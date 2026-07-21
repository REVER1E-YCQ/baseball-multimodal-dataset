"""训练 BEATs 区分 fly_ball 与 ground_ball。

使用方式（请在 model_test 根目录执行）：
    python train_beats_classifier.py --config config.json --run-name beats_crop1s

本脚本的设计前提：sample.csv 的 event_start / event_end 是人工给出的击球时间区间。
它先以区间中点裁剪音频，再做分类。因此本实验回答的是“对齐到击球附近时，音频能否
区分飞球和地滚球”，不会把定位器误差混入首轮分类结论。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.beats_classifier import BEATsBinaryClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
CLASS_TO_ID = {"fly_ball": 0, "ground_ball": 1}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}


def configure_text_output() -> None:
    """强制标准输出与错误输出使用 UTF-8，避免重定向训练日志时产生 GBK 乱码。

    Windows 在把输出重定向到文件时，Python 可能使用系统默认的 GBK 编码；
    VS Code 默认按 UTF-8 打开日志，于是中文会显示为乱码。这里仅改变文本输出
    编码，不会影响音频读取、模型参数、训练速度或训练结果。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


configure_text_output()


@dataclass(frozen=True)
class Sample:
    """单条样本的最小元数据；label 一律从目录名确定，避免读取模型可见的派生字段。"""

    sample_id: str
    label: str
    audio_path: Path
    event_start: float
    event_end: float
    crop_center: float
    source_key: str

    @property
    def event_midpoint(self) -> float:
        """分类裁剪中心；优先使用精化定位器输出，名称保留以兼容下游调用。"""
        return self.crop_center


def read_config(path: Path) -> dict[str, Any]:
    """读取 JSON 配置并阻止把 epoch 上限偷偷调高到 30 以上。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    epochs = int(config["training"]["max_epochs"])
    if not 1 <= epochs <= 30:
        raise ValueError("training.max_epochs 必须在 1 到 30 之间；本项目禁止超过 30。")
    if not 0 < config["split"]["validation_ratio"] < 1:
        raise ValueError("validation_ratio 必须位于 (0, 1)。")
    if not 0 < config["split"]["test_ratio"] < 1:
        raise ValueError("test_ratio 必须位于 (0, 1)。")
    if config["split"]["validation_ratio"] + config["split"]["test_ratio"] >= 1:
        raise ValueError("验证集与测试集比例之和必须小于 1。")
    pooling_candidates = config.get("model", {}).get("pooling_candidates", ["mean"])
    if not pooling_candidates or not isinstance(pooling_candidates, list):
        raise ValueError("model.pooling_candidates 必须是非空列表。")
    if len(pooling_candidates) != len(set(pooling_candidates)):
        raise ValueError("model.pooling_candidates 不能包含重复项。")
    unsupported = [name for name in pooling_candidates if name not in {"mean", "max"}]
    if unsupported:
        raise ValueError(f"不支持的池化方式：{unsupported}；当前只允许 mean 和 max。")
    return config


def source_key_from_file(source_file: Path) -> str:
    """提取视频 URL 作为分组键，避免同一视频的片段同时落入训练与测试。

    当前公布的数据几乎每条样本来自不同 URL；但保留这个逻辑可防止后续补充数据产生泄漏。
    """
    if not source_file.exists():
        return source_file.parent.name
    for line in source_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if "video_url:" in line:
            return line.split("video_url:", maxsplit=1)[1].strip()
    return source_file.parent.name


def load_refined_centers(manifest_path: Path, minimum_confidence: float) -> dict[tuple[str, str], float]:
    """读取定位器产物，只接纳置信度足够的精化时间点。

    低置信度样本不会悄悄退回粗标注中点；这样可避免把最可疑的时间标签混入 BEATs 训练。
    在首次尚未运行定位器时，返回空字典，由调用者明确打印提示后使用粗标注作为临时回退。
    """
    if not manifest_path.is_file():
        return {}
    accepted: dict[tuple[str, str], float] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                float(row["localizer_confidence"]) > minimum_confidence
                and row.get("review_status") == "auto_usable"
            ):
                accepted[(row["label"], row["sample_id"])] = float(row["refined_impact_time"])
    return accepted


def scan_samples(
    data_root: Path,
    refined_manifest_path: Path | None = None,
    minimum_refined_confidence: float = 0.60,
) -> list[Sample]:
    """扫描当前 Codex_Workstation 数据，不读取 landing_zone、region、bounce 等派生标签。

    这些字段若作为输入会造成标签泄漏：例如 fly 的 trajectory_type 与 ground 的 bounce
    在两类间并不对称。本任务的模型输入只能是 audio.wav。
    """
    refined_centers = (
        load_refined_centers(refined_manifest_path, minimum_refined_confidence)
        if refined_manifest_path is not None
        else {}
    )
    if refined_manifest_path is not None and not refined_manifest_path.is_file():
        print("[提示] 未发现精化时间清单；本次暂时使用原始 event 中点。请先运行 refine_impact_times.py。")
    samples: list[Sample] = []
    skipped_ids: list[str] = []
    for label, prefix in (("fly_ball", "F_*"), ("ground_ball", "G_*")):
        sample_root = data_root / label / "Codex_Workstation"
        if not sample_root.is_dir():
            raise FileNotFoundError(f"未找到数据目录：{sample_root}")
        for sample_dir in sorted(sample_root.glob(prefix)):
            audio_path, csv_path = sample_dir / "audio.wav", sample_dir / "sample.csv"
            if not audio_path.is_file() or not csv_path.is_file():
                skipped_ids.append(sample_dir.name)
                continue
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            start, end = float(row["event_start"]), float(row["event_end"])
            if not (0 <= start < end):
                skipped_ids.append(sample_dir.name)
                continue
            raw_midpoint = (start + end) / 2.0
            key = (label, sample_dir.name)
            if refined_manifest_path is not None and refined_manifest_path.is_file() and key not in refined_centers:
                skipped_ids.append(sample_dir.name)
                continue
            samples.append(
                Sample(
                    sample_id=sample_dir.name,
                    label=label,
                    audio_path=audio_path,
                    event_start=start,
                    event_end=end,
                    crop_center=refined_centers.get(key, raw_midpoint),
                    source_key=source_key_from_file(sample_dir / "source.txt"),
                )
            )
    if skipped_ids:
        print(f"跳过样本（{len(skipped_ids)}）：")
        # 每行最多 16 个编号，避免终端出现一条超长信息；这里只输出序列号，不重复原因。
        for begin in range(0, len(skipped_ids), 16):
            print("  " + "  ".join(skipped_ids[begin : begin + 16]))
    if not samples:
        raise RuntimeError("没有扫描到任何可训练样本。")
    return samples


def grouped_stratified_split(samples: list[Sample], val_ratio: float, test_ratio: float, seed: int):
    """按类别分层，并按 source_key 分组切分为 train/val/test。

    对每个类别的“来源组”分别打乱后分配。来源组不足时仍会工作，但统计波动会变大；
    因此每次运行都会把确切 split 写入 outputs，保证后续能复现同一结果。
    """
    rng = random.Random(seed)
    partitions = {"train": [], "val": [], "test": []}
    for label in CLASS_TO_ID:
        groups: dict[str, list[Sample]] = {}
        for sample in (item for item in samples if item.label == label):
            groups.setdefault(sample.source_key, []).append(sample)
        group_list = list(groups.values())
        rng.shuffle(group_list)
        n_groups = len(group_list)
        n_test = max(1, round(n_groups * test_ratio))
        n_val = max(1, round(n_groups * val_ratio))
        # 至少留一个来源组训练；现有数据规模远大于这个下限。
        if n_test + n_val >= n_groups:
            n_val = max(1, n_groups - n_test - 1)
        for index, group in enumerate(group_list):
            target = "test" if index < n_test else "val" if index < n_test + n_val else "train"
            partitions[target].extend(group)
    for key in partitions:
        rng.shuffle(partitions[key])
    return partitions


def read_centered_clip(
    audio_path: Path,
    center_seconds: float,
    crop_seconds: float,
    target_sample_rate: int,
    jitter_seconds: float = 0.0,
    remove_dc_offset: bool = True,
    peak_normalize: bool = False,
) -> torch.Tensor:
    """读取指定时间中心附近的固定长度片段，并输出目标采样率的一维波形。

    ``crop_seconds`` 控制模型可见上下文：0.5 s 更聚焦球棒撞击，1.0 s 是推荐起点，
    1.5--2.0 s 能包含更多回声/解说但也更容易把背景噪声当作类别线索。
    ``jitter_seconds`` 仅训练集使用，模拟定位器的轻微偏差；太大将导致真正击球声离开裁剪窗。
    """
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0)  # 数据目前是单声道；此写法兼容未来的双声道文件。
    desired_length = int(round(crop_seconds * sample_rate))
    center = center_seconds + jitter_seconds
    begin = int(round(center * sample_rate - desired_length / 2))
    end = begin + desired_length

    # 事件位于音频开头/结尾时，用 0 补齐，保证 batch 内张量长度恒定。
    clip = torch.zeros(desired_length, dtype=waveform.dtype)
    source_begin, source_end = max(0, begin), min(len(waveform), end)
    if source_end > source_begin:
        target_begin = source_begin - begin
        clip[target_begin : target_begin + source_end - source_begin] = waveform[source_begin:source_end]

    if sample_rate != target_sample_rate:
        clip = torchaudio.functional.resample(clip, sample_rate, target_sample_rate)
    expected = int(round(crop_seconds * target_sample_rate))
    clip = clip[:expected]
    if len(clip) < expected:
        clip = torch.nn.functional.pad(clip, (0, expected - len(clip)))

    if remove_dc_offset:
        clip = clip - clip.mean()
    if peak_normalize:
        # 关闭时保留原始响度信息；开启可降低录音音量差，但也可能抹去真实击球强弱线索。
        clip = clip / clip.abs().max().clamp_min(1e-6)
    return clip.contiguous()


class ImpactAudioDataset(Dataset):
    """按需读取波形的 Dataset；只对训练集进行轻量增强。"""

    def __init__(self, samples: list[Sample], audio_cfg: dict, aug_cfg: dict, training: bool) -> None:
        self.samples, self.audio_cfg, self.aug_cfg, self.training = samples, audio_cfg, aug_cfg, training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        jitter = 0.0
        if self.training:
            jitter_limit = float(self.audio_cfg["train_crop_jitter_seconds"])
            jitter = random.uniform(-jitter_limit, jitter_limit)
        clip = read_centered_clip(
            sample.audio_path,
            sample.event_midpoint,
            float(self.audio_cfg["crop_seconds"]),
            int(self.audio_cfg["target_sample_rate"]),
            jitter_seconds=jitter,
            remove_dc_offset=bool(self.audio_cfg["remove_dc_offset"]),
            peak_normalize=bool(self.audio_cfg["peak_normalize"]),
        )
        if self.training and self.aug_cfg.get("enabled", False):
            # 随机增益减少模型把录音音量误当类别的风险。gain_db 太大（>8）会失真。
            gain = 10 ** (random.uniform(-float(self.aug_cfg["gain_db"]), float(self.aug_cfg["gain_db"])) / 20)
            clip = clip * gain
            # 微量高斯噪声仅模拟编码/背景扰动；数值过大会淹没短促击球声。
            noise_std = float(self.aug_cfg["gaussian_noise_std"])
            if noise_std > 0:
                clip = clip + torch.randn_like(clip) * random.uniform(0, noise_std)
            clip = clip.clamp(-1.0, 1.0)
        return clip, CLASS_TO_ID[sample.label], sample.sample_id


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 可复现优先；如需最快速度可设为 False，但不同运行的分数会有轻微波动。
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    """让 Windows DataLoader 的每个 worker 都有不同且可复现的随机增强序列。"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def compute_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    """不依赖 sklearn 计算二分类研究所需的 Accuracy、Balanced Accuracy 与 Macro-F1。"""
    confusion = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        confusion[truth, pred] += 1
    recalls, f1s = [], []
    for class_id in range(2):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "accuracy": float(np.trace(confusion) / max(confusion.sum(), 1)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "confusion_matrix_rows_true_cols_pred": confusion.tolist(),
    }


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module):
    model.eval()
    total_loss, total_count, y_true, y_pred = 0.0, 0, [], []
    for clips, labels, _ in loader:
        clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        logits = model(clips)
        total_loss += criterion(logits, labels).item() * labels.size(0)
        total_count += labels.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(total_count, 1)
    return metrics


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train_pooling_candidate(
    pooling_mode: str,
    splits: dict[str, list[Sample]],
    config: dict[str, Any],
    device: torch.device,
    criterion: nn.Module,
    run_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """独立训练一种池化方式，并仅依据验证集 Macro-F1 保存其最佳 epoch。

    每个候选开始前都恢复相同随机种子，并重新创建 DataLoader。这样 mean/max 使用
    完全相同的数据划分、分类头初始化和随机增强序列，比较差异主要来自池化方式，
    而不是随机性。测试集不会在候选比较阶段被读取。
    """
    seed_everything(seed)
    candidate_dir = run_dir / "pooling_candidates" / pooling_mode
    candidate_dir.mkdir(parents=True, exist_ok=False)

    train_set = ImpactAudioDataset(splits["train"], config["audio"], config["augmentation"], training=True)
    val_set = ImpactAudioDataset(splits["val"], config["audio"], config["augmentation"], training=False)
    loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_init_fn,
    }
    if loader_args["num_workers"] > 0:
        loader_args["persistent_workers"] = True
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    train_loader = DataLoader(
        train_set, shuffle=True, generator=loader_generator, **loader_args
    )
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = BEATsBinaryClassifier(
        PROJECT_ROOT / config["paths"]["beats_checkpoint"],
        head_dropout=float(config["model"]["head_dropout"]),
        unfreeze_last_blocks=int(config["model"]["unfreeze_last_blocks"]),
        pooling_mode=pooling_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            float(config["training"]["head_learning_rate"]),
            float(config["training"]["backbone_learning_rate"]),
            float(config["training"]["weight_decay"]),
        )
    )
    use_amp = bool(config["training"]["use_mixed_precision"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"\n开始候选池化：{pooling_mode} | 设备={device} | AMP={use_amp} | 可训练参数={trainable:,}")

    best_f1, best_val_metrics, stale_epochs, history = -math.inf, None, 0, []
    max_epochs = int(config["training"]["max_epochs"])
    patience = int(config["training"]["early_stopping_patience"])
    stopped_early = False
    checkpoint_path = candidate_dir / "best_model.pt"
    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        for clips, labels, sample_ids in train_loader:
            clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)
            if not torch.isfinite(logits).all() or not torch.isfinite(loss):
                raise FloatingPointError(
                    f"{pooling_mode} 池化训练出现 NaN/Inf，已立即中止。问题样本编号："
                    + ", ".join(sample_ids)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_metrics = evaluate(model, val_loader, device, criterion)
        record = {"epoch": epoch, "train_loss": running_loss / max(seen, 1), "val": val_metrics}
        history.append(record)
        print(
            f"[{pooling_mode}] 轮次 {epoch:02d}/{max_epochs} | 训练损失 {record['train_loss']:.4f} | "
            f"验证损失 {val_metrics['loss']:.4f} | 宏平均F1 {val_metrics['macro_f1']:.4f} | "
            f"平衡准确率 {val_metrics['balanced_accuracy']:.4f}"
        )
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_val_metrics, stale_epochs = val_metrics["macro_f1"], val_metrics.copy(), 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "class_to_id": CLASS_TO_ID,
                    "pooling_mode": pooling_mode,
                    "best_epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "best_val_metrics": best_val_metrics,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                stopped_early = True
                print(
                    f"[{pooling_mode}] 早停：验证集 Macro-F1 连续 {patience} 个 epoch 未提升。"
                )
                break

    save_json(candidate_dir / "history.json", history)
    summary = {
        "pooling_mode": pooling_mode,
        "best_epoch": max(history, key=lambda item: item["val"]["macro_f1"])["epoch"],
        "best_val_macro_f1": float(best_f1),
        "best_val_balanced_accuracy": float(best_val_metrics["balanced_accuracy"]),
        "best_val_loss": float(best_val_metrics["loss"]),
        "completed_epochs": len(history),
        "stopped_early": stopped_early,
        "checkpoint": str(checkpoint_path),
    }
    save_json(candidate_dir / "summary.json", summary)

    # 两个候选顺序训练；显式释放第一个候选，避免第二次加载 BEATs 时占用双倍显存。
    del train_loader, val_loader, train_set, val_set, optimizer, scaler, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, history


def main() -> None:
    parser = argparse.ArgumentParser(description="用预训练 BEATs 分类 fly_ball / ground_ball")
    parser.add_argument("--config", default="config.json", help="配置文件路径（默认 config.json）")
    parser.add_argument("--run-name", default=None, help="输出子目录名；默认使用时间戳")
    args = parser.parse_args()

    config_path = (PROJECT_ROOT / args.config).resolve()
    config = read_config(config_path)
    seed = int(config["split"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or datetime.now().strftime("beats_%Y%m%d_%H%M%S")
    run_dir = (PROJECT_ROOT / config["paths"]["output_root"] / run_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "config_used.json", config)

    refined_path = (PROJECT_ROOT / config["paths"].get("refined_events_csv", "")).resolve()
    use_refined = bool(config.get("localization", {}).get("use_refined_events", False))
    samples = scan_samples(
        (PROJECT_ROOT / config["paths"]["data_root"]).resolve(),
        refined_path if use_refined else None,
        float(config.get("localization", {}).get("minimum_confidence", 0.45)),
    )
    splits = grouped_stratified_split(
        samples,
        float(config["split"]["validation_ratio"]),
        float(config["split"]["test_ratio"]),
        seed,
    )
    for split_name, split_samples in splits.items():
        print(f"{split_name:>5}: {len(split_samples):3d} 条，类别={dict(Counter(x.label for x in split_samples))}")
    save_json(
        run_dir / "splits.json",
        {key: [sample.__dict__ | {"audio_path": str(sample.audio_path)} for sample in value] for key, value in splits.items()},
    )

    train_labels = [CLASS_TO_ID[item.label] for item in splits["train"]]
    counts = np.bincount(train_labels, minlength=2)
    # 较少的 fly_ball 得到较大 loss 权重，避免模型只预测 ground_ball 获得虚高 Accuracy。
    class_weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    pooling_candidates = config["model"].get("pooling_candidates", ["mean"])
    print(f"\n将依次比较池化方式：{', '.join(pooling_candidates)}")
    candidate_summaries: list[dict[str, Any]] = []
    candidate_histories: dict[str, list[dict[str, Any]]] = {}
    for pooling_mode in pooling_candidates:
        summary, history = train_pooling_candidate(
            pooling_mode, splits, config, device, criterion, run_dir, seed
        )
        candidate_summaries.append(summary)
        candidate_histories[pooling_mode] = history

    # 只使用验证集 Macro-F1 选择池化方式；列表顺序是完全相同时的固定决胜规则。
    selected = max(candidate_summaries, key=lambda item: item["best_val_macro_f1"])
    selected_pooling = selected["pooling_mode"]
    comparison = {
        "selection_metric": "validation_macro_f1",
        "selected_pooling": selected_pooling,
        "candidates": candidate_summaries,
        "test_set_used_during_selection": False,
    }
    save_json(run_dir / "pooling_comparison.json", comparison)
    save_json(run_dir / "history.json", candidate_histories[selected_pooling])
    print("\n池化方式验证集比较：")
    for item in candidate_summaries:
        marker = " <- 选中" if item["pooling_mode"] == selected_pooling else ""
        print(
            f"  {item['pooling_mode']}: 最佳 epoch={item['best_epoch']}，"
            f"验证 Macro-F1={item['best_val_macro_f1']:.4f}，"
            f"验证平衡准确率={item['best_val_balanced_accuracy']:.4f}{marker}"
        )

    # 候选选择结束后才创建测试 DataLoader，并且只评估胜出的池化模型一次。
    best = torch.load(selected["checkpoint"], map_location=device, weights_only=False)
    model = BEATsBinaryClassifier(
        PROJECT_ROOT / config["paths"]["beats_checkpoint"],
        head_dropout=float(config["model"]["head_dropout"]),
        unfreeze_last_blocks=int(config["model"]["unfreeze_last_blocks"]),
        pooling_mode=selected_pooling,
    ).to(device)
    model.load_state_dict(best["model_state_dict"])
    model.eval()
    test_set = ImpactAudioDataset(splits["test"], config["audio"], config["augmentation"], training=False)
    test_loader_args = {
        "batch_size": int(config["training"]["batch_size"]),
        "num_workers": int(config["training"]["num_workers"]),
        "pin_memory": device.type == "cuda",
        "worker_init_fn": worker_init_fn,
    }
    if test_loader_args["num_workers"] > 0:
        test_loader_args["persistent_workers"] = True
    test_loader = DataLoader(test_set, shuffle=False, **test_loader_args)
    test_metrics = evaluate(model, test_loader, device, criterion)
    # 根目录的 best_model.pt 始终是最终胜者，预测脚本可像以前一样直接使用它。
    torch.save(best, run_dir / "best_model.pt")
    report = {
        "selected_pooling": selected_pooling,
        "pooling_selection_metric": "validation_macro_f1",
        "pooling_candidates": candidate_summaries,
        "best_epoch": best["best_epoch"],
        "best_val_macro_f1": best["best_val_macro_f1"],
        "test": test_metrics,
        "class_weights": class_weights.detach().cpu().tolist(),
        "note": "测试集仅在选择最佳验证 epoch 后评估一次；不可据此继续调参。",
    }
    save_json(run_dir / "metrics.json", report)
    matrix = test_metrics["confusion_matrix_rows_true_cols_pred"]
    print("\n训练完成")
    print(f"最终采用的池化方式：{selected_pooling}（按验证集 Macro-F1 选择）")
    print(
        f"最佳轮次 {best['best_epoch']} | 测试损失 {test_metrics['loss']:.4f} | "
        f"测试准确率 {test_metrics['accuracy']:.4f} | 测试宏平均F1 {test_metrics['macro_f1']:.4f} | "
        f"测试平衡准确率 {test_metrics['balanced_accuracy']:.4f}"
    )
    print(f"混淆矩阵（真实类别为行、预测类别为列；顺序 fly/ground）：{matrix}")
    print(f"详细结果已保存到：{run_dir}")


if __name__ == "__main__":
    # Windows 的多进程 DataLoader 必须保留这个入口保护。
    main()
