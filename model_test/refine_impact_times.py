"""用“启发式候选 + 随机森林”精化击球时间，并生成 BEATs 可读取的时间清单。

为什么不直接取 event_start/event_end 的中点？
--------------------------------------------
当前 event 区间来自自动流程，通常覆盖真实击球但可能偏宽。区间中点不是物理真值，
直接用它裁剪会把背景声带入 BEATs。此脚本在原始 48 kHz 波形上以 5 ms 步长寻找
冲击候选点，再用随机森林排序。

重要限制
--------
这仍是“弱监督定位”，不是人工真值验证：随机森林的初始正样本由高能量 / 高谱通量的
候选构造，若解说、欢呼或剪辑音效比撞击更显著，仍可能选错。因此脚本从不覆盖
sample.csv；它只写 outputs/refined_events/refined_events.csv 和置信度，低置信度样本
必须先人工试听/查看波形，再决定是否用于正式训练。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torchaudio
from sklearn.ensemble import RandomForestClassifier


PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_NAMES = [
    "log_rms",
    "crest_factor",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "high_band_ratio",
    "spectral_flux",
    "peak_amplitude",
]


@dataclass(frozen=True)
class RawSample:
    sample_id: str
    label: str
    audio_path: Path
    event_start: float
    event_end: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def scan_samples(data_root: Path) -> list[RawSample]:
    """扫描两类音频，并严格保留原始 event 区间作为审计字段。"""
    samples: list[RawSample] = []
    for label, prefix in (("fly_ball", "F_*"), ("ground_ball", "G_*")):
        root = data_root / label / "Codex_Workstation"
        if not root.is_dir():
            raise FileNotFoundError(f"未找到数据目录：{root}")
        for directory in sorted(root.glob(prefix)):
            with (directory / "sample.csv").open("r", encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))
            start, end = float(row["event_start"]), float(row["event_end"])
            if (directory / "audio.wav").is_file() and 0 <= start < end:
                samples.append(RawSample(directory.name, label, directory / "audio.wav", start, end))
    if not samples:
        raise RuntimeError("没有找到可处理的音频样本。")
    return samples


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """使用中位数/MAD 标准化，避免单个极大欢呼声把均值和标准差拉偏。"""
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return np.clip((values - median) / (1.4826 * mad + 1e-8), -6.0, 12.0)


def extract_frame_features(audio_path: Path, frame_seconds: float, hop_seconds: float):
    """从原始采样率波形提取高时间分辨率的冲击声特征。

    - RMS / peak：瞬时能量；
    - crest factor：尖锐撞击相对持续背景声更高；
    - 谱通量：频谱突然变化，对球棒撞击很敏感；
    - 高频比例：撞击常带有较多高频成分；
    - ZCR / 谱质心：辅助区分人声、持续噪声和瞬态。

    不使用 librosa，避免不同环境下的隐式帧中心偏移；这里的时间戳明确是分析窗中心。
    """
    waveform, sample_rate = torchaudio.load(str(audio_path))
    waveform = waveform.mean(dim=0).float()
    frame_size = max(64, int(round(frame_seconds * sample_rate)))
    hop_size = max(1, int(round(hop_seconds * sample_rate)))
    if len(waveform) < frame_size:
        waveform = torch.nn.functional.pad(waveform, (0, frame_size - len(waveform)))
    frames = waveform.unfold(0, frame_size, hop_size)
    window = torch.hann_window(frame_size, device=frames.device, dtype=frames.dtype)
    spectrum = torch.fft.rfft(frames * window, dim=1).abs().clamp_min(1e-8)
    frequencies = torch.fft.rfftfreq(frame_size, d=1.0 / sample_rate).to(spectrum.device)

    rms = frames.square().mean(dim=1).sqrt().clamp_min(1e-8)
    peak = frames.abs().amax(dim=1)
    crest = peak / rms
    zcr = (frames[:, 1:] * frames[:, :-1] < 0).float().mean(dim=1)
    magnitude_sum = spectrum.sum(dim=1).clamp_min(1e-8)
    centroid = (spectrum * frequencies).sum(dim=1) / magnitude_sum
    high_mask = (frequencies >= 2000.0) & (frequencies <= min(10000.0, sample_rate / 2.0))
    high_ratio = spectrum[:, high_mask].sum(dim=1) / magnitude_sum
    log_spectrum = spectrum.log()
    flux = torch.zeros(len(frames), dtype=frames.dtype)
    flux[1:] = torch.relu(log_spectrum[1:] - log_spectrum[:-1]).mean(dim=1)

    features = torch.stack(
        [rms.log(), crest, zcr, centroid, high_ratio, flux, peak], dim=1
    ).cpu().numpy().astype(np.float32)
    # centre=False 等价：第 0 个 frame 从波形第 0 个采样开始，时间戳取窗中心。
    times = (np.arange(len(features)) * hop_size + frame_size / 2.0) / sample_rate
    return features, times


def heuristic_score(features: np.ndarray) -> np.ndarray:
    """以鲁棒标准化的多特征融合产生候选分数；只用于构造弱标签和 RF 辅助排序。"""
    return (
        0.30 * robust_zscore(features[:, 0])  # log RMS
        + 0.25 * robust_zscore(features[:, 5])  # spectral flux
        + 0.20 * robust_zscore(features[:, 4])  # high-band ratio
        + 0.15 * robust_zscore(features[:, 1])  # crest factor
        + 0.10 * robust_zscore(features[:, 6])  # peak amplitude
    )


def search_indices(times: np.ndarray, start: float, end: float, padding_seconds: float) -> np.ndarray:
    """只在粗标注附近搜索；padding 用来容忍边界误差，不能大到重新搜索整段解说。"""
    return np.flatnonzero((times >= max(0.0, start - padding_seconds)) & (times <= end + padding_seconds))


def non_maximum_peaks(score: np.ndarray, candidate_indices: np.ndarray, times: np.ndarray, minimum_distance: float):
    """按得分从高到低选局部峰，防止同一个撞击附近连续 5 ms 帧重复入选。"""
    selected: list[int] = []
    for index in candidate_indices[np.argsort(score[candidate_indices])[::-1]]:
        if all(abs(times[index] - times[old]) >= minimum_distance for old in selected):
            selected.append(int(index))
    return selected


def build_weak_training_set(records: list[dict[str, Any]], config: dict[str, Any]):
    """由每条音频最像冲击声的峰构建正样本，并从其它位置抽取困难负样本。

    这比把整个 event_start--event_end 区间都标成正帧更抗“区间偏宽”噪声，但仍不能
    替代人工时间标签。正样本会取候选点附近 ±positive_radius 的多帧，提高时间稳定性。
    """
    weak_cfg, audio_cfg = config["weak_labels"], config["audio"]
    rng = np.random.default_rng(int(weak_cfg["random_seed"]))
    x_parts, y_parts = [], []
    for record in records:
        features, times, score, search = record["features"], record["times"], record["heuristic"], record["search"]
        peaks = non_maximum_peaks(score, search, times, float(audio_cfg["minimum_peak_distance_seconds"]))
        if not peaks:
            continue
        anchor = peaks[0]
        positive = np.flatnonzero(np.abs(times - times[anchor]) <= float(weak_cfg["positive_radius_seconds"]))
        negative_pool = np.flatnonzero(np.abs(times - times[anchor]) >= float(weak_cfg["negative_exclusion_seconds"]))
        # 优先选择“得分较高但不是锚点”的困难负样本，再以随机背景帧补足。
        hard = [index for index in peaks[1:] if index in set(negative_pool)]
        desired = int(weak_cfg["negatives_per_sample"])
        selected_negative = hard[:desired]
        if len(selected_negative) < desired and len(negative_pool):
            remaining = np.setdiff1d(negative_pool, np.asarray(selected_negative, dtype=int), assume_unique=False)
            chosen = rng.choice(remaining, size=min(desired - len(selected_negative), len(remaining)), replace=False)
            selected_negative.extend(np.asarray(chosen, dtype=int).tolist())
        if not selected_negative:
            continue
        x_parts.extend([features[positive], features[np.asarray(selected_negative)]])
        y_parts.extend([np.ones(len(positive), dtype=np.int64), np.zeros(len(selected_negative), dtype=np.int64)])
        record["heuristic_anchor_index"] = anchor
    if not x_parts:
        raise RuntimeError("无法构造随机森林的弱监督训练集。")
    return np.vstack(x_parts), np.concatenate(y_parts)


def normalized_score(values: np.ndarray) -> np.ndarray:
    """把候选启发式分数压到 0--1，便于与 RF 概率做凸组合。"""
    low, high = np.percentile(values, 5), np.percentile(values, 95)
    return np.clip((values - low) / max(high - low, 1e-8), 0.0, 1.0)


def choose_refined_time(record: dict[str, Any], forest: RandomForestClassifier, config: dict[str, Any]):
    """使用 RF 概率与启发式分数融合，返回时间、置信度、峰间边际和诊断信息。"""
    features, times, score, search = record["features"], record["times"], record["heuristic"], record["search"]
    rf_probability = forest.predict_proba(features)[:, 1]
    rf_weight = float(config["selection"]["random_forest_weight"])
    combined = rf_weight * rf_probability + (1.0 - rf_weight) * normalized_score(score)
    peaks = non_maximum_peaks(
        combined, search, times, float(config["audio"]["minimum_peak_distance_seconds"])
    )
    if not peaks:
        raise RuntimeError(f"{record['sample'].sample_id} 没有候选击球点。")
    first, second = peaks[0], peaks[1] if len(peaks) > 1 else peaks[0]
    margin = max(0.0, float(combined[first] - combined[second]))
    # 概率高、且第一峰明显强于第二峰时置信度高；这是审核优先级，不是统计校准概率。
    confidence = float(combined[first] * (0.70 + 0.30 * min(margin / 0.20, 1.0)))
    return {
        "refined_impact_time": float(times[first]),
        "localizer_confidence": confidence,
        "rf_probability": float(rf_probability[first]),
        "combined_score": float(combined[first]),
        "peak_margin": margin,
        "heuristic_anchor_time": float(times[record["heuristic_anchor_index"]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="精化击球时间：RMS/onset 类特征 + 随机森林弱监督排序")
    parser.add_argument("--config", default="localizer_config.json", help="定位器 JSON 配置文件")
    args = parser.parse_args()
    config = read_json((PROJECT_ROOT / args.config).resolve())
    output_dir = (PROJECT_ROOT / config["paths"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = scan_samples((PROJECT_ROOT / config["paths"]["data_root"]).resolve())
    records: list[dict[str, Any]] = []
    for number, sample in enumerate(samples, start=1):
        features, times = extract_frame_features(
            sample.audio_path,
            float(config["audio"]["frame_seconds"]),
            float(config["audio"]["hop_seconds"]),
        )
        search = search_indices(
            times, sample.event_start, sample.event_end, float(config["audio"]["search_padding_seconds"])
        )
        if len(search) == 0:
            print(f"[跳过] {sample.sample_id} 的粗标注附近没有可分析帧")
            continue
        records.append({"sample": sample, "features": features, "times": times, "heuristic": heuristic_score(features), "search": search})
        if number % 50 == 0 or number == len(samples):
            print(f"已提取特征：{number}/{len(samples)}")

    train_x, train_y = build_weak_training_set(records, config)
    # 极端短音频等可能没有足够负样本；这些记录未参与 RF 构造，也不能假装有可靠锚点。
    records = [record for record in records if "heuristic_anchor_index" in record]
    rf_cfg = config["random_forest"]
    forest = RandomForestClassifier(
        n_estimators=int(rf_cfg["n_estimators"]),
        max_depth=int(rf_cfg["max_depth"]),
        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
        max_features=rf_cfg["max_features"],
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(config["weak_labels"]["random_seed"]),
    )
    forest.fit(train_x, train_y)
    joblib.dump({"model": forest, "feature_names": FEATURE_NAMES, "config": config}, output_dir / "impact_localizer_rf.joblib")

    threshold = float(config["selection"]["minimum_confidence_for_beats"])
    shift_review_threshold = float(config["selection"]["review_if_shift_from_raw_midpoint_seconds"])
    rows = []
    for record in records:
        sample = record["sample"]
        result = choose_refined_time(record, forest, config)
        raw_midpoint = (sample.event_start + sample.event_end) / 2.0
        shift_from_raw_midpoint = abs(result["refined_impact_time"] - raw_midpoint)
        # 大偏移不代表一定错误：真实撞击可能本来在粗区间边缘；但它是最值得人工核验的信号。
        confidence_ok = result["localizer_confidence"] > threshold
        shift_ok = shift_from_raw_midpoint <= shift_review_threshold
        review_status = "auto_usable" if confidence_ok and shift_ok else "needs_manual_review"
        # 将两种拦截原因分开记录，便于判断样本究竟是定位置信度不足，还是相对
        # AI 粗标注区间中点的偏移过大。当前偏移上限由配置设为 0.20 秒；调大
        # 该值会放宽偏移检查，但不会绕过 minimum_confidence_for_beats。
        if confidence_ok and shift_ok:
            review_reason = "passed"
        elif not confidence_ok and not shift_ok:
            review_reason = "low_confidence_and_excessive_shift"
        elif not confidence_ok:
            review_reason = "low_confidence"
        else:
            review_reason = "excessive_shift"
        rows.append(
            {
                "sample_id": sample.sample_id,
                "label": sample.label,
                "raw_event_start": sample.event_start,
                "raw_event_end": sample.event_end,
                "shift_from_raw_midpoint_seconds": shift_from_raw_midpoint,
                **result,
                "review_status": review_status,
                "review_reason": review_reason,
            }
        )
    fields = list(rows[0].keys())
    with (output_dir / "refined_events.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    # 单独输出人工复核队列：先看低置信度，其次看相对粗中点偏移大的候选。
    review_rows = sorted(
        (row for row in rows if row["review_status"] == "needs_manual_review"),
        key=lambda row: (float(row["localizer_confidence"]), -float(row["shift_from_raw_midpoint_seconds"])),
    )
    with (output_dir / "manual_review_priority.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(review_rows)
    (output_dir / "localizer_config_used.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    usable = sum(row["review_status"] == "auto_usable" for row in rows)
    reason_counts = Counter(row["review_reason"] for row in rows)
    print(f"\n已写入 {output_dir / 'refined_events.csv'}")
    print(f"人工复核队列：{output_dir / 'manual_review_priority.csv'}")
    print(
        f"筛选条件：置信度 > {threshold:.2f}，相对原始区间中点偏移 <= "
        f"{shift_review_threshold:.3f} 秒"
    )
    print(
        "筛选原因统计："
        f"通过={reason_counts.get('passed', 0)}，"
        f"低置信度={reason_counts.get('low_confidence', 0)}，"
        f"偏移过大={reason_counts.get('excessive_shift', 0)}，"
        f"两者均不满足={reason_counts.get('low_confidence_and_excessive_shift', 0)}"
    )
    print(f"可直接进入 BEATs 训练：{usable}/{len(rows)}；其余样本请优先人工复核。")
    print("注意：RF 的训练标签由候选峰弱监督生成，不能把它的训练内分数当作定位准确率。")


if __name__ == "__main__":
    main()
