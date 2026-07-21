from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from transformers import AutoModel

matplotlib.use("Agg")


MODEL_ID = "worstchan/EAT-base_epoch30_finetune_AS2M"
MODEL_REVISION = "60d61e8b2e9e5ba3be6860285de80cb7d625ccbb"
WINDOWS = (0.25, 0.5, 1.0)
POOL_NAMES = ("cls", "max", "mean")
LABEL_TO_ID = {"ground_ball": 0, "fly_ball": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    sample_rate: int = 16_000
    validation_fraction: float = 0.15
    batch_size: int = 32
    norm_mean: float = -4.268
    norm_std: float = 4.569
    event_position_fraction: float = 0.2


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_frames(window_seconds: float) -> int:
    raw_frames = math.floor((window_seconds * 16_000 - 400) / 160) + 1
    return int(math.ceil(raw_frames / 16) * 16)


def extract_clip(
    audio: np.ndarray,
    sample_rate: int,
    event_center: float,
    window_seconds: float,
    event_position_fraction: float,
) -> np.ndarray:
    target_length = int(round(sample_rate * window_seconds))
    start_time = event_center - window_seconds * event_position_fraction
    source_start = max(0, int(round(start_time * sample_rate)))
    target_start = max(0, -int(round(start_time * sample_rate)))
    source_end = min(len(audio), source_start + target_length - target_start)
    clip = np.zeros(target_length, dtype=np.float32)
    available = max(0, source_end - source_start)
    if available:
        copy_length = min(available, target_length - target_start)
        clip[target_start : target_start + copy_length] = audio[source_start : source_start + copy_length]
    return clip


def clip_to_fbank(
    clip: np.ndarray,
    source_rate: int,
    frames: int,
    cfg: Config,
) -> torch.Tensor:
    waveform = torch.from_numpy(clip).float()
    if source_rate != cfg.sample_rate:
        waveform = torchaudio.functional.resample(waveform, source_rate, cfg.sample_rate)
    waveform = waveform - waveform.mean()
    mel = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        htk_compat=True,
        sample_frequency=cfg.sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=128,
        dither=0.0,
        frame_shift=10,
    )
    if mel.shape[0] < frames:
        mel = torch.nn.functional.pad(mel, (0, 0, 0, frames - mel.shape[0]))
    else:
        mel = mel[:frames]
    return (mel - cfg.norm_mean) / (cfg.norm_std * 2.0)


def read_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(sample_rate)


def extract_eat_features(
    split: pd.DataFrame,
    output_path: Path,
    cfg: Config,
    device: torch.device,
) -> np.ndarray:
    model = AutoModel.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
    ).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    features = np.zeros((len(split), len(WINDOWS), len(POOL_NAMES), 768), dtype=np.float16)
    started = time.time()
    for batch_start in range(0, len(split), cfg.batch_size):
        batch_end = min(len(split), batch_start + cfg.batch_size)
        rows = split.iloc[batch_start:batch_end]
        audio_batch = [read_audio(path) for path in rows["audio_path"]]
        centers = ((rows["event_start"] + rows["event_end"]) / 2.0).to_numpy()

        for window_index, window_seconds in enumerate(WINDOWS):
            frames = target_frames(window_seconds)
            mels = []
            for (audio, sample_rate), center in zip(audio_batch, centers):
                clip = extract_clip(
                    audio,
                    sample_rate,
                    float(center),
                    window_seconds,
                    cfg.event_position_fraction,
                )
                mels.append(clip_to_fbank(clip, sample_rate, frames, cfg))
            inputs = torch.stack(mels).unsqueeze(1).to(device, non_blocking=True)
            autocast = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda")
            with torch.inference_mode(), autocast:
                hidden = model.extract_features(inputs)
            cls = hidden[:, 0]
            frame_hidden = hidden[:, 1:]
            pooled = torch.stack([cls, frame_hidden.max(dim=1).values, frame_hidden.mean(dim=1)], dim=1)
            features[batch_start:batch_end, window_index] = pooled.float().cpu().numpy().astype(np.float16)

        elapsed = time.time() - started
        print(f"Extracted {batch_end}/{len(split)} samples in {elapsed:.1f}s", flush=True)

    np.savez_compressed(
        output_path,
        eat_features=features,
        windows=np.asarray(WINDOWS, dtype=np.float32),
        pool_names=np.asarray(POOL_NAMES),
        model_id=np.asarray(MODEL_ID),
        model_revision=np.asarray(MODEL_REVISION),
    )
    return features


def candidate_features(eat: np.ndarray, traditional: np.ndarray) -> dict[str, np.ndarray]:
    eat = eat.astype(np.float32)
    one_second = eat[:, 2]
    multi_max = eat[:, :, 1].reshape(len(eat), -1)
    multi_cls_max = eat[:, :, :2].reshape(len(eat), -1)
    multi_all = eat.reshape(len(eat), -1)
    return {
        "eat_1s_cls": one_second[:, 0],
        "eat_1s_max": one_second[:, 1],
        "eat_1s_all_pooling": one_second.reshape(len(eat), -1),
        "eat_multi_max": multi_max,
        "eat_multi_cls_max": multi_cls_max,
        "eat_multi_all": multi_all,
        "eat_multi_max_plus_traditional": np.concatenate([multi_max, traditional], axis=1),
        "eat_multi_all_plus_traditional": np.concatenate([multi_all, traditional], axis=1),
    }


def best_threshold(y_true: np.ndarray, scores: np.ndarray, default: float) -> tuple[float, float, float]:
    thresholds = np.unique(np.concatenate([np.linspace(0.15, 0.85, 141), np.asarray([default])]))
    best = (-1.0, -1.0, default)
    for threshold in thresholds:
        prediction = (scores >= threshold).astype(int)
        macro_f1 = f1_score(y_true, prediction, average="macro")
        balanced = balanced_accuracy_score(y_true, prediction)
        candidate = (macro_f1, balanced, float(threshold))
        if candidate[:2] > best[:2]:
            best = candidate
    return best[2], best[0], best[1]


def make_model(model_type: str, parameter: float) -> Pipeline:
    if model_type == "logistic":
        classifier = LogisticRegression(
            C=parameter,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        )
    elif model_type == "rbf_svm":
        classifier = SVC(
            C=parameter,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        )
    else:
        raise ValueError(model_type)
    return Pipeline([("scaler", StandardScaler()), ("classifier", classifier)])


def select_configuration(
    candidates: dict[str, np.ndarray],
    labels: np.ndarray,
    inner_train: np.ndarray,
    validation: np.ndarray,
) -> tuple[dict[str, object], pd.DataFrame]:
    rows = []
    best: dict[str, object] | None = None
    for feature_name, values in candidates.items():
        settings = [("logistic", value) for value in (0.001, 0.01, 0.1, 1.0, 10.0)]
        if feature_name in {
            "eat_1s_cls",
            "eat_1s_max",
            "eat_multi_max",
            "eat_multi_max_plus_traditional",
        }:
            settings += [("rbf_svm", value) for value in (0.1, 1.0, 10.0, 100.0)]
        for model_type, parameter in settings:
            started = time.time()
            model = make_model(model_type, parameter)
            model.fit(values[inner_train], labels[inner_train])
            scores = model.predict_proba(values[validation])[:, 1]
            threshold, macro_f1, balanced = best_threshold(labels[validation], scores, 0.5)
            auc = roc_auc_score(labels[validation], scores)
            row = {
                "feature_name": feature_name,
                "feature_dimension": int(values.shape[1]),
                "model_type": model_type,
                "parameter_C": parameter,
                "threshold": threshold,
                "validation_macro_f1": macro_f1,
                "validation_balanced_accuracy": balanced,
                "validation_roc_auc": auc,
                "fit_seconds": time.time() - started,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            ranking = (macro_f1, balanced, auc)
            if best is None or ranking > best["ranking"]:
                best = {**row, "ranking": ranking}
    assert best is not None
    best.pop("ranking")
    return best, pd.DataFrame(rows)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_plots(selection: pd.DataFrame, metrics: dict[str, object], output_dir: Path) -> None:
    top = selection.sort_values("validation_macro_f1", ascending=False).head(12).copy()
    top["label"] = top["feature_name"] + "\n" + top["model_type"] + " C=" + top["parameter_C"].astype(str)
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.barh(top["label"][::-1], top["validation_macro_f1"][::-1], color="#2878b5")
    axis.set_xlim(0, 1)
    axis.set_xlabel("Validation Macro-F1")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "validation_candidates.png", dpi=180)
    plt.close(figure)

    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_xticks([0, 1], ["ground_ball", "fly_ball"])
    axis.set_yticks([0, 1], ["ground_ball", "fly_ball"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("EAT audio fusion test confusion matrix")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "test_confusion_matrix.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--traditional-cache", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    cfg = Config()
    set_seed(cfg.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.feature_cache.parent.mkdir(parents=True, exist_ok=True)
    split = pd.read_csv(args.split_csv)
    labels = split["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    train_indices = np.flatnonzero(split["split"].to_numpy() == "train")
    test_indices = np.flatnonzero(split["split"].to_numpy() == "test")
    inner_train, validation = train_test_split(
        train_indices,
        test_size=cfg.validation_fraction,
        random_state=cfg.seed,
        stratify=labels[train_indices],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.feature_cache.exists():
        eat = extract_eat_features(split, args.feature_cache, cfg, device)
    else:
        eat = np.load(args.feature_cache)["eat_features"]
        print(f"Loaded EAT cache {eat.shape}", flush=True)
    if args.extract_only:
        return

    traditional_cache = np.load(args.traditional_cache)
    traditional = traditional_cache["traditional"].astype(np.float32)
    if len(traditional) != len(split) or len(eat) != len(split):
        raise RuntimeError("Feature cache and split have different sample counts")
    candidates = candidate_features(eat, traditional)
    best, selection = select_configuration(candidates, labels, inner_train, validation)
    selection.to_csv(args.output_dir / "validation_model_selection.csv", index=False, encoding="utf-8-sig")
    (args.output_dir / "selected_configuration.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    chosen = candidates[str(best["feature_name"])]
    final_model = make_model(str(best["model_type"]), float(best["parameter_C"]))
    final_model.fit(chosen[train_indices], labels[train_indices])
    test_scores = final_model.predict_proba(chosen[test_indices])[:, 1]
    test_prediction = (test_scores >= float(best["threshold"])).astype(int)
    metrics = calculate_metrics(labels[test_indices], test_prediction, test_scores)
    metrics.update(
        {
            "selected_configuration": best,
            "dataset_samples": int(len(split)),
            "train_samples": int(len(train_indices)),
            "inner_train_samples": int(len(inner_train)),
            "validation_samples": int(len(validation)),
            "test_samples": int(len(test_indices)),
            "eat_model_id": MODEL_ID,
            "eat_model_revision": MODEL_REVISION,
            "windows_seconds": list(WINDOWS),
            "pooling": list(POOL_NAMES),
            "device": str(device),
            "torch_version": torch.__version__,
            "config": asdict(cfg),
        }
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(final_model, args.output_dir / "eat_audio_fusion_model.joblib")
    predictions = split.iloc[test_indices][["sample_id", "label", "source_id", "audio_path"]].copy()
    predictions["predicted_label"] = [ID_TO_LABEL[int(value)] for value in test_prediction]
    predictions["fly_ball_probability"] = test_scores
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    save_plots(selection, metrics, args.output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
