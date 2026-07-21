from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


LABEL_TO_ID = {"ground_ball": 0, "fly_ball": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    sample_rate: int = 48_000
    clip_seconds: float = 0.5
    pre_event_seconds: float = 0.1
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 480
    n_mels: int = 64
    n_mfcc: int = 20


def parse_source_id(source_text: str) -> str:
    for line in source_text.splitlines():
        if line.startswith("source_id:"):
            return line.split(":", 1)[1].strip()
    return ""


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for audio_path in sorted(dataset_root.rglob("audio.wav")):
        sample_dir = audio_path.parent
        with (sample_dir / "sample.csv").open(encoding="utf-8-sig", newline="") as handle:
            metadata = next(csv.DictReader(handle))
        label = metadata["label"].strip()
        if label not in LABEL_TO_ID:
            continue
        source_text = (sample_dir / "source.txt").read_text(encoding="utf-8")
        source_id = parse_source_id(source_text)
        relative_audio_path = audio_path.relative_to(dataset_root).as_posix()
        if not source_id:
            source_id = f"path:{relative_audio_path}"
        rows.append(
            {
                "sample_id": metadata["sample_id"].strip(),
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "event_start": float(metadata["event_start"]),
                "event_end": float(metadata["event_end"]),
                "source_id": source_id,
                "audio_sha256": file_sha256(audio_path),
                "relative_audio_path": relative_audio_path,
                "audio_path": str(audio_path.resolve()),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["label_id", "sample_id"]).reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError(f"No samples found under {dataset_root}")
    if manifest["source_id"].duplicated().any():
        duplicates = manifest.loc[manifest["source_id"].duplicated(False), "source_id"].tolist()
        raise RuntimeError(f"Repeated source IDs require grouped splitting: {duplicates[:5]}")
    return manifest


def create_grouped_split(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create an exact 70/30 stratified split without duplicate-audio leakage."""
    rng = np.random.default_rng(seed)
    test_total = int(math.ceil(len(manifest) * 0.30))
    label_counts = manifest["label_id"].value_counts().sort_index()
    exact_targets = label_counts * 0.30
    test_targets = np.floor(exact_targets).astype(int)
    remaining = test_total - int(test_targets.sum())
    remainders = (exact_targets - test_targets).sort_values(ascending=False)
    for label_id in remainders.index[:remaining]:
        test_targets.loc[label_id] += 1

    test_indices: list[int] = []
    for label_id, target in test_targets.items():
        label_rows = manifest.index[manifest["label_id"] == label_id].to_numpy()
        groups = [
            group.index.to_numpy()
            for _, group in manifest.loc[label_rows].groupby("audio_sha256", sort=False)
        ]
        rng.shuffle(groups)
        needed = int(target)
        for group in groups:
            if len(group) <= needed:
                test_indices.extend(group.tolist())
                needed -= len(group)
        if needed:
            raise RuntimeError(f"Could not make an exact grouped split for label {label_id}")

    split = manifest.copy()
    split["split"] = "train"
    split.loc[np.asarray(test_indices, dtype=int), "split"] = "test"
    return split.sort_values(["split", "label_id", "sample_id"]).reset_index(drop=True)


def extract_event_clip(audio: np.ndarray, sample_rate: int, event_center: float, cfg: Config) -> np.ndarray:
    clip_start = event_center - cfg.pre_event_seconds
    clip_end = clip_start + cfg.clip_seconds
    target_length = int(round(cfg.clip_seconds * cfg.sample_rate))
    source_start = max(0, int(round(clip_start * sample_rate)))
    source_end = min(len(audio), int(round(clip_end * sample_rate)))
    target_start = max(0, -int(round(clip_start * sample_rate)))
    clip = np.zeros(target_length, dtype=np.float32)
    available = max(0, source_end - source_start)
    if available:
        copy_length = min(available, target_length - target_start)
        clip[target_start : target_start + copy_length] = audio[source_start : source_start + copy_length]
    return clip


def read_event_clip(row: pd.Series, cfg: Config) -> np.ndarray:
    audio, sample_rate = librosa.load(row["audio_path"], sr=None, mono=True)
    if sample_rate != cfg.sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=cfg.sample_rate)
        sample_rate = cfg.sample_rate
    event_center = (float(row["event_start"]) + float(row["event_end"])) / 2.0
    return extract_event_clip(audio, sample_rate, event_center, cfg)


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio.astype(np.float32, copy=True)
    return (audio * (target_peak / peak)).astype(np.float32)


def frame_statistics(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(1, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    maximum = flat.max(axis=1)
    minimum = flat.min(axis=1)
    median = np.median(flat, axis=1)
    centered = flat - mean[:, None]
    safe_std = np.maximum(std, 1e-10)
    skewness = np.mean(centered**3, axis=1) / safe_std**3
    kurtosis = np.mean(centered**4, axis=1) / safe_std**4
    return np.concatenate([mean, std, maximum, minimum, median, skewness, kurtosis]).astype(np.float32)


def traditional_features(audio: np.ndarray, cfg: Config) -> np.ndarray:
    original_peak = float(np.max(np.abs(audio)))
    original_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    original_energy = float(np.sum(audio**2))
    normalized = peak_normalize(audio)

    stft = librosa.stft(
        normalized,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hamming",
        center=True,
    )
    magnitude = np.abs(stft)
    power = magnitude**2
    mel = librosa.feature.melspectrogram(
        S=power,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        fmin=20,
        fmax=cfg.sample_rate / 2,
    )
    log_mel = librosa.power_to_db(mel + 1e-12, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=cfg.n_mfcc)
    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=cfg.sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=cfg.sample_rate)
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=cfg.sample_rate, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(
        normalized, frame_length=cfg.win_length, hop_length=cfg.hop_length
    )
    rms = librosa.feature.rms(S=magnitude, frame_length=cfg.n_fft, hop_length=cfg.hop_length)

    frequencies = librosa.fft_frequencies(sr=cfg.sample_rate, n_fft=cfg.n_fft)
    band_edges = [0, 500, 1000, 2000, 4000, 8000, 12000, 16000, 24000]
    total_power = np.maximum(power.sum(axis=0), 1e-12)
    band_ratios = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        band_ratios.append(power[mask].sum(axis=0) / total_power)
    band_ratios_array = np.asarray(band_ratios)

    frame_rms = rms.reshape(-1)
    peak_frame = int(np.argmax(frame_rms))
    tail = frame_rms[peak_frame:]
    if len(tail) >= 2:
        x = np.arange(len(tail), dtype=np.float64)
        decay_slope = float(np.polyfit(x, np.log(tail + 1e-8), 1)[0])
    else:
        decay_slope = 0.0

    blocks = [
        frame_statistics(log_mel),
        frame_statistics(mfcc),
        frame_statistics(centroid),
        frame_statistics(bandwidth),
        frame_statistics(rolloff),
        frame_statistics(zcr),
        frame_statistics(rms),
        frame_statistics(band_ratios_array),
        np.asarray([original_peak, original_rms, math.log1p(original_energy), decay_slope], dtype=np.float32),
    ]
    features = np.concatenate(blocks)
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the best handcrafted-feature RBF-SVM model.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("best_model_output"))
    args = parser.parse_args()

    cfg = Config()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.dataset_root.resolve())
    split = create_grouped_split(manifest, cfg.seed)
    features = []
    for index, row in split.iterrows():
        features.append(traditional_features(read_event_clip(row, cfg), cfg))
        if (index + 1) % 100 == 0 or index + 1 == len(split):
            print(f"Prepared {index + 1}/{len(split)} samples", flush=True)
    features_array = np.stack(features)
    labels = split["label_id"].to_numpy(dtype=np.int64)
    train_indices = np.flatnonzero(split["split"].to_numpy() == "train")
    test_indices = np.flatnonzero(split["split"].to_numpy() == "test")

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced", probability=True, random_state=cfg.seed)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        },
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg.seed),
        n_jobs=-1,
        refit=True,
    )
    start = time.time()
    search.fit(features_array[train_indices], labels[train_indices])
    y_pred = search.predict(features_array[test_indices])
    probabilities = search.predict_proba(features_array[test_indices])[:, 1]
    metrics = calculate_metrics(labels[test_indices], y_pred, probabilities)
    metrics.update(
        {
            "best_parameters": search.best_params_,
            "cross_validation_macro_f1": float(search.best_score_),
            "training_seconds": time.time() - start,
            "feature_dimension": int(features_array.shape[1]),
            "dataset_samples": int(len(split)),
            "train_samples": int(len(train_indices)),
            "test_samples": int(len(test_indices)),
            "config": asdict(cfg),
        }
    )

    joblib.dump(search.best_estimator_, args.output_dir / "best_model.joblib")
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    export_split = split.drop(columns=["audio_path"])
    export_split.to_csv(args.output_dir / "dataset_split.csv", index=False, encoding="utf-8-sig")
    predictions = export_split.iloc[test_indices][
        ["sample_id", "label", "source_id", "relative_audio_path"]
    ].copy()
    predictions["predicted_label"] = [ID_TO_LABEL[int(value)] for value in y_pred]
    predictions["fly_ball_probability"] = probabilities
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
