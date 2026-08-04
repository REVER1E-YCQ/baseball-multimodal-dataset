#!/usr/bin/env python3
"""Leakage-aware audio baselines for fly-ball vs ground-ball classification."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import zoom
from scipy.signal import resample, resample_poly, stft
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


LABEL_TO_INT = {"ground_ball": 0, "fly_ball": 1}
SOURCE_LINE = re.compile(r"^(source_id|video_url):\s*(.+?)\s*$", re.MULTILINE)


def read_manifest(repo: Path, manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        sample_dir = repo / Path(row["dataset_path"])
        row["audio_path"] = str(sample_dir / "audio.wav")
        row["source_path"] = str(sample_dir / "source.txt")
        source_text = (sample_dir / "source.txt").read_text(encoding="utf-8-sig")
        values = {key: value for key, value in SOURCE_LINE.findall(source_text)}
        row["source_group"] = values.get("source_id") or values.get("video_url") or row["dataset_path"]
        row["target"] = str(LABEL_TO_INT[row["label"]])
    return rows


def make_splits(rows: list[dict[str, str]], seed: int) -> dict[str, str]:
    y = np.asarray([int(row["target"]) for row in rows])
    groups = np.asarray([row["source_group"] for row in rows])
    indices = np.arange(len(rows))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(outer.split(indices, y, groups))
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + 1)
    train_rel, val_rel = next(
        inner.split(train_val_idx, y[train_val_idx], groups[train_val_idx])
    )
    split = {}
    for idx in train_val_idx[train_rel]:
        split[rows[int(idx)]["dataset_path"]] = "train"
    for idx in train_val_idx[val_rel]:
        split[rows[int(idx)]["dataset_path"]] = "val"
    for idx in test_idx:
        split[rows[int(idx)]["dataset_path"]] = "test"
    return split


def to_float_mono(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if np.issubdtype(samples.dtype, np.integer):
        scale = max(abs(np.iinfo(samples.dtype).min), np.iinfo(samples.dtype).max)
        samples = samples.astype(np.float32) / float(scale)
    else:
        samples = samples.astype(np.float32)
    return np.nan_to_num(samples)


def resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples
    divisor = math.gcd(source_rate, target_rate)
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


def fixed_window(samples: np.ndarray, rate: int, center: float, seconds: float) -> np.ndarray:
    length = int(round(seconds * rate))
    center_index = int(round(center * rate))
    start = center_index - length // 2
    end = start + length
    output = np.zeros(length, dtype=np.float32)
    source_start = max(0, start)
    source_end = min(len(samples), end)
    if source_end > source_start:
        output[source_start - start : source_end - start] = samples[source_start:source_end]
    return output


def background_center(duration: float, contact_center: float, window_seconds: float) -> float:
    left = window_seconds / 2
    right = max(left, duration - window_seconds / 2)
    return left if abs(contact_center - left) >= abs(contact_center - right) else right


def normalize_window(samples: np.ndarray) -> np.ndarray:
    samples = samples - float(np.mean(samples))
    rms = float(np.sqrt(np.mean(np.square(samples))) + 1e-8)
    return np.clip(samples / rms, -12.0, 12.0).astype(np.float32)


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (np.power(10.0, np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(rate: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_points = np.linspace(hz_to_mel(60.0), hz_to_mel(rate / 2), n_mels + 2)
    bins = np.floor((n_fft + 1) * mel_to_hz(mel_points) / rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    bank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = bins[index : index + 3]
        if center > left:
            bank[index, left:center] = np.arange(left, center) / (center - left)
        if right > center:
            bank[index, center:right] = (right - np.arange(center, right)) / (right - center)
    return bank


def logmel_feature(samples: np.ndarray, rate: int, n_mels: int = 48, frames: int = 48) -> np.ndarray:
    _freq, _time, spectrum = stft(
        samples,
        fs=rate,
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(spectrum)).astype(np.float32)
    mel = mel_filterbank(rate, 512, n_mels) @ power
    mel = np.log1p(10.0 * mel)
    if mel.shape[1] != frames:
        mel = zoom(mel, (1.0, frames / mel.shape[1]), order=1)
    mel = mel[:, :frames]
    return mel.reshape(-1).astype(np.float32)


def waveform_feature(samples: np.ndarray, points: int = 2048) -> np.ndarray:
    return resample(samples, points).astype(np.float32)


def extract_features(
    rows: list[dict[str, str]], rate: int, window_seconds: float
) -> dict[str, np.ndarray]:
    features = {
        "contact_logmel": [],
        "contact_waveform": [],
        "masked_contact_logmel": [],
        "background_logmel": [],
    }
    for index, row in enumerate(rows, start=1):
        source_rate, raw = wavfile.read(row["audio_path"])
        audio = resample_audio(to_float_mono(raw), int(source_rate), rate)
        start = float(row["final_event_start"])
        end = float(row["final_event_end"])
        center = (start + end) / 2
        contact = normalize_window(fixed_window(audio, rate, center, window_seconds))
        masked = contact.copy()
        half_mask = int(round(0.12 * rate))
        midpoint = len(masked) // 2
        masked[midpoint - half_mask : midpoint + half_mask] = 0.0
        bg_center = background_center(len(audio) / rate, center, window_seconds)
        background = normalize_window(fixed_window(audio, rate, bg_center, window_seconds))
        features["contact_logmel"].append(logmel_feature(contact, rate))
        features["contact_waveform"].append(waveform_feature(contact))
        features["masked_contact_logmel"].append(logmel_feature(masked, rate))
        features["background_logmel"].append(logmel_feature(background, rate))
        if index % 100 == 0:
            print(f"features={index}/{len(rows)}", flush=True)
    return {name: np.stack(values) for name, values in features.items()}


def metrics(y_true: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "confusion_matrix_ground_fly": confusion_matrix(y_true, predictions, labels=[0, 1]).tolist(),
    }


def train_and_evaluate(
    name: str,
    x: np.ndarray,
    y: np.ndarray,
    split_names: np.ndarray,
) -> tuple[dict, np.ndarray, np.ndarray]:
    train = split_names == "train"
    val = split_names == "val"
    test = split_names == "test"
    candidates = []
    for c_value in (0.01, 0.1, 1.0, 10.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                max_iter=3000,
                class_weight="balanced",
                solver="liblinear",
                random_state=0,
            ),
        )
        model.fit(x[train], y[train])
        val_predictions = model.predict(x[val])
        candidates.append((f1_score(y[val], val_predictions, average="macro"), c_value, model))
    candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    val_f1, c_value, _ = candidates[0]
    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=3000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0,
        ),
    )
    final_model.fit(x[train | val], y[train | val])
    predictions = final_model.predict(x[test])
    scores = final_model.decision_function(x[test])
    result = {
        "feature_set": name,
        "selected_C": c_value,
        "validation_macro_f1": float(val_f1),
        **metrics(y[test], predictions, scores),
    }
    all_predictions = np.full(len(y), -1, dtype=int)
    all_scores = np.full(len(y), np.nan, dtype=float)
    all_predictions[test] = predictions
    all_scores[test] = scores
    return result, all_predictions, all_scores


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/binary_classifier/results"))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir

    rows = read_manifest(repo, manifest)
    split = make_splits(rows, args.seed)
    split_names = np.asarray([split[row["dataset_path"]] for row in rows])
    y = np.asarray([int(row["target"]) for row in rows])
    output_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features(rows, args.sample_rate, args.window_seconds)
    np.savez_compressed(
        output_dir / "audio_features.npz",
        dataset_paths=np.asarray([row["dataset_path"] for row in rows]),
        **features,
    )

    split_rows = []
    for row, split_name in zip(rows, split_names):
        split_rows.append(
            {
                "dataset_path": row["dataset_path"],
                "sample_id": row["sample_id"],
                "label": row["label"],
                "source_group": row["source_group"],
                "split": split_name,
            }
        )
    write_csv(
        output_dir / "dataset_split.csv",
        ["dataset_path", "sample_id", "label", "source_group", "split"],
        split_rows,
    )

    results = []
    prediction_columns: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, x in features.items():
        result, predictions, scores = train_and_evaluate(name, x, y, split_names)
        results.append(result)
        prediction_columns[name] = (predictions, scores)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    test_mask = split_names == "test"
    majority = Counter(y[split_names == "train"]).most_common(1)[0][0]
    majority_predictions = np.full(int(test_mask.sum()), majority)
    majority_scores = np.full(int(test_mask.sum()), float(majority))
    majority_result = {
        "feature_set": "majority_baseline",
        "selected_C": None,
        "validation_macro_f1": None,
        **metrics(y[test_mask], majority_predictions, majority_scores),
    }
    results.append(majority_result)

    result_fields = list(results[0].keys())
    write_csv(output_dir / "audio_baseline_results.csv", result_fields, results)
    (output_dir / "audio_baseline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_rows = []
    for index, row in enumerate(rows):
        if not test_mask[index]:
            continue
        output = {
            "dataset_path": row["dataset_path"],
            "sample_id": row["sample_id"],
            "label": row["label"],
            "source_group": row["source_group"],
        }
        for name, (predictions, scores) in prediction_columns.items():
            output[f"{name}_prediction"] = (
                "fly_ball" if predictions[index] == 1 else "ground_ball"
            )
            output[f"{name}_score"] = scores[index]
        prediction_rows.append(output)
    write_csv(output_dir / "test_predictions.csv", list(prediction_rows[0]), prediction_rows)

    summary = {
        "seed": args.seed,
        "sample_rate": args.sample_rate,
        "window_seconds": args.window_seconds,
        "samples": len(rows),
        "source_groups": len({row["source_group"] for row in rows}),
        "split_counts": Counter(split_names.tolist()),
        "split_label_counts": {
            split_name: Counter(row["label"] for row in rows if split[row["dataset_path"]] == split_name)
            for split_name in ("train", "val", "test")
        },
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=dict) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
