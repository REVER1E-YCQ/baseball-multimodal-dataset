#!/usr/bin/env python3
"""Validation-selected audio representations and conservative late fusion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "results"


def load_rows() -> list[dict[str, str]]:
    manifest_path = ROOT / "reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv"
    split_path = RESULT_DIR / "dataset_split.csv"
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        splits = {row["dataset_path"]: row["split"] for row in csv.DictReader(handle)}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["split"] = splits[row["dataset_path"]]
        row["target"] = "1" if row["label"] == "fly_ball" else "0"
    return rows


def audio_variants(cache: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    contact = cache["contact_logmel"].reshape(-1, 48, 48)
    background = cache["background_logmel"].reshape(-1, 48, 48)
    waveform = cache["contact_waveform"]
    center_mel = contact[:, :, 17:31].reshape(len(contact), -1)
    side_mean = np.concatenate([contact[:, :, :12], contact[:, :, -12:]], axis=2).mean(axis=2)
    center_mean = contact[:, :, 20:28].mean(axis=2)
    transient_contrast = np.concatenate([center_mel, center_mean - side_mean], axis=1)
    midpoint = waveform.shape[1] // 2
    center_waveform = waveform[:, midpoint - 256 : midpoint + 256]
    center_spectrum = np.log1p(np.abs(np.fft.rfft(center_waveform, axis=1))).astype(np.float32)
    return {
        "contact_logmel": cache["contact_logmel"],
        "contact_minus_background": (contact - background).reshape(len(contact), -1),
        "transient_contrast": transient_contrast,
        "center_waveform": center_waveform,
        "center_spectrum": center_spectrum,
    }


def make_model(c_value: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=0,
        ),
    )


def metric(y: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "roc_auc": float(roc_auc_score(y, probabilities)),
    }


def tune_base(
    x: np.ndarray, y: np.ndarray, train: np.ndarray, val: np.ndarray
) -> tuple[float, object, np.ndarray]:
    candidates = []
    for c_value in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        model = make_model(c_value)
        model.fit(x[train], y[train])
        probability = model.predict_proba(x[val])[:, 1]
        score = metric(y[val], probability)
        candidates.append((score["macro_f1"], score["roc_auc"], -c_value, c_value, model, probability))
    _f1, _auc, _negative_c, c_value, model, probability = max(candidates, key=lambda item: item[:3])
    return c_value, model, probability


def tune_fusion(
    y: np.ndarray,
    video_probability: np.ndarray,
    audio_candidates: dict[str, np.ndarray],
) -> dict:
    candidates = []
    for audio_name, audio_probability in audio_candidates.items():
        for weight in np.linspace(0.0, 0.5, 21):
            fused = (1.0 - weight) * video_probability + weight * audio_probability
            for threshold in np.linspace(0.40, 0.60, 21):
                score = metric(y, fused, float(threshold))
                candidates.append(
                    (
                        score["macro_f1"],
                        score["balanced_accuracy"],
                        score["roc_auc"],
                        -abs(float(threshold) - 0.5),
                        -float(weight),
                        {
                            "method": "weighted_probability",
                            "audio_name": audio_name,
                            "audio_weight": float(weight),
                            "threshold": float(threshold),
                            "validation_metrics": score,
                        },
                    )
                )

        video_prediction = video_probability >= 0.5
        audio_prediction = audio_probability >= 0.5
        video_confidence = np.abs(video_probability - 0.5) * 2.0
        audio_confidence = np.abs(audio_probability - 0.5) * 2.0
        for audio_min in np.linspace(0.55, 0.95, 9):
            for video_max in np.linspace(0.10, 0.70, 13):
                use_audio = (
                    (video_prediction != audio_prediction)
                    & (audio_confidence >= audio_min)
                    & (video_confidence <= video_max)
                )
                gated = np.where(use_audio, audio_probability, video_probability)
                score = metric(y, gated)
                candidates.append(
                    (
                        score["macro_f1"],
                        score["balanced_accuracy"],
                        score["roc_auc"],
                        -float(audio_min),
                        -float(video_max),
                        {
                            "method": "confidence_gate",
                            "audio_name": audio_name,
                            "audio_min_confidence": float(audio_min),
                            "video_max_confidence": float(video_max),
                            "validation_metrics": score,
                        },
                    )
                )
    return max(candidates, key=lambda item: item[:5])[-1]


def apply_fusion(config: dict, video_probability: np.ndarray, audio_probability: np.ndarray):
    if config["method"] == "weighted_probability":
        weight = config["audio_weight"]
        probability = (1.0 - weight) * video_probability + weight * audio_probability
        threshold = config["threshold"]
        return probability, threshold
    video_prediction = video_probability >= 0.5
    audio_prediction = audio_probability >= 0.5
    video_confidence = np.abs(video_probability - 0.5) * 2.0
    audio_confidence = np.abs(audio_probability - 0.5) * 2.0
    use_audio = (
        (video_prediction != audio_prediction)
        & (audio_confidence >= config["audio_min_confidence"])
        & (video_confidence <= config["video_max_confidence"])
    )
    return np.where(use_audio, audio_probability, video_probability), 0.5


def main() -> None:
    rows = load_rows()
    paths = np.asarray([row["dataset_path"] for row in rows])
    y = np.asarray([int(row["target"]) for row in rows])
    split = np.asarray([row["split"] for row in rows])
    train, val, test = split == "train", split == "val", split == "test"
    train_val = train | val

    audio_cache = np.load(RESULT_DIR / "audio_features.npz")
    video_cache = np.load(RESULT_DIR / "video_features.npz")
    if not np.array_equal(paths, audio_cache["dataset_paths"]) or not np.array_equal(
        paths, video_cache["dataset_paths"]
    ):
        raise RuntimeError("Feature cache order does not match the manifest")
    audio = audio_variants(audio_cache)
    video = video_cache["video_combined"]

    video_c, _video_model, video_val = tune_base(video, y, train, val)
    audio_tuning = {}
    audio_val_probabilities = {}
    for name, values in audio.items():
        c_value, _model, probability = tune_base(values, y, train, val)
        audio_tuning[name] = c_value
        audio_val_probabilities[name] = probability

    fusion = tune_fusion(y[val], video_val, audio_val_probabilities)
    chosen_audio = fusion["audio_name"]

    video_model = make_model(video_c)
    video_model.fit(video[train_val], y[train_val])
    video_test = video_model.predict_proba(video[test])[:, 1]
    audio_model = make_model(audio_tuning[chosen_audio])
    audio_model.fit(audio[chosen_audio][train_val], y[train_val])
    audio_test = audio_model.predict_proba(audio[chosen_audio][test])[:, 1]
    fused_test, threshold = apply_fusion(fusion, video_test, audio_test)

    output = {
        "selection_policy": "all representation and fusion choices made on validation only",
        "video_C": video_c,
        "audio_C_by_representation": audio_tuning,
        "selected_fusion": fusion,
        "test": {
            "video_only": metric(y[test], video_test),
            "selected_audio_only": metric(y[test], audio_test),
            "late_fusion": metric(y[test], fused_test, threshold),
        },
    }
    (RESULT_DIR / "late_fusion_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    prediction_rows = []
    test_indices = np.flatnonzero(test)
    for local_index, global_index in enumerate(test_indices):
        prediction_rows.append(
            {
                "dataset_path": paths[global_index],
                "label": rows[global_index]["label"],
                "video_probability_fly": video_test[local_index],
                "audio_probability_fly": audio_test[local_index],
                "fusion_probability_fly": fused_test[local_index],
                "fusion_prediction": "fly_ball"
                if fused_test[local_index] >= threshold
                else "ground_ball",
            }
        )
    with (RESULT_DIR / "late_fusion_test_predictions.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
