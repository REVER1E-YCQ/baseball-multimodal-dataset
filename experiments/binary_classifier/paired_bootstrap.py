#!/usr/bin/env python3
"""Paired stratified bootstrap intervals for baseline model comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["dataset_path"]: row for row in csv.DictReader(handle)}


def prediction(rows: dict[str, dict[str, str]], keys: list[str], column: str) -> np.ndarray:
    return np.asarray([1 if rows[key][column] == "fly_ball" else 0 for key in keys])


def main() -> None:
    result_dir = Path(__file__).resolve().parent / "results"
    audio = read_rows(result_dir / "test_predictions.csv")
    multimodal = read_rows(result_dir / "multimodal_test_predictions.csv")
    keys = sorted(set(audio) & set(multimodal))
    y = np.asarray([1 if audio[key]["label"] == "fly_ball" else 0 for key in keys])
    models = {
        "audio_contact": prediction(audio, keys, "contact_logmel_prediction"),
        "video_combined": prediction(multimodal, keys, "video_combined_prediction"),
        "audio_video": prediction(multimodal, keys, "audio_video_early_fusion_prediction"),
        "background_audio_video": prediction(
            multimodal, keys, "background_audio_video_control_prediction"
        ),
    }
    rng = np.random.default_rng(20260804)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    bootstrap = {name: [] for name in models}
    for _ in range(10_000):
        indices = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        for name, values in models.items():
            bootstrap[name].append(balanced_accuracy_score(y[indices], values[indices]))

    output = {"models": {}, "paired_deltas": {}}
    for name, values in models.items():
        distribution = np.asarray(bootstrap[name])
        output["models"][name] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, values)),
            "ci95": np.quantile(distribution, [0.025, 0.975]).tolist(),
        }
    for left, right in (
        ("video_combined", "audio_contact"),
        ("audio_video", "video_combined"),
        ("background_audio_video", "video_combined"),
    ):
        delta = np.asarray(bootstrap[left]) - np.asarray(bootstrap[right])
        output["paired_deltas"][f"{left}_minus_{right}"] = {
            "mean": float(delta.mean()),
            "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "probability_positive": float(np.mean(delta > 0)),
        }
    (result_dir / "paired_bootstrap.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
