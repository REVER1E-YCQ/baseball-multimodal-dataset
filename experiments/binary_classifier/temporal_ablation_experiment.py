#!/usr/bin/env python3
"""Measure audio gain as progressively more post-contact video becomes available."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from late_fusion_experiment import (
    RESULT_DIR,
    apply_fusion,
    audio_variants,
    load_rows,
    make_model,
    metric,
    tune_base,
    tune_fusion,
)


ROOT = Path(__file__).resolve().parents[2]


def source_groups(rows: list[dict[str, str]]) -> np.ndarray:
    values = []
    for row in rows:
        text = (ROOT / row["dataset_path"] / "source.txt").read_text(encoding="utf-8-sig")
        source = next(
            line.split(":", 1)[1].strip()
            for line in text.splitlines()
            if line.startswith(("source_id:", "video_url:"))
        )
        values.append(source)
    return np.asarray(values)


def evaluate_setting(
    name: str,
    video: np.ndarray,
    audio: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260804)
    video_probability = np.full(len(y), np.nan)
    audio_probability = np.full(len(y), np.nan)
    fusion_probability = np.full(len(y), np.nan)
    fusion_threshold = np.full(len(y), np.nan)
    fold_reports = []
    indices = np.arange(len(y))
    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(indices, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260900 + fold)
        inner_train_rel, val_rel = next(
            inner.split(outer_train_idx, y[outer_train_idx], groups[outer_train_idx])
        )
        train = np.zeros(len(y), dtype=bool)
        val = np.zeros(len(y), dtype=bool)
        outer_train = np.zeros(len(y), dtype=bool)
        test = np.zeros(len(y), dtype=bool)
        train[outer_train_idx[inner_train_rel]] = True
        val[outer_train_idx[val_rel]] = True
        outer_train[outer_train_idx] = True
        test[test_idx] = True

        video_c, _model, video_val = tune_base(video, y, train, val)
        audio_c, _model, audio_val = tune_base(audio, y, train, val)
        fusion = tune_fusion(y[val], video_val, {"contact_minus_background": audio_val})

        video_model = make_model(video_c)
        video_model.fit(video[outer_train], y[outer_train])
        fold_video = video_model.predict_proba(video[test])[:, 1]
        audio_model = make_model(audio_c)
        audio_model.fit(audio[outer_train], y[outer_train])
        fold_audio = audio_model.predict_proba(audio[test])[:, 1]
        fold_fusion, threshold = apply_fusion(fusion, fold_video, fold_audio)

        video_probability[test] = fold_video
        audio_probability[test] = fold_audio
        fusion_probability[test] = fold_fusion
        fusion_threshold[test] = threshold
        fold_reports.append(
            {
                "fold": fold,
                "video_balanced_accuracy": metric(y[test], fold_video)["balanced_accuracy"],
                "audio_balanced_accuracy": metric(y[test], fold_audio)["balanced_accuracy"],
                "fusion_balanced_accuracy": metric(y[test], fold_fusion, threshold)[
                    "balanced_accuracy"
                ],
                "fusion": fusion,
            }
        )

    video_prediction = video_probability >= 0.5
    audio_prediction = audio_probability >= 0.5
    fusion_prediction = fusion_probability >= fusion_threshold
    video_balanced = metric(y, video_probability)["balanced_accuracy"]
    audio_balanced = metric(y, audio_probability)["balanced_accuracy"]
    fusion_balanced = 0.5 * (
        (fusion_prediction[y == 0] == 0).mean() + (fusion_prediction[y == 1] == 1).mean()
    )
    rng = np.random.default_rng(20260804)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    bootstrap = {"video": [], "audio": [], "fusion": []}
    for _ in range(10_000):
        sample = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        for model_name, prediction in (
            ("video", video_prediction),
            ("audio", audio_prediction),
            ("fusion", fusion_prediction),
        ):
            score = 0.5 * (
                (prediction[sample][y[sample] == 0] == 0).mean()
                + (prediction[sample][y[sample] == 1] == 1).mean()
            )
            bootstrap[model_name].append(score)
    fusion_minus_video = np.asarray(bootstrap["fusion"]) - np.asarray(bootstrap["video"])
    fusion_minus_audio = np.asarray(bootstrap["fusion"]) - np.asarray(bootstrap["audio"])
    return {
        "setting": name,
        "protocol": "5-fold source-grouped outer test with inner validation",
        "video_balanced_accuracy": float(video_balanced),
        "audio_balanced_accuracy": float(audio_balanced),
        "fusion_balanced_accuracy": float(fusion_balanced),
        "fusion_minus_video": float(fusion_balanced - video_balanced),
        "fusion_minus_audio": float(fusion_balanced - audio_balanced),
        "paired_bootstrap": {
            "fusion_minus_video_ci95": np.quantile(
                fusion_minus_video, [0.025, 0.975]
            ).tolist(),
            "fusion_minus_video_probability_positive": float(
                np.mean(fusion_minus_video > 0)
            ),
            "fusion_minus_audio_ci95": np.quantile(
                fusion_minus_audio, [0.025, 0.975]
            ).tolist(),
            "fusion_minus_audio_probability_positive": float(
                np.mean(fusion_minus_audio > 0)
            ),
        },
        "video_correct": int((video_prediction == y).sum()),
        "audio_correct": int((audio_prediction == y).sum()),
        "fusion_correct": int((fusion_prediction == y).sum()),
        "folds_fusion_beats_video": sum(
            fold["fusion_balanced_accuracy"] > fold["video_balanced_accuracy"]
            for fold in fold_reports
        ),
        "folds": fold_reports,
    }


def main() -> None:
    rows = load_rows()
    y = np.asarray([int(row["target"]) for row in rows])
    groups = source_groups(rows)
    audio_cache = np.load(RESULT_DIR / "audio_features.npz")
    video_cache = np.load(RESULT_DIR / "video_features.npz")
    audio = audio_variants(audio_cache)["contact_minus_background"]
    appearance = video_cache["video_appearance"]
    pixels_per_frame = appearance.shape[1] // 4
    frame_t0 = appearance[:, :pixels_per_frame]
    frame_t0p45 = appearance[:, pixels_per_frame : 2 * pixels_per_frame]
    settings = {
        "contact_frame_only_t0": frame_t0,
        "early_video_t0_to_0p45": np.concatenate([frame_t0, frame_t0p45], axis=1),
        "early_video_t0_to_0p45_with_motion": np.concatenate(
            [frame_t0, frame_t0p45, np.abs(frame_t0p45 - frame_t0)], axis=1
        ),
        "full_video_t0_to_1p35_with_motion": video_cache["video_combined"],
    }
    results = []
    for name, video in settings.items():
        result = evaluate_setting(name, video, audio, y, groups)
        results.append(result)
        print(json.dumps({key: value for key, value in result.items() if key != "folds"}, indent=2))
    (RESULT_DIR / "temporal_ablation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
