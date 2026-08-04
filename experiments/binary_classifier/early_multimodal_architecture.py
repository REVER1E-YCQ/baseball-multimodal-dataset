#!/usr/bin/env python3
"""Nested-CV comparison of early feature fusion and late probability fusion."""

from __future__ import annotations

import json

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
from temporal_ablation_experiment import source_groups


def balanced_from_predictions(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(
        0.5
        * ((prediction[y == 0] == 0).mean() + (prediction[y == 1] == 1).mean())
    )


def main() -> None:
    rows = load_rows()
    y = np.asarray([int(row["target"]) for row in rows])
    groups = source_groups(rows)
    audio_cache = np.load(RESULT_DIR / "audio_features.npz")
    video_cache = np.load(RESULT_DIR / "video_features.npz")
    audio = audio_variants(audio_cache)["contact_minus_background"]
    appearance = video_cache["video_appearance"]
    pixels = appearance.shape[1] // 4
    frame_t0 = appearance[:, :pixels]
    frame_t0p45 = appearance[:, pixels : 2 * pixels]
    video = np.concatenate(
        [frame_t0, frame_t0p45, np.abs(frame_t0p45 - frame_t0)], axis=1
    )
    combined = np.concatenate([video, audio], axis=1)

    predictions = {
        "video": np.full(len(y), -1, dtype=int),
        "audio": np.full(len(y), -1, dtype=int),
        "multimodal": np.full(len(y), -1, dtype=int),
    }
    fold_reports = []
    indices = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260804)
    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(indices, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20261000 + fold)
        train_rel, val_rel = next(
            inner.split(outer_train_idx, y[outer_train_idx], groups[outer_train_idx])
        )
        train = np.zeros(len(y), dtype=bool)
        val = np.zeros(len(y), dtype=bool)
        outer_train = np.zeros(len(y), dtype=bool)
        test = np.zeros(len(y), dtype=bool)
        train[outer_train_idx[train_rel]] = True
        val[outer_train_idx[val_rel]] = True
        outer_train[outer_train_idx] = True
        test[test_idx] = True

        video_c, _video, video_val = tune_base(video, y, train, val)
        audio_c, _audio, audio_val = tune_base(audio, y, train, val)
        concat_c, _concat, concat_val = tune_base(combined, y, train, val)
        late_config = tune_fusion(
            y[val], video_val, {"contact_minus_background": audio_val}
        )
        late_val, late_threshold = apply_fusion(late_config, video_val, audio_val)
        concat_score = metric(y[val], concat_val)
        late_score = metric(y[val], late_val, late_threshold)
        selected_method = (
            "feature_concat"
            if (concat_score["macro_f1"], concat_score["roc_auc"])
            >= (late_score["macro_f1"], late_score["roc_auc"])
            else "late_fusion"
        )

        video_model = make_model(video_c)
        video_model.fit(video[outer_train], y[outer_train])
        fold_video = video_model.predict_proba(video[test])[:, 1]
        audio_model = make_model(audio_c)
        audio_model.fit(audio[outer_train], y[outer_train])
        fold_audio = audio_model.predict_proba(audio[test])[:, 1]
        if selected_method == "feature_concat":
            concat_model = make_model(concat_c)
            concat_model.fit(combined[outer_train], y[outer_train])
            fold_multimodal = concat_model.predict_proba(combined[test])[:, 1]
            threshold = 0.5
        else:
            fold_multimodal, threshold = apply_fusion(
                late_config, fold_video, fold_audio
            )
        predictions["video"][test] = (fold_video >= 0.5).astype(int)
        predictions["audio"][test] = (fold_audio >= 0.5).astype(int)
        predictions["multimodal"][test] = (fold_multimodal >= threshold).astype(int)
        fold_reports.append(
            {
                "fold": fold,
                "selected_method": selected_method,
                "video_balanced_accuracy": balanced_from_predictions(
                    y[test], predictions["video"][test]
                ),
                "audio_balanced_accuracy": balanced_from_predictions(
                    y[test], predictions["audio"][test]
                ),
                "multimodal_balanced_accuracy": balanced_from_predictions(
                    y[test], predictions["multimodal"][test]
                ),
            }
        )

    scores = {
        name: balanced_from_predictions(y, values) for name, values in predictions.items()
    }
    rng = np.random.default_rng(20260804)
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    deltas = {"multimodal_minus_video": [], "multimodal_minus_audio": []}
    for _ in range(10_000):
        sample = np.concatenate(
            [rng.choice(values, size=len(values), replace=True) for values in class_indices]
        )
        sample_scores = {
            name: balanced_from_predictions(y[sample], values[sample])
            for name, values in predictions.items()
        }
        deltas["multimodal_minus_video"].append(
            sample_scores["multimodal"] - sample_scores["video"]
        )
        deltas["multimodal_minus_audio"].append(
            sample_scores["multimodal"] - sample_scores["audio"]
        )
    output = {
        "protocol": "5-fold source-grouped outer test; fusion architecture selected by inner validation",
        "video_window": "contact frame through +0.45 seconds, including first frame difference",
        "scores": scores,
        "deltas": {
            name: {
                "point": scores["multimodal"] - scores[name.rsplit("_", 1)[-1]],
                "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
                "probability_positive": float(np.mean(np.asarray(values) > 0)),
            }
            for name, values in deltas.items()
        },
        "folds": fold_reports,
    }
    (RESULT_DIR / "early_multimodal_architecture_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
