#!/usr/bin/env python3
"""Five-fold source-grouped evaluation of validation-selected late fusion."""

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


def main() -> None:
    rows = load_rows()
    paths = np.asarray([row["dataset_path"] for row in rows])
    y = np.asarray([int(row["target"]) for row in rows])
    groups = np.asarray(
        [
            next(
                line.split(":", 1)[1].strip()
                for line in (Path(row["dataset_path"]) / "source.txt").read_text(
                    encoding="utf-8-sig"
                ).splitlines()
                if line.startswith(("source_id:", "video_url:"))
            )
            for row in rows
        ]
    )

    audio_cache = np.load(RESULT_DIR / "audio_features.npz")
    video_cache = np.load(RESULT_DIR / "video_features.npz")
    if not np.array_equal(paths, audio_cache["dataset_paths"]) or not np.array_equal(
        paths, video_cache["dataset_paths"]
    ):
        raise RuntimeError("Feature cache order does not match the manifest")
    audio = audio_variants(audio_cache)
    video = video_cache["video_combined"]

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260804)
    video_probability = np.full(len(rows), np.nan)
    fusion_probability = np.full(len(rows), np.nan)
    fusion_threshold = np.full(len(rows), np.nan)
    fold_reports = []
    all_indices = np.arange(len(rows))

    for fold, (outer_train_idx, outer_test_idx) in enumerate(
        outer.split(all_indices, y, groups), start=1
    ):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20260804 + fold)
        inner_train_rel, inner_val_rel = next(
            inner.split(
                outer_train_idx,
                y[outer_train_idx],
                groups[outer_train_idx],
            )
        )
        train = np.zeros(len(rows), dtype=bool)
        val = np.zeros(len(rows), dtype=bool)
        outer_train = np.zeros(len(rows), dtype=bool)
        test = np.zeros(len(rows), dtype=bool)
        train[outer_train_idx[inner_train_rel]] = True
        val[outer_train_idx[inner_val_rel]] = True
        outer_train[outer_train_idx] = True
        test[outer_test_idx] = True

        video_c, _model, video_val = tune_base(video, y, train, val)
        audio_c = {}
        audio_val = {}
        for name, values in audio.items():
            c_value, _model, probability = tune_base(values, y, train, val)
            audio_c[name] = c_value
            audio_val[name] = probability
        fusion = tune_fusion(y[val], video_val, audio_val)
        audio_name = fusion["audio_name"]

        video_model = make_model(video_c)
        video_model.fit(video[outer_train], y[outer_train])
        fold_video = video_model.predict_proba(video[test])[:, 1]
        audio_model = make_model(audio_c[audio_name])
        audio_model.fit(audio[audio_name][outer_train], y[outer_train])
        fold_audio = audio_model.predict_proba(audio[audio_name][test])[:, 1]
        fold_fusion, threshold = apply_fusion(fusion, fold_video, fold_audio)
        video_probability[test] = fold_video
        fusion_probability[test] = fold_fusion
        fusion_threshold[test] = threshold
        fold_report = {
            "fold": fold,
            "test_size": int(test.sum()),
            "video_C": video_c,
            "selected_audio": audio_name,
            "audio_C": audio_c[audio_name],
            "fusion": fusion,
            "video_metrics": metric(y[test], fold_video),
            "fusion_metrics": metric(y[test], fold_fusion, threshold),
        }
        fold_reports.append(fold_report)
        print(json.dumps(fold_report, ensure_ascii=False), flush=True)

    if np.isnan(video_probability).any() or np.isnan(fusion_probability).any():
        raise RuntimeError("Cross-validation did not predict every sample")
    video_prediction = video_probability >= 0.5
    fusion_prediction = fusion_probability >= fusion_threshold
    overall = {
        "protocol": "5-fold stratified source-group cross-validation with inner validation",
        "samples": len(rows),
        "video_metrics": metric(y, video_probability),
        "fusion_metrics": {
            "accuracy": float((fusion_prediction == y).mean()),
            "balanced_accuracy": float(
                0.5
                * (
                    (fusion_prediction[y == 0] == 0).mean()
                    + (fusion_prediction[y == 1] == 1).mean()
                )
            ),
        },
        "video_correct": int((video_prediction == y).sum()),
        "fusion_correct": int((fusion_prediction == y).sum()),
        "net_correct_change": int((fusion_prediction == y).sum() - (video_prediction == y).sum()),
        "folds_improved": sum(
            report["fusion_metrics"]["balanced_accuracy"]
            > report["video_metrics"]["balanced_accuracy"]
            for report in fold_reports
        ),
        "folds_tied": sum(
            report["fusion_metrics"]["balanced_accuracy"]
            == report["video_metrics"]["balanced_accuracy"]
            for report in fold_reports
        ),
    }
    output = {"overall": overall, "folds": fold_reports}
    (RESULT_DIR / "crossval_late_fusion_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
