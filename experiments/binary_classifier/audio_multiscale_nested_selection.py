#!/usr/bin/env python3
"""Nested selection of multiscale audio features and SVM parameters."""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from audio_baseline import read_manifest
from audio_context_experiment import (
    RESULT_DIR,
    ROOT,
    build_model,
    candidates,
    feature_views,
    score,
)
from temporal_ablation_experiment import source_groups


def main() -> None:
    rows = read_manifest(
        ROOT,
        ROOT / "reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv",
    )
    y = np.asarray([1 if row["label"] == "fly_ball" else 0 for row in rows])
    groups = source_groups(rows)
    cache = np.load(RESULT_DIR / "audio_context_features.npz")
    views = {
        name: feature_views(cache[name])
        for name in ("contact_0p5s", "extended_context_4p0s", "full_clip")
    }
    feature_sets = {
        "full_map": views["full_clip"]["map"],
        "full_plus_4s": np.concatenate(
            [views["full_clip"]["map"], views["extended_context_4p0s"]["summary"]], axis=1
        ),
        "full_plus_contact": np.concatenate(
            [views["full_clip"]["map"], views["contact_0p5s"]["summary"]], axis=1
        ),
        "full_plus_4s_plus_contact": np.concatenate(
            [
                views["full_clip"]["map"],
                views["extended_context_4p0s"]["summary"],
                views["contact_0p5s"]["summary"],
            ],
            axis=1,
        ),
    }
    predictions = np.full(len(y), -1, dtype=int)
    decisions = np.full(len(y), np.nan)
    fold_reports = []
    indices = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260805)
    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(indices, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20261300 + fold)
        train_rel, val_rel = next(
            inner.split(outer_train_idx, y[outer_train_idx], groups[outer_train_idx])
        )
        train_idx = outer_train_idx[train_rel]
        val_idx = outer_train_idx[val_rel]
        ranked = []
        for feature_name, values in feature_sets.items():
            for params, model in candidates("rbf_svm"):
                model.fit(values[train_idx], y[train_idx])
                val_prediction = model.predict(values[val_idx])
                val_decision = model.decision_function(values[val_idx])
                validation = score(y[val_idx], val_prediction, val_decision)
                ranked.append(
                    (
                        validation["macro_f1"],
                        validation["roc_auc"],
                        feature_name,
                        params,
                    )
                )
        _f1, _auc, feature_name, params = max(ranked, key=lambda item: item[:2])
        model = build_model("rbf_svm", params)
        model.fit(feature_sets[feature_name][outer_train_idx], y[outer_train_idx])
        predictions[test_idx] = model.predict(feature_sets[feature_name][test_idx])
        decisions[test_idx] = model.decision_function(feature_sets[feature_name][test_idx])
        fold_result = {
            "fold": fold,
            "selected_feature": feature_name,
            "selected_params": params,
            **score(y[test_idx], predictions[test_idx], decisions[test_idx]),
        }
        fold_reports.append(fold_result)
        print(json.dumps(fold_result), flush=True)
    output = {
        "protocol": "5-fold source-grouped outer test with inner feature and parameter selection",
        **score(y, predictions, decisions),
        "fold_accuracy_mean": float(np.mean([row["accuracy"] for row in fold_reports])),
        "fold_accuracy_std": float(np.std([row["accuracy"] for row in fold_reports])),
        "fold_balanced_accuracy_mean": float(
            np.mean([row["balanced_accuracy"] for row in fold_reports])
        ),
        "selected_feature_counts": {
            name: sum(row["selected_feature"] == name for row in fold_reports)
            for name in feature_sets
        },
        "folds": fold_reports,
    }
    (RESULT_DIR / "audio_multiscale_nested_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in output.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
