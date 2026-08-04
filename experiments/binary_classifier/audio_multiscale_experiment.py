#!/usr/bin/env python3
"""Combine short contact and long context audio features in one classifier."""

from __future__ import annotations

import json

import numpy as np

from audio_baseline import read_manifest
from audio_context_experiment import RESULT_DIR, ROOT, evaluate, feature_views
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
        for name in (
            "contact_0p5s",
            "contact_1p0s",
            "early_context_2p0s",
            "extended_context_4p0s",
            "full_clip",
        )
    }
    features = {
        "full_map_plus_4s_summary": np.concatenate(
            [views["full_clip"]["map"], views["extended_context_4p0s"]["summary"]], axis=1
        ),
        "full_map_plus_contact_summary": np.concatenate(
            [views["full_clip"]["map"], views["contact_0p5s"]["summary"]], axis=1
        ),
        "full_map_plus_4s_and_contact_summary": np.concatenate(
            [
                views["full_clip"]["map"],
                views["extended_context_4p0s"]["summary"],
                views["contact_0p5s"]["summary"],
            ],
            axis=1,
        ),
        "all_window_summaries": np.concatenate(
            [views[name]["summary"] for name in views], axis=1
        ),
    }
    results = []
    for name, values in features.items():
        result = evaluate(name, "rbf_svm", values, y, groups)
        results.append(result)
        print(
            json.dumps({key: value for key, value in result.items() if key != "folds"}),
            flush=True,
        )
    results.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    (RESULT_DIR / "audio_multiscale_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("BEST", json.dumps({k: v for k, v in results[0].items() if k != "folds"}))


if __name__ == "__main__":
    main()
