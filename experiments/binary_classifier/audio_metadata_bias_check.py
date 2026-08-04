#!/usr/bin/env python3
"""Measure how much label signal exists in non-audio collection metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    with (
        ROOT / "reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv"
    ).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    x = []
    y = []
    groups = []
    for row in rows:
        source = {}
        for line in (ROOT / row["dataset_path"] / "source.txt").read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                source[key.strip()] = value.strip()
        x.append(
            [
                urlparse(source.get("video_url", "")).netloc,
                row["collector"],
                row["verification_source"],
            ]
        )
        y.append(1 if row["label"] == "fly_ball" else 0)
        groups.append(source.get("source_id") or source.get("video_url") or row["dataset_path"])
    x = np.asarray(x)
    y = np.asarray(y)
    groups = np.asarray(groups)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260805)
    results = []
    for name, columns in (
        ("source_domain", 1),
        ("source_domain_plus_collector", 2),
        ("domain_collector_plus_review_route", 3),
    ):
        model = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            LogisticRegression(class_weight="balanced", max_iter=1000),
        )
        prediction = cross_val_predict(
            model, x[:, :columns], y, groups=groups, cv=cv, method="predict"
        )
        results.append(
            {
                "metadata": name,
                "accuracy": float(accuracy_score(y, prediction)),
                "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            }
        )
    (RESULT_DIR / "audio_metadata_bias_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
