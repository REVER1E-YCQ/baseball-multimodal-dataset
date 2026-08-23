from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_linear_probe import (
    LABEL_TO_INT,
    calculate_metrics,
    fit_probe,
    load_paired_features,
)


def evaluate_external(
    features_path: Path,
    out_dir: Path,
    train_role: str,
    test_role: str,
    event_window: str,
    pre_window: str,
    c_value: float,
    seed: int,
) -> pd.DataFrame:
    if c_value <= 0:
        raise ValueError("c_value must be positive")
    train = load_paired_features(
        features_path,
        event_window,
        pre_window,
        train_role,
    )
    test = load_paired_features(
        features_path,
        event_window,
        pre_window,
        test_role,
    )
    if train.feature_columns != test.feature_columns:
        raise ValueError("Training and external-test feature columns differ")

    train_indices = np.arange(len(train.labels))
    event_probe = fit_probe(
        train.event,
        train.labels,
        train_indices,
        c_value,
        seed,
    )
    pre_probe = fit_probe(
        train.strict_pre,
        train.labels,
        train_indices,
        c_value,
        seed,
    )
    conditions = {
        "event_selected_event": event_probe.predict(test.event),
        "event_selected_pre": event_probe.predict(test.strict_pre),
        "pre_selected_pre": pre_probe.predict(test.strict_pre),
    }

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for condition, (prediction, score) in conditions.items():
        metric_rows.append(
            {
                "condition": condition,
                **calculate_metrics(test.labels, prediction, score),
            }
        )
        for index in range(len(test.labels)):
            prediction_rows.append(
                {
                    "condition": condition,
                    "uid": test.uids[index],
                    "source_id": test.groups[index],
                    "y_true": int(test.labels[index]),
                    "y_pred": int(prediction[index]),
                    "score": float(score[index]),
                }
            )

    summary = pd.DataFrame(metric_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        out_dir / "external_predictions.csv",
        index=False,
    )
    protocol = {
        "encoder": "feature_file_as_provided",
        "classifier": "StandardScaler + L2 logistic regression",
        "classifier_solver": "liblinear",
        "class_weight": "balanced",
        "c_value": c_value,
        "c_selection": "selected_without_using_external_test",
        "train_role": train_role,
        "test_role": test_role,
        "event_window": event_window,
        "pre_window": pre_window,
        "n_train_paired": int(len(train.labels)),
        "n_test_paired": int(len(test.labels)),
        "train_class_counts": {
            label: int((train.labels == value).sum())
            for label, value in LABEL_TO_INT.items()
        },
        "test_class_counts": {
            label: int((test.labels == value).sum())
            for label, value in LABEL_TO_INT.items()
        },
        "event_solver_iterations": int(event_probe.classifier.n_iter_[0]),
        "pre_solver_iterations": int(pre_probe.classifier.n_iter_[0]),
        "seed": seed,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit a prespecified linear probe on one role and evaluate a disjoint "
            "external role without external-test tuning."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-role", default="primary_dev")
    parser.add_argument("--test-role", default="external_test")
    parser.add_argument("--event-window", default="event_200ms")
    parser.add_argument("--pre-window", default="pre_200ms")
    parser.add_argument("--c-value", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20260716)
    args = parser.parse_args()
    summary = evaluate_external(
        args.features,
        args.out_dir,
        args.train_role,
        args.test_role,
        args.event_window,
        args.pre_window,
        args.c_value,
        args.seed,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
