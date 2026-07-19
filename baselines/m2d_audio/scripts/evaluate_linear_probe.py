from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


LABEL_TO_INT = {"fly_ball": 0, "ground_ball": 1}
DEFAULT_C_GRID = (0.001, 0.01, 0.1)
REQUIRED_COLUMNS = {
    "uid",
    "label",
    "source_id",
    "protocol_role",
    "window_name",
}


@dataclass(frozen=True)
class PairedFeatures:
    uids: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    event: np.ndarray
    strict_pre: np.ndarray
    feature_columns: tuple[str, ...]
    singleton_groups: int


@dataclass(frozen=True)
class FittedProbe:
    scaler: StandardScaler
    classifier: LogisticRegression

    def predict(self, matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformed = self.scaler.transform(matrix)
        prediction = self.classifier.predict(transformed)
        score = self.classifier.predict_proba(transformed)[:, 1]
        return prediction.astype(int), score.astype(np.float64)


def load_paired_features(
    path: Path,
    event_window: str,
    pre_window: str,
    protocol_role: str,
) -> PairedFeatures:
    frame = pd.read_csv(path, low_memory=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature file is missing columns: {sorted(missing)}")
    feature_columns = tuple(column for column in frame if column.startswith("feat_"))
    if not feature_columns:
        raise ValueError("Feature file has no feat_ columns")

    role = frame[frame["protocol_role"].eq(protocol_role)].copy()
    event = role[role["window_name"].eq(event_window)].copy()
    strict_pre = role[role["window_name"].eq(pre_window)].copy()
    for name, selected in [(event_window, event), (pre_window, strict_pre)]:
        if selected.empty:
            raise ValueError(f"No rows for {name!r} and role {protocol_role!r}")
        if selected["uid"].duplicated().any():
            raise ValueError(f"Duplicate UIDs in {name}")

    paired_uids = sorted(set(event["uid"]).intersection(strict_pre["uid"]))
    if not paired_uids:
        raise ValueError("Event and strict-pre windows have no paired UIDs")
    dropped_event = len(event) - len(paired_uids)
    dropped_pre = len(strict_pre) - len(paired_uids)
    if dropped_event or dropped_pre:
        print(
            f"Pairing retained {len(paired_uids)} UIDs; "
            f"dropped event={dropped_event}, pre={dropped_pre}",
            file=sys.stderr,
        )

    event = event.set_index("uid").loc[paired_uids]
    strict_pre = strict_pre.set_index("uid").loc[paired_uids]
    if not event["label"].equals(strict_pre["label"]):
        raise ValueError("Event and strict-pre labels do not match")
    if not event["source_id"].astype(str).equals(strict_pre["source_id"].astype(str)):
        raise ValueError("Event and strict-pre source IDs do not match")

    labels = event["label"].map(LABEL_TO_INT)
    if labels.isna().any():
        unexpected = sorted(event.loc[labels.isna(), "label"].unique())
        raise ValueError(f"Unexpected labels: {unexpected}")
    groups = event["source_id"].astype(str)

    event_matrix = event.loc[:, feature_columns].to_numpy(dtype=np.float64)
    pre_matrix = strict_pre.loc[:, feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(event_matrix).all() or not np.isfinite(pre_matrix).all():
        raise ValueError("Features contain NaN or infinite values")

    group_sizes = groups.value_counts()
    singleton_groups = int((group_sizes == 1).sum())
    if singleton_groups == len(group_sizes):
        print(
            "WARNING: every source_id is a singleton. Grouped CV cannot test "
            "cross-session generalization and is effectively stratified CV.",
            file=sys.stderr,
        )

    return PairedFeatures(
        uids=np.asarray(paired_uids, dtype=object),
        labels=labels.to_numpy(dtype=int),
        groups=groups.to_numpy(dtype=object),
        event=event_matrix,
        strict_pre=pre_matrix,
        feature_columns=feature_columns,
        singleton_groups=singleton_groups,
    )


def make_probe(c_value: float, seed: int) -> LogisticRegression:
    return LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=seed,
    )


def fit_probe(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    c_value: float,
    seed: int,
) -> FittedProbe:
    scaler = StandardScaler()
    transformed = scaler.fit_transform(matrix[train])
    classifier = make_probe(c_value, seed)
    classifier.fit(transformed, labels[train])
    return FittedProbe(scaler, classifier)


def select_c(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    outer_train: np.ndarray,
    c_grid: tuple[float, ...],
    inner_splits: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
    if len(c_grid) == 1:
        return float(c_grid[0]), [
            {
                "C": float(c_grid[0]),
                "inner_balanced_accuracy": float("nan"),
                "inner_balanced_accuracy_std": float("nan"),
            }
        ]
    splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    local_labels = labels[outer_train]
    local_groups = groups[outer_train]
    folds = list(
        splitter.split(
            np.zeros(len(outer_train)),
            local_labels,
            local_groups,
        )
    )
    scores = {c_value: [] for c_value in c_grid}
    for inner_fold, (train_position, validation_position) in enumerate(folds):
        train = outer_train[train_position]
        validation = outer_train[validation_position]
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(matrix[train])
        transformed_validation = scaler.transform(matrix[validation])
        for c_value in c_grid:
            classifier = make_probe(c_value, seed + inner_fold)
            classifier.fit(transformed_train, labels[train])
            prediction = classifier.predict(transformed_validation)
            scores[c_value].append(
                float(balanced_accuracy_score(labels[validation], prediction))
            )

    records = [
        {
            "C": float(c_value),
            "inner_balanced_accuracy": float(np.mean(values)),
            "inner_balanced_accuracy_std": float(np.std(values, ddof=0)),
        }
        for c_value, values in scores.items()
    ]
    best_index, best = max(
        enumerate(records),
        key=lambda item: (
            item[1]["inner_balanced_accuracy"],
            -item[1]["inner_balanced_accuracy_std"],
            -item[0],
        ),
    )
    del best_index
    return float(best["C"]), records


def calculate_metrics(
    labels: np.ndarray,
    prediction: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "f1_macro": float(f1_score(labels, prediction, average="macro")),
        "mcc": float(matthews_corrcoef(labels, prediction)),
    }


def run_repeat(
    data: PairedFeatures,
    repeat: int,
    outer_splits: int,
    inner_splits: int,
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    repeat_seed = seed + repeat * 1009
    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=repeat_seed,
    )
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for outer_fold, (train, test) in enumerate(
        splitter.split(np.zeros(len(data.labels)), data.labels, data.groups)
    ):
        if set(data.groups[train]).intersection(data.groups[test]):
            raise AssertionError("Source leakage in outer split")
        fold_seed = repeat_seed + outer_fold * 37

        event_c, event_inner = select_c(
            data.event,
            data.labels,
            data.groups,
            train,
            c_grid,
            inner_splits,
            fold_seed,
        )
        event_probe = fit_probe(data.event, data.labels, train, event_c, fold_seed)
        event_prediction, event_scores = event_probe.predict(data.event[test])
        transferred_prediction, transferred_scores = event_probe.predict(
            data.strict_pre[test]
        )

        pre_c, pre_inner = select_c(
            data.strict_pre,
            data.labels,
            data.groups,
            train,
            c_grid,
            inner_splits,
            fold_seed,
        )
        pre_probe = fit_probe(
            data.strict_pre,
            data.labels,
            train,
            pre_c,
            fold_seed,
        )
        pre_prediction, pre_scores = pre_probe.predict(data.strict_pre[test])

        condition_values = {
            "event_selected_event": (event_prediction, event_scores),
            "event_selected_pre": (transferred_prediction, transferred_scores),
            "pre_selected_pre": (pre_prediction, pre_scores),
        }
        for condition, (prediction, scores) in condition_values.items():
            for position, index in enumerate(test):
                prediction_rows.append(
                    {
                        "repeat": repeat,
                        "outer_fold": outer_fold,
                        "condition": condition,
                        "uid": data.uids[index],
                        "source_id": data.groups[index],
                        "y_true": int(data.labels[index]),
                        "y_pred": int(prediction[position]),
                        "score": float(scores[position]),
                    }
                )

        selection_rows.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "training_condition": "event",
                "selected_C": event_c,
                "solver_iterations": int(event_probe.classifier.n_iter_[0]),
                "inner_scores_json": json.dumps(event_inner, sort_keys=True),
            }
        )
        selection_rows.append(
            {
                "repeat": repeat,
                "outer_fold": outer_fold,
                "training_condition": "strict_pre",
                "selected_C": pre_c,
                "solver_iterations": int(pre_probe.classifier.n_iter_[0]),
                "inner_scores_json": json.dumps(pre_inner, sort_keys=True),
            }
        )

    metrics: list[dict[str, object]] = []
    frame = pd.DataFrame(prediction_rows)
    for condition, selected in frame.groupby("condition", sort=True):
        if len(selected) != len(data.labels) or selected["uid"].duplicated().any():
            raise AssertionError(
                f"Each UID must receive one OOF prediction for {condition}"
            )
        values = calculate_metrics(
            selected["y_true"].to_numpy(),
            selected["y_pred"].to_numpy(),
            selected["score"].to_numpy(),
        )
        metrics.append({"repeat": repeat, "condition": condition, **values})
    return metrics, prediction_rows, selection_rows


def evaluate(
    features_path: Path,
    out_dir: Path,
    event_window: str = "event_200ms",
    pre_window: str = "pre_200ms",
    protocol_role: str = "primary_dev",
    outer_splits: int = 5,
    inner_splits: int = 3,
    repeats: int = 5,
    c_grid: tuple[float, ...] = DEFAULT_C_GRID,
    seed: int = 20260716,
) -> pd.DataFrame:
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must both be at least 2")
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if not c_grid or any(value <= 0 for value in c_grid):
        raise ValueError("c_grid must contain positive values")
    data = load_paired_features(
        features_path,
        event_window,
        pre_window,
        protocol_role,
    )
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for repeat in range(repeats):
        metrics, predictions, selections = run_repeat(
            data,
            repeat,
            outer_splits,
            inner_splits,
            c_grid,
            seed,
        )
        metric_rows.extend(metrics)
        prediction_rows.extend(predictions)
        selection_rows.extend(selections)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    summary = (
        metrics.groupby("condition", as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            mcc_mean=("mcc", "mean"),
            mcc_std=("mcc", "std"),
        )
        .fillna(0.0)
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "repeat_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    predictions.to_csv(out_dir / "outer_predictions.csv", index=False)
    selections.to_csv(out_dir / "selections.csv", index=False)
    protocol = {
        "encoder": "frozen_m2d_40ms",
        "encoder_training_epochs": 0,
        "classifier": "StandardScaler + L2 logistic regression",
        "classifier_solver": "liblinear",
        "classifier_max_iter": 5000,
        "class_weight": "balanced",
        "event_window": event_window,
        "pre_window": pre_window,
        "strict_pre_definition": "same duration ending 50 ms before event_start",
        "protocol_role": protocol_role,
        "n_paired_samples": int(len(data.labels)),
        "class_counts": {
            label: int((data.labels == value).sum())
            for label, value in LABEL_TO_INT.items()
        },
        "n_source_groups": int(len(np.unique(data.groups))),
        "n_singleton_source_groups": data.singleton_groups,
        "grouping_effective_beyond_stratification": bool(
            data.singleton_groups < len(np.unique(data.groups))
        ),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "repeats": repeats,
        "c_grid": list(c_grid),
        "c_selection_method": (
            "prespecified_single_value" if len(c_grid) == 1 else "inner_cv"
        ),
        "seed": seed,
        "feature_dimension": len(data.feature_columns),
        "locked_test_used": False,
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a frozen M2D linear probe and strict pre-event negative control "
            "with repeated nested source-grouped CV."
        )
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--event-window", default="event_200ms")
    parser.add_argument("--pre-window", default="pre_200ms")
    parser.add_argument("--protocol-role", default="primary_dev")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--c-grid", type=float, nargs="+", default=list(DEFAULT_C_GRID))
    parser.add_argument("--seed", type=int, default=20260716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate(
        args.features,
        args.out_dir,
        args.event_window,
        args.pre_window,
        args.protocol_role,
        args.outer_splits,
        args.inner_splits,
        args.repeats,
        tuple(args.c_grid),
        args.seed,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
