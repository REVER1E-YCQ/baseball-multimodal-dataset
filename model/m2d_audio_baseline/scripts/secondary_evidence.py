from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from .short_contact_benchmark import ArtifactBundle, LABEL_TO_INT

FIXED_C_GRID = (0.001, 0.01, 0.1)
RBF_C_GRID = (0.3, 1.0, 3.0)
RBF_GAMMA_GRID = ("scale", 0.001)


class SecondaryEvidenceError(RuntimeError):
    """Raised when the fixed benchmark split cannot be reproduced."""


@dataclass(frozen=True)
class SecondaryEvidence:
    output_root: Path
    summary: dict[str, object]

    def path(self, name: str) -> Path:
        return self.output_root / name


def _uid_from_dataset_path(dataset_path: str) -> str:
    parts = dataset_path.replace("\\", "/").split("/")
    if len(parts) != 4 or parts[0] != "dataset":
        raise SecondaryEvidenceError(
            f"Unexpected dataset path in fixed split: {dataset_path}"
        )
    label, collector, sample_id = parts[1], parts[2], parts[3]
    return f"{label}__{collector}__{sample_id}"


def _load_fixed_split(split_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(split_path, encoding="utf-8-sig")
    missing = {"dataset_path", "split"}.difference(frame.columns)
    if missing:
        raise SecondaryEvidenceError(
            f"Fixed split file is missing columns: {sorted(missing)}"
        )
    frame["uid"] = frame["dataset_path"].map(_uid_from_dataset_path)
    allowed = {"train", "val", "test"}
    unexpected = set(frame["split"]) - allowed
    if unexpected:
        raise SecondaryEvidenceError(
            f"Fixed split contains unexpected partitions: {sorted(unexpected)}"
        )
    if frame["uid"].duplicated().any():
        raise SecondaryEvidenceError("Fixed split contains duplicate UIDs")
    return frame[["uid", "label", "source_group", "split"]].copy()


def _event_features(bundle: ArtifactBundle) -> pd.DataFrame:
    features_path = bundle.root / "features" / _feature_filename(bundle)
    features = pd.read_csv(features_path)
    return features[features["window_name"].eq("event_200ms")]


def _feature_filename(bundle: ArtifactBundle) -> str:
    candidates = sorted((bundle.root / "features").glob("*.csv"))
    if not candidates:
        raise SecondaryEvidenceError(f"No feature files in {bundle.root / 'features'}")
    return candidates[0].name


def _split_counts(frame: pd.DataFrame) -> dict[str, int]:
    return frame["split"].value_counts().to_dict()


def _crossings(frame: pd.DataFrame, split_by_uid: pd.Series) -> dict[str, int]:
    """Count source groups whose members span more than one partition."""
    counts: dict[str, int] = {}
    for group, members in frame.groupby("source_group"):
        partitions = set(split_by_uid.loc[members["uid"]])
        if len(partitions) > 1:
            counts[str(group)] = int(len(partitions))
    return counts


def _game_crossings(
    split_by_uid: pd.Series,
    game_by_uid: dict[str, str],
) -> int:
    """Count MLB games whose plays span more than one fixed partition."""
    games: dict[str, set[str]] = {}
    for uid, game in game_by_uid.items():
        if uid in split_by_uid.index:
            games.setdefault(game, set()).add(str(split_by_uid.loc[uid]))
    return int(sum(len(partitions) > 1 for partitions in games.values()))


def _fixed_split_evaluation(
    encoder_name: str,
    features: pd.DataFrame,
    split_frame: pd.DataFrame,
    c_grid: tuple[float, ...],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    by_uid = split_frame.set_index("uid")
    eligible = features[features["uid"].isin(split_frame["uid"])].sort_values(
        "uid"
    )
    train_uids = by_uid.index[by_uid["split"].eq("train")]
    val_uids = by_uid.index[by_uid["split"].eq("val")]
    test_uids = by_uid.index[by_uid["split"].eq("test")]
    train = eligible[eligible["uid"].isin(train_uids)]
    val = eligible[eligible["uid"].isin(val_uids)]
    test = eligible[eligible["uid"].isin(test_uids)]
    if len(train) < 2 or len(val) < 2 or len(test) < 2:
        raise SecondaryEvidenceError(
            "Fixed split partitions are too small after eligibility filtering"
        )

    feature_columns = [
        column for column in features if column.startswith("feat_")
    ]
    train_matrix = train[feature_columns].to_numpy(dtype=np.float64)
    val_matrix = val[feature_columns].to_numpy(dtype=np.float64)
    test_matrix = test[feature_columns].to_numpy(dtype=np.float64)
    train_labels = train["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    val_labels = val["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    test_labels = test["label"].map(LABEL_TO_INT).to_numpy(dtype=int)

    c_scores: list[dict[str, object]] = []
    for c_value in c_grid:
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(train_matrix)
        transformed_val = scaler.transform(val_matrix)
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed,
        )
        classifier.fit(transformed_train, train_labels)
        prediction = classifier.predict(transformed_val)
        c_scores.append(
            {
                "C": c_value,
                "validation_balanced_accuracy": float(
                    balanced_accuracy_score(val_labels, prediction)
                ),
            }
        )
    best = max(
        c_scores,
        key=lambda record: (
            float(record["validation_balanced_accuracy"]),
            -c_scores.index(record),
        ),
    )
    selected_c = float(best["C"])

    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(train_matrix)
    transformed_test = scaler.transform(test_matrix)
    classifier = LogisticRegression(
        C=selected_c,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5_000,
        random_state=seed,
    )
    classifier.fit(transformed_train, train_labels)
    prediction = classifier.predict(transformed_test)
    score = classifier.predict_proba(transformed_test)[:, 1]

    prediction_rows = [
        {
            "encoder": encoder_name,
            "uid": str(row.uid),
            "split": by_uid.loc[str(row.uid), "split"],
            "y_true": int(test_labels[position]),
            "y_pred": int(prediction[position]),
            "score_ground_ball": float(score[position]),
        }
        for position, row in enumerate(test.itertuples(index=False))
    ]
    matrix_counts = confusion_matrix(test_labels, prediction, labels=[0, 1])
    metric_row = {
        "encoder": encoder_name,
        "selected_C": selected_c,
        "balanced_accuracy": float(
            balanced_accuracy_score(test_labels, prediction)
        ),
        "accuracy": float(accuracy_score(test_labels, prediction)),
        "roc_auc": float(roc_auc_score(test_labels, score)),
        "macro_f1": float(f1_score(test_labels, prediction, average="macro")),
        "true_fly_pred_fly": int(matrix_counts[0, 0]),
        "true_fly_pred_ground": int(matrix_counts[0, 1]),
        "true_ground_pred_fly": int(matrix_counts[1, 0]),
        "true_ground_pred_ground": int(matrix_counts[1, 1]),
        "n_train": len(train),
        "n_val": len(val),
        "n_test": len(test),
        "development_evidence": True,
        "not_source_transfer_evidence": True,
    }
    summary = {
        "selected_C": selected_c,
        "c_validation_scores": c_scores,
        "partition_counts": {
            "train": int(len(train)),
            "val": int(len(val)),
            "test": int(len(test)),
        },
    }
    return pd.DataFrame(prediction_rows), pd.DataFrame([metric_row]), summary


def _select_rbf_inner(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    inner_splits: int,
    seed: int,
) -> tuple[tuple[float, object], list[dict[str, object]]]:
    splitter = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=seed,
    )
    local_labels = labels[train]
    local_groups = groups[train]
    folds = list(
        splitter.split(
            np.zeros(len(train)),
            local_labels,
            local_groups,
        )
    )
    candidates = [
        (c_value, gamma) for c_value in RBF_C_GRID for gamma in RBF_GAMMA_GRID
    ]
    scores: dict[tuple[float, object], list[float]] = {
        candidate: [] for candidate in candidates
    }
    for inner_fold, (train_position, validation_position) in enumerate(folds):
        inner_train = train[train_position]
        inner_validation = train[validation_position]
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(matrix[inner_train])
        transformed_validation = scaler.transform(matrix[inner_validation])
        for candidate in candidates:
            classifier = SVC(
                C=candidate[0],
                gamma=candidate[1],
                class_weight="balanced",
                kernel="rbf",
                random_state=seed + inner_fold,
            )
            classifier.fit(transformed_train, labels[inner_train])
            prediction = classifier.predict(transformed_validation)
            scores[candidate].append(
                float(
                    balanced_accuracy_score(
                        labels[inner_validation], prediction
                    )
                )
            )
    records = [
        {
            "C": float(candidate[0]),
            "gamma": candidate[1],
            "inner_balanced_accuracy": float(np.mean(values)),
            "inner_balanced_accuracy_std": float(np.std(values)),
        }
        for candidate, values in scores.items()
    ]
    best = max(
        records,
        key=lambda record: (
            record["inner_balanced_accuracy"],
            -record["inner_balanced_accuracy_std"],
            -records.index(record),
        ),
    )
    return (float(best["C"]), best["gamma"]), records


def _rbf_evaluation(
    encoder_name: str,
    features: pd.DataFrame,
    folds: pd.DataFrame,
    protocol_seed: int,
    outer_splits: int,
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_columns = [
        column for column in features if column.startswith("feat_")
    ]
    aligned = folds.merge(
        features[["uid", *feature_columns]],
        on="uid",
        how="inner",
        validate="one_to_one",
    ).sort_values("uid").reset_index(drop=True)
    matrix = aligned[feature_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(matrix).all():
        raise SecondaryEvidenceError(
            f"Missing or non-finite features for {encoder_name} RBF probe"
        )
    labels = aligned["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = aligned["lineage_group_id"].to_numpy(dtype=object)
    predictions = np.full(len(aligned), -1, dtype=int)
    scores = np.full(len(aligned), np.nan, dtype=np.float64)
    selection_rows: list[dict[str, object]] = []

    for outer_fold in sorted(aligned["outer_fold"].unique()):
        test = np.flatnonzero(aligned["outer_fold"].to_numpy() == outer_fold)
        train = np.flatnonzero(aligned["outer_fold"].to_numpy() != outer_fold)
        fold_seed = protocol_seed + int(outer_fold)
        (c_value, gamma), records = _select_rbf_inner(
            matrix,
            labels,
            groups,
            train,
            inner_splits,
            fold_seed,
        )
        selection_rows.append(
            {
                "encoder": encoder_name,
                "outer_fold": int(outer_fold),
                "selected_C": c_value,
                "selected_gamma": gamma,
                "inner_scores_json": json.dumps(records, sort_keys=True),
            }
        )
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(matrix[train])
        transformed_test = scaler.transform(matrix[test])
        classifier = SVC(
            C=c_value,
            gamma=gamma,
            class_weight="balanced",
            kernel="rbf",
            random_state=fold_seed,
        )
        classifier.fit(transformed_train, labels[train])
        predictions[test] = classifier.predict(transformed_test)
        scores[test] = classifier.decision_function(transformed_test)

    if (predictions < 0).any() or not np.isfinite(scores).all():
        raise SecondaryEvidenceError(
            f"Incomplete RBF out-of-fold predictions for {encoder_name}"
        )
    prediction_rows = [
        {
            "encoder": encoder_name,
            "uid": str(row.uid),
            "lineage_group_id": str(row.lineage_group_id),
            "outer_fold": int(row.outer_fold),
            "y_true": int(labels[position]),
            "y_pred": int(predictions[position]),
            "score_ground_ball": float(scores[position]),
        }
        for position, row in enumerate(aligned.itertuples(index=False))
    ]
    matrix_counts = confusion_matrix(labels, predictions, labels=[0, 1])
    metric_row = {
        "encoder": encoder_name,
        "probe": "balanced_rbf_svm",
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "true_fly_pred_fly": int(matrix_counts[0, 0]),
        "true_fly_pred_ground": int(matrix_counts[0, 1]),
        "true_ground_pred_fly": int(matrix_counts[1, 0]),
        "true_ground_pred_ground": int(matrix_counts[1, 1]),
        "eligible_samples": len(labels),
        "lineage_groups": int(aligned["lineage_group_id"].nunique()),
        "exploratory": True,
        "primary_ranking_unchanged": True,
    }
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame([metric_row]),
        pd.DataFrame(selection_rows),
    )


def compute_secondary_evidence(
    bundles: dict[str, ArtifactBundle],
    fixed_split_path: Path,
    output_root: Path,
    seed: int = 20260805,
) -> SecondaryEvidence:
    """Run the fixed-benchmark-split and RBF SVM development evidence."""

    split_frame = _load_fixed_split(Path(fixed_split_path).resolve())
    split_by_uid = split_frame.set_index("uid")["split"]
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    fixed_predictions: list[pd.DataFrame] = []
    fixed_metrics: list[pd.DataFrame] = []
    rbf_predictions: list[pd.DataFrame] = []
    rbf_metrics: list[pd.DataFrame] = []
    rbf_selections: list[pd.DataFrame] = []
    summaries: dict[str, object] = {}

    for name, bundle in sorted(bundles.items()):
        features = _event_features(bundle)
        snapshot_uids = set(features["uid"]) | set(
            pd.read_csv(bundle.path("exclusions"))["uid"]
        )
        if set(split_frame["uid"]) != snapshot_uids:
            missing = sorted(set(split_frame["uid"]) - snapshot_uids)
            unexpected = sorted(snapshot_uids - set(split_frame["uid"]))
            raise SecondaryEvidenceError(
                "Fixed split does not reproduce the snapshot membership: "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        folds = pd.read_csv(bundle.path("fold_assignments"))
        game_by_uid = {
            str(row.uid): match.group(1)
            for row in folds.itertuples(index=False)
            if (
                match := re.match(
                    r"^mlb_game_pk:(\d+)$", str(row.lineage_group_id)
                )
            )
        }
        pred, metric, summary = _fixed_split_evaluation(
            name, features, split_frame, FIXED_C_GRID, seed
        )
        crossing_groups = _crossings(split_frame, split_by_uid)
        crossing_games = _game_crossings(split_by_uid, game_by_uid)
        fixed_predictions.append(pred)
        fixed_metrics.append(metric)
        summaries[name] = {
            "fixed_split": summary,
            "crossing_source_groups": len(crossing_groups),
            "crossing_mlb_games": crossing_games,
        }

        rbf_pred, rbf_metric, rbf_selection = _rbf_evaluation(
            name,
            features,
            folds,
            seed,
            int(
                json.loads(
                    bundle.path("protocol").read_text(encoding="utf-8")
                )["fold_policy"]["outer_splits"]
            ),
            int(
                json.loads(
                    bundle.path("protocol").read_text(encoding="utf-8")
                )["fold_policy"].get("inner_splits", 3)
            ),
        )
        rbf_predictions.append(rbf_pred)
        rbf_metrics.append(rbf_metric)
        rbf_selections.append(rbf_selection)

    pd.concat(fixed_predictions).to_csv(
        output_root / "fixed_split_predictions.csv", index=False
    )
    pd.concat(fixed_metrics).to_csv(
        output_root / "fixed_split_metrics.csv", index=False
    )
    pd.concat(rbf_predictions).to_csv(
        output_root / "rbf_predictions.csv", index=False
    )
    pd.concat(rbf_metrics).to_csv(output_root / "rbf_metrics.csv", index=False)
    pd.concat(rbf_selections).to_csv(
        output_root / "rbf_selections.csv", index=False
    )
    summary = {
        "encoders": sorted(bundles),
        "fixed_split": summaries,
        "fixed_split_membership_reproduced": True,
        "fixed_split_used_for_primary_folds": False,
        "rbf_grid": {
            "C": list(RBF_C_GRID),
            "gamma": list(RBF_GAMMA_GRID),
        },
        "development_evidence": True,
        "not_source_transfer_evidence": True,
        "primary_ranking_unchanged": True,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SecondaryEvidence(output_root=output_root, summary=summary)
