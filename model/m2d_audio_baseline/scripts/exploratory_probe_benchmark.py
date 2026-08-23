from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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

from .attention_control_representation import (
    CONTROL_CONDITIONS,
    AttentionControlRepresentation,
    attention_control_window_roles,
    load_token_table,
)
from .benchmark_artifact_roles import (
    ATTENTION_CONTROL_TRANSFORM_POLICY,
    M2D_ENCODER_NAME,
    VERIFIED_DATASET_REVISION,
)
from .short_contact_benchmark import (
    ArtifactBundle,
    LABEL_TO_INT,
    _canonical_sha256,
    _file_sha256,
    _write_json,
)


EXPLORATORY_PROBE_PROTOCOL_VERSION = "exploratory-probe-v1"
LOCKED_FOLD_SEED = 20260805


class ExploratoryProbeError(RuntimeError):
    """Raised when an exploratory probe run cannot preserve its protocol."""


@dataclass
class _FitAudit:
    representation_fits: int = 0
    model_selection_fits: int = 0
    threshold_selection_fits: int = 0
    outer_probe_fits: int = 0


@dataclass(frozen=True)
class ProbeConfig:
    """Named estimator and bounded development-only selection policy."""

    name: str
    estimator_family: str
    hyperparameter_grid: Mapping[str, tuple[float | str, ...]]
    score_output: str
    fixed_decision_threshold: float | None = None
    calibrate_threshold: bool = False


_FAMILY_POLICIES = {
    "balanced_l2_logistic_regression": {
        "parameters": ("C",),
        "score_output": "probability_ground_ball",
        "default_threshold": 0.5,
    },
    "balanced_linear_svm": {
        "parameters": ("C",),
        "score_output": "decision_function_ground_ball",
        "default_threshold": 0.0,
    },
    "balanced_rbf_svm": {
        "parameters": ("C", "gamma"),
        "score_output": "decision_function_ground_ball",
        "default_threshold": 0.0,
    },
}


def _probe_document(config: ProbeConfig) -> dict[str, object]:
    if not config.name or not config.name.strip():
        raise ValueError("Probe name must not be empty")
    policy = _FAMILY_POLICIES.get(config.estimator_family)
    if policy is None:
        raise ValueError(
            f"Unsupported estimator family: {config.estimator_family}"
        )
    expected_parameters = tuple(policy["parameters"])
    if set(config.hyperparameter_grid) != set(expected_parameters):
        raise ValueError(
            f"{config.estimator_family} requires exactly "
            f"{list(expected_parameters)}"
        )
    normalized_grid: dict[str, list[float | str]] = {}
    candidate_count = 1
    for parameter in expected_parameters:
        values = tuple(config.hyperparameter_grid[parameter])
        if not values:
            raise ValueError(f"Probe grid {parameter} must not be empty")
        candidate_count *= len(values)
        normalized: list[float | str] = []
        for value in values:
            if parameter == "gamma" and value in ("scale", "auto"):
                normalized.append(str(value))
                continue
            if (
                not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(
                    f"Probe grid {parameter} values must be finite and positive"
                )
            normalized.append(float(value))
        normalized_grid[parameter] = normalized
    if candidate_count > 64:
        raise ValueError(
            "Probe hyperparameter grid must contain at most 64 candidates"
        )
    if config.score_output != policy["score_output"]:
        raise ValueError(
            f"{config.estimator_family} requires "
            f"score_output={policy['score_output']!r}"
        )
    threshold = (
        float(policy["default_threshold"])
        if config.fixed_decision_threshold is None
        else float(config.fixed_decision_threshold)
    )
    if not math.isfinite(threshold):
        raise ValueError("fixed_decision_threshold must be finite")
    return {
        "name": config.name,
        "estimator_family": config.estimator_family,
        "hyperparameter_grid": normalized_grid,
        "score_output": config.score_output,
        "fixed_decision_threshold": threshold,
        "calibrate_threshold": bool(config.calibrate_threshold),
        "selection_metric": "inner_grouped_balanced_accuracy",
        "selection_scope": "outer_train_inner_grouped_validation",
        "scaling": "StandardScaler_fit_on_training_rows_only",
    }


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExploratoryProbeError(f"Cannot read JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExploratoryProbeError(f"JSON artifact must contain an object: {path}")
    return value


def _validated_source(source_root: Path) -> tuple[dict[str, object], Path]:
    protocol_path = source_root / "protocol.json"
    bundle_path = source_root / "artifact_bundle.json"
    if not protocol_path.is_file() or not bundle_path.is_file():
        raise ExploratoryProbeError(
            "Source must contain protocol.json and artifact_bundle.json"
        )
    protocol = _load_json_object(protocol_path)
    manifest = _load_json_object(bundle_path)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ExploratoryProbeError("Source artifact manifest is malformed")
    protocol_artifact_id = str(protocol.get("artifact_id", ""))
    expected_artifact_id = _canonical_sha256(
        {key: value for key, value in protocol.items() if key != "artifact_id"}
    )[:24]
    if (
        not protocol_artifact_id
        or manifest.get("artifact_id") != protocol_artifact_id
        or protocol_artifact_id != expected_artifact_id
    ):
        raise ExploratoryProbeError("Source artifact identity is inconsistent")
    for name in ("protocol", "fold_assignments", "exclusions"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise ExploratoryProbeError(f"Source manifest is missing {name}")
        artifact_path = source_root / str(record.get("path", ""))
        if not artifact_path.is_file() or _file_sha256(artifact_path) != record.get(
            "sha256"
        ):
            raise ExploratoryProbeError(f"Source artifact failed checksum: {name}")

    encoders = protocol.get("encoders")
    dataset = protocol.get("dataset")
    controls = protocol.get("controls")
    fold_policy = protocol.get("fold_policy")
    classifier = protocol.get("classifier")
    threshold = protocol.get("decision_threshold")
    model_input = protocol.get("model_input_policy")
    if not all(
        isinstance(value, dict)
        for value in (
            dataset,
            controls,
            fold_policy,
            classifier,
            threshold,
            model_input,
        )
    ) or not isinstance(encoders, list):
        raise ExploratoryProbeError("Source protocol has malformed role fields")
    role_matches = (
        len(encoders) == 1
        and isinstance(encoders[0], dict)
        and encoders[0].get("name") == M2D_ENCODER_NAME
        and int(encoders[0].get("training_epochs", -1)) == 0
        and dataset.get("revision") == VERIFIED_DATASET_REVISION
        and protocol.get("pooling") == "attention"
        and tuple(protocol.get("window_conditions", [])) == ("event_200ms",)
        and protocol.get("normalization") == "snapshot_level"
        and protocol.get("detector")
        == "absolute_amplitude_peak_within_event_interval"
        and int(protocol.get("event_window_shift_ms", 0)) == 0
        and bool(controls.get("enabled", False))
        and tuple(controls.get("conditions", []))
        == (*CONTROL_CONDITIONS, "contact_specific_increment")
        and fold_policy.get("name") == "StratifiedGroupKFold"
        and fold_policy.get("group") == "lineage_group_id"
        and int(fold_policy.get("outer_splits", 0)) == 5
        and int(fold_policy.get("seed", -1)) == LOCKED_FOLD_SEED
        and bool(fold_policy.get("shuffle", False))
        and classifier.get("name") == "balanced_l2_logistic_regression"
        and classifier.get("C_selection") == "inner_grouped_cv"
        and tuple(classifier.get("C_grid", [])) == (0.001, 0.01, 0.1)
        and int(classifier.get("inner_splits", 0)) == 3
        and classifier.get("class_weight") == "balanced"
        and classifier.get("penalty") == "l2"
        and classifier.get("solver") == "liblinear"
        and not bool(threshold.get("calibrate", False))
        and float(threshold.get("fixed_default", float("nan"))) == 0.5
        and bool(model_input.get("contact_window_only", False))
        and not bool(model_input.get("full_clips", True))
        and not bool(model_input.get("one_second_windows", True))
        and not bool(model_input.get("outcome_context", True))
        and not bool(model_input.get("project_label_visible_to_encoder", True))
        and not bool(model_input.get("waveform_padding", True))
        and protocol.get("attention_control_transform_policy")
        == ATTENTION_CONTROL_TRANSFORM_POLICY
        and protocol.get("feature_composition") is None
        and protocol.get("layers") is None
    )
    if not role_matches:
        raise ExploratoryProbeError(
            "Source is not the corrected locked M2D attention-control role"
        )

    feature_records = [
        record
        for name, record in artifacts.items()
        if str(name).startswith("features/") and isinstance(record, dict)
    ]
    if len(feature_records) != 1:
        raise ExploratoryProbeError(
            "Source role must contain exactly one frozen feature artifact"
        )
    feature_record = feature_records[0]
    feature_path = source_root / str(feature_record.get("path", ""))
    if (
        not feature_path.is_file()
        or _file_sha256(feature_path) != feature_record.get("sha256")
    ):
        raise ExploratoryProbeError("Source feature artifact failed checksum")
    return protocol, feature_path


def _inner_fold_assignments(
    folds: pd.DataFrame,
    *,
    inner_splits: int,
    seed: int,
) -> pd.DataFrame:
    required = {"uid", "label", "lineage_group_id", "outer_fold"}
    if not required.issubset(folds.columns):
        raise ExploratoryProbeError(
            f"Fold assignments are missing columns: {sorted(required - set(folds))}"
        )
    if (folds.groupby("lineage_group_id")["outer_fold"].nunique() != 1).any():
        raise ExploratoryProbeError("A lineage group crosses outer folds")
    rows: list[dict[str, object]] = []
    outer_values = folds["outer_fold"].to_numpy(dtype=int)
    labels = folds["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = folds["lineage_group_id"].to_numpy(dtype=object)
    for outer_fold in sorted(set(outer_values)):
        outer_train = np.flatnonzero(outer_values != outer_fold)
        for inner_fold, (_inner_train, inner_validation) in enumerate(
            _inner_splits(
                labels,
                groups,
                outer_train,
                inner_splits,
                seed + int(outer_fold),
            )
        ):
            for position in inner_validation:
                rows.append(
                    {
                        "outer_fold": int(outer_fold),
                        "inner_fold": int(inner_fold),
                        "uid": str(folds.iloc[position]["uid"]),
                        "lineage_group_id": str(
                            folds.iloc[position]["lineage_group_id"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _candidate_parameters(
    probe_document: dict[str, object],
) -> list[dict[str, float | str]]:
    grid = probe_document["hyperparameter_grid"]
    if not isinstance(grid, dict):
        raise AssertionError("Validated probe grid is not a mapping")
    names = list(grid)
    return [
        dict(zip(names, values, strict=True))
        for values in product(*(grid[name] for name in names))
    ]


def _make_estimator(
    family: str,
    parameters: dict[str, float | str],
    seed: int,
):
    c_value = float(parameters["C"])
    if family == "balanced_l2_logistic_regression":
        return LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed,
        )
    if family == "balanced_linear_svm":
        return SVC(
            C=c_value,
            class_weight="balanced",
            kernel="linear",
            random_state=seed,
        )
    if family == "balanced_rbf_svm":
        return SVC(
            C=c_value,
            gamma=parameters["gamma"],
            class_weight="balanced",
            kernel="rbf",
            random_state=seed,
        )
    raise AssertionError(f"Validated estimator family is unknown: {family}")


def _estimator_scores(
    estimator,
    matrix: np.ndarray,
    score_output: str,
) -> np.ndarray:
    if score_output == "probability_ground_ball":
        scores = estimator.predict_proba(matrix)[:, 1]
    else:
        scores = estimator.decision_function(matrix)
    result = np.asarray(scores, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ExploratoryProbeError("Probe produced non-finite decision scores")
    return result


def _inner_splits(
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    inner_split_count: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=inner_split_count,
        shuffle=True,
        random_state=seed,
    )
    return [
        (train[inner_train], train[inner_validation])
        for inner_train, inner_validation in splitter.split(
            np.zeros(len(train)), labels[train], groups[train]
        )
    ]


def _select_candidate(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    candidates: list[dict[str, float | str]],
    family: str,
    score_output: str,
    fixed_threshold: float,
    inner_split_count: int,
    seed: int,
    audit: _FitAudit | None = None,
) -> tuple[dict[str, float | str], list[dict[str, object]]]:
    scores_by_candidate: list[list[float]] = [
        [] for _candidate in candidates
    ]
    for inner_fold, (inner_train, inner_validation) in enumerate(
        _inner_splits(
            labels, groups, train, inner_split_count, seed
        )
    ):
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(matrix[inner_train])
        transformed_validation = scaler.transform(matrix[inner_validation])
        for candidate_index, candidate in enumerate(candidates):
            estimator = _make_estimator(
                family, candidate, seed + inner_fold
            )
            estimator.fit(transformed_train, labels[inner_train])
            if audit is not None:
                audit.model_selection_fits += 1
            validation_scores = _estimator_scores(
                estimator, transformed_validation, score_output
            )
            prediction = (validation_scores >= fixed_threshold).astype(int)
            scores_by_candidate[candidate_index].append(
                float(
                    balanced_accuracy_score(
                        labels[inner_validation], prediction
                    )
                )
            )
    records = [
        {
            "parameters": candidate,
            "inner_balanced_accuracy": float(np.mean(scores)),
            "inner_balanced_accuracy_std": float(np.std(scores)),
        }
        for candidate, scores in zip(
            candidates, scores_by_candidate, strict=True
        )
    ]
    best_index = max(
        range(len(records)),
        key=lambda index: (
            records[index]["inner_balanced_accuracy"],
            -records[index]["inner_balanced_accuracy_std"],
            -index,
        ),
    )
    return candidates[best_index], records


def _select_threshold(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    parameters: dict[str, float | str],
    family: str,
    score_output: str,
    fixed_threshold: float,
    inner_split_count: int,
    seed: int,
    audit: _FitAudit | None = None,
) -> tuple[float, list[dict[str, float]]]:
    inner_scores = np.full(len(labels), np.nan, dtype=np.float64)
    for inner_fold, (inner_train, inner_validation) in enumerate(
        _inner_splits(
            labels, groups, train, inner_split_count, seed
        )
    ):
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(matrix[inner_train])
        transformed_validation = scaler.transform(matrix[inner_validation])
        estimator = _make_estimator(
            family, parameters, seed + inner_fold
        )
        estimator.fit(transformed_train, labels[inner_train])
        if audit is not None:
            audit.threshold_selection_fits += 1
        inner_scores[inner_validation] = _estimator_scores(
            estimator, transformed_validation, score_output
        )
    training_scores = inner_scores[train]
    if not np.isfinite(training_scores).all():
        raise ExploratoryProbeError(
            "Threshold calibration did not cover every outer-training row"
        )
    ordered = np.sort(np.unique(training_scores))
    midpoints = (ordered[1:] + ordered[:-1]) / 2.0
    candidates = np.unique(
        np.concatenate([[fixed_threshold], midpoints])
    )
    records = [
        {
            "threshold": float(threshold),
            "inner_balanced_accuracy": float(
                balanced_accuracy_score(
                    labels[train],
                    (training_scores >= threshold).astype(int),
                )
            ),
        }
        for threshold in candidates
    ]
    best = max(
        records,
        key=lambda record: (
            record["inner_balanced_accuracy"],
            -abs(record["threshold"] - fixed_threshold),
        ),
    )
    return float(best["threshold"]), records


def _fit_estimator(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    parameters: dict[str, float | str],
    family: str,
    seed: int,
    audit: _FitAudit | None = None,
):
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(matrix[train])
    estimator = _make_estimator(family, parameters, seed)
    estimator.fit(transformed_train, labels[train])
    if audit is not None:
        audit.outer_probe_fits += 1
    return scaler, estimator


def _fixed_rule_name(threshold: float) -> str:
    return f"fixed_{float(threshold)}"


def _evaluate_attention_controls(
    feature_path: Path,
    folds: pd.DataFrame,
    source_protocol: dict[str, object],
    probe_document: dict[str, object],
    *,
    labels_by_uid: Mapping[str, int] | None = None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    _FitAudit,
]:
    required_fold_columns = {
        "uid",
        "label",
        "lineage_group_id",
        "outer_fold",
    }
    if not required_fold_columns.issubset(folds.columns):
        raise ExploratoryProbeError(
            "Source fold assignments are missing required columns"
        )
    if folds["uid"].duplicated().any():
        raise ExploratoryProbeError("Source fold assignments repeat a uid")
    if (folds.groupby("lineage_group_id")["outer_fold"].nunique() != 1).any():
        raise ExploratoryProbeError("A lineage group crosses outer folds")
    expected_outer_folds = set(
        range(int(source_protocol["fold_policy"]["outer_splits"]))
    )
    if set(folds["outer_fold"].astype(int)) != expected_outer_folds:
        raise ExploratoryProbeError(
            "Source fold assignments do not match the outer-fold protocol"
        )

    roles = attention_control_window_roles(
        "event_200ms", "pre_200ms", "removed_200ms"
    )
    try:
        representation = AttentionControlRepresentation.from_token_table(
            load_token_table(feature_path), folds, roles
        )
    except ValueError as error:
        raise ExploratoryProbeError(str(error)) from error
    paired = representation.paired
    window_names = roles.window_names
    fit_window_by_condition = roles.fit_window_by_condition
    apply_window_by_condition = roles.apply_window_by_condition
    if len(paired) < 2:
        raise ExploratoryProbeError("Too few timing-eligible paired samples")
    mapped_labels = paired["label"].map(LABEL_TO_INT)
    if mapped_labels.isna().any():
        raise ExploratoryProbeError("Fold assignments contain unknown labels")
    labels = mapped_labels.to_numpy(dtype=int)
    if labels_by_uid is not None:
        paired_uids = tuple(paired["uid"].astype(str))
        if set(labels_by_uid) != set(paired_uids):
            raise ExploratoryProbeError(
                "Permuted labels do not match the paired sample population"
            )
        labels = np.asarray(
            [labels_by_uid[uid] for uid in paired_uids], dtype=int
        )
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ExploratoryProbeError("Permuted labels must be binary")
    groups = paired["lineage_group_id"].to_numpy(dtype=object)
    fold_array = paired["outer_fold"].to_numpy(dtype=int)
    seed = int(source_protocol["fold_policy"]["seed"])
    inner_split_count = int(source_protocol["classifier"]["inner_splits"])
    attention_k = int(source_protocol.get("attention_k", 3))
    family = str(probe_document["estimator_family"])
    score_output = str(probe_document["score_output"])
    fixed_threshold = float(probe_document["fixed_decision_threshold"])
    candidates = _candidate_parameters(probe_document)
    calibrate = bool(probe_document["calibrate_threshold"])
    rules = [_fixed_rule_name(fixed_threshold)]
    if calibrate:
        rules.append("calibrated")

    fold_scores = {
        condition: np.full(len(paired), np.nan, dtype=np.float64)
        for condition in CONTROL_CONDITIONS
    }
    fold_predictions = {
        (condition, rule): np.full(len(paired), -1, dtype=int)
        for condition in CONTROL_CONDITIONS
        for rule in rules
    }
    selection_rows: list[dict[str, object]] = []
    audit = _FitAudit()

    for outer_fold in sorted(set(fold_array)):
        test = np.flatnonzero(fold_array == outer_fold)
        train = np.flatnonzero(fold_array != outer_fold)
        fold_seed = seed + int(outer_fold)
        try:
            source_matrices, condition_matrices = (
                representation.fold_matrices(
                    train,
                    labels,
                    "attention",
                    attention_k,
                )
            )
            audit.representation_fits += len(window_names)
        except ValueError as error:
            raise ExploratoryProbeError(str(error)) from error
        selected_parameters: dict[str, dict[str, float | str]] = {}
        selected_thresholds: dict[str, float] = {}
        for window_name in window_names:
            selected, candidate_records = _select_candidate(
                source_matrices[window_name],
                labels,
                groups,
                train,
                candidates,
                family,
                score_output,
                fixed_threshold,
                inner_split_count,
                fold_seed,
                audit,
            )
            selected_parameters[window_name] = selected
            row: dict[str, object] = {
                "condition": window_name.rsplit("_", 1)[0],
                "window_ms": 200,
                "outer_fold": int(outer_fold),
                "selected_parameters_json": json.dumps(
                    selected, sort_keys=True
                ),
                "candidate_scores_json": json.dumps(
                    candidate_records, sort_keys=True
                ),
            }
            if calibrate:
                selected_threshold, threshold_records = _select_threshold(
                    source_matrices[window_name],
                    labels,
                    groups,
                    train,
                    selected,
                    family,
                    score_output,
                    fixed_threshold,
                    inner_split_count,
                    fold_seed,
                    audit,
                )
                selected_thresholds[window_name] = selected_threshold
                row["selected_threshold"] = selected_threshold
                row["threshold_scores_json"] = json.dumps(
                    threshold_records, sort_keys=True
                )
            selection_rows.append(row)

        probes = {
            window_name: _fit_estimator(
                source_matrices[window_name],
                labels,
                train,
                selected_parameters[window_name],
                family,
                fold_seed,
                audit,
            )
            for window_name in window_names
        }
        for condition in CONTROL_CONDITIONS:
            fit_window = fit_window_by_condition[condition]
            scaler, estimator = probes[fit_window]
            transformed = scaler.transform(condition_matrices[condition][test])
            scores = _estimator_scores(estimator, transformed, score_output)
            fold_scores[condition][test] = scores
            fold_predictions[(condition, rules[0])][test] = (
                scores >= fixed_threshold
            ).astype(int)
            if calibrate:
                fold_predictions[(condition, "calibrated")][test] = (
                    scores >= selected_thresholds[fit_window]
                ).astype(int)

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for condition in CONTROL_CONDITIONS:
        if not np.isfinite(fold_scores[condition]).all():
            raise ExploratoryProbeError(
                f"Incomplete scores for condition {condition}"
            )
        for rule in rules:
            predictions = fold_predictions[(condition, rule)]
            if (predictions < 0).any():
                raise ExploratoryProbeError(
                    f"Incomplete predictions for {condition}/{rule}"
                )
            for position, row in enumerate(paired.itertuples(index=False)):
                prediction_rows.append(
                    {
                        "condition": condition,
                        "window_ms": 200,
                        "decision_rule": rule,
                        "representation_fit_window": (
                            fit_window_by_condition[condition]
                        ),
                        "representation_apply_window": (
                            apply_window_by_condition[condition]
                        ),
                        "uid": str(row.uid),
                        "label": str(row.label),
                        "lineage_group_id": str(row.lineage_group_id),
                        "outer_fold": int(row.outer_fold),
                        "y_true": int(labels[position]),
                        "y_pred": int(predictions[position]),
                        "score_ground_ball": float(
                            fold_scores[condition][position]
                        ),
                    }
                )
            counts = confusion_matrix(labels, predictions, labels=[0, 1])
            metric_rows.append(
                {
                    "condition": condition,
                    "window_ms": 200,
                    "decision_rule": rule,
                    "primary_metric": "balanced_accuracy",
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels, predictions)
                    ),
                    "accuracy": float(accuracy_score(labels, predictions)),
                    "roc_auc": float(
                        roc_auc_score(labels, fold_scores[condition])
                    ),
                    "macro_f1": float(
                        f1_score(labels, predictions, average="macro")
                    ),
                    "true_fly_pred_fly": int(counts[0, 0]),
                    "true_fly_pred_ground": int(counts[0, 1]),
                    "true_ground_pred_fly": int(counts[1, 0]),
                    "true_ground_pred_ground": int(counts[1, 1]),
                    "eligible_samples": len(paired),
                    "lineage_groups": int(
                        paired["lineage_group_id"].nunique()
                    ),
                }
            )
    for rule in rules:
        increment = float(
            balanced_accuracy_score(
                labels,
                fold_predictions[("event_selected_event", rule)],
            )
            - balanced_accuracy_score(
                labels,
                fold_predictions[("event_selected_pre", rule)],
            )
        )
        metric_rows.append(
            {
                "condition": "contact_specific_increment",
                "window_ms": 200,
                "decision_rule": rule,
                "primary_metric": "balanced_accuracy",
                "balanced_accuracy": increment,
                "accuracy": float("nan"),
                "roc_auc": float("nan"),
                "macro_f1": float("nan"),
                "true_fly_pred_fly": 0,
                "true_fly_pred_ground": 0,
                "true_ground_pred_fly": 0,
                "true_ground_pred_ground": 0,
                "eligible_samples": len(paired),
                "lineage_groups": int(
                    paired["lineage_group_id"].nunique()
                ),
            }
        )
    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
        paired,
        audit,
    )


def run_exploratory_probe_benchmark(
    source_bundle: Path,
    output_dir: Path,
    config: ProbeConfig,
) -> ArtifactBundle:
    """Evaluate one development-only probe on a locked frozen representation."""

    source_root = Path(source_bundle).resolve()
    probe_document = _probe_document(config)
    source_protocol, feature_path = _validated_source(source_root)
    source_artifact_id = str(source_protocol.get("artifact_id", ""))
    source_folds_path = source_root / "fold_assignments.csv"
    source_exclusions_path = source_root / "exclusions.csv"
    provenance = {
        "source_artifact_id": source_artifact_id,
        "source_protocol_sha256": _file_sha256(source_root / "protocol.json"),
        "source_features_sha256": _file_sha256(feature_path),
        "source_folds_sha256": _file_sha256(source_folds_path),
        "source_exclusions_sha256": _file_sha256(source_exclusions_path),
        "encoder_inference_runs": 0,
    }
    protocol_document = {
        "protocol_version": EXPLORATORY_PROBE_PROTOCOL_VERSION,
        "evidence_role": "development_exploratory",
        "primary_common_benchmark_unchanged": True,
        "source_artifact_id": source_artifact_id,
        "source_representation": {
            "encoder": M2D_ENCODER_NAME,
            "pooling": "attention",
            "attention_control_transform_policy": (
                ATTENTION_CONTROL_TRANSFORM_POLICY
            ),
            "window_conditions": ["event_200ms"],
        },
        "fold_policy": source_protocol["fold_policy"],
        "probe": probe_document,
        "provenance_fingerprint": provenance,
    }
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = Path(output_dir).resolve() / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    source_folds = pd.read_csv(source_folds_path).sort_values(
        "uid"
    ).reset_index(drop=True)
    predictions, metrics, selections, folds, _audit = (
        _evaluate_attention_controls(
            feature_path,
            source_folds,
            source_protocol,
            probe_document,
        )
    )
    predictions.insert(0, "encoder", M2D_ENCODER_NAME)
    predictions.insert(0, "probe", config.name)
    metrics.insert(0, "encoder", M2D_ENCODER_NAME)
    metrics.insert(0, "probe", config.name)
    metrics["exploratory"] = True
    metrics["primary_ranking_unchanged"] = True
    selections.insert(0, "encoder", M2D_ENCODER_NAME)
    selections.insert(0, "probe", config.name)
    inner_folds = _inner_fold_assignments(
        folds,
        inner_splits=int(source_protocol["classifier"]["inner_splits"]),
        seed=int(source_protocol["fold_policy"]["seed"]),
    )

    artifact_paths: dict[str, Path] = {}
    for name, frame, filename in (
        ("fold_assignments", folds, "fold_assignments.csv"),
        ("inner_fold_assignments", inner_folds, "inner_fold_assignments.csv"),
        ("oof_predictions", predictions, "oof_predictions.csv"),
        ("metrics", metrics, "metrics.csv"),
        ("selections", selections, "selections.csv"),
    ):
        path = bundle_root / filename
        frame.to_csv(path, index=False)
        artifact_paths[name] = path
    exclusions_path = bundle_root / "exclusions.csv"
    shutil.copy2(source_exclusions_path, exclusions_path)
    artifact_paths["exclusions"] = exclusions_path

    protocol_path = bundle_root / "protocol.json"
    _write_json(protocol_path, {"artifact_id": artifact_id, **protocol_document})
    artifact_paths["protocol"] = protocol_path
    provenance_path = bundle_root / "provenance.json"
    _write_json(provenance_path, provenance)
    artifact_paths["provenance"] = provenance_path

    manifest_path = bundle_root / "artifact_bundle.json"
    _write_json(
        manifest_path,
        {
            "artifact_id": artifact_id,
            "artifacts": {
                name: {
                    "path": path.relative_to(bundle_root).as_posix(),
                    "sha256": _file_sha256(path),
                }
                for name, path in sorted(artifact_paths.items())
            },
        },
    )
    artifact_paths["artifact_bundle"] = manifest_path
    return ArtifactBundle(
        artifact_id=artifact_id,
        root=bundle_root,
        _artifacts=tuple(sorted(artifact_paths.items())),
    )
