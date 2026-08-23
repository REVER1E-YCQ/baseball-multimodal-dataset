from __future__ import annotations

import json
import math
import shutil
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

from .attention_control_representation import (
    AttentionControlRepresentation,
    attention_control_window_roles,
    fit_attention_directions,
    load_token_table,
    pool_attention_tokens,
)
from .benchmark_artifact_roles import M2D_ENCODER_NAME
from .exploratory_probe_benchmark import (
    _candidate_parameters,
    _estimator_scores,
    _fit_estimator,
    _probe_document,
    _select_candidate,
    _validated_source,
    ProbeConfig,
)
from .short_contact_benchmark import (
    ArtifactBundle,
    LABEL_TO_INT,
    _canonical_sha256,
    _file_sha256,
    _write_json,
)


PAIRED_CONTRAST_PROTOCOL_VERSION = "paired-event-pre-contrast-v1"
LOCKED_FOLD_SEEDS = (20260805, 20260806, 20260807)
LOCKED_MINIMUM_HEADLINE_BA_GAIN = 0.02
ARMS = ("event_alone", "event_minus_pre", "event_plus_delta")
CONDITIONS = ("event", "strict_pre", "transient_removed")
WINDOW_BY_CONDITION = {
    "event": "event_200ms",
    "strict_pre": "pre_200ms",
    "transient_removed": "removed_200ms",
}


class PairedContrastError(RuntimeError):
    """Raised when a paired contrast run cannot preserve its protocol."""


@dataclass(frozen=True)
class PairedContrastEvaluationConfig:
    """Locked three-seed uncertainty policy for event/Pre contrasts."""

    fold_seeds: tuple[int, ...] = LOCKED_FOLD_SEEDS
    n_bootstrap: int = 2000
    seed: int = 20260805
    minimum_headline_ba_gain: float = 0.02


def _validate_config(
    config: PairedContrastEvaluationConfig,
) -> dict[str, object]:
    if tuple(config.fold_seeds) != LOCKED_FOLD_SEEDS:
        raise ValueError(
            "fold_seeds are locked at (20260805, 20260806, 20260807)"
        )
    if config.n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not isinstance(config.seed, int):
        raise ValueError("seed must be an integer")
    gain = float(config.minimum_headline_ba_gain)
    if (
        not math.isfinite(gain)
        or not math.isclose(
            gain, LOCKED_MINIMUM_HEADLINE_BA_GAIN, rel_tol=0, abs_tol=1e-12
        )
    ):
        raise ValueError("minimum_headline_ba_gain is locked at 0.02")
    return {
        "fold_seeds": list(LOCKED_FOLD_SEEDS),
        "n_bootstrap": int(config.n_bootstrap),
        "bootstrap_seed": int(config.seed),
        "minimum_headline_ba_gain": gain,
    }


def _locked_probe() -> tuple[dict[str, object], list[dict[str, float | str]]]:
    document = _probe_document(
        ProbeConfig(
            name="paired-contrast-logistic",
            estimator_family="balanced_l2_logistic_regression",
            hyperparameter_grid={"C": (0.001, 0.01, 0.1)},
            score_output="probability_ground_ball",
            fixed_decision_threshold=0.5,
            calibrate_threshold=False,
        )
    )
    return document, _candidate_parameters(document)


def _seeded_folds(
    source_folds: pd.DataFrame,
    paired: pd.DataFrame,
    fold_seeds: tuple[int, ...],
) -> pd.DataFrame:
    source = source_folds.reset_index(drop=True).copy()
    labels = source["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = source["lineage_group_id"].astype(str).to_numpy(dtype=object)
    rows: list[pd.DataFrame] = []
    for seed in fold_seeds:
        if seed == LOCKED_FOLD_SEEDS[0]:
            assigned = source["outer_fold"].to_numpy(dtype=int)
        else:
            assigned = np.full(len(source), -1, dtype=int)
            splitter = StratifiedGroupKFold(
                n_splits=5,
                shuffle=True,
                random_state=int(seed),
            )
            for outer_fold, (_train, test) in enumerate(
                splitter.split(np.zeros(len(source)), labels, groups)
            ):
                assigned[test] = int(outer_fold)
            if (assigned < 0).any():
                raise PairedContrastError(
                    f"Fold seed {seed} did not assign every source row"
                )
        mapping = dict(
            zip(source["uid"].astype(str), assigned, strict=True)
        )
        seeded = paired[
            ["uid", "label", "lineage_group_id"]
        ].copy()
        seeded.insert(0, "fold_seed", int(seed))
        seeded["outer_fold"] = seeded["uid"].astype(str).map(mapping)
        if seeded["outer_fold"].isna().any():
            raise PairedContrastError(
                f"Fold seed {seed} is missing exact paired rows"
            )
        seeded["outer_fold"] = seeded["outer_fold"].astype(int)
        rows.append(seeded)
    result = pd.concat(rows, ignore_index=True)
    if (
        result.groupby(["fold_seed", "lineage_group_id"])["outer_fold"]
        .nunique()
        .gt(1)
        .any()
    ):
        raise PairedContrastError("A lineage group crosses outer folds")
    return result


def _pool_event_fitted_windows(
    representation: AttentionControlRepresentation,
    train: np.ndarray,
    labels: np.ndarray,
    attention_k: int,
) -> dict[str, np.ndarray]:
    directions = fit_attention_directions(
        [
            representation.token_table[
                (representation.paired_uids[position], "event_200ms")
            ]
            for position in train
        ],
        labels[train],
        "attention",
        attention_k,
    )
    matrices: dict[str, np.ndarray] = {}
    for window_name in WINDOW_BY_CONDITION.values():
        matrix = np.stack(
            [
                pool_attention_tokens(
                    representation.token_table[(uid, window_name)],
                    directions,
                    "attention",
                    attention_k,
                )
                for uid in representation.paired_uids
            ]
        )
        if not np.isfinite(matrix).all():
            raise PairedContrastError(
                f"Non-finite event-fitted features for {window_name}"
            )
        matrices[window_name] = matrix
    return matrices


def _arm_matrices(
    pooled: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    event = pooled["event_200ms"]
    pre = pooled["pre_200ms"]
    removed = pooled["removed_200ms"]
    zero_delta = pre - pre
    return {
        "event_alone": {
            "event": event,
            "strict_pre": pre,
            "transient_removed": removed,
        },
        "event_minus_pre": {
            "event": event - pre,
            "strict_pre": zero_delta,
            "transient_removed": removed - pre,
        },
        "event_plus_delta": {
            "event": np.concatenate([event, event - pre], axis=1),
            "strict_pre": np.concatenate([pre, zero_delta], axis=1),
            "transient_removed": np.concatenate(
                [removed, removed - pre], axis=1
            ),
        },
    }


def _metric_row(
    arm: str,
    fold_seed: int,
    condition: str,
    labels: np.ndarray,
    predictions: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, object]:
    counts = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "arm": arm,
        "fold_seed": int(fold_seed),
        "condition": condition,
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "accuracy": float(accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "true_fly_pred_fly": int(counts[0, 0]),
        "true_fly_pred_ground": int(counts[0, 1]),
        "true_ground_pred_fly": int(counts[1, 0]),
        "true_ground_pred_ground": int(counts[1, 1]),
        "eligible_samples": int(len(labels)),
        "lineage_groups": int(len(set(groups))),
    }


def _evaluate_family(
    representation: AttentionControlRepresentation,
    seeded_folds: pd.DataFrame,
    source_protocol: dict[str, object],
    probe_document: dict[str, object],
    candidates: list[dict[str, float | str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    paired = representation.paired
    labels = paired["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = paired["lineage_group_id"].astype(str).to_numpy(dtype=object)
    attention_k = int(source_protocol.get("attention_k", 3))
    inner_splits = int(source_protocol["classifier"]["inner_splits"])
    predictions_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fit_audit = {
        "event_attention_fits": 0,
        "model_selection_fits": 0,
        "outer_probe_fits": 0,
    }

    for fold_seed in LOCKED_FOLD_SEEDS:
        fold_rows = seeded_folds[
            seeded_folds["fold_seed"] == fold_seed
        ].set_index("uid")
        fold_array = np.asarray(
            [
                fold_rows.loc[uid, "outer_fold"]
                for uid in representation.paired_uids
            ],
            dtype=int,
        )
        score_arrays = {
            (arm, condition): np.full(len(paired), np.nan, dtype=np.float64)
            for arm in ARMS
            for condition in CONDITIONS
        }
        prediction_arrays = {
            (arm, condition): np.full(len(paired), -1, dtype=int)
            for arm in ARMS
            for condition in CONDITIONS
        }
        for outer_fold in sorted(set(fold_array)):
            test = np.flatnonzero(fold_array == outer_fold)
            train = np.flatnonzero(fold_array != outer_fold)
            fold_seed_value = int(fold_seed + outer_fold)
            pooled = _pool_event_fitted_windows(
                representation, train, labels, attention_k
            )
            fit_audit["event_attention_fits"] += 1
            matrices_by_arm = _arm_matrices(pooled)
            for arm in ARMS:
                event_matrix = matrices_by_arm[arm]["event"]
                selected, candidate_records = _select_candidate(
                    event_matrix,
                    labels,
                    groups,
                    train,
                    candidates,
                    str(probe_document["estimator_family"]),
                    str(probe_document["score_output"]),
                    float(probe_document["fixed_decision_threshold"]),
                    inner_splits,
                    fold_seed_value,
                )
                fit_audit["model_selection_fits"] += (
                    inner_splits * len(candidates)
                )
                scaler, estimator = _fit_estimator(
                    event_matrix,
                    labels,
                    train,
                    selected,
                    str(probe_document["estimator_family"]),
                    fold_seed_value,
                )
                fit_audit["outer_probe_fits"] += 1
                selection_rows.append(
                    {
                        "arm": arm,
                        "fold_seed": int(fold_seed),
                        "outer_fold": int(outer_fold),
                        "selected_parameters_json": json.dumps(
                            selected, sort_keys=True
                        ),
                        "candidate_scores_json": json.dumps(
                            candidate_records, sort_keys=True
                        ),
                        "selection_scope": (
                            "outer_train_inner_lineage_grouped_validation"
                        ),
                    }
                )
                for condition in CONDITIONS:
                    transformed = scaler.transform(
                        matrices_by_arm[arm][condition][test]
                    )
                    scores = _estimator_scores(
                        estimator,
                        transformed,
                        str(probe_document["score_output"]),
                    )
                    predicted = (scores >= 0.5).astype(int)
                    score_arrays[(arm, condition)][test] = scores
                    prediction_arrays[(arm, condition)][test] = predicted

        for arm in ARMS:
            for condition in CONDITIONS:
                scores = score_arrays[(arm, condition)]
                predicted = prediction_arrays[(arm, condition)]
                if not np.isfinite(scores).all() or (predicted < 0).any():
                    raise PairedContrastError(
                        f"Incomplete OOF values for {fold_seed}/{arm}/{condition}"
                    )
                metric_rows.append(
                    _metric_row(
                        arm,
                        fold_seed,
                        condition,
                        labels,
                        predicted,
                        scores,
                        groups,
                    )
                )
                for position, row in enumerate(paired.itertuples(index=False)):
                    predictions_rows.append(
                        {
                            "arm": arm,
                            "fold_seed": int(fold_seed),
                            "condition": condition,
                            "uid": str(row.uid),
                            "label": str(row.label),
                            "lineage_group_id": str(row.lineage_group_id),
                            "outer_fold": int(fold_array[position]),
                            "y_true": int(labels[position]),
                            "y_pred": int(predicted[position]),
                            "score_ground_ball": float(scores[position]),
                        }
                    )
            event_ba = next(
                row["balanced_accuracy"]
                for row in reversed(metric_rows)
                if row["arm"] == arm
                and row["fold_seed"] == fold_seed
                and row["condition"] == "event"
            )
            pre_ba = next(
                row["balanced_accuracy"]
                for row in reversed(metric_rows)
                if row["arm"] == arm
                and row["fold_seed"] == fold_seed
                and row["condition"] == "strict_pre"
            )
            metric_rows.append(
                {
                    "arm": arm,
                    "fold_seed": int(fold_seed),
                    "condition": "contact_specific_increment",
                    "balanced_accuracy": float(event_ba - pre_ba),
                    "accuracy": float("nan"),
                    "roc_auc": float("nan"),
                    "macro_f1": float("nan"),
                    "true_fly_pred_fly": 0,
                    "true_fly_pred_ground": 0,
                    "true_ground_pred_fly": 0,
                    "true_ground_pred_ground": 0,
                    "eligible_samples": int(len(labels)),
                    "lineage_groups": int(len(set(groups))),
                }
            )
    return (
        pd.DataFrame(predictions_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
        fit_audit,
    )


def _paired_differences(
    predictions: pd.DataFrame,
    config: PairedContrastEvaluationConfig,
) -> pd.DataFrame:
    reference = (
        predictions[
            (predictions["arm"] == "event_alone")
            & (predictions["fold_seed"] == LOCKED_FOLD_SEEDS[0])
            & (predictions["condition"] == "event")
        ]
        .sort_values("uid")
        .reset_index(drop=True)
    )
    uids = reference["uid"].astype(str).tolist()
    labels = reference["y_true"].to_numpy(dtype=int)
    groups = reference["lineage_group_id"].astype(str).to_numpy(dtype=object)
    unique_groups = np.asarray(sorted(set(groups)), dtype=object)
    positions_by_group = {
        group: np.flatnonzero(groups == group) for group in unique_groups
    }
    rng = np.random.default_rng(config.seed)
    bootstrap_positions = [
        np.concatenate(
            [positions_by_group[group] for group in rng.choice(
                unique_groups, size=len(unique_groups), replace=True
            )]
        )
        for _ in range(config.n_bootstrap)
    ]

    vectors: dict[tuple[int, str, str], np.ndarray] = {}
    for fold_seed in LOCKED_FOLD_SEEDS:
        for arm in ARMS:
            for condition in CONDITIONS:
                arm_rows = (
                    predictions[
                        (predictions["arm"] == arm)
                        & (predictions["fold_seed"] == fold_seed)
                        & (predictions["condition"] == condition)
                    ]
                    .set_index("uid")
                    .loc[uids]
                )
                if not np.array_equal(
                    arm_rows["y_true"].to_numpy(dtype=int), labels
                ):
                    raise PairedContrastError(
                        "Paired predictions do not share labels and membership"
                    )
                vectors[(fold_seed, arm, condition)] = arm_rows[
                    "y_pred"
                ].to_numpy(dtype=int)

    def measure(
        arm: str,
        comparison: str,
        fold_seed: int,
        positions: np.ndarray,
    ) -> float:
        y_true = labels[positions]

        def ba(candidate_arm: str, condition: str) -> float:
            return float(
                balanced_accuracy_score(
                    y_true,
                    vectors[(fold_seed, candidate_arm, condition)][positions],
                )
            )

        if comparison == "event_ba_gain":
            return ba(arm, "event") - ba("event_alone", "event")
        if comparison == "strict_pre_ba_change":
            return ba(arm, "strict_pre") - ba(
                "event_alone", "strict_pre"
            )
        if comparison == "transient_removed_ba_change":
            return ba(arm, "transient_removed") - ba(
                "event_alone", "transient_removed"
            )
        if comparison == "contact_specific_increment_gain":
            candidate_increment = ba(arm, "event") - ba(
                arm, "strict_pre"
            )
            baseline_increment = ba("event_alone", "event") - ba(
                "event_alone", "strict_pre"
            )
            return candidate_increment - baseline_increment
        raise AssertionError(f"Unknown paired comparison: {comparison}")

    comparisons = (
        "event_ba_gain",
        "strict_pre_ba_change",
        "transient_removed_ba_change",
        "contact_specific_increment_gain",
    )
    all_positions = np.arange(len(reference), dtype=int)
    rows: list[dict[str, object]] = []
    for arm in ARMS[1:]:
        for comparison in comparisons:
            seed_bootstrap: dict[int, np.ndarray] = {}
            seed_observed: dict[int, float] = {}
            for fold_seed in LOCKED_FOLD_SEEDS:
                observed = measure(
                    arm, comparison, fold_seed, all_positions
                )
                bootstrapped = np.asarray(
                    [
                        measure(arm, comparison, fold_seed, positions)
                        for positions in bootstrap_positions
                    ],
                    dtype=np.float64,
                )
                seed_observed[fold_seed] = observed
                seed_bootstrap[fold_seed] = bootstrapped
                rows.append(
                    {
                        "arm": arm,
                        "comparison": comparison,
                        "fold_seed": str(fold_seed),
                        "observed_difference": observed,
                        "ci_low": float(np.quantile(bootstrapped, 0.025)),
                        "ci_high": float(np.quantile(bootstrapped, 0.975)),
                        "n_bootstrap": int(config.n_bootstrap),
                        "resampling_unit": "lineage_group_id",
                    }
                )
            mean_bootstrap = np.mean(
                np.stack(list(seed_bootstrap.values())), axis=0
            )
            rows.append(
                {
                    "arm": arm,
                    "comparison": comparison,
                    "fold_seed": "mean_across_seeds",
                    "observed_difference": float(
                        np.mean(list(seed_observed.values()))
                    ),
                    "ci_low": float(np.quantile(mean_bootstrap, 0.025)),
                    "ci_high": float(np.quantile(mean_bootstrap, 0.975)),
                    "n_bootstrap": int(config.n_bootstrap),
                    "resampling_unit": "lineage_group_id_shared_across_seeds",
                }
            )
    return pd.DataFrame(rows)


def _screening_verdict(
    paired_differences: pd.DataFrame,
    config: PairedContrastEvaluationConfig,
) -> dict[str, object]:
    evaluated: list[dict[str, object]] = []
    qualifying: list[str] = []
    for arm in ARMS[1:]:
        arm_rows = paired_differences[paired_differences["arm"] == arm]
        mean_rows = arm_rows[
            arm_rows["fold_seed"] == "mean_across_seeds"
        ].set_index("comparison")
        seed_event = arm_rows[
            (arm_rows["comparison"] == "event_ba_gain")
            & (arm_rows["fold_seed"] != "mean_across_seeds")
        ]
        event = mean_rows.loc["event_ba_gain"]
        same_direction = bool(
            (seed_event["observed_difference"] > 0).all()
        )
        seed_controls = arm_rows[
            arm_rows["comparison"].isin(
                ("strict_pre_ba_change", "transient_removed_ba_change")
            )
            & (arm_rows["fold_seed"] != "mean_across_seeds")
        ]
        controls_not_weakened = bool(
            (seed_controls["observed_difference"] <= 0).all()
        )
        gain_at_least_minimum = bool(
            event["observed_difference"]
            >= config.minimum_headline_ba_gain
        )
        paired_interval_above_zero = bool(event["ci_low"] > 0)
        qualifies = bool(
            same_direction
            and controls_not_weakened
            and gain_at_least_minimum
            and paired_interval_above_zero
        )
        if qualifies:
            qualifying.append(arm)
        evaluated.append(
            {
                "arm": arm,
                "mean_event_ba_gain": float(event["observed_difference"]),
                "mean_event_ba_gain_ci_low": float(event["ci_low"]),
                "mean_event_ba_gain_ci_high": float(event["ci_high"]),
                "all_three_seeds_improve": same_direction,
                "gain_at_least_minimum": gain_at_least_minimum,
                "paired_interval_above_zero": paired_interval_above_zero,
                "no_control_rise_in_any_seed": controls_not_weakened,
                "qualifies_for_downstream_validation": qualifies,
            }
        )
    decision = "continue" if qualifying else "stop"
    return {
        "decision": decision,
        "continue_contrast_direction": bool(qualifying),
        "eligible_for_downstream_validation": bool(qualifying),
        "qualifying_arms": qualifying,
        "evaluated_arms": evaluated,
        "preferred_representation_selected": False,
        "headline_replacement_allowed": False,
        "primary_common_benchmark_unchanged": True,
        "development_evidence_only": True,
        "minimum_headline_ba_gain": float(
            config.minimum_headline_ba_gain
        ),
        "interpretation_zh": (
            "至少一个预声明 contrast 臂满足三 seed、配对区间和负控门槛；"
            "仅进入后续 family-aware 验证，不选择 preferred representation。"
            if qualifying
            else "没有 contrast 臂同时满足三 seed、配对区间和负控门槛；停止该方向。"
        ),
    }


def _report_zh(
    metrics: pd.DataFrame,
    paired_differences: pd.DataFrame,
    dimensions: pd.DataFrame,
    selections: pd.DataFrame,
    pairing_audit: dict[str, object],
    verdict: dict[str, object],
) -> str:
    lines = [
        "# M2D paired Event–Pre contrast 三-seed 锁定复测",
        "",
        "本报告属于开发集探索证据，不改变 ADR-0004 共同比较，也不选择 preferred representation。",
        "",
        "## 表示与推理契约",
        "",
        "三个预声明表示均使用仅在 outer-training event tokens 上拟合的同一个 attention transform。",
        "event-minus-Pre 与 event-plus-delta 在推理时均要求同一样本的 exact strict-Pre 音频；不允许 padding。",
        "其中 `P-P` 是表示公式产生的结构性零差控制，不能当作独立 strict-Pre 实证性能；`R-P` 才保留实际 removed 对照变化。",
        "",
        "| 表示臂 | 公式 | 维度 | 推理时需要 strict-Pre |",
        "|---|---|---:|---|",
    ]
    for row in dimensions.itertuples(index=False):
        lines.append(
            f"| {row.arm} | `{row.formula}` | {int(row.dimension)} | "
            f"{'是' if row.strict_pre_required_at_inference else '否'} |"
        )
    lines.extend(
        [
            "",
            "## 三个 fold seeds 的 OOF 结果",
            "",
            "| seed | 表示臂 | Event BA | strict-Pre BA | Removed BA | Contact increment |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    indexed_metrics = metrics.set_index(["fold_seed", "arm", "condition"])
    for fold_seed in LOCKED_FOLD_SEEDS:
        for arm in ARMS:
            lines.append(
                f"| {fold_seed} | {arm} | "
                f"{float(indexed_metrics.loc[(fold_seed, arm, 'event'), 'balanced_accuracy']):.3f} | "
                f"{float(indexed_metrics.loc[(fold_seed, arm, 'strict_pre'), 'balanced_accuracy']):.3f} | "
                f"{float(indexed_metrics.loc[(fold_seed, arm, 'transient_removed'), 'balanced_accuracy']):.3f} | "
                f"{float(indexed_metrics.loc[(fold_seed, arm, 'contact_specific_increment'), 'balanced_accuracy']):+.3f} |"
            )
    lines.extend(
        [
            "",
            "## 相对 event-alone 的 paired lineage-group 差异",
            "",
            "| 表示臂 | 比较 | 三-seed 平均差 | 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    mean_rows = paired_differences[
        paired_differences["fold_seed"] == "mean_across_seeds"
    ]
    for row in mean_rows.itertuples(index=False):
        lines.append(
            f"| {row.arm} | {row.comparison} | "
            f"{float(row.observed_difference):+.3f} | "
            f"[{float(row.ci_low):+.3f}, {float(row.ci_high):+.3f}] |"
        )
    selected_counts = (
        selections.groupby(["arm", "selected_parameters_json"])
        .size()
        .reset_index(name="fold_count")
    )
    lines.extend(
        [
            "",
            "## Inner-only probe selection",
            "",
            "| 表示臂 | 参数 | outer-fold 数 |",
            "|---|---|---:|",
        ]
    )
    for row in selected_counts.itertuples(index=False):
        lines.append(
            f"| {row.arm} | `{row.selected_parameters_json}` | {int(row.fold_count)} |"
        )
    lines.extend(
        [
            "",
            "## 判定与证据边界",
            "",
            str(verdict["interpretation_zh"]),
            "低于 +0.02 BA、三 seed 不同方向、配对区间跨零或负控同步上升时均不得晋升。",
            f"本次使用 {int(pairing_audit['n_exact_pairs'])} 条 exact pairs、"
            f"{int(pairing_audit['n_lineage_groups'])} 个 lineage groups，其中 "
            f"{int(pairing_audit['n_singleton_lineage_groups'])} 个为 singleton；"
            f"源 event-eligible 集为 {int(pairing_audit['n_source_event_eligible'])} 条，"
            f"因配对要求另排除 {int(pairing_audit['excluded_from_exact_pairs'])} 条。",
            f"源 exclusions artifact 共 {int(pairing_audit['n_source_exclusion_records'])} 条记录。",
            "这些结果不能证明跨比赛、设备、采集者或跨采集流程泛化。",
            "完整 Accuracy、ROC-AUC、Macro-F1、confusion counts、每折选择和样本级 OOF 见机器可读 artifacts。",
        ]
    )
    return "\n".join(lines) + "\n"


def _representation_dimensions(token_dimension: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "arm": "event_alone",
                "formula": "E",
                "dimension": token_dimension,
                "strict_pre_required_at_inference": False,
            },
            {
                "arm": "event_minus_pre",
                "formula": "E-P",
                "dimension": token_dimension,
                "strict_pre_required_at_inference": True,
            },
            {
                "arm": "event_plus_delta",
                "formula": "concat(E,E-P)",
                "dimension": token_dimension * 2,
                "strict_pre_required_at_inference": True,
            },
        ]
    )


def run_paired_contrast_evaluation(
    source_bundle: Path,
    output_dir: Path,
    config: PairedContrastEvaluationConfig,
) -> ArtifactBundle:
    """Evaluate the locked event/Pre contrast family on exact pairs."""

    config_document = _validate_config(config)
    source_root = Path(source_bundle).resolve()
    source_protocol, feature_path = _validated_source(source_root)
    source_folds_path = source_root / "fold_assignments.csv"
    source_exclusions_path = source_root / "exclusions.csv"
    source_folds = pd.read_csv(source_folds_path)
    source_exclusions = pd.read_csv(source_exclusions_path)
    roles = attention_control_window_roles(
        "event_200ms", "pre_200ms", "removed_200ms"
    )
    try:
        representation = AttentionControlRepresentation.from_token_table(
            load_token_table(feature_path), source_folds, roles
        )
    except ValueError as error:
        raise PairedContrastError(str(error)) from error
    if not representation.paired_uids:
        raise PairedContrastError("No exact event/Pre/removed pairs are available")

    probe_document, candidates = _locked_probe()
    seeded_folds = _seeded_folds(
        source_folds, representation.paired, LOCKED_FOLD_SEEDS
    )
    predictions, metrics, selections, fit_audit = _evaluate_family(
        representation,
        seeded_folds,
        source_protocol,
        probe_document,
        candidates,
    )
    first_tokens = representation.token_table[
        (representation.paired_uids[0], "event_200ms")
    ]
    dimensions = _representation_dimensions(int(first_tokens.shape[1]))
    paired_differences = _paired_differences(predictions, config)
    verdict = _screening_verdict(paired_differences, config)
    group_sizes = representation.paired.groupby("lineage_group_id").size()
    pairing_audit = {
        "n_source_event_eligible": int(len(source_folds)),
        "n_exact_pairs": int(len(representation.paired)),
        "n_lineage_groups": int(len(group_sizes)),
        "n_singleton_lineage_groups": int(group_sizes.eq(1).sum()),
        "identical_membership_across_arms_and_seeds": True,
        "exact_windows_only": True,
        "waveform_padding_samples": 0,
        "required_windows": list(roles.window_names),
        "excluded_from_exact_pairs": int(
            len(source_folds) - len(representation.paired)
        ),
        "n_source_exclusion_records": int(len(source_exclusions)),
        "source_exclusion_reason_counts": {
            str(reason): int(count)
            for reason, count in source_exclusions["reason"].value_counts().items()
        },
    }
    representation_family = {
        "arms": list(ARMS),
        "attention_fit_scope": "outer_training_event_tokens_only",
        "shared_transform_windows": list(roles.window_names),
        "condition_recipes": {
            "event_alone": {"event": "E", "strict_pre": "P", "removed": "R"},
            "event_minus_pre": {
                "event": "E-P",
                "strict_pre": "P-P",
                "removed": "R-P",
            },
            "event_plus_delta": {
                "event": "concat(E,E-P)",
                "strict_pre": "concat(P,P-P)",
                "removed": "concat(R,R-P)",
            },
        },
        "representation_selection": "none_predeclared_arms_reported_separately",
        "strict_pre_required_at_inference": {
            "event_alone": False,
            "event_minus_pre": True,
            "event_plus_delta": True,
        },
    }
    provenance = {
        "source_artifact_id": str(source_protocol["artifact_id"]),
        "source_protocol_sha256": _file_sha256(source_root / "protocol.json"),
        "source_features_sha256": _file_sha256(feature_path),
        "source_folds_sha256": _file_sha256(source_folds_path),
        "source_exclusions_sha256": _file_sha256(source_exclusions_path),
        "encoder_inference_runs": 0,
    }
    protocol_document = {
        "protocol_version": PAIRED_CONTRAST_PROTOCOL_VERSION,
        "evidence_role": "development_exploratory",
        "primary_common_benchmark_unchanged": True,
        "source_artifact_id": str(source_protocol["artifact_id"]),
        "config": config_document,
        "representation_family": representation_family,
        "probe": probe_document,
        "screening_policy": {
            "baseline_arm": "event_alone",
            "event_ba_gain_required": LOCKED_MINIMUM_HEADLINE_BA_GAIN,
            "all_three_seeds_must_improve": True,
            "paired_lineage_interval_must_be_above_zero": True,
            "strict_pre_and_removed_must_not_rise_in_any_seed": True,
            "outer_results_select_preferred_representation": False,
        },
        "fold_policy": {
            "name": "StratifiedGroupKFold",
            "group": "lineage_group_id",
            "outer_splits": 5,
            "shuffle": True,
            "seeds": list(LOCKED_FOLD_SEEDS),
            "locked_source_assignments_reused_for_first_seed": True,
        },
        "pairing": pairing_audit,
        "provenance_fingerprint": provenance,
    }
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = Path(output_dir).resolve() / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}
    for name, frame, filename in (
        ("fold_assignments", seeded_folds, "fold_assignments.csv"),
        ("oof_predictions", predictions, "oof_predictions.csv"),
        ("metrics", metrics, "metrics.csv"),
        ("selections", selections, "selections.csv"),
        (
            "representation_dimensions",
            dimensions,
            "representation_dimensions.csv",
        ),
        (
            "paired_differences",
            paired_differences,
            "paired_differences.csv",
        ),
    ):
        path = bundle_root / filename
        frame.to_csv(path, index=False)
        artifact_paths[name] = path
    exclusions_path = bundle_root / "exclusions.csv"
    shutil.copy2(source_exclusions_path, exclusions_path)
    artifact_paths["exclusions"] = exclusions_path
    for name, document, filename in (
        ("pairing_audit", pairing_audit, "pairing_audit.json"),
        ("fit_audit", fit_audit, "fit_audit.json"),
        ("verdict", verdict, "verdict.json"),
        ("provenance", provenance, "provenance.json"),
    ):
        path = bundle_root / filename
        _write_json(path, document)
        artifact_paths[name] = path
    protocol_path = bundle_root / "protocol.json"
    _write_json(protocol_path, {"artifact_id": artifact_id, **protocol_document})
    artifact_paths["protocol"] = protocol_path
    report_path = bundle_root / "report_zh.md"
    report_path.write_text(
        _report_zh(
            metrics,
            paired_differences,
            dimensions,
            selections,
            pairing_audit,
            verdict,
        ),
        encoding="utf-8",
    )
    artifact_paths["report_zh"] = report_path

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
