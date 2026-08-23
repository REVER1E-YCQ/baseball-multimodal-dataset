from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .exploratory_probe_benchmark import (
    ProbeConfig,
    run_exploratory_probe_benchmark,
)
from .short_contact_benchmark import (
    ArtifactBundle,
    _canonical_sha256,
    _file_sha256,
    _write_json,
)

MARGIN_CLASSIFIER_PROTOCOL_VERSION = "margin-classifier-evaluation-v1"
LOCKED_MINIMUM_HEADLINE_BA_GAIN = 0.02
BASELINE_PROBE = "attention-logistic"
BASELINE_RULE = "fixed_0.5"
EVENT_CONDITION = "event_selected_event"
PRE_CONTROL = "event_selected_pre"
REMOVED_CONTROL = "event_selected_removed"
STARTING_OBSERVATIONS = {
    "attention-logistic": {
        "balanced_accuracy": 0.667,
        "roc_auc": 0.693,
    },
    "attention-linear-svm": {
        "balanced_accuracy": 0.669,
        "roc_auc": 0.698,
    },
    "attention-rbf-svm": {
        "balanced_accuracy": 0.659,
        "roc_auc": 0.714,
    },
}


@dataclass(frozen=True)
class MarginClassifierEvaluationConfig:
    """Locked uncertainty and promotion policy for the margin family."""

    n_bootstrap: int = 2000
    seed: int = 20260805
    minimum_headline_ba_gain: float = 0.02


def _candidate_family() -> tuple[ProbeConfig, ...]:
    return (
        ProbeConfig(
            name="attention-logistic",
            estimator_family="balanced_l2_logistic_regression",
            hyperparameter_grid={"C": (0.001, 0.01, 0.1)},
            score_output="probability_ground_ball",
            calibrate_threshold=True,
        ),
        ProbeConfig(
            name="attention-linear-svm",
            estimator_family="balanced_linear_svm",
            hyperparameter_grid={"C": (0.001, 0.01, 0.1, 1.0, 10.0)},
            score_output="decision_function_ground_ball",
            calibrate_threshold=True,
        ),
        ProbeConfig(
            name="attention-rbf-svm",
            estimator_family="balanced_rbf_svm",
            hyperparameter_grid={
                "C": (0.3, 1.0, 3.0),
                "gamma": ("scale", 0.001),
            },
            score_output="decision_function_ground_ball",
            calibrate_threshold=True,
        ),
    )


def _validate_config(
    config: MarginClassifierEvaluationConfig,
) -> dict[str, object]:
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
        "n_bootstrap": int(config.n_bootstrap),
        "seed": int(config.seed),
        "minimum_headline_ba_gain": gain,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _membership_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    return (
        predictions[predictions["condition"].eq(EVENT_CONDITION)][
            ["uid", "lineage_group_id", "outer_fold", "y_true"]
        ]
        .drop_duplicates()
        .sort_values("uid")
        .reset_index(drop=True)
    )


def _validate_matched_candidates(
    predictions_by_probe: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, object]]:
    reference_name = BASELINE_PROBE
    reference = _membership_frame(predictions_by_probe[reference_name])
    checks: dict[str, bool] = {}
    for name, predictions in predictions_by_probe.items():
        candidate = _membership_frame(predictions)
        checks[name] = candidate.equals(reference)
    all_matched = all(checks.values())
    if not all_matched:
        raise ValueError(
            "Margin classifier candidates do not share samples, labels, groups, "
            "and outer folds"
        )
    return reference, {
        "reference_probe": reference_name,
        "candidate_matches_reference": checks,
        "all_candidates_matched": all_matched,
        "n_timing_eligible_pairs": int(len(reference)),
        "n_lineage_groups": int(reference["lineage_group_id"].nunique()),
        "n_singleton_lineage_groups": int(
            reference.groupby("lineage_group_id")["uid"].size().eq(1).sum()
        ),
        "outer_folds": sorted(reference["outer_fold"].astype(int).unique().tolist()),
        "matched_fields": [
            "uid",
            "lineage_group_id",
            "outer_fold",
            "y_true",
        ],
    }


def _prediction_arm(
    predictions: pd.DataFrame,
    probe: str,
    rule: str,
    condition: str,
    reference: pd.DataFrame,
) -> np.ndarray:
    selected = predictions[
        predictions["probe"].eq(probe)
        & predictions["decision_rule"].eq(rule)
        & predictions["condition"].eq(condition)
    ][["uid", "lineage_group_id", "outer_fold", "y_true", "y_pred"]]
    aligned = reference.merge(
        selected,
        on=["uid", "lineage_group_id", "outer_fold", "y_true"],
        how="left",
        validate="one_to_one",
    )
    if aligned["y_pred"].isna().any():
        raise ValueError(f"Incomplete prediction arm: {probe}/{rule}/{condition}")
    return aligned["y_pred"].to_numpy(dtype=int)


def _group_bootstrap_indices(
    reference: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> list[np.ndarray]:
    groups = reference["lineage_group_id"].astype(str).to_numpy(dtype=object)
    names = sorted(set(groups))
    rows = [np.flatnonzero(groups == name) for name in names]
    rng = np.random.default_rng(seed)
    return [
        np.concatenate(
            [rows[index] for index in rng.integers(0, len(rows), len(rows))]
        )
        for _ in range(n_bootstrap)
    ]


def _paired_improvements(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    y_true = reference["y_true"].to_numpy(dtype=int)
    baseline = {
        condition: _prediction_arm(
            predictions,
            BASELINE_PROBE,
            BASELINE_RULE,
            condition,
            reference,
        )
        for condition in (EVENT_CONDITION, PRE_CONTROL, REMOVED_CONTROL)
    }
    bootstrap_indices = _group_bootstrap_indices(
        reference, n_bootstrap, seed + 510_013
    )
    arms = (
        predictions[["probe", "decision_rule"]]
        .drop_duplicates()
        .sort_values(["probe", "decision_rule"])
    )
    rows: list[dict[str, object]] = []
    for arm in arms.itertuples(index=False):
        candidate = {
            condition: _prediction_arm(
                predictions,
                str(arm.probe),
                str(arm.decision_rule),
                condition,
                reference,
            )
            for condition in (EVENT_CONDITION, PRE_CONTROL, REMOVED_CONTROL)
        }
        condition_results: dict[str, tuple[float, float, float]] = {}
        condition_draws: dict[str, np.ndarray] = {}
        for condition in (EVENT_CONDITION, PRE_CONTROL, REMOVED_CONTROL):
            observed = float(
                balanced_accuracy_score(y_true, candidate[condition])
                - balanced_accuracy_score(y_true, baseline[condition])
            )
            draws = np.asarray(
                [
                    balanced_accuracy_score(
                        y_true[indices], candidate[condition][indices]
                    )
                    - balanced_accuracy_score(
                        y_true[indices], baseline[condition][indices]
                    )
                    for indices in bootstrap_indices
                ],
                dtype=np.float64,
            )
            low, high = np.percentile(draws, [2.5, 97.5])
            condition_results[condition] = (
                observed,
                float(low),
                float(high),
            )
            condition_draws[condition] = draws
        event = condition_results[EVENT_CONDITION]
        pre = condition_results[PRE_CONTROL]
        removed = condition_results[REMOVED_CONTROL]
        increment_gain = event[0] - pre[0]
        increment_draws = (
            condition_draws[EVENT_CONDITION]
            - condition_draws[PRE_CONTROL]
        )
        increment_low, increment_high = np.percentile(
            increment_draws, [2.5, 97.5]
        )
        rows.append(
            {
                "probe": str(arm.probe),
                "decision_rule": str(arm.decision_rule),
                "baseline_probe": BASELINE_PROBE,
                "baseline_decision_rule": BASELINE_RULE,
                "event_ba_gain": event[0],
                "event_ba_gain_ci_low": event[1],
                "event_ba_gain_ci_high": event[2],
                "strict_pre_ba_rise": pre[0],
                "strict_pre_ba_rise_ci_low": pre[1],
                "strict_pre_ba_rise_ci_high": pre[2],
                "removed_ba_rise": removed[0],
                "removed_ba_rise_ci_low": removed[1],
                "removed_ba_rise_ci_high": removed[2],
                "contact_specific_increment_gain": increment_gain,
                "contact_specific_increment_gain_ci_low": float(
                    increment_low
                ),
                "contact_specific_increment_gain_ci_high": float(
                    increment_high
                ),
                "n_bootstrap": n_bootstrap,
                "bootstrap_unit": "lineage_group_id",
                "n_groups": int(reference["lineage_group_id"].nunique()),
                "n_samples": int(len(reference)),
            }
        )
    return pd.DataFrame(rows)


def _verdict(
    improvements: pd.DataFrame,
    minimum_gain: float,
) -> dict[str, object]:
    evaluated: list[dict[str, object]] = []
    for row in improvements.itertuples(index=False):
        is_baseline = (
            row.probe == BASELINE_PROBE
            and row.decision_rule == BASELINE_RULE
        )
        gain_large_enough = float(row.event_ba_gain) >= minimum_gain
        interval_positive = float(row.event_ba_gain_ci_low) > 0
        controls_not_weakened = (
            float(row.strict_pre_ba_rise) <= 0
            and float(row.removed_ba_rise) <= 0
        )
        qualifies = bool(
            not is_baseline
            and gain_large_enough
            and interval_positive
            and controls_not_weakened
        )
        evaluated.append(
            {
                "probe": str(row.probe),
                "decision_rule": str(row.decision_rule),
                "event_ba_gain": float(row.event_ba_gain),
                "event_ba_gain_ci_low": float(row.event_ba_gain_ci_low),
                "event_ba_gain_ci_high": float(row.event_ba_gain_ci_high),
                "gain_at_least_minimum": gain_large_enough,
                "paired_interval_above_zero": interval_positive,
                "strict_pre_and_removed_not_weakened": controls_not_weakened,
                "qualifies_for_continuation": qualifies,
            }
        )
    qualifying = [item for item in evaluated if item["qualifies_for_continuation"]]
    return {
        "decision": "continue" if qualifying else "stop",
        "continue_classifier_direction": bool(qualifying),
        "eligible_for_downstream_validation": bool(qualifying),
        "headline_replacement_allowed": False,
        "minimum_headline_ba_gain": minimum_gain,
        "baseline": {
            "probe": BASELINE_PROBE,
            "decision_rule": BASELINE_RULE,
        },
        "qualifying_arms": qualifying,
        "evaluated_arms": evaluated,
        "interpretation_zh": (
            "存在满足预注册门槛的候选；仅允许进入后续跨 seed 和完整家族校正。"
            if qualifying
            else "没有候选同时满足增益、配对区间和负控门槛；不替换 attention logistic headline。"
        ),
        "development_evidence_only": True,
        "primary_common_benchmark_unchanged": True,
    }


def _starting_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    fixed_rule = {
        "attention-logistic": "fixed_0.5",
        "attention-linear-svm": "fixed_0.0",
        "attention-rbf-svm": "fixed_0.0",
    }
    rows: list[dict[str, object]] = []
    for probe, starting in STARTING_OBSERVATIONS.items():
        locked = metrics[
            metrics["probe"].eq(probe)
            & metrics["decision_rule"].eq(fixed_rule[probe])
            & metrics["condition"].eq(EVENT_CONDITION)
        ].iloc[0]
        rows.append(
            {
                "probe": probe,
                "decision_rule": fixed_rule[probe],
                "starting_balanced_accuracy": starting["balanced_accuracy"],
                "starting_roc_auc": starting["roc_auc"],
                "locked_balanced_accuracy": float(locked["balanced_accuracy"]),
                "locked_roc_auc": float(locked["roc_auc"]),
                "balanced_accuracy_delta_from_starting": float(
                    locked["balanced_accuracy"] - starting["balanced_accuracy"]
                ),
                "roc_auc_delta_from_starting": float(
                    locked["roc_auc"] - starting["roc_auc"]
                ),
                "starting_role": "pre_ticket_exploratory_observation",
                "locked_role": "leak_safe_locked_rerun",
            }
        )
    return pd.DataFrame(rows)


def _report_zh(
    starting: pd.DataFrame,
    metrics: pd.DataFrame,
    improvements: pd.DataFrame,
    verdict: dict[str, object],
    membership_audit: dict[str, object],
) -> str:
    lines = [
        "# M2D attention margin classifier 锁定复测",
        "",
        "本报告属于开发集探索证据，不改变 ADR-0004 的 mean-pooling L2 Logistic Regression 共同比较。",
        "",
        "## 探索起点与锁定复测",
        "",
        "| 候选 | 起点 BA | 起点 AUC | 锁定 BA | 锁定 AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in starting.itertuples(index=False):
        lines.append(
            f"| {row.probe} | {row.starting_balanced_accuracy:.3f} | "
            f"{row.starting_roc_auc:.3f} | {row.locked_balanced_accuracy:.3f} | "
            f"{row.locked_roc_auc:.3f} |"
        )
    locked_by_probe = starting.set_index("probe")
    linear = locked_by_probe.loc["attention-linear-svm"]
    rbf = locked_by_probe.loc["attention-rbf-svm"]
    lines.extend(
        [
            "",
            f"Linear SVM 的早期 BA 约 0.669 未在锁定复测中重现："
            f"固定规则 BA 为 {float(linear['locked_balanced_accuracy']):.3f}。",
            f"RBF SVM 保留较高 AUC {float(rbf['locked_roc_auc']):.3f}，但固定规则 "
            f"BA 仅为 {float(rbf['locked_balanced_accuracy']):.3f}；排序质量未转化为 "
            "Balanced Accuracy 增益。",
            "",
            "## Fixed 与 inner-OOF calibrated 结果",
            "",
            "| 候选 | 规则 | Event BA | Accuracy | AUC | Macro-F1 | Event-Pre 增量 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    event = metrics[metrics["condition"].eq(EVENT_CONDITION)]
    increments = metrics[
        metrics["condition"].eq("contact_specific_increment")
    ].set_index(["probe", "decision_rule"])
    for row in event.sort_values(["probe", "decision_rule"]).itertuples(index=False):
        increment = increments.loc[(row.probe, row.decision_rule)]
        lines.append(
            f"| {row.probe} | {row.decision_rule} | "
            f"{row.balanced_accuracy:.3f} | {row.accuracy:.3f} | "
            f"{row.roc_auc:.3f} | {row.macro_f1:.3f} | "
            f"{float(increment['balanced_accuracy']):+.3f} |"
        )
    best = improvements[
        ~(
            improvements["probe"].eq(BASELINE_PROBE)
            & improvements["decision_rule"].eq(BASELINE_RULE)
        )
    ].sort_values("event_ba_gain", ascending=False).iloc[0]
    lines.extend(
        [
            "",
            "## 判定",
            "",
            str(verdict["interpretation_zh"]),
            "",
            f"最大配对 BA 差为 {float(best['event_ba_gain']):+.3f}，"
            f"95% lineage-group bootstrap 区间 "
            f"[{float(best['event_ba_gain_ci_low']):+.3f}, "
            f"{float(best['event_ba_gain_ci_high']):+.3f}]。",
            "低于 +0.02 的差异不得晋升为新 headline。",
            "",
            "完整 confusion counts、负控、每折参数和阈值见机器可读 artifacts。",
            "",
            "## 证据边界",
            "",
            f"本次共有 {int(membership_audit['n_timing_eligible_pairs'])} 条配对样本、"
            f"{int(membership_audit['n_lineage_groups'])} 个 lineage groups，其中 "
            f"{int(membership_audit['n_singleton_lineage_groups'])} 个为 singleton。"
            "分组 OOF 和 lineage-group bootstrap 可降低已记录组内依赖造成的偏差，"
            "但不能证明跨比赛、设备、采集者或跨采集流程泛化。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_margin_classifier_evaluation(
    source_bundle: Path,
    output_dir: Path,
    config: MarginClassifierEvaluationConfig = MarginClassifierEvaluationConfig(),
) -> ArtifactBundle:
    """Run and compare the locked calibrated margin-classifier family."""

    config_document = _validate_config(config)
    source_root = Path(source_bundle).resolve()
    output_root = Path(output_dir).resolve()
    probe_bundles = {
        candidate.name: run_exploratory_probe_benchmark(
            source_root,
            output_root / "probe-runs",
            candidate,
        )
        for candidate in _candidate_family()
    }
    predictions_by_probe = {
        name: pd.read_csv(bundle.path("oof_predictions"))
        for name, bundle in probe_bundles.items()
    }
    reference, membership_audit = _validate_matched_candidates(
        predictions_by_probe
    )
    predictions = pd.concat(
        [predictions_by_probe[name] for name in sorted(predictions_by_probe)],
        ignore_index=True,
    )
    fold_assignments = pd.read_csv(
        probe_bundles[BASELINE_PROBE].path("fold_assignments")
    )[["uid", "label", "lineage_group_id", "outer_fold"]]
    metrics = pd.concat(
        [
            pd.read_csv(probe_bundles[name].path("metrics"))
            for name in sorted(probe_bundles)
        ],
        ignore_index=True,
    )
    selections = pd.concat(
        [
            pd.read_csv(probe_bundles[name].path("selections"))
            for name in sorted(probe_bundles)
        ],
        ignore_index=True,
    )
    paired_improvements = _paired_improvements(
        predictions,
        reference,
        int(config.n_bootstrap),
        int(config.seed),
    )
    verdict = _verdict(
        paired_improvements, float(config.minimum_headline_ba_gain)
    )
    starting = _starting_comparison(metrics)
    probe_protocols = {
        name: _read_json(bundle.path("protocol"))
        for name, bundle in probe_bundles.items()
    }
    probe_provenances = {
        name: _read_json(bundle.path("provenance"))
        for name, bundle in probe_bundles.items()
    }
    candidate_documents = [
        probe_protocols[candidate.name]["probe"]
        for candidate in _candidate_family()
    ]
    source_roles = {
        _canonical_sha256(protocol["source_representation"])
        for protocol in probe_protocols.values()
    }
    feature_hashes_by_probe = {
        name: str(document["source_features_sha256"])
        for name, document in probe_provenances.items()
    }
    if len(source_roles) != 1 or len(set(feature_hashes_by_probe.values())) != 1:
        raise ValueError(
            "Margin classifier candidates do not share one source representation"
        )
    shared_source_representation = {
        **probe_protocols[BASELINE_PROBE]["source_representation"],
        "source_features_sha256": next(iter(feature_hashes_by_probe.values())),
    }
    candidate_input_audit = {
        "all_matched": True,
        "source_representation_document_sha256_by_probe": {
            name: _canonical_sha256(protocol["source_representation"])
            for name, protocol in probe_protocols.items()
        },
        "source_features_sha256_by_probe": feature_hashes_by_probe,
    }
    source_protocol = _read_json(source_root / "protocol.json")
    provenance = {
        "source_artifact_id": source_protocol["artifact_id"],
        "source_protocol_sha256": _file_sha256(source_root / "protocol.json"),
        "source_folds_sha256": _file_sha256(source_root / "fold_assignments.csv"),
        "source_exclusions_sha256": _file_sha256(source_root / "exclusions.csv"),
        "probe_artifact_ids": {
            name: bundle.artifact_id for name, bundle in sorted(probe_bundles.items())
        },
        "encoder_inference_runs": 0,
    }
    protocol_document = {
        "protocol_version": MARGIN_CLASSIFIER_PROTOCOL_VERSION,
        "evidence_role": "development_exploratory",
        "source_artifact_id": source_protocol["artifact_id"],
        "primary_common_benchmark_unchanged": True,
        "candidate_family": candidate_documents,
        "shared_source_representation": shared_source_representation,
        "candidate_input_audit": candidate_input_audit,
        "starting_observations": STARTING_OBSERVATIONS,
        "comparison_policy": {
            "baseline_probe": BASELINE_PROBE,
            "baseline_decision_rule": BASELINE_RULE,
            "paired_unit": "lineage_group_id",
            **config_document,
            "continue_requires": [
                "event_ba_gain_at_least_minimum",
                "paired_95pct_interval_above_zero",
                "no_observed_strict_pre_ba_rise",
                "no_observed_removed_ba_rise",
            ],
        },
        "matched_input_policy": membership_audit,
        "provenance_fingerprint": provenance,
    }
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = output_root / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}
    frames = (
        ("fold_assignments", fold_assignments, "fold_assignments.csv"),
        ("oof_predictions", predictions, "oof_predictions.csv"),
        ("metrics", metrics, "metrics.csv"),
        ("selections", selections, "selections.csv"),
        (
            "paired_improvements",
            paired_improvements,
            "paired_improvements.csv",
        ),
        ("starting_comparison", starting, "starting_comparison.csv"),
    )
    for name, frame, filename in frames:
        path = bundle_root / filename
        frame.to_csv(path, index=False)
        artifact_paths[name] = path
    exclusions_path = bundle_root / "exclusions.csv"
    shutil.copy2(source_root / "exclusions.csv", exclusions_path)
    artifact_paths["exclusions"] = exclusions_path

    for name, document, filename in (
        ("membership_audit", membership_audit, "membership_audit.json"),
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
            starting,
            metrics,
            paired_improvements,
            verdict,
            membership_audit,
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
