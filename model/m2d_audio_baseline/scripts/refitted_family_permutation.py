from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .attention_control_representation import (
    AttentionControlRepresentation,
    attention_control_window_roles,
    load_token_table,
)
from .exploratory_probe_benchmark import (
    ProbeConfig,
    _evaluate_attention_controls,
    _probe_document,
    _validated_source,
)
from .short_contact_benchmark import (
    ArtifactBundle,
    LABEL_TO_INT,
    _canonical_sha256,
    _file_sha256,
    _write_json,
)


REFITTED_FAMILY_PROTOCOL_VERSION = "refitted-family-permutation-v1"
PRIMARY_CONDITION = "event_selected_event"


@dataclass(frozen=True)
class PermutationFamilyConfig:
    """Complete development-candidate family and permutation policy."""

    name: str
    candidates: tuple[ProbeConfig, ...]
    n_permutations: int = 999
    seed: int = 20260805


def _family_document(config: PermutationFamilyConfig) -> dict[str, object]:
    if not config.name or not config.name.strip():
        raise ValueError("Permutation family name must not be empty")
    if len(config.candidates) < 2:
        raise ValueError(
            "A family correction requires at least two candidate arms"
        )
    candidate_documents = [_probe_document(item) for item in config.candidates]
    names = [str(item["name"]) for item in candidate_documents]
    if len(set(names)) != len(names):
        raise ValueError("Permutation candidate names must be unique")
    scientific_fingerprints = {
        _canonical_sha256(
            {key: value for key, value in item.items() if key != "name"}
        )
        for item in candidate_documents
    }
    if len(scientific_fingerprints) != len(candidate_documents):
        raise ValueError(
            "Permutation candidates must be scientifically distinct"
        )
    if (
        not isinstance(config.n_permutations, int)
        or config.n_permutations < 1
    ):
        raise ValueError("n_permutations must be a positive integer")
    if not isinstance(config.seed, int):
        raise ValueError("Permutation seed must be an integer")
    return {
        "name": config.name,
        "candidates": candidate_documents,
        "n_permutations": config.n_permutations,
        "seed": config.seed,
        "family_scope": "all_declared_development_candidates",
        "minimum_family_size_enforced": 2,
        "primary_condition": PRIMARY_CONDITION,
    }


def _label_assignment_sha256(
    uids: tuple[str, ...], labels: np.ndarray
) -> str:
    document = [
        {"uid": uid, "label": int(label)}
        for uid, label in zip(uids, labels, strict=True)
    ]
    return hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _permute_group_label_vectors(
    paired: pd.DataFrame,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exchange whole lineage label vectors within fold/size strata."""

    result = labels.copy()
    working = paired[["uid", "lineage_group_id", "outer_fold"]].copy()
    working["position"] = np.arange(len(working), dtype=int)
    sizes = working.groupby("lineage_group_id")["uid"].transform("size")
    working["lineage_group_size"] = sizes.astype(int)
    for (_fold, _size), stratum in working.groupby(
        ["outer_fold", "lineage_group_size"], sort=True
    ):
        target_groups = [
            group.sort_values("uid")["position"].to_numpy(dtype=int)
            for _group_name, group in stratum.groupby(
                "lineage_group_id", sort=True
            )
        ]
        source_order = rng.permutation(len(target_groups))
        source_vectors = [labels[positions].copy() for positions in target_groups]
        for target_positions, source_index in zip(
            target_groups, source_order, strict=True
        ):
            result[target_positions] = source_vectors[int(source_index)]
    return result


def _metric_rows(
    metrics: pd.DataFrame,
    *,
    candidate: str,
    permutation: int,
    label_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in metrics.itertuples(index=False):
        rows.append(
            {
                "candidate": candidate,
                "decision_rule": str(row.decision_rule),
                "condition": str(row.condition),
                "permutation": permutation,
                "is_observed": permutation == -1,
                "label_assignment_sha256": label_sha256,
                "balanced_accuracy": float(row.balanced_accuracy),
            }
        )
    return rows


def _summary_tables(
    scores: pd.DataFrame,
    n_permutations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event = scores[scores["condition"].eq(PRIMARY_CONDITION)].copy()
    observed = event[event["is_observed"]].set_index(
        ["candidate", "decision_rule"]
    )
    null = event[~event["is_observed"]]
    hypotheses = list(observed.index)
    if len(hypotheses) < 2:
        raise ValueError(
            "Family correction requires at least two candidate/decision hypotheses"
        )
    max_null = (
        null.groupby("permutation")["balanced_accuracy"].max().sort_index()
    )
    if len(max_null) != n_permutations:
        raise RuntimeError("Permutation scores do not cover every replicate")

    summary_rows: list[dict[str, object]] = []
    screening_rows: list[dict[str, object]] = []
    for candidate, decision_rule in hypotheses:
        observed_score = float(
            observed.loc[(candidate, decision_rule), "balanced_accuracy"]
        )
        candidate_null = null[
            null["candidate"].eq(candidate)
            & null["decision_rule"].eq(decision_rule)
        ].sort_values("permutation")["balanced_accuracy"]
        raw_p = float(
            (1 + int((candidate_null >= observed_score).sum()))
            / (n_permutations + 1)
        )
        family_p = float(
            (1 + int((max_null >= observed_score).sum()))
            / (n_permutations + 1)
        )
        summary_rows.append(
            {
                "candidate": candidate,
                "decision_rule": decision_rule,
                "observed_balanced_accuracy": observed_score,
                "null_mean": float(candidate_null.mean()),
                "null_std": float(candidate_null.std(ddof=0)),
                "raw_p_value": raw_p,
                "max_stat_familywise_p_value": family_p,
                "n_permutations": n_permutations,
                "n_family_hypotheses": len(hypotheses),
            }
        )
        observed_arm = scores[
            scores["is_observed"]
            & scores["candidate"].eq(candidate)
            & scores["decision_rule"].eq(decision_rule)
        ].set_index("condition")["balanced_accuracy"]
        screening_rows.append(
            {
                "candidate": candidate,
                "decision_rule": decision_rule,
                "event_balanced_accuracy": observed_score,
                "event_applied_to_pre_balanced_accuracy": float(
                    observed_arm.loc["event_selected_pre"]
                ),
                "pre_fitted_control_balanced_accuracy": float(
                    observed_arm.loc["pre_selected_pre"]
                ),
                "event_applied_to_removed_balanced_accuracy": float(
                    observed_arm.loc["event_selected_removed"]
                ),
                "removed_fitted_control_balanced_accuracy": float(
                    observed_arm.loc["removed_selected_removed"]
                ),
                "contact_specific_increment": float(
                    observed_arm.loc["contact_specific_increment"]
                ),
                "raw_p_value": raw_p,
                "max_stat_familywise_p_value": family_p,
                "screening_alpha": 0.05,
                "family_complete_as_declared": True,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(screening_rows)


def _fixed_prediction_diagnostic(
    summary: pd.DataFrame,
    observed_predictions: pd.DataFrame,
    assignments: pd.DataFrame,
    n_permutations: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for summary_row in summary.itertuples(index=False):
        candidate = str(summary_row.candidate)
        decision_rule = str(summary_row.decision_rule)
        event = observed_predictions[
            observed_predictions["candidate"].eq(candidate)
            & observed_predictions["decision_rule"].eq(decision_rule)
            & observed_predictions["condition"].eq(PRIMARY_CONDITION)
        ].set_index("uid")
        if event.empty or event.index.has_duplicates:
            raise RuntimeError(
                f"Observed event predictions are malformed for {candidate}"
            )
        observed_score = float(
            balanced_accuracy_score(event["y_true"], event["y_pred"])
        )
        fixed_null_scores: list[float] = []
        for permutation in range(n_permutations):
            permuted = assignments[
                assignments["permutation"].eq(permutation)
            ].set_index("uid")
            if set(permuted.index) != set(event.index):
                raise RuntimeError(
                    "Permutation assignments do not match observed predictions"
                )
            aligned = permuted.loc[event.index]
            fixed_null_scores.append(
                float(
                    balanced_accuracy_score(
                        aligned["y_true"], event["y_pred"]
                    )
                )
            )
        fixed_raw_p = float(
            (1 + int(np.sum(np.asarray(fixed_null_scores) >= observed_score)))
            / (n_permutations + 1)
        )
        full_raw_p = float(summary_row.raw_p_value)
        rows.append(
            {
                "candidate": candidate,
                "decision_rule": decision_rule,
                "observed_balanced_accuracy": observed_score,
                "fixed_prediction_null_mean": float(
                    np.mean(fixed_null_scores)
                ),
                "fixed_prediction_raw_p_value": fixed_raw_p,
                "full_refit_raw_p_value": full_raw_p,
                "screening_alpha": 0.05,
                "verdict_changed_by_refitting": (
                    (fixed_raw_p < 0.05) != (full_raw_p < 0.05)
                ),
                "fixed_prediction_method_valid_for_claim": False,
            }
        )
    return pd.DataFrame(rows)


def _load_checkpoint(
    path: Path, expected_fingerprint: str
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_fingerprint") != expected_fingerprint
    ):
        return None
    payload_sha256 = payload.get("payload_sha256")
    content = {
        key: value for key, value in payload.items() if key != "payload_sha256"
    }
    if payload_sha256 != _canonical_sha256(content):
        return None
    for name in ("scores", "selections", "fit_audit", "predictions"):
        if not isinstance(payload.get(name), list):
            return None
    return payload


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = dict(payload)
    content["payload_sha256"] = _canonical_sha256(payload)
    temporary_path = path.with_suffix(".tmp")
    _write_json(temporary_path, content)
    temporary_path.replace(path)


def run_refitted_family_permutation(
    source_bundle: Path,
    output_dir: Path,
    config: PermutationFamilyConfig,
) -> ArtifactBundle:
    """Refit a synchronized candidate family under groupwise permutation."""

    family_document = _family_document(config)
    source_root = Path(source_bundle).resolve()
    source_protocol, feature_path = _validated_source(source_root)
    source_folds_path = source_root / "fold_assignments.csv"
    source_exclusions_path = source_root / "exclusions.csv"
    source_folds = pd.read_csv(source_folds_path).sort_values("uid").reset_index(
        drop=True
    )
    mapped = source_folds["label"].map(LABEL_TO_INT)
    if mapped.isna().any():
        raise ValueError("Source folds contain unknown labels")

    provenance = {
        "source_artifact_id": str(source_protocol["artifact_id"]),
        "source_protocol_sha256": _file_sha256(source_root / "protocol.json"),
        "source_features_sha256": _file_sha256(feature_path),
        "source_folds_sha256": _file_sha256(source_folds_path),
        "source_exclusions_sha256": _file_sha256(source_exclusions_path),
        "encoder_inference_runs": 0,
    }
    protocol_document = {
        "protocol_version": REFITTED_FAMILY_PROTOCOL_VERSION,
        "evidence_role": "development_exploratory_family_corrected",
        "primary_common_benchmark_unchanged": True,
        "source_artifact_id": source_protocol["artifact_id"],
        "fold_policy": source_protocol["fold_policy"],
        "permutation_policy": {
            "unit": "lineage_group_label_vector",
            "exchangeability_blocks": [
                "locked_outer_fold",
                "lineage_group_size",
            ],
            "preserves": [
                "locked_outer_folds",
                "per_fold_class_totals",
                "lineage_group_membership",
                "within_group_label_vector",
                "mixed_label_lineage_groups",
            ],
            "does_not_exchange_across": [
                "outer_fold",
                "lineage_group_size",
            ],
        },
        "family": family_document,
        "provenance_fingerprint": provenance,
    }
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = Path(output_dir).resolve() / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    candidate_documents = family_document["candidates"]
    if not isinstance(candidate_documents, list):
        raise AssertionError("Validated family candidates are not a list")

    roles = attention_control_window_roles(
        "event_200ms", "pre_200ms", "removed_200ms"
    )
    try:
        paired_reference = AttentionControlRepresentation.from_token_table(
            load_token_table(feature_path), source_folds, roles
        ).paired
    except ValueError as error:
        raise ValueError(str(error)) from error
    paired_uids = tuple(paired_reference["uid"].astype(str))
    original_labels = (
        paired_reference["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    )
    labels_by_replicate: list[tuple[int, np.ndarray]] = [
        (-1, original_labels)
    ]
    rng = np.random.default_rng(config.seed)
    for permutation in range(config.n_permutations):
        labels_by_replicate.append(
            (
                permutation,
                _permute_group_label_vectors(
                    paired_reference, original_labels, rng
                ),
            )
        )

    score_rows: list[dict[str, object]] = []
    selection_frames: list[pd.DataFrame] = []
    observed_prediction_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    checkpoint_root = bundle_root / "checkpoints"

    for permutation, labels in labels_by_replicate:
        label_sha256 = _label_assignment_sha256(paired_uids, labels)
        labels_by_uid = None
        if permutation >= 0:
            labels_by_uid = dict(
                zip(paired_uids, labels.tolist(), strict=True)
            )
        replicate_name = (
            "observed"
            if permutation == -1
            else f"permutation-{permutation:06d}"
        )
        for candidate_index, candidate_document in enumerate(
            candidate_documents
        ):
            candidate = str(candidate_document["name"])
            checkpoint_path = (
                checkpoint_root
                / replicate_name
                / f"candidate-{candidate_index:03d}.json"
            )
            checkpoint_fingerprint = _canonical_sha256(
                {
                    "artifact_id": artifact_id,
                    "candidate": candidate_document,
                    "permutation": permutation,
                    "label_assignment_sha256": label_sha256,
                }
            )
            checkpoint = _load_checkpoint(
                checkpoint_path, checkpoint_fingerprint
            )
            if checkpoint is None:
                predictions, metrics, selections, paired, audit = (
                    _evaluate_attention_controls(
                        feature_path,
                        source_folds,
                        source_protocol,
                        candidate_document,
                        labels_by_uid=labels_by_uid,
                    )
                )
                if not paired[["uid", "outer_fold"]].equals(
                    paired_reference[["uid", "outer_fold"]]
                ):
                    raise RuntimeError(
                        "A refitted candidate changed the paired population"
                    )
                candidate_score_rows = _metric_rows(
                    metrics,
                    candidate=candidate,
                    permutation=permutation,
                    label_sha256=label_sha256,
                )
                selections.insert(
                    0, "label_assignment_sha256", label_sha256
                )
                selections.insert(0, "is_observed", permutation == -1)
                selections.insert(0, "permutation", permutation)
                selections.insert(0, "candidate", candidate)
                audit_row = {
                    "candidate": candidate,
                    "permutation": permutation,
                    "is_observed": permutation == -1,
                    "label_assignment_sha256": label_sha256,
                    "representation_fits": audit.representation_fits,
                    "model_selection_fits": audit.model_selection_fits,
                    "threshold_selection_fits": (
                        audit.threshold_selection_fits
                    ),
                    "outer_probe_fits": audit.outer_probe_fits,
                }
                prediction_rows: list[dict[str, object]] = []
                if permutation == -1:
                    predictions.insert(0, "candidate", candidate)
                    prediction_rows = predictions.to_dict(orient="records")
                checkpoint = {
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                    "scores": candidate_score_rows,
                    "selections": selections.to_dict(orient="records"),
                    "fit_audit": [audit_row],
                    "predictions": prediction_rows,
                }
                _write_checkpoint(checkpoint_path, checkpoint)
            score_rows.extend(checkpoint["scores"])
            selection_frames.append(pd.DataFrame(checkpoint["selections"]))
            audit_rows.extend(checkpoint["fit_audit"])
            if permutation == -1:
                observed_prediction_frames.append(
                    pd.DataFrame(checkpoint["predictions"])
                )
            checkpoint_records.append(
                {
                    "candidate": candidate,
                    "permutation": permutation,
                    "label_assignment_sha256": label_sha256,
                    "checkpoint_fingerprint": checkpoint_fingerprint,
                    "path": checkpoint_path.relative_to(bundle_root).as_posix(),
                    "sha256": _file_sha256(checkpoint_path),
                }
            )

    scores = pd.DataFrame(score_rows).sort_values(
        ["permutation", "candidate", "decision_rule", "condition"]
    ).reset_index(drop=True)
    summary, screening = _summary_tables(scores, config.n_permutations)
    selections = pd.concat(selection_frames, ignore_index=True)
    observed_predictions = pd.concat(
        observed_prediction_frames, ignore_index=True
    )
    fit_audit = pd.DataFrame(audit_rows).sort_values(
        ["permutation", "candidate"]
    ).reset_index(drop=True)
    assignment_rows: list[dict[str, object]] = []
    for permutation, labels in labels_by_replicate:
        label_sha256 = _label_assignment_sha256(paired_uids, labels)
        for position, row in enumerate(
            paired_reference.itertuples(index=False)
        ):
            assignment_rows.append(
                {
                    "permutation": permutation,
                    "is_observed": permutation == -1,
                    "label_assignment_sha256": label_sha256,
                    "uid": str(row.uid),
                    "lineage_group_id": str(row.lineage_group_id),
                    "outer_fold": int(row.outer_fold),
                    "y_true": int(labels[position]),
                }
            )
    assignments = pd.DataFrame(assignment_rows)
    fixed_prediction_diagnostic = _fixed_prediction_diagnostic(
        summary,
        observed_predictions,
        assignments,
        config.n_permutations,
    )

    artifact_paths: dict[str, Path] = {}
    for name, frame, filename in (
        ("permutation_scores", scores, "permutation_scores.csv"),
        ("permutation_summary", summary, "permutation_summary.csv"),
        ("screening_inputs", screening, "screening_inputs.csv"),
        (
            "permutation_selections",
            selections,
            "permutation_selections.csv",
        ),
        ("fit_audit", fit_audit, "fit_audit.csv"),
        (
            "fixed_prediction_diagnostic",
            fixed_prediction_diagnostic,
            "fixed_prediction_diagnostic.csv",
        ),
        (
            "permutation_assignments",
            assignments,
            "permutation_assignments.csv",
        ),
        (
            "observed_predictions",
            observed_predictions,
            "observed_predictions.csv",
        ),
        ("fold_assignments", paired_reference, "fold_assignments.csv"),
    ):
        path = bundle_root / filename
        frame.to_csv(path, index=False)
        artifact_paths[name] = path

    exclusions_path = bundle_root / "exclusions.csv"
    shutil.copy2(source_exclusions_path, exclusions_path)
    artifact_paths["exclusions"] = exclusions_path
    checkpoint_index_path = bundle_root / "checkpoint_index.json"
    _write_json(
        checkpoint_index_path,
        {
            "artifact_id": artifact_id,
            "checkpoints": checkpoint_records,
        },
    )
    artifact_paths["checkpoint_index"] = checkpoint_index_path
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
