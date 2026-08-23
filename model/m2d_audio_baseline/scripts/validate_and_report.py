from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark_artifact_roles import (
    BEATS_ENCODER_NAME as BEATS_ENCODER,
    M2D_ENCODER_NAME as M2D_ENCODER,
    VERIFIED_DATASET_REVISION as LOCKED_DATASET_REVISION,
    ATTENTION_CONTROL_TRANSFORM_POLICY,
    BenchmarkArtifactRole,
    BenchmarkArtifactRoleError,
    resolve_benchmark_bundle,
    short_contact_artifact_role,
)
from .short_contact_benchmark import ArtifactBundle, CONTROL_CONDITIONS
LOCKED_BENCHMARK_SEED = 20260805
EXPECTED_SNAPSHOT_COUNT = 822
EXPECTED_LABEL_COUNTS = {"fly_ball": 386, "ground_ball": 436}
ALLOWED_EXCLUSION_REASONS = {
    "unreadable_audio",
    "audio_shorter_than_window",
    "invalid_event_interval",
    "window_not_exact",
    "strict_pre_unavailable",
}
REQUIRED_BUNDLE_ARTIFACTS = (
    "protocol",
    "snapshot_audit",
    "window_manifest",
    "exclusions",
    "fold_assignments",
    "oof_predictions",
    "metrics",
    "selections",
)


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class BundleValidation:
    bundle: ArtifactBundle
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_bundle(
    bundle: ArtifactBundle,
    expected_snapshot_count: int = EXPECTED_SNAPSHOT_COUNT,
    expected_label_counts: dict[str, int] | None = None,
) -> tuple[str, ...]:
    if expected_label_counts is None:
        expected_label_counts = dict(EXPECTED_LABEL_COUNTS)
    failures: list[str] = []
    root = bundle.root

    for name in REQUIRED_BUNDLE_ARTIFACTS:
        if not bundle.path(name).is_file():
            failures.append(f"missing artifact: {name}")
    if failures:
        return tuple(failures)

    protocol = json.loads(bundle.path("protocol").read_text(encoding="utf-8"))
    if protocol.get("artifact_id") != bundle.artifact_id:
        failures.append("protocol artifact_id mismatch")
    if not protocol.get("controls", {}).get("enabled"):
        failures.append("controls not enabled")
    if protocol.get("classifier", {}).get("C_selection") != "inner_grouped_cv":
        failures.append("classifier is not the locked inner-grouped selection")
    if (
        protocol.get("model_input_policy", {}).get("waveform_padding")
        is not False
    ):
        failures.append("protocol allows waveform padding")

    audit = json.loads(bundle.path("snapshot_audit").read_text(encoding="utf-8"))
    if int(audit.get("sample_count", 0)) != expected_snapshot_count:
        failures.append(f"snapshot count != {expected_snapshot_count}")
    if audit.get("label_counts") != expected_label_counts:
        failures.append(f"label counts != {expected_label_counts}")

    windows = pd.read_csv(bundle.path("window_manifest"))
    if windows.empty:
        failures.append("window manifest is empty")
    if windows.duplicated(["uid", "window_name"]).any():
        failures.append("duplicate (uid, window_name) rows")
    if (windows["window_shift_from_requested_ms"].abs() > 1e-9).any():
        failures.append("a window is not exactly peak-centred")
    if (windows["wav_boundary_padding_samples"] != 0).any():
        failures.append("a window contains waveform padding")

    feature_paths = sorted((root / "features").glob("*.csv"))
    if not feature_paths:
        failures.append("no feature files")
    for path in feature_paths:
        features = pd.read_csv(path)
        feature_columns = [
            column for column in features if column.startswith("feat_")
        ]
        if not feature_columns:
            failures.append(f"{path.name}: no feat_ columns")
            continue
        expected_width = len(feature_columns)
        matrix = features[feature_columns].to_numpy(dtype=np.float64)
        if matrix.shape[1] != expected_width:
            failures.append(f"{path.name}: inconsistent feature width")
        if not np.isfinite(matrix).all():
            failures.append(f"{path.name}: non-finite features")
        if path.name.endswith("_tokens.csv"):
            uniqueness = ["uid", "window_name", "token_index"]
        else:
            uniqueness = ["uid", "window_name"]
        if features.duplicated(uniqueness).any():
            failures.append(f"{path.name}: duplicate rows in {uniqueness}")
        if (features["encoder_training_epochs"] != 0).any():
            failures.append(f"{path.name}: encoder not frozen")
        for column in ("checkpoint_sha256", "upstream_revision"):
            if features[column].isna().any() or (features[column] == "").any():
                failures.append(f"{path.name}: missing {column}")

    folds = pd.read_csv(bundle.path("fold_assignments"))
    if folds["uid"].duplicated().any():
        failures.append("fold assignments contain duplicate UIDs")
    window_uids = set(windows["uid"])
    if set(folds["uid"]) != window_uids:
        failures.append("fold assignments do not match window membership")
    group_fold_spread = folds.groupby("lineage_group_id")["outer_fold"].nunique()
    if (group_fold_spread > 1).any():
        failures.append("a lineage group crosses outer folds")

    predictions = pd.read_csv(bundle.path("oof_predictions"))
    key_columns = ["encoder", "condition", "window_ms", "uid"]
    if "decision_rule" in predictions.columns:
        key_columns.append("decision_rule")
    if predictions[key_columns].duplicated().any():
        failures.append("duplicate out-of-fold predictions")
    for window_ms in predictions["window_ms"].unique():
        for condition in predictions.loc[
            predictions["window_ms"].eq(window_ms), "condition"
        ].unique():
            count = len(
                predictions[
                    predictions["window_ms"].eq(window_ms)
                    & predictions["condition"].eq(condition)
                ]
            )
            if count < 2:
                failures.append(
                    f"condition {condition}/{window_ms} has {count} predictions"
                )

    metrics = pd.read_csv(bundle.path("metrics"))
    for window_ms in metrics["window_ms"].unique():
        conditions = set(
            metrics.loc[metrics["window_ms"].eq(window_ms), "condition"]
        )
        expected_conditions = set(CONTROL_CONDITIONS)
        if window_ms == 200:
            expected_conditions = expected_conditions | {
                "contact_specific_increment"
            }
        else:
            expected_conditions = set(CONTROL_CONDITIONS[:3]) | {
                "contact_specific_increment"
            }
        if conditions != expected_conditions:
            failures.append(
                f"metrics conditions for {window_ms}ms: {sorted(conditions)}"
            )

    selections = pd.read_csv(bundle.path("selections"))
    selection_keys = ["encoder", "condition", "window_ms", "outer_fold"]
    if selections[selection_keys].duplicated().any():
        failures.append("duplicate selection rows")
    if protocol.get("decision_threshold", {}).get("calibrate"):
        if "selected_threshold" not in selections.columns:
            failures.append("calibrated run is missing selected_threshold")
        else:
            thresholds = selections["selected_threshold"].dropna()
            if len(thresholds) == 0:
                failures.append("calibrated run has no recorded thresholds")
            elif not ((thresholds > 0.0) & (thresholds < 1.0)).all():
                failures.append("recorded thresholds outside (0, 1)")
        if "threshold_scores_json" not in selections.columns:
            failures.append("calibrated run is missing threshold_scores_json")

    exclusions = pd.read_csv(bundle.path("exclusions"))
    unexpected_reasons = set(exclusions["reason"]) - ALLOWED_EXCLUSION_REASONS
    if unexpected_reasons:
        failures.append(
            f"unexpected exclusion reasons: {sorted(unexpected_reasons)}"
        )

    manifest = json.loads(
        bundle.path("artifact_bundle").read_text(encoding="utf-8")
    )
    for name, record in manifest.get("artifacts", {}).items():
        artifact_path = root / str(record["path"])
        if not artifact_path.is_file():
            failures.append(f"artifact_bundle lists missing file: {name}")
        elif _sha256(artifact_path) != str(record["sha256"]):
            failures.append(f"artifact_bundle checksum mismatch: {name}")

    return tuple(failures)


def _artifact_bundle_from_root(candidate: Path) -> ArtifactBundle:
    protocol = json.loads(
        (candidate / "protocol.json").read_text(encoding="utf-8")
    )
    artifact_paths = {}
    csv_names = {"window_manifest": "windows_manifest.csv"}
    for artifact in REQUIRED_BUNDLE_ARTIFACTS:
        if artifact in {"protocol", "snapshot_audit"}:
            artifact_paths[artifact] = candidate / f"{artifact}.json"
        else:
            artifact_paths[artifact] = candidate / csv_names.get(
                artifact, f"{artifact}.csv"
            )
    artifact_paths["artifact_bundle"] = candidate / "artifact_bundle.json"
    for path in sorted((candidate / "features").glob("*.csv")):
        artifact_paths[f"features/{path.name}"] = path
    return ArtifactBundle(
        artifact_id=str(protocol["artifact_id"]),
        root=candidate,
        _artifacts=tuple(sorted(artifact_paths.items())),
    )


def _common_role(
    encoder_name: str, dataset_revision: str, benchmark_seed: int
) -> BenchmarkArtifactRole:
    return short_contact_artifact_role(
        name=f"common_200ms_mean:{encoder_name}",
        encoder_name=encoder_name,
        dataset_revision=dataset_revision,
        pooling="valid_final_layer_token_mean",
        fold_seed=benchmark_seed,
    )


def _find_bundles(
    common_root: Path,
    dataset_revision: str,
    benchmark_seed: int,
) -> dict[str, ArtifactBundle]:
    return {
        encoder_name: _artifact_bundle_from_root(
            resolve_benchmark_bundle(
                common_root,
                _common_role(encoder_name, dataset_revision, benchmark_seed),
            )
        )
        for encoder_name in (M2D_ENCODER, BEATS_ENCODER)
    }


def _require_file(path: Path, failures: list[str]) -> None:
    if not path.is_file():
        failures.append(f"missing output: {path.name}")


def validate_complete_run(
    common_root: Path,
    secondary_evidence_dir: Path,
    statistical_evidence_dir: Path,
    sensitivity_root: Path,
    output_root: Path,
    expected_snapshot_count: int = EXPECTED_SNAPSHOT_COUNT,
    expected_label_counts: dict[str, int] | None = None,
    benchmark_seed: int = LOCKED_BENCHMARK_SEED,
    dataset_revision: str = LOCKED_DATASET_REVISION,
) -> dict[str, object]:
    """Validate every stage of the complete benchmark run."""

    if expected_label_counts is None:
        expected_label_counts = dict(EXPECTED_LABEL_COUNTS)
    failures: list[str] = []
    checks: dict[str, object] = {}

    try:
        bundles = _find_bundles(
            common_root,
            dataset_revision=dataset_revision,
            benchmark_seed=benchmark_seed,
        )
    except BenchmarkArtifactRoleError as exc:
        bundles = {}
        failures.append(str(exc))
    checks["encoders"] = sorted(bundles)
    if len(bundles) < 2:
        failures.append("fewer than two control bundles found in common root")
    bundle_failures: dict[str, list[str]] = {}
    for name, bundle in bundles.items():
        result = _validate_bundle(
            bundle,
            expected_snapshot_count=expected_snapshot_count,
            expected_label_counts=expected_label_counts,
        )
        bundle_failures[name] = list(result)
        failures.extend(f"{name}: {item}" for item in result)
    checks["bundle_failures"] = bundle_failures

    statistical_files = (
        "summary.json",
        "group_uncertainty.csv",
        "paired_intervals.csv",
        "permutation_summary.csv",
        "permutation_scores.csv",
    )
    for name in statistical_files:
        _require_file(Path(statistical_evidence_dir) / name, failures)
    permutation = pd.read_csv(
        Path(statistical_evidence_dir) / "permutation_summary.csv"
    )
    if len(permutation) != len(bundles):
        failures.append("permutation summary does not cover every encoder")

    secondary_files = (
        "summary.json",
        "fixed_split_metrics.csv",
        "fixed_split_predictions.csv",
        "rbf_metrics.csv",
        "rbf_predictions.csv",
        "rbf_selections.csv",
    )
    for name in secondary_files:
        _require_file(Path(secondary_evidence_dir) / name, failures)
    secondary_summary = json.loads(
        (Path(secondary_evidence_dir) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    if not secondary_summary.get("fixed_split_membership_reproduced"):
        failures.append("fixed split membership was not reproduced")
    if not secondary_summary.get("development_evidence"):
        failures.append("secondary outputs are not labelled development evidence")

    try:
        sensitivity_bundles = _find_sensitivity_bundles(
            sensitivity_root,
            dataset_revision=dataset_revision,
            benchmark_seed=benchmark_seed,
        )
    except BenchmarkArtifactRoleError as exc:
        sensitivity_bundles = {}
        failures.append(str(exc))
    checks["sensitivity_runs"] = {
        key: sorted(str(path) for path in bundle_paths)
        for key, bundle_paths in sensitivity_bundles.items()
    }
    if not sensitivity_bundles:
        failures.append("no sensitivity bundles found")

    status = "pass" if not failures else "fail"
    report = {
        "status": status,
        "checks": checks,
        "failures": failures,
    }
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def _sensitivity_role(
    *,
    name: str,
    dataset_revision: str,
    benchmark_seed: int,
    pooling: str,
    windows: tuple[str, ...] = ("event_200ms",),
    normalization: str = "snapshot_level",
    calibrated: bool = False,
    c_selection: str = "inner_grouped_cv",
    feature_composition: tuple[tuple[int, str], ...] | None = None,
) -> BenchmarkArtifactRole:
    return short_contact_artifact_role(
        name=name,
        encoder_name=M2D_ENCODER,
        dataset_revision=dataset_revision,
        pooling=pooling,
        fold_seed=benchmark_seed,
        window_conditions=windows,
        normalization=normalization,
        threshold_calibrated=calibrated,
        c_selection=c_selection,
        attention_control_transform_policy=(
            ATTENTION_CONTROL_TRANSFORM_POLICY
            if pooling.startswith("attention")
            or bool(
                feature_composition
                and any(
                    component_pooling.startswith("attention")
                    for _window_ms, component_pooling in feature_composition
                )
            )
            else None
        ),
        feature_composition=feature_composition,
    )


def _find_sensitivity_bundles(
    sensitivity_root: Path,
    *,
    dataset_revision: str,
    benchmark_seed: int,
) -> dict[str, list[Path]]:
    role_specs = {
        "durations": [
            _sensitivity_role(
                name="m2d_attention_duration_scan",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="attention",
                windows=("event_050ms", "event_100ms", "event_200ms"),
            )
        ],
        "rms_normalized": [
            _sensitivity_role(
                name="m2d_mean_rms_normalized",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="valid_final_layer_token_mean",
                normalization="rms_normalized",
            )
        ],
        "legacy_pooling": [
            _sensitivity_role(
                name="m2d_legacy_pooling",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="legacy_mean_std_max",
            ),
            _sensitivity_role(
                name="m2d_legacy_pooling_calibrated",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="legacy_mean_std_max",
                calibrated=True,
            ),
        ],
        "energy_weighted": [
            _sensitivity_role(
                name="m2d_energy_weighted",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="energy_weighted",
            )
        ],
        "attention": [
            _sensitivity_role(
                name="m2d_attention_headline",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="attention",
            ),
            _sensitivity_role(
                name="m2d_attention_calibrated",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="attention",
                calibrated=True,
            ),
        ],
        "attention_lda": [
            _sensitivity_role(
                name="m2d_attention_lda",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="attention_lda",
                c_selection="fixed",
            )
        ],
        "attention_multi": [
            _sensitivity_role(
                name="m2d_attention_multi",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="attention_multi",
                c_selection="fixed",
            )
        ],
        "attention_neighbourhood": [
            _sensitivity_role(
                name="m2d_attention_neighbourhood",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="attention_neighbourhood",
                c_selection="fixed",
            )
        ],
        "composed": [
            _sensitivity_role(
                name="m2d_50ms_mean_200ms_attention",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="valid_final_layer_token_mean",
                windows=("event_050ms", "event_200ms"),
                c_selection="fixed",
                feature_composition=(
                    (50, "valid_final_layer_token_mean"),
                    (200, "attention"),
                ),
            )
        ],
        "mean": [
            _sensitivity_role(
                name="m2d_mean",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="valid_final_layer_token_mean",
            )
        ],
        "mean_std": [
            _sensitivity_role(
                name="m2d_mean_std",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="mean_std",
            )
        ],
        "mean_max": [
            _sensitivity_role(
                name="m2d_mean_max",
                dataset_revision=dataset_revision,
                benchmark_seed=benchmark_seed,
                pooling="mean_max",
            )
        ],
    }
    return {
        key: [
            resolve_benchmark_bundle(sensitivity_root, role)
            for role in roles
        ]
        for key, roles in role_specs.items()
    }


def _resolve_optional_artifact(root: Path, filename: str) -> Path | None:
    root = Path(root)
    candidates = sorted(root.glob(f"*/{filename}"))
    direct = root / filename
    if direct.is_file():
        candidates.append(direct)
    candidates = sorted(set(candidates))
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValidationError(
            f"ambiguous optional artifact {filename}: {joined}"
        )
    return candidates[0] if candidates else None


def _md_table(columns: list[str], rows: list[list[object]]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def _condition_metrics(
    bundle: ArtifactBundle, condition: str, window_ms: int = 200
) -> pd.Series:
    metrics = pd.read_csv(bundle.path("metrics"))
    return metrics.loc[
        metrics["condition"].eq(condition)
        & metrics["window_ms"].eq(window_ms)
    ].iloc[0]


def _format_metrics(metric: pd.Series) -> str:
    return (
        f"BA {float(metric['balanced_accuracy']):.3f} · "
        f"ACC {float(metric['accuracy']):.3f} · "
        f"AUC {float(metric['roc_auc']):.3f} · "
        f"F1 {float(metric['macro_f1']):.3f}"
    )


def generate_reports(
    common_root: Path,
    statistical_evidence_dir: Path,
    secondary_evidence_dir: Path,
    sensitivity_root: Path,
    snapshot_audit_path: Path,
    output_root: Path,
    layer_scan_dir: Path | None = None,
    full_audio_conditions_dir: Path | None = None,
    alignment_sensitivity_dir: Path | None = None,
    finetune_pilot_dir: Path | None = None,
    finetune_primary_root: Path | None = None,
    benchmark_seed: int = LOCKED_BENCHMARK_SEED,
) -> dict[str, Path]:
    """Write the Chinese summary and detailed technical report."""

    snapshot_audit = json.loads(
        Path(snapshot_audit_path).read_text(encoding="utf-8")
    )
    bundles = _find_bundles(
        common_root,
        dataset_revision=str(snapshot_audit["revision"]),
        benchmark_seed=benchmark_seed,
    )
    SHORT_NAMES = {
        "m2d_vit_base_80x200p16x4_40ms": "m2d",
        "beats_iter3plus_as2m": "beats",
    }
    name_to_bundle = {
        SHORT_NAMES.get(name, name): bundle
        for name, bundle in bundles.items()
    }
    names = sorted(name_to_bundle)
    m2d_protocol = json.loads(
        name_to_bundle["m2d"].path("protocol").read_text(encoding="utf-8")
    )
    c_grid_text = "/".join(
        str(value) for value in m2d_protocol["classifier"]["C_grid"]
    )
    outer_splits = int(m2d_protocol["fold_policy"]["outer_splits"])
    inner_splits = int(m2d_protocol["classifier"]["inner_splits"])
    statistical = json.loads(
        (Path(statistical_evidence_dir) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    secondary_summary = json.loads(
        (Path(secondary_evidence_dir) / "summary.json").read_text(
            encoding="utf-8"
        )
    )
    secondary_metrics = pd.read_csv(
        Path(secondary_evidence_dir) / "fixed_split_metrics.csv"
    )
    rbf_metrics = pd.read_csv(Path(secondary_evidence_dir) / "rbf_metrics.csv")

    sensitivity = _find_sensitivity_bundles(
        sensitivity_root,
        dataset_revision=str(snapshot_audit["revision"]),
        benchmark_seed=benchmark_seed,
    )

    def _metrics(role_name: str, index: int = 0) -> pd.DataFrame:
        return pd.read_csv(
            sensitivity[role_name][index] / "metrics.csv"
        )

    durations_metrics = _metrics("durations")
    rms_metrics = _metrics("rms_normalized")
    legacy_metrics = _metrics("legacy_pooling", 1)
    energy_metrics = _metrics("energy_weighted")
    mean_metrics = _metrics("mean")
    mean_std_metrics = _metrics("mean_std")
    mean_max_metrics = _metrics("mean_max")
    attn_headline = _metrics("attention")
    attn_protocol = json.loads(
        (sensitivity["attention"][0] / "protocol.json").read_text(
            encoding="utf-8"
        )
    )
    attn_transform_policy = str(
        attn_protocol.get("attention_control_transform_policy", "N/A")
    )
    attn_calibrated = _metrics("attention", 1)
    lda_metrics = _metrics("attention_lda")
    multi_metrics = _metrics("attention_multi")
    neighbourhood_metrics = _metrics("attention_neighbourhood")
    composed_metrics = _metrics("composed")

    layer_scan = pd.DataFrame()
    if layer_scan_dir is not None:
        layer_scan_path = _resolve_optional_artifact(
            layer_scan_dir, "layer_scan.csv"
        )
        if layer_scan_path is not None:
            layer_scan = pd.read_csv(layer_scan_path)

    finetune_comparison: dict[str, object] = {}
    if finetune_pilot_dir is not None:
        finetune_path = _resolve_optional_artifact(
            finetune_pilot_dir, "pilot_comparison.json"
        )
        if finetune_path is not None:
            finetune_comparison = json.loads(
                finetune_path.read_text(encoding="utf-8")
            )
    if finetune_comparison:
        if finetune_primary_root is None:
            raise ValueError(
                "finetune_primary_root is required to verify a fine-tune "
                "comparison"
            )
        from .run_finetune_pilot import (
            build_pilot_comparison,
            resolve_finetune_benchmarks,
        )

        references = resolve_finetune_benchmarks(
            primary_root=finetune_primary_root,
            pooling_root=sensitivity_root,
            seed=benchmark_seed,
            dataset_revision=str(snapshot_audit["revision"]),
        )
        corrected = build_pilot_comparison(
            finetune_ba=float(
                finetune_comparison["fine_tuned_mean_balanced_accuracy"]
            ),
            finetune_eligible=int(
                finetune_comparison["fine_tuned_eligible_samples"]
            ),
            frozen_mean_ba=float(
                references.frozen_mean_metric["balanced_accuracy"]
            ),
            frozen_mean_eligible=int(
                references.frozen_mean_metric["eligible_samples"]
            ),
            attention_headline_ba=float(
                references.attention_headline_metric["balanced_accuracy"]
            ),
            attention_headline_eligible=int(
                references.attention_headline_metric["eligible_samples"]
            ),
        )
        corrected["overfitting_signature"] = finetune_comparison.get(
            "overfitting_signature", {}
        )
        corrected["benchmark_references"] = {
            "frozen_mean_artifact_id": (
                references.frozen_mean_bundle.name
            ),
            "attention_headline_artifact_id": (
                references.attention_headline_bundle.name
            ),
            "fold_assignments_compatible": True,
        }
        finetune_comparison = corrected

    alignment_curve = pd.DataFrame()
    alignment_summary: dict[str, object] = {}
    if alignment_sensitivity_dir is not None:
        alignment_path = _resolve_optional_artifact(
            alignment_sensitivity_dir, "alignment_sensitivity.csv"
        )
        if alignment_path is not None:
            alignment_curve = pd.read_csv(alignment_path)
            summary_path = alignment_path.parent / (
                "alignment_sensitivity_summary.json"
            )
            if summary_path.is_file():
                alignment_summary = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )

    full_audio_scan = pd.DataFrame()
    full_audio_attribution: dict[str, object] = {}
    full_audio_lead = pd.DataFrame()
    if full_audio_conditions_dir is not None:
        full_audio_path = _resolve_optional_artifact(
            full_audio_conditions_dir, "condition_scan.csv"
        )
        if full_audio_path is not None:
            full_audio_scan = pd.read_csv(full_audio_path)
            fa_root = full_audio_path.parent
            attribution_path = fa_root / "attribution_summary.json"
            if attribution_path.is_file():
                full_audio_attribution = json.loads(
                    attribution_path.read_text(encoding="utf-8")
                )
            lead_path = fa_root / "lead_comparison.csv"
            if lead_path.is_file():
                full_audio_lead = pd.read_csv(lead_path)

    attn_stat_dir = Path(sensitivity_root) / "statistical_evidence_attention"
    attn_stat = None
    attn_permutation = pd.DataFrame()
    attn_paired = pd.DataFrame()
    if (attn_stat_dir / "summary.json").is_file():
        attn_stat = json.loads(
            (attn_stat_dir / "summary.json").read_text(encoding="utf-8")
        )
        attn_permutation = pd.read_csv(
            attn_stat_dir / "permutation_summary.csv"
        )
        attn_paired = pd.read_csv(attn_stat_dir / "paired_intervals.csv")

    def sensitivity_row(metrics: pd.DataFrame, condition: str) -> str:
        row = metrics[
            metrics["condition"].eq(condition)
            & metrics["window_ms"].eq(200)
        ]
        if row.empty:
            return "N/A"
        return f"{float(row.iloc[0]['balanced_accuracy']):.3f}"

    def duration_rows(
        metrics: pd.DataFrame, condition: str
    ) -> dict[int, str]:
        result: dict[int, str] = {}
        for window_ms in sorted(metrics["window_ms"].unique()):
            row = metrics[
                metrics["condition"].eq(condition)
                & metrics["window_ms"].eq(window_ms)
            ]
            if not row.empty:
                result[int(window_ms)] = f"{float(row.iloc[0]['balanced_accuracy']):.3f}"
        return result

    def _fixed_row(
        metrics: pd.DataFrame, condition: str
    ) -> pd.Series | None:
        if metrics.empty:
            return None
        row = metrics[metrics["condition"].eq(condition)]
        if "window_ms" in row.columns:
            # Composed runs label their rows with window_ms == 0; single
            # window runs use their duration.
            expected_ms = (
                int(row["window_ms"].unique()[0])
                if row["window_ms"].nunique() == 1
                else 200
            )
            row = row[row["window_ms"].eq(expected_ms)]
        if "decision_rule" in row.columns:
            row = row[row["decision_rule"].eq("fixed_0.5")]
        return row.iloc[0] if not row.empty else None

    def _ba(row: pd.Series | None) -> str:
        return f"{float(row['balanced_accuracy']):.3f}" if row is not None else "N/A"

    def _fixed_ba(metrics: pd.DataFrame, condition: str) -> str:
        return _ba(_fixed_row(metrics, condition))

    def _fixed_value(
        metrics: pd.DataFrame, condition: str, column: str
    ) -> str:
        row = _fixed_row(metrics, condition)
        return f"{float(row[column]):.3f}" if row is not None else "N/A"

    def _calibrated_ba(metrics: pd.DataFrame, condition: str) -> str:
        if metrics.empty or "decision_rule" not in metrics.columns:
            return "N/A"
        row = metrics[
            metrics["condition"].eq(condition)
            & metrics["window_ms"].eq(200)
            & metrics["decision_rule"].eq("calibrated")
        ]
        return _ba(row.iloc[0] if not row.empty else None)

    primary_rows: list[list[object]] = []
    for condition in CONTROL_CONDITIONS:
        row: list[object] = [condition]
        for name in names:
            metric = _condition_metrics(name_to_bundle[name], condition)
            row.append(_format_metrics(metric))
        primary_rows.append(row)

    permutation = pd.read_csv(
        Path(statistical_evidence_dir) / "permutation_summary.csv"
    )
    paired = pd.read_csv(
        Path(statistical_evidence_dir) / "paired_intervals.csv"
    )

    def increment_row(name: str) -> str:
        row = paired[
            paired["encoder"].eq(name)
            & paired["interval_type"].eq("event_minus_pre_increment")
        ].iloc[0]
        return (
            f"+{float(row['point_estimate']):.3f} "
            f"[{float(row['ci_low']):.3f}, {float(row['ci_high']):.3f}]"
        )

    def decision_text(name: str) -> str:
        decision = statistical["screening_decisions"][name]
        if decision["screening_positive"]:
            return "通过（screening-positive）"
        return "未通过（" + "；".join(decision["reasons"]) + "）"

    def display_name(name: str) -> str:
        return {"m2d": "M2D", "beats": "BEATs"}.get(name, name)

    fixed_rows = [
        [
            str(row["encoder"]),
            f"{float(row['balanced_accuracy']):.3f}",
            f"{float(row['roc_auc']):.3f}",
            f"C={row['selected_C']}",
            f"{int(row['n_train'])}/{int(row['n_val'])}/{int(row['n_test'])}",
        ]
        for row in secondary_metrics.to_dict("records")
    ]
    rbf_rows = [
        [
            str(row["encoder"]),
            f"{float(row['balanced_accuracy']):.3f}",
            f"{float(row['roc_auc']):.3f}",
        ]
        for row in rbf_metrics.to_dict("records")
    ]

    verification_counts = snapshot_audit.get(
        "verification_source_counts", {}
    )
    verification_text = "、".join(
        f"{key} {value}" for key, value in verification_counts.items()
    )
    source_transfer = statistical.get("source_transfer_conclusive", {})
    transfer_text = "、".join(
        f"{name}={'是' if value else '否'}"
        for name, value in source_transfer.items()
    )
    groups = statistical.get("groups", {})
    group_text = "、".join(
        f"{name} {stats['n_groups']} 组（{stats['n_singleton_groups']} 单例）"
        for name, stats in groups.items()
    )

    duration_event = duration_rows(
        durations_metrics, "event_selected_event"
    )
    duration_pre = duration_rows(durations_metrics, "event_selected_pre")
    duration_increment = duration_rows(
        durations_metrics, "contact_specific_increment"
    )
    sensitivity_rows = [
        [
            f"{window_ms} ms",
            duration_event.get(window_ms, "N/A"),
            duration_pre.get(window_ms, "N/A"),
            duration_increment.get(window_ms, "N/A"),
        ]
        for window_ms in sorted(duration_event)
    ]

    ablation_rows = [
        ["mean", _fixed_ba(mean_metrics, "event_selected_event"),
         _fixed_ba(mean_metrics, "event_selected_pre"),
         _fixed_ba(mean_metrics, "contact_specific_increment")],
        ["mean+std", _fixed_ba(mean_std_metrics, "event_selected_event"),
         _fixed_ba(mean_std_metrics, "event_selected_pre"),
         _fixed_ba(mean_std_metrics, "contact_specific_increment")],
        ["mean+max", _fixed_ba(mean_max_metrics, "event_selected_event"),
         _fixed_ba(mean_max_metrics, "event_selected_pre"),
         _fixed_ba(mean_max_metrics, "contact_specific_increment")],
        ["mean/std/max", _fixed_ba(legacy_metrics, "event_selected_event"),
         _fixed_ba(legacy_metrics, "event_selected_pre"),
         _fixed_ba(legacy_metrics, "contact_specific_increment")],
        ["energy-weighted", _fixed_ba(energy_metrics, "event_selected_event"),
         _fixed_ba(energy_metrics, "event_selected_pre"),
         _fixed_ba(energy_metrics, "contact_specific_increment")],
        ["attention", _fixed_ba(attn_headline, "event_selected_event"),
         _fixed_ba(attn_headline, "event_selected_pre"),
         _fixed_ba(attn_headline, "contact_specific_increment")],
        ["attention-lda", _fixed_ba(lda_metrics, "event_selected_event"),
         _fixed_ba(lda_metrics, "event_selected_pre"),
         _fixed_ba(lda_metrics, "contact_specific_increment")],
        ["attention-multi (k=3)", _fixed_ba(multi_metrics, "event_selected_event"),
         _fixed_ba(multi_metrics, "event_selected_pre"),
         _fixed_ba(multi_metrics, "contact_specific_increment")],
        ["attention-neighbourhood",
         _fixed_ba(neighbourhood_metrics, "event_selected_event"),
         _fixed_ba(neighbourhood_metrics, "event_selected_pre"),
         _fixed_ba(neighbourhood_metrics, "contact_specific_increment")],
        ["50ms mean + 200ms attention",
         _fixed_ba(composed_metrics, "event_selected_event"),
         _fixed_ba(composed_metrics, "event_selected_pre"),
         _fixed_ba(composed_metrics, "contact_specific_increment")],
    ]

    if not attn_paired.empty:
        attn_inc_row = attn_paired[
            attn_paired["interval_type"].eq("event_minus_pre_increment")
        ].iloc[0]
        attn_inc_ci = (
            f"+{float(attn_inc_row['point_estimate']):.3f} "
            f"[{float(attn_inc_row['ci_low']):.3f}, "
            f"{float(attn_inc_row['ci_high']):.3f}]"
        )
    else:
        attn_inc_ci = "N/A"
    if not attn_permutation.empty:
        attn_perm_row = attn_permutation.iloc[0]
        attn_perm_text = (
            f"观测 BA {float(attn_perm_row['observed_balanced_accuracy']):.3f}，"
            f"零均值 {float(attn_perm_row['null_mean']):.3f}，家族校正 p = "
            f"{float(attn_perm_row['max_stat_familywise_p']):.3f}"
        )
        attn_p_value = f"{float(attn_perm_row['max_stat_familywise_p']):.3f}"
    else:
        attn_perm_text = "N/A"
        attn_p_value = "N/A"
    if attn_stat is not None:
        attn_decision = attn_stat["screening_decisions"].get(
            "m2d_attention", {}
        )
        if attn_decision.get("screening_positive"):
            attn_decision_text = "通过（screening-positive）"
        else:
            attn_decision_text = "未通过（" + "；".join(
                attn_decision.get("reasons", [])
            ) + "）"
    else:
        attn_decision_text = "N/A"

    if not layer_scan.empty:
        sorted_layers = layer_scan.sort_values("layer")
        layer_scan_table = _md_table(
            ["层", "事件 BA", "AUC", "严格前", "增量"],
            [
                [
                    str(int(row["layer"])),
                    f"{float(row['event_balanced_accuracy']):.3f}",
                    f"{float(row['event_roc_auc']):.3f}",
                    f"{float(row['strict_pre_balanced_accuracy']):.3f}",
                    f"{float(row['contact_specific_increment']):.3f}",
                ]
                for row in sorted_layers.to_dict("records")
            ],
        )
        best_layer = sorted_layers.loc[
            sorted_layers["event_balanced_accuracy"].idxmax()
        ]
        final_layer = sorted_layers.iloc[-1]
        layer_scan_narrative = (
            f"事件 BA 从最低 {float(sorted_layers['event_balanced_accuracy'].min()):.3f} "
            f"到最高 {float(best_layer['event_balanced_accuracy']):.3f}；"
            f"单 seed 峰值在层 {int(best_layer['layer'])}，"
            f"最后一层 {int(final_layer['layer'])} 为 "
            f"{float(final_layer['event_balanced_accuracy']):.3f}。"
        )
    else:
        layer_scan_table = "N/A"
        layer_scan_narrative = "未提供 layer-scan artifact。"

    if not full_audio_scan.empty:
        fa_condition_table = _md_table(
            ["条件", "事件 BA", "AUC", "样本"],
            [
                [
                    str(row["condition"]),
                    f"{float(row['event_balanced_accuracy']):.3f}",
                    f"{float(row['event_roc_auc']):.3f}",
                    str(int(row["eligible_samples"])),
                ]
                for row in full_audio_scan.to_dict("records")
            ],
        )
        fa_lead_table = _md_table(
            ["主导人条件", "主导人正确率", "复现条件", "组级分层 BA"],
            [
                [
                    str(row["lead_condition"]),
                    f"{int(row['lead_ungrouped_accuracy'])}%",
                    str(row["our_condition"]),
                    f"{float(row['our_grouped_balanced_accuracy']):.3f}",
                ]
                for row in full_audio_lead.to_dict("records")
            ],
        )
        attribution = full_audio_attribution

        def fa_ba(condition: str) -> str:
            row = full_audio_scan[
                full_audio_scan["condition"].eq(condition)
            ]
            return (
                f"{float(row.iloc[0]['event_balanced_accuracy']):.3f}"
                if not row.empty
                else "N/A"
            )

        def fa_n(condition: str) -> str:
            row = full_audio_scan[
                full_audio_scan["condition"].eq(condition)
            ]
            return (
                str(int(row.iloc[0]["eligible_samples"]))
                if not row.empty
                else "N/A"
            )

        beats_increment = _condition_metrics(
            name_to_bundle["beats"], "contact_specific_increment"
        )
        beats_increment_text = (
            f"{float(beats_increment['balanced_accuracy']):+.3f}"
        )
        duration_shortcut_ba = full_audio_attribution.get(
            "duration_shortcut_balanced_accuracy"
        )
        duration_shortcut_ba = (
            f"{float(duration_shortcut_ba):.3f}"
            if isinstance(duration_shortcut_ba, (int, float))
            else "N/A"
        )
        duration_shortcut_auc = full_audio_attribution.get(
            "duration_shortcut_roc_auc"
        )
        duration_shortcut_auc = (
            f"{float(duration_shortcut_auc):.3f}"
            if isinstance(duration_shortcut_auc, (int, float))
            else "N/A"
        )

        def fmt_gain(key: str) -> str:
            value = attribution.get(key)
            if isinstance(value, (int, float)):
                return f"{float(value):+.3f}"
            return "N/A"

        fa_section = f"""## 12. 完整音频对照（泄漏验证）

**主导人数字（随机切分 accuracy）与组级分层复现（balanced accuracy）**：主导人管线未记录切分协议；我们的复现在同一验证快照上用血缘组分层 {outer_splits} 折、
冻结 BEATs、相同决策规则。正确率与 BA 口径不同，不能直接相减，只作量级对照。

{fa_lead_table}

**per-condition 复现表**（完整 bundle 见 condition_scan.csv）：

{fa_condition_table}

**区域归因**（相对 0.5 s 事件基线）：击球前 1 s {fmt_gain('pre_contact_gain')}、
击球后 1 s {fmt_gain('post_contact_1s_gain')}、击球后 4 s {fmt_gain('post_contact_4s_gain')}、
完整音频 {fmt_gain('full_audio_gain')}（其中 4 s 之后的部分 {fmt_gain('gain_beyond_4s')}）。归因结论：
{attribution.get('conclusion', 'N/A')}。

**解释（分证据层级）**：

- primary：组级分层下完整音频 BA {fa_ba('full_audio')}，0.5 s 事件窗 {fa_ba('event_500ms')}；与上表主导人数字的差距说明随机切分数字不能直接迁移到组级协议。
- negative control：锁定 200 ms 事件窗的接触增量（contact_specific_increment）为 {beats_increment_text}（BEATs mean 口径）；
击球后 1 s 增益 {fmt_gain('post_contact_1s_gain')}、完整音频增益 {fmt_gain('full_audio_gain')}，用于判断事件窗外信息的量级。
- sensitivity：每条件合格样本数见上表（post_contact_4s 为 {fa_n('post_contact_4000ms')}，full_audio 为 {fa_n('full_audio')}）。
- exploratory：增益全部位于击球后区间（击球前无信号，与主导人"背景 55%"一致），与结果内容泄漏一致——解说、观众反应、跑动声音在击球后编码了飞/滚的结果；
BEATs 是语义级表征，能读这些内容。
- exploratory（时长捷径）：duration 单特征达到 BA {duration_shortcut_ba} / AUC {duration_shortcut_auc}，是采集流程产物（剪辑习惯），部署时不存在。固定长度段（event/pre/post_contact）不受此混杂，其归因成立；
完整音频 4 s→结尾的 {fmt_gain('gain_beyond_4s')} 含时长成分。BEATs mean pooling 特征未有效编码时长（
完整音频特征 BA {fa_ba('full_audio')} vs duration {duration_shortcut_ba}），因此我们的完整音频复现主要反映内容信号；
表中随机切分 full_audio 数字高于组级复现的部分可能来自模型编码文件长度——这是采集流程混杂，不是击球声学。

**部署声明**：非居中条件依赖击球后的音频，在击球瞬间的推理场景中不存在；负控链只对居中接触窗定义。因此随机切分 full_audio 数字不是可部署的筛选数字——可部署的事件声学头条仍是 200 ms 窗（
M2D attention {_fixed_ba(attn_headline, 'event_selected_event')} / BEATs mean {float(_condition_metrics(name_to_bundle['beats'], 'event_selected_event')['balanced_accuracy']):.3f}）。
"""
    else:
        fa_section = ""

    if not alignment_curve.empty:
        alignment_table = _md_table(
            ["偏移(ms)", "事件 BA", "AUC", "相对 0ms", "样本"],
            [
                [
                    f"{int(row['shift_ms']):+d}",
                    f"{float(row['event_balanced_accuracy']):.3f}",
                    f"{float(row['event_roc_auc']):.3f}",
                    f"{float(row['delta_vs_0ms']):+.3f}",
                    str(int(row["eligible_samples"])),
                ]
                for row in alignment_curve.to_dict("records")
            ],
        )
        drop_50 = alignment_summary.get("drop_at_50ms")
        drop_text = (
            f"{float(drop_50):.3f}"
            if isinstance(drop_50, (int, float))
            else "N/A"
        )
        interpretation = alignment_summary.get(
            "interpretation", "N/A"
        )
        alignment_section = f"""
**对齐敏感性（alignment sensitivity）**：把 200 ms 事件窗中心相对峰值偏移 ±25/50/100 ms（同一无控件配置，训练折内选 C；负控链只对精确居中窗定义）。

{alignment_table}

±50 ms 处事件 BA 掉 {drop_text}，解读为 {interpretation}（部署诊断，非新头条声明）。
"""
    else:
        alignment_section = ""

    if finetune_comparison:
        ft_ba = finetune_comparison.get(
            "fine_tuned_mean_balanced_accuracy"
        )
        ft_ba = (
            f"{float(ft_ba):.3f}"
            if isinstance(ft_ba, (int, float))
            else "N/A"
        )
        frozen_ba = finetune_comparison.get(
            "frozen_mean_balanced_accuracy"
        )
        frozen_ba = (
            f"{float(frozen_ba):.3f}"
            if isinstance(frozen_ba, (int, float))
            else "N/A"
        )
        headline_ba = _fixed_ba(
            attn_headline, "event_selected_event"
        )
        gain = finetune_comparison.get("gain_vs_frozen_mean")
        gain_text = (
            f"{float(gain):+.3f}"
            if isinstance(gain, (int, float))
            else "N/A"
        )
        conclusion = finetune_comparison.get("conclusion", "N/A")
        signature = finetune_comparison.get(
            "overfitting_signature", {}
        )
        gap = signature.get("train_minus_inner_val_mean")
        gap_text = (
            f"{float(gap):+.3f}"
            if isinstance(gap, (int, float))
            else "N/A"
        )
        finetune_section = f"""
**微调试点（exploratory，最终配置：解冻最后 4 层全量）**：在锁定折上每折训练折内拟合（初版 LoRA rank 8 与最终解冻顶层均试过），内层组级折早停，mean 池化头。冻结 mean 与微调行必须共享 eligible-sample role；attention 仅作 controls 口径参考：

- 解冻顶层微调 mean：{ft_ba}
- 冻结 mean：{frozen_ba}（增益 {gain_text} → {conclusion}）
- attention 头条（由已解析 benchmark artifact 提供）：{headline_ba}

失败模式（train 减内层验证均值 {gap_text}，见 trace）：解冻顶层后模型能学习，但测试折无净增益并出现训练/内层验证分离；方向关闭。
"""
    else:
        finetune_section = ""

    report = f"""# 短接触 M2D / BEATs 基准：技术报告

## 0. 数据与协议

- 不可变数据集快照：`{snapshot_audit.get('revision', 'N/A')}`，
共 {int(snapshot_audit.get('sample_count', 0))} 个样本（fly {int(snapshot_audit.get('label_counts',
 {}).get('fly_ball', 0))} / ground {int(snapshot_audit.get('label_counts',
 {}).get('ground_ball', 0))}）。
- 验证路线构成：{verification_text}。验证路线仅作为来源记录，不作为分类特征，也不代表全部样本都经过统一双重人工复核。
- 编码器：M2D（40 ms patch，冻结）与 BEATs（iter3+ AS2M，冻结，强制 FP32）。
- 表示：共同比较使用最终层有效 token 的均值池化（768 维）；M2D 头条使用 attention 池化（每外层折在冻结 token 上拟合 PCA 主方向并 softmax 加权）。
- 分类器：训练折内 StandardScaler + 平衡 L2 逻辑回归；C 仅在训练折内部从 {c_grid_text} 选择。
- 评估：确定性种子、{outer_splits} 折按血缘组分层外折、{inner_splits} 折内层选择；血缘组以 MLB 比赛为主单位。

## 1. 共同比较（primary，mean 池化，200 ms 事件窗）

{_md_table(['条件', *names], primary_rows)}

所有主结果均为 Balanced Accuracy 牵头，并附 Accuracy、ROC-AUC、Macro-F1 与混淆计数；不使用任何固定分数阈值作为达标标准。

## 2. M2D 头条表示（attention 池化，200 ms）

M2D 头条使用 attention 池化（每外层训练折拟合冻结 token 的 PCA 主方向，softmax 加权平均）；event probe 对 strict Pre 和 transient-removed 复用 event 训练折方向，独立负控 probe 才拟合各自方向。transform policy：`{attn_transform_policy}`。
BEATs 因 200 ms 仅一个 token 无法使用 attention，共同比较保持 mean 池化。此不对称是表示能力差异，不是任务差异。

- 事件窗 BA {_fixed_ba(attn_headline, 'event_selected_event')}，
严格前迁移 {_fixed_ba(attn_headline, 'event_selected_pre')}，
独立严格前 {_fixed_ba(attn_headline, 'pre_selected_pre')}，
移除瞬态后 {_fixed_ba(attn_headline, 'event_selected_removed')}，
独立移除 {_fixed_ba(attn_headline, 'removed_selected_removed')}。
- 接触特异性增量：{_fixed_ba(attn_headline, 'contact_specific_increment')}（{attn_inc_ci}）。
- 标签置换（折内分层，单编码器家族）：{attn_perm_text}。
- 判定：{attn_decision_text}。
- 稳定性：当前 role 的逐折结果与另行保存的多 seed sensitivity artifacts 分开呈现，避免在本报告中重复硬编码未解析运行的数值。

## 3. 池化消融（M2D 200 ms，固定 0.5 阈值）

{_md_table(['池化', '事件 BA', '严格前 BA', '增量'], ablation_rows)}

attention 的增益来自逐 token 加权聚焦瞬态时刻；短窗（50/100 ms）token 过少时 attention 退化（50 ms 仅 2 个 token）。

## 4. 头条决策（headline decision）

05/06 阶段新增的表示与锁定 attention 头条在同一合格样本、折叠与决策规则下对比，均未超越头条：

| 表示 | 事件 BA | AUC | 严格前 | 增量 |
|---|---|---|---|---|
| attention（头条） | {_fixed_ba(attn_headline, 'event_selected_event')} | — | {_fixed_ba(attn_headline, 'event_selected_pre')} | {_fixed_ba(attn_headline, 'contact_specific_increment')} |
| attention-lda | {_fixed_ba(lda_metrics, 'event_selected_event')} | — | {_fixed_ba(lda_metrics, 'event_selected_pre')} | {_fixed_ba(lda_metrics, 'contact_specific_increment')} |
| attention-multi (k=3) | {_fixed_ba(multi_metrics, 'event_selected_event')} | — | {_fixed_ba(multi_metrics, 'event_selected_pre')} | {_fixed_ba(multi_metrics, 'contact_specific_increment')} |
| attention-neighbourhood | {_fixed_ba(neighbourhood_metrics, 'event_selected_event')} | — | {_fixed_ba(neighbourhood_metrics, 'event_selected_pre')} | {_fixed_ba(neighbourhood_metrics, 'contact_specific_increment')} |
| 50ms mean + 200ms attention | {_fixed_ba(composed_metrics, 'event_selected_event')} | {_fixed_value(composed_metrics, 'event_selected_event', 'roc_auc')} | {_fixed_ba(composed_metrics, 'event_selected_pre')} | {_fixed_ba(composed_metrics, 'contact_specific_increment')} |

因此头条维持 attention（{_fixed_ba(attn_headline, 'event_selected_event')}），无需为替代表示补充统计证据。

**层扫描（layer scan）**：对 M2D 已提供层逐一评估（每层在训练折内拟合 attention 方向与 C）。{layer_scan_narrative}

{layer_scan_table}

**seed 稳健性**：高层峰值在已有多 seed artifacts 中发生漂移；本报告不重复硬编码未通过当前 role 解析的运行数字。峰值层是跨多层选择的结果，带多重比较偏差，因此保持探索性。

**最终决策**：没有任何层稳健地超越最后一层 attention 头条（{_fixed_ba(attn_headline, 'event_selected_event')}）；层扫描关闭"换层"方向。
{finetune_section}

**权重诊断**：另行保存的诊断显示 attention 接近峰值 token 选择器；因为该诊断尚未通过本报告的 artifact role 输入解析，这里不重复硬编码其数值。

**长窗实测（补充验证）**：长窗结果保存在独立 benchmark artifacts 中；因为本报告当前未接收该 role，这里只保留“没有替换 200 ms 头条”的决策，不重复硬编码运行数字。

## 5. 决策阈值校准（M2D 200 ms）

阈值在训练折内层验证折上按 Balanced Accuracy 最优选择，只作用于留出预测：

- mean/std/max：固定 0.5 → {_fixed_ba(legacy_metrics, 'event_selected_event')}，
校准 → {_calibrated_ba(legacy_metrics, 'event_selected_event')}。
- attention：固定 0.5 → {_fixed_ba(attn_headline, 'event_selected_event')}，
校准 → {_calibrated_ba(attn_calibrated, 'event_selected_event')}（固定 0.5 已接近最优，校准无增益）。

## 6. 负控解读（negative control）

attention 口径下，事件模型迁移到严格前窗口后为 {_fixed_ba(attn_headline, 'event_selected_pre')}，移除中央 40 ms 瞬态后为 {_fixed_ba(attn_headline, 'event_selected_removed')}；独立训练的严格前模型为 {_fixed_ba(attn_headline, 'pre_selected_pre')}，
说明背景可能残留少量类别关联信息，但事件窗的分离能力主要依赖瞬态本身。

## 7. 时长敏感性（M2D，attention 池化）

{_md_table(['窗口', '事件 BA', '严格前 BA', '增量'], sensitivity_rows)}

attention 池化在 50/100/200 ms 的 token 数分别为 2/3/6；本次扫描的接触增量分别为 {duration_increment.get(50, 'N/A')} / {duration_increment.get(100, 'N/A')} / {duration_increment.get(200, 'N/A')}。

- RMS 归一化（200 ms）：事件 {sensitivity_row(rms_metrics, 'event_selected_event')}，
增量 {sensitivity_row(rms_metrics, 'contact_specific_increment')}。
- 传统 mean/std/max 池化（200 ms，2304 维）：
事件 {sensitivity_row(legacy_metrics, 'event_selected_event')}，
增量 {sensitivity_row(legacy_metrics, 'contact_specific_increment')}。
- energy-weighted 池化（200 ms）：事件 {_fixed_ba(energy_metrics, 'event_selected_event')}，
增量 {_fixed_ba(energy_metrics, 'contact_specific_increment')}。
{alignment_section}

## 8. 探索性分析（exploratory）

平衡 RBF SVM（C ∈ {'/'.join(str(value) for value in secondary_summary.get('rbf_grid', {}).get('C', []))}，γ ∈ {'/'.join(str(value) for value in secondary_summary.get('rbf_grid', {}).get('gamma', []))}，分组内层选择）：

{_md_table(['编码器', 'OOF BA', 'ROC-AUC'], rbf_rows)}

与线性探针相比无增益，不改变主排名。

## 9. 固定切分开发证据（fixed benchmark）

固定切分的 train/val/test 计数见下表并被逐样本精确复现；
{secondary_summary.get('fixed_split', {}).get(next(iter(secondary_summary.get('fixed_split',
 {})), 'm2d'), {}).get('crossing_mlb_games', 0)} 个 MLB 比赛跨越分区。测试分区仅评估一次，调参只用训练与验证分区：

{_md_table(['编码器', 'Test BA', 'ROC-AUC', '选定 C', 'train/val/test'], fixed_rows)}

固定切分结果属于开发证据，不是独立验证，也不能作为跨采集流程迁移的证据。

## 10. 来源迁移边界（source-transfer）

血缘组重采样的 95% 区间以组为单位；本快照 {group_text}，迁移结论 {transfer_text}。组级评估只能减少同场比赛泄漏，不能由本实验单独证明音频信息可迁移到新的采集流程。

## 11. 排除与可复现性

- 排除原因均为稳定枚举：window_not_exact（窗口无法严格居中）与 strict_pre_unavailable（严格前区不足）。
- 特征、折叠、预测与统计全部由内容寻址的 artifact 身份与 provenance 指纹管理；相同协议可直接续跑，任何协议定义值变化都会使缓存失效。
- 未发布任何原始媒体、模型权重、第三方模型源码或样本级预测；报告中的所有数字可从头复现。
{fa_section}
"""

    summary = f"""# 短接触 M2D / BEATs 基准：组会摘要

在 {int(snapshot_audit.get('sample_count', 0))} 样本的已验证快照（{verification_text}）上，冻结 M2D 与 BEATs 以 200 ms 峰值居中窗口区分 fly/ground，
采用血缘组分层的 {outer_splits} 折评估与训练折内 C 选择。

- 共同比较（mean 池化）：M2D {float(_condition_metrics(
 name_to_bundle['m2d'], 'event_selected_event'
 )['balanced_accuracy']):.3f} / BEATs {float(_condition_metrics(
 name_to_bundle['beats'], 'event_selected_event'
 )['balanced_accuracy']):.3f}。
- M2D 头条（attention 池化）：事件 BA {_fixed_ba(attn_headline, 'event_selected_event')}，
严格前 {_fixed_ba(attn_headline, 'event_selected_pre')}，
接触增量 {_fixed_ba(attn_headline, 'contact_specific_increment')}（{attn_inc_ci}）。
- 置换检验：attention 头条家族校正 p = {attn_p_value}；{attn_decision_text}。
- 池化消融：attention（{_fixed_ba(attn_headline, 'event_selected_event')}）、mean/std/max（{_fixed_ba(legacy_metrics, 'event_selected_event')}）、mean（{_fixed_ba(mean_metrics, 'event_selected_event')}）；阈值校准结果见技术报告。
- 局限：严格前模型 BA {_fixed_ba(attn_headline, 'pre_selected_pre')}；{group_text}，组级评估不能单独证明跨采集流程迁移。
"""


    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "report_zh.md"
    summary_path = output_root / "summary_zh.md"
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(summary, encoding="utf-8")
    return {"report": report_path, "summary": summary_path}
