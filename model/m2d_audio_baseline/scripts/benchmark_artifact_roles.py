from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


M2D_ENCODER_NAME = "m2d_vit_base_80x200p16x4_40ms"
BEATS_ENCODER_NAME = "beats_iter3plus_as2m"
VERIFIED_DATASET_REVISION = "4b6ed0e1cea1425121b075212ddb49b820e27cda"
ATTENTION_CONTROL_TRANSFORM_POLICY = "event_fitted_transfer_v1"


class BenchmarkArtifactRoleError(RuntimeError):
    """Raised when a scientific artifact role does not resolve uniquely."""


@dataclass(frozen=True)
class BenchmarkArtifactRole:
    """Complete protocol identity required by one benchmark consumer."""

    name: str
    encoder_name: str
    dataset_revision: str
    pooling: str
    window_conditions: tuple[str, ...]
    normalization: str
    controls_enabled: bool
    fold_seed: int
    lineage_group: str
    classifier_name: str
    c_selection: str
    threshold_calibrated: bool
    fixed_threshold: float
    event_window_shift_ms: int = 0
    attention_control_transform_policy: str | None = None
    feature_composition: tuple[tuple[int, str], ...] | None = None


def short_contact_artifact_role(
    *,
    name: str,
    encoder_name: str,
    dataset_revision: str,
    pooling: str,
    fold_seed: int,
    window_conditions: tuple[str, ...] = ("event_200ms",),
    normalization: str = "snapshot_level",
    controls_enabled: bool = True,
    threshold_calibrated: bool = False,
    c_selection: str = "inner_grouped_cv",
    event_window_shift_ms: int = 0,
    attention_control_transform_policy: str | None = None,
    feature_composition: tuple[tuple[int, str], ...] | None = None,
) -> BenchmarkArtifactRole:
    """Describe one locked short-contact logistic benchmark role."""

    return BenchmarkArtifactRole(
        name=name,
        encoder_name=encoder_name,
        dataset_revision=dataset_revision,
        pooling=pooling,
        window_conditions=window_conditions,
        normalization=normalization,
        controls_enabled=controls_enabled,
        fold_seed=fold_seed,
        lineage_group="lineage_group_id",
        classifier_name="balanced_l2_logistic_regression",
        c_selection=c_selection,
        threshold_calibrated=threshold_calibrated,
        fixed_threshold=0.5,
        event_window_shift_ms=event_window_shift_ms,
        attention_control_transform_policy=(
            attention_control_transform_policy
        ),
        feature_composition=feature_composition,
    )


def _feature_composition(protocol: dict[str, object]) -> (
    tuple[tuple[int, str], ...] | None
):
    raw = protocol.get("feature_composition")
    if raw is None:
        return None
    if not isinstance(raw, list):
        return ()
    result: list[tuple[int, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return ()
        try:
            result.append((int(item["window_ms"]), str(item["pooling"])))
        except (KeyError, TypeError, ValueError):
            return ()
    return tuple(result)


def _matches(protocol: dict[str, object], role: BenchmarkArtifactRole) -> bool:
    encoders = protocol.get("encoders", [])
    if not isinstance(encoders, list) or len(encoders) != 1:
        return False
    encoder = encoders[0]
    if not isinstance(encoder, dict):
        return False
    dataset = protocol.get("dataset", {})
    controls = protocol.get("controls", {})
    fold_policy = protocol.get("fold_policy", {})
    classifier = protocol.get("classifier", {})
    threshold = protocol.get("decision_threshold", {})
    if not all(
        isinstance(value, dict)
        for value in (dataset, controls, fold_policy, classifier, threshold)
    ):
        return False
    return (
        encoder.get("name") == role.encoder_name
        and dataset.get("revision") == role.dataset_revision
        and protocol.get("pooling") == role.pooling
        and tuple(protocol.get("window_conditions", []))
        == role.window_conditions
        and protocol.get("normalization") == role.normalization
        and bool(controls.get("enabled", False)) == role.controls_enabled
        and fold_policy.get("seed") == role.fold_seed
        and fold_policy.get("group") == role.lineage_group
        and classifier.get("name") == role.classifier_name
        and classifier.get("C_selection") == role.c_selection
        and bool(threshold.get("calibrate", False))
        == role.threshold_calibrated
        and float(threshold.get("fixed_default", 0.5))
        == role.fixed_threshold
        and int(protocol.get("event_window_shift_ms", 0))
        == role.event_window_shift_ms
        and protocol.get("attention_control_transform_policy")
        == role.attention_control_transform_policy
        and _feature_composition(protocol) == role.feature_composition
    )


def resolve_benchmark_bundle(
    root: Path, role: BenchmarkArtifactRole
) -> Path:
    """Resolve exactly one immediate child bundle for a scientific role."""

    root = Path(root)
    matches: list[Path] = []
    for candidate in sorted(root.glob("*")):
        protocol_path = candidate / "protocol.json"
        if not protocol_path.is_file():
            continue
        try:
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(protocol, dict) and _matches(protocol, role):
            matches.append(candidate)
    if not matches:
        raise BenchmarkArtifactRoleError(
            f"{role.name}: no matching artifact under {root}"
        )
    if len(matches) != 1:
        joined = ", ".join(path.name for path in matches)
        raise BenchmarkArtifactRoleError(
            f"{role.name}: {len(matches)} matching artifacts under {root}: "
            f"{joined}"
        )
    return matches[0]
