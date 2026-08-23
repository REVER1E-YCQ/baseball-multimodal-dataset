from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from .benchmark_artifact_roles import ATTENTION_CONTROL_TRANSFORM_POLICY
from .short_contact_benchmark import (
    ATTENTION_POOLINGS,
    ArtifactBundle,
    BenchmarkProtocol,
    _canonical_sha256,
    _evaluate_encoders,
    _file_sha256,
    _uses_attention_controls,
    _validate_protocol,
    _write_json,
)


def _cached_protocol(document: dict[str, object]) -> BenchmarkProtocol:
    classifier = document.get("classifier", {})
    fold_policy = document.get("fold_policy", {})
    controls = document.get("controls", {})
    threshold = document.get("decision_threshold", {})
    if not all(
        isinstance(value, dict)
        for value in (classifier, fold_policy, controls, threshold)
    ):
        raise ValueError("Cached protocol has malformed policy sections")
    non_centered = document.get("non_centered_windows", {})
    if non_centered:
        raise ValueError(
            "Cached attention-control reevaluation does not support "
            "non-centered windows"
        )
    composition_document = document.get("feature_composition")
    feature_composition = None
    if composition_document is not None:
        if not isinstance(composition_document, list):
            raise ValueError("Cached feature composition is malformed")
        feature_composition = tuple(
            (int(item["window_ms"]), str(item["pooling"]))
            for item in composition_document
        )
    pooling = str(document.get("pooling", ""))
    window_conditions = tuple(
        int(str(value).removeprefix("event_").removesuffix("ms"))
        for value in document.get("window_conditions", [])
    )
    c_grid_document = classifier.get("C_grid")
    c_grid = (
        tuple(float(value) for value in c_grid_document)
        if isinstance(c_grid_document, list)
        else None
    )
    layers_document = document.get("layers")
    layers = None
    if isinstance(layers_document, dict):
        indices = layers_document.get("indices")
        if isinstance(indices, list):
            layers = tuple(int(value) for value in indices)
    return BenchmarkProtocol(
        seed=int(fold_policy["seed"]),
        outer_splits=int(fold_policy["outer_splits"]),
        logistic_c=float(classifier.get("C") or 0.01),
        c_grid=c_grid,
        inner_splits=int(classifier.get("inner_splits") or 3),
        window_conditions=window_conditions,
        window_shift_ms=int(document.get("event_window_shift_ms", 0)),
        detector=str(document["detector"]),
        normalization=str(document["normalization"]),
        pooling=pooling,
        classifier=str(classifier["name"]),
        include_controls=bool(controls.get("enabled", False)),
        calibrate_threshold=bool(threshold.get("calibrate", False)),
        attention_k=int(document.get("attention_k", 3)),
        feature_composition=feature_composition,
        layers=layers,
        protocol_version=str(document["protocol_version"]),
    )


def _copy_cached_artifact(source: Path, destination: Path) -> None:
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def reevaluate_cached_attention_controls(
    source_bundle: Path,
    output_dir: Path,
) -> ArtifactBundle:
    """Re-evaluate attention controls from frozen feature artifacts only."""

    source_bundle = Path(source_bundle).resolve()
    source_protocol_path = source_bundle / "protocol.json"
    if not source_protocol_path.is_file():
        raise FileNotFoundError(
            f"Cached source has no protocol.json: {source_bundle}"
        )
    source_document = json.loads(
        source_protocol_path.read_text(encoding="utf-8")
    )
    if not isinstance(source_document, dict):
        raise ValueError("Cached protocol must be a JSON object")
    source_artifact_id = str(source_document.get("artifact_id", ""))
    source_policy = source_document.get(
        "attention_control_transform_policy"
    )
    if source_policy not in (None, ATTENTION_CONTROL_TRANSFORM_POLICY):
        raise ValueError(
            f"Unsupported attention-control transform policy: {source_policy}"
        )
    protocol = _cached_protocol(source_document)
    _validate_protocol(protocol)
    if not protocol.include_controls:
        raise ValueError("Cached source does not contain negative controls")
    if not _uses_attention_controls(protocol):
        raise ValueError(
            "Cached attention-control reevaluation requires an attention "
            "representation"
        )

    encoders = source_document.get("encoders", [])
    if not isinstance(encoders, list) or len(encoders) != 1:
        raise ValueError("Cached reevaluation requires exactly one encoder")
    encoder_document = encoders[0]
    if not isinstance(encoder_document, dict):
        raise ValueError("Cached encoder provenance is malformed")
    encoder_name = str(encoder_document["name"])
    source_feature_paths: dict[str, Path] = {}
    if protocol.feature_composition is None:
        token_paths = sorted(
            (source_bundle / "features").glob("*_tokens.csv")
        )
        if len(token_paths) != 1:
            raise ValueError(
                "Cached reevaluation requires exactly one token feature "
                f"table; found {len(token_paths)}"
            )
        source_feature_paths[encoder_name] = token_paths[0]
    else:
        for milliseconds, component_pooling in protocol.feature_composition:
            key = f"{milliseconds}ms_{component_pooling}"
            suffix = (
                f"__{key}_tokens.csv"
                if component_pooling in ATTENTION_POOLINGS
                else f"__{key}.csv"
            )
            matches = sorted(
                (source_bundle / "features").glob(f"*{suffix}")
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Cached composition component {key} has "
                    f"{len(matches)} feature tables"
                )
            source_feature_paths[key] = matches[0]

    protocol_document = {
        key: value
        for key, value in source_document.items()
        if key != "artifact_id"
    }
    protocol_document["attention_control_transform_policy"] = (
        ATTENTION_CONTROL_TRANSFORM_POLICY
    )
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = Path(output_dir).resolve() / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    artifact_paths: dict[str, Path] = {}
    source_files = {
        "snapshot_audit": "snapshot_audit.json",
        "window_manifest": "windows_manifest.csv",
        "exclusions": "exclusions.csv",
        "fold_assignments": "fold_assignments.csv",
    }
    for artifact_name, filename in source_files.items():
        source_path = source_bundle / filename
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Cached source is missing {filename}: {source_bundle}"
            )
        destination = bundle_root / filename
        _copy_cached_artifact(source_path, destination)
        artifact_paths[artifact_name] = destination

    windows = pd.read_csv(artifact_paths["window_manifest"])
    for raw_relative_path in windows["window_path"]:
        relative_path = Path(str(raw_relative_path))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"Cached window path is not bundle-relative: {relative_path}"
            )
        source_window = source_bundle / relative_path
        if not source_window.is_file():
            raise FileNotFoundError(
                f"Cached source is missing window: {source_window}"
            )
        destination_window = bundle_root / relative_path
        destination_window.parent.mkdir(parents=True, exist_ok=True)
        _copy_cached_artifact(source_window, destination_window)

    features_root = bundle_root / "features"
    features_root.mkdir(parents=True, exist_ok=True)
    feature_paths: dict[str, Path] = {}
    for feature_key, source_feature_path in source_feature_paths.items():
        destination = features_root / source_feature_path.name
        _copy_cached_artifact(source_feature_path, destination)
        feature_paths[feature_key] = destination
        artifact_paths[f"features/{feature_key}"] = destination

    folds = pd.read_csv(artifact_paths["fold_assignments"])
    predictions, metrics, selections = _evaluate_encoders(
        feature_paths, folds, protocol
    )
    for artifact_name, frame, filename in (
        ("oof_predictions", predictions, "oof_predictions.csv"),
        ("metrics", metrics, "metrics.csv"),
        ("selections", selections, "selections.csv"),
    ):
        destination = bundle_root / filename
        frame.to_csv(destination, index=False)
        artifact_paths[artifact_name] = destination

    protocol_path = bundle_root / "protocol.json"
    _write_json(
        protocol_path,
        {"artifact_id": artifact_id, **protocol_document},
    )
    artifact_paths["protocol"] = protocol_path
    provenance_path = bundle_root / "reevaluation_provenance.json"
    _write_json(
        provenance_path,
        {
            "source_artifact_id": source_artifact_id,
            "source_feature_sha256": {
                key: _file_sha256(path)
                for key, path in sorted(source_feature_paths.items())
            },
            "transform_policy": ATTENTION_CONTROL_TRANSFORM_POLICY,
            "encoder_inference_runs": 0,
        },
    )
    artifact_paths["reevaluation_provenance"] = provenance_path

    bundle_manifest_path = bundle_root / "artifact_bundle.json"
    _write_json(
        bundle_manifest_path,
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
    artifact_paths["artifact_bundle"] = bundle_manifest_path
    return ArtifactBundle(
        artifact_id=artifact_id,
        root=bundle_root,
        _artifacts=tuple(sorted(artifact_paths.items())),
    )
