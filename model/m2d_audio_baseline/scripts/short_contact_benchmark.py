from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd
from scipy.io import wavfile
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

from .attention_control_representation import (
    CONTROL_CONDITIONS,
    PRE_ONLY_CONDITIONS,
    AttentionControlRepresentation,
    attention_control_window_roles,
    fit_attention_directions as _attention_fit_directions,
    load_token_table as _load_token_table,
    pool_attention_tokens as _attention_family_pool,
)
from .benchmark_artifact_roles import ATTENTION_CONTROL_TRANSFORM_POLICY
from .prepare_windows import (
    find_peak_time,
    parse_non_centered_spec,
    prepare_windows,
    to_float_mono,
    to_int16,
    validate_non_centered_specs,
)


__all__ = [
    "ArtifactBundle",
    "BenchmarkProtocol",
    "CONTROL_CONDITIONS",
    "DatasetSnapshot",
    "EncoderAdapter",
    "EncoderProvenance",
    "SnapshotSample",
    "run_short_contact_benchmark",
]


LABEL_TO_INT = {"fly_ball": 0, "ground_ball": 1}


@dataclass(frozen=True)
class BenchmarkProtocol:
    """Frozen choices that define one short-contact benchmark run."""

    seed: int = 20260805
    outer_splits: int = 5
    logistic_c: float = 0.01
    c_grid: tuple[float, ...] | None = None
    inner_splits: int = 3
    event_window_ms: int = 200
    window_conditions: tuple[int, ...] | None = None
    window_shift_ms: int = 0
    detector: str = "absolute_amplitude_peak_within_event_interval"
    normalization: str = "snapshot_level"
    pooling: str = "valid_final_layer_token_mean"
    classifier: str = "balanced_l2_logistic_regression"
    include_controls: bool = False
    calibrate_threshold: bool = False
    feature_composition: tuple[tuple[int, str], ...] | None = None
    attention_k: int = 3
    layers: tuple[int, ...] | None = None
    non_centered_windows: tuple[str, ...] = ()
    protocol_version: str = "short-contact-tracer-v1"

    @property
    def resolved_window_conditions(self) -> tuple[int, ...]:
        base = set(self.window_conditions or (self.event_window_ms,))
        if self.feature_composition is not None:
            base.update(milliseconds for milliseconds, _pooling in self.feature_composition)
        return tuple(sorted(base))


@dataclass(frozen=True)
class SnapshotSample:
    uid: str
    label: str
    lineage_group_id: str
    audio_path: Path
    event_start: float
    event_end: float


@dataclass(frozen=True)
class DatasetSnapshot:
    revision: str
    samples: tuple[SnapshotSample, ...]


@dataclass(frozen=True)
class EncoderProvenance:
    name: str
    upstream_revision: str
    checkpoint_sha256: str
    precision: str
    token_dimension: int
    training_epochs: int = 0


class EncoderAdapter(Protocol):
    provenance: EncoderProvenance

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return only valid final-layer tokens as [token, feature]."""


@dataclass(frozen=True)
class ArtifactBundle:
    artifact_id: str
    root: Path
    _artifacts: tuple[tuple[str, Path], ...]

    @property
    def artifact_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._artifacts)

    def path(self, name: str) -> Path:
        for artifact_name, path in self._artifacts:
            if artifact_name == name:
                return path
        raise KeyError(f"Unknown benchmark artifact: {name!r}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_encoder_name(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
    if not value or any(character not in allowed for character in value):
        raise ValueError(
            "Encoder names must contain only letters, digits, dots, underscores, or hyphens"
        )
    return value


ALLOWED_POOLING = {
    "valid_final_layer_token_mean",
    "mean_std",
    "mean_max",
    "legacy_mean_std_max",
    "energy_weighted",
    "attention",
    "attention_lda",
    "attention_multi",
    "attention_neighbourhood",
}
ATTENTION_POOLINGS = {
    "attention",
    "attention_lda",
    "attention_multi",
    "attention_neighbourhood",
}
POOLING_PARTS = {
    "valid_final_layer_token_mean": ("mean",),
    "mean_std": ("mean", "std"),
    "mean_max": ("mean", "max"),
    "legacy_mean_std_max": ("mean", "std", "max"),
}
ALLOWED_NORMALIZATION = {"snapshot_level", "rms_normalized"}
MIN_BEATS_WINDOW_MS = 200


def _validate_protocol(protocol: BenchmarkProtocol) -> None:
    if protocol.event_window_ms != 200:
        raise ValueError("The benchmark supports the locked 200 ms reference window")
    if protocol.window_conditions is not None:
        if not protocol.window_conditions or any(
            value <= 0 for value in protocol.window_conditions
        ):
            raise ValueError("window_conditions must contain positive durations")
    if protocol.window_shift_ms != 0 and protocol.include_controls:
        raise ValueError(
            "window_shift_ms requires include_controls=False: the "
            "negative-control chain is defined for the exact "
            "peak-centred window only"
        )
    if protocol.detector != "absolute_amplitude_peak_within_event_interval":
        raise ValueError("The benchmark requires label-blind absolute-amplitude alignment")
    if protocol.normalization not in ALLOWED_NORMALIZATION:
        raise ValueError(
            f"normalization must be one of {sorted(ALLOWED_NORMALIZATION)}"
        )
    if protocol.pooling not in ALLOWED_POOLING:
        raise ValueError(f"pooling must be one of {sorted(ALLOWED_POOLING)}")
    if protocol.classifier != "balanced_l2_logistic_regression":
        raise ValueError("The benchmark requires balanced L2 logistic regression")
    if protocol.outer_splits < 2:
        raise ValueError("outer_splits must be at least 2")
    if protocol.logistic_c <= 0:
        raise ValueError("logistic_c must be positive")
    if protocol.c_grid is not None:
        if not protocol.c_grid or any(value <= 0 for value in protocol.c_grid):
            raise ValueError("c_grid must contain positive values")
        if protocol.inner_splits < 2:
            raise ValueError("inner_splits must be at least 2")
    if protocol.calibrate_threshold and protocol.inner_splits < 2:
        raise ValueError("threshold calibration requires inner_splits >= 2")
    if protocol.feature_composition is not None:
        if not protocol.feature_composition:
            raise ValueError("feature_composition must not be empty")
        seen_windows: set[int] = set()
        for milliseconds, pooling in protocol.feature_composition:
            if milliseconds <= 0:
                raise ValueError("composition windows must be positive")
            if pooling not in ALLOWED_POOLING:
                raise ValueError(
                    f"composition pooling must be one of "
                    f"{sorted(ALLOWED_POOLING)}"
                )
            if milliseconds in seen_windows:
                raise ValueError(
                    f"composition repeats window {milliseconds}ms"
                )
            seen_windows.add(milliseconds)
    if protocol.attention_k < 1:
        raise ValueError("attention_k must be at least 1")
    validate_non_centered_specs(protocol.non_centered_windows)
    if protocol.non_centered_windows:
        if protocol.feature_composition is not None:
            raise ValueError(
                "non_centered_windows cannot be combined with "
                "feature_composition"
            )
        if protocol.layers is not None:
            raise ValueError(
                "non_centered_windows cannot be combined with layers mode"
            )
    if protocol.layers is not None:
        if not protocol.layers or any(layer < 0 for layer in protocol.layers):
            raise ValueError("layers must contain non-negative layer indices")
        if protocol.pooling not in ATTENTION_POOLINGS:
            raise ValueError(
                "layers mode requires an attention-family pooling"
            )
        if protocol.feature_composition is not None:
            raise ValueError(
                "layers mode cannot be combined with feature_composition"
            )


def _audit_snapshot(snapshot: DatasetSnapshot) -> tuple[dict[str, object], str]:
    if not snapshot.revision.strip():
        raise ValueError("The dataset snapshot needs an immutable revision")
    if not snapshot.samples:
        raise ValueError("The dataset snapshot contains no samples")

    seen_uids: set[str] = set()
    label_counts = {label: 0 for label in LABEL_TO_INT}
    group_counts: dict[str, int] = {}
    identity_rows: list[dict[str, object]] = []

    for sample in sorted(snapshot.samples, key=lambda item: item.uid):
        if not sample.uid.strip() or sample.uid in seen_uids:
            raise ValueError(f"Duplicate or empty UID: {sample.uid!r}")
        if sample.label not in LABEL_TO_INT:
            raise ValueError(f"Unsupported label for {sample.uid}: {sample.label!r}")
        if not sample.lineage_group_id.strip():
            raise ValueError(f"Missing lineage group for {sample.uid}")

        audio_path = Path(sample.audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        sample_rate, raw = wavfile.read(audio_path)
        if int(sample_rate) <= 0 or raw.size == 0:
            raise ValueError(f"Unreadable or empty waveform for {sample.uid}")
        if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
            raise ValueError(f"Non-finite waveform for {sample.uid}")
        waveform = to_float_mono(raw)
        duration = len(waveform) / float(sample_rate)
        if not (
            np.isfinite(sample.event_start)
            and np.isfinite(sample.event_end)
            and 0.0 <= sample.event_start < sample.event_end <= duration
        ):
            raise ValueError(f"Invalid event interval for {sample.uid}")

        audio_hash = _file_sha256(audio_path)
        identity_rows.append(
            {
                "uid": sample.uid,
                "label": sample.label,
                "lineage_group_id": sample.lineage_group_id,
                "event_start": sample.event_start,
                "event_end": sample.event_end,
                "audio_sha256": audio_hash,
            }
        )
        seen_uids.add(sample.uid)
        label_counts[sample.label] += 1
        group_counts[sample.lineage_group_id] = (
            group_counts.get(sample.lineage_group_id, 0) + 1
        )

    missing_labels = [label for label, count in label_counts.items() if count == 0]
    if missing_labels:
        raise ValueError(f"Snapshot is missing classes: {missing_labels}")

    snapshot_fingerprint = _canonical_sha256(
        {"revision": snapshot.revision, "samples": identity_rows}
    )
    audit = {
        "revision": snapshot.revision,
        "snapshot_fingerprint": snapshot_fingerprint,
        "sample_count": len(identity_rows),
        "label_counts": label_counts,
        "lineage_group_count": len(group_counts),
        "singleton_lineage_group_count": sum(
            count == 1 for count in group_counts.values()
        ),
        "largest_lineage_group": max(group_counts.values()),
        "audio_identity": "sha256_file_content",
        "machine_paths_used_as_identity": False,
    }
    return audit, snapshot_fingerprint


def _encoder_documents(
    encoder_adapters: Sequence[EncoderAdapter],
) -> tuple[tuple[EncoderAdapter, ...], list[dict[str, object]]]:
    if not encoder_adapters:
        raise ValueError("At least one encoder adapter is required")
    by_name: dict[str, EncoderAdapter] = {}
    for adapter in encoder_adapters:
        provenance = adapter.provenance
        name = _safe_encoder_name(provenance.name)
        if name in by_name:
            raise ValueError(f"Duplicate encoder name: {name}")
        if provenance.training_epochs != 0:
            raise ValueError(f"Encoder {name} is not frozen")
        if provenance.token_dimension < 1:
            raise ValueError(f"Encoder {name} has an invalid token dimension")
        if not provenance.upstream_revision or not provenance.checkpoint_sha256:
            raise ValueError(f"Encoder {name} has incomplete provenance")
        by_name[name] = adapter
    ordered = tuple(by_name[name] for name in sorted(by_name))
    documents = [asdict(adapter.provenance) for adapter in ordered]
    return ordered, documents


def _non_centered_document(
    specs: tuple[str, ...],
) -> dict[str, object]:
    document: dict[str, object] = {}
    for spec in specs:
        kind, duration_ms = parse_non_centered_spec(spec)
        definition = {
            "full_audio": "full clip from audio start",
            "post_contact": "fixed segment starting at estimated peak",
            "pre_contact": (
                f"fixed segment ending {PRE_GAP_MS}ms before "
                "annotated event start"
            ),
        }[kind]
        entry: dict[str, object] = {"definition": definition}
        if duration_ms is not None:
            entry["duration_ms"] = duration_ms
        document[spec] = entry
    return document


def _uses_attention_controls(protocol: BenchmarkProtocol) -> bool:
    if not protocol.include_controls:
        return False
    if protocol.pooling in ATTENTION_POOLINGS:
        return True
    return bool(
        protocol.feature_composition
        and any(
            pooling in ATTENTION_POOLINGS
            for _milliseconds, pooling in protocol.feature_composition
        )
    )


def _protocol_document(
    protocol: BenchmarkProtocol,
    snapshot: DatasetSnapshot,
    snapshot_fingerprint: str,
    encoder_documents: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "protocol_version": protocol.protocol_version,
        "dataset": {
            "revision": snapshot.revision,
            "snapshot_fingerprint": snapshot_fingerprint,
        },
        "detector": protocol.detector,
        "window_conditions": [
            f"event_{milliseconds:03d}ms"
            for milliseconds in protocol.resolved_window_conditions
        ],
        "event_window_shift_ms": protocol.window_shift_ms,
        "non_centered_windows": _non_centered_document(
            protocol.non_centered_windows
        ),
        "normalization": protocol.normalization,
        "pooling": protocol.pooling,
        "attention_k": protocol.attention_k,
        "attention_control_transform_policy": (
            ATTENTION_CONTROL_TRANSFORM_POLICY
            if _uses_attention_controls(protocol)
            else None
        ),
        "model_input_policy": {
            "contact_window_only": True,
            "waveform_padding": False,
            "full_clips": bool(
                "full_audio" in protocol.non_centered_windows
            ),
            "one_second_windows": False,
            "outcome_context": False,
            "project_label_visible_to_encoder": False,
        },
        "controls": (
            {
                "enabled": True,
                "applies_to": "centered event windows only",
                "strict_pre_gap_ms": PRE_GAP_MS,
                "strict_pre_window_ms": 200,
                "transient_removal_ms": REMOVED_REPLACEMENT_MS,
                "transient_removal_crossfade_ms": REMOVED_CROSSFADE_MS,
                "conditions": [
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                    "event_selected_removed",
                    "removed_selected_removed",
                    "contact_specific_increment",
                ],
            }
            if protocol.include_controls
            else {"enabled": False}
        ),
        "classifier": {
            "name": protocol.classifier,
            "C": protocol.logistic_c if protocol.c_grid is None else None,
            "C_grid": list(protocol.c_grid) if protocol.c_grid is not None else None,
            "C_selection": (
                "inner_grouped_cv" if protocol.c_grid is not None else "fixed"
            ),
            "inner_splits": (
                protocol.inner_splits if protocol.c_grid is not None else None
            ),
            "class_weight": "balanced",
            "penalty": "l2",
            "solver": "liblinear",
        },
        "decision_threshold": {
            "calibrate": protocol.calibrate_threshold,
            "fixed_default": 0.5,
            "selection_scope": "outer_train_inner_validation",
        },
        "feature_composition": (
            [
                {"window_ms": milliseconds, "pooling": pooling}
                for milliseconds, pooling in protocol.feature_composition
            ]
            if protocol.feature_composition is not None
            else None
        ),
        "layers": (
            {
                "indices": list(protocol.layers),
                "layer_definition": (
                    "output after each transformer block, normalised with "
                    "the backbone final norm, CLS dropped, averaged per "
                    "patch frame; identical post-processing to the last layer"
                ),
            }
            if protocol.layers is not None
            else None
        ),
        "fold_policy": {
            "name": "StratifiedGroupKFold",
            "group": "lineage_group_id",
            "outer_splits": protocol.outer_splits,
            "shuffle": True,
            "seed": protocol.seed,
        },
        "encoders": encoder_documents,
    }


REMOVED_REPLACEMENT_MS = 40
REMOVED_CROSSFADE_MS = 5
PRE_GAP_MS = 50


def _removed_window_row(
    waveform: np.ndarray,
    sample_rate: int,
    peak: float,
    event_start: float,
    uid: str,
    label: str,
    lineage_group_id: str,
    output_manifest: Path,
) -> dict[str, object]:
    window_seconds = 0.2
    dest_start = peak - REMOVED_REPLACEMENT_MS / 2 / 1000.0
    dest_end = peak + REMOVED_REPLACEMENT_MS / 2 / 1000.0
    pre_end = event_start - PRE_GAP_MS / 1000.0
    pre_start = pre_end - window_seconds
    segment_duration = REMOVED_REPLACEMENT_MS / 1000.0
    source_start = pre_start
    source_end = pre_start + segment_duration

    def sample_index(value: float) -> int:
        return int(round(value * sample_rate))

    window_start = peak - window_seconds / 2.0
    window_start_sample = sample_index(window_start)
    window_end_sample = sample_index(window_start + window_seconds)
    window = waveform[window_start_sample:window_end_sample].copy()
    dest_start_sample = sample_index(dest_start)
    dest_end_sample = sample_index(dest_end)
    segment_length = dest_end_sample - dest_start_sample
    source_start_sample = sample_index(source_start)
    segment = waveform[
        source_start_sample : source_start_sample + segment_length
    ].copy()
    if len(segment) != segment_length:
        raise AssertionError(
            f"Removed-window background has an unexpected sample count for {uid}"
        )
    fade = sample_index(REMOVED_CROSSFADE_MS / 1000.0)
    if segment_length <= 2 * fade:
        raise AssertionError(
            f"Removed-window segment is shorter than its crossfades for {uid}"
        )
    relative_dest_start = dest_start_sample - window_start_sample
    blend = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    window[relative_dest_start : relative_dest_start + fade] = (
        window[relative_dest_start : relative_dest_start + fade]
        * (1.0 - blend)
        + segment[:fade] * blend
    )
    window[
        relative_dest_start + fade : relative_dest_start + segment_length - fade
    ] = segment[fade : segment_length - fade]
    window[
        relative_dest_start + segment_length - fade :
        relative_dest_start + segment_length
    ] = (
        window[
            relative_dest_start + segment_length - fade :
            relative_dest_start + segment_length
        ]
        * np.linspace(1.0, 0.0, fade, dtype=np.float32)
        + segment[segment_length - fade : segment_length]
        * np.linspace(0.0, 1.0, fade, dtype=np.float32)
    )

    filename = uid.replace("/", "_").replace("\\", "_")
    window_path = output_manifest.parent / "windows" / "removed_200ms" / f"{filename}.wav"
    window_path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(window_path, sample_rate, to_int16(window))
    relative = Path(
        os.path.relpath(window_path, output_manifest.parent)
    ).as_posix()
    return {
        "uid": uid,
        "label": label,
        "lineage_group_id": lineage_group_id,
        "window_name": "removed_200ms",
        "window_kind": "transient_removed",
        "window_path": relative,
        "window_start": window_start,
        "window_end": window_start + window_seconds,
        "window_duration": window_seconds,
        "sample_rate": sample_rate,
        "event_start": event_start,
        "event_end": peak,
        "estimated_peak_time": peak,
        "window_shift_from_requested_ms": 0.0,
        "alignment_method": "absolute_amplitude_peak_within_annotated_event_interval",
        "wav_boundary_padding_samples": 0,
        "removed_source_start": source_start,
        "removed_source_end": source_end,
        "removed_dest_start": dest_start,
        "removed_dest_end": dest_end,
        "removed_crossfade_seconds": REMOVED_CROSSFADE_MS / 1000.0,
    }


def _prepare_contact_windows(
    snapshot: DatasetSnapshot,
    bundle_root: Path,
    protocol: BenchmarkProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    durations = protocol.resolved_window_conditions
    longest_window_seconds = max(durations) / 1000.0
    exclusion_columns = ["uid", "label", "scope", "reason", "detail"]
    exclusions: list[dict[str, object]] = []
    eligible_rows: list[dict[str, object]] = []
    removed_rows: list[dict[str, object]] = []

    for sample in sorted(snapshot.samples, key=lambda item: item.uid):
        audio_path = Path(sample.audio_path).resolve()
        try:
            sample_rate, raw = wavfile.read(audio_path)
            waveform = to_float_mono(raw)
        except Exception as error:
            exclusions.append(
                {
                    "uid": sample.uid,
                    "label": sample.label,
                    "scope": "event",
                    "reason": "unreadable_audio",
                    "detail": str(error),
                }
            )
            continue
        duration = len(waveform) / float(sample_rate)
        if duration < longest_window_seconds - 1e-9:
            exclusions.append(
                {
                    "uid": sample.uid,
                    "label": sample.label,
                    "scope": "event",
                    "reason": "audio_shorter_than_window",
                    "detail": f"duration={duration:.6f}",
                }
            )
            continue
        if not (
            0.0 <= sample.event_start < sample.event_end <= duration + 1e-6
        ):
            exclusions.append(
                {
                    "uid": sample.uid,
                    "label": sample.label,
                    "scope": "event",
                    "reason": "invalid_event_interval",
                    "detail": (
                        f"[{sample.event_start}, {sample.event_end}] "
                        f"vs duration {duration:.6f}"
                    ),
                }
            )
            continue
        peak = find_peak_time(
            waveform,
            int(sample_rate),
            sample.event_start,
            sample.event_end,
        )
        requested_start = peak - longest_window_seconds / 2.0 + (
            protocol.window_shift_ms / 1000.0
        )
        if (
            requested_start < -1e-9
            or requested_start + longest_window_seconds > duration + 1e-9
        ):
            exclusions.append(
                {
                    "uid": sample.uid,
                    "label": sample.label,
                    "scope": "event",
                    "reason": "window_not_exact",
                    "detail": (
                        f"requested_start={requested_start:.6f}, "
                        f"duration={duration:.6f}"
                    ),
                }
            )
            continue
        unavailable: list[int] = []
        for milliseconds in durations:
            window_seconds = milliseconds / 1000.0
            if (
                sample.event_start
                - PRE_GAP_MS / 1000.0
                - window_seconds
                < -1e-9
            ):
                unavailable.append(milliseconds)
        if protocol.include_controls and unavailable:
            exclusions.append(
                {
                    "uid": sample.uid,
                    "label": sample.label,
                    "scope": "controls",
                    "reason": "strict_pre_unavailable",
                    "detail": (
                        f"event_start={sample.event_start:.6f}, "
                        f"durations={unavailable}"
                    ),
                }
            )
        if (
            protocol.include_controls
            and 200 in durations
            and not unavailable
        ):
            removed_rows.append(
                _removed_window_row(
                    waveform,
                    int(sample_rate),
                    peak,
                    sample.event_start,
                    sample.uid,
                    sample.label,
                    sample.lineage_group_id,
                    bundle_root / "windows_manifest.csv",
                )
            )
        eligible_rows.append(
            {
                "uid": sample.uid,
                "label": sample.label,
                "source_id": sample.lineage_group_id,
                "protocol_role": "snapshot",
                "audio_path": str(audio_path),
                "event_start": sample.event_start,
                "event_end": sample.event_end,
            }
        )

    if not eligible_rows:
        raise ValueError("No samples are eligible for the contact window")
    with tempfile.TemporaryDirectory() as directory:
        manifest_path = Path(directory) / "snapshot.csv"
        pd.DataFrame(eligible_rows).to_csv(manifest_path, index=False)
        windows = prepare_windows(
            manifest_path,
            bundle_root,
            tuple(durations),
            include_strict_pre=protocol.include_controls,
            non_centered_windows=protocol.non_centered_windows,
            event_shift_ms=protocol.window_shift_ms,
        )

    windows = windows.rename(columns={"source_id": "lineage_group_id"})
    if protocol.include_controls and removed_rows:
        windows = pd.concat(
            [windows, pd.DataFrame(removed_rows)], ignore_index=True
        )
    expected_windows: set[str] = set()
    for milliseconds in durations:
        suffix = f"{milliseconds:03d}ms"
        expected_windows.add(f"event_{suffix}")
        if protocol.include_controls:
            expected_windows.add(f"pre_{suffix}")
    expected_windows.update(protocol.non_centered_windows)
    if protocol.include_controls and 200 in durations:
        expected_windows.add("removed_200ms")
    if set(windows["window_name"]) != expected_windows:
        raise AssertionError("The benchmark prepared a non-contact input condition")
    if (windows["window_shift_from_requested_ms"].abs() > 1e-9).any():
        raise AssertionError("A prepared window is not exactly peak-centred")
    if (windows["wav_boundary_padding_samples"] != 0).any():
        raise AssertionError("A prepared window contains waveform padding")
    windows.to_csv(bundle_root / "windows_manifest.csv", index=False)
    exclusions_df = pd.DataFrame(exclusions, columns=exclusion_columns)
    exclusions_df.to_csv(bundle_root / "exclusions.csv", index=False)
    return windows, exclusions_df


def _feature_fingerprint(
    protocol: BenchmarkProtocol,
    snapshot_fingerprint: str,
    adapter: EncoderAdapter,
    pooling: str | None = None,
    layer: int | None = None,
) -> str:
    return _canonical_sha256(
        {
            "snapshot_fingerprint": snapshot_fingerprint,
            "detector": protocol.detector,
            "window_conditions": list(protocol.resolved_window_conditions),
            "window_shift_ms": protocol.window_shift_ms,
            "non_centered_windows": list(protocol.non_centered_windows),
            "normalization": protocol.normalization,
            "pooling": pooling or protocol.pooling,
            "layer": layer,
            "encoder": asdict(adapter.provenance),
        }
    )


def _extract_features(
    adapters: tuple[EncoderAdapter, ...],
    windows: pd.DataFrame,
    windows_manifest: Path,
    bundle_root: Path,
    protocol: BenchmarkProtocol,
    snapshot_fingerprint: str,
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    composed = protocol.feature_composition is not None
    components = (
        protocol.feature_composition
        if composed
        else ((protocol.event_window_ms, protocol.pooling),)
    )
    layer_mode = protocol.layers is not None
    requested_layers = protocol.layers or ()
    for adapter in adapters:
        provenance = adapter.provenance
        layerwise = getattr(adapter, "encode_layer_tokens", None)
        if layer_mode and layerwise is None:
            raise ValueError(
                f"Encoder {provenance.name} does not support layer-wise "
                "extraction"
            )
        for milliseconds, pooling in components:
            suffix = f"__{milliseconds}ms_{pooling}" if composed else ""
            component_key = (
                f"{milliseconds}ms_{pooling}" if composed else provenance.name
            )
            layer_keys = (
                (component_key,)
                if not layer_mode
                else tuple(
                    f"{component_key}__layer{layer}"
                    for layer in requested_layers
                )
            )
            layer_files: dict[str, Path] = {}
            layer_markers: dict[str, Path] = {}
            layer_fingerprints: dict[str, str] = {}
            for layer, layer_key in zip(
                (None,) if not layer_mode else requested_layers,
                layer_keys,
                strict=True,
            ):
                layer_suffix = "" if layer is None else f"__layer{layer}"
                output_path = (
                    bundle_root
                    / "features"
                    / f"{provenance.name}{suffix}{layer_suffix}.csv"
                )
                if pooling in ATTENTION_POOLINGS:
                    output_path = output_path.with_name(
                        f"{provenance.name}{suffix}{layer_suffix}_tokens.csv"
                    )
                marker_path = output_path.with_name(
                    f"{provenance.name}{suffix}{layer_suffix}.provenance.json"
                )
                fingerprint = _feature_fingerprint(
                    protocol,
                    snapshot_fingerprint,
                    adapter,
                    pooling,
                    layer,
                )
                if output_path.is_file() and marker_path.is_file():
                    try:
                        marker = json.loads(
                            marker_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError):
                        marker = {}
                    if marker.get("fingerprint") == fingerprint:
                        outputs[layer_key] = output_path
                        continue
                layer_files[layer_key] = output_path
                layer_markers[layer_key] = marker_path
                layer_fingerprints[layer_key] = fingerprint

            if not layer_files:
                continue
            rows_by_key: dict[str, list[dict[str, object]]] = {
                key: [] for key in layer_files
            }
            pooling_parts = POOLING_PARTS.get(pooling, ("mean",))
            token_columns = [
                f"feat_{index:04d}"
                for index in range(provenance.token_dimension)
            ]
            if pooling in ATTENTION_POOLINGS:
                feature_columns = token_columns
            else:
                feature_columns = [
                    f"feat_{index:04d}"
                    for index in range(
                        provenance.token_dimension * len(pooling_parts)
                    )
                ]
            stream_writers: dict[str, object] = {}
            stream_handles: dict[str, object] = {}
            if layer_mode:
                import csv as _csv

                for layer_key, output_path in layer_files.items():
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    handle = output_path.open("w", encoding="utf-8", newline="")
                    stream_handles[layer_key] = handle
                    stream_writers[layer_key] = _csv.DictWriter(
                        handle,
                        fieldnames=[
                            "uid", "label", "lineage_group_id",
                            "window_name", "encoder",
                            "encoder_training_epochs", "embedding_pooling",
                            "embedding_input_policy", "upstream_revision",
                            "checkpoint_sha256", "inference_precision",
                            "one_token_degeneracy", "layer", "token_index",
                            *token_columns,
                        ],
                    )
                    stream_writers[layer_key].writeheader()
            component_windows = None
            if composed:
                component_windows = {
                    f"event_{milliseconds:03d}ms",
                    f"pre_{milliseconds:03d}ms",
                }
            for row in windows.sort_values("uid").itertuples(index=False):
                if (
                    component_windows is not None
                    and str(row.window_name) not in component_windows
                ):
                    continue
                window_path = Path(str(row.window_path))
                if not window_path.is_absolute():
                    window_path = windows_manifest.parent / window_path
                sample_rate, raw = wavfile.read(window_path.resolve())
                waveform = to_float_mono(raw)
                if protocol.normalization == "rms_normalized":
                    rms = float(np.sqrt(np.mean(np.square(waveform))))
                    if rms > 1e-12:
                        waveform = waveform / rms * 0.1
                one_token_degeneracy = False
                if layer_mode:
                    all_tokens = np.asarray(
                        layerwise(waveform, int(sample_rate)),
                        dtype=np.float64,
                    )
                    if all_tokens.ndim != 3:
                        raise ValueError(
                            f"Encoder {provenance.name} returned invalid "
                            f"layer token shape {all_tokens.shape}"
                        )
                    if not np.isfinite(all_tokens).all():
                        raise FloatingPointError(
                            f"Encoder {provenance.name} returned "
                            "non-finite layer tokens"
                        )
                    for layer, layer_key in zip(
                        requested_layers, layer_keys, strict=True
                    ):
                        if layer_key not in rows_by_key:
                            continue
                        if layer >= all_tokens.shape[0]:
                            raise ValueError(
                                f"Encoder {provenance.name} has "
                                f"{all_tokens.shape[0]} layers but layer "
                                f"{layer} was requested"
                            )
                        tokens = all_tokens[layer]
                        one_token_degeneracy = bool(tokens.shape[0] == 1)
                        for token_index, token in enumerate(tokens):
                            token_row: dict[str, object] = {
                                "uid": str(row.uid),
                                "label": str(row.label),
                                "lineage_group_id": str(
                                    row.lineage_group_id
                                ),
                                "window_name": str(row.window_name),
                                "encoder": provenance.name,
                                "encoder_training_epochs": (
                                    provenance.training_epochs
                                ),
                                "embedding_pooling": pooling,
                                "embedding_input_policy": (
                                    "resample_16khz_no_padding"
                                ),
                                "upstream_revision": (
                                    provenance.upstream_revision
                                ),
                                "checkpoint_sha256": (
                                    provenance.checkpoint_sha256
                                ),
                                "inference_precision": provenance.precision,
                                "one_token_degeneracy": (
                                    one_token_degeneracy
                                ),
                                "layer": int(layer),
                                "token_index": int(token_index),
                            }
                            token_row.update(
                                {
                                    column: float(value)
                                    for column, value in zip(
                                        token_columns, token, strict=True
                                    )
                                }
                            )
                            if layer_mode:
                                stream_writers[layer_key].writerow(token_row)
                            else:
                                if layer_mode:
                                    stream_writers[layer_key].writerow(
                                        token_row
                                    )
                                else:
                                    rows_by_key[layer_key].append(token_row)
                    continue
                tokens = np.asarray(
                    adapter.encode_tokens(waveform, int(sample_rate)),
                    dtype=np.float64,
                )
                expected_tail = provenance.token_dimension
                if (
                    tokens.ndim != 2
                    or tokens.shape[0] < 1
                    or tokens.shape[1] != expected_tail
                ):
                    raise ValueError(
                        f"Encoder {provenance.name} returned invalid "
                        f"token shape {tokens.shape}"
                    )
                if not np.isfinite(tokens).all():
                    raise FloatingPointError(
                        f"Encoder {provenance.name} returned non-finite tokens"
                    )
                one_token_degeneracy = bool(tokens.shape[0] == 1)
                if pooling in ATTENTION_POOLINGS:
                    for token_index, token in enumerate(tokens):
                        token_row: dict[str, object] = {
                            "uid": str(row.uid),
                            "label": str(row.label),
                            "lineage_group_id": str(row.lineage_group_id),
                            "window_name": str(row.window_name),
                            "encoder": provenance.name,
                            "encoder_training_epochs": (
                                provenance.training_epochs
                            ),
                            "embedding_pooling": pooling,
                            "embedding_input_policy": (
                                "resample_16khz_no_padding"
                            ),
                            "upstream_revision": provenance.upstream_revision,
                            "checkpoint_sha256": provenance.checkpoint_sha256,
                            "inference_precision": provenance.precision,
                            "one_token_degeneracy": one_token_degeneracy,
                            "token_index": int(token_index),
                        }
                        token_row.update(
                            {
                                column: float(value)
                                for column, value in zip(
                                    token_columns, token, strict=True
                                )
                            }
                        )
                        rows_by_key[component_key].append(token_row)
                    continue
                parts: dict[str, np.ndarray] = {
                    "mean": tokens.mean(axis=0),
                    "std": tokens.std(axis=0),
                    "max": tokens.max(axis=0),
                }
                if pooling == "energy_weighted":
                    energies = np.square(tokens).sum(axis=1)
                    energy_sum = energies.sum()
                    if energy_sum > 1e-12:
                        weights = energies / energy_sum
                    else:
                        weights = np.full(
                            tokens.shape[0], 1.0 / tokens.shape[0]
                        )
                    embedding = weights @ tokens
                else:
                    embedding = np.concatenate(
                        [parts[part] for part in pooling_parts]
                    )
                feature_row: dict[str, object] = {
                    "uid": str(row.uid),
                    "label": str(row.label),
                    "lineage_group_id": str(row.lineage_group_id),
                    "window_name": str(row.window_name),
                    "encoder": provenance.name,
                    "encoder_training_epochs": provenance.training_epochs,
                    "embedding_pooling": pooling,
                    "embedding_input_policy": "resample_16khz_no_padding",
                    "upstream_revision": provenance.upstream_revision,
                    "checkpoint_sha256": provenance.checkpoint_sha256,
                    "inference_precision": provenance.precision,
                    "one_token_degeneracy": one_token_degeneracy,
                }
                feature_row.update(
                    {
                        column: float(value)
                        for column, value in zip(
                            feature_columns, embedding, strict=True
                        )
                    }
                )
                rows_by_key[component_key].append(feature_row)

            for layer_key, output_path in layer_files.items():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if layer_mode:
                    stream_handles[layer_key].close()
                else:
                    pd.DataFrame(rows_by_key[layer_key]).to_csv(
                        output_path, index=False
                    )
                _write_json(
                    layer_markers[layer_key],
                    {
                        "fingerprint": layer_fingerprints[layer_key],
                        "snapshot_fingerprint": snapshot_fingerprint,
                        "encoder": asdict(provenance),
                    },
                )
                outputs[layer_key] = output_path
    return outputs


def _make_fold_assignments(
    snapshot: DatasetSnapshot,
    protocol: BenchmarkProtocol,
    eligible_uids: set[str],
) -> pd.DataFrame:
    ordered = sorted(
        (sample for sample in snapshot.samples if sample.uid in eligible_uids),
        key=lambda item: item.uid,
    )
    labels = np.asarray([LABEL_TO_INT[sample.label] for sample in ordered], dtype=int)
    groups = np.asarray([sample.lineage_group_id for sample in ordered], dtype=object)
    if len(set(groups)) < protocol.outer_splits:
        raise ValueError("The snapshot has fewer lineage groups than outer folds")

    splitter = StratifiedGroupKFold(
        n_splits=protocol.outer_splits,
        shuffle=True,
        random_state=protocol.seed,
    )
    assignments = np.full(len(ordered), -1, dtype=int)
    for outer_fold, (train, test) in enumerate(
        splitter.split(np.zeros(len(labels)), labels, groups)
    ):
        if set(groups[train]).intersection(groups[test]):
            raise AssertionError("A lineage group crosses an outer fold")
        if assignments[test].max(initial=-1) != -1:
            raise AssertionError("A sample was assigned to more than one outer fold")
        assignments[test] = outer_fold
    if (assignments < 0).any():
        raise AssertionError("A sample has no outer-fold assignment")

    return pd.DataFrame(
        {
            "uid": [sample.uid for sample in ordered],
            "label": [sample.label for sample in ordered],
            "lineage_group_id": groups,
            "outer_fold": assignments,
        }
    )


def _select_threshold_inner(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    c_value: float,
    inner_splits: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
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

    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(matrix[train])
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5_000,
        random_state=seed,
    )
    classifier.fit(transformed_train, labels[train])
    train_scores = classifier.predict_proba(transformed_train)[:, 1]
    ordered = np.sort(train_scores)
    midpoints = (ordered[1:] + ordered[:-1]) / 2.0
    candidates = np.unique(np.concatenate([[0.5], midpoints]))

    scores_by_candidate: dict[float, list[float]] = {
        float(candidate): [] for candidate in candidates
    }
    for inner_fold, (train_position, validation_position) in enumerate(folds):
        inner_train = train[train_position]
        inner_validation = train[validation_position]
        inner_scaler = StandardScaler()
        transformed_inner_train = inner_scaler.fit_transform(
            matrix[inner_train]
        )
        transformed_inner_validation = inner_scaler.transform(
            matrix[inner_validation]
        )
        inner_classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed + inner_fold,
        )
        inner_classifier.fit(transformed_inner_train, labels[inner_train])
        validation_scores = inner_classifier.predict_proba(
            transformed_inner_validation
        )[:, 1]
        for candidate in candidates:
            prediction = (validation_scores >= candidate).astype(int)
            scores_by_candidate[float(candidate)].append(
                float(
                    balanced_accuracy_score(
                        labels[inner_validation], prediction
                    )
                )
            )

    records = [
        {
            "threshold": candidate,
            "inner_balanced_accuracy": float(np.mean(values)),
        }
        for candidate, values in scores_by_candidate.items()
    ]
    best = max(
        records,
        key=lambda record: (
            record["inner_balanced_accuracy"],
            -abs(record["threshold"] - 0.5),
        ),
    )
    return float(best["threshold"]), records


def _select_c_inner(
    matrix: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    c_grid: tuple[float, ...],
    inner_splits: int,
    seed: int,
) -> tuple[float, list[dict[str, float]]]:
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
    scores: dict[float, list[float]] = {value: [] for value in c_grid}
    for inner_fold, (train_position, validation_position) in enumerate(folds):
        inner_train = train[train_position]
        inner_validation = train[validation_position]
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(matrix[inner_train])
        transformed_validation = scaler.transform(matrix[inner_validation])
        for c_value in c_grid:
            classifier = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                solver="liblinear",
                max_iter=5_000,
                random_state=seed + inner_fold,
            )
            classifier.fit(transformed_train, labels[inner_train])
            prediction = classifier.predict(transformed_validation)
            scores[c_value].append(
                float(
                    balanced_accuracy_score(
                        labels[inner_validation], prediction
                    )
                )
            )
    records = [
        {
            "C": c_value,
            "inner_balanced_accuracy": float(np.mean(values)),
            "inner_balanced_accuracy_std": float(np.std(values)),
        }
        for c_value, values in scores.items()
    ]
    best = max(
        records,
        key=lambda record: (
            record["inner_balanced_accuracy"],
            -record["inner_balanced_accuracy_std"],
            -records.index(record),
        ),
    )
    return float(best["C"]), records


def _composed_load(feature_paths: dict[str, Path], protocol: BenchmarkProtocol):
    """Load per-component feature data for a composed run."""
    loaded: dict[str, dict[str, np.ndarray | dict]] = {}
    for milliseconds, pooling in protocol.feature_composition:
        key = f"{milliseconds}ms_{pooling}"
        path = feature_paths[key]
        if pooling in ATTENTION_POOLINGS:
            loaded[key] = {"tokens": _load_token_table(path)}
        else:
            frame = pd.read_csv(path)
            loaded[key] = {"frame": frame}
    return loaded


def _evaluate_encoders_composed(
    feature_paths: dict[str, Path],
    folds: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    loaded = _composed_load(feature_paths, protocol)

    # Paired samples: intersection of every component's event windows AND
    # every component's pre windows (samples without a strict-pre window for
    # any component cannot form a matched pre combination).
    event_available: set[str] | None = None
    pre_available: set[str] | None = None
    for milliseconds, pooling in protocol.feature_composition:
        key = f"{milliseconds}ms_{pooling}"
        event_name = f"event_{milliseconds:03d}ms"
        pre_name = f"pre_{milliseconds:03d}ms"
        if pooling in ATTENTION_POOLINGS:
            token_table = loaded[key]["tokens"]
            event_uids = {
                uid for (uid, name) in token_table if name == event_name
            }
            pre_uids = {
                uid for (uid, name) in token_table if name == pre_name
            }
        else:
            frame = loaded[key]["frame"]
            event_uids = set(
                frame.loc[frame["window_name"].eq(event_name), "uid"]
            )
            pre_uids = set(
                frame.loc[frame["window_name"].eq(pre_name), "uid"]
            )
        event_available = (
            event_uids
            if event_available is None
            else event_available & event_uids
        )
        pre_available = (
            pre_uids if pre_available is None else pre_available & pre_uids
        )
    available = event_available & pre_available
    paired = folds[
        folds["uid"].isin(available)
    ].sort_values("uid").reset_index(drop=True)
    if len(paired) < 2:
        raise ValueError("Too few paired samples for composed evaluation")
    labels = paired["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = paired["lineage_group_id"].to_numpy(dtype=object)
    paired_uids = [str(uid) for uid in paired["uid"]]
    fold_array = paired["outer_fold"].to_numpy(dtype=int)

    def composed_matrices(
        train: np.ndarray,
        fit_prefix: str,
        apply_prefix: str,
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for milliseconds, pooling in protocol.feature_composition:
            key = f"{milliseconds}ms_{pooling}"
            fit_window = f"{fit_prefix}_{milliseconds:03d}ms"
            apply_window = f"{apply_prefix}_{milliseconds:03d}ms"
            if pooling in ATTENTION_POOLINGS:
                token_table = loaded[key]["tokens"]
                directions = _attention_fit_directions(
                    [
                        token_table[(paired_uids[i], fit_window)]
                        for i in train
                    ],
                    labels[train],
                    pooling,
                    protocol.attention_k,
                )
                parts.append(
                    np.stack(
                        [
                            _attention_family_pool(
                                token_table[(uid, apply_window)],
                                directions,
                                pooling,
                                protocol.attention_k,
                            )
                            for uid in paired_uids
                        ]
                    )
                )
            else:
                frame = loaded[key]["frame"]
                feature_columns = [
                    column
                    for column in frame
                    if column.startswith("feat_")
                ]
                selected = frame[
                    frame["window_name"].eq(apply_window)
                ].set_index("uid")
                parts.append(
                    selected.loc[paired_uids, feature_columns].to_numpy(
                        dtype=np.float64
                    )
                )
        return np.concatenate(parts, axis=1)

    conditions = ("event_selected_event", "event_selected_pre", "pre_selected_pre")
    rules = (
        ("fixed_0.5", "calibrated")
        if protocol.calibrate_threshold
        else ("fixed_0.5",)
    )
    fold_scores: dict[str, np.ndarray] = {
        condition: np.full(len(paired), np.nan, dtype=np.float64)
        for condition in conditions
    }
    fold_predictions: dict[tuple[str, str], np.ndarray] = {
        (condition, rule): np.full(len(paired), -1, dtype=int)
        for condition in conditions
        for rule in rules
    }

    for outer_fold in sorted(set(fold_array)):
        test = np.flatnonzero(fold_array == outer_fold)
        train = np.flatnonzero(fold_array != outer_fold)
        fold_seed = protocol.seed + int(outer_fold)
        event_matrix = composed_matrices(train, "event", "event")
        event_to_pre_matrix = composed_matrices(train, "event", "pre")
        pre_matrix = composed_matrices(train, "pre", "pre")

        if protocol.c_grid is not None:
            c_value, records = _select_c_inner(
                event_matrix,
                labels,
                groups,
                train,
                protocol.c_grid,
                protocol.inner_splits,
                fold_seed,
            )
        else:
            c_value = protocol.logistic_c
            records = []
        selection_row = {
            "encoder": "composed",
            "condition": "event",
            "window_ms": 0,
            "outer_fold": int(outer_fold),
            "selected_C": c_value,
            "inner_scores_json": json.dumps(records, sort_keys=True),
        }
        pre_c_value = c_value
        if protocol.c_grid is not None:
            pre_c_value, pre_records = _select_c_inner(
                pre_matrix,
                labels,
                groups,
                train,
                protocol.c_grid,
                protocol.inner_splits,
                fold_seed,
            )
            selection_row["pre_selected_C"] = pre_c_value
            selection_row["pre_inner_scores_json"] = json.dumps(
                pre_records, sort_keys=True
            )
        if protocol.calibrate_threshold:
            threshold, threshold_records = _select_threshold_inner(
                event_matrix,
                labels,
                groups,
                train,
                c_value,
                protocol.inner_splits,
                fold_seed,
            )
            selection_row["selected_threshold"] = threshold
            selection_row["threshold_scores_json"] = json.dumps(
                threshold_records, sort_keys=True
            )
            pre_threshold, _pre_threshold_records = _select_threshold_inner(
                pre_matrix,
                labels,
                groups,
                train,
                pre_c_value,
                protocol.inner_splits,
                fold_seed,
            )
            selection_row["pre_selected_threshold"] = pre_threshold
        selection_rows.append(selection_row)

        event_probe = _fit_probe(
            event_matrix, labels, train, c_value, fold_seed
        )
        pre_probe = _fit_probe(
            pre_matrix, labels, train, pre_c_value, fold_seed
        )
        transformed_event = event_probe[0].transform(event_matrix[test])
        event_scores = event_probe[1].predict_proba(transformed_event)[:, 1]
        transformed_event_to_pre = event_probe[0].transform(
            event_to_pre_matrix[test]
        )
        event_to_pre_scores = event_probe[1].predict_proba(
            transformed_event_to_pre
        )[:, 1]
        transformed_pre = pre_probe[0].transform(pre_matrix[test])
        pre_scores = pre_probe[1].predict_proba(transformed_pre)[:, 1]
        fold_scores["event_selected_event"][test] = event_scores
        fold_scores["event_selected_pre"][test] = event_to_pre_scores
        fold_scores["pre_selected_pre"][test] = pre_scores
        for rule in rules:
            threshold = (
                selection_row["selected_threshold"]
                if rule == "calibrated"
                else 0.5
            )
            pre_threshold = (
                selection_row["pre_selected_threshold"]
                if rule == "calibrated"
                else 0.5
            )
            fold_predictions[("event_selected_event", rule)][test] = (
                event_scores >= threshold
            ).astype(int)
            fold_predictions[("event_selected_pre", rule)][test] = (
                event_to_pre_scores >= threshold
            ).astype(int)
            fold_predictions[("pre_selected_pre", rule)][test] = (
                pre_scores >= pre_threshold
            ).astype(int)

    representation_roles = {
        "event_selected_event": ("event_composed", "event_composed"),
        "event_selected_pre": ("event_composed", "pre_composed"),
        "pre_selected_pre": ("pre_composed", "pre_composed"),
    }
    for condition in conditions:
        for rule in rules:
            rule_predictions = fold_predictions[(condition, rule)]
            if (
                (rule_predictions < 0).any()
                or not np.isfinite(fold_scores[condition]).all()
            ):
                raise AssertionError(
                    f"Incomplete composed predictions for {condition}/{rule}"
                )
            for position, row in enumerate(paired.itertuples(index=False)):
                prediction_rows.append(
                    {
                        "encoder": "composed",
                        "condition": condition,
                        "window_ms": 0,
                        "decision_rule": rule,
                        "representation_fit_window": (
                            representation_roles[condition][0]
                        ),
                        "representation_apply_window": (
                            representation_roles[condition][1]
                        ),
                        "uid": str(row.uid),
                        "label": str(row.label),
                        "lineage_group_id": str(row.lineage_group_id),
                        "outer_fold": int(row.outer_fold),
                        "y_true": int(labels[position]),
                        "y_pred": int(rule_predictions[position]),
                        "score_ground_ball": float(
                            fold_scores[condition][position]
                        ),
                    }
                )
            matrix_counts = confusion_matrix(
                labels, rule_predictions, labels=[0, 1]
            )
            metric_rows.append(
                {
                    "encoder": "composed",
                    "condition": condition,
                    "window_ms": 0,
                    "decision_rule": rule,
                    "primary_metric": "balanced_accuracy",
                    "balanced_accuracy": float(
                        balanced_accuracy_score(labels, rule_predictions)
                    ),
                    "accuracy": float(accuracy_score(labels, rule_predictions)),
                    "roc_auc": float(roc_auc_score(labels, fold_scores[condition])),
                    "macro_f1": float(
                        f1_score(labels, rule_predictions, average="macro")
                    ),
                    "true_fly_pred_fly": int(matrix_counts[0, 0]),
                    "true_fly_pred_ground": int(matrix_counts[0, 1]),
                    "true_ground_pred_fly": int(matrix_counts[1, 0]),
                    "true_ground_pred_ground": int(matrix_counts[1, 1]),
                    "eligible_samples": len(paired),
                    "lineage_groups": int(paired["lineage_group_id"].nunique()),
                }
            )
    for rule in rules:
        increment = (
            float(
                balanced_accuracy_score(
                    labels, fold_predictions[("event_selected_event", rule)]
                )
            )
            - float(
                balanced_accuracy_score(
                    labels, fold_predictions[("event_selected_pre", rule)]
                )
            )
        )
        metric_rows.append(
            {
                "encoder": "composed",
                "condition": "contact_specific_increment",
                "window_ms": 0,
                "decision_rule": rule,
                "primary_metric": "balanced_accuracy",
                "balanced_accuracy": increment,
                "accuracy": float("nan"),
                "roc_auc": float("nan"),
                "macro_f1": float("nan"),
                "true_fly_pred_fly": int(0),
                "true_fly_pred_ground": int(0),
                "true_ground_pred_fly": int(0),
                "true_ground_pred_ground": int(0),
                "eligible_samples": len(paired),
                "lineage_groups": int(paired["lineage_group_id"].nunique()),
            }
        )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(
            selection_rows,
            columns=[
                "encoder",
                "condition",
                "window_ms",
                "outer_fold",
                "selected_C",
                "inner_scores_json",
                "pre_selected_C",
                "pre_inner_scores_json",
                "selected_threshold",
                "threshold_scores_json",
                "pre_selected_threshold",
            ],
        ).dropna(axis=1, how="all"),
    )


def _validate_requested_layers(
    feature_paths: dict[str, Path],
    protocol: BenchmarkProtocol,
    encoder_names: Sequence[str],
) -> None:
    """Layer mode must produce a feature file for every requested layer."""
    if protocol.layers is None:
        return
    for name in encoder_names:
        for layer in protocol.layers:
            key = f"{name}__layer{layer}"
            if key not in feature_paths:
                raise ValueError(
                    f"Missing feature file for {key}; the layer was "
                    "requested but extraction produced no tokens"
                )


def _evaluate_encoders(
    feature_paths: dict[str, Path],
    folds: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if protocol.feature_composition is not None:
        return _evaluate_encoders_composed(
            feature_paths, folds, protocol
        )
    if protocol.include_controls:
        return _evaluate_encoders_with_controls(
            feature_paths, folds, protocol
        )
    return _evaluate_encoders_primary(feature_paths, folds, protocol)


def _evaluate_non_centered_condition(
    encoder_name: str,
    spec: str,
    window_ms: int,
    features: pd.DataFrame | None,
    feature_columns: list[str],
    token_table: dict[tuple[str, str], np.ndarray] | None,
    folds: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Evaluate one non-centered condition as an independent condition.

    Non-centered windows have no negative-control chain by design; each
    condition uses its own available-sample set. Predictions and metrics
    use the window spec itself as the condition name.
    """
    attention = token_table is not None
    if attention:
        uids = sorted(
            uid for (uid, name) in token_table if name == spec
        )
    else:
        uids = sorted(
            features.loc[features["window_name"].eq(spec), "uid"]
        )
    aligned = folds[folds["uid"].isin(uids)].sort_values(
        "uid"
    ).reset_index(drop=True)
    if len(aligned) < 2:
        return [], [], []
    aligned_uids = [str(uid) for uid in aligned["uid"]]
    if attention:
        matrix = None
    else:
        selected = features[features["window_name"].eq(spec)]
        aligned = aligned.merge(
            selected[["uid", *feature_columns]],
            on="uid",
            how="left",
            validate="one_to_one",
        )
        matrix = aligned[feature_columns].to_numpy(dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError(
                f"Missing or non-finite features for "
                f"{encoder_name}/{spec}"
            )
    labels = aligned["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = aligned["lineage_group_id"].to_numpy(dtype=object)
    scores = np.full(len(aligned), np.nan, dtype=np.float64)
    predictions_by_rule: dict[str, np.ndarray] = {}
    thresholds: dict[int, float] = {}

    def build_matrix(train: np.ndarray) -> np.ndarray:
        if not attention:
            return matrix
        directions = _attention_fit_directions(
            [token_table[(aligned_uids[i], spec)] for i in train],
            labels[train],
            protocol.pooling,
            protocol.attention_k,
        )
        return np.stack(
            [
                _attention_family_pool(
                    token_table[(uid, spec)],
                    directions,
                    protocol.pooling,
                    protocol.attention_k,
                )
                for uid in aligned_uids
            ]
        )

    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    for outer_fold in sorted(aligned["outer_fold"].unique()):
        test = np.flatnonzero(
            aligned["outer_fold"].to_numpy() == outer_fold
        )
        train = np.flatnonzero(
            aligned["outer_fold"].to_numpy() != outer_fold
        )
        fold_matrix = build_matrix(train)
        scaler = StandardScaler()
        transformed_train = scaler.fit_transform(fold_matrix[train])
        transformed_test = scaler.transform(fold_matrix[test])
        if protocol.c_grid is not None:
            c_value, records = _select_c_inner(
                fold_matrix,
                labels,
                groups,
                train,
                protocol.c_grid,
                protocol.inner_splits,
                protocol.seed + int(outer_fold),
            )
        else:
            c_value = protocol.logistic_c
            records = []
        classifier = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=protocol.seed + int(outer_fold),
        )
        classifier.fit(transformed_train, labels[train])
        test_scores = classifier.predict_proba(transformed_test)[:, 1]
        scores[test] = test_scores
        predictions_by_rule.setdefault(
            "fixed_0.5", np.full(len(aligned), -1, dtype=int)
        )[test] = (test_scores >= 0.5).astype(int)
        selection_row: dict[str, object] = {
            "encoder": encoder_name,
            "condition": spec,
            "window_ms": window_ms,
            "outer_fold": int(outer_fold),
            "selected_C": c_value,
            "inner_scores_json": json.dumps(records, sort_keys=True),
        }
        if protocol.calibrate_threshold:
            threshold, threshold_records = _select_threshold_inner(
                fold_matrix,
                labels,
                groups,
                train,
                c_value,
                protocol.inner_splits,
                protocol.seed + int(outer_fold),
            )
            thresholds[int(outer_fold)] = threshold
            selection_row["selected_threshold"] = threshold
            selection_row["threshold_scores_json"] = json.dumps(
                threshold_records, sort_keys=True
            )
            predictions_by_rule.setdefault(
                "calibrated", np.full(len(aligned), -1, dtype=int)
            )[test] = (test_scores >= threshold).astype(int)
        selection_rows.append(selection_row)

    for rule in predictions_by_rule:
        rule_predictions = predictions_by_rule[rule]
        if (rule_predictions < 0).any():
            raise AssertionError(
                f"Incomplete out-of-fold predictions for "
                f"{encoder_name}/{spec}/{rule}"
            )
        if not np.isfinite(scores).all():
            raise AssertionError(
                f"Non-finite scores for {encoder_name}/{spec}"
            )
        for position, row in enumerate(aligned.itertuples(index=False)):
            prediction_rows.append(
                {
                    "encoder": encoder_name,
                    "condition": spec,
                    "window_ms": window_ms,
                    "decision_rule": rule,
                    "uid": str(row.uid),
                    "label": str(row.label),
                    "lineage_group_id": str(row.lineage_group_id),
                    "outer_fold": int(row.outer_fold),
                    "y_true": int(labels[position]),
                    "y_pred": int(rule_predictions[position]),
                    "score_ground_ball": float(scores[position]),
                }
            )
        matrix_counts = confusion_matrix(
            labels, rule_predictions, labels=[0, 1]
        )
        metric_rows.append(
            {
                "encoder": encoder_name,
                "condition": spec,
                "window_ms": window_ms,
                "decision_rule": rule,
                "primary_metric": "balanced_accuracy",
                "balanced_accuracy": float(
                    balanced_accuracy_score(labels, rule_predictions)
                ),
                "accuracy": float(accuracy_score(labels, rule_predictions)),
                "roc_auc": float(roc_auc_score(labels, scores)),
                "macro_f1": float(
                    f1_score(labels, rule_predictions, average="macro")
                ),
                "true_fly_pred_fly": int(matrix_counts[0, 0]),
                "true_fly_pred_ground": int(matrix_counts[0, 1]),
                "true_ground_pred_fly": int(matrix_counts[1, 0]),
                "true_ground_pred_ground": int(matrix_counts[1, 1]),
                "eligible_samples": len(labels),
                "lineage_groups": int(
                    aligned["lineage_group_id"].nunique()
                ),
            }
        )
    return prediction_rows, metric_rows, selection_rows


def _evaluate_encoders_primary(
    feature_paths: dict[str, Path],
    folds: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for encoder_name in sorted(feature_paths):
        attention = protocol.pooling in ATTENTION_POOLINGS
        if attention:
            token_table = _load_token_table(feature_paths[encoder_name])
        else:
            features = pd.read_csv(feature_paths[encoder_name])
            feature_columns = [
                column for column in features if column.startswith("feat_")
            ]
        for spec in protocol.non_centered_windows:
            kind, duration_ms = parse_non_centered_spec(spec)
            condition_rows, condition_metrics, condition_selections = (
                _evaluate_non_centered_condition(
                    encoder_name,
                    spec,
                    duration_ms if duration_ms is not None else 0,
                    None if attention else features,
                    feature_columns,
                    token_table if attention else None,
                    folds,
                    protocol,
                )
            )
            prediction_rows.extend(condition_rows)
            metric_rows.extend(condition_metrics)
            selection_rows.extend(condition_selections)
        for duration in protocol.resolved_window_conditions:
            window_name = f"event_{duration:03d}ms"
            if attention:
                uids = sorted(
                    uid
                    for (uid, name) in token_table
                    if name == window_name
                )
                aligned = folds[folds["uid"].isin(uids)].sort_values(
                    "uid"
                ).reset_index(drop=True)
                aligned_uids = [str(uid) for uid in aligned["uid"]]
            else:
                selected = features[features["window_name"].eq(window_name)]
                aligned = folds.merge(
                    selected[["uid", *feature_columns]],
                    on="uid",
                    how="left",
                    validate="one_to_one",
                )
                matrix = aligned[feature_columns].to_numpy(dtype=np.float64)
                if not np.isfinite(matrix).all():
                    raise ValueError(
                        f"Missing or non-finite features for "
                        f"{encoder_name}/{window_name}"
                    )
            labels = aligned["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
            groups = aligned["lineage_group_id"].to_numpy(dtype=object)
            scores = np.full(len(aligned), np.nan, dtype=np.float64)
            predictions_by_rule: dict[str, np.ndarray] = {}
            thresholds: dict[int, float] = {}

            def build_matrix(train: np.ndarray) -> np.ndarray:
                if not attention:
                    return matrix
                directions = _attention_fit_directions(
                    [
                        token_table[(aligned_uids[i], window_name)]
                        for i in train
                    ],
                    labels[train],
                    protocol.pooling,
                    protocol.attention_k,
                )
                return np.stack(
                    [
                        _attention_family_pool(
                            token_table[(uid, window_name)],
                            directions,
                            protocol.pooling,
                            protocol.attention_k,
                        )
                        for uid in aligned_uids
                    ]
                )

            for outer_fold in sorted(aligned["outer_fold"].unique()):
                test = np.flatnonzero(
                    aligned["outer_fold"].to_numpy() == outer_fold
                )
                train = np.flatnonzero(
                    aligned["outer_fold"].to_numpy() != outer_fold
                )
                fold_matrix = build_matrix(train)
                scaler = StandardScaler()
                transformed_train = scaler.fit_transform(
                    fold_matrix[train]
                )
                transformed_test = scaler.transform(fold_matrix[test])
                if protocol.c_grid is not None:
                    c_value, records = _select_c_inner(
                        fold_matrix,
                        labels,
                        groups,
                        train,
                        protocol.c_grid,
                        protocol.inner_splits,
                        protocol.seed + int(outer_fold),
                    )
                else:
                    c_value = protocol.logistic_c
                    records = []
                classifier = LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=5_000,
                    random_state=protocol.seed + int(outer_fold),
                )
                classifier.fit(transformed_train, labels[train])
                test_scores = classifier.predict_proba(transformed_test)[:, 1]
                scores[test] = test_scores
                predictions_by_rule.setdefault("fixed_0.5", np.full(
                    len(aligned), -1, dtype=int
                ))[test] = (test_scores >= 0.5).astype(int)
                selection_row = {
                    "encoder": encoder_name,
                    "condition": "event",
                    "window_ms": duration,
                    "outer_fold": int(outer_fold),
                    "selected_C": c_value,
                    "inner_scores_json": json.dumps(
                        records, sort_keys=True
                    ),
                }
                if protocol.calibrate_threshold:
                    threshold, threshold_records = _select_threshold_inner(
                        fold_matrix,
                        labels,
                        groups,
                        train,
                        c_value,
                        protocol.inner_splits,
                        protocol.seed + int(outer_fold),
                    )
                    thresholds[int(outer_fold)] = threshold
                    selection_row["selected_threshold"] = threshold
                    selection_row["threshold_scores_json"] = json.dumps(
                        threshold_records, sort_keys=True
                    )
                    predictions_by_rule.setdefault("calibrated", np.full(
                        len(aligned), -1, dtype=int
                    ))[test] = (test_scores >= threshold).astype(int)
                selection_rows.append(selection_row)

            rules = list(predictions_by_rule)
            for rule in rules:
                rule_predictions = predictions_by_rule[rule]
                if (rule_predictions < 0).any():
                    raise AssertionError(
                        f"Incomplete out-of-fold predictions for "
                        f"{encoder_name}/{window_name}/{rule}"
                    )
                if not np.isfinite(scores).all():
                    raise AssertionError(
                        f"Non-finite scores for {encoder_name}/{window_name}"
                    )
                for position, row in enumerate(aligned.itertuples(index=False)):
                    prediction_rows.append(
                        {
                            "encoder": encoder_name,
                            "condition": "event_selected_event",
                            "window_ms": duration,
                            "decision_rule": rule,
                            "uid": str(row.uid),
                            "label": str(row.label),
                            "lineage_group_id": str(row.lineage_group_id),
                            "outer_fold": int(row.outer_fold),
                            "y_true": int(labels[position]),
                            "y_pred": int(rule_predictions[position]),
                            "score_ground_ball": float(scores[position]),
                        }
                    )

                matrix_counts = confusion_matrix(
                    labels, rule_predictions, labels=[0, 1]
                )
                metric_rows.append(
                    {
                        "encoder": encoder_name,
                        "condition": "event_selected_event",
                        "window_ms": duration,
                        "decision_rule": rule,
                        "primary_metric": "balanced_accuracy",
                        "balanced_accuracy": float(
                            balanced_accuracy_score(labels, rule_predictions)
                        ),
                        "accuracy": float(
                            accuracy_score(labels, rule_predictions)
                        ),
                        "roc_auc": float(roc_auc_score(labels, scores)),
                        "macro_f1": float(
                            f1_score(
                                labels, rule_predictions, average="macro"
                            )
                        ),
                        "true_fly_pred_fly": int(matrix_counts[0, 0]),
                        "true_fly_pred_ground": int(matrix_counts[0, 1]),
                        "true_ground_pred_fly": int(matrix_counts[1, 0]),
                        "true_ground_pred_ground": int(matrix_counts[1, 1]),
                        "eligible_samples": len(labels),
                        "lineage_groups": int(
                            aligned["lineage_group_id"].nunique()
                        ),
                    }
                )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(
            selection_rows,
            columns=[
                "encoder",
                "condition",
                "window_ms",
                "outer_fold",
                "selected_C",
                "inner_scores_json",
                "selected_threshold",
                "threshold_scores_json",
            ],
        ).dropna(axis=1, how="all"),
    )


def _fit_probe(
    matrix: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    c_value: float,
    seed: int,
):
    scaler = StandardScaler()
    transformed_train = scaler.fit_transform(matrix[train])
    classifier = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5_000,
        random_state=seed,
    )
    classifier.fit(transformed_train, labels[train])
    return scaler, classifier


@dataclass(frozen=True)
class _DurationSpec:
    duration_ms: int
    event_name: str
    pre_name: str
    removed_name: str | None
    conditions: tuple[str, ...]


def _duration_specs(protocol: BenchmarkProtocol) -> list[_DurationSpec]:
    specs: list[_DurationSpec] = []
    for duration in protocol.resolved_window_conditions:
        suffix = f"{duration:03d}ms"
        event_name = f"event_{suffix}"
        pre_name = f"pre_{suffix}"
        removed_name = None
        if duration == 200 and protocol.include_controls:
            removed_name = "removed_200ms"
        conditions = (
            CONTROL_CONDITIONS if removed_name else PRE_ONLY_CONDITIONS
        )
        specs.append(
            _DurationSpec(
                duration_ms=duration,
                event_name=event_name,
                pre_name=pre_name,
                removed_name=removed_name,
                conditions=conditions,
            )
        )
    return specs


def _evaluate_encoders_with_controls(
    feature_paths: dict[str, Path],
    folds: pd.DataFrame,
    protocol: BenchmarkProtocol,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []

    for encoder_name in sorted(feature_paths):
        attention = protocol.pooling in ATTENTION_POOLINGS
        if attention:
            token_table = _load_token_table(feature_paths[encoder_name])
        else:
            features = pd.read_csv(feature_paths[encoder_name])
            feature_columns = [
                column for column in features if column.startswith("feat_")
            ]
        for spec in protocol.non_centered_windows:
            kind, duration_ms = parse_non_centered_spec(spec)
            condition_rows, condition_metrics, condition_selections = (
                _evaluate_non_centered_condition(
                    encoder_name,
                    spec,
                    duration_ms if duration_ms is not None else 0,
                    None if attention else features,
                    feature_columns,
                    token_table if attention else None,
                    folds,
                    protocol,
                )
            )
            prediction_rows.extend(condition_rows)
            metric_rows.extend(condition_metrics)
            selection_rows.extend(condition_selections)
        for spec in _duration_specs(protocol):
            roles = attention_control_window_roles(
                spec.event_name,
                spec.pre_name,
                spec.removed_name,
            )
            window_names = list(roles.window_names)
            fit_window_by_condition = roles.fit_window_by_condition
            apply_window_by_condition = roles.apply_window_by_condition
            attention_representation = None
            if attention:
                attention_representation = (
                    AttentionControlRepresentation.from_token_table(
                        token_table, folds, roles
                    )
                )
                paired = attention_representation.paired
            else:
                available: set[str] | None = None
                for window_name in window_names:
                    uids = set(
                        features.loc[
                            features["window_name"].eq(window_name), "uid"
                        ]
                    )
                    available = (
                        uids if available is None else available & uids
                    )
                paired = folds[
                    folds["uid"].isin(available)
                ].sort_values("uid").reset_index(drop=True)
            if len(paired) < 2:
                raise ValueError(
                    f"Too few paired samples for "
                    f"{encoder_name}/{spec.duration_ms}ms"
                )
            labels = paired["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
            groups = paired["lineage_group_id"].to_numpy(dtype=object)
            paired_uids = [str(uid) for uid in paired["uid"]]

            static_window_matrices: dict[str, np.ndarray] = {}
            if not attention:
                for window_name in window_names:
                    selected = features[
                        features["window_name"].eq(window_name)
                    ].set_index("uid")
                    matrix = selected.loc[
                        paired_uids, feature_columns
                    ].to_numpy(dtype=np.float64)
                    if not np.isfinite(matrix).all():
                        raise ValueError(
                            f"Missing or non-finite {window_name} features "
                            f"for {encoder_name}"
                        )
                    static_window_matrices[window_name] = matrix

            def matrices_for_fold(
                train: np.ndarray,
            ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
                if attention_representation is not None:
                    return attention_representation.fold_matrices(
                        train,
                        labels,
                        protocol.pooling,
                        protocol.attention_k,
                    )
                return (
                    static_window_matrices,
                    {
                        condition: static_window_matrices[apply_window]
                        for condition, apply_window in (
                            apply_window_by_condition.items()
                        )
                    },
                )

            fold_scores: dict[str, np.ndarray] = {
                condition: np.full(len(paired), np.nan, dtype=np.float64)
                for condition in spec.conditions
            }
            fold_predictions: dict[tuple[str, str], np.ndarray] = {
                (condition, rule): np.full(len(paired), -1, dtype=int)
                for condition in spec.conditions
                for rule in (
                    ("fixed_0.5", "calibrated")
                    if protocol.calibrate_threshold
                    else ("fixed_0.5",)
                )
            }
            fold_array = paired["outer_fold"].to_numpy(dtype=int)

            for outer_fold in sorted(set(fold_array)):
                test = np.flatnonzero(fold_array == outer_fold)
                train = np.flatnonzero(fold_array != outer_fold)
                fold_seed = protocol.seed + int(outer_fold)
                source_matrices, condition_matrices = matrices_for_fold(train)
                c_values: dict[str, float] = {}
                thresholds: dict[str, float] = {}
                for window_name in window_names:
                    if protocol.c_grid is not None:
                        c_value, records = _select_c_inner(
                            source_matrices[window_name],
                            labels,
                            groups,
                            train,
                            protocol.c_grid,
                            protocol.inner_splits,
                            fold_seed,
                        )
                    else:
                        c_value = protocol.logistic_c
                        records = []
                    c_values[window_name] = c_value
                    selection_row = {
                        "encoder": encoder_name,
                        "condition": window_name.rsplit("_", 1)[0],
                        "window_ms": spec.duration_ms,
                        "outer_fold": int(outer_fold),
                        "selected_C": c_value,
                        "inner_scores_json": json.dumps(
                            records, sort_keys=True
                        ),
                    }
                    if protocol.calibrate_threshold:
                        threshold, threshold_records = _select_threshold_inner(
                            source_matrices[window_name],
                            labels,
                            groups,
                            train,
                            c_value,
                            protocol.inner_splits,
                            fold_seed,
                        )
                        thresholds[window_name] = threshold
                        selection_row["selected_threshold"] = threshold
                        selection_row["threshold_scores_json"] = json.dumps(
                            threshold_records, sort_keys=True
                        )
                    selection_rows.append(selection_row)

                probes: dict[str, object] = {}
                for window_name in window_names:
                    probes[window_name] = _fit_probe(
                        source_matrices[window_name],
                        labels,
                        train,
                        c_values[window_name],
                        fold_seed,
                    )
                event_probe = probes[spec.event_name]

                probe_conditions = (
                    ("event_selected_event", event_probe),
                    ("event_selected_pre", event_probe),
                    ("pre_selected_pre", probes[spec.pre_name]),
                )
                if spec.removed_name is not None:
                    probe_conditions = probe_conditions + (
                        ("event_selected_removed", event_probe),
                        (
                            "removed_selected_removed",
                            probes[spec.removed_name],
                        ),
                    )
                for condition, probe in probe_conditions:
                    transformed = probe[0].transform(
                        condition_matrices[condition][test]
                    )
                    score = probe[1].predict_proba(transformed)[:, 1]
                    fold_scores[condition][test] = score
                    fold_predictions[(condition, "fixed_0.5")][test] = (
                        score >= 0.5
                    ).astype(int)
                    if protocol.calibrate_threshold:
                        fit_window = fit_window_by_condition[condition]
                        fold_predictions[(condition, "calibrated")][test] = (
                            score >= thresholds[fit_window]
                        ).astype(int)

            rules = [
                rule
                for rule in (
                    ("fixed_0.5", "calibrated")
                    if protocol.calibrate_threshold
                    else ("fixed_0.5",)
                )
            ]
            for condition in spec.conditions:
                for rule in rules:
                    rule_predictions = fold_predictions[(condition, rule)]
                    if (
                        (rule_predictions < 0).any()
                        or not np.isfinite(fold_scores[condition]).all()
                    ):
                        raise AssertionError(
                            f"Incomplete control predictions for "
                            f"{encoder_name}/{spec.duration_ms}ms/"
                            f"{condition}/{rule}"
                        )
                    for position, row in enumerate(
                        paired.itertuples(index=False)
                    ):
                        prediction_rows.append(
                            {
                                "encoder": encoder_name,
                                "condition": condition,
                                "window_ms": spec.duration_ms,
                                "decision_rule": rule,
                                "representation_fit_window": (
                                    fit_window_by_condition[condition]
                                ),
                                "representation_apply_window": (
                                    apply_window_by_condition[condition]
                                ),
                                "uid": str(row.uid),
                                "label": str(row.label),
                                "lineage_group_id": str(
                                    row.lineage_group_id
                                ),
                                "outer_fold": int(row.outer_fold),
                                "y_true": int(labels[position]),
                                "y_pred": int(rule_predictions[position]),
                                "score_ground_ball": float(
                                    fold_scores[condition][position]
                                ),
                            }
                        )

                    matrix_counts = confusion_matrix(
                        labels, rule_predictions, labels=[0, 1]
                    )
                    metric_rows.append(
                        {
                            "encoder": encoder_name,
                            "condition": condition,
                            "window_ms": spec.duration_ms,
                            "decision_rule": rule,
                            "primary_metric": "balanced_accuracy",
                            "balanced_accuracy": float(
                                balanced_accuracy_score(
                                    labels, rule_predictions
                                )
                            ),
                            "accuracy": float(
                                accuracy_score(labels, rule_predictions)
                            ),
                            "roc_auc": float(
                                roc_auc_score(
                                    labels, fold_scores[condition]
                                )
                            ),
                            "macro_f1": float(
                                f1_score(
                                    labels, rule_predictions, average="macro"
                                )
                            ),
                            "true_fly_pred_fly": int(matrix_counts[0, 0]),
                            "true_fly_pred_ground": int(matrix_counts[0, 1]),
                            "true_ground_pred_fly": int(matrix_counts[1, 0]),
                            "true_ground_pred_ground": int(
                                matrix_counts[1, 1]
                            ),
                            "eligible_samples": len(paired),
                            "lineage_groups": int(
                                paired["lineage_group_id"].nunique()
                            ),
                        }
                    )

            for rule in rules:
                increment = (
                    float(
                        balanced_accuracy_score(
                            labels, fold_predictions[
                                ("event_selected_event", rule)
                            ]
                        )
                    )
                    - float(
                        balanced_accuracy_score(
                            labels, fold_predictions[
                                ("event_selected_pre", rule)
                            ]
                        )
                    )
                )
                metric_rows.append(
                    {
                        "encoder": encoder_name,
                        "condition": "contact_specific_increment",
                        "window_ms": spec.duration_ms,
                        "decision_rule": rule,
                        "primary_metric": "balanced_accuracy",
                        "balanced_accuracy": increment,
                        "accuracy": float("nan"),
                        "roc_auc": float("nan"),
                        "macro_f1": float("nan"),
                        "true_fly_pred_fly": int(0),
                        "true_fly_pred_ground": int(0),
                        "true_ground_pred_fly": int(0),
                        "true_ground_pred_ground": int(0),
                        "eligible_samples": len(paired),
                        "lineage_groups": int(
                            paired["lineage_group_id"].nunique()
                        ),
                    }
                )

    return (
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(
            selection_rows,
            columns=[
                "encoder",
                "condition",
                "window_ms",
                "outer_fold",
                "selected_C",
                "inner_scores_json",
                "selected_threshold",
                "threshold_scores_json",
            ],
        ).dropna(axis=1, how="all"),
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_short_contact_benchmark(
    protocol: BenchmarkProtocol,
    snapshot: DatasetSnapshot,
    encoder_adapters: Sequence[EncoderAdapter],
    output_dir: Path,
) -> ArtifactBundle:
    """Run the locked short-contact benchmark and return its artifacts."""

    _validate_protocol(protocol)
    audit, snapshot_fingerprint = _audit_snapshot(snapshot)
    adapters, encoder_documents = _encoder_documents(encoder_adapters)
    for adapter in adapters:
        if (
            adapter.provenance.name.startswith("beats")
            and min(protocol.resolved_window_conditions)
            < MIN_BEATS_WINDOW_MS
        ):
            raise ValueError(
                f"{adapter.provenance.name} supports only "
                f"{MIN_BEATS_WINDOW_MS} ms windows; got "
                f"{protocol.resolved_window_conditions}"
            )
        if adapter.provenance.name.startswith("beats"):
            for spec in protocol.non_centered_windows:
                _kind, duration_ms = parse_non_centered_spec(spec)
                if duration_ms is not None and duration_ms < MIN_BEATS_WINDOW_MS:
                    raise ValueError(
                        f"{adapter.provenance.name} supports only "
                        f"{MIN_BEATS_WINDOW_MS} ms windows; got "
                        f"non-centered spec {spec!r}"
                    )
    protocol_document = _protocol_document(
        protocol,
        snapshot,
        snapshot_fingerprint,
        encoder_documents,
    )
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = Path(output_dir).resolve() / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    protocol_path = bundle_root / "protocol.json"
    audit_path = bundle_root / "snapshot_audit.json"
    _write_json(protocol_path, {"artifact_id": artifact_id, **protocol_document})
    _write_json(audit_path, audit)

    windows, exclusions = _prepare_contact_windows(snapshot, bundle_root, protocol)
    windows_path = bundle_root / "windows_manifest.csv"
    exclusions_path = bundle_root / "exclusions.csv"
    feature_paths = _extract_features(
        adapters,
        windows,
        windows_path,
        bundle_root,
        protocol,
        snapshot_fingerprint,
    )
    _validate_requested_layers(
        feature_paths,
        protocol,
        [adapter.provenance.name for adapter in adapters],
    )

    eligible_uids = set(windows["uid"])
    folds = _make_fold_assignments(snapshot, protocol, eligible_uids)
    folds_path = bundle_root / "fold_assignments.csv"
    folds.to_csv(folds_path, index=False)
    predictions, metrics, selections = _evaluate_encoders(
        feature_paths, folds, protocol
    )
    predictions_path = bundle_root / "oof_predictions.csv"
    metrics_path = bundle_root / "metrics.csv"
    selections_path = bundle_root / "selections.csv"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    selections.to_csv(selections_path, index=False)

    artifact_paths: dict[str, Path] = {
        "protocol": protocol_path,
        "snapshot_audit": audit_path,
        "window_manifest": windows_path,
        "exclusions": exclusions_path,
        "fold_assignments": folds_path,
        "oof_predictions": predictions_path,
        "metrics": metrics_path,
        "selections": selections_path,
    }
    artifact_paths.update(
        {
            f"features/{name}": path
            for name, path in sorted(feature_paths.items())
        }
    )
    bundle_manifest_path = bundle_root / "artifact_bundle.json"
    manifest_document = {
        "artifact_id": artifact_id,
        "artifacts": {
            name: {
                "path": path.relative_to(bundle_root).as_posix(),
                "sha256": _file_sha256(path),
            }
            for name, path in sorted(artifact_paths.items())
        },
    }
    _write_json(bundle_manifest_path, manifest_document)
    artifact_paths["artifact_bundle"] = bundle_manifest_path

    return ArtifactBundle(
        artifact_id=artifact_id,
        root=bundle_root,
        _artifacts=tuple(sorted(artifact_paths.items())),
    )
