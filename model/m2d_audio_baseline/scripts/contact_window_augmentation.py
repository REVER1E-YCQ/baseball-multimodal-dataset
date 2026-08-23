from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

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
    _inner_splits,
    _make_estimator,
    _probe_document,
    _validated_source,
    ProbeConfig,
)
from .paired_contrast_evaluation import (
    LOCKED_FOLD_SEEDS,
    LOCKED_MINIMUM_HEADLINE_BA_GAIN,
    _seeded_folds,
)
from .prepare_windows import exact_slice, to_float_mono
from .short_contact_benchmark import (
    ArtifactBundle,
    DatasetSnapshot,
    EncoderAdapter,
    LABEL_TO_INT,
    SnapshotSample,
    _audit_snapshot,
    _canonical_sha256,
    _file_sha256,
    _write_json,
)


CONTACT_AUGMENTATION_PROTOCOL_VERSION = "contact-window-augmentation-v1"
LOCKED_AUGMENTATION_SEED = 20260805
ARMS = (
    "no_augmentation",
    "time_jitter",
    "gain",
    "light_eq",
    "combined",
)
AUGMENTED_ARMS = ARMS[1:]
CONDITIONS = (
    "event",
    "strict_pre",
    "transient_removed",
    "imposed_shift_minus_20ms",
    "imposed_shift_plus_20ms",
)
WINDOW_BY_CONDITION = {
    "event": "event_200ms",
    "strict_pre": "pre_200ms",
    "transient_removed": "removed_200ms",
}
JITTER_LIMIT_MS = 20
GAIN_DB_RANGE = (-6.0, -1.0)
EQ_CUTOFF_HZ_RANGE = (3_000.0, 6_000.0)
EQ_ATTENUATION_DB_RANGE = (-6.0, -3.0)
AGGREGATION_POLICY = "append_original_and_one_derivative_equal_source_weight"


class ContactWindowAugmentationError(RuntimeError):
    """Raised when augmentation evaluation cannot preserve its protocol."""


@dataclass(frozen=True)
class ContactWindowAugmentationConfig:
    """Locked train-only augmentation and uncertainty policy."""

    fold_seeds: tuple[int, ...] = LOCKED_FOLD_SEEDS
    augmentation_seed: int = LOCKED_AUGMENTATION_SEED
    n_bootstrap: int = 2000
    seed: int = 20260805
    minimum_headline_ba_gain: float = LOCKED_MINIMUM_HEADLINE_BA_GAIN


def _validate_config(
    config: ContactWindowAugmentationConfig,
) -> dict[str, object]:
    if tuple(config.fold_seeds) != LOCKED_FOLD_SEEDS:
        raise ValueError(
            "fold_seeds are locked at (20260805, 20260806, 20260807)"
        )
    if config.augmentation_seed != LOCKED_AUGMENTATION_SEED:
        raise ValueError("augmentation_seed is locked at 20260805")
    if config.n_bootstrap < 20:
        raise ValueError("n_bootstrap must be at least 20")
    if not isinstance(config.seed, int):
        raise ValueError("seed must be an integer")
    gain = float(config.minimum_headline_ba_gain)
    if not math.isfinite(gain) or not math.isclose(
        gain,
        LOCKED_MINIMUM_HEADLINE_BA_GAIN,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError("minimum_headline_ba_gain is locked at 0.02")
    return {
        "fold_seeds": list(LOCKED_FOLD_SEEDS),
        "augmentation_seed": LOCKED_AUGMENTATION_SEED,
        "n_bootstrap": int(config.n_bootstrap),
        "bootstrap_seed": int(config.seed),
        "minimum_headline_ba_gain": gain,
    }


def _locked_probe() -> tuple[dict[str, object], list[dict[str, float | str]]]:
    document = _probe_document(
        ProbeConfig(
            name="contact-augmentation-logistic",
            estimator_family="balanced_l2_logistic_regression",
            hyperparameter_grid={"C": (0.001, 0.01, 0.1)},
            score_output="probability_ground_ball",
            fixed_decision_threshold=0.5,
            calibrate_threshold=False,
        )
    )
    return document, _candidate_parameters(document)


def _encoder_document(encoder: EncoderAdapter) -> dict[str, object]:
    provenance = encoder.provenance
    return {
        "name": provenance.name,
        "upstream_revision": provenance.upstream_revision,
        "checkpoint_sha256": provenance.checkpoint_sha256,
        "precision": provenance.precision,
        "token_dimension": int(provenance.token_dimension),
        "training_epochs": int(provenance.training_epochs),
    }


def _validate_encoder(
    encoder: EncoderAdapter,
    source_protocol: dict[str, object],
) -> dict[str, object]:
    supplied = _encoder_document(encoder)
    source_encoders = source_protocol.get("encoders")
    if not isinstance(source_encoders, list) or len(source_encoders) != 1:
        raise ContactWindowAugmentationError(
            "Source protocol must declare exactly one encoder"
        )
    expected = source_encoders[0]
    if not isinstance(expected, dict):
        raise ContactWindowAugmentationError("Source encoder role is malformed")
    fields = (
        "name",
        "upstream_revision",
        "checkpoint_sha256",
        "precision",
        "token_dimension",
        "training_epochs",
    )
    if any(supplied[field] != expected.get(field) for field in fields):
        raise ContactWindowAugmentationError(
            "Augmentation encoder identity does not match the source encoder"
        )
    if supplied["name"] != M2D_ENCODER_NAME:
        raise ContactWindowAugmentationError("Only the locked M2D encoder is allowed")
    return supplied


def _source_artifact_path(source_root: Path, artifact_name: str) -> Path:
    manifest_path = source_root / "artifact_bundle.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["artifacts"][artifact_name]
        path = source_root / str(record["path"])
        expected_sha = str(record["sha256"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ContactWindowAugmentationError(
            f"Source manifest is missing {artifact_name}"
        ) from error
    if not path.is_file() or _file_sha256(path) != expected_sha:
        raise ContactWindowAugmentationError(
            f"Source artifact failed checksum: {artifact_name}"
        )
    return path


def _validated_population(
    source_root: Path,
    source_protocol: dict[str, object],
    feature_path: Path,
    snapshot: DatasetSnapshot,
) -> tuple[
    AttentionControlRepresentation,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, SnapshotSample],
    dict[str, object],
]:
    folds = pd.read_csv(source_root / "fold_assignments.csv")
    roles = attention_control_window_roles(
        "event_200ms", "pre_200ms", "removed_200ms"
    )
    try:
        representation = AttentionControlRepresentation.from_token_table(
            load_token_table(feature_path), folds, roles
        )
    except ValueError as error:
        raise ContactWindowAugmentationError(str(error)) from error
    if not representation.paired_uids:
        raise ContactWindowAugmentationError(
            "No exact event/Pre/removed pairs are available"
        )
    dataset = source_protocol.get("dataset")
    if not isinstance(dataset, dict) or snapshot.revision != dataset.get("revision"):
        raise ContactWindowAugmentationError(
            "Snapshot revision does not match the source protocol"
        )
    _snapshot_audit, snapshot_fingerprint = _audit_snapshot(snapshot)
    if snapshot_fingerprint != dataset.get("snapshot_fingerprint"):
        raise ContactWindowAugmentationError(
            "Snapshot fingerprint does not match the locked source protocol"
        )
    snapshot_by_uid = {sample.uid: sample for sample in snapshot.samples}
    if len(snapshot_by_uid) != len(snapshot.samples):
        raise ContactWindowAugmentationError("Snapshot repeats a uid")
    paired = representation.paired.set_index("uid")
    missing = set(representation.paired_uids) - set(snapshot_by_uid)
    if missing:
        raise ContactWindowAugmentationError(
            f"Snapshot is missing paired uids: {sorted(missing)[:3]}"
        )

    audio_records: list[dict[str, str]] = []
    sample_map: dict[str, SnapshotSample] = {}
    for uid in representation.paired_uids:
        sample = snapshot_by_uid[uid]
        expected = paired.loc[uid]
        if (
            sample.label != expected["label"]
            or sample.lineage_group_id != expected["lineage_group_id"]
        ):
            raise ContactWindowAugmentationError(
                f"Snapshot label or lineage mismatch for {uid}"
            )
        audio_path = Path(sample.audio_path).resolve()
        if not audio_path.is_file():
            raise ContactWindowAugmentationError(
                f"Snapshot audio is missing for {uid}: {audio_path}"
            )
        audio_records.append({"uid": uid, "sha256": _file_sha256(audio_path)})
        sample_map[uid] = sample
    snapshot_audio_fingerprint = _canonical_sha256(audio_records)

    manifest_path = _source_artifact_path(source_root, "window_manifest")
    windows = pd.read_csv(manifest_path)
    event_windows = windows[
        (windows["window_name"] == "event_200ms")
        & windows["uid"].astype(str).isin(representation.paired_uids)
    ].copy()
    if event_windows["uid"].duplicated().any() or len(event_windows) != len(paired):
        raise ContactWindowAugmentationError(
            "Source window manifest does not have one event window per pair"
        )
    if (
        event_windows["wav_boundary_padding_samples"].fillna(0).astype(int) != 0
    ).any():
        raise ContactWindowAugmentationError(
            "Augmentation requires source windows without waveform padding"
        )
    event_windows["window_path"] = event_windows["window_path"].map(
        lambda value: str((source_root / str(value)).resolve())
    )
    event_windows = event_windows.set_index("uid").loc[
        list(representation.paired_uids)
    ].reset_index()
    return (
        representation,
        folds,
        event_windows,
        sample_map,
        {
            "snapshot_revision": snapshot.revision,
            "snapshot_audio_fingerprint": snapshot_audio_fingerprint,
            "audio_identity": "sha256_file_content",
            "n_snapshot_audio_files": len(audio_records),
        },
    )


def _rng_for(
    augmentation_seed: int,
    fold_seed: int,
    recipe: str,
    uid: str,
) -> np.random.Generator:
    payload = f"{augmentation_seed}|{fold_seed}|{recipe}|{uid}".encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(seed)


def _read_waveform(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, raw = wavfile.read(path)
    return int(sample_rate), to_float_mono(raw)


def _shifted_window(
    full_waveform: np.ndarray,
    sample_rate: int,
    window_start: float,
    shift_ms: float,
) -> np.ndarray:
    return exact_slice(
        full_waveform,
        sample_rate,
        window_start + shift_ms / 1000.0,
        0.2,
    )


def _light_eq(
    waveform: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
    attenuation_db: float,
) -> np.ndarray:
    frequencies = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0
    transition = np.clip(
        (frequencies - cutoff_hz) / max(nyquist - cutoff_hz, 1.0),
        0.0,
        1.0,
    )
    smooth = transition * transition * (3.0 - 2.0 * transition)
    response = np.power(10.0, (attenuation_db * smooth) / 20.0)
    filtered = np.fft.irfft(
        np.fft.rfft(waveform.astype(np.float64)) * response,
        n=len(waveform),
    )
    return filtered.astype(np.float32)


def _make_derivative(
    recipe: str,
    uid: str,
    fold_seed: int,
    augmentation_seed: int,
    event_waveform: np.ndarray,
    full_waveform: np.ndarray,
    sample_rate: int,
    window_start: float,
) -> tuple[np.ndarray, dict[str, object]]:
    rng = _rng_for(augmentation_seed, fold_seed, recipe, uid)
    result = event_waveform.copy()
    jitter_ms = 0.0
    gain_db = 0.0
    cutoff_hz = float("nan")
    attenuation_db = 0.0
    if recipe in ("time_jitter", "combined"):
        jitter_samples_limit = int(round(JITTER_LIMIT_MS * sample_rate / 1000))
        jitter_samples = int(
            rng.integers(-jitter_samples_limit, jitter_samples_limit + 1)
        )
        jitter_ms = jitter_samples * 1000.0 / sample_rate
        result = _shifted_window(
            full_waveform, sample_rate, window_start, jitter_ms
        )
    if recipe in ("gain", "combined"):
        gain_db = float(rng.uniform(*GAIN_DB_RANGE))
        result = result * np.float32(10.0 ** (gain_db / 20.0))
    if recipe in ("light_eq", "combined"):
        cutoff_hz = float(rng.uniform(*EQ_CUTOFF_HZ_RANGE))
        attenuation_db = float(rng.uniform(*EQ_ATTENUATION_DB_RANGE))
        result = _light_eq(result, sample_rate, cutoff_hz, attenuation_db)
    if recipe not in AUGMENTED_ARMS:
        raise AssertionError(f"Unknown augmentation recipe: {recipe}")
    result = np.asarray(result, dtype=np.float32)
    if len(result) != len(event_waveform) or not np.isfinite(result).all():
        raise ContactWindowAugmentationError(
            f"Augmentation produced invalid duration or values for {uid}/{recipe}"
        )
    return result, {
        "jitter_ms": jitter_ms,
        "gain_db": gain_db,
        "eq_cutoff_hz": cutoff_hz,
        "eq_attenuation_db": attenuation_db,
    }


def _encode_cache(
    representation: AttentionControlRepresentation,
    event_windows: pd.DataFrame,
    sample_map: dict[str, SnapshotSample],
    encoder: EncoderAdapter,
    config: ContactWindowAugmentationConfig,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict[tuple[object, ...], int]]:
    event_by_uid = event_windows.set_index("uid")
    token_rows: list[np.ndarray] = []
    index_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    key_to_index: dict[tuple[object, ...], int] = {}
    expected_token_shape: tuple[int, int] | None = None

    def append(
        key: tuple[object, ...],
        waveform: np.ndarray,
        sample_rate: int,
        role: str,
        recipe: str,
        fold_seed: int | str,
        uid: str,
        parameters: dict[str, object],
    ) -> None:
        nonlocal expected_token_shape
        tokens = np.asarray(
            encoder.encode_tokens(waveform, sample_rate), dtype=np.float32
        )
        if tokens.ndim != 2 or not np.isfinite(tokens).all():
            raise ContactWindowAugmentationError(
                f"Encoder returned invalid tokens for {key}"
            )
        if expected_token_shape is None:
            expected_token_shape = tokens.shape
        if tokens.shape != expected_token_shape:
            raise ContactWindowAugmentationError(
                "Augmented fixed-duration windows produced unequal token shapes"
            )
        cache_index = len(token_rows)
        token_rows.append(tokens)
        key_to_index[key] = cache_index
        index_rows.append(
            {
                "cache_index": cache_index,
                "cache_role": role,
                "recipe": recipe,
                "fold_seed": str(fold_seed),
                "uid": uid,
            }
        )
        audit_rows.append(
            {
                "cache_index": cache_index,
                "cache_role": role,
                "recipe": recipe,
                "fold_seed": str(fold_seed),
                "uid": uid,
                "expected_duration_samples": int(round(0.2 * sample_rate)),
                "output_duration_samples": int(len(waveform)),
                "waveform_padding_samples": 0,
                "project_label_visible": False,
                **parameters,
            }
        )

    for fold_seed in LOCKED_FOLD_SEEDS:
        for recipe in AUGMENTED_ARMS:
            for uid in representation.paired_uids:
                row = event_by_uid.loc[uid]
                event_rate, event_waveform = _read_waveform(
                    Path(str(row["window_path"]))
                )
                sample = sample_map[uid]
                full_rate, full_waveform = _read_waveform(Path(sample.audio_path))
                if full_rate != event_rate:
                    raise ContactWindowAugmentationError(
                        f"Snapshot/window sample-rate mismatch for {uid}"
                    )
                derivative, parameters = _make_derivative(
                    recipe,
                    uid,
                    fold_seed,
                    config.augmentation_seed,
                    event_waveform,
                    full_waveform,
                    event_rate,
                    float(row["window_start"]),
                )
                append(
                    ("derivative", fold_seed, recipe, uid),
                    derivative,
                    event_rate,
                    "outer_train_derivative",
                    recipe,
                    fold_seed,
                    uid,
                    parameters,
                )
    for shift_ms, recipe in (
        (-JITTER_LIMIT_MS, "imposed_shift_minus_20ms"),
        (JITTER_LIMIT_MS, "imposed_shift_plus_20ms"),
    ):
        for uid in representation.paired_uids:
            row = event_by_uid.loc[uid]
            sample = sample_map[uid]
            full_rate, full_waveform = _read_waveform(Path(sample.audio_path))
            shifted = _shifted_window(
                full_waveform,
                full_rate,
                float(row["window_start"]),
                float(shift_ms),
            )
            append(
                ("robustness", recipe, uid),
                shifted,
                full_rate,
                "outer_test_imposed_shift_diagnostic",
                recipe,
                "shared",
                uid,
                {
                    "jitter_ms": float(shift_ms),
                    "gain_db": 0.0,
                    "eq_cutoff_hz": float("nan"),
                    "eq_attenuation_db": 0.0,
                },
            )
    if not token_rows:
        raise ContactWindowAugmentationError("Augmentation cache is empty")
    return (
        np.stack(token_rows),
        pd.DataFrame(index_rows),
        pd.DataFrame(audit_rows),
        key_to_index,
    )


def _load_token_cache(
    bundle_root: Path,
    protocol_document: dict[str, object],
    artifact_id: str,
) -> tuple[
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[object, ...], int],
] | None:
    protocol_path = bundle_root / "protocol.json"
    manifest_path = bundle_root / "artifact_bundle.json"
    cache_path = bundle_root / "augmented_tokens.npy"
    index_path = bundle_root / "augmented_token_index.csv"
    audit_path = bundle_root / "waveform_audit.csv"
    required = (protocol_path, manifest_path, cache_path, index_path, audit_path)
    if not any(path.exists() for path in required):
        return None
    if not all(path.is_file() for path in required):
        raise ContactWindowAugmentationError(
            "Existing augmentation cache is incomplete"
        )
    try:
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContactWindowAugmentationError(
            "Existing augmentation cache metadata is unreadable"
        ) from error
    expected_protocol = {"artifact_id": artifact_id, **protocol_document}
    if (
        existing_protocol != expected_protocol
        or manifest.get("artifact_id") != artifact_id
    ):
        raise ContactWindowAugmentationError(
            "Existing augmentation cache provenance does not match this run"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ContactWindowAugmentationError(
            "Existing augmentation cache manifest is malformed"
        )
    for name, path in (
        ("augmented_tokens", cache_path),
        ("augmented_token_index", index_path),
        ("waveform_audit", audit_path),
    ):
        record = artifacts.get(name)
        if (
            not isinstance(record, dict)
            or record.get("path") != path.name
            or record.get("sha256") != _file_sha256(path)
        ):
            raise ContactWindowAugmentationError(
                f"Existing augmentation cache failed checksum: {name}"
            )
    token_cache = np.load(cache_path, allow_pickle=False)
    token_index = pd.read_csv(index_path, dtype={"fold_seed": str})
    waveform_audit = pd.read_csv(audit_path, dtype={"fold_seed": str})
    expected_indices = list(range(len(token_index)))
    if (
        token_cache.ndim != 3
        or token_index["cache_index"].astype(int).tolist() != expected_indices
        or waveform_audit["cache_index"].astype(int).tolist() != expected_indices
        or len(token_cache) != len(token_index)
        or not np.isfinite(token_cache).all()
    ):
        raise ContactWindowAugmentationError(
            "Existing augmentation cache arrays and indices are inconsistent"
        )
    key_to_index: dict[tuple[object, ...], int] = {}
    for row in token_index.itertuples(index=False):
        if row.cache_role == "outer_train_derivative":
            key = (
                "derivative",
                int(row.fold_seed),
                str(row.recipe),
                str(row.uid),
            )
        elif row.cache_role == "outer_test_imposed_shift_diagnostic":
            key = ("robustness", str(row.recipe), str(row.uid))
        else:
            raise ContactWindowAugmentationError(
                f"Existing augmentation cache has unknown role: {row.cache_role}"
            )
        if key in key_to_index:
            raise ContactWindowAugmentationError(
                f"Existing augmentation cache repeats a key: {key}"
            )
        key_to_index[key] = int(row.cache_index)
    return token_cache, token_index, waveform_audit, key_to_index


def _augmentation_assignments(
    seeded_folds: pd.DataFrame,
    key_to_index: dict[tuple[object, ...], int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_seed in LOCKED_FOLD_SEEDS:
        seed_rows = seeded_folds[
            seeded_folds["fold_seed"] == fold_seed
        ]
        for outer_fold in sorted(seed_rows["outer_fold"].unique()):
            training = seed_rows[seed_rows["outer_fold"] != outer_fold]
            for recipe in AUGMENTED_ARMS:
                for row in training.itertuples(index=False):
                    rows.append(
                        {
                            "fold_seed": int(fold_seed),
                            "outer_fold": int(outer_fold),
                            "recipe": recipe,
                            "uid": str(row.uid),
                            "lineage_group_id": str(row.lineage_group_id),
                            "split_role": "outer_train",
                            "cache_index": key_to_index[
                                ("derivative", fold_seed, recipe, str(row.uid))
                            ],
                            "source_weight": 0.5,
                        }
                    )
    return pd.DataFrame(rows)


def _pooled_matrix(
    representation: AttentionControlRepresentation,
    window_name: str,
    directions: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            pool_attention_tokens(
                representation.token_table[(uid, window_name)],
                directions,
                "attention",
                3,
            )
            for uid in representation.paired_uids
        ]
    )


def _pooled_cache_matrix(
    token_cache: np.ndarray,
    indices: list[int],
    directions: np.ndarray,
) -> np.ndarray:
    return np.stack(
        [
            pool_attention_tokens(
                token_cache[index], directions, "attention", 3
            )
            for index in indices
        ]
    )


def _select_augmented_candidate(
    original: np.ndarray,
    derivative: np.ndarray | None,
    labels: np.ndarray,
    groups: np.ndarray,
    train: np.ndarray,
    candidates: list[dict[str, float | str]],
    inner_splits: int,
    seed: int,
) -> tuple[dict[str, float | str], list[dict[str, object]]]:
    scores_by_candidate = [[] for _ in candidates]
    for inner_fold, (inner_train, inner_validation) in enumerate(
        _inner_splits(labels, groups, train, inner_splits, seed)
    ):
        if derivative is None:
            train_matrix = original[inner_train]
            train_labels = labels[inner_train]
            weights = None
        else:
            train_matrix = np.concatenate(
                [original[inner_train], derivative[inner_train]], axis=0
            )
            train_labels = np.concatenate(
                [labels[inner_train], labels[inner_train]]
            )
            weights = np.full(len(train_matrix), 0.5, dtype=np.float64)
        scaler = StandardScaler()
        if weights is None:
            transformed_train = scaler.fit_transform(train_matrix)
        else:
            transformed_train = scaler.fit_transform(
                train_matrix, sample_weight=weights
            )
        transformed_validation = scaler.transform(original[inner_validation])
        for candidate_index, candidate in enumerate(candidates):
            estimator = _make_estimator(
                "balanced_l2_logistic_regression",
                candidate,
                seed + inner_fold,
            )
            if weights is None:
                estimator.fit(transformed_train, train_labels)
            else:
                estimator.fit(
                    transformed_train,
                    train_labels,
                    sample_weight=weights,
                )
            scores = _estimator_scores(
                estimator,
                transformed_validation,
                "probability_ground_ball",
            )
            scores_by_candidate[candidate_index].append(
                float(
                    balanced_accuracy_score(
                        labels[inner_validation], (scores >= 0.5).astype(int)
                    )
                )
            )
    records = [
        {
            "parameters": candidate,
            "inner_balanced_accuracy": float(np.mean(scores)),
            "inner_balanced_accuracy_std": float(np.std(scores)),
        }
        for candidate, scores in zip(candidates, scores_by_candidate, strict=True)
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


def _fit_augmented_probe(
    original: np.ndarray,
    derivative: np.ndarray | None,
    labels: np.ndarray,
    train: np.ndarray,
    parameters: dict[str, float | str],
    seed: int,
):
    if derivative is None:
        train_matrix = original[train]
        train_labels = labels[train]
        weights = None
    else:
        train_matrix = np.concatenate(
            [original[train], derivative[train]], axis=0
        )
        train_labels = np.concatenate([labels[train], labels[train]])
        weights = np.full(len(train_matrix), 0.5, dtype=np.float64)
    scaler = StandardScaler()
    if weights is None:
        transformed = scaler.fit_transform(train_matrix)
    else:
        transformed = scaler.fit_transform(
            train_matrix, sample_weight=weights
        )
    estimator = _make_estimator(
        "balanced_l2_logistic_regression", parameters, seed
    )
    if weights is None:
        estimator.fit(transformed, train_labels)
    else:
        estimator.fit(transformed, train_labels, sample_weight=weights)
    return scaler, estimator


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
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
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
    token_cache: np.ndarray,
    key_to_index: dict[tuple[object, ...], int],
    candidates: list[dict[str, float | str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    paired = representation.paired
    labels = paired["label"].map(LABEL_TO_INT).to_numpy(dtype=int)
    groups = paired["lineage_group_id"].astype(str).to_numpy(dtype=object)
    inner_splits = int(source_protocol["classifier"]["inner_splits"])
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    fit_audit = {
        "attention_fits": 0,
        "model_selection_fits": 0,
        "outer_probe_fits": 0,
    }

    for fold_seed in LOCKED_FOLD_SEEDS:
        fold_rows = seeded_folds[
            seeded_folds["fold_seed"] == fold_seed
        ].set_index("uid")
        fold_array = np.asarray(
            [fold_rows.loc[uid, "outer_fold"] for uid in representation.paired_uids],
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
            for arm in ARMS:
                direction_tokens = [
                    representation.token_table[
                        (representation.paired_uids[position], "event_200ms")
                    ]
                    for position in train
                ]
                direction_labels = list(labels[train])
                derivative_indices: list[int] | None = None
                if arm != "no_augmentation":
                    derivative_indices = [
                        key_to_index[
                            (
                                "derivative",
                                fold_seed,
                                arm,
                                representation.paired_uids[position],
                            )
                        ]
                        for position in range(len(paired))
                    ]
                    direction_tokens.extend(
                        token_cache[derivative_indices[position]]
                        for position in train
                    )
                    direction_labels.extend(labels[train])
                directions = fit_attention_directions(
                    direction_tokens,
                    np.asarray(direction_labels, dtype=int),
                    "attention",
                    int(source_protocol.get("attention_k", 3)),
                )
                fit_audit["attention_fits"] += 1
                original = _pooled_matrix(
                    representation, "event_200ms", directions
                )
                derivative = (
                    None
                    if derivative_indices is None
                    else _pooled_cache_matrix(
                        token_cache, derivative_indices, directions
                    )
                )
                condition_matrices = {
                    "event": original,
                    "strict_pre": _pooled_matrix(
                        representation, "pre_200ms", directions
                    ),
                    "transient_removed": _pooled_matrix(
                        representation, "removed_200ms", directions
                    ),
                    "imposed_shift_minus_20ms": _pooled_cache_matrix(
                        token_cache,
                        [
                            key_to_index[
                                ("robustness", "imposed_shift_minus_20ms", uid)
                            ]
                            for uid in representation.paired_uids
                        ],
                        directions,
                    ),
                    "imposed_shift_plus_20ms": _pooled_cache_matrix(
                        token_cache,
                        [
                            key_to_index[("robustness", "imposed_shift_plus_20ms", uid)]
                            for uid in representation.paired_uids
                        ],
                        directions,
                    ),
                }
                selected, candidate_records = _select_augmented_candidate(
                    original,
                    derivative,
                    labels,
                    groups,
                    train,
                    candidates,
                    inner_splits,
                    fold_seed_value,
                )
                fit_audit["model_selection_fits"] += inner_splits * len(candidates)
                scaler, estimator = _fit_augmented_probe(
                    original,
                    derivative,
                    labels,
                    train,
                    selected,
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
                            "outer_train_augmented_fit_inner_lineage_grouped_"
                            "unaugmented_validation"
                        ),
                    }
                )
                for condition, matrix in condition_matrices.items():
                    scores = _estimator_scores(
                        estimator,
                        scaler.transform(matrix[test]),
                        "probability_ground_ball",
                    )
                    predictions = (scores >= 0.5).astype(int)
                    score_arrays[(arm, condition)][test] = scores
                    prediction_arrays[(arm, condition)][test] = predictions

        for arm in ARMS:
            for condition in CONDITIONS:
                scores = score_arrays[(arm, condition)]
                predictions = prediction_arrays[(arm, condition)]
                if not np.isfinite(scores).all() or (predictions < 0).any():
                    raise ContactWindowAugmentationError(
                        f"Incomplete OOF values for {fold_seed}/{arm}/{condition}"
                    )
                metric_rows.append(
                    _metric_row(
                        arm,
                        fold_seed,
                        condition,
                        labels,
                        predictions,
                        scores,
                        groups,
                    )
                )
                for position, row in enumerate(paired.itertuples(index=False)):
                    prediction_rows.append(
                        {
                            "arm": arm,
                            "fold_seed": int(fold_seed),
                            "condition": condition,
                            "uid": str(row.uid),
                            "label": str(row.label),
                            "lineage_group_id": str(row.lineage_group_id),
                            "outer_fold": int(fold_array[position]),
                            "y_true": int(labels[position]),
                            "y_pred": int(predictions[position]),
                            "score_ground_ball": float(scores[position]),
                        }
                    )
            event_ba = float(
                balanced_accuracy_score(
                    labels, prediction_arrays[(arm, "event")]
                )
            )
            pre_ba = float(
                balanced_accuracy_score(
                    labels, prediction_arrays[(arm, "strict_pre")]
                )
            )
            metric_rows.append(
                {
                    "arm": arm,
                    "fold_seed": int(fold_seed),
                    "condition": "contact_specific_increment",
                    "balanced_accuracy": event_ba - pre_ba,
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
        pd.DataFrame(prediction_rows),
        pd.DataFrame(metric_rows),
        pd.DataFrame(selection_rows),
        fit_audit,
    )


def _paired_differences(
    predictions: pd.DataFrame,
    config: ContactWindowAugmentationConfig,
) -> pd.DataFrame:
    reference = (
        predictions[
            (predictions["arm"] == "no_augmentation")
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
            [
                positions_by_group[group]
                for group in rng.choice(
                    unique_groups, size=len(unique_groups), replace=True
                )
            ]
        )
        for _ in range(config.n_bootstrap)
    ]
    vectors: dict[tuple[int, str, str], np.ndarray] = {}
    for fold_seed in LOCKED_FOLD_SEEDS:
        for arm in ARMS:
            for condition in CONDITIONS:
                rows = (
                    predictions[
                        (predictions["fold_seed"] == fold_seed)
                        & (predictions["arm"] == arm)
                        & (predictions["condition"] == condition)
                    ]
                    .set_index("uid")
                    .loc[uids]
                )
                if not np.array_equal(
                    rows["y_true"].to_numpy(dtype=int), labels
                ):
                    raise ContactWindowAugmentationError(
                        "Paired predictions do not share labels and membership"
                    )
                vectors[(fold_seed, arm, condition)] = rows["y_pred"].to_numpy(
                    dtype=int
                )

    def ba(
        fold_seed: int,
        arm: str,
        condition: str,
        positions: np.ndarray,
    ) -> float:
        return float(
            balanced_accuracy_score(
                labels[positions],
                vectors[(fold_seed, arm, condition)][positions],
            )
        )

    def measure(
        fold_seed: int,
        arm: str,
        comparison: str,
        positions: np.ndarray,
    ) -> float:
        if comparison == "event_ba_gain":
            return ba(fold_seed, arm, "event", positions) - ba(
                fold_seed, "no_augmentation", "event", positions
            )
        if comparison == "strict_pre_ba_change":
            return ba(fold_seed, arm, "strict_pre", positions) - ba(
                fold_seed, "no_augmentation", "strict_pre", positions
            )
        if comparison == "transient_removed_ba_change":
            return ba(fold_seed, arm, "transient_removed", positions) - ba(
                fold_seed, "no_augmentation", "transient_removed", positions
            )
        if comparison == "contact_specific_increment_gain":
            candidate = ba(fold_seed, arm, "event", positions) - ba(
                fold_seed, arm, "strict_pre", positions
            )
            baseline = ba(
                fold_seed, "no_augmentation", "event", positions
            ) - ba(fold_seed, "no_augmentation", "strict_pre", positions)
            return candidate - baseline
        if comparison == "imposed_shift_robustness_gain":
            candidate = np.mean(
                [
                    ba(fold_seed, arm, condition, positions)
                    for condition in (
                        "imposed_shift_minus_20ms",
                        "imposed_shift_plus_20ms",
                    )
                ]
            )
            baseline = np.mean(
                [
                    ba(fold_seed, "no_augmentation", condition, positions)
                    for condition in (
                        "imposed_shift_minus_20ms",
                        "imposed_shift_plus_20ms",
                    )
                ]
            )
            return float(candidate - baseline)
        raise AssertionError(f"Unknown comparison: {comparison}")

    comparisons = (
        "event_ba_gain",
        "strict_pre_ba_change",
        "transient_removed_ba_change",
        "contact_specific_increment_gain",
        "imposed_shift_robustness_gain",
    )
    all_positions = np.arange(len(reference), dtype=int)
    rows: list[dict[str, object]] = []
    for arm in AUGMENTED_ARMS:
        for comparison in comparisons:
            observed_by_seed: list[float] = []
            bootstrap_by_seed: list[np.ndarray] = []
            for fold_seed in LOCKED_FOLD_SEEDS:
                observed = measure(
                    fold_seed, arm, comparison, all_positions
                )
                bootstrapped = np.asarray(
                    [
                        measure(fold_seed, arm, comparison, positions)
                        for positions in bootstrap_positions
                    ]
                )
                observed_by_seed.append(observed)
                bootstrap_by_seed.append(bootstrapped)
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
            mean_bootstrap = np.mean(np.stack(bootstrap_by_seed), axis=0)
            rows.append(
                {
                    "arm": arm,
                    "comparison": comparison,
                    "fold_seed": "mean_across_seeds",
                    "observed_difference": float(np.mean(observed_by_seed)),
                    "ci_low": float(np.quantile(mean_bootstrap, 0.025)),
                    "ci_high": float(np.quantile(mean_bootstrap, 0.975)),
                    "n_bootstrap": int(config.n_bootstrap),
                    "resampling_unit": "lineage_group_id_shared_across_seeds",
                }
            )
    return pd.DataFrame(rows)


def _verdict(
    paired_differences: pd.DataFrame,
    config: ContactWindowAugmentationConfig,
) -> dict[str, object]:
    evaluated: list[dict[str, object]] = []
    qualifying: list[str] = []
    for arm in AUGMENTED_ARMS:
        rows = paired_differences[paired_differences["arm"] == arm]
        mean = rows[rows["fold_seed"] == "mean_across_seeds"].set_index(
            "comparison"
        )
        event = mean.loc["event_ba_gain"]
        seed_event = rows[
            (rows["comparison"] == "event_ba_gain")
            & (rows["fold_seed"] != "mean_across_seeds")
        ]
        seed_controls = rows[
            rows["comparison"].isin(
                ("strict_pre_ba_change", "transient_removed_ba_change")
            )
            & (rows["fold_seed"] != "mean_across_seeds")
        ]
        same_direction = bool((seed_event["observed_difference"] > 0).all())
        controls_not_weakened = bool(
            (seed_controls["observed_difference"] <= 0).all()
        )
        gain_at_least_minimum = bool(
            event["observed_difference"] >= config.minimum_headline_ba_gain
        )
        interval_above_zero = bool(event["ci_low"] > 0)
        qualifies = bool(
            same_direction
            and controls_not_weakened
            and gain_at_least_minimum
            and interval_above_zero
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
                "paired_interval_above_zero": interval_above_zero,
                "no_control_rise_in_any_seed": controls_not_weakened,
                "qualifies_for_downstream_validation": qualifies,
            }
        )
    return {
        "decision": "continue" if qualifying else "stop",
        "continue_augmentation_direction": bool(qualifying),
        "eligible_for_downstream_validation": bool(qualifying),
        "qualifying_arms": qualifying,
        "evaluated_arms": evaluated,
        "preferred_recipe_selected": False,
        "headline_replacement_allowed": False,
        "primary_common_benchmark_unchanged": True,
        "development_evidence_only": True,
        "minimum_headline_ba_gain": float(config.minimum_headline_ba_gain),
    }


def _report_zh(
    metrics: pd.DataFrame,
    paired_differences: pd.DataFrame,
    verdict: dict[str, object],
    population_audit: dict[str, object],
) -> str:
    lines = [
        "# M2D contact-window augmentation invariance 三-seed 复测",
        "",
        "本报告属于开发集探索证据；ADR-0004 共同比较保持不变，真实 outer-test 音频不做训练增强。",
        "",
        "## 锁定增强协议",
        "",
        "预声明五臂：不增强、contact-time jitter、衰减式 gain、轻度高频 EQ，以及三者 combined。",
        "每个 outer-training 原样本追加一个同 UID、同 lineage 的 derivative；原样本与 derivative 各占 0.5 source weight。",
        "outer-test 只使用原始 Event、strict-Pre、transient-removed；固定 ±20 ms 仅作为 imposed-shift robustness 诊断。",
        "所有输出保持 200 ms，waveform padding 为 0，增强器看不到项目标签。",
        "",
        "## 三个 fold seeds 的 OOF 结果",
        "",
        "| seed | 臂 | Event BA | Pre BA | Removed BA | -20 ms BA | +20 ms BA | Contact increment |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    indexed = metrics.set_index(["fold_seed", "arm", "condition"])
    for fold_seed in LOCKED_FOLD_SEEDS:
        for arm in ARMS:
            def value(condition: str) -> float:
                return float(
                    indexed.loc[
                        (fold_seed, arm, condition), "balanced_accuracy"
                    ]
                )

            lines.append(
                f"| {fold_seed} | {arm} | {value('event'):.3f} | "
                f"{value('strict_pre'):.3f} | {value('transient_removed'):.3f} | "
                f"{value('imposed_shift_minus_20ms'):.3f} | "
                f"{value('imposed_shift_plus_20ms'):.3f} | "
                f"{value('contact_specific_increment'):+.3f} |"
            )
    lines.extend(
        [
            "",
            "## 三-seed 平均 paired lineage-group 差异",
            "",
            "| 臂 | 比较 | 差异 | 95% CI |",
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
    lines.extend(
        [
            "",
            "## 判定与边界",
            "",
            (
                "至少一个增强臂满足预声明门槛；仅进入后续完整 family-aware 验证，不选择 preferred recipe。"
                if verdict["decision"] == "continue"
                else "没有增强臂同时满足三 seed、+0.02 BA、paired CI 与逐-seed负控门槛；停止该方向。"
            ),
            "不选择 preferred recipe，也不根据 outer OOF 直接替换 headline。",
            f"本次使用 {int(population_audit['n_exact_pairs'])} 条 exact pairs、"
            f"{int(population_audit['n_lineage_groups'])} 个 lineage groups。",
            "这些开发集 OOF 结果不能证明跨比赛、设备、采集者或采集流程泛化。",
        ]
    )
    return "\n".join(lines) + "\n"


def run_contact_window_augmentation_evaluation(
    source_bundle: Path,
    output_dir: Path,
    snapshot: DatasetSnapshot,
    encoder: EncoderAdapter,
    config: ContactWindowAugmentationConfig,
) -> ArtifactBundle:
    """Evaluate the locked train-only contact-window augmentation family."""

    config_document = _validate_config(config)
    source_root = Path(source_bundle).resolve()
    source_protocol, feature_path = _validated_source(source_root)
    encoder_document = _validate_encoder(encoder, source_protocol)
    (
        representation,
        source_folds,
        event_windows,
        sample_map,
        snapshot_provenance,
    ) = _validated_population(
        source_root, source_protocol, feature_path, snapshot
    )
    seeded_folds = _seeded_folds(
        source_folds, representation.paired, LOCKED_FOLD_SEEDS
    )
    probe_document, candidates = _locked_probe()

    augmentation_family = {
        "arms": list(ARMS),
        "single_derivative_arms": list(AUGMENTED_ARMS),
        "recipes": {
            "time_jitter": {"uniform_jitter_ms": [-20, 20]},
            "gain": {"uniform_gain_db": list(GAIN_DB_RANGE)},
            "light_eq": {
                "uniform_cutoff_hz": list(EQ_CUTOFF_HZ_RANGE),
                "uniform_nyquist_attenuation_db": list(EQ_ATTENUATION_DB_RANGE),
            },
            "combined": {
                "composition": ["time_jitter", "gain", "light_eq"]
            },
        },
        "aggregation_policy": AGGREGATION_POLICY,
        "source_weight": {"original": 0.5, "derivative": 0.5},
        "outer_test_policy": (
            "unaugmented_except_predeclared_imposed_shift_diagnostics"
        ),
        "imposed_shift_diagnostics_ms": [-20, 20],
        "project_label_visible_to_augmentation": False,
        "waveform_padding_samples": 0,
        "post_contact_outcome_context": False,
    }
    provenance = {
        "source_artifact_id": str(source_protocol["artifact_id"]),
        "source_protocol_sha256": _file_sha256(source_root / "protocol.json"),
        "source_features_sha256": _file_sha256(feature_path),
        "source_folds_sha256": _file_sha256(source_root / "fold_assignments.csv"),
        "source_exclusions_sha256": _file_sha256(source_root / "exclusions.csv"),
        "source_window_manifest_sha256": _file_sha256(
            source_root / "windows_manifest.csv"
        ),
        **snapshot_provenance,
        "encoder": encoder_document,
        "augmentation_seed": int(config.augmentation_seed),
        "aggregation_policy": AGGREGATION_POLICY,
        "encoder_inference_runs": int(
            len(representation.paired) * (
                len(LOCKED_FOLD_SEEDS) * len(AUGMENTED_ARMS) + 2
            )
        ),
        "feature_cache_rows": int(
            len(representation.paired) * (
                len(LOCKED_FOLD_SEEDS) * len(AUGMENTED_ARMS) + 2
            )
        ),
    }
    protocol_document = {
        "protocol_version": CONTACT_AUGMENTATION_PROTOCOL_VERSION,
        "evidence_role": "development_exploratory",
        "primary_common_benchmark_unchanged": True,
        "source_artifact_id": str(source_protocol["artifact_id"]),
        "config": config_document,
        "fold_policy": {
            "name": "StratifiedGroupKFold",
            "group": "lineage_group_id",
            "outer_splits": 5,
            "shuffle": True,
            "seeds": list(LOCKED_FOLD_SEEDS),
            "locked_source_assignments_reused_for_first_seed": True,
        },
        "augmentation_family": augmentation_family,
        "probe": probe_document,
        "screening_policy": {
            "baseline_arm": "no_augmentation",
            "event_ba_gain_required": LOCKED_MINIMUM_HEADLINE_BA_GAIN,
            "all_three_seeds_must_improve": True,
            "paired_lineage_interval_must_be_above_zero": True,
            "strict_pre_and_removed_must_not_rise_in_any_seed": True,
            "outer_results_select_preferred_recipe": False,
        },
        "provenance_fingerprint": provenance,
    }
    artifact_id = _canonical_sha256(protocol_document)[:24]
    bundle_root = Path(output_dir).resolve() / artifact_id
    bundle_root.mkdir(parents=True, exist_ok=True)

    cached = _load_token_cache(bundle_root, protocol_document, artifact_id)
    if cached is None:
        token_cache, token_index, waveform_audit, key_to_index = _encode_cache(
            representation, event_windows, sample_map, encoder, config
        )
    else:
        token_cache, token_index, waveform_audit, key_to_index = cached
    assignments = _augmentation_assignments(seeded_folds, key_to_index)
    predictions, metrics, selections, fit_audit = _evaluate_family(
        representation,
        seeded_folds,
        source_protocol,
        token_cache,
        key_to_index,
        candidates,
    )
    differences = _paired_differences(predictions, config)
    verdict = _verdict(differences, config)
    group_sizes = representation.paired.groupby("lineage_group_id").size()
    population_audit = {
        "n_source_event_eligible": int(len(source_folds)),
        "n_exact_pairs": int(len(representation.paired)),
        "n_lineage_groups": int(len(group_sizes)),
        "n_singleton_lineage_groups": int(group_sizes.eq(1).sum()),
        "identical_membership_across_arms_and_seeds": True,
        "outer_test_derivatives_used": 0,
        "waveform_padding_samples": 0,
        "required_windows": ["event_200ms", "pre_200ms", "removed_200ms"],
    }
    if len(token_cache) != provenance["encoder_inference_runs"]:
        raise AssertionError("Encoder inference audit does not match the cache")

    artifact_paths: dict[str, Path] = {}
    for name, frame, filename in (
        ("fold_assignments", seeded_folds, "fold_assignments.csv"),
        (
            "augmentation_assignments",
            assignments,
            "augmentation_assignments.csv",
        ),
        ("augmented_token_index", token_index, "augmented_token_index.csv"),
        ("waveform_audit", waveform_audit, "waveform_audit.csv"),
        ("oof_predictions", predictions, "oof_predictions.csv"),
        ("metrics", metrics, "metrics.csv"),
        ("selections", selections, "selections.csv"),
        ("paired_differences", differences, "paired_differences.csv"),
    ):
        path = bundle_root / filename
        frame.to_csv(path, index=False)
        artifact_paths[name] = path
    token_cache_path = bundle_root / "augmented_tokens.npy"
    np.save(token_cache_path, token_cache, allow_pickle=False)
    artifact_paths["augmented_tokens"] = token_cache_path
    exclusions_path = bundle_root / "exclusions.csv"
    shutil.copy2(source_root / "exclusions.csv", exclusions_path)
    artifact_paths["exclusions"] = exclusions_path
    for name, document, filename in (
        ("fit_audit", fit_audit, "fit_audit.json"),
        ("population_audit", population_audit, "population_audit.json"),
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
        _report_zh(metrics, differences, verdict, population_audit),
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
