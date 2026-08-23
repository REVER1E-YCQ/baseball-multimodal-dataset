from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from scipy.io import wavfile

from .prepare_windows import to_float_mono
from .short_contact_benchmark import DatasetSnapshot, SnapshotSample


PINNED_REVISION = "4b6ed0e1cea1425121b075212ddb49b820e27cda"
MANIFEST_RELPATH = Path("reports") / "verified_dataset_20260804" / "VERIFIED_DATASET_MANIFEST.csv"
EXPECTED_TOTAL = 822
EXPECTED_LABEL_COUNTS = {"fly_ball": 386, "ground_ball": 436}
GAME_RE = re.compile(r"^MLB_(\d+)_(.+)$")


class SnapshotAuditError(RuntimeError):
    """Raised when the verified snapshot is missing, malformed, or changed."""


@dataclass(frozen=True)
class AuditedSample:
    uid: str
    sample_id: str
    label: str
    collector: str
    dataset_path: Path
    event_start: float
    event_end: float
    verification_source: str
    verification_detail: str
    timing_was_corrected: bool
    audio_sha256: str
    lineage_group_id: str
    lineage_group_basis: str
    lineage_fallback: str
    source_id: str


@dataclass(frozen=True)
class SnapshotAudit:
    revision: str
    samples: tuple[AuditedSample, ...]
    summary: dict[str, object]
    sample_rows: pd.DataFrame
    lineage_rows: pd.DataFrame
    failures: tuple[dict[str, str], ...]


def _repository_revision(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SnapshotAuditError(
            f"Dataset root is not a Git worktree: {root}"
        ) from error
    return completed.stdout.strip()


def _canonical_url(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return ""
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path, "", "")
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_one_row_csv(path: Path) -> dict[str, str]:
    rows = list(pd.read_csv(path, encoding="utf-8-sig", dtype=str).to_dict("records"))
    if len(rows) != 1:
        raise ValueError(f"Expected one data row, found {len(rows)} in {path}")
    return {str(key).strip(): str(value or "").strip() for key, value in rows[0].items()}


def _parse_source(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.replace("\uff1a", ":")
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip().lower()] = value.strip()
    return values


def _clean_uid(label: str, collector: str, sample_id: str) -> str:
    raw = f"{label}__{collector}__{sample_id}".strip().replace(" ", "_")
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    if not cleaned:
        raise ValueError("UID cannot be empty")
    return cleaned


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self._parent[value] != value:
            self._parent[value] = self.find(self._parent[value])
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)


def _assign_lineage(
    sample_rows: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, object]]:
    uids = list(sample_rows["uid"])
    union = _UnionFind(uids)
    basis: dict[str, str] = {}
    group_key: dict[str, str] = {}
    fallback: dict[str, str] = {}

    for row in sample_rows.itertuples(index=False):
        uid = str(row.uid)
        source_id = str(row.source_id or "")
        video_url = str(row.video_url or "")
        clip_id = str(row.clip_id or "")
        match = GAME_RE.match(source_id)
        if match:
            basis[uid] = "mlb_game_pk"
            group_key[uid] = match.group(1)
            fallback[uid] = ""
        else:
            canonical = _canonical_url(video_url)
            if canonical:
                basis[uid] = "canonical_video_url"
                group_key[uid] = canonical
                fallback[uid] = ""
            elif source_id:
                basis[uid] = "source_id"
                group_key[uid] = source_id
                fallback[uid] = ""
            elif clip_id:
                basis[uid] = "clip_id"
                group_key[uid] = clip_id
                fallback[uid] = ""
            else:
                basis[uid] = "singleton_uid"
                group_key[uid] = uid
                fallback[uid] = "singleton_uid"

    game_to_uids: dict[str, list[str]] = {}
    for uid in uids:
        if basis[uid] == "mlb_game_pk":
            game_to_uids.setdefault(group_key[uid], []).append(uid)
    for members in game_to_uids.values():
        for right in members[1:]:
            union.union(members[0], right)

    audio_by_hash: dict[str, list[str]] = {}
    for row in sample_rows.itertuples(index=False):
        audio_by_hash.setdefault(str(row.audio_sha256), []).append(str(row.uid))
    duplicate_unions = 0
    for members in audio_by_hash.values():
        if len(members) > 1:
            duplicate_unions += 1
            for right in members[1:]:
                union.union(members[0], right)

    components: dict[str, list[str]] = {}
    for uid in uids:
        components.setdefault(union.find(uid), []).append(uid)
    group_id: dict[str, str] = {}
    group_basis: dict[str, str] = {}
    for members in components.values():
        sorted_members = sorted(members)
        if len(members) == 1:
            first = sorted_members[0]
            group_id[first] = f"{basis[first]}:{group_key[first]}"
            group_basis[first] = basis[first]
            continue
        bases = sorted({basis[uid] for uid in members})
        keys = sorted({group_key[uid] for uid in members})
        if len(bases) == 1 and len(keys) == 1:
            group_id[sorted_members[0]] = f"{bases[0]}:{keys[0]}"
            group_basis[sorted_members[0]] = bases[0]
        else:
            group_id[sorted_members[0]] = (
                "duplicate_lineage:"
                + hashlib.sha256(
                    "|".join(sorted_members).encode("utf-8")
                ).hexdigest()[:16]
            )
            group_basis[sorted_members[0]] = "exact_audio_duplicate"
        for uid in sorted_members:
            group_id[uid] = group_id[sorted_members[0]]
            group_basis[uid] = group_basis[sorted_members[0]]

    group_series = pd.Series(
        [group_id[uid] for uid in uids],
        index=sample_rows.index,
        dtype="string",
    )
    basis_series = pd.Series(
        [group_basis[uid] for uid in uids],
        index=sample_rows.index,
        dtype="string",
    )
    fallback_series = pd.Series(
        [fallback[uid] for uid in uids],
        index=sample_rows.index,
        dtype="string",
    )

    diagnostics = {
        "duplicate_unions": duplicate_unions,
        "fallback_counts": {
            str(key): int(value)
            for key, value in fallback_series.value_counts().items()
        },
    }
    return group_series, basis_series, fallback_series, diagnostics


def _build_lineage_table(
    sample_rows: pd.DataFrame,
    group_series: pd.Series,
    basis_series: pd.Series,
) -> pd.DataFrame:
    frame = sample_rows.assign(
        lineage_group_id=group_series,
        lineage_group_basis=basis_series,
    )
    groups = (
        frame.groupby("lineage_group_id", as_index=False)
        .agg(
            n_samples=("uid", "size"),
            labels=("label", lambda values: ";".join(sorted(set(values)))),
            basis=("lineage_group_basis", "first"),
            uids=("uid", lambda values: "|".join(sorted(values))),
        )
        .sort_values(["n_samples", "lineage_group_id"], ascending=[False, True])
    )
    return groups


def audit_verified_snapshot(
    dataset_root: Path,
    expected_revision: str = PINNED_REVISION,
    manifest_relpath: Path = MANIFEST_RELPATH,
    expected_total: int = EXPECTED_TOTAL,
    expected_label_counts: dict[str, int] | None = None,
) -> tuple[DatasetSnapshot, SnapshotAudit]:
    """Audit the immutable verified snapshot and return a benchmark-ready dataset."""

    if expected_label_counts is None:
        expected_label_counts = dict(EXPECTED_LABEL_COUNTS)
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        raise SnapshotAuditError(f"Dataset root does not exist: {dataset_root}")

    actual_revision = _repository_revision(dataset_root)
    if actual_revision != expected_revision:
        raise SnapshotAuditError(
            "Dataset revision mismatch: "
            f"expected {expected_revision}, found {actual_revision}"
        )

    manifest_path = dataset_root / manifest_relpath
    if not manifest_path.is_file():
        raise SnapshotAuditError(f"Verified manifest is missing: {manifest_path}")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig", dtype=str)
    required_columns = {
        "dataset_path",
        "sample_id",
        "label",
        "collector",
        "verification_source",
        "verification_detail",
        "timing_was_corrected",
        "final_event_start",
        "final_event_end",
    }
    missing_columns = required_columns.difference(manifest.columns)
    if missing_columns:
        raise SnapshotAuditError(
            f"Verified manifest is missing columns: {sorted(missing_columns)}"
        )

    failures: list[dict[str, str]] = []
    seen_uids: set[str] = set()
    per_sample_rows: list[dict[str, object]] = []

    for manifest_row in manifest.itertuples(index=False):
        dataset_path = str(manifest_row.dataset_path or "").strip()
        sample_id = str(manifest_row.sample_id or "").strip()
        label = str(manifest_row.label or "").strip()
        collector = str(manifest_row.collector or "").strip()
        verification_source = str(manifest_row.verification_source or "").strip()
        verification_detail = str(manifest_row.verification_detail or "").strip()
        timing_was_corrected = (
            str(manifest_row.timing_was_corrected or "").strip().upper()
            in {"Y", "TRUE", "1"}
        )
        sample_dir = dataset_root / dataset_path

        if not dataset_path.startswith("dataset/"):
            failures.append(
                {"uid": dataset_path, "reason": "invalid_dataset_path", "detail": dataset_path}
            )
            continue
        try:
            uid = _clean_uid(label, collector, sample_id)
        except ValueError:
            uid = f"invalid__{dataset_path}"
        if uid in seen_uids:
            failures.append(
                {"uid": uid, "reason": "duplicate_uid", "detail": dataset_path}
            )
            continue
        seen_uids.add(uid)

        required_files = ["audio.wav", "label.txt", "sample.csv", "source.txt"]
        missing = [name for name in required_files if not (sample_dir / name).is_file()]
        if missing:
            failures.append(
                {
                    "uid": uid,
                    "reason": "missing_file",
                    "detail": f"{dataset_path}: {','.join(missing)}",
                }
            )
            continue

        try:
            sample_row = _read_one_row_csv(sample_dir / "sample.csv")
        except Exception as error:
            failures.append(
                {"uid": uid, "reason": "malformed_sample_csv", "detail": str(error)}
            )
            continue
        label_txt = (sample_dir / "label.txt").read_text(
            encoding="utf-8", errors="replace"
        ).strip()
        labels = {label, sample_row.get("label", ""), label_txt, dataset_path.split("/")[1]}
        if len(labels) != 1:
            failures.append(
                {"uid": uid, "reason": "label_mismatch", "detail": f"{sorted(labels)}"}
            )
            continue

        try:
            sample_rate, raw = wavfile.read(sample_dir / "audio.wav")
            waveform = to_float_mono(raw)
        except Exception as error:
            failures.append(
                {"uid": uid, "reason": "unreadable_audio", "detail": str(error)}
            )
            continue
        duration = len(waveform) / float(sample_rate)
        try:
            event_start = float(manifest_row.final_event_start)
            event_end = float(manifest_row.final_event_end)
        except (TypeError, ValueError) as error:
            failures.append(
                {"uid": uid, "reason": "invalid_event_interval", "detail": str(error)}
            )
            continue
        if not (
            np.isfinite(event_start)
            and np.isfinite(event_end)
            and 0.0 <= event_start < event_end <= duration + 1e-6
        ):
            failures.append(
                {
                    "uid": uid,
                    "reason": "invalid_event_interval",
                    "detail": (
                        f"[{event_start}, {event_end}] vs duration {duration}"
                    ),
                }
            )
            continue

        source = _parse_source(sample_dir / "source.txt")
        source_id = source.get("source_id", "")
        video_url = source.get("video_url", "")
        clip_id = source.get("clip_id", "")
        # A source.txt without a usable identity is an ambiguous grouping
        # case handled by the singleton-UID fallback, not a membership failure.

        audio_sha256 = _file_sha256(sample_dir / "audio.wav")
        per_sample_rows.append(
            {
                "uid": uid,
                "sample_id": sample_id,
                "label": label,
                "collector": collector,
                "dataset_path": dataset_path,
                "event_start": event_start,
                "event_end": event_end,
                "verification_source": verification_source,
                "verification_detail": verification_detail,
                "timing_was_corrected": timing_was_corrected,
                "audio_sha256": audio_sha256,
                "source_id": source_id,
                "video_url": video_url,
                "clip_id": clip_id,
            }
        )

    if failures:
        raise SnapshotAuditError(
            f"Snapshot audit found {len(failures)} failures: "
            + "; ".join(
                f"{item['uid']}:{item['reason']}" for item in failures[:10]
            )
        )

    sample_rows = pd.DataFrame(per_sample_rows)
    if len(sample_rows) != expected_total:
        raise SnapshotAuditError(
            f"Snapshot membership mismatch: expected {expected_total} samples, "
            f"found {len(sample_rows)}"
        )
    label_counts = sample_rows["label"].value_counts().to_dict()
    for label, expected in expected_label_counts.items():
        actual = int(label_counts.get(label, 0))
        if actual != expected:
            raise SnapshotAuditError(
                f"Snapshot label mismatch for {label}: expected {expected}, found {actual}"
            )

    group_series, basis_series, fallback_series, lineage_diagnostics = _assign_lineage(
        sample_rows
    )
    sample_rows = sample_rows.assign(
        lineage_group_id=group_series,
        lineage_group_basis=basis_series,
        lineage_fallback=fallback_series,
    )
    lineage_rows = _build_lineage_table(sample_rows, group_series, basis_series)

    group_counts = (
        sample_rows.groupby("lineage_group_id")["uid"]
        .size()
        .rename("n_samples")
        .reset_index()
    )
    group_labels = (
        sample_rows.groupby("lineage_group_id")["label"]
        .nunique()
        .rename("n_labels")
        .reset_index()
    )
    stats = group_counts.merge(
        group_labels, on="lineage_group_id", validate="one_to_one"
    )
    mixed = int(stats["n_labels"].gt(1).sum())
    summary = {
        "revision": expected_revision,
        "sample_count": int(len(sample_rows)),
        "label_counts": label_counts,
        "verification_source_counts": (
            sample_rows["verification_source"].value_counts().to_dict()
        ),
        "collector_counts": sample_rows["collector"].value_counts().to_dict(),
        "timing_corrections": int(sample_rows["timing_was_corrected"].sum()),
        "lineage_groups": {
            "count": int(len(stats)),
            "singleton_groups": int(stats["n_samples"].eq(1).sum()),
            "multi_sample_groups": int(stats["n_samples"].gt(1).sum()),
            "mixed_label_groups": mixed,
            "largest_group_size": int(stats["n_samples"].max()),
            **lineage_diagnostics,
        },
        "audit_passed": True,
    }

    snapshot = DatasetSnapshot(
        revision=expected_revision,
        samples=tuple(
            SnapshotSample(
                uid=str(row.uid),
                label=str(row.label),
                lineage_group_id=str(row.lineage_group_id),
                audio_path=(dataset_root / str(row.dataset_path) / "audio.wav").resolve(),
                event_start=float(row.event_start),
                event_end=float(row.event_end),
            )
            for row in sample_rows.itertuples(index=False)
        ),
    )
    audited = tuple(
        AuditedSample(
            uid=str(row.uid),
            sample_id=str(row.sample_id),
            label=str(row.label),
            collector=str(row.collector),
            dataset_path=Path(str(row.dataset_path)),
            event_start=float(row.event_start),
            event_end=float(row.event_end),
            verification_source=str(row.verification_source),
            verification_detail=str(row.verification_detail),
            timing_was_corrected=bool(row.timing_was_corrected),
            audio_sha256=str(row.audio_sha256),
            lineage_group_id=str(row.lineage_group_id),
            lineage_group_basis=str(row.lineage_group_basis),
            lineage_fallback=str(row.lineage_fallback),
            source_id=str(row.source_id),
        )
        for row in sample_rows.itertuples(index=False)
    )
    return snapshot, SnapshotAudit(
        revision=expected_revision,
        samples=audited,
        summary=summary,
        sample_rows=sample_rows,
        lineage_rows=lineage_rows,
        failures=tuple(failures),
    )


def _write_audit_outputs(audit: SnapshotAudit, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit.sample_rows.to_csv(out_dir / "sample_inventory.csv", index=False)
    audit.lineage_rows.to_csv(out_dir / "lineage_groups.csv", index=False)
    (out_dir / "audit_summary.json").write_text(
        json.dumps(audit.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the immutable verified-dataset snapshot and lineage."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, audit = audit_verified_snapshot(
        dataset_root=args.dataset_root,
        expected_revision=args.expected_revision,
    )
    _write_audit_outputs(audit, args.out_dir.resolve())
    print(json.dumps(audit.summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
