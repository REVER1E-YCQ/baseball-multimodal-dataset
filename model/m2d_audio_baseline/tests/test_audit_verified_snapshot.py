from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.audit_verified_snapshot import (
    SnapshotAuditError,
    audit_verified_snapshot,
    _write_audit_outputs,
)

MLB_SOURCE = {"source_id": "MLB_100001_slug", "video_url": "https://a.example.com/1.mp4"}


def _write_wav(path: Path, seconds: float = 1.0, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    waveform = (rng.standard_normal(int(16_000 * seconds)) * 0.1).astype(np.float32)
    wavfile.write(path, 16_000, waveform)


def _sample_files(
    root: Path,
    dataset_path: str,
    label: str,
    event_start: float = 0.4,
    event_end: float = 0.6,
    source: dict[str, str] | None = MLB_SOURCE,
    corrupt_audio: bool = False,
    seed: int = 0,
) -> None:
    sample_dir = root / dataset_path
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "label.txt").write_text(label + "\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "sample_id": dataset_path.rsplit("/", 1)[1],
                "label": label,
                "event_start": event_start,
                "event_end": event_end,
            }
        ]
    ).to_csv(sample_dir / "sample.csv", index=False)
    if corrupt_audio:
        (sample_dir / "audio.wav").write_bytes(b"not a wav file at all")
    else:
        _write_wav(sample_dir / "audio.wav", seed=seed)
    if source is not None:
        (sample_dir / "source.txt").write_text(
            "\n".join(f"{key}: {value}" for key, value in source.items()) + "\n",
            encoding="utf-8",
        )
    (sample_dir / "video.mp4").write_bytes(b"fake-video")


def _write_samples(
    root: Path,
    rows: list[dict[str, object]],
    source: dict[str, str] | None = MLB_SOURCE,
    corrupt_audio: bool = False,
    seed: int = 0,
) -> None:
    for row in rows:
        _sample_files(
            root,
            str(row["dataset_path"]),
            str(row["label"]),
            source=source,
            corrupt_audio=corrupt_audio,
            seed=seed,
        )


def _manifest_csv(root: Path, rows: list[dict[str, object]]) -> None:
    manifest_dir = root / "reports" / "verified_dataset_20260804"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(
        manifest_dir / "VERIFIED_DATASET_MANIFEST.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _make_repo(root: Path) -> str:
    subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-q",
            "-m",
            "synthetic snapshot",
        ],
        check=True,
    )
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _base_row(dataset_path: str, label: str, **overrides: object) -> dict[str, object]:
    collector = (
        "Codex_Workstation"
        if "/Codex_Workstation/" in dataset_path
        else "Zhengxuan_Liu"
    )
    row = {
        "dataset_path": dataset_path,
        "sample_id": dataset_path.rsplit("/", 1)[1],
        "label": label,
        "collector": collector,
        "verification_source": "human_binary_review",
        "verification_detail": "test-detail",
        "timing_was_corrected": "N",
        "original_event_start": 0.4,
        "original_event_end": 0.6,
        "final_event_start": 0.4,
        "final_event_end": 0.6,
    }
    row.update(overrides)
    return row


class AuditVerifiedSnapshotTest(unittest.TestCase):
    def _audit(
        self,
        root: Path,
        rows: list[dict[str, object]],
        expected_total: int,
        expected_label_counts: dict[str, int],
        expected_revision: str | None = None,
    ):
        _manifest_csv(root, rows)
        revision = _make_repo(root)
        if expected_revision is None:
            expected_revision = revision
        return audit_verified_snapshot(
            root,
            expected_revision=expected_revision,
            expected_total=expected_total,
            expected_label_counts=expected_label_counts,
        )

    def test_audits_valid_snapshot_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
                _base_row("dataset/fly_ball/Codex_Workstation/F_0002", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0002", "ground_ball"),
            ]
            _write_samples(root, rows)

            snapshot, audit = self._audit(
                root,
                rows,
                expected_total=4,
                expected_label_counts={"fly_ball": 2, "ground_ball": 2},
            )

            self.assertEqual(snapshot.revision, audit.revision)
            self.assertEqual(len(snapshot.samples), 4)
            self.assertEqual(audit.summary["sample_count"], 4)
            self.assertEqual(
                audit.summary["label_counts"], {"fly_ball": 2, "ground_ball": 2}
            )
            self.assertTrue(audit.summary["audit_passed"])
            for sample in snapshot.samples:
                self.assertTrue(
                    sample.lineage_group_id.startswith("mlb_game_pk:100001")
                )
                self.assertTrue(sample.audio_path.is_absolute())
            for sample in audit.samples:
                self.assertEqual(sample.lineage_group_basis, "mlb_game_pk")
                self.assertEqual(sample.lineage_fallback, "")

    def test_unions_exact_audio_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
            ]
            _write_samples(root, rows, seed=7)
            # Overwrite the second audio with a byte-identical copy of the first.
            duplicate = root / "dataset/fly_ball/Codex_Workstation/F_0001/audio.wav"
            wavfile.write(
                root / "dataset/ground_ball/Codex_Workstation/G_0001/audio.wav",
                16_000,
                wavfile.read(duplicate)[1],
            )

            _, audit = self._audit(
                root,
                rows,
                expected_total=2,
                expected_label_counts={"fly_ball": 1, "ground_ball": 1},
            )
            group_ids = {sample.lineage_group_id for sample in audit.samples}
            self.assertEqual(len(group_ids), 1)
            self.assertGreaterEqual(
                audit.summary["lineage_groups"]["duplicate_unions"], 1
            )

    def test_rejects_revision_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
            ]
            _write_samples(root, rows)
            with self.assertRaises(SnapshotAuditError) as context:
                self._audit(
                    root,
                    rows,
                    expected_total=2,
                    expected_label_counts={"fly_ball": 1, "ground_ball": 1},
                    expected_revision="f" * 40,
                )
            self.assertIn("revision mismatch", str(context.exception))

    def test_rejects_count_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
            ]
            _write_samples(root, rows)
            with self.assertRaises(SnapshotAuditError) as context:
                self._audit(
                    root,
                    rows,
                    expected_total=3,
                    expected_label_counts={"fly_ball": 1, "ground_ball": 1},
                )
            self.assertIn("membership mismatch", str(context.exception))

    def test_rejects_duplicate_uid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
            ]
            _write_samples(root, rows)
            with self.assertRaises(SnapshotAuditError) as context:
                self._audit(
                    root,
                    rows,
                    expected_total=2,
                    expected_label_counts={"fly_ball": 2, "ground_ball": 0},
                )
            self.assertIn("duplicate_uid", str(context.exception))

    def test_rejects_missing_and_unreadable_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
            ]
            _write_samples(root, rows, corrupt_audio=True)
            (root / str(rows[0]["dataset_path"]) / "audio.wav").unlink()
            with self.assertRaises(SnapshotAuditError) as context:
                self._audit(
                    root,
                    rows,
                    expected_total=2,
                    expected_label_counts={"fly_ball": 1, "ground_ball": 1},
                )
            self.assertIn("missing_file", str(context.exception))
            self.assertIn("unreadable_audio", str(context.exception))

    def test_rejects_label_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Codex_Workstation/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
            ]
            _write_samples(root, rows)
            (root / "dataset/fly_ball/Codex_Workstation/F_0001/label.txt").write_text(
                "ground_ball\n", encoding="utf-8"
            )
            with self.assertRaises(SnapshotAuditError) as context:
                self._audit(
                    root,
                    rows,
                    expected_total=2,
                    expected_label_counts={"fly_ball": 1, "ground_ball": 1},
                )
            self.assertIn("label_mismatch", str(context.exception))

    def test_rejects_invalid_event_interval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row(
                    "dataset/fly_ball/Codex_Workstation/F_0001",
                    "fly_ball",
                    final_event_start=0.4,
                    final_event_end=5.0,
                ),
                _base_row("dataset/ground_ball/Codex_Workstation/G_0001", "ground_ball"),
            ]
            _write_samples(root, rows)
            with self.assertRaises(SnapshotAuditError) as context:
                self._audit(
                    root,
                    rows,
                    expected_total=2,
                    expected_label_counts={"fly_ball": 1, "ground_ball": 1},
                )
            self.assertIn("invalid_event_interval", str(context.exception))

    def test_reports_ambiguous_grouping_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                _base_row("dataset/fly_ball/Zhengxuan_Liu/F_0001", "fly_ball"),
                _base_row("dataset/ground_ball/Zhengxuan_Liu/G_0001", "ground_ball"),
            ]
            # source.txt exists but carries no usable identity: the strongest
            # available identity is the singleton-UID fallback. Distinct seeds
            # keep the two samples from being exact audio duplicates.
            for position, row in enumerate(rows):
                _sample_files(
                    root,
                    str(row["dataset_path"]),
                    str(row["label"]),
                    source={"note": "no identity fields"},
                    seed=position,
                )
            _, audit = self._audit(
                root,
                rows,
                expected_total=2,
                expected_label_counts={"fly_ball": 1, "ground_ball": 1},
            )
            for sample in audit.samples:
                self.assertEqual(sample.lineage_fallback, "singleton_uid")
                self.assertEqual(sample.lineage_group_basis, "singleton_uid")
            self.assertEqual(
                audit.summary["lineage_groups"]["fallback_counts"]["singleton_uid"], 2
            )
            out_dir = root / "audit-output"
            _write_audit_outputs(audit, out_dir)
            self.assertTrue((out_dir / "sample_inventory.csv").is_file())
            self.assertTrue((out_dir / "lineage_groups.csv").is_file())
            summary = json.loads(
                (out_dir / "audit_summary.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["audit_passed"])


if __name__ == "__main__":
    unittest.main()
