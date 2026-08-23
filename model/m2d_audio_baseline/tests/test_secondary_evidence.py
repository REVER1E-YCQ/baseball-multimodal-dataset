from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.secondary_evidence import (
    SecondaryEvidenceError,
    compute_secondary_evidence,
)
from scripts.short_contact_benchmark import (
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderAdapter,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)


@dataclass
class _FakeEncoder(EncoderAdapter):
    provenance: EncoderProvenance = field(
        default_factory=lambda: EncoderProvenance(
            name="fake",
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
        )
    )

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        center = float(waveform[len(waveform) // 2])
        energy = float(np.mean(np.square(waveform)))
        token = np.asarray(
            [center, energy, abs(center), float(sample_rate) / 16_000.0]
        )
        return np.stack([token, token])


def _make_snapshot(root: Path, prefix: str) -> DatasetSnapshot:
    samples: list[SnapshotSample] = []
    for game_index in range(16):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"{label}__Collector_A__{prefix}-{game_index:02d}"
            waveform = np.zeros(16_000, dtype=np.float32)
            waveform[8_000] = polarity
            audio_path = root / "snapshot" / f"{uid}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio_path, 16_000, waveform)
            samples.append(
                SnapshotSample(
                    uid=uid,
                    label=label,
                    lineage_group_id=f"game-{game_index:02d}",
                    audio_path=audio_path,
                    event_start=0.45,
                    event_end=0.55,
                )
            )
    return DatasetSnapshot(revision=f"{prefix}-snapshot", samples=tuple(samples))


def _fixed_split_csv(root: Path, snapshot: DatasetSnapshot) -> Path:
    rows = []
    uids = sorted(sample.uid for sample in snapshot.samples)
    for position, uid in enumerate(uids):
        label, collector, sample_id = uid.split("__")
        partition = ["train", "train", "val", "test"][position % 4]
        rows.append(
            {
                "dataset_path": f"dataset/{label}/{collector}/{sample_id}",
                "sample_id": sample_id,
                "label": label,
                "source_group": f"src-{position % 4}",
                "split": partition,
            }
        )
    path = root / "dataset_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


class SecondaryEvidenceTest(unittest.TestCase):
    def _bundles(self, root: Path, snapshot: DatasetSnapshot) -> dict[str, object]:
        protocol = BenchmarkProtocol(seed=71, outer_splits=2)
        first = run_short_contact_benchmark(
            protocol,
            snapshot,
            (
                _FakeEncoder(
                    EncoderProvenance(
                        name="encoder-a",
                        upstream_revision="r",
                        checkpoint_sha256="c",
                        precision="fp32",
                        token_dimension=4,
                    )
                ),
            ),
            root / "a",
        )
        second = run_short_contact_benchmark(
            protocol,
            snapshot,
            (
                _FakeEncoder(
                    EncoderProvenance(
                        name="encoder-b",
                        upstream_revision="r",
                        checkpoint_sha256="c",
                        precision="fp32",
                        token_dimension=4,
                    )
                ),
            ),
            root / "b",
        )
        return {"encoder-a": first, "encoder-b": second}

    def test_fixed_split_and_rbf_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "sec")
            bundles = self._bundles(root, snapshot)
            split_path = _fixed_split_csv(root, snapshot)

            evidence = compute_secondary_evidence(
                bundles, split_path, root / "evidence", seed=73
            )

            self.assertTrue(
                evidence.summary["fixed_split_membership_reproduced"]
            )
            self.assertFalse(evidence.summary["fixed_split_used_for_primary_folds"])
            fixed_metrics = pd.read_csv(evidence.path("fixed_split_metrics.csv"))
            self.assertEqual(set(fixed_metrics["encoder"]), {"encoder-a", "encoder-b"})
            self.assertTrue(
                (fixed_metrics["development_evidence"] == True).all()
            )
            self.assertTrue(
                (fixed_metrics["not_source_transfer_evidence"] == True).all()
            )
            fixed_predictions = pd.read_csv(
                evidence.path("fixed_split_predictions.csv")
            )
            self.assertEqual(set(fixed_predictions["split"]), {"test"})
            self.assertFalse(
                fixed_predictions[["encoder", "uid"]].duplicated().any()
            )

            rbf_metrics = pd.read_csv(evidence.path("rbf_metrics.csv"))
            self.assertEqual(set(rbf_metrics["encoder"]), {"encoder-a", "encoder-b"})
            self.assertTrue((rbf_metrics["exploratory"] == True).all())
            rbf_predictions = pd.read_csv(evidence.path("rbf_predictions.csv"))
            self.assertEqual(len(rbf_predictions), 64)
            self.assertFalse(
                rbf_predictions[["encoder", "uid"]].duplicated().any()
            )
            rbf_selections = pd.read_csv(evidence.path("rbf_selections.csv"))
            self.assertEqual(len(rbf_selections), 4)
            self.assertTrue(
                rbf_selections["selected_C"].isin((0.3, 1.0, 3.0)).all()
            )
            self.assertTrue(
                rbf_selections["selected_gamma"].isin(("scale", 0.001)).all()
            )

    def test_test_partition_never_informs_fitting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "sec")
            # Invert the class polarity of every sample that the fixed split
            # assigns to the test partition, before any bundle is built. A
            # leak-free model must fail on the test partition. The split file
            # uses sorted UIDs, so invert in the same order.
            sorted_uids = sorted(sample.uid for sample in snapshot.samples)
            by_uid = {sample.uid: sample for sample in snapshot.samples}
            for position, uid in enumerate(sorted_uids):
                if position % 4 == 3:
                    sample = by_uid[uid]
                    polarity = -1.0 if sample.label == "fly_ball" else 1.0
                    waveform = np.zeros(16_000, dtype=np.float32)
                    waveform[8_000] = polarity
                    wavfile.write(sample.audio_path, 16_000, waveform)
            bundles = self._bundles(root, snapshot)
            split_path = _fixed_split_csv(root, snapshot)

            evidence = compute_secondary_evidence(
                bundles, split_path, root / "evidence", seed=73
            )
            fixed_metrics = pd.read_csv(evidence.path("fixed_split_metrics.csv"))
            for _index, row in fixed_metrics.iterrows():
                self.assertLess(float(row["balanced_accuracy"]), 0.35)

    def test_membership_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "sec")
            bundles = self._bundles(root, snapshot)
            split_path = _fixed_split_csv(root, snapshot)
            split = pd.read_csv(split_path, encoding="utf-8-sig")
            split = split.iloc[:-1]
            split.to_csv(split_path, index=False, encoding="utf-8-sig")
            with self.assertRaises(SecondaryEvidenceError) as context:
                compute_secondary_evidence(
                    bundles, split_path, root / "evidence", seed=73
                )
            self.assertIn("does not reproduce", str(context.exception))


if __name__ == "__main__":
    unittest.main()
