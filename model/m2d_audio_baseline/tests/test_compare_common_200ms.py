from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.compare_common_200ms import (
    CommonComparisonError,
    validate_common_200ms,
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
    for game_index in range(4):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"{prefix}-{game_index:02d}-{label}"
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
    return DatasetSnapshot(revision="common-snapshot", samples=tuple(samples))


def _encoder(name: str) -> _FakeEncoder:
    return _FakeEncoder(
        EncoderProvenance(
            name=name,
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
        )
    )


class CompareCommon200msTest(unittest.TestCase):
    def test_matching_bundles_produce_paired_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "common")
            protocol = BenchmarkProtocol(
                seed=23, outer_splits=2, include_controls=True
            )
            m2d_bundle = run_short_contact_benchmark(
                protocol, snapshot, (_encoder("encoder-a"),), root / "a"
            )
            beats_bundle = run_short_contact_benchmark(
                protocol, snapshot, (_encoder("encoder-b"),), root / "b"
            )

            comparison = validate_common_200ms(
                {"encoder-a": m2d_bundle, "encoder-b": beats_bundle},
                root / "comparison",
            )

            self.assertTrue(comparison.summary["checks"]["protocols_compatible"])
            self.assertTrue(comparison.summary["checks"]["fold_assignments_identical"])
            self.assertTrue(
                comparison.summary["checks"]["prediction_cardinalities_identical"]
            )
            self.assertEqual(comparison.summary["n_common_samples"], 8)
            self.assertEqual(comparison.summary["n_paired_samples"], 8)

            common = comparison.common_metrics
            self.assertEqual(len(common), 6)
            self.assertEqual(
                set(common["condition"]),
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                    "event_selected_removed",
                    "removed_selected_removed",
                    "contact_specific_increment",
                },
            )
            self.assertTrue(
                (common["encoder-a_minus_encoder-b"].notna()).all()
            )
            increment_row = common[
                common["condition"].eq("contact_specific_increment")
            ].iloc[0]
            self.assertEqual(int(increment_row["n_paired_samples"]), 8)
            self.assertTrue((root / "comparison" / "common_metrics.csv").is_file())
            self.assertTrue(
                (root / "comparison" / "encoder-a_oof_predictions.csv").is_file()
            )
            self.assertTrue(
                (root / "comparison" / "encoder-b_oof_predictions.csv").is_file()
            )

    def test_fails_on_fold_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "fold")
            first = run_short_contact_benchmark(
                BenchmarkProtocol(seed=23, outer_splits=2, include_controls=True),
                snapshot,
                (_encoder("encoder-a"),),
                root / "a",
            )
            second = run_short_contact_benchmark(
                BenchmarkProtocol(seed=99, outer_splits=2, include_controls=True),
                snapshot,
                (_encoder("encoder-b"),),
                root / "b",
            )
            with self.assertRaises(CommonComparisonError) as context:
                validate_common_200ms(
                    {"encoder-a": first, "encoder-b": second}, root / "out"
                )
            self.assertIn("seed", str(context.exception))

    def test_fails_on_missing_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "controls")
            with_controls = run_short_contact_benchmark(
                BenchmarkProtocol(seed=23, outer_splits=2, include_controls=True),
                snapshot,
                (_encoder("encoder-a"),),
                root / "a",
            )
            without_controls = run_short_contact_benchmark(
                BenchmarkProtocol(seed=23, outer_splits=2, include_controls=False),
                snapshot,
                (_encoder("encoder-b"),),
                root / "b",
            )
            with self.assertRaises(CommonComparisonError) as context:
                validate_common_200ms(
                    {"encoder-a": with_controls, "encoder-b": without_controls},
                    root / "out",
                )
            self.assertIn("controls", str(context.exception))


if __name__ == "__main__":
    unittest.main()
