from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.encoder_fusion import (
    evaluate_fusion,
    load_pooled_table,
    load_source_table,
    verify_fold_consistency,
)
from scripts.short_contact_benchmark import (
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)


@dataclass
class FakePooledEncoder:
    provenance: EncoderProvenance = field(
        default_factory=lambda: EncoderProvenance(
            name="fake-pooled",
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
            training_epochs=0,
        )
    )

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        center = float(waveform[len(waveform) // 2])
        energy = float(np.mean(np.square(waveform)))
        token = np.asarray(
            [center, energy, abs(center), float(sample_rate) / 16_000.0]
        )
        return np.stack([token, token])


@dataclass
class FakeAttentionEncoder:
    provenance: EncoderProvenance = field(
        default_factory=lambda: EncoderProvenance(
            name="fake-attention",
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
            training_epochs=0,
        )
    )

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        center = float(waveform[len(waveform) // 2])
        energy = float(np.mean(np.square(waveform)))
        token = np.asarray(
            [center, energy, abs(center), float(sample_rate) / 16_000.0]
        )
        return np.stack([token * (index + 1) for index in range(3)])


def _make_snapshot(root: Path) -> DatasetSnapshot:
    sample_rate = 16_000
    samples: list[SnapshotSample] = []
    for game_index in range(6):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"game-{game_index:02d}-{label}"
            waveform = np.zeros(sample_rate, dtype=np.float32)
            waveform[sample_rate // 2] = polarity
            audio_path = root / "snapshot" / f"{uid}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio_path, sample_rate, waveform)
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
    return DatasetSnapshot(
        revision="synthetic-snapshot-1", samples=tuple(samples)
    )


class EncoderFusionTest(unittest.TestCase):
    def _bundles(self, root: Path):
        snapshot = _make_snapshot(root)
        protocol = BenchmarkProtocol(
            seed=20260805,
            outer_splits=3,
            logistic_c=0.1,
            include_controls=True,
        )
        pooled = run_short_contact_benchmark(
            protocol,
            snapshot,
            (FakePooledEncoder(),),
            root / "pooled",
        )
        attention = run_short_contact_benchmark(
            BenchmarkProtocol(
                seed=20260805,
                outer_splits=3,
                logistic_c=0.1,
                include_controls=True,
                pooling="attention",
            ),
            snapshot,
            (FakeAttentionEncoder(),),
            root / "attention",
        )
        return pooled, attention, snapshot

    def _feature_path(self, bundle, name):
        return next(
            bundle.path(item)
            for item in bundle.artifact_names
            if item.startswith("features/") and name in item
        )

    def test_fusion_evaluates_sources_and_concatenation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pooled, attention, _snapshot = self._bundles(root)

            pooled_path = self._feature_path(pooled, "fake-pooled")
            attention_path = self._feature_path(attention, "fake-attention")
            # The attention bundle writes a token table.
            pooled_table = load_pooled_table(pooled_path)
            attention_table = load_source_table(attention_path, True)
            folds = pd.read_csv(
                pooled.path("fold_assignments")
            )
            verify_fold_consistency(
                folds,
                pd.read_csv(attention.path("fold_assignments")),
            )

            result = evaluate_fusion(
                ("pooled", "attention"),
                (pooled_table, attention_table),
                (False, True),
                folds,
                c_grid=(0.001, 0.01, 0.1),
                seed=20260805,
                inner_splits=2,
            )
            table = result["table"]
            self.assertEqual(len(table), 3)
            self.assertEqual(
                set(table["name"]), {"pooled", "attention", "pooled+attention"}
            )
            for row in table.itertuples(index=False):
                self.assertEqual(row.condition, "event_selected_event")
                self.assertEqual(int(row.eligible_samples), 12)
                self.assertGreater(float(row.balanced_accuracy), 0.9)
                self.assertLess(
                    float(row.pre_transfer_balanced_accuracy), 0.7
                )
                self.assertGreater(
                    float(row.contact_specific_increment), 0.2
                )
            self.assertEqual(result["summary"]["event_eligible_samples"], 12)
            self.assertEqual(len(result["selections"]), 3 * 3)

    def test_fold_mismatch_fails_visibly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pooled, attention, _snapshot = self._bundles(root)
            folds_a = pd.read_csv(pooled.path("fold_assignments"))
            folds_b = pd.read_csv(attention.path("fold_assignments"))
            folds_b.loc[0, "outer_fold"] = (
                int(folds_b.loc[0, "outer_fold"]) % 2 + 1
            )
            with self.assertRaises(ValueError) as context:
                verify_fold_consistency(folds_a, folds_b)
            self.assertIn("disagree", str(context.exception))

    def test_sample_set_is_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pooled, attention, _snapshot = self._bundles(root)
            pooled_path = self._feature_path(pooled, "fake-pooled")
            attention_path = self._feature_path(attention, "fake-attention")
            pooled_table = load_pooled_table(pooled_path)
            attention_table = load_source_table(attention_path, True)
            # Drop one uid from the pooled source: the fusion must use the
            # intersection, not either source's full set.
            pooled_table = {
                key: value
                for key, value in pooled_table.items()
                if key[0] != "game-00-fly_ball"
            }
            result = evaluate_fusion(
                ("pooled", "attention"),
                (pooled_table, attention_table),
                (False, True),
                pd.read_csv(pooled.path("fold_assignments")),
                c_grid=(0.001, 0.01, 0.1),
                seed=20260805,
                inner_splits=2,
            )
            self.assertEqual(
                result["summary"]["event_eligible_samples"], 11
            )
            self.assertTrue(
                (result["table"]["eligible_samples"] == 11).all()
            )

    def test_attention_source_table_shape_is_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pooled, attention, _snapshot = self._bundles(root)
            attention_path = self._feature_path(attention, "fake-attention")
            table = load_source_table(attention_path, True)
            self.assertEqual(
                table[("game-00-fly_ball", "event_200ms")].shape, (3, 4)
            )


if __name__ == "__main__":
    unittest.main()
