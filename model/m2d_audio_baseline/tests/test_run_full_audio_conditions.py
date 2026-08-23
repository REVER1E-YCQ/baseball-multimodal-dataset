from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.run_full_audio_conditions import (
    build_attribution_summary,
    build_condition_table,
    build_lead_comparison,
    compute_duration_shortcut,
)
from scripts.short_contact_benchmark import (
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)


@dataclass
class FakeEncoder:
    provenance: EncoderProvenance = field(
        default_factory=lambda: EncoderProvenance(
            name="fake-full-audio",
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


def _make_snapshot(root: Path) -> DatasetSnapshot:
    sample_rate = 16_000
    samples: list[SnapshotSample] = []
    for game_index in range(6):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"game-{game_index:02d}-{label}"
            waveform = np.zeros(sample_rate * 8, dtype=np.float32)
            waveform[int(sample_rate * 3.5)] = polarity
            audio_path = root / "snapshot" / f"{uid}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio_path, sample_rate, waveform)
            samples.append(
                SnapshotSample(
                    uid=uid,
                    label=label,
                    lineage_group_id=f"game-{game_index:02d}",
                    audio_path=audio_path,
                    event_start=3.45,
                    event_end=3.55,
                )
            )
    return DatasetSnapshot(
        revision="synthetic-snapshot-1", samples=tuple(samples)
    )


class RunFullAudioConditionsTest(unittest.TestCase):
    def _run_bundle(self, root: Path):
        return run_short_contact_benchmark(
            BenchmarkProtocol(
                seed=20260805,
                outer_splits=3,
                logistic_c=0.1,
                include_controls=True,
                window_conditions=(200, 500),
                non_centered_windows=(
                    "full_audio",
                    "post_contact_4000ms",
                    "post_contact_1000ms",
                    "pre_contact_1000ms",
                ),
            ),
            _make_snapshot(root),
            (FakeEncoder(),),
            root / "artifacts",
        )

    def test_condition_table_covers_all_lead_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._run_bundle(root)
            metrics = pd.read_csv(bundle.root / "metrics.csv")
            table = build_condition_table(metrics, "fake-full-audio")

            self.assertEqual(
                list(table["condition"]),
                [
                    "event_200ms",
                    "event_500ms",
                    "post_contact_4000ms",
                    "full_audio",
                    "post_contact_1000ms",
                    "pre_contact_1000ms",
                ],
            )
            for row in table.itertuples(index=False):
                self.assertEqual(int(row.eligible_samples), 12)
                self.assertFalse(np.isnan(row.event_balanced_accuracy))
            # Centred windows keep the negative-control columns.
            for condition in ("event_200ms", "event_500ms"):
                row = table[table["condition"].eq(condition)].iloc[0]
                self.assertFalse(
                    np.isnan(row.strict_pre_balanced_accuracy)
                )
                self.assertFalse(np.isnan(row.contact_specific_increment))
            # Non-centred conditions have no negative-control chain.
            for condition in (
                "full_audio",
                "post_contact_4000ms",
                "post_contact_1000ms",
                "pre_contact_1000ms",
            ):
                row = table[table["condition"].eq(condition)].iloc[0]
                self.assertTrue(np.isnan(row.strict_pre_balanced_accuracy))
                self.assertTrue(np.isnan(row.contact_specific_increment))
            # Delta is measured against the 0.5 s reproduction baseline.
            baseline = table[
                table["condition"].eq("event_500ms")
            ].iloc[0]
            self.assertTrue(np.isnan(baseline.vs_500ms_delta))
            full = table[table["condition"].eq("full_audio")].iloc[0]
            self.assertAlmostEqual(
                full.vs_500ms_delta,
                full.event_balanced_accuracy - baseline.event_balanced_accuracy,
            )

    def test_lead_comparison_maps_the_lead_claims(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = self._run_bundle(root)
            metrics = pd.read_csv(bundle.root / "metrics.csv")
            table = build_condition_table(metrics, "fake-full-audio")
            comparison = build_lead_comparison(table)

            self.assertEqual(
                list(comparison["lead_condition"]),
                ["0.5s_contact", "4s_window", "full_audio"],
            )
            self.assertEqual(
                list(comparison["our_condition"]),
                ["event_500ms", "post_contact_4000ms", "full_audio"],
            )
            self.assertEqual(
                list(comparison["lead_ungrouped_accuracy"]), [73, 78, 88]
            )
            self.assertTrue(
                comparison["note"].iloc[0].startswith(
                    "lead splits randomly"
                )
            )

    def test_condition_table_fails_on_missing_condition(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "encoder": "fake-full-audio",
                    "condition": "event_200ms",
                    "window_ms": 200,
                    "decision_rule": "fixed_0.5",
                    "balanced_accuracy": 0.5,
                    "roc_auc": 0.5,
                    "eligible_samples": 12,
                    "lineage_groups": 6,
                }
            ]
        )
        with self.assertRaises(ValueError) as context:
            build_condition_table(metrics, "fake-full-audio")
        self.assertIn("event_200ms", str(context.exception))

    def test_attribution_summary_decomposes_the_full_audio_gain(self) -> None:
        table = pd.DataFrame(
            [
                {
                    "condition": "event_500ms",
                    "event_balanced_accuracy": 0.583,
                },
                {
                    "condition": "pre_contact_1000ms",
                    "event_balanced_accuracy": 0.570,
                },
                {
                    "condition": "post_contact_1000ms",
                    "event_balanced_accuracy": 0.694,
                },
                {
                    "condition": "post_contact_4000ms",
                    "event_balanced_accuracy": 0.705,
                },
                {
                    "condition": "full_audio",
                    "event_balanced_accuracy": 0.777,
                },
            ]
        )
        summary = build_attribution_summary(table)

        self.assertEqual(summary["baseline_event_500ms"], 0.583)
        self.assertEqual(summary["pre_contact_gain"], 0.570 - 0.583)
        self.assertEqual(summary["post_contact_1s_gain"], 0.694 - 0.583)
        self.assertEqual(summary["full_audio_gain"], 0.777 - 0.583)
        self.assertEqual(summary["gain_beyond_4s"], 0.777 - 0.705)
        self.assertEqual(summary["conclusion"], "gain_lives_after_contact")

    def test_attribution_summary_flags_unclear_attribution(self) -> None:
        table = pd.DataFrame(
            [
                {"condition": "event_500ms", "event_balanced_accuracy": 0.60},
                {
                    "condition": "pre_contact_1000ms",
                    "event_balanced_accuracy": 0.62,
                },
                {
                    "condition": "post_contact_1000ms",
                    "event_balanced_accuracy": 0.63,
                },
                {
                    "condition": "post_contact_4000ms",
                    "event_balanced_accuracy": 0.63,
                },
                {
                    "condition": "full_audio",
                    "event_balanced_accuracy": 0.64,
                },
            ]
        )
        summary = build_attribution_summary(table)
        self.assertEqual(summary["conclusion"], "no_clear_attribution")

    def test_duration_shortcut_measures_clip_length_confound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 16_000
            samples: list[SnapshotSample] = []
            for game_index in range(6):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"game-{game_index:02d}-{label}"
                    seconds = 8 if label == "fly_ball" else 4
                    waveform = np.zeros(
                        sample_rate * seconds, dtype=np.float32
                    )
                    waveform[int(sample_rate * 2.5)] = polarity
                    audio_path = root / "snapshot" / f"{uid}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    wavfile.write(audio_path, sample_rate, waveform)
                    samples.append(
                        SnapshotSample(
                            uid=uid,
                            label=label,
                            lineage_group_id=f"game-{game_index:02d}",
                            audio_path=audio_path,
                            event_start=2.45,
                            event_end=2.55,
                        )
                    )
            snapshot = DatasetSnapshot(
                revision="synthetic-snapshot-1", samples=tuple(samples)
            )
            shortcut = compute_duration_shortcut(
                snapshot, outer_splits=3, seed=1
            )
            self.assertGreaterEqual(
                shortcut["duration_shortcut_balanced_accuracy"], 0.99
            )
            self.assertGreaterEqual(shortcut["duration_shortcut_roc_auc"], 0.99)
            self.assertIn("confound", shortcut["note"])

    def test_duration_shortcut_is_chance_with_equal_lengths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root)
            shortcut = compute_duration_shortcut(
                snapshot, outer_splits=3, seed=1
            )
            self.assertLessEqual(
                shortcut["duration_shortcut_balanced_accuracy"], 0.6
            )
            self.assertLessEqual(shortcut["duration_shortcut_roc_auc"], 0.6)


if __name__ == "__main__":
    unittest.main()
