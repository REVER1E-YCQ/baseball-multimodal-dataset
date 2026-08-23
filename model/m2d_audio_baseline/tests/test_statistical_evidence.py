from __future__ import annotations

import json
import tempfile
import unittest
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.short_contact_benchmark import (
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderAdapter,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)
from scripts.statistical_evidence import compute_statistical_evidence


@dataclass
class _SignalEncoder(EncoderAdapter):
    """Fake encoder whose token energy follows the planted class signal."""

    name: str = "fake-a"
    signal_in: str = "event"

    provenance: EncoderProvenance = field(init=False)

    def __post_init__(self) -> None:
        self.provenance = EncoderProvenance(
            name=self.name,
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
            training_epochs=0,
        )

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        center = float(waveform[len(waveform) // 2])
        energy = float(np.mean(np.square(waveform)))
        token = np.asarray(
            [center, energy, abs(center), float(sample_rate) / 16_000.0]
        )
        return np.stack([token, token])


def _make_snapshot(
    root: Path,
    prefix: str,
    signal_in: str = "event",
    singleton_groups: bool = False,
) -> DatasetSnapshot:
    samples: list[SnapshotSample] = []
    for game_index in range(24):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"{prefix}-{game_index:02d}-{label}"
            waveform = np.zeros(16_000, dtype=np.float32)
            if signal_in == "event":
                waveform[8_000] = polarity
            elif signal_in == "background":
                waveform[4_800] = polarity
            group = (
                f"game-{game_index:02d}"
                if not singleton_groups
                else f"singleton-{uid}"
            )
            audio_path = root / "snapshot" / f"{uid}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio_path, 16_000, waveform)
            samples.append(
                SnapshotSample(
                    uid=uid,
                    label=label,
                    lineage_group_id=group,
                    audio_path=audio_path,
                    event_start=0.45,
                    event_end=0.55,
                )
            )
    return DatasetSnapshot(revision=f"{prefix}-snapshot", samples=tuple(samples))


def _run_bundles(
    root: Path,
    snapshot: DatasetSnapshot,
    signal_in: str,
) -> dict[str, object]:
    protocol = BenchmarkProtocol(
        seed=31, outer_splits=2, include_controls=True
    )
    first = run_short_contact_benchmark(
        protocol,
        snapshot,
        (_SignalEncoder(name="encoder-a", signal_in=signal_in),),
        root / "a",
    )
    second = run_short_contact_benchmark(
        protocol,
        snapshot,
        (_SignalEncoder(name="encoder-b", signal_in=signal_in),),
        root / "b",
    )
    return {"encoder-a": first, "encoder-b": second}


class StatisticalEvidenceTest(unittest.TestCase):
    def test_event_signal_yields_screening_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = _run_bundles(root, _make_snapshot(root, "signal"), "event")
            evidence = compute_statistical_evidence(
                bundles,
                root / "evidence",
                n_bootstrap=199,
                n_permutations=199,
                seed=41,
            )

            summary = evidence.summary
            self.assertTrue(
                summary["source_transfer_conclusive"]["encoder-a"]
            )
            self.assertTrue(
                summary["source_transfer_conclusive"]["encoder-b"]
            )
            for encoder in ("encoder-a", "encoder-b"):
                decision = summary["screening_decisions"][encoder]
                self.assertTrue(decision["screening_positive"])
                self.assertLess(decision["max_stat_familywise_p"], 0.05)
                self.assertGreater(decision["increment_ci_low"], 0.0)

            uncertainty = pd.read_csv(evidence.path("group_uncertainty.csv"))
            self.assertEqual(set(uncertainty["method"]), {"group_resample"})
            event_row = uncertainty[
                uncertainty["condition"].eq("event_selected_event")
            ].iloc[0]
            self.assertEqual(int(event_row["n_groups"]), 24)
            self.assertEqual(int(event_row["n_samples"]), 48)
            self.assertLessEqual(
                float(event_row["ci_low"]), float(event_row["ci_high"])
            )

            permutation = pd.read_csv(evidence.path("permutation_summary.csv"))
            self.assertEqual(len(permutation), 2)
            self.assertTrue((permutation["uncorrected_p"] < 0.05).all())
            self.assertTrue((permutation["max_stat_familywise_p"] < 0.05).all())
            scores = pd.read_csv(evidence.path("permutation_scores.csv"))
            self.assertEqual(len(scores), 2 * (199 + 1))

    def test_no_signal_yields_not_positive_but_full_numerics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "noise")
            # Pure noise: event windows carry no label-correlated signal.
            for sample in snapshot.samples:
                # A stable cross-process seed: built-in hash() is salted per
                # process, which made this noise test flaky.
                rng = np.random.default_rng(
                    zlib.crc32(sample.uid.encode("utf-8"))
                )
                wavfile.write(
                    sample.audio_path,
                    16_000,
                    (rng.standard_normal(16_000) * 0.05).astype(np.float32),
                )
            bundles = _run_bundles(root, snapshot, "noise")
            evidence = compute_statistical_evidence(
                bundles,
                root / "evidence",
                n_bootstrap=199,
                n_permutations=199,
                seed=43,
            )
            for encoder in ("encoder-a", "encoder-b"):
                decision = evidence.summary["screening_decisions"][encoder]
                self.assertFalse(decision["screening_positive"])
                self.assertTrue(decision["reasons"])
            self.assertTrue(
                (evidence.path("group_uncertainty.csv")).is_file()
            )
            self.assertTrue((evidence.path("paired_intervals.csv")).is_file())

    def test_deterministic_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root, "det")
            bundles = _run_bundles(root, snapshot, "event")
            first = compute_statistical_evidence(
                bundles, root / "evidence-1", n_bootstrap=99, n_permutations=99
            )
            second = compute_statistical_evidence(
                bundles, root / "evidence-2", n_bootstrap=99, n_permutations=99
            )
            first_table = pd.read_csv(first.path("permutation_summary.csv"))
            second_table = pd.read_csv(second.path("permutation_summary.csv"))
            pd.testing.assert_frame_equal(first_table, second_table)

    def test_singleton_groups_mark_source_transfer_inconclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(
                root, "singleton", singleton_groups=True
            )
            bundles = _run_bundles(root, snapshot, "event")
            evidence = compute_statistical_evidence(
                bundles,
                root / "evidence",
                n_bootstrap=99,
                n_permutations=99,
            )
            self.assertFalse(
                evidence.summary["source_transfer_conclusive"]["encoder-a"]
            )
            self.assertFalse(
                evidence.summary["source_transfer_conclusive"]["encoder-b"]
            )

    def test_mixed_label_games_supported_in_permutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundles = _run_bundles(root, _make_snapshot(root, "mixed"), "event")
            evidence = compute_statistical_evidence(
                bundles,
                root / "evidence",
                n_bootstrap=99,
                n_permutations=99,
                seed=47,
            )
            scores = pd.read_csv(evidence.path("permutation_scores.csv"))
            # Every game contains both labels; permutation scores stay finite
            # and each fold's class totals are preserved by construction.
            self.assertTrue(np.isfinite(scores["balanced_accuracy"]).all())
            summary = json.loads(
                (evidence.path("summary.json")).read_text(encoding="utf-8")
            )
            self.assertIn(
                "mixed_label_games", summary["permutation_preserves"]
            )
            self.assertTrue(summary["screening_decisions"])


    def test_calibrated_bundle_uses_fixed_rule_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = BenchmarkProtocol(
                seed=91,
                outer_splits=2,
                inner_splits=2,
                c_grid=(0.001, 0.01, 0.1),
                include_controls=True,
                calibrate_threshold=True,
            )
            snapshot = _make_snapshot(root, "cal-evidence")
            bundles = {
                "encoder-a": run_short_contact_benchmark(
                    protocol,
                    snapshot,
                    (_SignalEncoder(name="encoder-a", signal_in="event"),),
                    root / "a",
                ),
                "encoder-b": run_short_contact_benchmark(
                    protocol,
                    snapshot,
                    (_SignalEncoder(name="encoder-b", signal_in="event"),),
                    root / "b",
                ),
            }
            evidence = compute_statistical_evidence(
                bundles,
                root / "evidence",
                n_bootstrap=49,
                n_permutations=49,
            )
            # Dual-rule bundles must be reduced to the locked fixed-0.5 rows
            # before statistical evidence; permutation scores stay finite.
            scores = pd.read_csv(evidence.path("permutation_scores.csv"))
            self.assertTrue(np.isfinite(scores["balanced_accuracy"]).all())
            self.assertTrue(evidence.summary["screening_decisions"])
if __name__ == "__main__":
    unittest.main()

