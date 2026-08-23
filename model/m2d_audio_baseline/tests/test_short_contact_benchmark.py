from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.cached_attention_controls import (
    reevaluate_cached_attention_controls,
)
from scripts.short_contact_benchmark import (
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)


@dataclass
class FakeContactEncoder:
    provenance: EncoderProvenance = field(
        default_factory=lambda: EncoderProvenance(
            name="fake-contact",
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
            training_epochs=0,
        )
    )
    received_sample_counts: list[int] = field(default_factory=list)
    signal_in: str = "event"

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        self.received_sample_counts.append(len(waveform))
        center = float(waveform[len(waveform) // 2])
        energy = float(np.mean(np.square(waveform)))
        token = np.asarray([center, energy, abs(center), float(sample_rate) / 16_000.0])
        return np.stack([token, token])


class ShortContactBenchmarkTest(unittest.TestCase):
    def test_runs_synthetic_snapshot_through_the_public_interface(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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

            encoder = FakeContactEncoder()
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(seed=20260805, outer_splits=3, logistic_c=0.1),
                DatasetSnapshot(revision="synthetic-snapshot-1", samples=tuple(samples)),
                (encoder,),
                root / "artifacts",
            )

            expected_artifacts = {
                "artifact_bundle",
                "exclusions",
                "features/fake-contact",
                "fold_assignments",
                "metrics",
                "oof_predictions",
                "protocol",
                "selections",
                "snapshot_audit",
                "window_manifest",
            }
            self.assertEqual(set(bundle.artifact_names), expected_artifacts)
            self.assertTrue(all(bundle.path(name).is_file() for name in expected_artifacts))

            audit = json.loads(bundle.path("snapshot_audit").read_text(encoding="utf-8"))
            self.assertEqual(audit["sample_count"], 12)
            self.assertEqual(audit["label_counts"], {"fly_ball": 6, "ground_ball": 6})
            self.assertEqual(audit["lineage_group_count"], 6)

            windows = pd.read_csv(bundle.path("window_manifest"))
            self.assertEqual(len(windows), 12)
            self.assertEqual(set(windows["window_name"]), {"event_200ms"})
            self.assertTrue((windows["window_duration"] == 0.2).all())
            self.assertTrue((windows["wav_boundary_padding_samples"] == 0).all())
            self.assertTrue(all(not Path(value).is_absolute() for value in windows["window_path"]))
            self.assertEqual(set(encoder.received_sample_counts), {3_200})

            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 12)
            self.assertFalse(predictions[["encoder", "uid"]].duplicated().any())
            group_fold_counts = predictions.groupby("lineage_group_id")["outer_fold"].nunique()
            self.assertTrue((group_fold_counts == 1).all())

            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertEqual(metrics.loc[0, "primary_metric"], "balanced_accuracy")
            self.assertGreater(float(metrics.loc[0, "balanced_accuracy"]), 0.95)

            protocol = json.loads(bundle.path("protocol").read_text(encoding="utf-8"))
            self.assertEqual(protocol["dataset"]["revision"], "synthetic-snapshot-1")
            self.assertEqual(protocol["window_conditions"], ["event_200ms"])
            self.assertEqual(protocol["detector"], "absolute_amplitude_peak_within_event_interval")
            self.assertEqual(protocol["normalization"], "snapshot_level")
            self.assertEqual(protocol["pooling"], "valid_final_layer_token_mean")
            self.assertEqual(protocol["classifier"]["name"], "balanced_l2_logistic_regression")
            self.assertEqual(protocol["fold_policy"]["outer_splits"], 3)
            self.assertEqual(protocol["encoders"][0]["upstream_revision"], "fake-revision-1")
            self.assertEqual(protocol["encoders"][0]["checkpoint_sha256"], "fake-checkpoint-sha256")
            self.assertEqual(protocol["encoders"][0]["training_epochs"], 0)
            self.assertNotIn(str(root.resolve()), json.dumps(protocol))

    def test_artifact_identity_is_portable_and_protocol_sensitive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def build_snapshot(location: Path) -> DatasetSnapshot:
                samples: list[SnapshotSample] = []
                for game_index in range(4):
                    for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                        uid = f"portable-{game_index:02d}-{label}"
                        waveform = np.zeros(16_000, dtype=np.float32)
                        waveform[8_000] = polarity
                        audio_path = location / f"{uid}.wav"
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
                return DatasetSnapshot(
                    revision="portable-snapshot-1",
                    samples=tuple(samples),
                )

            protocol = BenchmarkProtocol(seed=11, outer_splits=2, logistic_c=0.1)
            first = run_short_contact_benchmark(
                protocol,
                build_snapshot(root / "machine-a"),
                (FakeContactEncoder(),),
                root / "outputs-a",
            )
            relocated = run_short_contact_benchmark(
                protocol,
                build_snapshot(root / "machine-b"),
                (FakeContactEncoder(),),
                root / "outputs-b",
            )
            changed_protocol = run_short_contact_benchmark(
                BenchmarkProtocol(seed=12, outer_splits=2, logistic_c=0.1),
                build_snapshot(root / "machine-c"),
                (FakeContactEncoder(),),
                root / "outputs-c",
            )

            self.assertEqual(first.artifact_id, relocated.artifact_id)
            self.assertNotEqual(first.artifact_id, changed_protocol.artifact_id)

    def test_nested_c_selection_emits_selections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"nested-{game_index:02d}-{label}"
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
            protocol = BenchmarkProtocol(
                seed=5,
                outer_splits=2,
                inner_splits=2,
                c_grid=(0.001, 0.01, 0.1),
            )
            bundle = run_short_contact_benchmark(
                protocol,
                DatasetSnapshot(revision="nested-snapshot", samples=tuple(samples)),
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            selections = pd.read_csv(bundle.path("selections"))
            self.assertEqual(len(selections), 2)
            self.assertEqual(set(selections["outer_fold"]), {0, 1})
            self.assertTrue(
                selections["selected_C"].isin((0.001, 0.01, 0.1)).all()
            )
            self.assertTrue(
                selections["inner_scores_json"].map(json.loads).apply(
                    lambda rows: {row["C"] for row in rows}
                    == {0.001, 0.01, 0.1}
                ).all()
            )
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 8)
            self.assertFalse(predictions[["encoder", "uid"]].duplicated().any())
            protocol_doc = json.loads(
                bundle.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                protocol_doc["classifier"]["C_selection"], "inner_grouped_cv"
            )
            self.assertEqual(
                protocol_doc["classifier"]["C_grid"], [0.001, 0.01, 0.1]
            )

    def test_records_exclusion_reasons_for_ineligible_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_sample(
                uid: str, label: str, group: str, seconds: float, start: float, end: float
            ) -> SnapshotSample:
                waveform = np.zeros(int(16_000 * seconds), dtype=np.float32)
                waveform[min(int(0.055 * 16_000), len(waveform) - 1)] = 1.0
                audio_path = root / "snapshot" / f"{uid}.wav"
                audio_path.parent.mkdir(parents=True, exist_ok=True)
                wavfile.write(audio_path, 16_000, waveform)
                return SnapshotSample(
                    uid=uid,
                    label=label,
                    lineage_group_id=group,
                    audio_path=audio_path,
                    event_start=start,
                    event_end=end,
                )

            samples = [
                write_sample("good-fly-1", "fly_ball", "game-00", 1.0, 0.45, 0.55),
                write_sample("good-ground-1", "ground_ball", "game-01", 1.0, 0.45, 0.55),
                write_sample("good-fly-2", "fly_ball", "game-02", 1.0, 0.45, 0.55),
                write_sample("good-ground-2", "ground_ball", "game-03", 1.0, 0.45, 0.55),
                write_sample("short-fly", "fly_ball", "game-04", 0.1, 0.04, 0.05),
                write_sample("edge-ground", "ground_ball", "game-05", 0.21, 0.05, 0.06),
            ]
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(seed=7, outer_splits=2),
                DatasetSnapshot(revision="exclusion-snapshot", samples=tuple(samples)),
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            exclusions = pd.read_csv(bundle.path("exclusions"))
            self.assertEqual(set(exclusions["uid"]), {"short-fly", "edge-ground"})
            self.assertEqual(
                set(exclusions["reason"]),
                {"audio_shorter_than_window", "window_not_exact"},
            )
            windows = pd.read_csv(bundle.path("window_manifest"))
            self.assertEqual(
                set(windows["uid"]),
                {"good-fly-1", "good-ground-1", "good-fly-2", "good-ground-2"},
            )
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 4)
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertEqual(int(metrics.loc[0, "eligible_samples"]), 4)

    def test_reuses_features_and_invalidates_on_provenance_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(2):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"cache-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(revision="cache-snapshot", samples=tuple(samples))
            protocol = BenchmarkProtocol(seed=11, outer_splits=2)
            out_dir = root / "artifacts"

            encoder = FakeContactEncoder()
            first = run_short_contact_benchmark(protocol, snapshot, (encoder,), out_dir)
            calls_after_first = len(encoder.received_sample_counts)
            self.assertEqual(calls_after_first, 4)

            second = run_short_contact_benchmark(protocol, snapshot, (encoder,), out_dir)
            self.assertEqual(second.artifact_id, first.artifact_id)
            self.assertEqual(len(encoder.received_sample_counts), calls_after_first)
            marker = first.root / "features" / "fake-contact.provenance.json"
            self.assertTrue(marker.is_file())

            changed = FakeContactEncoder(
                EncoderProvenance(
                    name="fake-contact",
                    upstream_revision="fake-revision-1",
                    checkpoint_sha256="changed-checkpoint-sha256",
                    precision="fp32",
                    token_dimension=4,
                )
            )
            third = run_short_contact_benchmark(protocol, snapshot, (changed,), out_dir)
            self.assertNotEqual(third.artifact_id, first.artifact_id)
            self.assertEqual(len(changed.received_sample_counts), 4)

    def test_controls_emit_all_negative_control_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"control-{game_index:02d}-{label}"
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
            protocol = BenchmarkProtocol(
                seed=13,
                outer_splits=2,
                inner_splits=2,
                c_grid=(0.001, 0.01, 0.1),
                include_controls=True,
            )
            bundle = run_short_contact_benchmark(
                protocol,
                DatasetSnapshot(revision="control-snapshot", samples=tuple(samples)),
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            windows = pd.read_csv(bundle.path("window_manifest"))
            self.assertEqual(
                set(windows["window_name"]),
                {"event_200ms", "pre_200ms", "removed_200ms"},
            )
            self.assertTrue((windows["wav_boundary_padding_samples"] == 0).all())
            removed = windows[windows["window_name"].eq("removed_200ms")]
            self.assertTrue(
                removed["removed_source_start"].notna().all()
                and removed["removed_dest_end"].notna().all()
            )
            self.assertEqual(set(removed["removed_crossfade_seconds"]), {0.005})

            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(
                set(predictions["condition"]),
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                    "event_selected_removed",
                    "removed_selected_removed",
                },
            )
            per_condition_counts = (
                predictions.groupby("condition")["uid"].nunique()
            )
            self.assertTrue((per_condition_counts == 8).all())
            group_fold_counts = predictions.groupby(
                ["condition", "lineage_group_id"]
            )["outer_fold"].nunique()
            self.assertTrue((group_fold_counts == 1).all())

            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertEqual(
                set(metrics["condition"]),
                set(predictions["condition"]) | {"contact_specific_increment"},
            )
            increment = metrics[
                metrics["condition"].eq("contact_specific_increment")
            ].iloc[0]
            event_ba = float(
                metrics.loc[
                    metrics["condition"].eq("event_selected_event"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            pre_ba = float(
                metrics.loc[
                    metrics["condition"].eq("event_selected_pre"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            self.assertAlmostEqual(
                float(increment["balanced_accuracy"]), event_ba - pre_ba
            )
            self.assertEqual(int(increment["eligible_samples"]), 8)

            selections = pd.read_csv(bundle.path("selections"))
            self.assertEqual(
                set(selections["condition"]), {"event", "pre", "removed"}
            )

    def test_attention_event_transform_is_reused_for_negative_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class OrthogonalControlEncoder(FakeContactEncoder):
                def encode_tokens(self, waveform, sample_rate):
                    center = float(waveform[len(waveform) // 2])
                    marker = float(waveform[600])
                    if abs(center) > 0.75:
                        polarity = 1.0 if center > 0 else -1.0
                        amplitude = 2.0 + 0.5 * polarity
                        return np.asarray(
                            [[amplitude, 0.0, 0.0, 0.0],
                             [-amplitude, 0.0, 0.0, 0.0]]
                        )
                    if abs(center) > 0.25:
                        polarity = 1.0 if center > 0 else -1.0
                        first = 2.0 + 0.5 * polarity
                        second = 2.0 - 0.5 * polarity
                        return np.asarray(
                            [[first, 4.0, 0.0, 0.0],
                             [second, -4.0, 0.0, 0.0]]
                        )
                    polarity = 1.0 if marker > 0 else -1.0
                    first = 2.0 + 0.5 * polarity
                    second = 2.0 - 0.5 * polarity
                    return np.asarray(
                        [[first, 0.0, 4.0, 0.0],
                         [second, 0.0, -4.0, 0.0]]
                    )

            samples: list[SnapshotSample] = []
            for game_index in range(12):
                for label, polarity in (
                    ("fly_ball", 1.0),
                    ("ground_ball", -1.0),
                ):
                    uid = f"orthogonal-{game_index:02d}-{label}"
                    waveform = np.zeros(16_000, dtype=np.float32)
                    waveform[8_000] = polarity
                    waveform[4_800] = 0.5 * polarity
                    waveform[7_000] = 0.5 * polarity
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

            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=101,
                    outer_splits=3,
                    logistic_c=1.0,
                    pooling="attention",
                    include_controls=True,
                ),
                DatasetSnapshot(
                    revision="orthogonal-control-snapshot",
                    samples=tuple(samples),
                ),
                (OrthogonalControlEncoder(),),
                root / "artifacts",
            )

            predictions = pd.read_csv(bundle.path("oof_predictions"))
            representation_roles = (
                predictions.groupby("condition")[[
                    "representation_fit_window",
                    "representation_apply_window",
                ]]
                .first()
                .to_dict("index")
            )
            self.assertEqual(
                representation_roles["event_selected_pre"],
                {
                    "representation_fit_window": "event_200ms",
                    "representation_apply_window": "pre_200ms",
                },
            )
            self.assertEqual(
                representation_roles["event_selected_removed"],
                {
                    "representation_fit_window": "event_200ms",
                    "representation_apply_window": "removed_200ms",
                },
            )
            self.assertEqual(
                representation_roles["pre_selected_pre"],
                {
                    "representation_fit_window": "pre_200ms",
                    "representation_apply_window": "pre_200ms",
                },
            )
            self.assertEqual(
                representation_roles["removed_selected_removed"],
                {
                    "representation_fit_window": "removed_200ms",
                    "representation_apply_window": "removed_200ms",
                },
            )

            metrics = pd.read_csv(bundle.path("metrics")).set_index(
                "condition"
            )
            self.assertGreater(
                float(metrics.loc["event_selected_event", "balanced_accuracy"]),
                0.95,
            )
            self.assertLess(
                float(metrics.loc["event_selected_pre", "balanced_accuracy"]),
                0.6,
            )
            self.assertLess(
                float(metrics.loc[
                    "event_selected_removed", "balanced_accuracy"
                ]),
                0.6,
            )
            self.assertGreater(
                float(metrics.loc["pre_selected_pre", "balanced_accuracy"]),
                0.95,
            )
            self.assertGreater(
                float(metrics.loc[
                    "removed_selected_removed", "balanced_accuracy"
                ]),
                0.95,
            )
            protocol = json.loads(
                bundle.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                protocol["attention_control_transform_policy"],
                "event_fitted_transfer_v1",
            )

    def test_cached_attention_controls_reevaluate_without_encoding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (
                    ("fly_ball", 1.0),
                    ("ground_ball", -1.0),
                ):
                    uid = f"cached-{game_index:02d}-{label}"
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
            encoder = FakeContactEncoder()
            current = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=103,
                    outer_splits=2,
                    pooling="attention",
                    include_controls=True,
                ),
                DatasetSnapshot(
                    revision="cached-control-snapshot",
                    samples=tuple(samples),
                ),
                (encoder,),
                root / "current",
            )
            encoded_windows = len(encoder.received_sample_counts)
            legacy_root = root / "legacy-bundle"
            shutil.copytree(current.root, legacy_root)
            legacy_protocol_path = legacy_root / "protocol.json"
            legacy_protocol = json.loads(
                legacy_protocol_path.read_text(encoding="utf-8")
            )
            legacy_protocol["artifact_id"] = "legacy-attention-control"
            legacy_protocol.pop("attention_control_transform_policy")
            legacy_protocol_path.write_text(
                json.dumps(legacy_protocol), encoding="utf-8"
            )

            reevaluated = reevaluate_cached_attention_controls(
                legacy_root,
                root / "reevaluated",
            )

            self.assertEqual(len(encoder.received_sample_counts), encoded_windows)
            self.assertEqual(reevaluated.artifact_id, current.artifact_id)
            reevaluated_protocol = json.loads(
                reevaluated.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                reevaluated_protocol["attention_control_transform_policy"],
                "event_fitted_transfer_v1",
            )
            provenance = json.loads(
                reevaluated.path("reevaluation_provenance").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                provenance["source_artifact_id"],
                "legacy-attention-control",
            )
            reevaluated_windows = pd.read_csv(
                reevaluated.path("window_manifest")
            )
            self.assertTrue(
                all(
                    (reevaluated.root / relative_path).is_file()
                    for relative_path in reevaluated_windows["window_path"]
                )
            )
            current_metrics = pd.read_csv(current.path("metrics"))
            reevaluated_metrics = pd.read_csv(reevaluated.path("metrics"))
            pd.testing.assert_frame_equal(current_metrics, reevaluated_metrics)

    def test_event_only_signal_and_background_confound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def make_snapshot(signal_in: str) -> DatasetSnapshot:
                plant_event = signal_in == "event"
                samples: list[SnapshotSample] = []
                for game_index in range(4):
                    for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                        uid = f"{signal_in}-{game_index:02d}-{label}"
                        waveform = np.zeros(16_000, dtype=np.float32)
                        # The event-centred impulse carries the class polarity
                        # only in the event-only scenario.
                        if plant_event:
                            waveform[8_000] = polarity
                        # The strict-pre background carries the class polarity
                        # when the confound is planted in the background. The
                        # pre window spans [0.2 s, 0.4 s], so its centre is at
                        # sample 4800.
                        if signal_in == "background":
                            waveform[4_800] = polarity
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
                return DatasetSnapshot(
                    revision=f"{signal_in}-snapshot", samples=tuple(samples)
                )

            protocol = BenchmarkProtocol(
                seed=17, outer_splits=2, include_controls=True
            )
            bundle = run_short_contact_benchmark(
                protocol,
                make_snapshot("event"),
                (FakeContactEncoder(signal_in="event"),),
                root / "event-artifacts",
            )
            metrics = pd.read_csv(bundle.path("metrics"))
            event_ba = float(
                metrics.loc[
                    metrics["condition"].eq("event_selected_event"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            pre_ba = float(
                metrics.loc[
                    metrics["condition"].eq("event_selected_pre"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            self.assertGreater(event_ba, 0.95)
            self.assertLess(pre_ba, 0.6)

            bundle = run_short_contact_benchmark(
                protocol,
                make_snapshot("background"),
                (FakeContactEncoder(signal_in="background"),),
                root / "background-artifacts",
            )
            metrics = pd.read_csv(bundle.path("metrics"))
            pre_ba = float(
                metrics.loc[
                    metrics["condition"].eq("pre_selected_pre"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            event_ba = float(
                metrics.loc[
                    metrics["condition"].eq("event_selected_event"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            self.assertGreater(pre_ba, 0.95)
            self.assertLess(event_ba, 0.6)

    def test_sensitivity_durations_emit_comparable_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"sens-{game_index:02d}-{label}"
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
            protocol = BenchmarkProtocol(
                seed=53,
                outer_splits=2,
                window_conditions=(50, 100, 200),
                include_controls=True,
            )
            bundle = run_short_contact_benchmark(
                protocol,
                DatasetSnapshot(revision="sensitivity-snapshot", samples=tuple(samples)),
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            windows = pd.read_csv(bundle.path("window_manifest"))
            self.assertEqual(
                set(windows["window_name"]),
                {
                    "event_050ms",
                    "pre_050ms",
                    "event_100ms",
                    "pre_100ms",
                    "event_200ms",
                    "pre_200ms",
                    "removed_200ms",
                },
            )
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(set(predictions["window_ms"]), {50, 100, 200})
            condition_by_window = predictions.groupby("window_ms")[
                "condition"
            ].apply(set)
            self.assertEqual(
                condition_by_window[50],
                {"event_selected_event", "event_selected_pre", "pre_selected_pre"},
            )
            self.assertEqual(
                condition_by_window[200],
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                    "event_selected_removed",
                    "removed_selected_removed",
                },
            )
            metrics = pd.read_csv(bundle.path("metrics"))
            increment_windows = metrics[
                metrics["condition"].eq("contact_specific_increment")
            ]["window_ms"]
            self.assertEqual(set(increment_windows), {50, 100, 200})

    def test_beats_rejects_short_durations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                SnapshotSample(
                    uid=f"beats-{game}-{label}",
                    label=label,
                    lineage_group_id=f"game-{game:02d}",
                    audio_path=root / "a.wav",
                    event_start=0.45,
                    event_end=0.55,
                )
                for game in range(4)
                for label in ("fly_ball", "ground_ball")
            ]
            for sample in samples:
                waveform = np.zeros(16_000, dtype=np.float32)
                wavfile.write(sample.audio_path, 16_000, waveform)
            beats_like = FakeContactEncoder(
                EncoderProvenance(
                    name="beats_iter3plus_as2m",
                    upstream_revision="fake",
                    checkpoint_sha256="fake",
                    precision="fp32",
                    token_dimension=4,
                )
            )
            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=59, outer_splits=2, window_conditions=(50, 200)
                    ),
                    DatasetSnapshot(
                        revision="beats-snapshot", samples=tuple(samples)
                    ),
                    (beats_like,),
                    root / "artifacts",
                )
            self.assertIn("supports only 200 ms", str(context.exception))

    def test_rms_and_legacy_pooling_are_distinct_sensitivities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"pool-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="pool-snapshot", samples=tuple(samples)
            )
            encoder = FakeContactEncoder()
            primary = run_short_contact_benchmark(
                BenchmarkProtocol(seed=61, outer_splits=2),
                snapshot,
                (encoder,),
                root / "primary",
            )
            rms = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=61, outer_splits=2, normalization="rms_normalized"
                ),
                snapshot,
                (encoder,),
                root / "rms",
            )
            legacy = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=61, outer_splits=2, pooling="legacy_mean_std_max"
                ),
                snapshot,
                (encoder,),
                root / "legacy",
            )

            ids = {primary.artifact_id, rms.artifact_id, legacy.artifact_id}
            self.assertEqual(len(ids), 3)
            primary_features = pd.read_csv(primary.path("features/fake-contact"))
            legacy_features = pd.read_csv(legacy.path("features/fake-contact"))
            self.assertEqual(
                len([c for c in primary_features if c.startswith("feat_")]),
                4,
            )
            self.assertEqual(
                len([c for c in legacy_features if c.startswith("feat_")]),
                12,
            )
            self.assertEqual(
                set(legacy_features["one_token_degeneracy"]), {False}
            )
            self.assertEqual(
                set(legacy_features["embedding_pooling"]),
                {"legacy_mean_std_max"},
            )
            rms_features = pd.read_csv(rms.path("features/fake-contact"))
            self.assertEqual(
                set(rms_features["embedding_input_policy"]),
                {"resample_16khz_no_padding"},
            )
            rms_protocol = json.loads(
                rms.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(rms_protocol["normalization"], "rms_normalized")
            legacy_protocol = json.loads(
                legacy.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(legacy_protocol["pooling"], "legacy_mean_std_max")

    def test_energy_weighted_pooling_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"energy-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="energy-snapshot", samples=tuple(samples)
            )
            encoder = FakeContactEncoder()
            first = run_short_contact_benchmark(
                BenchmarkProtocol(seed=81, outer_splits=2, pooling="energy_weighted"),
                snapshot,
                (encoder,),
                root / "first",
            )
            second = run_short_contact_benchmark(
                BenchmarkProtocol(seed=81, outer_splits=2, pooling="energy_weighted"),
                snapshot,
                (encoder,),
                root / "second",
            )

            first_features = pd.read_csv(first.path("features/fake-contact"))
            self.assertEqual(
                len([c for c in first_features if c.startswith("feat_")]), 4
            )
            self.assertEqual(
                set(first_features["embedding_pooling"]), {"energy_weighted"}
            )
            second_features = pd.read_csv(second.path("features/fake-contact"))
            pd.testing.assert_frame_equal(first_features, second_features)

    def test_single_token_degeneracy_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            class SingleTokenEncoder(FakeContactEncoder):
                def encode_tokens(self, waveform, sample_rate):
                    token = super().encode_tokens(waveform, sample_rate)[0]
                    return np.stack([token])

            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"single-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="single-snapshot", samples=tuple(samples)
            )
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=89, outer_splits=2, pooling="legacy_mean_std_max"
                ),
                snapshot,
                (SingleTokenEncoder(),),
                root / "artifacts",
            )
            features = pd.read_csv(bundle.path("features/fake-contact"))
            self.assertEqual(set(features["one_token_degeneracy"]), {True})
            self.assertEqual(
                len([c for c in features if c.startswith("feat_")]), 12
            )

    def test_attention_pooling_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"attn-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="attn-snapshot", samples=tuple(samples)
            )
            encoder = FakeContactEncoder()
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(seed=83, outer_splits=2, pooling="attention"),
                snapshot,
                (encoder,),
                root / "artifacts",
            )

            tokens_path = bundle.root / "features" / "fake-contact_tokens.csv"
            self.assertTrue(tokens_path.is_file())
            tokens = pd.read_csv(tokens_path)
            self.assertTrue("token_index" in tokens.columns)
            self.assertFalse(tokens[["uid", "window_name", "token_index"]].duplicated().any())
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 8)
            self.assertFalse(predictions[["encoder", "uid"]].duplicated().any())
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertTrue((metrics["balanced_accuracy"] > 0.5).all())
            self.assertEqual(
                set(metrics["condition"]), {"event_selected_event"}
            )
            # Attention pooling must differ from plain mean pooling output.
            mean_bundle = run_short_contact_benchmark(
                BenchmarkProtocol(seed=83, outer_splits=2),
                snapshot,
                (encoder,),
                root / "mean",
            )
            mean_features = pd.read_csv(mean_bundle.path("features/fake-contact"))
            self.assertEqual(
                set(mean_features["embedding_pooling"]),
                {"valid_final_layer_token_mean"},
            )
            self.assertNotEqual(bundle.artifact_id, mean_bundle.artifact_id)

    def test_threshold_calibration_emits_both_decision_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"cal-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="cal-snapshot", samples=tuple(samples)
            )
            protocol = BenchmarkProtocol(
                seed=85,
                outer_splits=2,
                inner_splits=2,
                calibrate_threshold=True,
            )
            bundle = run_short_contact_benchmark(
                protocol,
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(
                set(predictions["decision_rule"]), {"fixed_0.5", "calibrated"}
            )
            self.assertFalse(
                predictions[
                    ["encoder", "condition", "uid", "decision_rule"]
                ].duplicated().any()
            )
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertEqual(set(metrics["decision_rule"]), {"fixed_0.5", "calibrated"})
            selections = pd.read_csv(bundle.path("selections"))
            self.assertTrue((selections["selected_threshold"].notna()).all())
            self.assertTrue(
                (selections["selected_threshold"] > 0.0).all()
                and (selections["selected_threshold"] < 1.0).all()
            )
            calibrated_ba = metrics.loc[
                metrics["decision_rule"].eq("calibrated"), "balanced_accuracy"
            ].iloc[0]
            fixed_ba = metrics.loc[
                metrics["decision_rule"].eq("fixed_0.5"), "balanced_accuracy"
            ].iloc[0]
            self.assertGreaterEqual(calibrated_ba, fixed_ba - 1e-12)

    def test_calibration_off_keeps_locked_single_rule(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"fixed-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="fixed-snapshot", samples=tuple(samples)
            )
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(seed=87, outer_splits=2),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(set(predictions["decision_rule"]), {"fixed_0.5"})
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertEqual(set(metrics["decision_rule"]), {"fixed_0.5"})
            selections = pd.read_csv(bundle.path("selections"))
            self.assertNotIn("selected_threshold", selections.columns)



    def test_threshold_selection_uses_training_fold_scores_only(self) -> None:
        # Regression guard on the calibration seam: the selection function
        # receives only training-fold indices, and the chosen threshold must
        # be a midpoint of the training scores, never influenced by held-out
        # scores that are passed nowhere near it.
        from scripts.short_contact_benchmark import _select_threshold_inner

        rng = np.random.default_rng(1234)
        n = 40
        matrix = rng.standard_normal((n, 4))
        labels = np.asarray([0, 1] * 20, dtype=int)
        groups = np.asarray([f"g{i // 2}" for i in range(n)], dtype=object)
        train = np.arange(n)
        threshold, records = _select_threshold_inner(
            matrix,
            labels,
            groups,
            train,
            c_value=0.01,
            inner_splits=2,
            seed=7,
        )
        self.assertGreater(threshold, 0.0)
        self.assertLess(threshold, 1.0)
        self.assertTrue(records)
        # Candidate thresholds are midpoints of training-fold scores plus the
        # default 0.5; the selection signature never receives held-out data.
        candidate_set = {0.5}
        candidate_set.update(float(record["threshold"]) for record in records)
        self.assertIn(threshold, candidate_set)
        self.assertTrue(np.isfinite(threshold))

    def test_composed_cross_scale_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples: list[SnapshotSample] = []
            for game_index in range(4):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"composed-{game_index:02d}-{label}"
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
            snapshot = DatasetSnapshot(
                revision="composed-snapshot", samples=tuple(samples)
            )
            protocol = BenchmarkProtocol(
                seed=93,
                outer_splits=2,
                window_conditions=(50, 200),
                feature_composition=(
                    (50, "valid_final_layer_token_mean"),
                    (200, "attention"),
                ),
                include_controls=True,
            )
            bundle = run_short_contact_benchmark(
                protocol,
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            # Component feature files exist with the expected widths.
            mean_path = (
                bundle.root
                / "features"
                / "fake-contact__50ms_valid_final_layer_token_mean.csv"
            )
            tokens_path = (
                bundle.root
                / "features"
                / "fake-contact__200ms_attention_tokens.csv"
            )
            self.assertTrue(mean_path.is_file())
            self.assertTrue(tokens_path.is_file())
            mean_features = pd.read_csv(mean_path)
            self.assertEqual(
                len([c for c in mean_features if c.startswith("feat_")]), 4
            )
            protocol_doc = json.loads(bundle.path("protocol").read_text(encoding="utf-8"))
            self.assertEqual(len(protocol_doc["feature_composition"]), 2)
            self.assertEqual(
                protocol_doc["window_conditions"],
                ["event_050ms", "event_200ms"],
            )
            self.assertEqual(
                protocol_doc["attention_control_transform_policy"],
                "event_fitted_transfer_v1",
            )

            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(
                set(predictions["condition"]),
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                },
            )
            self.assertEqual(set(predictions["window_ms"]), {0})
            roles = predictions.groupby("condition")[[
                "representation_fit_window",
                "representation_apply_window",
            ]].first()
            self.assertEqual(
                roles.loc[
                    "event_selected_pre", "representation_fit_window"
                ],
                "event_composed",
            )
            self.assertEqual(
                roles.loc[
                    "event_selected_pre", "representation_apply_window"
                ],
                "pre_composed",
            )
            self.assertFalse(
                predictions[
                    ["encoder", "condition", "uid", "decision_rule"]
                ].duplicated().any()
            )
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertIn(
                "contact_specific_increment", set(metrics["condition"])
            )
            event_ba = float(
                metrics.loc[
                    metrics["condition"].eq("event_selected_event"),
                    "balanced_accuracy",
                ].iloc[0]
            )
            self.assertGreater(event_ba, 0.95)

    def test_composed_rejects_invalid_component_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                SnapshotSample(
                    uid=f"bad-{game}-{label}",
                    label=label,
                    lineage_group_id=f"game-{game:02d}",
                    audio_path=root / "a.wav",
                    event_start=0.45,
                    event_end=0.55,
                )
                for game in range(4)
                for label in ("fly_ball", "ground_ball")
            ]
            for sample in samples:
                waveform = np.zeros(16_000, dtype=np.float32)
                wavfile.write(sample.audio_path, 16_000, waveform)
            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=95,
                        outer_splits=2,
                        feature_composition=((50, "not_a_pooling"),),
                    ),
                    DatasetSnapshot(
                        revision="bad-snapshot", samples=tuple(samples)
                    ),
                    (FakeContactEncoder(),),
                    root / "artifacts",
                )
            self.assertIn("composition pooling", str(context.exception))

    def test_composed_fails_when_component_window_has_no_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = [
                SnapshotSample(
                    uid=f"short-{game}-{label}",
                    label=label,
                    lineage_group_id=f"game-{game:02d}",
                    audio_path=root / "a.wav",
                    event_start=0.04,
                    event_end=0.05,
                )
                for game in range(4)
                for label in ("fly_ball", "ground_ball")
            ]
            for sample in samples:
                # 100 ms of audio: a 300 ms component window cannot exist.
                waveform = np.zeros(1_600, dtype=np.float32)
                wavfile.write(sample.audio_path, 16_000, waveform)
            with self.assertRaises((ValueError, AssertionError)):
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=97,
                        outer_splits=2,
                        feature_composition=((300, "valid_final_layer_token_mean"),),
                    ),
                    DatasetSnapshot(
                        revision="short-snapshot", samples=tuple(samples)
                    ),
                    (FakeContactEncoder(),),
                    root / "artifacts",
                )

    def _attention_family_snapshot(self, root: Path) -> DatasetSnapshot:
        samples: list[SnapshotSample] = []
        for game_index in range(4):
            for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                uid = f"fam-{game_index:02d}-{label}"
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
        return DatasetSnapshot(
            revision="family-snapshot", samples=tuple(samples)
        )

    def test_attention_lda_pooling(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._attention_family_snapshot(root)
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=101, outer_splits=2, pooling="attention_lda"
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "lda",
            )
            tokens_path = bundle.root / "features" / "fake-contact_tokens.csv"
            self.assertTrue(tokens_path.is_file())
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 8)
            self.assertFalse(
                predictions[["encoder", "uid"]].duplicated().any()
            )
            protocol_doc = json.loads(
                bundle.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol_doc["pooling"], "attention_lda")

    def test_attention_multi_pooling_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._attention_family_snapshot(root)
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=103,
                    outer_splits=2,
                    pooling="attention_multi",
                    attention_k=3,
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "multi",
            )
            protocol_doc = json.loads(
                bundle.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol_doc["attention_k"], 3)
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 8)
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertTrue((metrics["balanced_accuracy"] > 0.5).all())

    def test_attention_neighbourhood_and_single_token_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._attention_family_snapshot(root)
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=105,
                    outer_splits=2,
                    pooling="attention_neighbourhood",
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "neighbourhood",
            )
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(len(predictions), 8)

            class SingleTokenEncoder(FakeContactEncoder):
                def encode_tokens(self, waveform, sample_rate):
                    token = super().encode_tokens(waveform, sample_rate)[0]
                    return np.stack([token])

            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=107,
                        outer_splits=2,
                        pooling="attention_neighbourhood",
                    ),
                    snapshot,
                    (SingleTokenEncoder(),),
                    root / "single",
                )
            self.assertIn("at least two tokens", str(context.exception))

    def _layer_family_snapshot(self, root: Path) -> DatasetSnapshot:
        samples: list[SnapshotSample] = []
        for game_index in range(4):
            for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                uid = f"layers-{game_index:02d}-{label}"
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
        return DatasetSnapshot(
            revision="layers-snapshot", samples=tuple(samples)
        )

    def test_layer_wise_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._layer_family_snapshot(root)

            class MultiLayerEncoder(FakeContactEncoder):
                def encode_layer_tokens(self, waveform, sample_rate):
                    token = super().encode_tokens(waveform, sample_rate)
                    return np.stack([token * (k + 1) for k in range(4)])

            encoder = MultiLayerEncoder()
            protocol = BenchmarkProtocol(
                seed=109,
                outer_splits=2,
                pooling="attention",
                layers=(0, 2, 3),
            )
            bundle = run_short_contact_benchmark(
                protocol, snapshot, (encoder,), root / "artifacts"
            )
            for layer in (0, 2, 3):
                path = (
                    bundle.root
                    / "features"
                    / f"fake-contact__layer{layer}_tokens.csv"
                )
                self.assertTrue(path.is_file())
            self.assertFalse(
                (bundle.root / "features" / "fake-contact__layer1_tokens.csv").is_file()
            )
            protocol_doc = json.loads(
                bundle.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol_doc["layers"]["indices"], [0, 2, 3])

            # Cache reuse: the second run must not re-extract.
            calls_before = len(encoder.received_sample_counts)
            run_short_contact_benchmark(
                protocol, snapshot, (encoder,), root / "artifacts"
            )
            self.assertEqual(len(encoder.received_sample_counts), calls_before)

    def test_layer_mode_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._layer_family_snapshot(root)
            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=111,
                        outer_splits=2,
                        pooling="valid_final_layer_token_mean",
                        layers=(0, 1),
                    ),
                    snapshot,
                    (FakeContactEncoder(),),
                    root / "bad-pooling",
                )
            self.assertIn("attention-family", str(context.exception))
            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=113,
                        outer_splits=2,
                        pooling="attention",
                        layers=(-1,),
                    ),
                    snapshot,
                    (FakeContactEncoder(),),
                    root / "bad-layer",
                )
            self.assertIn("non-negative", str(context.exception))

    def test_layer_mode_requires_encoder_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._layer_family_snapshot(root)
            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=115,
                        outer_splits=2,
                        pooling="attention",
                        layers=(0, 1),
                    ),
                    snapshot,
                    (FakeContactEncoder(),),  # no encode_layer_tokens
                    root / "unsupported",
                )
            self.assertIn("does not support layer-wise", str(context.exception))

    def test_layer_scan_evaluates_every_requested_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._layer_family_snapshot(root)

            class MultiLayerEncoder(FakeContactEncoder):
                def encode_layer_tokens(self, waveform, sample_rate):
                    token = super().encode_tokens(waveform, sample_rate)
                    return np.stack([token * (k + 1) for k in range(3)])

            protocol = BenchmarkProtocol(
                seed=117,
                outer_splits=2,
                pooling="attention",
                include_controls=True,
                layers=(0, 1, 2),
            )
            bundle = run_short_contact_benchmark(
                protocol, snapshot, (MultiLayerEncoder(),), root / "artifacts"
            )
            metrics = pd.read_csv(bundle.path("metrics"))
            fixed = metrics[metrics["decision_rule"].eq("fixed_0.5")]
            layer_encoders = sorted(
                e for e in fixed["encoder"].unique() if "__layer" in e
            )
            self.assertEqual(layer_encoders, ["fake-contact__layer0",
                                              "fake-contact__layer1",
                                              "fake-contact__layer2"])
            for encoder_key in layer_encoders:
                conditions = set(
                    fixed.loc[fixed["encoder"].eq(encoder_key), "condition"]
                )
                self.assertEqual(
                    conditions,
                    {
                        "event_selected_event",
                        "event_selected_pre",
                        "pre_selected_pre",
                        "event_selected_removed",
                        "removed_selected_removed",
                        "contact_specific_increment",
                    },
                )
            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertFalse(
                predictions[["encoder", "uid", "condition", "decision_rule"]]
                .duplicated()
                .any()
            )
            protocol_doc = json.loads(
                bundle.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol_doc["layers"]["indices"], [0, 1, 2])

    def test_layer_beyond_encoder_depth_fails_visibly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._layer_family_snapshot(root)

            class TwoLayerEncoder(FakeContactEncoder):
                def encode_layer_tokens(self, waveform, sample_rate):
                    token = super().encode_tokens(waveform, sample_rate)
                    return np.stack([token, token * 2])

            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=119,
                        outer_splits=2,
                        pooling="attention",
                        layers=(0, 2),
                    ),
                    snapshot,
                    (TwoLayerEncoder(),),
                    root / "artifacts",
                )
            self.assertIn("layer 2", str(context.exception))

    def test_non_centered_windows_evaluate_and_coexist_with_centered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            snapshot = DatasetSnapshot(
                revision="synthetic-snapshot-1", samples=tuple(samples)
            )

            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=20260805,
                    outer_splits=3,
                    logistic_c=0.1,
                    non_centered_windows=(
                        "full_audio",
                        "post_contact_200ms",
                        "pre_contact_200ms",
                    ),
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            windows = pd.read_csv(bundle.path("window_manifest"))
            self.assertEqual(
                set(windows["window_name"]),
                {
                    "event_200ms",
                    "full_audio",
                    "post_contact_200ms",
                    "pre_contact_200ms",
                },
            )
            self.assertEqual(len(windows), 12 * 4)
            full = windows[windows["window_name"].eq("full_audio")]
            self.assertTrue((full["window_start"] == 0.0).all())
            self.assertTrue((full["window_duration"] == 1.0).all())
            self.assertTrue((full["window_kind"] == "full_audio").all())
            post = windows[windows["window_name"].eq("post_contact_200ms")]
            self.assertTrue((post["window_start"] == 0.5).all())
            self.assertTrue((post["window_end"] == 0.7).all())
            pre = windows[windows["window_name"].eq("pre_contact_200ms")]
            self.assertTrue((pre["window_end"] == 0.4).all())
            self.assertTrue((pre["window_start"] == 0.2).all())
            self.assertTrue((pre["window_kind"] == "pre_contact").all())

            metrics = pd.read_csv(bundle.path("metrics"))
            conditions = set(metrics["condition"])
            self.assertIn("event_selected_event", conditions)
            for condition in (
                "full_audio",
                "post_contact_200ms",
                "pre_contact_200ms",
            ):
                self.assertIn(condition, conditions)
                rows = metrics[metrics["condition"].eq(condition)]
                self.assertEqual(len(rows), 1)
                self.assertEqual(
                    int(rows["eligible_samples"].iloc[0]), 12
                )

            predictions = pd.read_csv(bundle.path("oof_predictions"))
            self.assertEqual(
                set(predictions["condition"]),
                {
                    "event_selected_event",
                    "full_audio",
                    "post_contact_200ms",
                    "pre_contact_200ms",
                },
            )
            duplicate = predictions.duplicated(
                ["encoder", "condition", "decision_rule", "uid"]
            )
            self.assertFalse(duplicate.any())

            protocol = json.loads(bundle.path("protocol").read_text(encoding="utf-8"))
            self.assertEqual(
                protocol["non_centered_windows"],
                {
                    "full_audio": {"definition": "full clip from audio start"},
                    "post_contact_200ms": {
                        "definition": "fixed segment starting at estimated peak",
                        "duration_ms": 200,
                    },
                    "pre_contact_200ms": {
                        "definition": (
                            "fixed segment ending 50ms before annotated event start"
                        ),
                        "duration_ms": 200,
                    },
                },
            )
            self.assertTrue(protocol["model_input_policy"]["full_clips"])

    def test_non_centered_windows_skip_out_of_bounds_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 16_000
            samples: list[SnapshotSample] = []
            for game_index in range(6):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"game-{game_index:02d}-{label}"
                    # fly clips from games 3-5 are too short to fit 0.5 s
                    # after the peak; the others fit.
                    duration = (
                        0.5 if label == "fly_ball" and game_index >= 3 else 1.0
                    )
                    waveform = np.zeros(
                        int(sample_rate * duration), dtype=np.float32
                    )
                    waveform[int(sample_rate * 0.10)] = polarity
                    audio_path = root / "snapshot" / f"{uid}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    wavfile.write(audio_path, sample_rate, waveform)
                    samples.append(
                        SnapshotSample(
                            uid=uid,
                            label=label,
                            lineage_group_id=f"game-{game_index:02d}",
                            audio_path=audio_path,
                            event_start=0.05,
                            event_end=0.15,
                        )
                    )
            snapshot = DatasetSnapshot(
                revision="synthetic-snapshot-1", samples=tuple(samples)
            )

            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=20260805,
                    outer_splits=3,
                    logistic_c=0.1,
                    non_centered_windows=("post_contact_500ms",),
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            windows = pd.read_csv(bundle.path("window_manifest"))
            post = windows[windows["window_name"].eq("post_contact_500ms")]
            # fly clips from games 3-5 (0.5 s) cannot fit 0.5 s after the
            # peak; every other clip can.
            self.assertEqual(len(post), 9)
            self.assertEqual(
                set(post["label"]), {"fly_ball", "ground_ball"}
            )
            self.assertEqual(len(post[post["label"].eq("fly_ball")]), 3)

            metrics = pd.read_csv(bundle.path("metrics"))
            rows = metrics[metrics["condition"].eq("post_contact_500ms")]
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows["eligible_samples"].iloc[0]), 9)

    def test_non_centered_window_spec_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            snapshot = DatasetSnapshot(
                revision="synthetic-snapshot-1", samples=tuple(samples)
            )

            def run_with(specs, **changes):
                return run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=20260805,
                        outer_splits=3,
                        logistic_c=0.1,
                        non_centered_windows=specs,
                        **changes,
                    ),
                    snapshot,
                    (FakeContactEncoder(),),
                    root / "artifacts-invalid",
                )

            for specs in (
                ("post_contact_0ms",),
                ("pre_contact_abc",),
                ("garbage",),
                ("full_audio", "full_audio"),
            ):
                with self.assertRaises(ValueError):
                    run_with(specs)

            with self.assertRaises(ValueError):
                run_with(
                    ("full_audio",),
                    feature_composition=((50, "valid_final_layer_token_mean"),),
                )
            with self.assertRaises(ValueError):
                run_with(("full_audio",), pooling="attention", layers=(0,))

            centered = run_with(())
            centered_doc = json.loads(
                centered.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(centered_doc["non_centered_windows"], {})
            full_audio_run = run_with(("full_audio",))
            self.assertNotEqual(
                centered.artifact_id,
                full_audio_run.artifact_id,
            )
            protocol_doc = json.loads(
                full_audio_run.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                list(protocol_doc["non_centered_windows"]), ["full_audio"]
            )

    def test_non_centered_with_controls_keeps_chain_and_adds_conditions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
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
            snapshot = DatasetSnapshot(
                revision="synthetic-snapshot-1", samples=tuple(samples)
            )

            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=20260805,
                    outer_splits=3,
                    logistic_c=0.1,
                    include_controls=True,
                    non_centered_windows=("full_audio",),
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )

            metrics = pd.read_csv(bundle.path("metrics"))
            conditions = set(metrics["condition"])
            for condition in (
                "event_selected_event",
                "event_selected_pre",
                "pre_selected_pre",
                "event_selected_removed",
                "removed_selected_removed",
                "contact_specific_increment",
            ):
                self.assertIn(condition, conditions)
            self.assertIn("full_audio", conditions)

    def _shift_snapshot(self, root: Path, audio_seconds: float = 1.0):
        sample_rate = 16_000
        samples: list[SnapshotSample] = []
        for game_index in range(6):
            for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                uid = f"game-{game_index:02d}-{label}"
                waveform = np.zeros(
                    int(sample_rate * audio_seconds), dtype=np.float32
                )
                waveform[int(sample_rate * 0.5)] = polarity
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

    def test_event_window_shift_slices_at_shifted_centre(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._shift_snapshot(root)
            for shift_ms, expected_start, expected_end in (
                (100, 0.5, 0.7),
                (-100, 0.3, 0.5),
                (25, 0.425, 0.625),
            ):
                bundle = run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=20260805,
                        outer_splits=3,
                        logistic_c=0.1,
                        window_shift_ms=shift_ms,
                    ),
                    snapshot,
                    (FakeContactEncoder(),),
                    root / f"artifacts-{shift_ms}",
                )
                windows = pd.read_csv(bundle.path("window_manifest"))
                event = windows[windows["window_name"].eq("event_200ms")]
                self.assertEqual(len(event), 12)
                self.assertTrue(
                    (event["window_start"] == expected_start).all(),
                    f"shift {shift_ms}",
                )
                self.assertTrue(
                    (event["window_end"] == expected_end).all(),
                    f"shift {shift_ms}",
                )
                metrics = pd.read_csv(bundle.path("metrics"))
                self.assertEqual(
                    int(
                        metrics[
                            metrics["condition"].eq("event_selected_event")
                        ]["eligible_samples"].iloc[0]
                    ),
                    12,
                )

    def test_event_window_shift_fingerprint_is_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._shift_snapshot(root)
            centred = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=20260805, outer_splits=3, logistic_c=0.1
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts-centred",
            )
            shifted = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=20260805,
                    outer_splits=3,
                    logistic_c=0.1,
                    window_shift_ms=50,
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts-shifted",
            )
            self.assertNotEqual(
                centred.artifact_id, shifted.artifact_id
            )
            protocol = json.loads(
                shifted.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol["event_window_shift_ms"], 50)

    def test_event_window_shift_skips_out_of_bounds_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_rate = 16_000
            samples: list[SnapshotSample] = []
            for game_index in range(6):
                for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
                    uid = f"game-{game_index:02d}-{label}"
                    # fly clips from games 3-5 are too short to fit a
                    # window centred 0.2 s after the peak.
                    seconds = (
                        0.5 if label == "fly_ball" and game_index >= 3 else 1.0
                    )
                    waveform = np.zeros(
                        int(sample_rate * seconds), dtype=np.float32
                    )
                    waveform[int(sample_rate * 0.25)] = polarity
                    audio_path = root / "snapshot" / f"{uid}.wav"
                    audio_path.parent.mkdir(parents=True, exist_ok=True)
                    wavfile.write(audio_path, sample_rate, waveform)
                    samples.append(
                        SnapshotSample(
                            uid=uid,
                            label=label,
                            lineage_group_id=f"game-{game_index:02d}",
                            audio_path=audio_path,
                            event_start=0.2,
                            event_end=0.3,
                        )
                    )
            snapshot = DatasetSnapshot(
                revision="synthetic-snapshot-1", samples=tuple(samples)
            )
            bundle = run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=20260805,
                    outer_splits=3,
                    logistic_c=0.1,
                    window_shift_ms=200,
                ),
                snapshot,
                (FakeContactEncoder(),),
                root / "artifacts",
            )
            windows = pd.read_csv(bundle.path("window_manifest"))
            event = windows[windows["window_name"].eq("event_200ms")]
            # fly clips from games 3-5 (0.5 s) cannot fit a window centred
            # at 0.45 s; every other clip can.
            self.assertEqual(len(event), 9)
            self.assertEqual(
                set(event["label"]), {"fly_ball", "ground_ball"}
            )
            self.assertEqual(len(event[event["label"].eq("fly_ball")]), 3)
            metrics = pd.read_csv(bundle.path("metrics"))
            self.assertEqual(
                int(
                    metrics[
                        metrics["condition"].eq("event_selected_event")
                    ]["eligible_samples"].iloc[0]
                ),
                9,
            )

    def test_event_window_shift_rejects_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = self._shift_snapshot(root)
            with self.assertRaises(ValueError) as context:
                run_short_contact_benchmark(
                    BenchmarkProtocol(
                        seed=20260805,
                        outer_splits=3,
                        logistic_c=0.1,
                        include_controls=True,
                        window_shift_ms=50,
                    ),
                    snapshot,
                    (FakeContactEncoder(),),
                    root / "artifacts",
                )
            self.assertIn("include_controls", str(context.exception))


if __name__ == "__main__":
    unittest.main()
