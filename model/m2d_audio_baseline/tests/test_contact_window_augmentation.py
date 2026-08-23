from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from locked_attention_fixture import SyntheticM2D, build_locked_attention_source
from scripts.contact_window_augmentation import (
    ContactWindowAugmentationConfig,
    run_contact_window_augmentation_evaluation,
)
from scripts.short_contact_benchmark import (
    DatasetSnapshot,
    SnapshotSample,
    _audit_snapshot,
    _canonical_sha256,
)


ARMS = {
    "no_augmentation",
    "time_jitter",
    "gain",
    "light_eq",
    "combined",
}
FOLD_SEEDS = {20260805, 20260806, 20260807}
CONDITIONS = {
    "event",
    "strict_pre",
    "transient_removed",
    "imposed_shift_minus_20ms",
    "imposed_shift_plus_20ms",
    "contact_specific_increment",
}


def update_manifest_checksum(source_root: Path, artifact_name: str) -> None:
    manifest_path = source_root / "artifact_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["artifacts"][artifact_name]
    artifact_path = source_root / record["path"]
    record["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def snapshot_for_fixture(root: Path, source_root: Path) -> DatasetSnapshot:
    folds = pd.read_csv(source_root / "fold_assignments.csv")
    return DatasetSnapshot(
        revision="4b6ed0e1cea1425121b075212ddb49b820e27cda",
        samples=tuple(
            SnapshotSample(
                uid=str(row.uid),
                label=str(row.label),
                lineage_group_id=str(row.lineage_group_id),
                audio_path=root / "snapshot" / f"{row.uid}.wav",
                event_start=0.45,
                event_end=0.55,
            )
            for row in folds.itertuples(index=False)
        ),
    )


def refresh_snapshot_identity(source_root: Path) -> None:
    fixture_root = source_root.parents[1]
    snapshot_audit, snapshot_fingerprint = _audit_snapshot(
        snapshot_for_fixture(fixture_root, source_root)
    )
    snapshot_audit_path = source_root / "snapshot_audit.json"
    snapshot_audit_path.write_text(
        json.dumps(snapshot_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_manifest_checksum(source_root, "snapshot_audit")

    protocol_path = source_root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["dataset"]["snapshot_fingerprint"] = snapshot_fingerprint
    protocol["artifact_id"] = _canonical_sha256(
        {key: value for key, value in protocol.items() if key != "artifact_id"}
    )[:24]
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    update_manifest_checksum(source_root, "protocol")
    manifest_path = source_root / "artifact_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_id"] = protocol["artifact_id"]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def flip_all_labels(source_root: Path) -> None:
    fold_path = source_root / "fold_assignments.csv"
    folds = pd.read_csv(fold_path)
    folds["label"] = folds["label"].map(
        {"fly_ball": "ground_ball", "ground_ball": "fly_ball"}
    )
    folds.to_csv(fold_path, index=False)
    update_manifest_checksum(source_root, "fold_assignments")
    refresh_snapshot_identity(source_root)


def flip_outer_test_labels(source_root: Path, outer_fold: int) -> None:
    fold_path = source_root / "fold_assignments.csv"
    folds = pd.read_csv(fold_path)
    test = folds["outer_fold"] == outer_fold
    folds.loc[test, "label"] = folds.loc[test, "label"].map(
        {"fly_ball": "ground_ball", "ground_ball": "fly_ball"}
    )
    folds.to_csv(fold_path, index=False)
    update_manifest_checksum(source_root, "fold_assignments")
    refresh_snapshot_identity(source_root)


class CountingSyntheticM2D(SyntheticM2D):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def encode_tokens(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        self.calls += 1
        return super().encode_tokens(waveform, sample_rate)


class ContactWindowAugmentationTest(unittest.TestCase):
    def test_locked_family_augments_outer_training_only_and_reports_robustness(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            snapshot = snapshot_for_fixture(root, source.root)

            result = run_contact_window_augmentation_evaluation(
                source.root,
                root / "augmentation",
                snapshot,
                SyntheticM2D(),
                ContactWindowAugmentationConfig(n_bootstrap=20),
            )

            metrics = pd.read_csv(result.path("metrics"))
            self.assertEqual(set(metrics["arm"]), ARMS)
            self.assertEqual(set(metrics["fold_seed"]), FOLD_SEEDS)
            self.assertEqual(set(metrics["condition"]), CONDITIONS)
            self.assertTrue((metrics["eligible_samples"] == 20).all())

            source_predictions = pd.read_csv(source.path("oof_predictions"))
            source_predictions = source_predictions[
                source_predictions["condition"].isin(
                    {
                        "event_selected_event",
                        "event_selected_pre",
                        "event_selected_removed",
                    }
                )
            ].copy()
            source_predictions["condition"] = source_predictions["condition"].map(
                {
                    "event_selected_event": "event",
                    "event_selected_pre": "strict_pre",
                    "event_selected_removed": "transient_removed",
                }
            )
            baseline_predictions = pd.read_csv(result.path("oof_predictions"))
            baseline_predictions = baseline_predictions[
                (baseline_predictions["arm"] == "no_augmentation")
                & (baseline_predictions["fold_seed"] == 20260805)
                & baseline_predictions["condition"].isin(
                    {"event", "strict_pre", "transient_removed"}
                )
            ]
            reproduced = source_predictions.merge(
                baseline_predictions,
                on=["uid", "condition"],
                suffixes=("_source", "_augmentation"),
                validate="one_to_one",
            )
            self.assertEqual(len(reproduced), 60)
            self.assertEqual(
                reproduced["score_ground_ball_source"].tolist(),
                reproduced["score_ground_ball_augmentation"].tolist(),
            )
            self.assertEqual(
                reproduced["y_pred_source"].tolist(),
                reproduced["y_pred_augmentation"].tolist(),
            )

            folds = pd.read_csv(result.path("fold_assignments"))
            assignments = pd.read_csv(result.path("augmentation_assignments"))
            self.assertEqual(set(assignments["recipe"]), ARMS - {"no_augmentation"})
            self.assertTrue((assignments["split_role"] == "outer_train").all())
            joined = assignments.merge(
                folds,
                on=["fold_seed", "uid", "lineage_group_id"],
                suffixes=("_assignment", "_source"),
                validate="many_to_one",
            )
            self.assertTrue(
                (
                    joined["outer_fold_assignment"]
                    != joined["outer_fold_source"]
                ).all()
            )
            self.assertTrue(
                (
                    folds.groupby(["fold_seed", "lineage_group_id"])[
                        "outer_fold"
                    ].nunique()
                    == 1
                ).all()
            )

            waveform_audit = pd.read_csv(result.path("waveform_audit"))
            self.assertTrue(
                (
                    waveform_audit["output_duration_samples"]
                    == waveform_audit["expected_duration_samples"]
                ).all()
            )
            self.assertTrue((waveform_audit["waveform_padding_samples"] == 0).all())
            self.assertFalse(waveform_audit["project_label_visible"].any())

            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                protocol["augmentation_family"]["arms"],
                [
                    "no_augmentation",
                    "time_jitter",
                    "gain",
                    "light_eq",
                    "combined",
                ],
            )
            self.assertEqual(
                protocol["augmentation_family"]["aggregation_policy"],
                "append_original_and_one_derivative_equal_source_weight",
            )
            self.assertEqual(
                protocol["augmentation_family"]["outer_test_policy"],
                "unaugmented_except_predeclared_imposed_shift_diagnostics",
            )
            self.assertEqual(
                result.artifact_id,
                _canonical_sha256(
                    {
                        key: value
                        for key, value in protocol.items()
                        if key != "artifact_id"
                    }
                )[:24],
            )
            manifest = json.loads(
                result.path("artifact_bundle").read_text(encoding="utf-8")
            )
            for record in manifest["artifacts"].values():
                artifact_path = result.root / record["path"]
                self.assertEqual(
                    hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                    record["sha256"],
                )
            report = result.path("report_zh").read_text(encoding="utf-8")
            self.assertIn("真实 outer-test 音频不做训练增强", report)
            self.assertIn("imposed-shift robustness", report)
            self.assertIn("不选择 preferred recipe", report)

    def test_replay_is_deterministic_and_waveform_randomization_is_label_blind(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_source = build_locked_attention_source(root / "original")
            changed_source = build_locked_attention_source(root / "changed")
            flip_all_labels(changed_source.root)
            config = ContactWindowAugmentationConfig(n_bootstrap=20, seed=41)

            first = run_contact_window_augmentation_evaluation(
                original_source.root,
                root / "first",
                snapshot_for_fixture(root / "original", original_source.root),
                SyntheticM2D(),
                config,
            )
            second = run_contact_window_augmentation_evaluation(
                original_source.root,
                root / "second",
                snapshot_for_fixture(root / "original", original_source.root),
                SyntheticM2D(),
                config,
            )
            relabelled = run_contact_window_augmentation_evaluation(
                changed_source.root,
                root / "relabelled",
                snapshot_for_fixture(root / "changed", changed_source.root),
                SyntheticM2D(),
                config,
            )

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertEqual(first.artifact_names, second.artifact_names)
            for name in first.artifact_names:
                self.assertEqual(
                    first.path(name).read_bytes(),
                    second.path(name).read_bytes(),
                    msg=name,
                )
            self.assertEqual(
                first.path("augmented_tokens").read_bytes(),
                relabelled.path("augmented_tokens").read_bytes(),
            )
            pd.testing.assert_frame_equal(
                pd.read_csv(first.path("waveform_audit")),
                pd.read_csv(relabelled.path("waveform_audit")),
            )

            with self.assertRaisesRegex(ValueError, "augmentation_seed is locked"):
                run_contact_window_augmentation_evaluation(
                    original_source.root,
                    root / "changed-seed",
                    snapshot_for_fixture(root / "original", original_source.root),
                    SyntheticM2D(),
                    ContactWindowAugmentationConfig(
                        augmentation_seed=20260806,
                        n_bootstrap=20,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "locked at 0.02"):
                run_contact_window_augmentation_evaluation(
                    original_source.root,
                    root / "weakened-gate",
                    snapshot_for_fixture(root / "original", original_source.root),
                    SyntheticM2D(),
                    ContactWindowAugmentationConfig(
                        n_bootstrap=20,
                        minimum_headline_ba_gain=0.019,
                    ),
                )

    def test_snapshot_audio_must_match_the_locked_source_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            snapshot = snapshot_for_fixture(root, source.root)
            audio_path = snapshot.samples[0].audio_path
            audio_path.write_bytes(audio_path.read_bytes() + b"changed")

            with self.assertRaisesRegex(
                RuntimeError, "Snapshot fingerprint does not match"
            ):
                run_contact_window_augmentation_evaluation(
                    source.root,
                    root / "output",
                    snapshot,
                    SyntheticM2D(),
                    ContactWindowAugmentationConfig(n_bootstrap=20),
                )

    def test_matching_artifact_identity_reuses_the_deterministic_token_cache(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            snapshot = snapshot_for_fixture(root, source.root)
            encoder = CountingSyntheticM2D()
            config = ContactWindowAugmentationConfig(n_bootstrap=20)

            first = run_contact_window_augmentation_evaluation(
                source.root,
                root / "output",
                snapshot,
                encoder,
                config,
            )
            self.assertEqual(encoder.calls, 280)
            second = run_contact_window_augmentation_evaluation(
                source.root,
                root / "output",
                snapshot,
                encoder,
                config,
            )

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertEqual(encoder.calls, 280)
            self.assertEqual(
                first.path("augmented_tokens").read_bytes(),
                second.path("augmented_tokens").read_bytes(),
            )

    def test_outer_test_labels_cannot_change_training_or_recipe_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_source = build_locked_attention_source(root / "original")
            changed_source = build_locked_attention_source(root / "changed")
            flip_outer_test_labels(changed_source.root, outer_fold=0)
            config = ContactWindowAugmentationConfig(n_bootstrap=20)

            original = run_contact_window_augmentation_evaluation(
                original_source.root,
                root / "original-output",
                snapshot_for_fixture(root / "original", original_source.root),
                SyntheticM2D(),
                config,
            )
            changed = run_contact_window_augmentation_evaluation(
                changed_source.root,
                root / "changed-output",
                snapshot_for_fixture(root / "changed", changed_source.root),
                SyntheticM2D(),
                config,
            )
            columns = [
                "arm",
                "fold_seed",
                "outer_fold",
                "selected_parameters_json",
                "candidate_scores_json",
                "selection_scope",
            ]
            original_selection = pd.read_csv(original.path("selections"))
            changed_selection = pd.read_csv(changed.path("selections"))
            original_locked_fold = original_selection[
                (original_selection["fold_seed"] == 20260805)
                & (original_selection["outer_fold"] == 0)
            ][columns].reset_index(drop=True)
            changed_locked_fold = changed_selection[
                (changed_selection["fold_seed"] == 20260805)
                & (changed_selection["outer_fold"] == 0)
            ][columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                original_locked_fold,
                changed_locked_fold,
            )

            fit_audit = json.loads(
                original.path("fit_audit").read_text(encoding="utf-8")
            )
            self.assertEqual(fit_audit["attention_fits"], 75)
            self.assertEqual(fit_audit["outer_probe_fits"], 75)
            self.assertEqual(fit_audit["model_selection_fits"], 675)
            provenance = json.loads(
                original.path("provenance").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["encoder_inference_runs"], 280)


if __name__ == "__main__":
    unittest.main()
