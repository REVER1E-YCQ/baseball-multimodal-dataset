from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from locked_attention_fixture import build_locked_attention_source
from scripts.paired_contrast_evaluation import (
    PairedContrastEvaluationConfig,
    run_paired_contrast_evaluation,
)


ARMS = {
    "event_alone",
    "event_minus_pre",
    "event_plus_delta",
}
FOLD_SEEDS = {20260805, 20260806, 20260807}
CONDITIONS = {
    "event",
    "strict_pre",
    "transient_removed",
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


def rewrite_tokens_for_scenario(source_root: Path, scenario: str) -> None:
    manifest_path = source_root / "artifact_bundle.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_name, feature_record = next(
        (name, record)
        for name, record in manifest["artifacts"].items()
        if name.startswith("features/")
    )
    feature_path = source_root / feature_record["path"]
    tokens = pd.read_csv(feature_path)
    labels = pd.read_csv(source_root / "fold_assignments.csv").set_index("uid")[
        "label"
    ]
    feature_columns = [
        column for column in tokens if column.startswith("feat_")
    ]
    tokens[feature_columns] = 0.0
    for index, row in tokens.iterrows():
        y_value = 1.0 if labels.loc[row["uid"]] == "ground_ball" else -1.0
        if scenario == "event_signal_with_cancelling_background":
            value = 0.0 if row["window_name"] == "event_200ms" else -y_value
        elif scenario == "background_only":
            value = y_value
        else:
            raise ValueError(f"Unknown synthetic scenario: {scenario}")
        tokens.at[index, feature_columns[0]] = value
    tokens.to_csv(feature_path, index=False)
    update_manifest_checksum(source_root, feature_name)


def flip_outer_test_labels(source_root: Path, outer_fold: int) -> None:
    fold_path = source_root / "fold_assignments.csv"
    folds = pd.read_csv(fold_path)
    test = folds["outer_fold"] == outer_fold
    folds.loc[test, "label"] = folds.loc[test, "label"].map(
        {"fly_ball": "ground_ball", "ground_ball": "fly_ball"}
    )
    folds.to_csv(fold_path, index=False)
    update_manifest_checksum(source_root, "fold_assignments")


class PairedContrastEvaluationTest(unittest.TestCase):
    def test_predeclared_family_uses_exact_pairs_and_reports_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)

            result = run_paired_contrast_evaluation(
                source.root,
                root / "contrast",
                PairedContrastEvaluationConfig(n_bootstrap=20),
            )

            dimensions = pd.read_csv(result.path("representation_dimensions"))
            self.assertEqual(set(dimensions["arm"]), ARMS)
            self.assertEqual(
                dict(zip(dimensions["arm"], dimensions["dimension"], strict=True)),
                {
                    "event_alone": 4,
                    "event_minus_pre": 4,
                    "event_plus_delta": 8,
                },
            )
            requires_pre = dict(
                zip(
                    dimensions["arm"],
                    dimensions["strict_pre_required_at_inference"],
                    strict=True,
                )
            )
            self.assertFalse(requires_pre["event_alone"])
            self.assertTrue(requires_pre["event_minus_pre"])
            self.assertTrue(requires_pre["event_plus_delta"])

            metrics = pd.read_csv(result.path("metrics"))
            self.assertEqual(set(metrics["arm"]), ARMS)
            self.assertEqual(set(metrics["fold_seed"]), FOLD_SEEDS)
            self.assertEqual(set(metrics["condition"]), CONDITIONS)
            self.assertTrue((metrics["eligible_samples"] == 20).all())

            folds = pd.read_csv(result.path("fold_assignments"))
            self.assertEqual(set(folds["fold_seed"]), FOLD_SEEDS)
            self.assertEqual(len(folds), 60)
            self.assertTrue(
                (
                    folds.groupby(["fold_seed", "lineage_group_id"])[
                        "outer_fold"
                    ].nunique()
                    == 1
                ).all()
            )

            pairing = json.loads(
                result.path("pairing_audit").read_text(encoding="utf-8")
            )
            self.assertEqual(pairing["n_exact_pairs"], 20)
            self.assertEqual(pairing["n_lineage_groups"], 10)
            self.assertEqual(pairing["n_singleton_lineage_groups"], 0)
            self.assertTrue(pairing["identical_membership_across_arms_and_seeds"])
            self.assertTrue(pairing["exact_windows_only"])
            self.assertEqual(pairing["waveform_padding_samples"], 0)

            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                protocol["representation_family"]["arms"],
                ["event_alone", "event_minus_pre", "event_plus_delta"],
            )
            self.assertEqual(
                protocol["representation_family"]["attention_fit_scope"],
                "outer_training_event_tokens_only",
            )
            self.assertEqual(
                protocol["representation_family"]["shared_transform_windows"],
                ["event_200ms", "pre_200ms", "removed_200ms"],
            )

    def test_contrast_recovers_event_signal_without_fabricating_background_gain(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            signal_source = build_locked_attention_source(root / "signal")
            rewrite_tokens_for_scenario(
                signal_source.root,
                "event_signal_with_cancelling_background",
            )
            signal = run_paired_contrast_evaluation(
                signal_source.root,
                root / "signal-output",
                PairedContrastEvaluationConfig(n_bootstrap=20),
            )
            signal_metrics = pd.read_csv(signal.path("metrics"))
            signal_event = signal_metrics[
                signal_metrics["condition"] == "event"
            ].groupby("arm")["balanced_accuracy"].mean()
            self.assertEqual(signal_event["event_alone"], 0.5)
            self.assertEqual(signal_event["event_minus_pre"], 1.0)
            self.assertEqual(signal_event["event_plus_delta"], 1.0)
            signal_differences = pd.read_csv(signal.path("paired_differences"))
            signal_event_gain = signal_differences[
                (signal_differences["comparison"] == "event_ba_gain")
                & (signal_differences["fold_seed"] == "mean_across_seeds")
            ].set_index("arm")
            self.assertEqual(
                signal_event_gain.loc["event_minus_pre", "observed_difference"],
                0.5,
            )
            self.assertGreater(
                signal_event_gain.loc["event_minus_pre", "ci_low"], 0
            )
            signal_verdict = json.loads(
                signal.path("verdict").read_text(encoding="utf-8")
            )
            self.assertEqual(signal_verdict["decision"], "continue")
            self.assertEqual(
                set(signal_verdict["qualifying_arms"]),
                {"event_minus_pre", "event_plus_delta"},
            )
            self.assertFalse(signal_verdict["preferred_representation_selected"])
            self.assertFalse(signal_verdict["headline_replacement_allowed"])
            self.assertTrue(
                all(
                    row["no_control_rise_in_any_seed"]
                    for row in signal_verdict["evaluated_arms"]
                )
            )

            background_source = build_locked_attention_source(root / "background")
            rewrite_tokens_for_scenario(background_source.root, "background_only")
            background = run_paired_contrast_evaluation(
                background_source.root,
                root / "background-output",
                PairedContrastEvaluationConfig(n_bootstrap=20),
            )
            background_metrics = pd.read_csv(background.path("metrics"))
            increments = background_metrics[
                background_metrics["condition"]
                == "contact_specific_increment"
            ]
            self.assertEqual(set(increments["arm"]), ARMS)
            self.assertTrue(
                (increments["balanced_accuracy"].abs() < 1e-12).all()
            )
            background_verdict = json.loads(
                background.path("verdict").read_text(encoding="utf-8")
            )
            self.assertEqual(background_verdict["decision"], "stop")
            self.assertEqual(background_verdict["qualifying_arms"], [])

    def test_rerun_is_deterministic_and_screening_policy_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            config = PairedContrastEvaluationConfig(
                n_bootstrap=20,
                seed=41,
            )
            first = run_paired_contrast_evaluation(
                source.root, root / "first", config
            )
            second = run_paired_contrast_evaluation(
                source.root, root / "second", config
            )
            changed = run_paired_contrast_evaluation(
                source.root,
                root / "changed",
                PairedContrastEvaluationConfig(
                    n_bootstrap=20,
                    seed=42,
                ),
            )

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertNotEqual(first.artifact_id, changed.artifact_id)
            self.assertEqual(first.artifact_names, second.artifact_names)
            for name in first.artifact_names:
                self.assertEqual(
                    first.path(name).read_bytes(),
                    second.path(name).read_bytes(),
                    msg=name,
                )
            report = first.path("report_zh").read_text(encoding="utf-8")
            self.assertIn("三个 fold seeds", report)
            self.assertIn("strict-Pre", report)
            self.assertIn("推理时", report)
            self.assertIn("不选择 preferred representation", report)
            self.assertIn("exact pairs", report)
            self.assertIn("结构性零差", report)
            provenance = json.loads(
                first.path("provenance").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["encoder_inference_runs"], 0)

            with self.assertRaisesRegex(ValueError, "fold_seeds are locked"):
                run_paired_contrast_evaluation(
                    source.root,
                    root / "two-seeds",
                    PairedContrastEvaluationConfig(
                        fold_seeds=(20260805, 20260806),
                        n_bootstrap=20,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "locked at 0.02"):
                run_paired_contrast_evaluation(
                    source.root,
                    root / "weakened-gate",
                    PairedContrastEvaluationConfig(
                        n_bootstrap=20,
                        minimum_headline_ba_gain=0.019,
                    ),
                )

    def test_outer_test_labels_cannot_change_training_fold_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_source = build_locked_attention_source(root / "original")
            changed_source = build_locked_attention_source(root / "changed")
            flip_outer_test_labels(changed_source.root, outer_fold=0)

            original = run_paired_contrast_evaluation(
                original_source.root,
                root / "original-output",
                PairedContrastEvaluationConfig(n_bootstrap=20),
            )
            changed = run_paired_contrast_evaluation(
                changed_source.root,
                root / "changed-output",
                PairedContrastEvaluationConfig(n_bootstrap=20),
            )
            columns = [
                "arm",
                "fold_seed",
                "outer_fold",
                "selected_parameters_json",
                "candidate_scores_json",
                "selection_scope",
            ]
            original_selections = pd.read_csv(original.path("selections"))
            changed_selections = pd.read_csv(changed.path("selections"))
            original_locked_fold = original_selections[
                (original_selections["fold_seed"] == 20260805)
                & (original_selections["outer_fold"] == 0)
            ][columns].reset_index(drop=True)
            changed_locked_fold = changed_selections[
                (changed_selections["fold_seed"] == 20260805)
                & (changed_selections["outer_fold"] == 0)
            ][columns].reset_index(drop=True)
            pd.testing.assert_frame_equal(
                original_locked_fold,
                changed_locked_fold,
            )

            fit_audit = json.loads(
                original.path("fit_audit").read_text(encoding="utf-8")
            )
            self.assertEqual(fit_audit["event_attention_fits"], 15)
            self.assertEqual(fit_audit["outer_probe_fits"], 45)
            self.assertEqual(fit_audit["model_selection_fits"], 405)


if __name__ == "__main__":
    unittest.main()
