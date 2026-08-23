from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from locked_attention_fixture import build_locked_attention_source
from scripts.margin_classifier_evaluation import (
    MarginClassifierEvaluationConfig,
    run_margin_classifier_evaluation,
)


class MarginClassifierEvaluationTest(unittest.TestCase):
    def test_locked_family_uses_matched_inputs_and_reports_both_rules(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_margin_classifier_evaluation(
                source.root,
                root / "margin-family",
                MarginClassifierEvaluationConfig(
                    n_bootstrap=40,
                    seed=20260805,
                ),
            )

            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            probes = {
                item["name"]: item for item in protocol["candidate_family"]
            }
            self.assertEqual(
                set(probes),
                {
                    "attention-logistic",
                    "attention-linear-svm",
                    "attention-rbf-svm",
                },
            )
            self.assertEqual(
                probes["attention-logistic"]["hyperparameter_grid"],
                {"C": [0.001, 0.01, 0.1]},
            )
            self.assertEqual(
                probes["attention-linear-svm"]["hyperparameter_grid"],
                {"C": [0.001, 0.01, 0.1, 1.0, 10.0]},
            )
            self.assertEqual(
                probes["attention-rbf-svm"]["hyperparameter_grid"],
                {"C": [0.3, 1.0, 3.0], "gamma": ["scale", 0.001]},
            )
            self.assertTrue(
                all(item["calibrate_threshold"] for item in probes.values())
            )

            predictions = pd.read_csv(result.path("oof_predictions"))
            expected_rules = {
                "attention-logistic": {"fixed_0.5", "calibrated"},
                "attention-linear-svm": {"fixed_0.0", "calibrated"},
                "attention-rbf-svm": {"fixed_0.0", "calibrated"},
            }
            event = predictions[
                predictions["condition"].eq("event_selected_event")
            ]
            reference = None
            for probe, frame in event.groupby("probe"):
                self.assertEqual(
                    set(frame["decision_rule"]), expected_rules[probe]
                )
                membership = (
                    frame[
                        [
                            "uid",
                            "lineage_group_id",
                            "outer_fold",
                            "y_true",
                        ]
                    ]
                    .drop_duplicates()
                    .sort_values("uid")
                    .reset_index(drop=True)
                )
                if reference is None:
                    reference = membership
                else:
                    pd.testing.assert_frame_equal(reference, membership)

            metrics = pd.read_csv(result.path("metrics"))
            self.assertEqual(len(metrics), 3 * 2 * 6)
            self.assertEqual(
                set(metrics["condition"]),
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                    "event_selected_removed",
                    "removed_selected_removed",
                    "contact_specific_increment",
                },
            )
            event_metrics = metrics[
                metrics["condition"].eq("event_selected_event")
            ]
            required_metric_columns = {
                "balanced_accuracy",
                "accuracy",
                "roc_auc",
                "macro_f1",
                "true_fly_pred_fly",
                "true_fly_pred_ground",
                "true_ground_pred_fly",
                "true_ground_pred_ground",
            }
            self.assertTrue(
                np.isfinite(
                    event_metrics[list(required_metric_columns)].to_numpy()
                ).all()
            )

            selections = pd.read_csv(result.path("selections"))
            self.assertEqual(len(selections), 3 * 5 * 3)
            self.assertTrue(np.isfinite(selections["selected_threshold"]).all())
            folds = pd.read_csv(result.path("fold_assignments"))
            self.assertEqual(
                list(folds.columns),
                ["uid", "label", "lineage_group_id", "outer_fold"],
            )
            self.assertEqual(len(folds), 20)
            membership_audit = json.loads(
                result.path("membership_audit").read_text(encoding="utf-8")
            )
            self.assertTrue(membership_audit["all_candidates_matched"])
            self.assertEqual(membership_audit["n_timing_eligible_pairs"], 20)
            self.assertEqual(
                membership_audit["n_singleton_lineage_groups"], 0
            )

            starting = pd.read_csv(result.path("starting_comparison"))
            self.assertEqual(len(starting), 3)
            self.assertAlmostEqual(
                float(
                    starting.loc[
                        starting["probe"].eq("attention-logistic"),
                        "starting_balanced_accuracy",
                    ].iloc[0]
                ),
                0.667,
            )
            report = result.path("report_zh").read_text(encoding="utf-8")
            self.assertIn("0.667", report)
            self.assertIn("0.714", report)
            self.assertIn("不替换", report)
            self.assertIn("不能证明跨比赛", report)
            self.assertIn("跨采集流程泛化", report)

    def test_protocol_audits_shared_representation_and_inner_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_margin_classifier_evaluation(
                source.root,
                root / "margin-family",
                MarginClassifierEvaluationConfig(n_bootstrap=20, seed=23),
            )

            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            shared = protocol["shared_source_representation"]
            self.assertEqual(shared["pooling"], "attention")
            self.assertEqual(shared["window_conditions"], ["event_200ms"])
            self.assertEqual(
                shared["attention_control_transform_policy"],
                "event_fitted_transfer_v1",
            )
            self.assertEqual(len(shared["source_features_sha256"]), 64)
            self.assertTrue(protocol["candidate_input_audit"]["all_matched"])
            self.assertEqual(
                set(
                    protocol["candidate_input_audit"][
                        "source_features_sha256_by_probe"
                    ].values()
                ),
                {shared["source_features_sha256"]},
            )
            self.assertTrue(
                all(
                    candidate["selection_scope"]
                    == "outer_train_inner_grouped_validation"
                    for candidate in protocol["candidate_family"]
                )
            )

            selections = pd.read_csv(result.path("selections"))
            expected_candidates = {
                "attention-logistic": 3,
                "attention-linear-svm": 5,
                "attention-rbf-svm": 6,
            }
            for probe, frame in selections.groupby("probe"):
                self.assertTrue(
                    frame["candidate_scores_json"].map(
                        lambda value: len(json.loads(value))
                        == expected_candidates[probe]
                    ).all()
                )
                self.assertTrue(
                    frame["threshold_scores_json"].map(
                        lambda value: len(json.loads(value)) > 0
                    ).all()
                )

    def test_rerun_is_deterministic_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            config = MarginClassifierEvaluationConfig(
                n_bootstrap=20,
                seed=31,
            )
            first = run_margin_classifier_evaluation(
                source.root, root / "first", config
            )
            second = run_margin_classifier_evaluation(
                source.root, root / "second", config
            )
            changed = run_margin_classifier_evaluation(
                source.root,
                root / "changed",
                MarginClassifierEvaluationConfig(
                    n_bootstrap=20,
                    seed=32,
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
            provenance = json.loads(
                first.path("provenance").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["encoder_inference_runs"], 0)
            with self.assertRaisesRegex(ValueError, "locked at 0.02"):
                run_margin_classifier_evaluation(
                    source.root,
                    root / "weakened-gate",
                    MarginClassifierEvaluationConfig(
                        n_bootstrap=20,
                        seed=31,
                        minimum_headline_ba_gain=0.019,
                    ),
                )

    def test_paired_group_intervals_drive_the_stop_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_margin_classifier_evaluation(
                source.root,
                root / "margin-family",
                MarginClassifierEvaluationConfig(
                    n_bootstrap=40,
                    seed=19,
                    minimum_headline_ba_gain=0.02,
                ),
            )

            paired = pd.read_csv(result.path("paired_improvements"))
            self.assertEqual(set(paired["bootstrap_unit"]), {"lineage_group_id"})
            self.assertEqual(set(paired["n_groups"]), {10})
            baseline = paired[
                paired["probe"].eq("attention-logistic")
                & paired["decision_rule"].eq("fixed_0.5")
            ].iloc[0]
            for column in (
                "event_ba_gain",
                "event_ba_gain_ci_low",
                "event_ba_gain_ci_high",
                "contact_specific_increment_gain",
                "contact_specific_increment_gain_ci_low",
                "contact_specific_increment_gain_ci_high",
            ):
                self.assertEqual(float(baseline[column]), 0.0)

            metrics = pd.read_csv(result.path("metrics"))
            baseline_metrics = metrics[
                metrics["probe"].eq("attention-logistic")
                & metrics["decision_rule"].eq("fixed_0.5")
            ].set_index("condition")
            for row in paired.itertuples(index=False):
                candidate = metrics[
                    metrics["probe"].eq(row.probe)
                    & metrics["decision_rule"].eq(row.decision_rule)
                ].set_index("condition")
                expected_event_gain = float(
                    candidate.loc["event_selected_event", "balanced_accuracy"]
                    - baseline_metrics.loc[
                        "event_selected_event", "balanced_accuracy"
                    ]
                )
                expected_increment_gain = float(
                    candidate.loc[
                        "contact_specific_increment", "balanced_accuracy"
                    ]
                    - baseline_metrics.loc[
                        "contact_specific_increment", "balanced_accuracy"
                    ]
                )
                self.assertAlmostEqual(row.event_ba_gain, expected_event_gain)
                self.assertAlmostEqual(
                    row.contact_specific_increment_gain,
                    expected_increment_gain,
                )

            verdict = json.loads(
                result.path("verdict").read_text(encoding="utf-8")
            )
            self.assertEqual(verdict["decision"], "stop")
            self.assertFalse(verdict["headline_replacement_allowed"])
            self.assertIn("eligible_for_downstream_validation", verdict)
            self.assertFalse(
                any(
                    item["qualifies_for_continuation"]
                    for item in verdict["evaluated_arms"]
                    if item["event_ba_gain"] < 0.02
                )
            )


if __name__ == "__main__":
    unittest.main()
