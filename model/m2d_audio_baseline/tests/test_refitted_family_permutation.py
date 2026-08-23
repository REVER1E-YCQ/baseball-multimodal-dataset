from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from locked_attention_fixture import build_locked_attention_source
from scripts.exploratory_probe_benchmark import ProbeConfig
from scripts.refitted_family_permutation import (
    PermutationFamilyConfig,
    run_refitted_family_permutation,
)


def _logistic(name: str = "logistic") -> ProbeConfig:
    return ProbeConfig(
        name=name,
        estimator_family="balanced_l2_logistic_regression",
        hyperparameter_grid={"C": (0.01,)},
        score_output="probability_ground_ball",
    )


def _linear_svm(name: str = "linear-svm") -> ProbeConfig:
    return ProbeConfig(
        name=name,
        estimator_family="balanced_linear_svm",
        hyperparameter_grid={"C": (0.1,)},
        score_output="decision_function_ground_ball",
    )


def _update_manifest_checksum(source, artifact_name: str) -> Path:
    manifest_path = source.path("artifact_bundle")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest["artifacts"][artifact_name]
    artifact_path = source.root / record["path"]
    record["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return artifact_path


def _make_lineages_singleton(source) -> None:
    folds = pd.read_csv(source.path("fold_assignments"))
    folds["lineage_group_id"] = "singleton-" + folds["uid"].astype(str)
    folds.to_csv(source.path("fold_assignments"), index=False)
    _update_manifest_checksum(source, "fold_assignments")


def _make_partly_mixed_lineages(source) -> None:
    folds = pd.read_csv(source.path("fold_assignments"))
    keep_mixed = folds["lineage_group_id"].isin({"game-00", "game-01"})
    folds.loc[~keep_mixed, "lineage_group_id"] = (
        "singleton-" + folds.loc[~keep_mixed, "uid"].astype(str)
    )
    folds.to_csv(source.path("fold_assignments"), index=False)
    _update_manifest_checksum(source, "fold_assignments")


class RefittedFamilyPermutationTest(unittest.TestCase):
    def test_rejects_a_one_member_post_selection_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)

            with self.assertRaisesRegex(ValueError, "at least two"):
                run_refitted_family_permutation(
                    source.root,
                    root / "family-evidence",
                    PermutationFamilyConfig(
                        name="invalid-single-winner",
                        candidates=(_logistic(),),
                        n_permutations=2,
                        seed=17,
                    ),
                )
            with self.assertRaisesRegex(ValueError, "scientifically distinct"):
                run_refitted_family_permutation(
                    source.root,
                    root / "renamed-duplicates",
                    PermutationFamilyConfig(
                        name="renamed-duplicate-family",
                        candidates=(_logistic("first"), _logistic("second")),
                        n_permutations=2,
                        seed=17,
                    ),
                )

    def test_each_replicate_refits_the_synchronized_complete_family(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_refitted_family_permutation(
                source.root,
                root / "family-evidence",
                PermutationFamilyConfig(
                    name="margin-family",
                    candidates=(_logistic(), _linear_svm()),
                    n_permutations=2,
                    seed=23,
                ),
            )

            scores = pd.read_csv(result.path("permutation_scores"))
            self.assertEqual(set(scores["permutation"]), {-1, 0, 1})
            self.assertEqual(
                set(scores["candidate"]), {"logistic", "linear-svm"}
            )
            self.assertEqual(
                set(scores["condition"]),
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                    "event_selected_removed",
                    "removed_selected_removed",
                    "contact_specific_increment",
                },
            )

            audit = pd.read_csv(result.path("fit_audit"))
            self.assertEqual(len(audit), 2 * 3)
            self.assertTrue((audit["representation_fits"] > 0).all())
            self.assertTrue((audit["model_selection_fits"] > 0).all())
            self.assertTrue((audit["outer_probe_fits"] > 0).all())
            self.assertEqual(
                audit.groupby("permutation")["label_assignment_sha256"]
                .nunique()
                .to_dict(),
                {-1: 1, 0: 1, 1: 1},
            )

            selections = pd.read_csv(result.path("permutation_selections"))
            self.assertEqual(
                set(selections["permutation"]), {-1, 0, 1}
            )
            summary = pd.read_csv(result.path("permutation_summary"))
            self.assertEqual(len(summary), 2)
            self.assertEqual(set(summary["n_family_hypotheses"]), {2})
            self.assertTrue(
                (
                    summary["max_stat_familywise_p_value"]
                    >= summary["raw_p_value"]
                ).all()
            )
            event_null = scores[
                ~scores["is_observed"]
                & scores["condition"].eq("event_selected_event")
            ]
            max_null = event_null.groupby("permutation")[
                "balanced_accuracy"
            ].max()
            for row in summary.itertuples(index=False):
                expected_family_p = (
                    1
                    + int(
                        (
                            max_null
                            >= float(row.observed_balanced_accuracy)
                        ).sum()
                    )
                ) / 3
                self.assertAlmostEqual(
                    float(row.max_stat_familywise_p_value),
                    expected_family_p,
                )
            screening = pd.read_csv(result.path("screening_inputs"))
            self.assertEqual(len(screening), 2)
            self.assertTrue(screening["family_complete_as_declared"].all())

    def test_calibrated_hypothesis_refits_thresholds_in_every_replicate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            calibrated = ProbeConfig(
                name="calibrated-logistic",
                estimator_family="balanced_l2_logistic_regression",
                hyperparameter_grid={"C": (0.01,)},
                score_output="probability_ground_ball",
                calibrate_threshold=True,
            )
            result = run_refitted_family_permutation(
                source.root,
                root / "family-evidence",
                PermutationFamilyConfig(
                    name="threshold-family",
                    candidates=(calibrated, _linear_svm()),
                    n_permutations=1,
                    seed=27,
                ),
            )

            audit = pd.read_csv(result.path("fit_audit"))
            calibrated_audit = audit[
                audit["candidate"].eq("calibrated-logistic")
            ]
            self.assertEqual(set(calibrated_audit["permutation"]), {-1, 0})
            self.assertTrue(
                (calibrated_audit["threshold_selection_fits"] > 0).all()
            )
            fixed_audit = audit[audit["candidate"].eq("linear-svm")]
            self.assertTrue(
                (fixed_audit["threshold_selection_fits"] == 0).all()
            )
            summary = pd.read_csv(result.path("permutation_summary"))
            self.assertEqual(
                set(
                    summary[summary["candidate"].eq("calibrated-logistic")][
                        "decision_rule"
                    ]
                ),
                {"fixed_0.5", "calibrated"},
            )
            self.assertEqual(set(summary["n_family_hypotheses"]), {3})

    def test_groupwise_assignments_preserve_folds_totals_and_mixed_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            _make_partly_mixed_lineages(source)
            result = run_refitted_family_permutation(
                source.root,
                root / "family-evidence",
                PermutationFamilyConfig(
                    name="auditable-exchangeability",
                    candidates=(_logistic(), _linear_svm()),
                    n_permutations=3,
                    seed=29,
                ),
            )

            assignments = pd.read_csv(
                result.path("permutation_assignments")
            )
            observed = assignments[assignments["permutation"].eq(-1)]
            expected_fold_totals = (
                observed.groupby("outer_fold")["y_true"].sum().to_dict()
            )
            mixed_groups = set(
                observed.groupby("lineage_group_id")["uid"]
                .size()
                .loc[lambda sizes: sizes > 1]
                .index
            )
            self.assertTrue(mixed_groups)
            for permutation, frame in assignments.groupby("permutation"):
                self.assertEqual(
                    frame.groupby("outer_fold")["y_true"].sum().to_dict(),
                    expected_fold_totals,
                    msg=f"class totals changed in permutation {permutation}",
                )
                mixed_label_counts = (
                    frame[frame["lineage_group_id"].isin(mixed_groups)]
                    .groupby("lineage_group_id")["y_true"]
                    .nunique()
                )
                self.assertTrue(
                    (mixed_label_counts == 2).all(),
                    msg=f"mixed-label group collapsed in {permutation}",
                )
            self.assertGreater(
                assignments["label_assignment_sha256"].nunique(), 1
            )

            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            policy = protocol["permutation_policy"]
            self.assertEqual(policy["unit"], "lineage_group_label_vector")
            self.assertIn("per_fold_class_totals", policy["preserves"])
            self.assertIn(
                "mixed_label_lineage_groups", policy["preserves"]
            )
            self.assertEqual(
                policy["exchangeability_blocks"],
                ["locked_outer_fold", "lineage_group_size"],
            )

    def test_rerun_is_content_addressed_deterministic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            config = PermutationFamilyConfig(
                name="resumable-family",
                candidates=(_logistic(), _linear_svm()),
                n_permutations=2,
                seed=31,
            )
            first = run_refitted_family_permutation(
                source.root, root / "family-evidence", config
            )
            index_path = first.path("checkpoint_index")
            first_index_bytes = index_path.read_bytes()
            index = json.loads(first_index_bytes)
            self.assertEqual(len(index["checkpoints"]), 6)
            missing_checkpoint = index["checkpoints"][-1]["path"]
            checkpoint_mtimes = {
                record["path"]: (first.root / record["path"]).stat().st_mtime_ns
                for record in index["checkpoints"]
                if record["path"] != missing_checkpoint
            }
            first_summary_bytes = first.path("permutation_summary").read_bytes()
            (first.root / missing_checkpoint).unlink()
            first.path("checkpoint_index").unlink()
            first.path("permutation_summary").unlink()
            first.path("artifact_bundle").unlink()

            second = run_refitted_family_permutation(
                source.root, root / "family-evidence", config
            )

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertEqual(
                first_index_bytes, second.path("checkpoint_index").read_bytes()
            )
            self.assertEqual(
                first_summary_bytes,
                second.path("permutation_summary").read_bytes(),
            )
            for relative_path, first_mtime in checkpoint_mtimes.items():
                self.assertEqual(
                    (second.root / relative_path).stat().st_mtime_ns,
                    first_mtime,
                )

    def test_full_refit_avoids_a_misleading_fixed_prediction_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            _make_lineages_singleton(source)
            candidates = (_logistic(), _linear_svm())
            discovery = run_refitted_family_permutation(
                source.root,
                root / "assignment-discovery",
                PermutationFamilyConfig(
                    name="assignment-discovery",
                    candidates=candidates,
                    n_permutations=1,
                    seed=37,
                ),
            )
            assignments = pd.read_csv(
                discovery.path("permutation_assignments")
            )
            observed_labels = assignments[
                assignments["permutation"].eq(-1)
            ].set_index("uid")["y_true"]
            first_null_labels = assignments[
                assignments["permutation"].eq(0)
            ].set_index("uid")["y_true"]

            manifest = json.loads(
                source.path("artifact_bundle").read_text(encoding="utf-8")
            )
            feature_name = next(
                name
                for name in manifest["artifacts"]
                if name.startswith("features/")
            )
            feature_path = source.root / manifest["artifacts"][feature_name]["path"]
            features = pd.read_csv(feature_path)
            features["feat_0"] = features["uid"].map(observed_labels) * 2.0 - 1.0
            features["feat_1"] = (
                features["uid"].map(first_null_labels) * 2.0 - 1.0
            )
            features["feat_2"] = features["token_index"].astype(float) * 0.01
            features["feat_3"] = 0.0
            features.to_csv(feature_path, index=False)
            _update_manifest_checksum(source, feature_name)

            result = run_refitted_family_permutation(
                source.root,
                root / "family-evidence",
                PermutationFamilyConfig(
                    name="refit-versus-fixed",
                    candidates=candidates,
                    n_permutations=20,
                    seed=37,
                ),
            )

            diagnostic = pd.read_csv(
                result.path("fixed_prediction_diagnostic")
            )
            linear = diagnostic[
                diagnostic["candidate"].eq("linear-svm")
                & diagnostic["decision_rule"].eq("fixed_0.0")
            ].iloc[0]
            self.assertLessEqual(
                float(linear["fixed_prediction_raw_p_value"]), 0.05
            )
            self.assertGreater(
                float(linear["full_refit_raw_p_value"]), 0.05
            )
            self.assertTrue(bool(linear["verdict_changed_by_refitting"]))

            predictions = pd.read_csv(
                result.path("observed_predictions")
            )
            event = predictions[
                predictions["candidate"].eq("linear-svm")
                & predictions["decision_rule"].eq("fixed_0.0")
                & predictions["condition"].eq("event_selected_event")
            ].set_index("uid")
            observed_score = balanced_accuracy_score(
                event["y_true"], event["y_pred"]
            )
            fixed_null_scores = []
            final_assignments = pd.read_csv(
                result.path("permutation_assignments")
            )
            for permutation in range(20):
                permuted = final_assignments[
                    final_assignments["permutation"].eq(permutation)
                ].set_index("uid").loc[event.index]
                fixed_null_scores.append(
                    balanced_accuracy_score(
                        permuted["y_true"], event["y_pred"]
                    )
                )
            independently_computed_p = (
                1 + int(np.sum(np.asarray(fixed_null_scores) >= observed_score))
            ) / 21
            self.assertAlmostEqual(
                float(linear["fixed_prediction_raw_p_value"]),
                independently_computed_p,
            )


if __name__ == "__main__":
    unittest.main()
