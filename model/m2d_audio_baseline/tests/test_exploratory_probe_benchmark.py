from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from locked_attention_fixture import build_locked_attention_source
from scripts.exploratory_probe_benchmark import (
    ExploratoryProbeError,
    ProbeConfig,
    run_exploratory_probe_benchmark,
)

class ExploratoryProbeBenchmarkTest(unittest.TestCase):
    def test_logistic_arm_reproduces_locked_attention_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_exploratory_probe_benchmark(
                source.root,
                root / "exploratory",
                ProbeConfig(
                    name="logistic-reproduction",
                    estimator_family="balanced_l2_logistic_regression",
                    hyperparameter_grid={"C": (0.001, 0.01, 0.1)},
                    score_output="probability_ground_ball",
                    calibrate_threshold=False,
                ),
            )

            source_predictions = pd.read_csv(
                source.path("oof_predictions")
            ).sort_values(["condition", "uid"]).reset_index(drop=True)
            result_predictions = pd.read_csv(
                result.path("oof_predictions")
            ).sort_values(["condition", "uid"]).reset_index(drop=True)
            columns = [
                "condition",
                "uid",
                "lineage_group_id",
                "outer_fold",
                "y_true",
                "y_pred",
            ]
            pd.testing.assert_frame_equal(
                source_predictions[columns], result_predictions[columns]
            )
            np.testing.assert_allclose(
                source_predictions["score_ground_ball"],
                result_predictions["score_ground_ball"],
                rtol=0,
                atol=1e-12,
            )

            expected_artifacts = {
                "artifact_bundle",
                "exclusions",
                "fold_assignments",
                "inner_fold_assignments",
                "metrics",
                "oof_predictions",
                "protocol",
                "provenance",
                "selections",
            }
            self.assertEqual(set(result.artifact_names), expected_artifacts)
            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(protocol["evidence_role"], "development_exploratory")
            self.assertTrue(protocol["primary_common_benchmark_unchanged"])
            self.assertEqual(
                protocol["source_artifact_id"], source.artifact_id
            )
            self.assertEqual(
                protocol["probe"]["estimator_family"],
                "balanced_l2_logistic_regression",
            )
            provenance = json.loads(
                result.path("provenance").read_text(encoding="utf-8")
            )
            self.assertEqual(provenance["encoder_inference_runs"], 0)
            self.assertIn("source_exclusions_sha256", provenance)

    def test_margin_probe_emits_default_and_calibrated_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_exploratory_probe_benchmark(
                source.root,
                root / "exploratory",
                ProbeConfig(
                    name="linear-margin",
                    estimator_family="balanced_linear_svm",
                    hyperparameter_grid={"C": (0.01, 0.1)},
                    score_output="decision_function_ground_ball",
                    calibrate_threshold=True,
                ),
            )

            predictions = pd.read_csv(result.path("oof_predictions"))
            self.assertEqual(
                set(predictions["decision_rule"]),
                {"fixed_0.0", "calibrated"},
            )
            self.assertTrue(
                np.isfinite(predictions["score_ground_ball"]).all()
            )
            selections = pd.read_csv(result.path("selections"))
            self.assertEqual(len(selections), 15)
            self.assertTrue(
                selections["selected_parameters_json"].map(
                    lambda value: set(json.loads(value)) == {"C"}
                ).all()
            )
            self.assertTrue(
                np.isfinite(selections["selected_threshold"]).all()
            )
            metrics = pd.read_csv(result.path("metrics"))
            self.assertTrue(metrics["exploratory"].all())
            self.assertTrue(metrics["primary_ranking_unchanged"].all())

    def test_outer_and_inner_selection_keep_lineage_groups_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            result = run_exploratory_probe_benchmark(
                source.root,
                root / "exploratory",
                ProbeConfig(
                    name="group-audit",
                    estimator_family="balanced_l2_logistic_regression",
                    hyperparameter_grid={"C": (0.01, 0.1)},
                    score_output="probability_ground_ball",
                ),
            )

            outer = pd.read_csv(result.path("fold_assignments"))
            self.assertTrue(
                (
                    outer.groupby("lineage_group_id")["outer_fold"].nunique()
                    == 1
                ).all()
            )
            inner = pd.read_csv(result.path("inner_fold_assignments"))
            self.assertTrue(
                (
                    inner.groupby(["outer_fold", "lineage_group_id"])[
                        "inner_fold"
                    ].nunique()
                    == 1
                ).all()
            )
            outer_fold_by_uid = outer.set_index("uid")["outer_fold"]
            self.assertTrue(
                all(
                    int(row.outer_fold)
                    != int(outer_fold_by_uid.loc[row.uid])
                    for row in inner.itertuples(index=False)
                )
            )
            expected_outer_training_appearances = outer["outer_fold"].nunique() - 1
            self.assertTrue(
                (
                    inner.groupby("uid").size()
                    == expected_outer_training_appearances
                ).all()
            )
            protocol = json.loads(
                result.path("protocol").read_text(encoding="utf-8")
            )
            self.assertEqual(
                protocol["probe"]["selection_scope"],
                "outer_train_inner_grouped_validation",
            )

    def test_reruns_are_deterministic_and_configuration_is_content_addressed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            config = ProbeConfig(
                name="deterministic-linear",
                estimator_family="balanced_linear_svm",
                hyperparameter_grid={"C": (0.01, 0.1)},
                score_output="decision_function_ground_ball",
            )
            first = run_exploratory_probe_benchmark(
                source.root, root / "first", config
            )
            second = run_exploratory_probe_benchmark(
                source.root, root / "second", config
            )
            changed = run_exploratory_probe_benchmark(
                source.root,
                root / "changed",
                ProbeConfig(
                    name="deterministic-linear",
                    estimator_family="balanced_linear_svm",
                    hyperparameter_grid={"C": (0.001, 0.01, 0.1)},
                    score_output="decision_function_ground_ball",
                ),
            )

            self.assertEqual(first.artifact_id, second.artifact_id)
            self.assertNotEqual(first.artifact_id, changed.artifact_id)
            for artifact_name in first.artifact_names:
                self.assertEqual(
                    first.path(artifact_name).read_bytes(),
                    second.path(artifact_name).read_bytes(),
                )

    def test_rejects_incompatible_identity_and_non_finite_cached_features(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_locked_attention_source(root)
            config = ProbeConfig(
                name="source-validation",
                estimator_family="balanced_l2_logistic_regression",
                hyperparameter_grid={"C": (0.01,)},
                score_output="probability_ground_ball",
            )

            incompatible = root / "incompatible"
            shutil.copytree(source.root, incompatible)
            manifest_path = incompatible / "artifact_bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["artifact_id"] = "not-the-source-protocol-id"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(
                ExploratoryProbeError, "artifact identity"
            ):
                run_exploratory_probe_benchmark(
                    incompatible, root / "rejected", config
                )

            wrong_role = root / "wrong-role"
            shutil.copytree(source.root, wrong_role)
            wrong_protocol_path = wrong_role / "protocol.json"
            wrong_protocol = json.loads(
                wrong_protocol_path.read_text(encoding="utf-8")
            )
            wrong_protocol["fold_policy"]["seed"] = 99
            identity_document = {
                key: value
                for key, value in wrong_protocol.items()
                if key != "artifact_id"
            }
            wrong_protocol["artifact_id"] = hashlib.sha256(
                json.dumps(
                    identity_document,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:24]
            wrong_protocol_path.write_text(
                json.dumps(wrong_protocol), encoding="utf-8"
            )
            wrong_manifest_path = wrong_role / "artifact_bundle.json"
            wrong_manifest = json.loads(
                wrong_manifest_path.read_text(encoding="utf-8")
            )
            wrong_manifest["artifact_id"] = wrong_protocol["artifact_id"]
            wrong_manifest["artifacts"]["protocol"]["sha256"] = (
                hashlib.sha256(wrong_protocol_path.read_bytes()).hexdigest()
            )
            wrong_manifest_path.write_text(
                json.dumps(wrong_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExploratoryProbeError, "locked M2D"
            ):
                run_exploratory_probe_benchmark(
                    wrong_role, root / "wrong-role-output", config
                )

            non_finite = root / "non-finite"
            shutil.copytree(source.root, non_finite)
            non_finite_manifest_path = non_finite / "artifact_bundle.json"
            non_finite_manifest = json.loads(
                non_finite_manifest_path.read_text(encoding="utf-8")
            )
            feature_name, feature_record = next(
                (name, record)
                for name, record in non_finite_manifest["artifacts"].items()
                if name.startswith("features/")
            )
            feature_path = non_finite / feature_record["path"]
            features = pd.read_csv(feature_path)
            feature_column = next(
                column for column in features if column.startswith("feat_")
            )
            features.loc[0, feature_column] = np.nan
            features.to_csv(feature_path, index=False)
            feature_record["sha256"] = hashlib.sha256(
                feature_path.read_bytes()
            ).hexdigest()
            non_finite_manifest["artifacts"][feature_name] = feature_record
            non_finite_manifest_path.write_text(
                json.dumps(non_finite_manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                ExploratoryProbeError, "Non-finite"
            ):
                run_exploratory_probe_benchmark(
                    non_finite, root / "non-finite-output", config
                )


if __name__ == "__main__":
    unittest.main()
