from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.benchmark_artifact_roles import BenchmarkArtifactRoleError
from scripts.run_encoder_fusion import (
    build_fusion_comparison,
    find_bundle,
    fusion_source_role,
)


M2D = "m2d_vit_base_80x200p16x4_40ms"
REVISION = "verified-snapshot"


def _write_protocol(
    root: Path,
    artifact_id: str,
    *,
    pooling: str = "attention",
    windows: tuple[str, ...] = ("event_200ms",),
    normalization: str = "snapshot_level",
    calibrated: bool = False,
    attention_control_transform_policy: str | None = (
        "event_fitted_transfer_v1"
    ),
) -> Path:
    bundle = root / artifact_id
    bundle.mkdir(parents=True)
    (bundle / "protocol.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "dataset": {"revision": REVISION},
                "encoders": [{"name": M2D}],
                "pooling": pooling,
                "window_conditions": list(windows),
                "normalization": normalization,
                "controls": {"enabled": True},
                "fold_policy": {
                    "group": "lineage_group_id",
                    "seed": 20260805,
                },
                "classifier": {
                    "name": "balanced_l2_logistic_regression",
                    "C_selection": "inner_grouped_cv",
                },
                "decision_threshold": {
                    "calibrate": calibrated,
                    "fixed_default": 0.5,
                },
                "attention_control_transform_policy": (
                    attention_control_transform_policy
                ),
            }
        ),
        encoding="utf-8",
    )
    return bundle


def _table(fusion_ba: float, best_single_ba: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "name": "m2d_attention",
                "condition": "event_selected_event",
                "balanced_accuracy": best_single_ba,
                "roc_auc": 0.70,
                "eligible_samples": 803,
            },
            {
                "name": "beats_mean",
                "condition": "event_selected_event",
                "balanced_accuracy": 0.606,
                "roc_auc": 0.63,
                "eligible_samples": 803,
            },
            {
                "name": "m2d_attention+beats_mean",
                "condition": "event_selected_event",
                "balanced_accuracy": fusion_ba,
                "roc_auc": 0.72,
                "eligible_samples": 803,
            },
        ]
    )


class FusionComparisonTest(unittest.TestCase):
    def test_resolves_the_exact_fusion_source_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intended = _write_protocol(root, "intended")
            _write_protocol(root, "wrong-window", windows=("event_050ms",))
            _write_protocol(root, "wrong-normalization", normalization="rms_normalized")
            _write_protocol(root, "wrong-threshold", calibrated=True)
            _write_protocol(
                root,
                "legacy-policy",
                attention_control_transform_policy=None,
            )

            bundle, artifact_id = find_bundle(
                root,
                fusion_source_role(
                    name="m2d_attention_fusion_source",
                    encoder_name=M2D,
                    pooling="attention",
                    seed=20260805,
                    dataset_revision=REVISION,
                ),
            )

            self.assertEqual(bundle, intended)
            self.assertEqual(artifact_id, "intended")

    def test_rejects_ambiguous_fusion_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_protocol(root, "first")
            _write_protocol(root, "second")
            role = fusion_source_role(
                name="m2d_attention_fusion_source",
                encoder_name=M2D,
                pooling="attention",
                seed=20260805,
                dataset_revision=REVISION,
            )

            with self.assertRaisesRegex(
                BenchmarkArtifactRoleError, "2 matching artifacts"
            ):
                find_bundle(root, role)

    def test_positive_gain_opens_direction(self) -> None:
        summary = build_fusion_comparison(_table(0.690, 0.667))
        self.assertEqual(summary["direction"], "positive")
        self.assertEqual(summary["conclusion"], "fusion_open")
        self.assertAlmostEqual(summary["fusion_gain_vs_best_single"], 0.023)

    def test_negative_gain_closes_direction(self) -> None:
        summary = build_fusion_comparison(_table(0.640, 0.667))
        self.assertEqual(summary["direction"], "negative")
        self.assertEqual(summary["conclusion"], "fusion_closed")

    def test_neutral_gain_closes_direction(self) -> None:
        summary = build_fusion_comparison(_table(0.668, 0.667))
        self.assertEqual(summary["direction"], "neutral")
        self.assertEqual(summary["conclusion"], "fusion_closed")

    def test_missing_concatenation_row_fails_visibly(self) -> None:
        table = _table(0.69, 0.667).iloc[:2]
        with self.assertRaises(ValueError) as context:
            build_fusion_comparison(table)
        self.assertIn("concatenation", str(context.exception))

    def test_empty_table_fails_visibly(self) -> None:
        with self.assertRaises(ValueError) as context:
            build_fusion_comparison(pd.DataFrame())
        self.assertIn("Empty", str(context.exception))


if __name__ == "__main__":
    unittest.main()
