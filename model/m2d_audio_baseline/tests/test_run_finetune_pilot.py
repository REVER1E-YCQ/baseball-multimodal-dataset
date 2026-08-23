from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.run_finetune_pilot import (
    build_pilot_comparison,
    resolve_finetune_benchmarks,
)


M2D = "m2d_vit_base_80x200p16x4_40ms"
REVISION = "verified-snapshot"


def _write_bundle(
    root: Path,
    artifact_id: str,
    *,
    pooling: str,
    controls: bool,
    normalization: str = "snapshot_level",
    windows: tuple[str, ...] = ("event_200ms",),
    seed: int = 20260805,
    calibrated: bool = False,
    fold_offset: int = 0,
) -> Path:
    bundle = root / artifact_id
    bundle.mkdir(parents=True)
    protocol = {
        "artifact_id": artifact_id,
        "dataset": {
            "revision": REVISION,
            "snapshot_fingerprint": "snapshot-fingerprint",
        },
        "encoders": [{"name": M2D}],
        "pooling": pooling,
        "window_conditions": list(windows),
        "normalization": normalization,
        "controls": {"enabled": controls},
        "fold_policy": {
            "group": "lineage_group_id",
            "seed": seed,
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
            "event_fitted_transfer_v1"
            if pooling == "attention" and controls
            else None
        ),
    }
    (bundle / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    rows = [
        {
            "uid": "u1",
            "label": "fly_ball",
            "lineage_group_id": "game-1",
            "outer_fold": fold_offset,
        },
        {
            "uid": "u2",
            "label": "ground_ball",
            "lineage_group_id": "game-2",
            "outer_fold": 1 + fold_offset,
        },
    ]
    pd.DataFrame(rows).to_csv(bundle / "fold_assignments.csv", index=False)
    pd.DataFrame([
        {"uid": "u1", "window_name": "event_200ms"},
        {"uid": "u2", "window_name": "event_200ms"},
    ]).to_csv(bundle / "windows_manifest.csv", index=False)
    metric = {
        "encoder": M2D,
        "balanced_accuracy": 0.619 if not controls else 0.667,
        "eligible_samples": 2,
    }
    if controls:
        metric.update({
            "condition": "event_selected_event",
            "window_ms": 200,
            "decision_rule": "fixed_0.5",
        })
    pd.DataFrame([metric]).to_csv(bundle / "metrics.csv", index=False)
    return bundle


class PilotComparisonTest(unittest.TestCase):
    def test_positive_gain_opens_direction(self) -> None:
        summary = build_pilot_comparison(
            finetune_ba=0.70,
            finetune_eligible=817,
            frozen_mean_ba=0.619,
            frozen_mean_eligible=817,
            attention_headline_ba=0.667,
            attention_headline_eligible=803,
        )
        self.assertEqual(summary["direction"], "positive")
        self.assertEqual(summary["conclusion"], "fine_tuning_open")
        self.assertAlmostEqual(summary["gain_vs_frozen_mean"], 0.081)
        self.assertIn(
            "reference", summary["attention_headline_note"]
        )

    def test_negative_gain_closes_direction(self) -> None:
        summary = build_pilot_comparison(
            finetune_ba=0.60,
            finetune_eligible=817,
            frozen_mean_ba=0.619,
            frozen_mean_eligible=817,
            attention_headline_ba=0.667,
            attention_headline_eligible=803,
        )
        self.assertEqual(summary["direction"], "negative")
        self.assertEqual(summary["conclusion"], "fine_tuning_closed")

    def test_neutral_gain_closes_direction(self) -> None:
        summary = build_pilot_comparison(
            finetune_ba=0.621,
            finetune_eligible=817,
            frozen_mean_ba=0.619,
            frozen_mean_eligible=817,
            attention_headline_ba=0.667,
            attention_headline_eligible=803,
        )
        self.assertEqual(summary["direction"], "neutral")
        self.assertEqual(summary["conclusion"], "fine_tuning_closed")

    def test_resolves_exact_mean_and_attention_benchmark_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            pooling = root / "pooling"
            intended_mean = _write_bundle(
                primary,
                "mean-primary",
                pooling="valid_final_layer_token_mean",
                controls=False,
            )
            _write_bundle(
                primary,
                "mean-rms",
                pooling="valid_final_layer_token_mean",
                controls=False,
                normalization="rms_normalized",
            )
            intended_attention = _write_bundle(
                pooling,
                "attention-headline",
                pooling="attention",
                controls=True,
            )
            _write_bundle(
                pooling,
                "attention-50ms",
                pooling="attention",
                controls=True,
                windows=("event_050ms",),
            )
            _write_bundle(
                pooling,
                "attention-calibrated",
                pooling="attention",
                controls=True,
                calibrated=True,
            )

            references = resolve_finetune_benchmarks(
                primary_root=primary,
                pooling_root=pooling,
                seed=20260805,
                dataset_revision=REVISION,
            )

            self.assertEqual(references.frozen_mean_bundle, intended_mean)
            self.assertEqual(
                references.attention_headline_bundle, intended_attention
            )
            self.assertAlmostEqual(
                float(references.frozen_mean_metric["balanced_accuracy"]),
                0.619,
            )
            self.assertAlmostEqual(
                float(references.attention_headline_metric["balanced_accuracy"]),
                0.667,
            )

    def test_rejects_incompatible_fold_roles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = root / "primary"
            pooling = root / "pooling"
            _write_bundle(
                primary,
                "mean-primary",
                pooling="valid_final_layer_token_mean",
                controls=False,
            )
            _write_bundle(
                pooling,
                "attention-headline",
                pooling="attention",
                controls=True,
                fold_offset=1,
            )

            with self.assertRaisesRegex(ValueError, "incompatible fold roles"):
                resolve_finetune_benchmarks(
                    primary_root=primary,
                    pooling_root=pooling,
                    seed=20260805,
                    dataset_revision=REVISION,
                )

    def test_rejects_a_gain_between_incompatible_sample_roles(self) -> None:
        with self.assertRaisesRegex(ValueError, "eligible-sample role"):
            build_pilot_comparison(
                finetune_ba=0.60,
                finetune_eligible=817,
                frozen_mean_ba=0.619,
                frozen_mean_eligible=803,
                attention_headline_ba=0.667,
                attention_headline_eligible=803,
            )


if __name__ == "__main__":
    unittest.main()
