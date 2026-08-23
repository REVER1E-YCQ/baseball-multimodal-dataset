from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_alignment_sensitivity import build_alignment_curve


class AlignmentCurveTest(unittest.TestCase):
    def _rows(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"shift_ms": -100, "event_balanced_accuracy": 0.56,
                 "event_roc_auc": 0.60, "eligible_samples": 780},
                {"shift_ms": -50, "event_balanced_accuracy": 0.60,
                 "event_roc_auc": 0.63, "eligible_samples": 790},
                {"shift_ms": -25, "event_balanced_accuracy": 0.64,
                 "event_roc_auc": 0.67, "eligible_samples": 795},
                {"shift_ms": 0, "event_balanced_accuracy": 0.67,
                 "event_roc_auc": 0.70, "eligible_samples": 803},
                {"shift_ms": 25, "event_balanced_accuracy": 0.63,
                 "event_roc_auc": 0.66, "eligible_samples": 795},
                {"shift_ms": 50, "event_balanced_accuracy": 0.58,
                 "event_roc_auc": 0.61, "eligible_samples": 790},
                {"shift_ms": 100, "event_balanced_accuracy": 0.52,
                 "event_roc_auc": 0.56, "eligible_samples": 780},
            ]
        )

    def test_curve_orders_shifts_and_computes_deltas(self) -> None:
        curve, summary = build_alignment_curve(self._rows())
        self.assertEqual(
            list(curve["shift_ms"]),
            [-100, -50, -25, 0, 25, 50, 100],
        )
        self.assertAlmostEqual(
            float(curve.loc[curve["shift_ms"].eq(50), "delta_vs_0ms"].iloc[0]),
            0.58 - 0.67,
        )
        self.assertAlmostEqual(summary["reference_0ms_balanced_accuracy"], 0.67)
        self.assertAlmostEqual(summary["drop_at_50ms"], 0.67 - 0.58)
        self.assertAlmostEqual(summary["symmetry_abs_diff_50ms"], 0.02)
        self.assertTrue(summary["monotonic_away_from_0ms"])
        self.assertEqual(summary["interpretation"], "precise_alignment_dependence")

    def test_coarse_curve_gets_coarse_interpretation(self) -> None:
        rows = self._rows()
        rows.loc[rows["shift_ms"].eq(100), "event_balanced_accuracy"] = 0.65
        rows.loc[rows["shift_ms"].eq(-100), "event_balanced_accuracy"] = 0.66
        rows.loc[rows["shift_ms"].eq(50), "event_balanced_accuracy"] = 0.66
        rows.loc[rows["shift_ms"].eq(-50), "event_balanced_accuracy"] = 0.665
        rows.loc[rows["shift_ms"].eq(25), "event_balanced_accuracy"] = 0.68
        rows.loc[rows["shift_ms"].eq(-25), "event_balanced_accuracy"] = 0.675
        _curve, summary = build_alignment_curve(rows)
        self.assertEqual(summary["interpretation"], "coarse_content_dependence")
        self.assertFalse(summary["monotonic_away_from_0ms"])

    def test_missing_reference_or_50ms_fails_visibly(self) -> None:
        rows = self._rows()
        rows = rows[rows["shift_ms"].ne(0)]
        with self.assertRaises(ValueError) as context:
            build_alignment_curve(rows)
        self.assertIn("0 ms", str(context.exception))

        rows = self._rows()
        rows = rows[rows["shift_ms"].abs().ne(50)]
        with self.assertRaises(ValueError) as context:
            build_alignment_curve(rows)
        self.assertIn("50 ms", str(context.exception))

    def test_missing_columns_fail_visibly(self) -> None:
        rows = self._rows().drop(columns=["eligible_samples"])
        with self.assertRaises(ValueError) as context:
            build_alignment_curve(rows)
        self.assertIn("eligible_samples", str(context.exception))


if __name__ == "__main__":
    unittest.main()
