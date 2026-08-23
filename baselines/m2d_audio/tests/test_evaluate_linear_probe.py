from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_linear_probe import evaluate


class EvaluateLinearProbeTest(unittest.TestCase):
    def test_signal_and_group_isolation(self) -> None:
        rng = np.random.default_rng(20260716)
        rows: list[dict[str, object]] = []
        for group_index in range(30):
            for sample_index in range(2):
                label = "fly_ball" if sample_index == 0 else "ground_ball"
                class_value = -1.0 if label == "fly_ball" else 1.0
                uid = f"g{group_index:02d}_s{sample_index}"
                event_features = np.array(
                    [
                        3.0 * class_value + rng.normal(0.0, 0.15),
                        rng.normal(),
                        rng.normal(),
                        rng.normal(),
                    ]
                )
                pre_features = rng.normal(size=4)
                for window_name, values in [
                    ("event_200ms", event_features),
                    ("pre_200ms", pre_features),
                ]:
                    row: dict[str, object] = {
                        "uid": uid,
                        "label": label,
                        "source_id": f"group_{group_index:02d}",
                        "protocol_role": "primary_dev",
                        "window_name": window_name,
                    }
                    row.update(
                        {
                            f"feat_{feature_index:03d}": float(value)
                            for feature_index, value in enumerate(values)
                        }
                    )
                    rows.append(row)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.csv"
            out_dir = root / "evaluation"
            pd.DataFrame(rows).to_csv(features, index=False)
            summary = evaluate(
                features,
                out_dir,
                outer_splits=3,
                inner_splits=2,
                repeats=1,
                c_grid=(0.01,),
            )

            self.assertEqual(
                set(summary["condition"]),
                {
                    "event_selected_event",
                    "event_selected_pre",
                    "pre_selected_pre",
                },
            )
            event_ba = float(
                summary.loc[
                    summary["condition"].eq("event_selected_event"),
                    "balanced_accuracy_mean",
                ].iloc[0]
            )
            self.assertGreater(event_ba, 0.95)

            predictions = pd.read_csv(out_dir / "outer_predictions.csv")
            group_fold_counts = predictions.groupby(
                ["repeat", "condition", "source_id"]
            )["outer_fold"].nunique()
            self.assertTrue((group_fold_counts == 1).all())

            protocol = json.loads((out_dir / "protocol.json").read_text(encoding="utf-8"))
            self.assertEqual(protocol["n_source_groups"], 30)
            self.assertEqual(protocol["n_singleton_source_groups"], 0)
            self.assertTrue(protocol["grouping_effective_beyond_stratification"])
            self.assertFalse(protocol["locked_test_used"])


if __name__ == "__main__":
    unittest.main()

