from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .cached_attention_controls import reevaluate_cached_attention_controls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate attention negative controls from an existing frozen "
            "token bundle without running the encoder."
        )
    )
    parser.add_argument("source_bundle", type=Path)
    parser.add_argument("output_root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = reevaluate_cached_attention_controls(
        args.source_bundle,
        args.output_root,
    )
    metrics = pd.read_csv(bundle.path("metrics"))
    if "decision_rule" in metrics.columns:
        metrics = metrics[metrics["decision_rule"].eq("fixed_0.5")]
    columns = ["condition", "window_ms", "balanced_accuracy"]
    print(metrics[columns].to_string(index=False))
    print(f"Re-evaluated bundle: {bundle.root}")


if __name__ == "__main__":
    main()
