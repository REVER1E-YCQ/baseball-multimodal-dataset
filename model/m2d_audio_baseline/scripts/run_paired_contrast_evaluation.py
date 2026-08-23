from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .paired_contrast_evaluation import (
    PairedContrastEvaluationConfig,
    run_paired_contrast_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked three-seed M2D Event/Pre contrast family."
    )
    parser.add_argument(
        "source_bundle",
        type=Path,
        help="Corrected locked M2D attention-control artifact bundle.",
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_paired_contrast_evaluation(
        args.source_bundle,
        args.output_dir,
        PairedContrastEvaluationConfig(
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        ),
    )
    metrics = pd.read_csv(result.path("metrics"))
    print(
        metrics[metrics["condition"] == "event"][
            [
                "fold_seed",
                "arm",
                "balanced_accuracy",
                "accuracy",
                "roc_auc",
                "macro_f1",
            ]
        ].to_string(index=False)
    )
    print(result.path("report_zh"))


if __name__ == "__main__":
    main()
