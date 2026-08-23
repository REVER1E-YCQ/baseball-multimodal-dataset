from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .margin_classifier_evaluation import (
    MarginClassifierEvaluationConfig,
    run_margin_classifier_evaluation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the locked Logistic/Linear-SVM/RBF-SVM attention family "
            "with inner-OOF threshold calibration."
        )
    )
    parser.add_argument("source_bundle", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_margin_classifier_evaluation(
        args.source_bundle,
        args.output_root,
        MarginClassifierEvaluationConfig(
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        ),
    )
    metrics = pd.read_csv(result.path("metrics"))
    event = metrics[metrics["condition"].eq("event_selected_event")]
    print(
        event[
            [
                "probe",
                "decision_rule",
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
