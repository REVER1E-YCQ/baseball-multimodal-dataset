from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit_verified_snapshot import PINNED_REVISION, audit_verified_snapshot
from .contact_window_augmentation import (
    ContactWindowAugmentationConfig,
    run_contact_window_augmentation_evaluation,
)
from .m2d_encoder import M2DEncoderAdapter


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_SOURCE_BUNDLE = (
    REPO_ROOT / "outputs/m2d_pooling_ablation/4347fe8b746bfe7d9c827727"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/m2d_contact_window_augmentation"
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "data/branch_datasets_20260804/baseball-multimodal-dataset"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f"
)
DEFAULT_M2D_ROOT = REPO_ROOT / "external/m2d"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the locked train-only M2D contact-window augmentation "
            "family with unmodified outer-test audio."
        )
    )
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m2d-root", type=Path, default=DEFAULT_M2D_ROOT)
    parser.add_argument(
        "--expected-checkpoint-sha256",
        default=DEFAULT_CHECKPOINT_SHA256,
    )
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision",
        default="amp_fp16",
        choices=["fp32", "amp_fp16"],
    )
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, _audit = audit_verified_snapshot(
        dataset_root=args.dataset_root,
        expected_revision=args.expected_revision,
    )
    encoder = M2DEncoderAdapter(
        checkpoint=args.checkpoint,
        m2d_root=args.m2d_root,
        device=args.device,
        precision=args.precision,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    result = run_contact_window_augmentation_evaluation(
        args.source_bundle,
        args.output_root,
        snapshot,
        encoder,
        ContactWindowAugmentationConfig(
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        ),
    )
    metrics = pd.read_csv(result.path("metrics"))
    event = metrics[metrics["condition"] == "event"]
    print(
        event[
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
