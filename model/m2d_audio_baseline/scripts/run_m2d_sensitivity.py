from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit_verified_snapshot import (
    PINNED_REVISION,
    audit_verified_snapshot,
)
from .m2d_encoder import M2DEncoderAdapter
from .short_contact_benchmark import (
    BenchmarkProtocol,
    run_short_contact_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f"
)
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "data/branch_datasets_20260804/baseball-multimodal-dataset"
)
DEFAULT_M2D_ROOT = REPO_ROOT / "external/m2d"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/m2d_sensitivity_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked M2D short-contact sensitivity conditions."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1]
    )
    parser.add_argument(
        "--durations", type=int, nargs="+", default=[50, 100, 200]
    )
    parser.add_argument(
        "--rms-normalized",
        action="store_true",
        help="run the RMS-normalized sensitivity instead of snapshot level",
    )
    parser.add_argument(
        "--legacy-pooling",
        action="store_true",
        help="run the legacy mean/std/max pooling sensitivity",
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="select the decision threshold inside each outer training set",
    )
    parser.add_argument(
        "--pooling",
        choices=[
            "valid_final_layer_token_mean",
            "mean_std",
            "mean_max",
            "legacy_mean_std_max",
            "energy_weighted",
            "attention",
        ],
        default=None,
        help="explicit pooling mode (overrides --legacy-pooling)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, _ = audit_verified_snapshot(
        dataset_root=args.dataset_root,
        expected_revision=args.expected_revision,
    )
    adapter = M2DEncoderAdapter(
        checkpoint=DEFAULT_CHECKPOINT,
        m2d_root=DEFAULT_M2D_ROOT,
        device="auto",
        precision="amp_fp16",
        expected_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    pooling = args.pooling
    if pooling is None:
        pooling = (
            "legacy_mean_std_max"
            if args.legacy_pooling
            else "valid_final_layer_token_mean"
        )
    protocol = BenchmarkProtocol(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        c_grid=tuple(args.c_grid),
        window_conditions=tuple(args.durations),
        include_controls=True,
        normalization=(
            "rms_normalized" if args.rms_normalized else "snapshot_level"
        ),
        pooling=pooling,
        calibrate_threshold=args.calibrate_threshold,
    )
    bundle = run_short_contact_benchmark(
        protocol=protocol,
        snapshot=snapshot,
        encoder_adapters=(adapter,),
        output_dir=args.output_root,
    )
    metrics = pd.read_csv(bundle.root / "metrics.csv")
    print(f"Artifact bundle: {bundle.artifact_id}")
    print(bundle.root)
    summary = metrics[
        metrics["condition"].isin(
            {"event_selected_event", "event_selected_pre", "contact_specific_increment"}
        )
    ][
        [
            "window_ms",
            "condition",
            "balanced_accuracy",
            "roc_auc",
            "eligible_samples",
        ]
    ]
    print(summary.to_string(index=False))
    print(f"Exclusions: {len(pd.read_csv(bundle.root / 'exclusions.csv'))}")


if __name__ == "__main__":
    main()
