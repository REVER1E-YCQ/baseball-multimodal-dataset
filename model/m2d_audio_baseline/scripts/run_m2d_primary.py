from __future__ import annotations

import argparse
from pathlib import Path

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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/m2d_primary_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked primary 200 ms M2D benchmark."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--m2d-root", type=Path, default=DEFAULT_M2D_ROOT)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256
    )
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", default="amp_fp16", choices=["fp32", "amp_fp16"])
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--c-grid", type=float, nargs="+", default=[0.001, 0.01, 0.1]
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="include strict-pre and transient-removal negative controls",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, _ = audit_verified_snapshot(
        dataset_root=args.dataset_root,
        expected_revision=args.expected_revision,
    )
    adapter = M2DEncoderAdapter(
        checkpoint=args.checkpoint,
        m2d_root=args.m2d_root,
        device=args.device,
        precision=args.precision,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
    )
    protocol = BenchmarkProtocol(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        c_grid=tuple(args.c_grid),
        include_controls=args.controls,
    )
    bundle = run_short_contact_benchmark(
        protocol=protocol,
        snapshot=snapshot,
        encoder_adapters=(adapter,),
        output_dir=args.output_root,
    )
    import pandas as pd

    summary = pd.read_csv(bundle.root / "metrics.csv")
    print(f"Artifact bundle: {bundle.artifact_id}")
    print(bundle.root)
    print(summary.to_string(index=False))
    print(f"Exclusions: {len(pd.read_csv(bundle.root / 'exclusions.csv'))}")


if __name__ == "__main__":
    main()
