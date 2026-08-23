from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .audit_verified_snapshot import (
    PINNED_REVISION,
    audit_verified_snapshot,
)
from .beats_encoder import BEATsEncoderAdapter
from .short_contact_benchmark import (
    BenchmarkProtocol,
    run_short_contact_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data/models/beats_iter3plus_as2m/BEATs_iter3_plus_AS2M.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
)
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "data/branch_datasets_20260804/baseball-multimodal-dataset"
)
DEFAULT_BEATS_ROOT = REPO_ROOT / "external/unilm/beats"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/beats_primary_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the locked primary 200 ms BEATs benchmark."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--beats-root", type=Path, default=DEFAULT_BEATS_ROOT)
    parser.add_argument(
        "--expected-checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256
    )
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--device", default="auto")
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
    adapter = BEATsEncoderAdapter(
        checkpoint=args.checkpoint,
        beats_root=args.beats_root,
        device=args.device,
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
    summary = pd.read_csv(bundle.root / "metrics.csv")
    print(f"Artifact bundle: {bundle.artifact_id}")
    print(bundle.root)
    print(summary.to_string(index=False))
    print(f"Exclusions: {len(pd.read_csv(bundle.root / 'exclusions.csv'))}")


if __name__ == "__main__":
    main()
