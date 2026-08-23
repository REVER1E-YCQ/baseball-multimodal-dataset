from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .audit_verified_snapshot import (
    PINNED_REVISION,
    audit_verified_snapshot,
)
from .beats_encoder import BEATsEncoderAdapter
from .m2d_encoder import M2DEncoderAdapter
from .secondary_evidence import compute_secondary_evidence
from .short_contact_benchmark import (
    BenchmarkProtocol,
    run_short_contact_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_DATASET_ROOT = (
    REPO_ROOT / "data/branch_datasets_20260804/baseball-multimodal-dataset"
)
DEFAULT_FIXED_SPLIT = (
    DEFAULT_DATASET_ROOT
    / "experiments/binary_classifier/results/dataset_split.csv"
)
DEFAULT_M2D_CHECKPOINT = (
    REPO_ROOT
    / "data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth"
)
DEFAULT_M2D_CHECKPOINT_SHA256 = (
    "63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f"
)
DEFAULT_M2D_ROOT = REPO_ROOT / "external/m2d"
DEFAULT_BEATS_CHECKPOINT = (
    REPO_ROOT / "data/models/beats_iter3plus_as2m/BEATs_iter3_plus_AS2M.pt"
)
DEFAULT_BEATS_CHECKPOINT_SHA256 = (
    "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
)
DEFAULT_BEATS_ROOT = REPO_ROOT / "external/unilm/beats"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/secondary_evidence_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fixed-split and RBF SVM development evidence."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fixed-split", type=Path, default=DEFAULT_FIXED_SPLIT)
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, _ = audit_verified_snapshot(
        dataset_root=args.dataset_root,
        expected_revision=args.expected_revision,
    )
    protocol = BenchmarkProtocol(seed=args.seed, include_controls=True)
    m2d_adapter = M2DEncoderAdapter(
        checkpoint=DEFAULT_M2D_CHECKPOINT,
        m2d_root=DEFAULT_M2D_ROOT,
        device="auto",
        precision="amp_fp16",
        expected_checkpoint_sha256=DEFAULT_M2D_CHECKPOINT_SHA256,
    )
    beats_adapter = BEATsEncoderAdapter(
        checkpoint=DEFAULT_BEATS_CHECKPOINT,
        beats_root=DEFAULT_BEATS_ROOT,
        device="auto",
        expected_checkpoint_sha256=DEFAULT_BEATS_CHECKPOINT_SHA256,
    )
    bundles = {
        "m2d": run_short_contact_benchmark(
            protocol, snapshot, (m2d_adapter,), args.output_root
        ),
        "beats": run_short_contact_benchmark(
            protocol, snapshot, (beats_adapter,), args.output_root
        ),
    }
    evidence = compute_secondary_evidence(
        bundles, args.fixed_split, args.output_root / "evidence", seed=args.seed
    )
    print(
        pd.read_csv(evidence.path("fixed_split_metrics.csv")).to_string(
            index=False
        )
    )
    print(
        pd.read_csv(evidence.path("rbf_metrics.csv")).to_string(index=False)
    )
    print(
        json.dumps(evidence.summary["fixed_split"], indent=2, sort_keys=True)
    )
    print(f"Outputs: {evidence.output_root}")


if __name__ == "__main__":
    main()
