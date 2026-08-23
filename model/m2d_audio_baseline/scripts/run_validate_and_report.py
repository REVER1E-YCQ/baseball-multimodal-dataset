from __future__ import annotations

import argparse
import json
from pathlib import Path

from .validate_and_report import (
    generate_reports,
    validate_complete_run,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_COMMON_ROOT = REPO_ROOT / "outputs/common_200ms_benchmark"
DEFAULT_SECONDARY_ROOT = (
    REPO_ROOT / "outputs/secondary_evidence_benchmark/evidence"
)
DEFAULT_SENSITIVITY_ROOT = REPO_ROOT / "outputs/m2d_pooling_ablation"
DEFAULT_SNAPSHOT_AUDIT = (
    REPO_ROOT / "outputs/verified_snapshot_audit/audit_summary.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/validated_run"
DEFAULT_LAYER_SCAN_ROOT = REPO_ROOT / "outputs/m2d_layer_scan"
DEFAULT_FULL_AUDIO_ROOT = REPO_ROOT / "outputs/full_audio_conditions"
DEFAULT_ALIGNMENT_ROOT = REPO_ROOT / "outputs/m2d_alignment_sensitivity"
DEFAULT_FINETUNE_ROOT = REPO_ROOT / "outputs/m2d_finetune_pilot"
DEFAULT_FINETUNE_PRIMARY_ROOT = REPO_ROOT / "outputs/m2d_primary_benchmark"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the complete benchmark run and write reports."
    )
    parser.add_argument("--common-root", type=Path, default=DEFAULT_COMMON_ROOT)
    parser.add_argument("--secondary-root", type=Path, default=DEFAULT_SECONDARY_ROOT)
    parser.add_argument("--sensitivity-root", type=Path, default=DEFAULT_SENSITIVITY_ROOT)
    parser.add_argument("--snapshot-audit", type=Path, default=DEFAULT_SNAPSHOT_AUDIT)
    parser.add_argument("--layer-scan-root", type=Path, default=DEFAULT_LAYER_SCAN_ROOT)
    parser.add_argument("--full-audio-root", type=Path, default=DEFAULT_FULL_AUDIO_ROOT)
    parser.add_argument("--alignment-root", type=Path, default=DEFAULT_ALIGNMENT_ROOT)
    parser.add_argument("--finetune-root", type=Path, default=DEFAULT_FINETUNE_ROOT)
    parser.add_argument(
        "--finetune-primary-root",
        type=Path,
        default=DEFAULT_FINETUNE_PRIMARY_ROOT,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statistical_dir = args.common_root / "statistical_evidence"
    report = validate_complete_run(
        common_root=args.common_root,
        secondary_evidence_dir=args.secondary_root,
        statistical_evidence_dir=statistical_dir,
        sensitivity_root=args.sensitivity_root,
        output_root=args.output_root,
    )
    print(f"Validation status: {report['status']}")
    if report["failures"]:
        for failure in report["failures"]:
            print(f"  FAIL {failure}")
    paths = generate_reports(
        common_root=args.common_root,
        statistical_evidence_dir=statistical_dir,
        secondary_evidence_dir=args.secondary_root,
        sensitivity_root=args.sensitivity_root,
        snapshot_audit_path=args.snapshot_audit,
        output_root=args.output_root,
        layer_scan_dir=args.layer_scan_root,
        full_audio_conditions_dir=args.full_audio_root,
        alignment_sensitivity_dir=args.alignment_root,
        finetune_pilot_dir=args.finetune_root,
        finetune_primary_root=args.finetune_primary_root,
    )
    print(f"Report: {paths['report']}")
    print(f"Summary: {paths['summary']}")


if __name__ == "__main__":
    main()
