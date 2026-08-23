from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/m2d_alignment_sensitivity"
SHIFTS_MS = (-100, -50, -25, 0, 25, 50, 100)


def build_alignment_curve(
    shift_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build the alignment-sensitivity curve from per-shift metric rows.

    ``shift_rows`` must have columns shift_ms, event_balanced_accuracy,
    event_roc_auc, eligible_samples. The 0 ms row is the reference.
    """
    required = {
        "shift_ms",
        "event_balanced_accuracy",
        "event_roc_auc",
        "eligible_samples",
    }
    missing = required.difference(shift_rows.columns)
    if missing:
        raise ValueError(f"Missing curve columns: {sorted(missing)}")
    if shift_rows["shift_ms"].duplicated().any():
        raise AssertionError("Duplicate shift_ms in the alignment curve")
    if not (shift_rows["shift_ms"] == 0).any():
        raise ValueError("The alignment curve needs the 0 ms reference row")
    curve = shift_rows.sort_values("shift_ms").reset_index(drop=True).copy()
    reference = float(
        curve.loc[curve["shift_ms"].eq(0), "event_balanced_accuracy"].iloc[0]
    )
    curve["delta_vs_0ms"] = curve["event_balanced_accuracy"] - reference
    positive = curve.loc[curve["shift_ms"].eq(50)]
    negative = curve.loc[curve["shift_ms"].eq(-50)]
    if positive.empty or negative.empty:
        raise ValueError(
            "The alignment curve needs both -50 ms and +50 ms points"
        )
    drop_50 = reference - min(
        float(positive.iloc[0]["event_balanced_accuracy"]),
        float(negative.iloc[0]["event_balanced_accuracy"]),
    )
    symmetry = abs(
        float(positive.iloc[0]["event_balanced_accuracy"])
        - float(negative.iloc[0]["event_balanced_accuracy"])
    )
    monotonic_away = True
    previous_ba = reference
    for shift_ms in (25, 50, 100):
        row = curve.loc[curve["shift_ms"].eq(shift_ms)]
        if row.empty:
            continue
        value = float(row.iloc[0]["event_balanced_accuracy"])
        if value > previous_ba + 1e-9:
            monotonic_away = False
        previous_ba = value
    previous_ba = reference
    for shift_ms in (-25, -50, -100):
        row = curve.loc[curve["shift_ms"].eq(shift_ms)]
        if row.empty:
            continue
        value = float(row.iloc[0]["event_balanced_accuracy"])
        if value > previous_ba + 1e-9:
            monotonic_away = False
        previous_ba = value
    if drop_50 >= 0.05:
        interpretation = "precise_alignment_dependence"
    elif drop_50 <= 0.02:
        interpretation = "coarse_content_dependence"
    else:
        interpretation = "moderate_alignment_dependence"
    summary = {
        "reference_0ms_balanced_accuracy": reference,
        "drop_at_50ms": drop_50,
        "symmetry_abs_diff_50ms": symmetry,
        "monotonic_away_from_0ms": bool(monotonic_away),
        "interpretation": interpretation,
        "interpretation_note": (
            "drop >= 0.05 at +/-50 ms: the model depends on precise "
            "peak-centring (deployment needs an automatic aligner); "
            "drop <= 0.02: the model uses coarse window content; the "
            "0 ms point was re-run in the same no-controls configuration "
            "as every other shift."
        ),
    }
    return curve, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M2D alignment-sensitivity scan."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--shifts", type=int, nargs="+", default=list(SHIFTS_MS))
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
    bundles: list[tuple[int, object]] = []
    for shift_ms in args.shifts:
        protocol = BenchmarkProtocol(
            seed=args.seed,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            c_grid=(0.001, 0.01, 0.1),
            pooling="attention",
            window_shift_ms=shift_ms,
        )
        bundle = run_short_contact_benchmark(
            protocol=protocol,
            snapshot=snapshot,
            encoder_adapters=(adapter,),
            output_dir=args.output_root,
        )
        bundles.append((shift_ms, bundle))
        print(f"shift {shift_ms:+d} ms -> {bundle.artifact_id}")

    encoder_base = adapter.provenance.name
    rows: list[dict[str, object]] = []
    for shift_ms, bundle in bundles:
        metrics = pd.read_csv(bundle.root / "metrics.csv")
        fixed = metrics[metrics["decision_rule"].eq("fixed_0.5")]
        event = fixed[
            fixed["encoder"].eq(encoder_base)
            & fixed["condition"].eq("event_selected_event")
        ]
        if event.empty:
            raise ValueError(f"Missing event metrics for shift {shift_ms}")
        row = event.iloc[0]
        rows.append(
            {
                "shift_ms": shift_ms,
                "event_balanced_accuracy": float(row["balanced_accuracy"]),
                "event_roc_auc": float(row["roc_auc"]),
                "eligible_samples": int(row["eligible_samples"]),
                "artifact_id": str(bundle.artifact_id),
            }
        )
    curve, summary = build_alignment_curve(pd.DataFrame(rows))
    last_bundle = bundles[-1][1]
    curve_path = last_bundle.root / "alignment_sensitivity.csv"
    summary_path = last_bundle.root / "alignment_sensitivity_summary.json"
    curve.to_csv(curve_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(curve.to_string(index=False))
    print(
        f"drop at +/-50ms: {summary['drop_at_50ms']:.4f} "
        f"({summary['interpretation']})"
    )
    print(f"Bundle: {last_bundle.artifact_id}")


if __name__ == "__main__":
    main()
