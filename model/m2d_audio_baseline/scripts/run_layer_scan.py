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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/m2d_layer_scan"
ALL_LAYERS = tuple(range(12))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M2D per-layer attention scan."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--layers", type=int, nargs="+", default=list(ALL_LAYERS)
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
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
    protocol = BenchmarkProtocol(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        pooling="attention",
        include_controls=True,
        layers=tuple(args.layers),
    )
    bundle = run_short_contact_benchmark(
        protocol=protocol,
        snapshot=snapshot,
        encoder_adapters=(adapter,),
        output_dir=args.output_root,
    )
    metrics = pd.read_csv(bundle.root / "metrics.csv")
    fixed = metrics[metrics["decision_rule"].eq("fixed_0.5")]
    encoder_base = adapter.provenance.name
    scan_rows: list[dict[str, object]] = []
    for layer in args.layers:
        encoder_key = f"{encoder_base}__layer{layer}"
        event = fixed[
            fixed["encoder"].eq(encoder_key)
            & fixed["condition"].eq("event_selected_event")
        ]
        pre = fixed[
            fixed["encoder"].eq(encoder_key)
            & fixed["condition"].eq("event_selected_pre")
        ]
        increment = fixed[
            fixed["encoder"].eq(encoder_key)
            & fixed["condition"].eq("contact_specific_increment")
        ]
        if event.empty:
            raise ValueError(f"Missing event metrics for layer {layer}")
        scan_rows.append(
            {
                "layer": layer,
                "event_balanced_accuracy": float(
                    event.iloc[0]["balanced_accuracy"]
                ),
                "event_roc_auc": float(event.iloc[0]["roc_auc"]),
                "strict_pre_balanced_accuracy": (
                    float(pre.iloc[0]["balanced_accuracy"])
                    if not pre.empty
                    else float("nan")
                ),
                "contact_specific_increment": (
                    float(increment.iloc[0]["balanced_accuracy"])
                    if not increment.empty
                    else float("nan")
                ),
                "eligible_samples": int(event.iloc[0]["eligible_samples"]),
            }
        )
    scan = pd.DataFrame(scan_rows).sort_values("layer")
    peak = scan.loc[
        scan["event_balanced_accuracy"].idxmax()
    ]
    summary = {
        "encoders": sorted(set(fixed["encoder"])),
        "peak_layer": int(peak["layer"]),
        "peak_event_balanced_accuracy": float(
            peak["event_balanced_accuracy"]
        ),
        "peak_exploratory_note": (
            "The peak layer is selected across multiple layers and carries "
            "multiple-comparison bias; it is exploratory until confirmed "
            "across seeds."
        ),
    }
    scan_path = bundle.root / "layer_scan.csv"
    summary_path = bundle.root / "layer_scan_summary.json"
    scan.to_csv(scan_path, index=False)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(scan.to_string(index=False))
    print(f"Peak layer: {summary['peak_layer']} "
          f"({summary['peak_event_balanced_accuracy']:.4f})")
    print(f"Bundle: {bundle.artifact_id}")


if __name__ == "__main__":
    main()
