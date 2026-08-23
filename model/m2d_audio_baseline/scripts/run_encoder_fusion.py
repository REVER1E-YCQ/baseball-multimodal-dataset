from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .benchmark_artifact_roles import (
    BEATS_ENCODER_NAME as BEATS_ENCODER,
    M2D_ENCODER_NAME as M2D_ENCODER,
    VERIFIED_DATASET_REVISION as LOCKED_DATASET_REVISION,
    ATTENTION_CONTROL_TRANSFORM_POLICY,
    BenchmarkArtifactRole,
    resolve_benchmark_bundle,
    short_contact_artifact_role,
)
from .encoder_fusion import (
    evaluate_fusion,
    load_source_table,
    verify_fold_consistency,
)

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/encoder_fusion"
M2D_POOLING_ROOT = REPO_ROOT / "outputs/m2d_pooling_ablation"
COMMON_200MS_ROOT = REPO_ROOT / "outputs/common_200ms_benchmark"


def fusion_source_role(
    *,
    name: str,
    encoder_name: str,
    pooling: str,
    seed: int,
    dataset_revision: str = LOCKED_DATASET_REVISION,
) -> BenchmarkArtifactRole:
    """Describe one locked 200 ms source used by encoder fusion."""

    return short_contact_artifact_role(
        name=name,
        encoder_name=encoder_name,
        dataset_revision=dataset_revision,
        pooling=pooling,
        fold_seed=seed,
        attention_control_transform_policy=(
            ATTENTION_CONTROL_TRANSFORM_POLICY
            if pooling == "attention"
            else None
        ),
    )


def find_bundle(
    root: Path, role: BenchmarkArtifactRole
) -> tuple[Path, str]:
    """Resolve the unique fusion source matching a scientific role."""

    bundle = resolve_benchmark_bundle(root, role)
    protocol = json.loads(
        (bundle / "protocol.json").read_text(encoding="utf-8")
    )
    return bundle, str(protocol["artifact_id"])


def build_fusion_comparison(
    table: pd.DataFrame,
) -> dict[str, object]:
    """Summarise the fusion table and decide the direction.

    ``table`` has name / balanced_accuracy columns with the sources and
    the concatenation row; the concatenation row name is the source names
    joined with '+'.
    """
    if table.empty:
        raise ValueError("Empty fusion comparison table")
    event = table[table["condition"].eq("event_selected_event")]
    if event.empty:
        raise ValueError("Fusion table has no event rows")
    names = list(event["name"])
    combined = event[event["name"].str.contains("+", regex=False)]
    if combined.empty:
        raise ValueError("Fusion table is missing the concatenation row")
    singles = event[~event["name"].isin([combined.iloc[0]["name"]])]
    best_single = float(singles["balanced_accuracy"].max())
    fusion_ba = float(combined.iloc[0]["balanced_accuracy"])
    gain = fusion_ba - best_single
    if gain > 0.005:
        direction = "positive"
    elif gain < -0.005:
        direction = "negative"
    else:
        direction = "neutral"
    return {
        "fusion_balanced_accuracy": fusion_ba,
        "best_single_balanced_accuracy": best_single,
        "fusion_gain_vs_best_single": gain,
        "direction": direction,
        "conclusion": (
            "fusion_open"
            if direction == "positive"
            else "fusion_closed"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the M2D attention + BEATs mean fusion "
        "on the locked 200 ms bundles."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--m2d-root", type=Path, default=M2D_POOLING_ROOT)
    parser.add_argument("--beats-root", type=Path, default=COMMON_200MS_ROOT)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--c-grid", type=float, nargs="+", default=(0.001, 0.01, 0.1))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    m2d_root, m2d_id = find_bundle(
        args.m2d_root,
        fusion_source_role(
            name="m2d_attention_fusion_source",
            encoder_name=M2D_ENCODER,
            pooling="attention",
            seed=args.seed,
        ),
    )
    beats_root, beats_id = find_bundle(
        args.beats_root,
        fusion_source_role(
            name="beats_mean_fusion_source",
            encoder_name=BEATS_ENCODER,
            pooling="valid_final_layer_token_mean",
            seed=args.seed,
        ),
    )
    m2d_features = sorted(
        (m2d_root / "features").glob("*_tokens.csv")
    )
    beats_features = sorted((beats_root / "features").glob("*.csv"))
    if len(m2d_features) != 1 or len(beats_features) != 1:
        raise ValueError(
            f"Expected one feature table per bundle; got "
            f"{len(m2d_features)} M2D and {len(beats_features)} BEATs"
        )
    m2d_table = load_source_table(m2d_features[0], True)
    beats_table = load_source_table(beats_features[0], False)

    m2d_folds = pd.read_csv(m2d_root / "fold_assignments.csv")
    beats_folds = pd.read_csv(beats_root / "fold_assignments.csv")
    verify_fold_consistency(
        m2d_folds, beats_folds, name_a="m2d", name_b="beats"
    )

    result = evaluate_fusion(
        ("m2d_attention", "beats_mean"),
        (m2d_table, beats_table),
        (True, False),
        m2d_folds,
        c_grid=tuple(args.c_grid),
        seed=args.seed,
    )
    table = result["table"]
    summary = build_fusion_comparison(table)
    summary.update(
        {
            "source_bundles": {
                "m2d_attention": m2d_id,
                "beats_mean": beats_id,
            },
            "sample_set": result["summary"]["sample_set"],
            "event_eligible_samples": result["summary"][
                "event_eligible_samples"
            ],
            "c_grid": list(args.c_grid),
            "seed": args.seed,
        }
    )
    digest = hashlib.sha256(
        json.dumps(summary, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    bundle_root = args.output_root / digest
    bundle_root.mkdir(parents=True, exist_ok=True)
    table.to_csv(bundle_root / "fusion_comparison.csv", index=False)
    (bundle_root / "fusion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print(
        f"gain vs best single: {summary['fusion_gain_vs_best_single']:+.4f} "
        f"({summary['direction']}) -> {summary['conclusion']}"
    )
    print(f"Fusion bundle: {bundle_root}")


if __name__ == "__main__":
    main()
