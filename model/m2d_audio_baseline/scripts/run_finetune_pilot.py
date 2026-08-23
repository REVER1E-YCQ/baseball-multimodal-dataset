from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .benchmark_artifact_roles import (
    M2D_ENCODER_NAME as M2D_ENCODER,
    VERIFIED_DATASET_REVISION as LOCKED_DATASET_REVISION,
    ATTENTION_CONTROL_TRANSFORM_POLICY,
    BenchmarkArtifactRole,
    resolve_benchmark_bundle,
    short_contact_artifact_role,
)
from .finetune_m2d import (
    FinetuneConfig,
    FinetuneProvenance,
    run_finetune_pilot,
)
from .m2d_encoder import M2DEncoderAdapter

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f"
)
DEFAULT_M2D_ROOT = REPO_ROOT / "external/m2d"
DEFAULT_PRIMARY_ROOT = REPO_ROOT / "outputs/m2d_primary_benchmark"
DEFAULT_POOLING_ROOT = REPO_ROOT / "outputs/m2d_pooling_ablation"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/m2d_finetune_pilot"


@dataclass(frozen=True)
class FinetuneBenchmarkReferences:
    frozen_mean_bundle: Path
    attention_headline_bundle: Path
    frozen_mean_metric: pd.Series
    attention_headline_metric: pd.Series


def _role(
    *,
    name: str,
    pooling: str,
    controls_enabled: bool,
    seed: int,
    dataset_revision: str,
) -> BenchmarkArtifactRole:
    return short_contact_artifact_role(
        name=name,
        encoder_name=M2D_ENCODER,
        dataset_revision=dataset_revision,
        pooling=pooling,
        fold_seed=seed,
        controls_enabled=controls_enabled,
        attention_control_transform_policy=(
            ATTENTION_CONTROL_TRANSFORM_POLICY
            if pooling == "attention" and controls_enabled
            else None
        ),
    )


def _metric_row(bundle: Path, *, controls_enabled: bool) -> pd.Series:
    metrics = pd.read_csv(bundle / "metrics.csv")
    rows = metrics[metrics["encoder"].eq(M2D_ENCODER)]
    if controls_enabled:
        rows = rows[
            rows["condition"].eq("event_selected_event")
            & rows["window_ms"].eq(200)
        ]
        if "decision_rule" in rows.columns:
            rows = rows[rows["decision_rule"].eq("fixed_0.5")]
    if len(rows) != 1:
        raise ValueError(
            f"Expected one M2D event metric row in {bundle}, found {len(rows)}"
        )
    return rows.iloc[0]


def _fold_membership(bundle: Path) -> pd.DataFrame:
    folds = pd.read_csv(bundle / "fold_assignments.csv")
    required = ["uid", "label", "lineage_group_id", "outer_fold"]
    missing = set(required) - set(folds.columns)
    if missing:
        raise ValueError(f"Fold assignments in {bundle} miss {sorted(missing)}")
    return folds[required].sort_values("uid").reset_index(drop=True)


def resolve_finetune_benchmarks(
    *,
    primary_root: Path,
    pooling_root: Path,
    seed: int,
    dataset_revision: str = LOCKED_DATASET_REVISION,
) -> FinetuneBenchmarkReferences:
    """Resolve comparable frozen-mean and attention reference roles."""

    mean_bundle = resolve_benchmark_bundle(
        primary_root,
        _role(
            name="finetune_frozen_mean",
            pooling="valid_final_layer_token_mean",
            controls_enabled=False,
            seed=seed,
            dataset_revision=dataset_revision,
        ),
    )
    attention_bundle = resolve_benchmark_bundle(
        pooling_root,
        _role(
            name="finetune_attention_headline",
            pooling="attention",
            controls_enabled=True,
            seed=seed,
            dataset_revision=dataset_revision,
        ),
    )
    mean_protocol = json.loads(
        (mean_bundle / "protocol.json").read_text(encoding="utf-8")
    )
    attention_protocol = json.loads(
        (attention_bundle / "protocol.json").read_text(encoding="utf-8")
    )
    if mean_protocol["dataset"].get("snapshot_fingerprint") != (
        attention_protocol["dataset"].get("snapshot_fingerprint")
    ):
        raise ValueError("Fine-tune benchmark references use different snapshots")
    if not _fold_membership(mean_bundle).equals(
        _fold_membership(attention_bundle)
    ):
        raise ValueError(
            "Fine-tune benchmark references use incompatible fold roles"
        )
    mean_metric = _metric_row(mean_bundle, controls_enabled=False)
    event_windows = pd.read_csv(mean_bundle / "windows_manifest.csv")
    event_windows = event_windows[
        event_windows["window_name"].eq("event_200ms")
    ]
    if int(mean_metric["eligible_samples"]) != len(event_windows):
        raise ValueError(
            "Frozen mean metric and manifest use incompatible eligible-sample roles"
        )
    return FinetuneBenchmarkReferences(
        frozen_mean_bundle=mean_bundle,
        attention_headline_bundle=attention_bundle,
        frozen_mean_metric=mean_metric,
        attention_headline_metric=_metric_row(
            attention_bundle, controls_enabled=True
        ),
    )


def build_pilot_comparison(
    finetune_ba: float,
    finetune_eligible: int,
    frozen_mean_ba: float,
    frozen_mean_eligible: int,
    attention_headline_ba: float,
    attention_headline_eligible: int,
) -> dict[str, object]:
    """Build the pilot comparison summary and the direction decision."""
    if finetune_eligible != frozen_mean_eligible:
        raise ValueError(
            "Fine-tuned and frozen mean rows have incompatible "
            "eligible-sample roles"
        )
    gain = finetune_ba - frozen_mean_ba
    if gain > 0.005:
        direction = "positive"
        conclusion = "fine_tuning_open"
    elif gain < -0.005:
        direction = "negative"
        conclusion = "fine_tuning_closed"
    else:
        direction = "neutral"
        conclusion = "fine_tuning_closed"
    return {
        "fine_tuned_mean_balanced_accuracy": finetune_ba,
        "fine_tuned_eligible_samples": finetune_eligible,
        "frozen_mean_balanced_accuracy": frozen_mean_ba,
        "frozen_mean_eligible_samples": frozen_mean_eligible,
        "gain_vs_frozen_mean": gain,
        "attention_headline_reference_balanced_accuracy": (
            attention_headline_ba
        ),
        "attention_headline_eligible_samples": attention_headline_eligible,
        "attention_headline_note": (
            "reference row with the controls-enabled sample set; not the "
            "same sample set as the pilot rows"
            if attention_headline_eligible != finetune_eligible
            else "reference row on the same eligible-sample role"
        ),
        "direction": direction,
        "conclusion": conclusion,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the M2D LoRA fine-tuning pilot on the locked folds."
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--pooling-root", type=Path, default=DEFAULT_POOLING_ROOT)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--mode", default="unfreeze_top")
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    references = resolve_finetune_benchmarks(
        primary_root=Path(args.primary_root),
        pooling_root=Path(args.pooling_root),
        seed=args.seed,
    )
    mean_bundle = references.frozen_mean_bundle
    frozen_mean_row = references.frozen_mean_metric
    attention_row = references.attention_headline_metric

    windows = pd.read_csv(mean_bundle / "windows_manifest.csv")
    windows = windows[windows["window_name"].eq("event_200ms")]
    folds = pd.read_csv(mean_bundle / "fold_assignments.csv")

    adapter = M2DEncoderAdapter(
        checkpoint=DEFAULT_CHECKPOINT,
        m2d_root=DEFAULT_M2D_ROOT,
        device="auto",
        precision="amp_fp16",
        expected_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
    )

    def model_factory():
        model, device = adapter._load_model()
        return model, device

    provenance = FinetuneProvenance(
        backbone=adapter.provenance.name,
        upstream_revision=adapter.provenance.upstream_revision,
        checkpoint_sha256=adapter.provenance.checkpoint_sha256,
        mode=args.mode,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        unfreeze_layers=args.unfreeze_layers,
        lr=args.lr,
        head_lr=args.head_lr,
        max_epochs=args.epochs,
        inner_splits=args.inner_splits,
    )
    config = FinetuneConfig(
        mode=args.mode,
        unfreeze_layers=args.unfreeze_layers,
        lr=args.lr,
        head_lr=args.head_lr,
        max_epochs=args.epochs,
        batch_size=32,
        inner_splits=args.inner_splits,
        patience=args.patience,
    )
    result = run_finetune_pilot(
        model_factory=model_factory,
        token_dimension=adapter.provenance.token_dimension,
        provenance=provenance,
        config=config,
        windows=windows,
        manifest_root=mean_bundle,
        folds=folds,
        seed=args.seed,
        output_dir=args.output_root,
    )
    comparison = build_pilot_comparison(
        finetune_ba=result["event_balanced_accuracy"],
        finetune_eligible=len(result["oof_predictions"]),
        frozen_mean_ba=float(frozen_mean_row["balanced_accuracy"]),
        frozen_mean_eligible=int(frozen_mean_row["eligible_samples"]),
        attention_headline_ba=float(attention_row["balanced_accuracy"]),
        attention_headline_eligible=int(
            attention_row["eligible_samples"]
        ),
    )
    comparison["benchmark_references"] = {
        "frozen_mean_artifact_id": mean_bundle.name,
        "attention_headline_artifact_id": (
            references.attention_headline_bundle.name
        ),
        "fold_assignments_compatible": True,
    }
    trace = result["trace"]
    comparison["overfitting_signature"] = {
        "train_minus_inner_val_mean": float(
            trace["train_balanced_accuracy"].mean()
            - trace["inner_val_balanced_accuracy"].mean()
        ),
        "final_epochs_by_fold": [
            int(value)
            for value in trace.groupby("outer_fold")["epoch"].max().sort_index()
        ],
    }
    (Path(result["output_dir"]) / "pilot_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"fine-tuned BA: {comparison['fine_tuned_mean_balanced_accuracy']:.4f} "
        f"({comparison['fine_tuned_eligible_samples']} samples)"
    )
    print(
        f"frozen mean BA: {comparison['frozen_mean_balanced_accuracy']:.4f} "
        f"({comparison['frozen_mean_eligible_samples']})"
    )
    print(
        f"attention headline (reference): "
        f"{comparison['attention_headline_reference_balanced_accuracy']:.4f}"
    )
    print(
        f"gain vs frozen mean: {comparison['gain_vs_frozen_mean']:+.4f} "
        f"-> {comparison['conclusion']}"
    )
    print(f"overfitting: {comparison['overfitting_signature']}")
    print(f"Pilot bundle: {result['output_dir']}")


if __name__ == "__main__":
    main()
