from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from .audit_verified_snapshot import (
    PINNED_REVISION,
    audit_verified_snapshot,
)
from .beats_encoder import BEATsEncoderAdapter
from .short_contact_benchmark import (
    BenchmarkProtocol,
    LABEL_TO_INT,
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/full_audio_conditions"

# The project lead's un-grouped numbers (accuracy, random split) for the
# side-by-side comparison; these are reproduction references, not claims.
LEAD_NUMBERS = {
    "0.5s_contact": 73,
    "4s_window": 78,
    "full_audio": 88,
    "full_audio_minus_contact": 55,
    "svm_full_audio": 76,
}

# Condition columns used by the comparison table. The 4 s lead condition
# is reproduced as post_contact_4000ms (a fixed segment starting at the
# peak): a centred 4 s window needs 4 s of pre-contact audio that most
# clips do not have, and the lead's 0.5 s -> 4 s -> full progression reads
# as segments starting at the contact point.
CONDITION_ORDER = (
    "event_200ms",
    "event_500ms",
    "post_contact_4000ms",
    "full_audio",
    "post_contact_1000ms",
    "pre_contact_1000ms",
)


def build_condition_table(
    metrics: pd.DataFrame, encoder_base: str
) -> pd.DataFrame:
    """Build the per-condition comparison table from a metrics frame.

    Centred event windows keep their strict-pre and increment columns;
    non-centred conditions have no negative-control chain by design, so
    those cells are NaN.
    """
    fixed = metrics[metrics["decision_rule"].eq("fixed_0.5")]
    fixed = fixed[fixed["encoder"].eq(encoder_base)]
    rows: list[dict[str, object]] = []
    baseline: float | None = None
    for condition in CONDITION_ORDER:
        if condition.startswith("event_"):
            window_ms = int(condition.split("_")[1][:-2])
            event = fixed[
                fixed["condition"].eq("event_selected_event")
                & fixed["window_ms"].eq(window_ms)
            ]
            pre = fixed[
                fixed["condition"].eq("event_selected_pre")
                & fixed["window_ms"].eq(window_ms)
            ]
            increment = fixed[
                fixed["condition"].eq("contact_specific_increment")
                & fixed["window_ms"].eq(window_ms)
            ]
        else:
            event = fixed[fixed["condition"].eq(condition)]
            pre = pd.DataFrame()
            increment = pd.DataFrame()
        if event.empty:
            raise ValueError(f"Missing metrics for condition {condition!r}")
        row = event.iloc[0]
        entry: dict[str, object] = {
            "condition": condition,
            "window_ms": int(row["window_ms"]),
            "event_balanced_accuracy": float(row["balanced_accuracy"]),
            "event_roc_auc": float(row["roc_auc"]),
            "eligible_samples": int(row["eligible_samples"]),
            "lineage_groups": int(row["lineage_groups"]),
        }
        entry["strict_pre_balanced_accuracy"] = (
            float(pre.iloc[0]["balanced_accuracy"]) if not pre.empty else float("nan")
        )
        entry["contact_specific_increment"] = (
            float(increment.iloc[0]["balanced_accuracy"])
            if not increment.empty
            else float("nan")
        )
        if condition == "event_500ms":
            baseline = float(row["balanced_accuracy"])
        entry["vs_500ms_delta"] = (
            float(row["balanced_accuracy"]) - baseline
            if baseline is not None and condition != "event_500ms"
            else float("nan")
        )
        rows.append(entry)
    table = pd.DataFrame(rows)
    if table["condition"].duplicated().any():
        raise AssertionError("Duplicate conditions in the comparison table")
    return table


def build_attribution_summary(table: pd.DataFrame) -> dict[str, object]:
    """Decompose the full-audio gain over the 0.5 s baseline by region.

    The conclusion is ``gain_lives_after_contact`` when the post-contact
    1 s gain clearly exceeds the pre-contact gain, which is the outcome-
    content-leakage signature.
    """
    def ba(condition: str) -> float:
        return float(
            table.loc[
                table["condition"].eq(condition),
                "event_balanced_accuracy",
            ].iloc[0]
        )

    baseline = ba("event_500ms")
    pre = ba("pre_contact_1000ms")
    post1 = ba("post_contact_1000ms")
    post4 = ba("post_contact_4000ms")
    full = ba("full_audio")
    pre_gain = pre - baseline
    post1_gain = post1 - baseline
    conclusion = (
        "gain_lives_after_contact"
        if post1_gain > pre_gain + 0.02
        else "no_clear_attribution"
    )
    return {
        "baseline_event_500ms": baseline,
        "pre_contact_1s": pre,
        "post_contact_1s": post1,
        "post_contact_4s": post4,
        "full_audio": full,
        "pre_contact_gain": pre_gain,
        "post_contact_1s_gain": post1_gain,
        "post_contact_4s_gain": post4 - baseline,
        "full_audio_gain": full - baseline,
        "gain_beyond_4s": full - post4,
        "conclusion": conclusion,
        "note": (
            "Gains beyond the contact window are attributed by region; "
            "post-contact content (commentary, crowd, runner sounds) can "
            "encode the outcome itself, so these gains are not contact "
            "acoustics."
        ),
    }


def build_lead_comparison(table: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "0.5s_contact": "event_500ms",
        "4s_window": "post_contact_4000ms",
        "full_audio": "full_audio",
    }
    rows = [
        {
            "lead_condition": lead,
            "lead_ungrouped_accuracy": LEAD_NUMBERS[lead],
            "our_condition": table.loc[
                table["condition"].eq(our), "condition"
            ].iloc[0],
            "our_grouped_balanced_accuracy": float(
                table.loc[table["condition"].eq(our), "event_balanced_accuracy"].iloc[0]
            ),
            "note": (
                "lead splits randomly and reports accuracy; ours use "
                "game-grouped 5-fold and balanced accuracy"
            ),
        }
        for lead, our in mapping.items()
    ]
    return pd.DataFrame(rows)


def compute_duration_shortcut(
    snapshot: object,
    outer_splits: int = 5,
    seed: int = 20260805,
) -> dict[str, object]:
    """Measure how well clip duration alone predicts the label.

    The snapshot's clips have near-separated durations across classes
    (ground clips are edited to ~6 s, fly clips are longer), so duration is
    a collection-process confound that any variable-length condition can
    exploit. Fixed-length segments (event/pre/post_contact) are immune.
    The threshold is fitted inside each training fold with the same
    game-grouped protocol as the benchmark, so the number is leak-safe.
    """
    durations: list[float] = []
    labels: list[int] = []
    groups: list[str] = []
    for sample in snapshot.samples:
        sample_rate, raw = wavfile.read(Path(sample.audio_path))
        durations.append(len(raw) / float(sample_rate))
        labels.append(LABEL_TO_INT[sample.label])
        groups.append(sample.lineage_group_id)
    duration_array = np.asarray(durations, dtype=float)
    label_array = np.asarray(labels, dtype=int)
    group_array = np.asarray(groups, dtype=object)
    auc = float(roc_auc_score(label_array, duration_array))
    if auc < 0.5:
        auc = 1.0 - auc
    splitter = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=seed,
    )
    predictions = np.full(len(label_array), -1, dtype=int)
    for train, test in splitter.split(
        np.zeros(len(label_array)), label_array, group_array
    ):
        scaler = StandardScaler().fit(duration_array[train].reshape(-1, 1))
        classifier = LogisticRegression(
            C=0.01,
            class_weight="balanced",
            solver="liblinear",
            max_iter=5_000,
            random_state=seed,
        )
        classifier.fit(
            scaler.transform(duration_array[train].reshape(-1, 1)),
            label_array[train],
        )
        predictions[test] = classifier.predict(
            scaler.transform(duration_array[test].reshape(-1, 1))
        )
    ba = float(
        balanced_accuracy_score(label_array, predictions)
    )
    return {
        "duration_shortcut_balanced_accuracy": ba,
        "duration_shortcut_roc_auc": auc,
        "duration_shortcut_protocol": (
            "game_grouped_outer_fold_logistic_on_duration"
        ),
        "note": (
            "Clip duration alone is a near-perfect label predictor in this "
            "snapshot (a collection-process confound); it only affects "
            "variable-length conditions such as full audio, never the "
            "fixed-length contact segments."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the lead's longer-window conditions under "
        "the grouped leak-safe protocol (frozen BEATs)."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-revision", default=PINNED_REVISION)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip the GPU extraction and reuse the existing bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, _ = audit_verified_snapshot(
        dataset_root=args.dataset_root,
        expected_revision=args.expected_revision,
    )
    adapter = BEATsEncoderAdapter(
        checkpoint=DEFAULT_CHECKPOINT,
        beats_root=DEFAULT_BEATS_ROOT,
        device="auto",
        expected_checkpoint_sha256=DEFAULT_CHECKPOINT_SHA256,
    )
    protocol = BenchmarkProtocol(
        seed=args.seed,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        pooling="valid_final_layer_token_mean",
        include_controls=True,
        window_conditions=(200, 500),
        non_centered_windows=(
            "full_audio",
            "post_contact_4000ms",
            "post_contact_1000ms",
            "pre_contact_1000ms",
        ),
    )
    bundle = run_short_contact_benchmark(
        protocol=protocol,
        snapshot=snapshot,
        encoder_adapters=(adapter,),
        output_dir=args.output_root,
    )
    metrics = pd.read_csv(bundle.root / "metrics.csv")
    encoder_base = adapter.provenance.name
    table = build_condition_table(metrics, encoder_base)
    peak = table.loc[table["event_balanced_accuracy"].idxmax()]
    lead_table = build_lead_comparison(table)
    attribution = build_attribution_summary(table)
    duration_shortcut = compute_duration_shortcut(
        snapshot,
        outer_splits=protocol.outer_splits,
        seed=protocol.seed,
    )
    attribution.update(duration_shortcut)
    table_path = bundle.root / "condition_scan.csv"
    lead_path = bundle.root / "lead_comparison.csv"
    attribution_path = bundle.root / "attribution_summary.json"
    table.to_csv(table_path, index=False)
    lead_table.to_csv(lead_path, index=False)
    attribution_path.write_text(
        json.dumps(
            attribution, ensure_ascii=False, indent=2, sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    full_row = table[table["condition"].eq("full_audio")].iloc[0]
    summary = {
        "encoders": sorted(set(metrics["encoder"])),
        "peak_condition": str(peak["condition"]),
        "peak_event_balanced_accuracy": float(
            peak["event_balanced_accuracy"]
        ),
        "full_audio_event_balanced_accuracy": float(
            full_row["event_balanced_accuracy"]
        ),
        "lead_full_audio_accuracy": LEAD_NUMBERS["full_audio"],
        "full_audio_shrinkage_vs_lead": (
            LEAD_NUMBERS["full_audio"] / 100.0
            - float(full_row["event_balanced_accuracy"])
        ),
        "exploratory_note": (
            "Non-centred conditions are deployment-irrelevant and carry no "
            "negative-control chain; the full-audio reproduction is "
            "reported to test the leak interpretation, not as a screening "
            "candidate."
        ),
    }
    summary_path = bundle.root / "condition_scan_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(table.to_string(index=False))
    print()
    print(lead_table.to_string(index=False))
    print(f"Peak condition: {summary['peak_condition']} "
          f"({summary['peak_event_balanced_accuracy']:.4f})")
    print(f"Bundle: {bundle.artifact_id}")


if __name__ == "__main__":
    main()
