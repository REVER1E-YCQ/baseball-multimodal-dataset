from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from .short_contact_benchmark import (
    ArtifactBundle,
    CONTROL_CONDITIONS,
)

MIN_GROUPS_FOR_SOURCE_TRANSFER = 20
MAX_SINGLETON_FRACTION = 0.5


@dataclass(frozen=True)
class StatisticalEvidence:
    output_root: Path
    summary: dict[str, object]

    def path(self, name: str) -> Path:
        return self.output_root / name


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def _group_bootstrap_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_codes: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> np.ndarray:
    n_groups = int(group_codes.max()) + 1
    group_rows = [
        np.flatnonzero(group_codes == code) for code in range(n_groups)
    ]
    statistics = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        chosen = np.concatenate(
            [
                group_rows[code]
                for code in rng.integers(0, n_groups, size=n_groups)
            ]
        )
        statistics[index] = balanced_accuracy_score(
            y_true[chosen], y_pred[chosen]
        )
    return statistics


def _condition_frame(
    predictions: pd.DataFrame, condition: str
) -> pd.DataFrame:
    return predictions[predictions["condition"].eq(condition)]


def _group_uncertainty_rows(
    predictions: pd.DataFrame,
    group_codes: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for condition in CONTROL_CONDITIONS:
        frame = _condition_frame(predictions, condition)
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        observed = float(balanced_accuracy_score(y_true, y_pred))
        statistics = _group_bootstrap_statistics(
            y_true, y_pred, group_codes, rng, n_bootstrap
        )
        low, high = _percentile_interval(statistics)
        rows.append(
            {
                "condition": condition,
                "observed_balanced_accuracy": observed,
                "ci_low": low,
                "ci_high": high,
                "method": "group_resample",
                "unit": "lineage_group_id",
                "n_bootstrap": n_bootstrap,
                "n_groups": int(group_codes.max()) + 1,
                "n_samples": int(len(frame)),
            }
        )
    return rows


def _paired_bootstrap_interval(
    y_true: np.ndarray,
    first_scores: np.ndarray,
    second_scores: np.ndarray,
    group_codes: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    n_groups = int(group_codes.max()) + 1
    group_rows = [
        np.flatnonzero(group_codes == code) for code in range(n_groups)
    ]
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        chosen = np.concatenate(
            [
                group_rows[code]
                for code in rng.integers(0, n_groups, size=n_groups)
            ]
        )
        first = balanced_accuracy_score(y_true[chosen], first_scores[chosen])
        second = balanced_accuracy_score(y_true[chosen], second_scores[chosen])
        differences[index] = first - second
    low, high = _percentile_interval(differences)
    return float(differences.mean()), low, high


def _permute_labels_within_folds(
    y_true: np.ndarray,
    fold_array: np.ndarray,
    rng: np.random.Generator,
    n_permutations: int,
) -> list[np.ndarray]:
    """Stratified label permutation inside each locked outer fold.

    Class totals per fold are preserved and a mixed-label game keeps samples
    with different labels instead of collapsing the whole game to one label.
    """
    permutations: list[np.ndarray] = []
    for _ in range(n_permutations):
        permuted = y_true.copy()
        for fold in np.unique(fold_array):
            indices = np.flatnonzero(fold_array == fold)
            permuted[indices] = rng.permutation(y_true[indices])
        permutations.append(permuted)
    return permutations


def _permutation_summary(
    bundles: dict[str, ArtifactBundle],
    event_frame_by_encoder: dict[str, pd.DataFrame],
    n_permutations: int,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    names = list(bundles)
    rng = np.random.default_rng(seed + 900_001)
    observed: dict[str, float] = {}
    null_scores: dict[str, np.ndarray] = {}
    score_rows: list[dict[str, object]] = []
    for name in names:
        frame = event_frame_by_encoder[name]
        y_true = frame["y_true"].to_numpy(dtype=int)
        y_pred = frame["y_pred"].to_numpy(dtype=int)
        fold_array = frame["outer_fold"].to_numpy(dtype=int)
        observed[name] = float(balanced_accuracy_score(y_true, y_pred))
        permutations = _permute_labels_within_folds(
            y_true, fold_array, rng, n_permutations
        )
        scores = np.array(
            [
                balanced_accuracy_score(permuted, y_pred)
                for permuted in permutations
            ],
            dtype=np.float64,
        )
        null_scores[name] = scores
        for index, score in enumerate(scores):
            score_rows.append(
                {
                    "encoder": name,
                    "permutation": index,
                    "is_observed": False,
                    "balanced_accuracy": float(score),
                }
            )
        score_rows.append(
            {
                "encoder": name,
                "permutation": -1,
                "is_observed": True,
                "balanced_accuracy": observed[name],
            }
        )

    max_null = np.max(
        np.stack([null_scores[name] for name in names]), axis=0
    )
    summary_rows: list[dict[str, object]] = []
    for name in names:
        scores = null_scores[name]
        raw_p = float(
            (1 + int(np.sum(scores >= observed[name])))
            / (n_permutations + 1)
        )
        family_p = float(
            (1 + int(np.sum(max_null >= observed[name])))
            / (n_permutations + 1)
        )
        summary_rows.append(
            {
                "encoder": name,
                "observed_balanced_accuracy": observed[name],
                "null_mean": float(scores.mean()),
                "null_std": float(scores.std(ddof=0)),
                "uncorrected_p": raw_p,
                "max_stat_familywise_p": family_p,
                "n_permutations": n_permutations,
                "n_encoders_in_family": len(names),
                "permutation_unit": "sample_within_fold_stratified",
            }
        )
    return summary_rows, score_rows


def compute_statistical_evidence(
    bundles: dict[str, ArtifactBundle],
    output_root: Path,
    n_bootstrap: int = 1000,
    n_permutations: int = 999,
    seed: int = 20260805,
) -> StatisticalEvidence:
    """Compute group-aware intervals, paired increments, and permutation tests."""

    names = list(bundles)
    if not names:
        raise ValueError("At least one encoder bundle is required")
    predictions = {
        name: pd.read_csv(bundle.path("oof_predictions"))
        for name, bundle in bundles.items()
    }
    for name in names:
        frame = predictions[name]
        # Bundles with dual decision rules keep the locked 0.5-rule rows for
        # statistical evidence; calibrated rows are a decision-layer variant.
        if "decision_rule" in frame.columns:
            frame = frame[frame["decision_rule"].eq("fixed_0.5")]
        required = {
            "encoder",
            "condition",
            "uid",
            "lineage_group_id",
            "outer_fold",
            "y_true",
            "y_pred",
            "score_ground_ball",
        }
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} predictions are missing columns: {missing}")
        predictions[name] = frame.reset_index(drop=True)

    rng = np.random.default_rng(seed + 700_007)
    uncertainty_rows: list[dict[str, object]] = []
    paired_rows: list[dict[str, object]] = []
    group_stats: dict[str, dict[str, object]] = {}

    for name in names:
        frame = predictions[name]
        event = _condition_frame(frame, "event_selected_event")
        pre = _condition_frame(frame, "event_selected_pre")
        removed = _condition_frame(frame, "removed_selected_removed")
        group_ids = event["lineage_group_id"].astype("category")
        group_codes = group_ids.cat.codes.to_numpy(dtype=int)
        n_groups = int(group_codes.max()) + 1
        n_singletons = int(
            event.groupby("lineage_group_id")["uid"].nunique().eq(1).sum()
        )
        group_stats[name] = {
            "n_groups": n_groups,
            "n_singleton_groups": n_singletons,
            "n_samples": int(len(event)),
        }

        for condition, condition_frame in (
            (condition_name, _condition_frame(frame, condition_name))
            for condition_name in CONTROL_CONDITIONS
        ):
            y_true = condition_frame["y_true"].to_numpy(dtype=int)
            y_pred = condition_frame["y_pred"].to_numpy(dtype=int)
            observed = float(balanced_accuracy_score(y_true, y_pred))
            statistics = _group_bootstrap_statistics(
                y_true, y_pred, group_codes, rng, n_bootstrap
            )
            low, high = _percentile_interval(statistics)
            uncertainty_rows.append(
                {
                    "encoder": name,
                    "condition": condition,
                    "observed_balanced_accuracy": observed,
                    "ci_low": low,
                    "ci_high": high,
                    "method": "group_resample",
                    "unit": "lineage_group_id",
                    "n_bootstrap": n_bootstrap,
                    "n_groups": n_groups,
                    "n_samples": int(len(condition_frame)),
                    "seed": seed,
                }
            )

        # Paired event-minus-pre increment on identical samples and groups.
        aligned = event.set_index("uid").join(
            pre.set_index("uid"), lsuffix="_event", rsuffix="_pre"
        )
        if len(aligned) != len(event) or aligned["y_true_event"].isna().any():
            raise ValueError(f"{name} event/pre predictions are not fully paired")
        y_true = aligned["y_true_event"].to_numpy(dtype=int)
        event_pred = aligned["y_pred_event"].to_numpy(dtype=int)
        pre_pred = aligned["y_pred_pre"].to_numpy(dtype=int)
        event_pre_point = float(
            balanced_accuracy_score(y_true, event_pred)
            - balanced_accuracy_score(y_true, pre_pred)
        )
        _mean, low, high = _paired_bootstrap_interval(
            y_true,
            event_pred,
            pre_pred,
            group_codes,
            rng,
            n_bootstrap,
        )
        paired_rows.append(
            {
                "encoder": name,
                "interval_type": "event_minus_pre_increment",
                "point_estimate": event_pre_point,
                "ci_low": low,
                "ci_high": high,
                "n_groups": n_groups,
                "n_samples": len(aligned),
            }
        )

    # Paired encoder difference on the common event condition (requires at
    # least two encoders; single-encoder bundles skip this interval).
    first_event = _condition_frame(
        predictions[names[0]], "event_selected_event"
    )
    if len(names) >= 2:
        first, second = names[0], names[1]
        second_event = _condition_frame(
            predictions[second], "event_selected_event"
        )
        joined = first_event.set_index("uid").join(
            second_event.set_index("uid"), lsuffix="_a", rsuffix="_b"
        )
        if len(joined) != len(first_event) or joined["y_true_a"].isna().any():
            raise ValueError("Encoder event predictions are not fully paired")
        y_true = joined["y_true_a"].to_numpy(dtype=int)
        first_pred = joined["y_pred_a"].to_numpy(dtype=int)
        second_pred = joined["y_pred_b"].to_numpy(dtype=int)
        group_ids = first_event["lineage_group_id"].astype("category")
        group_codes = group_ids.cat.codes.to_numpy(dtype=int)
        point = float(
            balanced_accuracy_score(y_true, first_pred)
            - balanced_accuracy_score(y_true, second_pred)
        )
        _mean, low, high = _paired_bootstrap_interval(
            y_true,
            first_pred,
            second_pred,
            group_codes,
            rng,
            n_bootstrap,
        )
        paired_rows.append(
            {
                "encoder": f"{first}_minus_{second}",
                "interval_type": "encoder_event_difference",
                "point_estimate": point,
                "ci_low": low,
                "ci_high": high,
                "n_groups": int(group_codes.max()) + 1,
                "n_samples": len(joined),
            }
        )

    event_frames = {
        name: _condition_frame(predictions[name], "event_selected_event")
        for name in names
    }
    permutation_rows, score_rows = _permutation_summary(
        bundles, event_frames, n_permutations, seed
    )

    decisions: dict[str, object] = {}
    for row in permutation_rows:
        encoder = str(row["encoder"])
        increment_row = next(
            item
            for item in paired_rows
            if item["encoder"] == encoder
            and item["interval_type"] == "event_minus_pre_increment"
        )
        family_p = float(row["max_stat_familywise_p"])
        increment_low = float(increment_row["ci_low"])
        positive = family_p < 0.05 and increment_low > 0.0
        reasons: list[str] = []
        if family_p >= 0.05:
            reasons.append("corrected permutation evidence not significant")
        if increment_low <= 0.0:
            reasons.append("contact-specific increment interval not positive")
        decisions[encoder] = {
            "screening_positive": positive,
            "max_stat_familywise_p": family_p,
            "increment_ci_low": increment_low,
            "reasons": reasons,
        }

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(uncertainty_rows).to_csv(
        output_root / "group_uncertainty.csv", index=False
    )
    pd.DataFrame(paired_rows).to_csv(
        output_root / "paired_intervals.csv", index=False
    )
    pd.DataFrame(permutation_rows).to_csv(
        output_root / "permutation_summary.csv", index=False
    )
    pd.DataFrame(score_rows).to_csv(
        output_root / "permutation_scores.csv", index=False
    )
    summary = {
        "method": "group_resample_and_stratified_within_fold_permutation",
        "bootstrap_unit": "lineage_group_id",
        "permutation_unit": "sample_within_fold_stratified",
        "permutation_preserves": [
            "locked_outer_folds",
            "per_fold_class_totals",
            "mixed_label_games",
        ],
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
        "seed": seed,
        "encoders": names,
        "groups": group_stats,
        "source_transfer_conclusive": {
            name: (
                stats["n_groups"] >= MIN_GROUPS_FOR_SOURCE_TRANSFER
                and stats["n_singleton_groups"] / stats["n_groups"]
                <= MAX_SINGLETON_FRACTION
            )
            for name, stats in group_stats.items()
        },
        "screening_decisions": decisions,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return StatisticalEvidence(output_root=output_root, summary=summary)
