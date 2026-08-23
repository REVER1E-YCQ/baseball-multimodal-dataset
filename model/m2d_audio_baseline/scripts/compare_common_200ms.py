from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .short_contact_benchmark import (
    ArtifactBundle,
    CONTROL_CONDITIONS,
)

REQUIRED_CONTROL_CONDITIONS = tuple(CONTROL_CONDITIONS) + (
    "contact_specific_increment",
)


class CommonComparisonError(RuntimeError):
    """Raised when two bundles are not directly comparable."""


@dataclass(frozen=True)
class CommonComparison:
    output_root: Path
    summary: dict[str, object]
    common_metrics: pd.DataFrame


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise CommonComparisonError(message)


def _load(name: str, bundle: ArtifactBundle, artifact: str) -> pd.DataFrame:
    return pd.read_csv(bundle.path(artifact))


def validate_common_200ms(
    bundles: dict[str, ArtifactBundle],
    output_root: Path,
) -> CommonComparison:
    """Validate that bundles share samples, folds, and conditions; emit the paired table."""

    if len(bundles) < 2:
        raise CommonComparisonError("At least two encoder bundles are required")
    names = list(bundles)
    checks: dict[str, object] = {}

    protocols = {
        name: json.loads(bundle.path("protocol").read_text(encoding="utf-8"))
        for name, bundle in bundles.items()
    }
    first_protocol = protocols[names[0]]
    _check(
        first_protocol.get("controls", {}).get("enabled") is True,
        f"{names[0]} bundle did not run with controls enabled",
    )
    for name in names[1:]:
        _check(
            protocols[name].get("controls", {}).get("enabled") is True,
            f"{name} bundle did not run with controls enabled",
        )
        for key in ("seed", "outer_splits", "inner_splits"):
            _check(
                protocols[name]["fold_policy"].get(key)
                == first_protocol["fold_policy"].get(key),
                f"Fold policy {key} differs between {names[0]} and {name}",
            )
        _check(
            protocols[name]["classifier"].get("C_grid")
            == first_protocol["classifier"].get("C_grid"),
            f"Classifier C grid differs between {names[0]} and {name}",
        )
        _check(
            protocols[name]["dataset"].get("revision")
            == first_protocol["dataset"].get("revision"),
            f"Dataset revision differs between {names[0]} and {name}",
        )
    checks["protocols_compatible"] = True

    folds = {
        name: _load(name, bundle, "fold_assignments")
        .sort_values("uid")
        .reset_index(drop=True)
        for name, bundle in bundles.items()
    }
    first_folds = folds[names[0]]
    for name in names[1:]:
        _check(
            set(folds[name]["uid"]) == set(first_folds["uid"]),
            f"Fold membership differs between {names[0]} and {name}",
        )
        merged = first_folds[["uid", "outer_fold"]].merge(
            folds[name][["uid", "outer_fold"]],
            on="uid",
            how="inner",
            suffixes=("_a", "_b"),
        )
        _check(
            (merged["outer_fold_a"] == merged["outer_fold_b"]).all(),
            f"Fold assignments differ between {names[0]} and {name}",
        )
    checks["fold_assignments_identical"] = True

    windows = {
        name: _load(name, bundle, "window_manifest")
        for name, bundle in bundles.items()
    }
    first_windows = windows[names[0]]
    for name in names[1:]:
        _check(
            set(windows[name]["window_name"])
            == set(first_windows["window_name"]),
            f"Window conditions differ between {names[0]} and {name}",
        )
        for window_name in sorted(set(first_windows["window_name"])):
            _check(
                set(
                    windows[name]
                    .loc[windows[name]["window_name"].eq(window_name), "uid"]
                )
                == set(
                    first_windows.loc[
                        first_windows["window_name"].eq(window_name), "uid"
                    ]
                ),
                f"Window membership for {window_name} differs between "
                f"{names[0]} and {name}",
            )
    checks["window_membership_identical"] = True

    predictions = {
        name: _load(name, bundle, "oof_predictions")
        for name, bundle in bundles.items()
    }
    condition_sets = {
        name: set(frame["condition"]) for name, frame in predictions.items()
    }
    _check(
        all(
            condition_sets[name] == condition_sets[names[0]]
            for name in names[1:]
        ),
        f"Prediction condition sets differ: {condition_sets}",
    )
    _check(
        condition_sets[names[0]] == set(REQUIRED_CONTROL_CONDITIONS[:-1]),
        f"Bundles are missing control conditions: "
        f"{sorted(set(REQUIRED_CONTROL_CONDITIONS[:-1]) - condition_sets[names[0]])}",
    )
    for name in names:
        frame = predictions[name]
        _check(
            not frame[["encoder", "condition", "uid"]].duplicated().any(),
            f"{name} bundle has duplicate predictions",
        )
    first_predictions = predictions[names[0]]
    for name in names[1:]:
        for condition in condition_sets[names[0]]:
            left = first_predictions[
                first_predictions["condition"].eq(condition)
            ].set_index("uid")
            right = predictions[name][
                predictions[name]["condition"].eq(condition)
            ].set_index("uid")
            _check(
                set(left.index) == set(right.index),
                f"Prediction membership for {condition} differs between "
                f"{names[0]} and {name}",
            )
            _check(
                (left["y_true"] == right["y_true"]).all(),
                f"Labels for {condition} differ between {names[0]} and {name}",
            )
    checks["prediction_cardinalities_identical"] = True

    metrics = {
        name: _load(name, bundle, "metrics")
        for name, bundle in bundles.items()
    }
    for name in names:
        found = set(metrics[name]["condition"])
        _check(
            found == set(REQUIRED_CONTROL_CONDITIONS),
            f"{name} bundle metrics are missing conditions: "
            f"{sorted(set(REQUIRED_CONTROL_CONDITIONS) - found)}",
        )
    checks["metrics_conditions_complete"] = True

    common_rows: list[dict[str, object]] = []
    for condition in REQUIRED_CONTROL_CONDITIONS:
        row: dict[str, object] = {"condition": condition}
        values: dict[str, float] = {}
        for name in names:
            value = float(
                metrics[name]
                .loc[metrics[name]["condition"].eq(condition), "balanced_accuracy"]
                .iloc[0]
            )
            row[f"{name}_balanced_accuracy"] = value
            values[name] = value
        row[f"{names[0]}_minus_{names[1]}"] = values[names[0]] - values[names[1]]
        paired = int(
            metrics[names[0]]
            .loc[metrics[names[0]]["condition"].eq(condition), "eligible_samples"]
            .iloc[0]
        )
        row["n_paired_samples"] = paired
        common_rows.append(row)
    common_metrics = pd.DataFrame(common_rows)

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    common_metrics.to_csv(output_root / "common_metrics.csv", index=False)
    for name in names:
        predictions[name].to_csv(
            output_root / f"{name}_oof_predictions.csv", index=False
        )
    summary = {
        "encoders": names,
        "checks": checks,
        "n_common_samples": int(len(first_folds)),
        "n_paired_samples": int(
            metrics[names[0]]
            .loc[
                metrics[names[0]]["condition"].eq("event_selected_event"),
                "eligible_samples",
            ]
            .iloc[0]
        ),
        "common_conditions": list(REQUIRED_CONTROL_CONDITIONS),
    }
    (output_root / "validation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return CommonComparison(
        output_root=output_root,
        summary=summary,
        common_metrics=common_metrics,
    )
