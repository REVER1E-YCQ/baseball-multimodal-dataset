from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from run_eat_audio_fusion import (
    LABEL_TO_ID,
    best_threshold,
    calculate_metrics,
    candidate_features,
    make_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-csv", type=Path, required=True)
    parser.add_argument("--traditional-cache", type=Path, required=True)
    parser.add_argument("--eat-cache", type=Path, required=True)
    parser.add_argument("--eat-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split = pd.read_csv(args.split_csv)
    labels = split["label"].map(LABEL_TO_ID).to_numpy(dtype=np.int64)
    train_indices = np.flatnonzero(split["split"].to_numpy() == "train")
    test_indices = np.flatnonzero(split["split"].to_numpy() == "test")
    inner_train, validation = train_test_split(
        train_indices,
        test_size=0.15,
        random_state=42,
        stratify=labels[train_indices],
    )

    traditional = np.load(args.traditional_cache)["traditional"].astype(np.float32)
    eat = np.load(args.eat_cache)["eat_features"]
    eat_config = json.loads(args.eat_selection.read_text(encoding="utf-8"))
    eat_values = candidate_features(eat, traditional)[eat_config["feature_name"]]

    eat_model = make_model(eat_config["model_type"], float(eat_config["parameter_C"]))
    traditional_model = make_model("rbf_svm", 100.0)
    eat_model.fit(eat_values[inner_train], labels[inner_train])
    traditional_model.fit(traditional[inner_train], labels[inner_train])
    eat_validation = eat_model.predict_proba(eat_values[validation])[:, 1]
    traditional_validation = traditional_model.predict_proba(traditional[validation])[:, 1]

    rows = []
    best = None
    for eat_weight in np.linspace(0.0, 1.0, 41):
        scores = eat_weight * eat_validation + (1.0 - eat_weight) * traditional_validation
        threshold, macro_f1, balanced = best_threshold(labels[validation], scores, 0.5)
        row = {
            "eat_weight": float(eat_weight),
            "traditional_weight": float(1.0 - eat_weight),
            "threshold": float(threshold),
            "validation_macro_f1": float(macro_f1),
            "validation_balanced_accuracy": float(balanced),
            "validation_roc_auc": float(roc_auc_score(labels[validation], scores)),
        }
        rows.append(row)
        ranking = (row["validation_macro_f1"], row["validation_balanced_accuracy"], row["validation_roc_auc"])
        if best is None or ranking > best[0]:
            best = (ranking, row)
    assert best is not None
    selected = best[1]
    pd.DataFrame(rows).to_csv(args.output_dir / "late_fusion_validation.csv", index=False, encoding="utf-8-sig")

    eat_model.fit(eat_values[train_indices], labels[train_indices])
    traditional_model.fit(traditional[train_indices], labels[train_indices])
    eat_test = eat_model.predict_proba(eat_values[test_indices])[:, 1]
    traditional_test = traditional_model.predict_proba(traditional[test_indices])[:, 1]
    test_scores = selected["eat_weight"] * eat_test + selected["traditional_weight"] * traditional_test
    test_prediction = (test_scores >= selected["threshold"]).astype(int)
    metrics = calculate_metrics(labels[test_indices], test_prediction, test_scores)
    metrics.update(
        {
            "selected_fusion": selected,
            "eat_configuration": eat_config,
            "traditional_configuration": {"model_type": "rbf_svm", "C": 100.0},
            "train_samples": int(len(train_indices)),
            "validation_samples": int(len(validation)),
            "test_samples": int(len(test_indices)),
            "selection_rule": "Weights and threshold selected only on the fixed training split's validation subset.",
        }
    )
    (args.output_dir / "late_fusion_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    joblib.dump(
        {
            "eat_model": eat_model,
            "traditional_model": traditional_model,
            "eat_weight": selected["eat_weight"],
            "traditional_weight": selected["traditional_weight"],
            "threshold": selected["threshold"],
            "eat_feature_name": eat_config["feature_name"],
        },
        args.output_dir / "eat_traditional_late_fusion.joblib",
    )
    predictions = split.iloc[test_indices][["sample_id", "label", "source_id", "audio_path"]].copy()
    predictions["eat_probability"] = eat_test
    predictions["traditional_probability"] = traditional_test
    predictions["fused_fly_ball_probability"] = test_scores
    predictions["predicted_label"] = np.where(test_prediction == 1, "fly_ball", "ground_ball")
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions.to_csv(args.output_dir / "late_fusion_test_predictions.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
