#!/usr/bin/env python3
"""Source-grouped audio-only comparison across contact and context windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.ndimage import zoom
from scipy.signal import stft
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from audio_baseline import (
    mel_filterbank,
    read_manifest,
    resample_audio,
    to_float_mono,
)
from temporal_ablation_experiment import source_groups


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "results"
CACHE_PATH = RESULT_DIR / "audio_context_features.npz"
WINDOWS = {
    "contact_0p5s": (-0.25, 0.25),
    "contact_1p0s": (-0.50, 0.50),
    "early_context_2p0s": (-0.25, 1.75),
    "extended_context_4p0s": (-0.25, 3.75),
    "full_clip": None,
}


def extract_range(audio: np.ndarray, rate: int, start: float, end: float) -> np.ndarray:
    length = max(1, int(round((end - start) * rate)))
    output = np.zeros(length, dtype=np.float32)
    source_start = max(0, int(round(start * rate)))
    source_end = min(len(audio), int(round(end * rate)))
    if source_end > source_start:
        target_start = source_start - int(round(start * rate))
        output[target_start : target_start + source_end - source_start] = audio[source_start:source_end]
    return output


def normalize(samples: np.ndarray) -> np.ndarray:
    samples = samples.astype(np.float32) - float(np.mean(samples))
    scale = float(np.sqrt(np.mean(np.square(samples))) + 1e-7)
    return np.clip(samples / scale, -12.0, 12.0)


def logmel(samples: np.ndarray, rate: int = 16000) -> np.ndarray:
    _frequency, _time, spectrum = stft(
        samples,
        fs=rate,
        nperseg=400,
        noverlap=240,
        nfft=512,
        boundary=None,
        padded=False,
    )
    power = np.square(np.abs(spectrum)).astype(np.float32)
    values = mel_filterbank(rate, 512, 64) @ power
    values = np.log1p(10.0 * values)
    if values.shape[1] != 96:
        values = zoom(values, (1.0, 96 / values.shape[1]), order=1)
    return values[:, :96].astype(np.float32)


def feature_views(values: np.ndarray) -> dict[str, np.ndarray]:
    frequency_summary = np.concatenate(
        [
            values.mean(axis=2),
            values.std(axis=2),
            values.max(axis=2),
            np.quantile(values, 0.9, axis=2),
        ],
        axis=1,
    )
    temporal = values.mean(axis=1)
    temporal_summary = np.concatenate(
        [temporal, np.diff(temporal, axis=1, prepend=temporal[:, :1])], axis=1
    )
    return {
        "summary": np.concatenate([frequency_summary, temporal_summary], axis=1).astype(
            np.float32
        ),
        "map": values.reshape(len(values), -1),
    }


def load_or_extract(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    paths = np.asarray([row["dataset_path"] for row in rows])
    if CACHE_PATH.exists():
        cache = np.load(CACHE_PATH)
        if np.array_equal(cache["dataset_paths"], paths):
            return {name: cache[name] for name in cache.files if name != "dataset_paths"}
    output = {name: [] for name in WINDOWS}
    for index, row in enumerate(rows, start=1):
        rate, raw = wavfile.read(ROOT / row["dataset_path"] / "audio.wav")
        audio = resample_audio(to_float_mono(raw), int(rate), 16000)
        center = (float(row["final_event_start"]) + float(row["final_event_end"])) / 2
        for name, offsets in WINDOWS.items():
            if offsets is None:
                window = audio
            else:
                window = extract_range(audio, 16000, center + offsets[0], center + offsets[1])
            output[name].append(logmel(normalize(window)))
        if index % 100 == 0:
            print(f"audio_context_features={index}/{len(rows)}", flush=True)
    arrays = {name: np.stack(values) for name, values in output.items()}
    np.savez_compressed(CACHE_PATH, dataset_paths=paths, **arrays)
    return arrays


def score(y: np.ndarray, prediction: np.ndarray, decision: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(y, decision)),
    }


def candidates(model_kind: str):
    if model_kind == "rbf_svm":
        for c_value in (0.3, 1.0, 3.0):
            for gamma in ("scale", 0.001):
                yield (
                    {"C": c_value, "gamma": gamma},
                    make_pipeline(
                        StandardScaler(),
                        SVC(
                            C=c_value,
                            gamma=gamma,
                            class_weight="balanced",
                            probability=False,
                            random_state=0,
                        ),
                    ),
                )
    elif model_kind == "extra_trees":
        for min_leaf in (1, 4):
            for max_features in ("sqrt", 0.3):
                yield (
                    {"min_samples_leaf": min_leaf, "max_features": max_features},
                    ExtraTreesClassifier(
                        n_estimators=250,
                        min_samples_leaf=min_leaf,
                        max_features=max_features,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=0,
                    ),
                )
    else:
        raise ValueError(model_kind)


def fit_selected(
    model_kind: str,
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
):
    ranked = []
    for params, model in candidates(model_kind):
        model.fit(x[train], y[train])
        prediction = model.predict(x[val])
        decision = (
            model.decision_function(x[val])
            if hasattr(model, "decision_function")
            else model.predict_proba(x[val])[:, 1]
        )
        metrics = score(y[val], prediction, decision)
        ranked.append((metrics["macro_f1"], metrics["roc_auc"], params))
    return max(ranked, key=lambda item: item[:2])[-1]


def build_model(model_kind: str, params: dict):
    if model_kind == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=params["C"],
                gamma=params["gamma"],
                class_weight="balanced",
                probability=False,
                random_state=0,
            ),
        )
    return ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=params["min_samples_leaf"],
        max_features=params["max_features"],
        class_weight="balanced",
        n_jobs=-1,
        random_state=0,
    )


def evaluate(
    name: str,
    model_kind: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    predictions = np.full(len(y), -1, dtype=int)
    decisions = np.full(len(y), np.nan)
    fold_reports = []
    indices = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260805)
    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(indices, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20261100 + fold)
        train_rel, val_rel = next(
            inner.split(outer_train_idx, y[outer_train_idx], groups[outer_train_idx])
        )
        train = np.zeros(len(y), dtype=bool)
        val = np.zeros(len(y), dtype=bool)
        outer_train = np.zeros(len(y), dtype=bool)
        test = np.zeros(len(y), dtype=bool)
        train[outer_train_idx[train_rel]] = True
        val[outer_train_idx[val_rel]] = True
        outer_train[outer_train_idx] = True
        test[test_idx] = True
        params = fit_selected(model_kind, x, y, train, val)
        model = build_model(model_kind, params)
        model.fit(x[outer_train], y[outer_train])
        predictions[test] = model.predict(x[test])
        decisions[test] = (
            model.decision_function(x[test])
            if hasattr(model, "decision_function")
            else model.predict_proba(x[test])[:, 1]
        )
        fold_reports.append(
            {
                "fold": fold,
                "params": params,
                **score(y[test], predictions[test], decisions[test]),
            }
        )
    return {
        "feature": name,
        "model": model_kind,
        **score(y, predictions, decisions),
        "fold_balanced_accuracy_mean": float(
            np.mean([row["balanced_accuracy"] for row in fold_reports])
        ),
        "fold_balanced_accuracy_std": float(
            np.std([row["balanced_accuracy"] for row in fold_reports])
        ),
        "folds": fold_reports,
    }


def main() -> None:
    rows = read_manifest(
        ROOT,
        ROOT / "reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv",
    )
    y = np.asarray([1 if row["label"] == "fly_ball" else 0 for row in rows])
    groups = source_groups(rows)
    maps = load_or_extract(rows)
    results = []
    for window_name, values in maps.items():
        views = feature_views(values)
        for model_kind in ("rbf_svm", "extra_trees"):
            result = evaluate(
                f"{window_name}_{'map' if model_kind == 'rbf_svm' else 'summary'}",
                model_kind,
                views["map"] if model_kind == "rbf_svm" else views["summary"],
                y,
                groups,
            )
            results.append(result)
            print(
                json.dumps({key: value for key, value in result.items() if key != "folds"}),
                flush=True,
            )
    results.sort(key=lambda row: row["balanced_accuracy"], reverse=True)
    (RESULT_DIR / "audio_context_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("BEST", json.dumps({k: v for k, v in results[0].items() if k != "folds"}))


if __name__ == "__main__":
    main()
