#!/usr/bin/env python3
"""Pretrained raw-waveform features and validation-selected video fusion."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torchaudio

from audio_baseline import (
    fixed_window,
    read_manifest,
    resample_audio,
    to_float_mono,
)
from late_fusion_experiment import (
    RESULT_DIR,
    apply_fusion,
    load_rows,
    make_model,
    metric,
    tune_base,
    tune_fusion,
)
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[2]
RAW_CACHE = RESULT_DIR / "raw_contact_waveforms.npz"
FEATURE_CACHE = RESULT_DIR / "wav2vec_features.npz"


def load_or_create_waveforms(rows: list[dict[str, str]]) -> np.ndarray:
    paths = np.asarray([row["dataset_path"] for row in rows])
    if RAW_CACHE.exists():
        cache = np.load(RAW_CACHE)
        if np.array_equal(cache["dataset_paths"], paths):
            return cache["waveforms"]
    waveforms = []
    for index, row in enumerate(rows, start=1):
        rate, samples = wavfile.read(ROOT / row["dataset_path"] / "audio.wav")
        samples = resample_audio(to_float_mono(samples), int(rate), 16000)
        center = (float(row["final_event_start"]) + float(row["final_event_end"])) / 2
        window = fixed_window(samples, 16000, center, 1.0)
        window -= float(window.mean())
        peak = float(np.max(np.abs(window)) + 1e-8)
        waveforms.append(np.clip(window / peak, -1.0, 1.0).astype(np.float32))
        if index % 100 == 0:
            print(f"raw_waveforms={index}/{len(rows)}", flush=True)
    values = np.stack(waveforms)
    np.savez_compressed(RAW_CACHE, dataset_paths=paths, waveforms=values)
    return values


def pool_layer(values: torch.Tensor) -> np.ndarray:
    pooled = torch.cat(
        [values.mean(dim=1), values.std(dim=1), values.amax(dim=1)], dim=1
    )
    return pooled.cpu().numpy().astype(np.float32)


def load_or_extract_features(rows: list[dict[str, str]], waveforms: np.ndarray):
    paths = np.asarray([row["dataset_path"] for row in rows])
    if FEATURE_CACHE.exists():
        cache = np.load(FEATURE_CACHE)
        if np.array_equal(cache["dataset_paths"], paths):
            return {name: cache[name] for name in cache.files if name != "dataset_paths"}

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model().eval()
    layer_outputs = {"wav2vec_layer4": [], "wav2vec_layer8": [], "wav2vec_layer12": []}
    with torch.inference_mode():
        for start in range(0, len(waveforms), 16):
            batch = torch.from_numpy(waveforms[start : start + 16])
            features, _lengths = model.extract_features(batch)
            for name, layer_index in (
                ("wav2vec_layer4", 3),
                ("wav2vec_layer8", 7),
                ("wav2vec_layer12", 11),
            ):
                layer_outputs[name].append(pool_layer(features[layer_index]))
            print(f"wav2vec_features={min(start + 16, len(waveforms))}/{len(waveforms)}", flush=True)
    output = {name: np.concatenate(values) for name, values in layer_outputs.items()}
    np.savez_compressed(FEATURE_CACHE, dataset_paths=paths, **output)
    return output


def main() -> None:
    rows = read_manifest(
        ROOT,
        ROOT / "reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv",
    )
    split_rows = load_rows()
    if [row["dataset_path"] for row in rows] != [row["dataset_path"] for row in split_rows]:
        raise RuntimeError("Manifest and split row order differ")
    split = np.asarray([row["split"] for row in split_rows])
    y = np.asarray([int(row["target"]) for row in split_rows])
    train, val, test = split == "train", split == "val", split == "test"
    train_val = train | val

    waveforms = load_or_create_waveforms(rows)
    audio = load_or_extract_features(rows, waveforms)
    video_cache = np.load(RESULT_DIR / "video_features.npz")
    video = video_cache["video_combined"]

    video_c, _video_model, video_val = tune_base(video, y, train, val)
    audio_c = {}
    audio_val = {}
    for name, values in audio.items():
        c_value, _model, probability = tune_base(values, y, train, val)
        audio_c[name] = c_value
        audio_val[name] = probability
    fusion = tune_fusion(y[val], video_val, audio_val)
    audio_name = fusion["audio_name"]

    video_model = make_model(video_c)
    video_model.fit(video[train_val], y[train_val])
    video_test = video_model.predict_proba(video[test])[:, 1]
    audio_model = make_model(audio_c[audio_name])
    audio_model.fit(audio[audio_name][train_val], y[train_val])
    audio_test = audio_model.predict_proba(audio[audio_name][test])[:, 1]
    fusion_test, threshold = apply_fusion(fusion, video_test, audio_test)

    output = {
        "audio_encoder": "torchaudio.pipelines.WAV2VEC2_BASE",
        "input": "one-second 16 kHz raw waveform centered on verified contact",
        "selection_policy": "layer, regularization, and fusion selected on validation only",
        "audio_C_by_layer": audio_c,
        "selected_fusion": fusion,
        "test": {
            "video_only": metric(y[test], video_test),
            "pretrained_audio_only": metric(y[test], audio_test),
            "pretrained_audio_video": metric(y[test], fusion_test, threshold),
        },
    }
    (RESULT_DIR / "pretrained_wav2vec_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
