#!/usr/bin/env python3
"""Video-only and early-fusion baselines using the audio experiment split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from audio_baseline import (
    extract_features,
    make_splits,
    read_manifest,
    train_and_evaluate,
    write_csv,
)


FRAME_OFFSETS = (-0.35, -0.15, 0.0, 0.2, 0.45, 0.75, 1.05, 1.35)
APPEARANCE_INDICES = (2, 4, 6, 7)


def normalized_gray(frame: np.ndarray, width: int = 24, height: int = 14) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    gray -= float(gray.mean())
    gray /= float(gray.std() + 1e-6)
    return np.clip(gray, -4.0, 4.0)


def read_video_frames(path: Path, center: float) -> tuple[list[np.ndarray], int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 else center + max(FRAME_OFFSETS) + 0.1
    frames = []
    fallback_count = 0
    try:
        for offset in FRAME_OFFSETS:
            timestamp = min(max(0.0, center + offset), max(0.0, duration - 0.001))
            frame = None
            for backoff in (0.0, 1.0 / max(fps, 25.0), 0.1, 0.25, 0.5, 1.0):
                candidate = max(0.0, timestamp - backoff)
                capture.set(cv2.CAP_PROP_POS_MSEC, candidate * 1000.0)
                ok, decoded = capture.read()
                if ok and decoded is not None:
                    frame = decoded
                    if backoff:
                        fallback_count += 1
                    break
            if frame is None:
                raise RuntimeError(f"Cannot decode {path} at {timestamp:.3f}s")
            frames.append(normalized_gray(frame))
    finally:
        capture.release()
    return frames, fallback_count


def video_features(frames: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stack = np.stack(frames)
    appearance = stack[list(APPEARANCE_INDICES)].reshape(-1)
    differences = np.abs(np.diff(stack, axis=0))
    motion = np.concatenate(
        [differences.mean(axis=0).reshape(-1), differences.max(axis=0).reshape(-1)]
    )
    combined = np.concatenate([appearance, motion])
    return appearance.astype(np.float32), motion.astype(np.float32), combined.astype(np.float32)


def extract_video_features(repo: Path, rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    appearances = []
    motions = []
    combined = []
    fallback_total = 0
    for index, row in enumerate(rows, start=1):
        center = (float(row["final_event_start"]) + float(row["final_event_end"])) / 2
        frames, fallback_count = read_video_frames(
            repo / row["dataset_path"] / "video.mp4", center
        )
        fallback_total += fallback_count
        appearance, motion, full = video_features(frames)
        appearances.append(appearance)
        motions.append(motion)
        combined.append(full)
        if index % 50 == 0:
            print(
                f"video_features={index}/{len(rows)} frame_fallbacks={fallback_total}",
                flush=True,
            )
    return {
        "video_appearance": np.stack(appearances),
        "video_motion": np.stack(motions),
        "video_combined": np.stack(combined),
    }


def load_or_extract_audio(
    rows: list[dict[str, str]], cache_path: Path, sample_rate: int, window_seconds: float
) -> dict[str, np.ndarray]:
    expected = np.asarray([row["dataset_path"] for row in rows])
    if cache_path.exists():
        cache = np.load(cache_path)
        if np.array_equal(cache["dataset_paths"], expected):
            return {name: cache[name] for name in cache.files if name != "dataset_paths"}
    features = extract_features(rows, sample_rate, window_seconds)
    np.savez_compressed(cache_path, dataset_paths=expected, **features)
    return features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/binary_classifier/results"))
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--window-seconds", type=float, default=1.0)
    args = parser.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(repo, manifest)
    split_map = make_splits(rows, args.seed)
    split_names = np.asarray([split_map[row["dataset_path"]] for row in rows])
    y = np.asarray([int(row["target"]) for row in rows])

    audio = load_or_extract_audio(
        rows, output_dir / "audio_features.npz", args.sample_rate, args.window_seconds
    )
    video_cache_path = output_dir / "video_features.npz"
    expected = np.asarray([row["dataset_path"] for row in rows])
    if video_cache_path.exists():
        cache = np.load(video_cache_path)
        if np.array_equal(cache["dataset_paths"], expected):
            video = {name: cache[name] for name in cache.files if name != "dataset_paths"}
        else:
            video = extract_video_features(repo, rows)
    else:
        video = extract_video_features(repo, rows)
    np.savez_compressed(video_cache_path, dataset_paths=expected, **video)

    feature_sets = {
        **video,
        "audio_video_early_fusion": np.concatenate(
            [audio["contact_logmel"], video["video_combined"]], axis=1
        ),
        "background_audio_video_control": np.concatenate(
            [audio["background_logmel"], video["video_combined"]], axis=1
        ),
    }
    results = []
    prediction_columns = {}
    for name, values in feature_sets.items():
        result, predictions, scores = train_and_evaluate(name, values, y, split_names)
        results.append(result)
        prediction_columns[name] = (predictions, scores)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    fields = list(results[0])
    write_csv(output_dir / "multimodal_baseline_results.csv", fields, results)
    (output_dir / "multimodal_baseline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    test_rows = []
    for index, row in enumerate(rows):
        if split_names[index] != "test":
            continue
        output = {
            "dataset_path": row["dataset_path"],
            "sample_id": row["sample_id"],
            "label": row["label"],
            "source_group": row["source_group"],
        }
        for name, (predictions, scores) in prediction_columns.items():
            output[f"{name}_prediction"] = (
                "fly_ball" if predictions[index] == 1 else "ground_ball"
            )
            output[f"{name}_score"] = scores[index]
        test_rows.append(output)
    write_csv(output_dir / "multimodal_test_predictions.csv", list(test_rows[0]), test_rows)


if __name__ == "__main__":
    main()
