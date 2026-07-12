from __future__ import annotations

import argparse
import copy
import json
import math
import wave
from pathlib import Path
from typing import Any

from common import load_jsonl, read_csv, repo_path


def latest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        clip_id = record.get("clip_id")
        if clip_id:
            latest[clip_id] = record
    return latest


def read_wav_mono(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit WAV, got sample width {width}")
    values = []
    for idx in range(0, len(frames), width * channels):
        channel_samples = []
        for ch in range(channels):
            offset = idx + ch * width
            raw = int.from_bytes(frames[offset : offset + width], "little", signed=True)
            channel_samples.append(raw / 32768.0)
        values.append(sum(channel_samples) / len(channel_samples))
    return sample_rate, values


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(x * x for x in samples) / len(samples))


def peak_near_event(
    samples: list[float],
    sample_rate: int,
    event_start: float,
    event_end: float,
    search_margin: float,
    window_ms: float,
) -> float:
    window = max(1, int(sample_rate * window_ms / 1000.0))
    start_idx = max(0, int((event_start - search_margin) * sample_rate))
    end_idx = min(len(samples), int((event_end + search_margin) * sample_rate))
    if end_idx <= start_idx:
        end_idx = min(len(samples), start_idx + window)
    best_idx = start_idx
    best_energy = -1.0
    for idx in range(start_idx, max(start_idx + 1, end_idx - window + 1), window):
        energy = rms(samples[idx : idx + window])
        if energy > best_energy:
            best_energy = energy
            best_idx = idx
    return best_idx / sample_rate


def refine_record(record: dict[str, Any], clip_row: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    refined = copy.deepcopy(record)
    payload = refined.get("label") or {}
    if payload.get("label") not in {"ground_ball", "fly_ball"}:
        return refined
    audio_path = Path(clip_row.get("audio_path", ""))
    if not audio_path.is_absolute():
        audio_path = repo_path(str(audio_path))
    sample_rate, samples = read_wav_mono(audio_path)
    event_start = float(payload.get("event_start") or 0.0)
    event_end = float(payload.get("event_end") or 0.0)
    peak = peak_near_event(samples, sample_rate, event_start, event_end, args.search_margin, args.window_ms)
    original_mid = (event_start + event_end) / 2.0
    if args.max_shift > 0 and abs(peak - original_mid) > args.max_shift:
        refined["event_refinement_review"] = {
            "reason": "audio_peak_shift_exceeds_limit",
            "method": "audio_peak_near_qwen_interval",
            "original_event_start": event_start,
            "original_event_end": event_end,
            "peak_time": round(peak, 3),
            "max_shift": args.max_shift,
        }
        return refined
    half_width = args.event_width / 2.0
    duration = len(samples) / sample_rate
    new_start = max(0.0, peak - half_width)
    new_end = min(duration, peak + half_width)
    if new_end <= new_start:
        new_end = min(duration, new_start + args.event_width)
    payload["event_start"] = round(new_start, 3)
    payload["event_end"] = round(new_end, 3)
    refined["label"] = payload
    refined["event_refinement"] = {
        "method": "audio_peak_near_qwen_interval",
        "original_event_start": event_start,
        "original_event_end": event_end,
        "peak_time": round(peak, 3),
        "event_width": args.event_width,
    }
    return refined


def main() -> int:
    parser = argparse.ArgumentParser(description="Refine Qwen contact event times using local audio transients.")
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels.jsonl"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--output", type=Path, default=repo_path("reports", "qwen_labels_refined.jsonl"))
    parser.add_argument("--search-margin", type=float, default=0.750)
    parser.add_argument("--event-width", type=float, default=0.100)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument(
        "--max-shift",
        type=float,
        default=0.350,
        help="Do not rewrite event timing when the selected audio peak is this many seconds from the original event midpoint. Use 0 to disable.",
    )
    args = parser.parse_args()

    clip_rows = {row["clip_id"]: row for row in read_csv(args.clips_manifest)}
    records = latest_records(load_jsonl(args.labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    refined_count = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for clip_id, record in records.items():
            clip_row = clip_rows.get(clip_id)
            if not clip_row:
                continue
            try:
                refined = refine_record(record, clip_row, args)
            except Exception as exc:
                refined = copy.deepcopy(record)
                refined["event_refinement_error"] = f"{type(exc).__name__}: {exc}"
            if refined.get("event_refinement"):
                refined_count += 1
            fh.write(json.dumps(refined, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Refined {refined_count} label records; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
