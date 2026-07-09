from __future__ import annotations

import argparse
import math
import statistics
import wave
from pathlib import Path

from common import read_csv, repo_path, write_csv
from extract_candidates import CLIP_FIELDS


def read_wav_mono(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit wav, got sample width {width}")
    samples: list[float] = []
    frame_width = width * channels
    for idx in range(0, len(frames), frame_width):
        values = []
        for channel in range(channels):
            offset = idx + channel * width
            values.append(int.from_bytes(frames[offset : offset + width], "little", signed=True) / 32768.0)
        samples.append(sum(values) / len(values))
    return sample_rate, samples


def rms(samples: list[float]) -> float:
    if not samples:
        return 0.0
    return math.sqrt(sum(value * value for value in samples) / len(samples))


def derivative(samples: list[float]) -> list[float]:
    return [samples[index] - samples[index - 1] for index in range(1, len(samples))]


def window_energies(samples: list[float], sample_rate: int, window_ms: float) -> list[tuple[float, float]]:
    window = max(1, int(sample_rate * window_ms / 1000.0))
    values: list[tuple[float, float]] = []
    for idx in range(0, max(1, len(samples) - window), window):
        values.append((idx / sample_rate, rms(samples[idx : idx + window])))
    return values


def peak_ratio(windows: list[tuple[float, float]]) -> tuple[float, float, float]:
    if not windows:
        return 0.0, 0.0, 0.0
    peak_time, peak = max(windows, key=lambda item: item[1])
    baseline = statistics.median([energy for _time, energy in windows])
    return peak_time, peak, peak / max(baseline, 1e-9)


def score_audio(path: Path, window_ms: float) -> tuple[float, float, float, float, float]:
    sample_rate, samples = read_wav_mono(path)
    rms_time, rms_peak, rms_ratio = peak_ratio(window_energies(samples, sample_rate, window_ms))
    diff_time, diff_peak, diff_ratio = peak_ratio(window_energies(derivative(samples), sample_rate, window_ms))
    if diff_ratio > rms_ratio:
        return diff_time, diff_peak, diff_ratio, rms_ratio, diff_ratio
    return rms_time, rms_peak, rms_ratio, rms_ratio, diff_ratio


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark weak pending clips before sending them to Qwen.")
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--statuses", default="pending", help="Comma-separated statuses to score.")
    parser.add_argument("--reject-status", default="prefilter_reject")
    parser.add_argument("--min-ratio", type=float, default=1.60)
    parser.add_argument("--min-peak-time", type=float, default=0.25)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = {item.strip() for item in args.statuses.split(",") if item.strip()}
    rows = read_csv(args.clips_manifest)
    selected = [row for row in rows if row.get("status") in wanted]
    if args.limit:
        selected = selected[: args.limit]

    passed = 0
    rejected = 0
    failed = 0
    for row in selected:
        audio_path = Path(row.get("audio_path", ""))
        if not audio_path.is_absolute():
            audio_path = repo_path(str(audio_path))
        try:
            peak_time, peak, ratio, rms_ratio, diff_ratio = score_audio(audio_path, args.window_ms)
        except Exception as exc:
            failed += 1
            if not args.dry_run:
                row["status"] = args.reject_status
                row["notes"] = f"prefilter_error={type(exc).__name__}: {str(exc)[:180]}"
            print(f"REJECT {row['clip_id']}: {type(exc).__name__}: {exc}")
            continue

        note = f"prefilter_peak={peak_time:.3f}; peak={peak:.6f}; ratio={ratio:.2f}; rms_ratio={rms_ratio:.2f}; diff_ratio={diff_ratio:.2f}"
        if ratio < args.min_ratio or peak_time < args.min_peak_time:
            rejected += 1
            if not args.dry_run:
                row["status"] = args.reject_status
                row["notes"] = note
            print(f"REJECT {row['clip_id']}: {note}")
        else:
            passed += 1
            row["notes"] = (row.get("notes", "") + "; " + note).strip("; ")
            print(f"PASS {row['clip_id']}: {note}")

    if not args.dry_run:
        write_csv(args.clips_manifest, rows, CLIP_FIELDS)
    print(f"Scored {len(selected)} clips; pass={passed}; reject={rejected}; errors={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
