from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import wave
from pathlib import Path

from common import repo_path


def sample_dirs(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.glob("*/*/*") if p.is_dir()])


def sample_number(sample_id: str) -> tuple[str, int] | None:
    match = re.fullmatch(r"([A-Za-z]+)_(\d+)", sample_id)
    if not match:
        return None
    return match.group(1).upper(), int(match.group(2))


def parse_id_mins(raw: str) -> dict[str, int]:
    mins: dict[str, int] = {}
    if not raw:
        return mins
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit("--id-min entries must look like F=104,G=161")
        prefix, value = item.split("=", 1)
        mins[prefix.strip().upper()] = int(value)
    return mins


def filter_sample_dirs(dirs: list[Path], id_mins: dict[str, int], limit: int) -> list[Path]:
    if id_mins:
        filtered = []
        for path in dirs:
            parsed = sample_number(path.name)
            if not parsed:
                continue
            prefix, number = parsed
            if prefix in id_mins and number >= id_mins[prefix]:
                filtered.append(path)
        dirs = filtered
    if limit:
        dirs = dirs[:limit]
    return dirs


def window_energies_from_wav(path: Path, window_ms: float = 20.0) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit wav, got sample width {width}")
    window = max(1, int(sample_rate * window_ms / 1000.0))
    frame_width = width * channels
    energies: list[tuple[float, float]] = []
    diff_energies: list[tuple[float, float]] = []
    window_start = 0
    count = 0
    sum_sq = 0.0
    diff_count = 0
    diff_sum_sq = 0.0
    previous: float | None = None

    def flush() -> None:
        nonlocal count, sum_sq, diff_count, diff_sum_sq, window_start
        if count:
            energies.append((window_start / sample_rate, math.sqrt(sum_sq / count)))
        if diff_count:
            diff_energies.append((window_start / sample_rate, math.sqrt(diff_sum_sq / diff_count)))
        window_start += count
        count = 0
        sum_sq = 0.0
        diff_count = 0
        diff_sum_sq = 0.0

    for offset in range(0, len(frames), frame_width):
        sample = 0.0
        for channel in range(channels):
            channel_offset = offset + channel * width
            raw = int.from_bytes(frames[channel_offset : channel_offset + width], "little", signed=True)
            sample += raw / 32768.0
        sample /= channels

        sum_sq += sample * sample
        count += 1
        if previous is not None:
            diff = sample - previous
            diff_sum_sq += diff * diff
            diff_count += 1
        previous = sample
        if count >= window:
            flush()
    flush()
    return energies, diff_energies


def read_event_interval(path: Path) -> tuple[float, float]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        row = next(csv.DictReader(fh))
    return float(row["event_start"]), float(row["event_end"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a strong audio transient is near the annotated contact interval.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--tolerance", type=float, default=0.50)
    parser.add_argument("--min-ratio", type=float, default=2.0)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--id-min", default="", help="Comma-separated sample minimums, for example F=104,G=161.")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    failures = 0
    dirs = filter_sample_dirs(sample_dirs(args.dataset_root), parse_id_mins(args.id_min), args.limit)
    for path in dirs:
        try:
            event_start, event_end = read_event_interval(path)
            energies, diff_energies = window_energies_from_wav(path / "audio.wav", args.window_ms)
        except Exception as exc:
            print(f"FAIL {path.relative_to(args.dataset_root.parent)}: {exc}")
            failures += 1
            continue

        event_windows = [
            item for item in energies if event_start - args.tolerance <= item[0] <= event_end + args.tolerance
        ]
        if not event_windows:
            print(f"FAIL {path.relative_to(args.dataset_root.parent)}: no audio windows near event")
            failures += 1
            continue
        event_peak_time, event_peak_energy = max(event_windows, key=lambda item: item[1])
        baseline = statistics.median([energy for _time, energy in energies]) if energies else 0.0
        rms_ratio = event_peak_energy / max(baseline, 1e-9)

        diff_event_windows = [
            item for item in diff_energies if event_start - args.tolerance <= item[0] <= event_end + args.tolerance
        ]
        diff_peak_energy = max((energy for _time, energy in diff_event_windows), default=0.0)
        diff_baseline = statistics.median([energy for _time, energy in diff_energies]) if diff_energies else 0.0
        diff_ratio = diff_peak_energy / max(diff_baseline, 1e-9)
        ratio = max(rms_ratio, diff_ratio)

        if ratio < args.min_ratio:
            print(
                f"FAIL {path.relative_to(args.dataset_root.parent)}: event_peak_time={event_peak_time:.3f}s "
                f"event={event_start:.3f}-{event_end:.3f}s "
                f"rms_ratio={rms_ratio:.2f} diff_ratio={diff_ratio:.2f}"
            )
            failures += 1
    print(f"Checked {len(dirs)} samples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
