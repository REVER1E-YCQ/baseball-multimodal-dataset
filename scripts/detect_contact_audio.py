from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path

from common import repo_path


def sample_dirs(dataset_root: Path) -> list[Path]:
    return [p for p in dataset_root.glob("*/*/*") if p.is_dir()]


def read_wav_mono(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit wav, got sample width {width}")
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


def peak_window_time(samples: list[float], sample_rate: int, window_ms: float = 20.0) -> tuple[float, float]:
    window = max(1, int(sample_rate * window_ms / 1000.0))
    best_idx = 0
    best_energy = -1.0
    for idx in range(0, max(1, len(samples) - window), window):
        energy = rms(samples[idx : idx + window])
        if energy > best_energy:
            best_energy = energy
            best_idx = idx
    return best_idx / sample_rate, best_energy


def read_event_interval(path: Path) -> tuple[float, float]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        row = next(csv.DictReader(fh))
    return float(row["event_start"]), float(row["event_end"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether a strong audio transient is near the annotated contact interval.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--tolerance", type=float, default=0.50)
    parser.add_argument("--min-ratio", type=float, default=2.0)
    args = parser.parse_args()

    failures = 0
    dirs = sample_dirs(args.dataset_root)
    for path in dirs:
        try:
            sample_rate, samples = read_wav_mono(path / "audio.wav")
            event_start, event_end = read_event_interval(path)
            peak_time, peak_energy = peak_window_time(samples, sample_rate)
            global_rms = rms(samples)
        except Exception as exc:
            print(f"FAIL {path.relative_to(repo_path())}: {exc}")
            failures += 1
            continue

        near_event = event_start - args.tolerance <= peak_time <= event_end + args.tolerance
        strong = global_rms == 0 or peak_energy / max(global_rms, 1e-9) >= args.min_ratio
        if not near_event or not strong:
            print(
                f"FAIL {path.relative_to(repo_path())}: peak_time={peak_time:.3f}s "
                f"event={event_start:.3f}-{event_end:.3f}s ratio={peak_energy / max(global_rms, 1e-9):.2f}"
            )
            failures += 1
    print(f"Checked {len(dirs)} samples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

