from __future__ import annotations

import argparse
import subprocess
import tempfile
import wave
from pathlib import Path

from common import ffprobe_duration, read_csv, repo_path, safe_slug, tool_path, write_csv
from extract_candidates import CLIP_FIELDS, run_ffmpeg


def extract_temp_wav(source: Path, temp_dir: Path) -> Path:
    out = temp_dir / "source_audio.wav"
    subprocess.run(
        [
            tool_path("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            str(out),
        ],
        check=True,
    )
    return out


def read_wav(path: Path) -> tuple[int, list[float]]:
    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        channels = wav.getnchannels()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit WAV, got width={width}")
    values: list[float] = []
    frame_width = width * channels
    for idx in range(0, len(frames), frame_width):
        raw = int.from_bytes(frames[idx : idx + width], "little", signed=True)
        values.append(abs(raw) / 32768.0)
    return sample_rate, values


def energy_windows(samples: list[float], sample_rate: int, window_ms: float) -> list[tuple[float, float]]:
    window = max(1, int(sample_rate * window_ms / 1000.0))
    values: list[tuple[float, float]] = []
    for idx in range(0, len(samples), window):
        chunk = samples[idx : idx + window]
        if not chunk:
            continue
        energy = sum(x * x for x in chunk) / len(chunk)
        values.append((idx / sample_rate, energy))
    return values


def choose_peaks(windows: list[tuple[float, float]], count: int, min_gap: float) -> list[float]:
    ranked = sorted(windows, key=lambda item: item[1], reverse=True)
    chosen: list[float] = []
    for time_s, _energy in ranked:
        if all(abs(time_s - other) >= min_gap for other in chosen):
            chosen.append(time_s)
        if len(chosen) >= count:
            break
    return sorted(chosen)


def clamp_window(peak: float, duration: float, clip_duration: float, pre_roll: float) -> tuple[float, float]:
    start = max(0.0, peak - pre_roll)
    end = start + clip_duration
    if end > duration:
        end = duration
        start = max(0.0, end - clip_duration)
    return start, end


def selected_sources(rows: list[dict[str, str]], statuses: set[str], labels: set[str]) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        if row.get("status") not in statuses:
            continue
        if row.get("expected_label") not in labels:
            continue
        if not row.get("local_path"):
            continue
        selected.append(row)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut candidate clips around likely audio contact peaks.")
    parser.add_argument("--sources-manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidates-per-source", type=int, default=1)
    parser.add_argument("--pre-roll", type=float, default=2.0)
    parser.add_argument("--ground-duration", type=float, default=6.0)
    parser.add_argument("--fly-duration", type=float, default=7.0)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--min-peak-gap", type=float, default=3.0)
    parser.add_argument("--statuses", default="downloaded")
    parser.add_argument("--labels", default="ground_ball,fly_ball")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_rows = read_csv(args.sources_manifest)
    clip_rows = read_csv(args.clips_manifest)
    existing_ids = {row.get("clip_id") for row in clip_rows}
    statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    sources = selected_sources(source_rows, statuses, labels)
    if args.limit:
        sources = sources[: args.limit]

    created = 0
    for row in sources:
        source = repo_path(row["local_path"])
        duration = ffprobe_duration(source)
        if duration is None:
            print(f"SKIP {row['source_id']}: cannot read duration")
            continue
        try:
            with tempfile.TemporaryDirectory() as temp:
                wav = extract_temp_wav(source, Path(temp))
                sample_rate, samples = read_wav(wav)
                windows = energy_windows(samples, sample_rate, args.window_ms)
                peaks = choose_peaks(windows, args.candidates_per_source, args.min_peak_gap)
        except Exception as exc:
            print(f"SKIP {row['source_id']}: audio peak detection failed: {exc}")
            continue

        label = row["expected_label"]
        clip_duration = args.fly_duration if label == "fly_ball" else args.ground_duration
        for index, peak in enumerate(peaks, start=1):
            start, end = clamp_window(peak, duration, clip_duration, args.pre_roll)
            clip_id = f"{safe_slug(row['source_id'])}_auto{index}_{start:.3f}_{end:.3f}".replace(".", "p")
            if clip_id in existing_ids:
                continue
            clip_dir = repo_path("clips", "pending", clip_id)
            clip_path = clip_dir / "video.mp4"
            audio_path = clip_dir / "audio.wav"
            if not args.dry_run:
                run_ffmpeg(source, start, end, clip_path, audio_path)
            clip_rows.append(
                {
                    "clip_id": clip_id,
                    "source_id": row["source_id"],
                    "source_path": row["local_path"],
                    "clip_path": str(clip_path.relative_to(repo_path())),
                    "audio_path": str(audio_path.relative_to(repo_path())),
                    "start_time": f"{start:.3f}",
                    "end_time": f"{end:.3f}",
                    "expected_label": label,
                    "status": "pending" if not args.dry_run else "dry_run",
                    "notes": f"auto_peak={peak:.3f}",
                }
            )
            existing_ids.add(clip_id)
            created += 1
            print(f"{'Would create' if args.dry_run else 'Created'} {clip_id}")

    if not args.dry_run:
        write_csv(args.clips_manifest, clip_rows, CLIP_FIELDS)
    print(f"Auto-extracted {created} candidate clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

