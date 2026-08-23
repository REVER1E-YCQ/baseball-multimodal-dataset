from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


REQUIRED_COLUMNS = {
    "uid",
    "label",
    "source_id",
    "protocol_role",
    "audio_path",
    "event_start",
    "event_end",
}


def to_float_mono(data: np.ndarray) -> np.ndarray:
    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        scale = float(max(abs(info.min), info.max))
        data = data.astype(np.float32) / scale
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1:
        raise ValueError(f"Expected mono or stereo waveform, got shape {data.shape}")
    return np.nan_to_num(data).astype(np.float32, copy=False)


def to_int16(data: np.ndarray) -> np.ndarray:
    return np.round(np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)


def resolve_input_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def safe_filename(uid: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", uid.strip())
    if not value:
        raise ValueError(f"UID cannot be converted to a filename: {uid!r}")
    return value


def find_peak_time(
    waveform: np.ndarray,
    sample_rate: int,
    event_start: float,
    event_end: float,
) -> float:
    start = max(0, int(round(event_start * sample_rate)))
    end = min(len(waveform), int(np.ceil(event_end * sample_rate)) + 1)
    if start >= end:
        raise ValueError(
            f"Invalid annotated event interval [{event_start:.6f}, {event_end:.6f}]"
        )
    local_peak = int(np.argmax(np.abs(waveform[start:end])))
    return (start + local_peak) / float(sample_rate)


def bounded_centered_start(center: float, duration: float, audio_duration: float) -> float:
    if audio_duration < duration:
        raise ValueError(
            f"Audio duration {audio_duration:.6f}s is shorter than window {duration:.6f}s"
        )
    return min(max(0.0, center - duration / 2.0), audio_duration - duration)


def exact_slice(
    waveform: np.ndarray,
    sample_rate: int,
    start: float,
    duration: float,
) -> np.ndarray:
    target_samples = int(round(duration * sample_rate))
    start_sample = int(round(start * sample_rate))
    end_sample = start_sample + target_samples
    if start_sample < 0 or end_sample > len(waveform):
        raise ValueError("Requested window exceeds available waveform")
    result = waveform[start_sample:end_sample]
    if len(result) != target_samples:
        raise RuntimeError("Window has an unexpected sample count")
    return result


def relative_to_manifest(path: Path, output_manifest: Path) -> str:
    return Path(os.path.relpath(path, output_manifest.parent)).as_posix()


def prepare_windows(
    manifest_path: Path,
    out_root: Path,
    window_ms: tuple[int, ...] = (200,),
    pre_gap_ms: int = 50,
) -> pd.DataFrame:
    manifest_path = manifest_path.resolve()
    out_root = out_root.resolve()
    source = pd.read_csv(manifest_path)
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Input manifest is missing columns: {sorted(missing)}")
    if source["uid"].duplicated().any():
        raise ValueError("Input manifest must contain one row per uid")
    if any(value <= 0 for value in window_ms):
        raise ValueError("All window lengths must be positive")
    if pre_gap_ms < 0:
        raise ValueError("pre_gap_ms cannot be negative")

    output_manifest = out_root / "windows_manifest.csv"
    rows: list[dict[str, object]] = []
    used_filenames: dict[str, str] = {}
    gap_seconds = pre_gap_ms / 1000.0

    for row in source.itertuples(index=False):
        uid = str(row.uid)
        filename = safe_filename(uid)
        previous = used_filenames.setdefault(filename, uid)
        if previous != uid:
            raise ValueError(f"Filename collision between UIDs {previous!r} and {uid!r}")

        audio_path = resolve_input_path(str(row.audio_path), manifest_path)
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        sample_rate, raw = wavfile.read(audio_path)
        waveform = to_float_mono(raw)
        audio_duration = len(waveform) / float(sample_rate)
        event_start = float(row.event_start)
        event_end = float(row.event_end)
        if not (0 <= event_start < event_end <= audio_duration):
            raise ValueError(
                f"Invalid event interval for {uid}: "
                f"{event_start:.6f} < {event_end:.6f}, audio={audio_duration:.6f}"
            )
        peak_time = find_peak_time(
            waveform,
            int(sample_rate),
            event_start,
            event_end,
        )

        for milliseconds in sorted(set(window_ms)):
            duration = milliseconds / 1000.0
            suffix = f"{milliseconds:03d}ms"
            event_name = f"event_{suffix}"
            requested_start = peak_time - duration / 2.0
            event_window_start = bounded_centered_start(
                peak_time,
                duration,
                audio_duration,
            )
            event_audio = exact_slice(
                waveform,
                int(sample_rate),
                event_window_start,
                duration,
            )
            event_path = out_root / "windows" / event_name / f"{filename}.wav"
            event_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(event_path, int(sample_rate), to_int16(event_audio))
            rows.append(
                {
                    "uid": uid,
                    "label": str(row.label),
                    "source_id": str(row.source_id),
                    "protocol_role": str(row.protocol_role),
                    "window_name": event_name,
                    "window_kind": "event",
                    "window_path": relative_to_manifest(event_path, output_manifest),
                    "window_start": event_window_start,
                    "window_end": event_window_start + duration,
                    "window_duration": duration,
                    "sample_rate": int(sample_rate),
                    "event_start": event_start,
                    "event_end": event_end,
                    "estimated_peak_time": peak_time,
                    "window_shift_from_requested_ms": (
                        event_window_start - requested_start
                    )
                    * 1000.0,
                    "alignment_method": (
                        "absolute_amplitude_peak_within_annotated_event_interval"
                    ),
                    "wav_boundary_padding_samples": 0,
                }
            )

            pre_end = event_start - gap_seconds
            pre_start = pre_end - duration
            if pre_start < 0:
                continue
            pre_name = f"pre_{suffix}"
            pre_audio = exact_slice(
                waveform,
                int(sample_rate),
                pre_start,
                duration,
            )
            pre_path = out_root / "windows" / pre_name / f"{filename}.wav"
            pre_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(pre_path, int(sample_rate), to_int16(pre_audio))
            rows.append(
                {
                    "uid": uid,
                    "label": str(row.label),
                    "source_id": str(row.source_id),
                    "protocol_role": str(row.protocol_role),
                    "window_name": pre_name,
                    "window_kind": "strict_pre",
                    "window_path": relative_to_manifest(pre_path, output_manifest),
                    "window_start": pre_start,
                    "window_end": pre_end,
                    "window_duration": duration,
                    "sample_rate": int(sample_rate),
                    "event_start": event_start,
                    "event_end": event_end,
                    "estimated_peak_time": peak_time,
                    "window_shift_from_requested_ms": 0.0,
                    "alignment_method": (
                        f"strict_pre_ending_{pre_gap_ms}ms_before_event_start"
                    ),
                    "wav_boundary_padding_samples": 0,
                }
            )

    result = pd.DataFrame(rows).sort_values(["window_name", "uid"]).reset_index(drop=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_manifest, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create peak-centred event windows and strict pre-event controls "
            "without waveform padding."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--window-ms", type=int, nargs="+", default=[200])
    parser.add_argument("--pre-gap-ms", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare_windows(
        args.manifest,
        args.out_root,
        tuple(args.window_ms),
        args.pre_gap_ms,
    )
    print(f"Wrote {len(result)} rows to {(args.out_root / 'windows_manifest.csv').resolve()}")
    print(result.groupby(["window_name", "protocol_role", "label"]).size().to_string())


if __name__ == "__main__":
    main()
