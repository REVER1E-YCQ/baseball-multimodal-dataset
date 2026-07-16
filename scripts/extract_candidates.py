from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import read_csv, repo_path, safe_slug, tool_path, write_csv


CLIP_FIELDS = [
    "clip_id",
    "source_id",
    "source_path",
    "clip_path",
    "audio_path",
    "start_time",
    "end_time",
    "expected_label",
    "status",
    "notes",
]


def run_ffmpeg(source: Path, start: float, end: float, clip_path: Path, audio_path: Path) -> None:
    duration = max(0.01, end - start)
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            tool_path("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(clip_path),
        ],
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            tool_path("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(clip_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "48000",
            "-sample_fmt",
            "s16",
            str(audio_path),
        ],
        check=True,
        timeout=180,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cut candidate clips and audio from source videos.")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    parser.add_argument("--expected-label", choices=["ground_ball", "fly_ball", "unknown"], default="unknown")
    parser.add_argument("--clip-id")
    parser.add_argument("--manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start")

    source = args.source_path
    if not source.is_absolute():
        source = repo_path(str(source))
    if not source.exists() and not args.dry_run:
        raise SystemExit(f"Source does not exist: {source}")

    clip_id = args.clip_id or f"{safe_slug(args.source_id)}_{args.start:.3f}_{args.end:.3f}".replace(".", "p")
    clip_dir = repo_path("clips", "pending", safe_slug(clip_id))
    clip_path = clip_dir / "video.mp4"
    audio_path = clip_dir / "audio.wav"

    if not args.dry_run:
        run_ffmpeg(source, args.start, args.end, clip_path, audio_path)

    rows = read_csv(args.manifest)
    row = {
        "clip_id": clip_id,
        "source_id": args.source_id,
        "source_path": str(source.relative_to(repo_path()) if source.is_relative_to(repo_path()) else source),
        "clip_path": str(clip_path.relative_to(repo_path())),
        "audio_path": str(audio_path.relative_to(repo_path())),
        "start_time": f"{args.start:.3f}",
        "end_time": f"{args.end:.3f}",
        "expected_label": args.expected_label,
        "status": "pending" if not args.dry_run else "dry_run",
        "notes": "",
    }
    rows = [r for r in rows if r.get("clip_id") != clip_id]
    rows.append(row)
    write_csv(args.manifest, rows, CLIP_FIELDS)
    print(f"Wrote candidate {clip_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
