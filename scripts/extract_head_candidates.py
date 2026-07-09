from __future__ import annotations

import argparse
from pathlib import Path

from common import ffprobe_duration, read_csv, repo_path, safe_slug, write_csv
from extract_candidates import CLIP_FIELDS, run_ffmpeg


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
    parser = argparse.ArgumentParser(description="Cut head-window candidate clips from downloaded source videos.")
    parser.add_argument("--sources-manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--ground-duration", type=float, default=6.0)
    parser.add_argument("--fly-duration", type=float, default=7.0)
    parser.add_argument("--statuses", default="downloaded")
    parser.add_argument("--labels", default="ground_ball,fly_ball")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    source_rows = read_csv(args.sources_manifest)
    clip_rows = read_csv(args.clips_manifest)
    existing_ids = {row.get("clip_id") for row in clip_rows}
    existing_windows = {
        (row.get("source_id", ""), row.get("start_time", ""), row.get("end_time", "")) for row in clip_rows
    }
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
        label = row["expected_label"]
        clip_duration = args.fly_duration if label == "fly_ball" else args.ground_duration
        start = min(args.start, max(0.0, duration - 0.1))
        end = min(duration, start + clip_duration)
        start_text = f"{start:.3f}"
        end_text = f"{end:.3f}"
        clip_id = f"{safe_slug(row['source_id'])}_head_{start:.3f}_{end:.3f}".replace(".", "p")
        if clip_id in existing_ids or (row["source_id"], start_text, end_text) in existing_windows:
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
                "start_time": start_text,
                "end_time": end_text,
                "expected_label": label,
                "status": "pending" if not args.dry_run else "dry_run",
                "notes": "head_window",
            }
        )
        existing_ids.add(clip_id)
        existing_windows.add((row["source_id"], start_text, end_text))
        created += 1
        print(f"{'Would create' if args.dry_run else 'Created'} {clip_id}")

    if not args.dry_run:
        write_csv(args.clips_manifest, clip_rows, CLIP_FIELDS)
    print(f"Extracted {created} head-window candidate clips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
