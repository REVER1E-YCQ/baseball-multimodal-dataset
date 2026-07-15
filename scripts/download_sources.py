from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path

from common import read_csv, repo_path, safe_slug, sha256_file, write_csv


FIELDS = [
    "source_id",
    "source_url",
    "video_title",
    "game_date",
    "teams",
    "batter",
    "event_text",
    "expected_label",
    "rights_note",
    "status",
    "local_path",
    "source_hash",
    "notes",
]

# These historical rows were verified as unsuitable large/unstable sources.
# Keep the exclusion explicit so a resumed queue cannot get stuck on them.
SKIP_SOURCE_IDS = {
    "MLB_825107_condensed-game-det-az-3-31-26",
    "MLB_823322_xander-bogaerts-grounds-out-shortstop-willy-adames-to-first-baseman-c",
    "MLB_823158_cole-wilcox-in-play-out-s-to-aaron-judge",
    "MLB_746249_trevor-story-departs-game-with-injury",
}


def ensure_source_ids(rows: list[dict[str, str]]) -> None:
    for idx, row in enumerate(rows, start=1):
        if not row.get("source_id"):
            row["source_id"] = f"S_{idx:05d}"


def download_one(row: dict[str, str], output_dir: Path, dry_run: bool) -> dict[str, str]:
    url = row.get("source_url", "").strip()
    if not url:
        row["status"] = "missing_url"
        return row

    source_id = safe_slug(row.get("source_id", ""), "source")
    title = safe_slug(row.get("video_title", ""), source_id)[:80]
    out_template = str(output_dir / f"{source_id}_{title}.%(ext)s")
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    direct_media = suffix in {".mp4", ".mkv", ".mov", ".webm"}

    if dry_run:
        print(f"Would download {row.get('source_id')} -> {out_template}")
        return row

    output_dir.mkdir(parents=True, exist_ok=True)
    if direct_media:
        media = output_dir / f"{source_id}_{title}{suffix}"
        part = media.with_suffix(media.suffix + ".part")
        try:
            subprocess.run(
                [
                    "curl.exe",
                    "--fail",
                    "--location",
                    "--connect-timeout",
                    "30",
                    "--max-time",
                    "180",
                    "--user-agent",
                    "baseball-dataset-research/0.1",
                    "--output",
                    str(part),
                    url,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(part, media)
            row["status"] = "downloaded"
            row["local_path"] = str(media.relative_to(repo_path()))
            row["source_hash"] = sha256_file(media)
            row["notes"] = row.get("notes", "")
        except (OSError, subprocess.SubprocessError) as exc:
            part.unlink(missing_ok=True)
            row["status"] = "download_failed"
            row["notes"] = f"{type(exc).__name__}: {str(exc)[:450]}"
        return row

    if not shutil.which("yt-dlp"):
        row["status"] = "yt_dlp_missing"
        row["notes"] = "Install yt-dlp or place a direct local_path manually."
        return row

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--socket-timeout",
        "30",
        "--retries",
        "1",
        "--fragment-retries",
        "1",
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        out_template,
        url,
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired:
        row["status"] = "download_failed"
        row["notes"] = "yt-dlp timed out after 180 seconds"
        return row
    if proc.returncode != 0:
        row["status"] = "download_failed"
        row["notes"] = (proc.stderr or proc.stdout).strip()[-500:]
        return row

    matches = sorted(output_dir.glob(f"{source_id}_{title}.*"))
    media = next((p for p in matches if p.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}), None)
    if not media:
        row["status"] = "download_missing_output"
        return row

    row["status"] = "downloaded"
    row["local_path"] = str(media.relative_to(repo_path()))
    row["source_hash"] = sha256_file(media)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="Download source videos listed in sources_manifest.csv.")
    parser.add_argument("--manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("raw_sources"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--source-ids",
        default="",
        help="Comma-separated source IDs to download. Defaults to every non-downloaded source.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    ensure_source_ids(rows)
    selected_ids = {item.strip() for item in args.source_ids.split(",") if item.strip()}
    if args.dry_run:
        for row in rows:
            if row.get("status") == "downloaded" and row.get("local_path"):
                continue
            download_one(dict(row), args.output_dir, True)
        print(f"Dry run checked {len(rows)} rows; manifest not modified")
        return 0

    processed = 0
    for row in rows:
        if selected_ids and row.get("source_id") not in selected_ids:
            continue
        if row.get("status") == "downloaded" and row.get("local_path"):
            continue
        if row.get("source_id") in SKIP_SOURCE_IDS:
            row["status"] = "skip_known_large_source"
            row["notes"] = "Known unstable or oversized source; skipped during batch download."
            write_csv(args.manifest, rows, FIELDS)
            continue
        download_one(row, args.output_dir, args.dry_run)
        processed += 1
        write_csv(args.manifest, rows, FIELDS)
        if args.limit and processed >= args.limit:
            break

    print(f"Updated {args.manifest} ({len(rows)} rows; processed={processed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
