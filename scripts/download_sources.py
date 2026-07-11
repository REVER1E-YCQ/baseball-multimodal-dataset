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
        req = urllib.request.Request(url, headers={"User-Agent": "baseball-dataset-research/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response, part.open("wb") as fh:
                shutil.copyfileobj(response, fh)
            os.replace(part, media)
            row["status"] = "downloaded"
            row["local_path"] = str(media.relative_to(repo_path()))
            row["source_hash"] = sha256_file(media)
            row["notes"] = row.get("notes", "")
        except (OSError, urllib.error.URLError) as exc:
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
        "--merge-output-format",
        "mp4",
        "-f",
        "bv*+ba/b",
        "-o",
        out_template,
        url,
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    ensure_source_ids(rows)
    if args.dry_run:
        for row in rows:
            if row.get("status") == "downloaded" and row.get("local_path"):
                continue
            download_one(dict(row), args.output_dir, True)
        print(f"Dry run checked {len(rows)} rows; manifest not modified")
        return 0

    processed = 0
    for row in rows:
        if row.get("status") == "downloaded" and row.get("local_path"):
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
