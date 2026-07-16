from __future__ import annotations

import argparse
from pathlib import Path

from collect_mlb_sources import FIELDS
from common import read_csv, repo_path, safe_slug, sha256_file, write_csv


MEDIA_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Recover source manifest download metadata from existing media files.")
    parser.add_argument("--manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--raw-dir", type=Path, default=repo_path("raw_sources"))
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    media = [
        path
        for path in args.raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    ]
    by_prefix: dict[str, list[Path]] = {}
    for path in media:
        source_prefix = path.name.split("_", 2)
        if len(source_prefix) >= 2:
            by_prefix.setdefault("_".join(source_prefix[:2]), []).append(path)

    recovered = 0
    missing = 0
    for row in rows:
        if row.get("status") == "downloaded" and row.get("local_path"):
            continue
        source_id = safe_slug(row.get("source_id", ""), "source")
        candidates = [
            path
            for path in by_prefix.get("_".join(source_id.split("_")[:2]), [])
            if path.name.startswith(source_id + "_")
        ]
        if not candidates:
            missing += 1
            continue
        selected = max(candidates, key=lambda path: (path.stat().st_size, path.stat().st_mtime_ns))
        row["status"] = "downloaded"
        row["local_path"] = str(selected.relative_to(repo_path()))
        row["source_hash"] = sha256_file(selected)
        row["notes"] = (row.get("notes") or "") + "; recovered_from_existing_media"
        recovered += 1

    write_csv(args.manifest, rows, FIELDS)
    print(f"rows={len(rows)} media={len(media)} recovered={recovered} unmatched_non_downloaded={missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
