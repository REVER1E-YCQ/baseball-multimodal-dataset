from __future__ import annotations

import argparse
from pathlib import Path

from common import read_csv, repo_path, write_csv


FIELDS = [
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset selected clip statuses in clips_manifest.csv.")
    parser.add_argument("--manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--from-status", default="label_failed")
    parser.add_argument("--to-status", default="pending")
    parser.add_argument("--clear-notes", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.manifest)
    changed = 0
    for row in rows:
        if row.get("status") == args.from_status:
            row["status"] = args.to_status
            if args.clear_notes:
                row["notes"] = ""
            changed += 1
    write_csv(args.manifest, rows, FIELDS)
    print(f"Reset {changed} rows from {args.from_status} to {args.to_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

