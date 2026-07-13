from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from common import repo_path, write_csv


def natural_key(value: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)_(\d+)", value)
    return (match.group(1), int(match.group(2))) if match else (value, -1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a stable numbered index for the materialized dataset.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--output", type=Path, default=repo_path("reports", "dataset_index_20260713.csv"))
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for sample_csv in sorted(args.dataset_root.glob("*/*/*/sample.csv")):
        with sample_csv.open("r", newline="", encoding="utf-8-sig") as handle:
            sample = next(csv.DictReader(handle))
        sample_dir = sample_csv.parent
        relative = sample_dir.relative_to(args.dataset_root)
        rows.append(
            {
                "sample_id": sample.get("sample_id", sample_dir.name),
                "label": sample.get("label", ""),
                "collector": relative.parts[1],
                "relative_path": str(relative).replace("\\", "/"),
                "event_start": sample.get("event_start", ""),
                "event_end": sample.get("event_end", ""),
                "region": sample.get("region", ""),
                "landing_zone": sample.get("landing_zone", ""),
                "strength": sample.get("strength", ""),
                "bounce": sample.get("bounce", ""),
                "trajectory_type": sample.get("trajectory_type", ""),
            }
        )
    rows.sort(key=lambda row: natural_key(row["sample_id"]))
    for number, row in enumerate(rows, start=1):
        row["dataset_no"] = str(number)
    fields = ["dataset_no", "sample_id", "label", "collector", "relative_path", "event_start", "event_end", "region", "landing_zone", "strength", "bounce", "trajectory_type"]
    write_csv(args.output, rows, fields)
    print(f"Indexed {len(rows)} samples at {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
