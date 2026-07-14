from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import repo_path, write_csv


def latest_successes(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result = record.get("result") or {}
        if record.get("stage") == "fielding" and result:
            latest[str(record["sample_id"])] = result
    return latest


def accepted(result: dict) -> int | None:
    region = result.get("region")
    try:
        region = int(region)
    except (TypeError, ValueError):
        return None
    if region not in {1, 2, 3, 4}:
        return None
    if result.get("decision") != "pass":
        return None
    if result.get("receiving_moment_visible") is not True:
        return None
    if float(result.get("confidence") or 0) < 0.85:
        return None
    return region


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply accepted absolute ground-ball region reviews.")
    parser.add_argument(
        "--review",
        type=Path,
        default=repo_path("reports", "qwen_absolute_ground_region_audit_20260714.jsonl"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    reviews = latest_successes(args.review)
    override_path = repo_path("reports", "manual_region_overrides_20260714.csv")
    overrides: dict[str, int] = {}
    if override_path.exists():
        with override_path.open("r", newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                try:
                    region = int(row.get("region", ""))
                except ValueError:
                    continue
                if region in {1, 2, 3, 4} and row.get("sample_id"):
                    overrides[row["sample_id"]] = region
    changes: list[dict[str, str]] = []
    by_id: dict[str, int] = {}
    for sample_csv in sorted(repo_path("dataset", "ground_ball").glob("*/*/sample.csv")):
        with sample_csv.open("r", newline="", encoding="utf-8-sig") as fh:
            row = next(csv.DictReader(fh))
        sample_id = row["sample_id"]
        if sample_id in overrides:
            region = overrides[sample_id]
            confidence = "manual"
            control_time = ""
            evidence = "user_manual_qc"
        else:
            region = accepted(reviews.get(sample_id, {}))
            confidence = str(reviews.get(sample_id, {}).get("confidence", ""))
            control_time = str(reviews.get(sample_id, {}).get("first_control_time_seconds", ""))
            evidence = str(reviews.get(sample_id, {}).get("evidence", ""))
        if region is None or str(region) == row["region"]:
            continue
        changes.append(
            {
                "sample_id": sample_id,
                "old_region": row["region"],
                "new_region": str(region),
                "confidence": confidence,
                "first_control_time_seconds": control_time,
                "evidence": evidence,
            }
        )
        by_id[sample_id] = region
        if not args.dry_run:
            row["region"] = str(region)
            write_csv(sample_csv, [row], list(row))

    index_path = repo_path("reports", "dataset_index_20260713.csv")
    with index_path.open("r", newline="", encoding="utf-8-sig") as fh:
        index_rows = list(csv.DictReader(fh))
        fields = list(index_rows[0]) if index_rows else []
    if not args.dry_run:
        for row in index_rows:
            if row.get("sample_id") in by_id:
                row["region"] = str(by_id[row["sample_id"]])
        write_csv(index_path, index_rows, fields)
        write_csv(repo_path("reports", "absolute_region_corrections_20260714.csv"), changes, list(changes[0]) if changes else ["sample_id", "old_region", "new_region", "confidence", "first_control_time_seconds", "evidence"])

    print(f"accepted_changes={len(changes)} dry_run={args.dry_run}")
    for change in changes:
        print(f"{change['sample_id']}: {change['old_region']} -> {change['new_region']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
