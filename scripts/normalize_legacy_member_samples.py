from __future__ import annotations

import argparse
import csv
from pathlib import Path


GROUND_FIELDS = ["sample_id", "label", "region", "strength", "bounce", "event_start", "event_end"]
FLY_FIELDS = ["sample_id", "label", "landing_zone", "strength", "trajectory_type", "event_start", "event_end"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize legacy member sample metadata to the dataset CSV/label format.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    missing_provenance: list[dict[str, str]] = []
    normalized = 0
    for sample_dir in sorted(p for p in args.dataset_root.glob("*/*/*") if p.is_dir()):
        label = sample_dir.parents[1].name
        if label not in {"ground_ball", "fly_ball"} or not (sample_dir / "sample.csv").is_file():
            continue
        fields = GROUND_FIELDS if label == "ground_ball" else FLY_FIELDS
        with (sample_dir / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        if len(rows) != 1:
            raise ValueError(f"{sample_dir}: expected one CSV row, found {len(rows)}")
        row = rows[0]
        row["sample_id"] = sample_dir.name
        row["label"] = label
        if label == "ground_ball":
            row["strength"] = {"L": "low", "M": "medium", "H": "high"}.get(row.get("strength", ""), row.get("strength", ""))
            row["bounce"] = {"Y": "yes", "N": "no"}.get(row.get("bounce", ""), row.get("bounce", ""))
        with (sample_dir / "sample.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in fields})
        (sample_dir / "label.txt").write_text(f"{label}\n", encoding="utf-8")
        normalized += 1

        source_path = sample_dir / "source.txt"
        source = source_path.read_text(encoding="utf-8", errors="replace") if source_path.is_file() else ""
        if "video_title:" not in source or "video_url:" not in source:
            missing_provenance.append({
                "sample_path": sample_dir.as_posix(),
                "collector": sample_dir.parents[0].name,
                "sample_id": sample_dir.name,
                "reason": "legacy source metadata lacks video_title and/or video_url",
            })

    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["sample_path", "collector", "sample_id", "reason"])
        writer.writeheader()
        writer.writerows(missing_provenance)
    print(f"Normalized {normalized} samples; missing provenance={len(missing_provenance)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
