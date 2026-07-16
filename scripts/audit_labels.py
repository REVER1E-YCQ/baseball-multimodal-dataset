from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import repo_path


VALID_STRENGTH = {"low", "medium", "high"}
VALID_BOUNCE = {"yes", "no"}
VALID_TRAJECTORY = {"fly", "line_drive", "pop_fly"}


def sample_dirs(dataset_root: Path) -> list[Path]:
    return [p for p in dataset_root.glob("*/*/*") if p.is_dir()]


def read_row(path: Path) -> dict[str, str]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        return next(csv.DictReader(fh))


def audit(path: Path, allow_pending_location: bool = False) -> list[str]:
    errors: list[str] = []
    row = read_row(path)
    label = row.get("label", "")
    try:
        event_start = float(row.get("event_start", "nan"))
        event_end = float(row.get("event_end", "nan"))
    except ValueError:
        errors.append("event_start/event_end must be floats")
        return errors
    if not (0 <= event_start < event_end):
        errors.append("event interval must satisfy 0 <= event_start < event_end")
    if event_end - event_start > 0.200 + 1e-9:
        errors.append("event interval should bracket contact only, not the whole play")

    if label == "ground_ball":
        if row.get("strength") not in VALID_STRENGTH:
            errors.append("invalid ground_ball strength")
        if row.get("bounce") not in VALID_BOUNCE:
            errors.append("invalid bounce")
        region_value = row.get("region", "")
        if not (allow_pending_location and region_value == "pending"):
            try:
                region = int(region_value)
                if region < 1 or region > 4:
                    errors.append("region must be 1-4")
            except ValueError:
                errors.append("region must be integer 1-4")
    elif label == "fly_ball":
        if row.get("strength") not in VALID_STRENGTH:
            errors.append("invalid fly_ball strength")
        if row.get("trajectory_type") not in VALID_TRAJECTORY:
            errors.append("invalid trajectory_type")
        zone_value = row.get("landing_zone", "")
        if not (allow_pending_location and zone_value == "pending"):
            try:
                zone = int(zone_value)
                if zone < 1 or zone > 9:
                    errors.append("landing_zone must be 1-9")
            except ValueError:
                errors.append("landing_zone must be integer 1-9")
    else:
        errors.append(f"invalid label: {label}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit final labels for value ranges and timing sanity.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument(
        "--allow-pending-location",
        action="store_true",
        help="Allow region/landing_zone=pending during the pre-2000 collection phase.",
    )
    args = parser.parse_args()

    failures = 0
    dirs = sample_dirs(args.dataset_root)
    for path in dirs:
        errors = audit(path, allow_pending_location=args.allow_pending_location)
        if errors:
            failures += 1
            print(f"FAIL {path.relative_to(args.dataset_root.parent)}: {'; '.join(errors)}")
    print(f"Checked {len(dirs)} samples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
