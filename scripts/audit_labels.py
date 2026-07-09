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


def audit(path: Path) -> list[str]:
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
    if event_end - event_start > 0.200:
        errors.append("event interval should bracket contact only, not the whole play")

    if label == "ground_ball":
        if row.get("strength") not in VALID_STRENGTH:
            errors.append("invalid ground_ball strength")
        if row.get("bounce") not in VALID_BOUNCE:
            errors.append("invalid bounce")
        try:
            region = int(row.get("region", ""))
            if region < 1 or region > 4:
                errors.append("region must be 1-4")
        except ValueError:
            errors.append("region must be integer 1-4")
    elif label == "fly_ball":
        if row.get("strength") not in VALID_STRENGTH:
            errors.append("invalid fly_ball strength")
        if row.get("trajectory_type") not in VALID_TRAJECTORY:
            errors.append("invalid trajectory_type")
        try:
            zone = int(row.get("landing_zone", ""))
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
    args = parser.parse_args()

    failures = 0
    dirs = sample_dirs(args.dataset_root)
    for path in dirs:
        errors = audit(path)
        if errors:
            failures += 1
            print(f"FAIL {path.relative_to(repo_path())}: {'; '.join(errors)}")
    print(f"Checked {len(dirs)} samples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

