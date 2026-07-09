from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import repo_path


REQUIRED_FILES = {"video.mp4", "audio.wav", "label.txt", "sample.csv", "source.txt"}
GROUND_FIELDS = ["sample_id", "label", "region", "strength", "bounce", "event_start", "event_end"]
FLY_FIELDS = ["sample_id", "label", "landing_zone", "strength", "trajectory_type", "event_start", "event_end"]


def sample_dirs(dataset_root: Path) -> list[Path]:
    return [p for p in dataset_root.glob("*/*/*") if p.is_dir()]


def validate_sample(path: Path) -> list[str]:
    errors: list[str] = []
    label_folder = path.parents[1].name
    sample_id = path.name
    names = {child.name for child in path.iterdir()}
    missing = REQUIRED_FILES - names
    if missing:
        errors.append(f"missing files: {', '.join(sorted(missing))}")
        return errors

    label_text = (path / "label.txt").read_text(encoding="utf-8").strip()
    if label_text != label_folder:
        errors.append(f"label.txt '{label_text}' does not match folder '{label_folder}'")
    if label_folder == "ground_ball" and not sample_id.startswith("G_"):
        errors.append("ground_ball sample_id must start with G_")
    if label_folder == "fly_ball" and not sample_id.startswith("F_"):
        errors.append("fly_ball sample_id must start with F_")

    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    expected = GROUND_FIELDS if label_folder == "ground_ball" else FLY_FIELDS if label_folder == "fly_ball" else []
    if list(reader.fieldnames or []) != expected:
        errors.append(f"bad csv fields: {reader.fieldnames}; expected {expected}")
    if len(rows) != 1:
        errors.append(f"sample.csv must contain exactly one row, found {len(rows)}")
    elif rows[0].get("sample_id") != sample_id:
        errors.append(f"csv sample_id '{rows[0].get('sample_id')}' does not match folder '{sample_id}'")

    source = (path / "source.txt").read_text(encoding="utf-8")
    if "video_title:" not in source or "video_url:" not in source:
        errors.append("source.txt must include video_title: and video_url:")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final dataset folder and CSV schema.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    args = parser.parse_args()

    failures = 0
    dirs = sample_dirs(args.dataset_root)
    for path in dirs:
        errors = validate_sample(path)
        if errors:
            failures += 1
            print(f"FAIL {path.relative_to(repo_path())}: {'; '.join(errors)}")
    print(f"Checked {len(dirs)} samples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

