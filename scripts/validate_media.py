from __future__ import annotations

import argparse
from pathlib import Path

from common import ffprobe_duration, repo_path, require_tool


def sample_dirs(dataset_root: Path) -> list[Path]:
    return [p for p in dataset_root.glob("*/*/*") if p.is_dir()]


def parse_id_mins(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    if not value:
        return result
    for item in value.split(","):
        prefix, separator, number = item.strip().partition("=")
        if separator != "=" or prefix not in {"F", "G"} or not number.isdigit():
            raise SystemExit("--id-min entries must look like F=104,G=161")
        result[prefix] = int(number)
    return result


def filter_sample_dirs(dirs: list[Path], id_mins: dict[str, int]) -> list[Path]:
    if not id_mins:
        return dirs
    selected: list[Path] = []
    for path in dirs:
        prefix, separator, number = path.name.partition("_")
        if separator != "_" or not number.isdigit():
            continue
        if prefix in id_mins and int(number) >= id_mins[prefix]:
            selected.append(path)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate media readability and duration agreement.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--max-delta", type=float, default=0.25)
    parser.add_argument("--id-min", default="", help="Only check IDs at or above thresholds, e.g. F=104,G=161.")
    args = parser.parse_args()

    require_tool("ffprobe")
    failures = 0
    dirs = filter_sample_dirs(sample_dirs(args.dataset_root), parse_id_mins(args.id_min))
    for path in dirs:
        video = path / "video.mp4"
        audio = path / "audio.wav"
        v_duration = ffprobe_duration(video)
        a_duration = ffprobe_duration(audio)
        if v_duration is None or a_duration is None:
            print(f"FAIL {path.relative_to(args.dataset_root.parent)}: unreadable media")
            failures += 1
            continue
        if abs(v_duration - a_duration) > args.max_delta:
            print(
                f"FAIL {path.relative_to(args.dataset_root.parent)}: duration mismatch "
                f"video={v_duration:.3f}s audio={a_duration:.3f}s"
            )
            failures += 1
    print(f"Checked {len(dirs)} samples; failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
