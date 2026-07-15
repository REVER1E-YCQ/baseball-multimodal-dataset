from __future__ import annotations

import argparse
from pathlib import Path

from common import ffprobe_duration, repo_path, require_tool


def sample_dirs(dataset_root: Path) -> list[Path]:
    return [p for p in dataset_root.glob("*/*/*") if p.is_dir()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate media readability and duration agreement.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--max-delta", type=float, default=0.25)
    args = parser.parse_args()

    require_tool("ffprobe")
    failures = 0
    dirs = sample_dirs(args.dataset_root)
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
