from __future__ import annotations

import argparse
import csv
import html
import subprocess
from pathlib import Path
from typing import Any

from common import ffprobe_duration, repo_path, tool_path
from qwen_review_dataset import successful_records


def read_sample(path: Path) -> dict[str, str]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        return next(csv.DictReader(fh))


def samples_by_id(root: Path) -> dict[str, Path]:
    return {path.name: path for path in root.glob("*/*/*") if (path / "sample.csv").exists()}


def run_sheet(video: Path, output: Path, start: float | None, duration: float | None, fps: int, tile: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [tool_path("ffmpeg"), "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd.extend(["-ss", f"{start:.3f}"])
    if duration is not None:
        cmd.extend(["-t", f"{duration:.3f}"])
    cmd.extend(
        [
            "-i",
            str(video),
            "-vf",
            f"fps={fps},scale=320:-1,tile={tile}:padding=2:margin=2",
            "-frames:v",
            "1",
            str(output),
        ]
    )
    subprocess.run(cmd, check=True)


def result_text(result: dict[str, Any]) -> str:
    keys = [
        "decision",
        "confidence",
        "corrected_event_start",
        "corrected_event_end",
        "visual_evidence",
        "audio_evidence",
        "failure_reason",
    ]
    return "\n".join(f"{key}: {result.get(key, '')}" for key in keys)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build frame-by-frame evidence for unresolved dataset reviews.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--review", type=Path, default=repo_path("reports", "qwen_dataset_review.jsonl"))
    parser.add_argument("--reconciliation", type=Path, default=repo_path("reports", "qwen_dataset_reconciliation.csv"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("reports", "manual_visual_full_audit"))
    parser.add_argument("--statuses", default="manual_review")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    wanted = {item.strip() for item in args.statuses.split(",") if item.strip()}
    with args.reconciliation.open("r", newline="", encoding="utf-8-sig") as fh:
        queue = [row for row in csv.DictReader(fh) if row.get("status") in wanted]
    paths = samples_by_id(args.dataset_root.resolve())
    records = successful_records(args.review)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []

    for row in queue:
        sample_id = row["sample_id"]
        path = paths.get(sample_id)
        if path is None:
            continue
        sample = read_sample(path)
        timing = (records.get((sample_id, "timing")) or {}).get("result") or {}
        current_mid = (float(sample["event_start"]) + float(sample["event_end"])) / 2.0
        try:
            proposed_mid = (float(timing["corrected_event_start"]) + float(timing["corrected_event_end"])) / 2.0
        except (KeyError, TypeError, ValueError):
            proposed_mid = current_mid
        sample_dir = args.output_dir / sample_id
        full = sample_dir / "full_sheet.png"
        current = sample_dir / "current_event_25fps.png"
        proposed = sample_dir / "proposed_event_25fps.png"
        video = path / "video.mp4"
        if args.force or not full.exists():
            duration = min(ffprobe_duration(video) or 7.0, 8.0)
            run_sheet(video, full, 0.0, duration, 5, "8x5")
        if args.force or not current.exists():
            run_sheet(video, current, max(0.0, current_mid - 0.4), 0.8, 25, "10x2")
        if args.force or not proposed.exists():
            run_sheet(video, proposed, max(0.0, proposed_mid - 0.4), 0.8, 25, "10x2")
        rel_full = full.relative_to(args.output_dir).as_posix()
        rel_current = current.relative_to(args.output_dir).as_posix()
        rel_proposed = proposed.relative_to(args.output_dir).as_posix()
        cards.append(
            f"""
<section>
  <h2>{html.escape(sample_id)}: {html.escape(row.get('reason', ''))}</h2>
  <p>current={sample['event_start']}-{sample['event_end']} proposed={timing.get('corrected_event_start', '')}-{timing.get('corrected_event_end', '')} audio_ratio={html.escape(row.get('audio_ratio', ''))}</p>
  <video controls preload="metadata" src="{video.resolve().as_uri()}"></video>
  <audio controls preload="metadata" src="{(path / 'audio.wav').resolve().as_uri()}"></audio>
  <h3>Whole clip, 5 fps</h3><img src="{rel_full}" alt="whole clip frames">
  <h3>Current interval neighborhood, 25 fps</h3><img src="{rel_current}" alt="current event frames">
  <h3>Proposed interval neighborhood, 25 fps</h3><img src="{rel_proposed}" alt="proposed event frames">
  <pre>{html.escape(result_text(timing))}</pre>
</section>
"""
        )

    page = """<!doctype html><html><head><meta charset="utf-8"><title>Full visual audit queue</title>
<style>body{font-family:Arial,sans-serif;margin:24px;background:#f5f5f2;color:#171717}section{border-top:2px solid #222;padding:20px 0}video{width:min(960px,100%);display:block}audio{width:min(960px,100%);margin:8px 0}img{max-width:100%;display:block;margin:8px 0;background:#111}pre{white-space:pre-wrap}</style>
</head><body><h1>Full visual audit queue</h1>""" + "\n".join(cards) + "</body></html>\n"
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"wrote {len(cards)} samples to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
