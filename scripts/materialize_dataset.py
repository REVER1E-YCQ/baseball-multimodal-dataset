from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Any

from common import load_jsonl, read_csv, repo_path


def next_sample_id(label: str, collector_dir: Path) -> str:
    prefix = "G" if label == "ground_ball" else "F"
    max_seen = 0
    if collector_dir.exists():
        for child in collector_dir.iterdir():
            if child.is_dir() and child.name.startswith(prefix + "_"):
                try:
                    max_seen = max(max_seen, int(child.name.split("_", 1)[1]))
                except ValueError:
                    pass
    return f"{prefix}_{max_seen + 1:03d}"


def write_sample_csv(path: Path, sample_id: str, label_payload: dict[str, Any]) -> None:
    label = label_payload["label"]
    if label == "ground_ball":
        fields = ["sample_id", "label", "region", "strength", "bounce", "event_start", "event_end"]
        gb = label_payload.get("ground_ball") or {}
        row = {
            "sample_id": sample_id,
            "label": label,
            "region": gb.get("region", ""),
            "strength": gb.get("strength", ""),
            "bounce": gb.get("bounce", ""),
            "event_start": f"{float(label_payload['event_start']):.3f}",
            "event_end": f"{float(label_payload['event_end']):.3f}",
        }
    else:
        fields = ["sample_id", "label", "landing_zone", "strength", "trajectory_type", "event_start", "event_end"]
        fb = label_payload.get("fly_ball") or {}
        row = {
            "sample_id": sample_id,
            "label": label,
            "landing_zone": fb.get("landing_zone", ""),
            "strength": fb.get("strength", ""),
            "trajectory_type": fb.get("trajectory_type", ""),
            "event_start": f"{float(label_payload['event_start']):.3f}",
            "event_end": f"{float(label_payload['event_end']):.3f}",
        }
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)


def source_text(clip_row: dict[str, str], source_rows: dict[str, dict[str, str]]) -> str:
    source = source_rows.get(clip_row.get("source_id", ""), {})
    return "\n".join(
        [
            f"video_title: {source.get('video_title') or source.get('event_text') or clip_row.get('clip_id', '')}",
            f"video_url: {source.get('source_url', '')}",
            f"source_id: {clip_row.get('source_id', '')}",
            f"clip_id: {clip_row.get('clip_id', '')}",
            f"source_path: {clip_row.get('source_path', '')}",
            f"clip_start_time: {clip_row.get('start_time', '')}",
            f"clip_end_time: {clip_row.get('end_time', '')}",
        ]
    ) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy accepted Qwen labels into dataset folders.")
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels.jsonl"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--sources-manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--collector", default="Codex_Workstation")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clip_rows = {row["clip_id"]: row for row in read_csv(args.clips_manifest)}
    source_rows = {row["source_id"]: row for row in read_csv(args.sources_manifest) if row.get("source_id")}
    records = load_jsonl(args.labels)
    created = 0

    for record in records:
        label_payload = record.get("label") or {}
        label = label_payload.get("label")
        if label not in {"ground_ball", "fly_ball"}:
            continue
        if float(label_payload.get("confidence") or 0.0) < args.min_confidence:
            continue
        clip_row = clip_rows.get(record.get("clip_id", ""))
        if not clip_row:
            continue

        collector_dir = repo_path("dataset", label, args.collector)
        sample_id = next_sample_id(label, collector_dir)
        out_dir = collector_dir / sample_id
        if out_dir.exists():
            continue
        if args.dry_run:
            print(f"Would create {out_dir}")
            created += 1
            continue

        out_dir.mkdir(parents=True, exist_ok=False)
        clip_path = repo_path(clip_row["clip_path"])
        audio_path = repo_path(clip_row["audio_path"])
        shutil.copy2(clip_path, out_dir / "video.mp4")
        shutil.copy2(audio_path, out_dir / "audio.wav")
        (out_dir / "label.txt").write_text(label + "\n", encoding="utf-8")
        write_sample_csv(out_dir / "sample.csv", sample_id, label_payload)
        (out_dir / "source.txt").write_text(source_text(clip_row, source_rows), encoding="utf-8")
        created += 1
        print(f"Created {out_dir.relative_to(repo_path())}")

    print(f"Materialized {created} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

