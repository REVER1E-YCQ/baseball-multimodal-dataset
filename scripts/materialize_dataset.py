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


def write_sample_csv(
    path: Path,
    sample_id: str,
    label_payload: dict[str, Any],
    defer_position: bool = False,
) -> None:
    label = label_payload["label"]
    if label == "ground_ball":
        fields = ["sample_id", "label", "region", "strength", "bounce", "event_start", "event_end"]
        gb = label_payload.get("ground_ball") or {}
        row = {
            "sample_id": sample_id,
            "label": label,
            "region": "pending" if defer_position else gb.get("region", ""),
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
            "landing_zone": "pending" if defer_position else fb.get("landing_zone", ""),
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


def latest_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        clip_id = record.get("clip_id")
        if clip_id:
            latest[clip_id] = record
    return latest


def load_audit_passes(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    passes: set[str] = set()
    for row in load_jsonl(path):
        if row.get("status") == "pass" and row.get("clip_id"):
            passes.add(row["clip_id"])
    return passes


def load_visual_passes(path: Path) -> set[str] | None:
    if not path.exists():
        return None
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return {row["clip_id"] for row in csv.DictReader(fh) if row.get("final_status") == "auto_accepted"}


def candidate_score(record: dict[str, Any], clip_row: dict[str, str]) -> tuple[float, float, float, str]:
    payload = record.get("label") or {}
    start = float(payload.get("event_start") or 0.0)
    end = float(payload.get("event_end") or 0.0)
    duration = max(0.0, float(clip_row.get("end_time") or 0.0) - float(clip_row.get("start_time") or 0.0))
    boundary_margin = min(start, max(0.0, duration - end))
    return (
        float(payload.get("confidence") or 0.0),
        boundary_margin,
        -abs((start + end) / 2.0 - 2.0),
        str(record.get("clip_id", "")),
    )


def existing_materialized_ids(dataset_root: Path) -> tuple[set[str], set[str]]:
    clip_ids: set[str] = set()
    source_ids: set[str] = set()
    for source_file in dataset_root.glob("*/*/*/source.txt"):
        try:
            for line in source_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("clip_id:"):
                    clip_id = line.split(":", 1)[1].strip()
                    if clip_id:
                        clip_ids.add(clip_id)
                if line.startswith("source_id:"):
                    source_id = line.split(":", 1)[1].strip()
                    if source_id:
                        source_ids.add(source_id)
        except OSError:
            continue
    return clip_ids, source_ids


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy accepted Qwen labels into dataset folders.")
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels.jsonl"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--sources-manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--collector", default="Codex_Workstation")
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--audit", type=Path, default=repo_path("reports", "qwen_label_audit.jsonl"))
    parser.add_argument("--require-audit-pass", action="store_true")
    parser.add_argument("--visual-audit", type=Path, default=repo_path("reports", "qwen_candidate_review_summary.csv"))
    parser.add_argument("--require-visual-audit-pass", action="store_true")
    parser.add_argument("--accepted-clip-statuses", default="labeled")
    parser.add_argument(
        "--defer-position",
        action="store_true",
        help="Write region/landing_zone as pending for later position-only backfill.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clip_rows = {row["clip_id"]: row for row in read_csv(args.clips_manifest)}
    accepted_statuses = {item.strip() for item in args.accepted_clip_statuses.split(",") if item.strip()}
    source_rows = {row["source_id"]: row for row in read_csv(args.sources_manifest) if row.get("source_id")}
    records = latest_records(load_jsonl(args.labels)).values()
    audit_passes = load_audit_passes(args.audit)
    visual_passes = load_visual_passes(args.visual_audit)
    already_materialized_clips, already_materialized_sources = existing_materialized_ids(repo_path("dataset"))
    created = 0

    eligible_by_source: dict[str, list[tuple[dict[str, Any], dict[str, str]]]] = {}
    for record in records:
        if record.get("clip_id") in already_materialized_clips:
            continue
        if args.require_audit_pass and (audit_passes is None or record.get("clip_id") not in audit_passes):
            continue
        if args.require_visual_audit_pass and (visual_passes is None or record.get("clip_id") not in visual_passes):
            continue
        label_payload = record.get("label") or {}
        label = label_payload.get("label")
        if label not in {"ground_ball", "fly_ball"}:
            continue
        if float(label_payload.get("confidence") or 0.0) < args.min_confidence:
            continue
        clip_row = clip_rows.get(record.get("clip_id", ""))
        if not clip_row:
            continue
        if accepted_statuses and clip_row.get("status") not in accepted_statuses:
            continue
        if clip_row.get("source_id") in already_materialized_sources:
            continue

        eligible_by_source.setdefault(clip_row.get("source_id", ""), []).append((record, clip_row))

    selected = [max(candidates, key=lambda item: candidate_score(item[0], item[1])) for candidates in eligible_by_source.values()]
    for record, clip_row in sorted(selected, key=lambda item: item[1].get("source_id", "")):
        label_payload = record.get("label") or {}
        label = label_payload.get("label")

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
        write_sample_csv(out_dir / "sample.csv", sample_id, label_payload, args.defer_position)
        (out_dir / "source.txt").write_text(source_text(clip_row, source_rows), encoding="utf-8")
        created += 1
        already_materialized_clips.add(record.get("clip_id", ""))
        already_materialized_sources.add(clip_row.get("source_id", ""))
        print(f"Created {out_dir.relative_to(repo_path())}")

    print(f"Materialized {created} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
