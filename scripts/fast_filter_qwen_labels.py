from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Select high-confidence one-pass Qwen labels for audio-gated materialization.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review-summary", type=Path)
    parser.add_argument("--reconciled", type=Path)
    parser.add_argument("--sources-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-confidence", type=float, default=0.78)
    args = parser.parse_args()

    review_status: dict[str, str] = {}
    if args.review_summary and args.review_summary.exists():
        with args.review_summary.open(encoding="utf-8-sig", newline="") as fh:
            review_status = {row["clip_id"]: row["final_status"] for row in csv.DictReader(fh)}
    reconciled = {row["clip_id"]: row for row in load_jsonl(args.reconciled)} if args.reconciled else {}
    expected_labels: dict[str, str] = {}
    if args.sources_manifest and args.sources_manifest.exists():
        with args.sources_manifest.open(encoding="utf-8-sig", newline="") as fh:
            expected_labels = {row["source_id"]: row.get("expected_label", "") for row in csv.DictReader(fh)}

    selected: list[dict] = []
    reasons: dict[str, int] = {}
    for record in load_jsonl(args.labels):
        clip_id = record.get("clip_id", "")
        status = review_status.get(clip_id, "incomplete")
        if status == "manual_review":
            reasons["review_rejected"] = reasons.get("review_rejected", 0) + 1
            continue
        if status == "auto_accepted" and clip_id in reconciled:
            chosen = reconciled[clip_id]
        else:
            chosen = record
        label = chosen.get("label") or {}
        try:
            start = float(label.get("event_start"))
            end = float(label.get("event_end"))
            confidence = float(label.get("confidence") or 0)
        except (TypeError, ValueError):
            reasons["bad_fields"] = reasons.get("bad_fields", 0) + 1
            continue
        if label.get("label") not in {"ground_ball", "fly_ball"}:
            reasons["bad_label"] = reasons.get("bad_label", 0) + 1
            continue
        expected = expected_labels.get(chosen.get("source_id", ""), "")
        if expected in {"ground_ball", "fly_ball"} and expected != label.get("label"):
            reasons["source_label_conflict"] = reasons.get("source_label_conflict", 0) + 1
            continue
        if confidence < args.min_confidence:
            reasons["low_confidence"] = reasons.get("low_confidence", 0) + 1
            continue
        if label.get("contact_sound_clear") is not True:
            reasons["contact_not_clear"] = reasons.get("contact_not_clear", 0) + 1
            continue
        if not (0 <= start < end and end - start <= 0.2000001):
            reasons["bad_event_interval"] = reasons.get("bad_event_interval", 0) + 1
            continue
        chosen["fast_path"] = True
        selected.append(chosen)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        for record in selected:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Selected {len(selected)} of {sum(reasons.values()) + len(selected)} records; rejected={reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
