from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import load_jsonl, read_csv, repo_path


def latest_by_clip(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        clip_id = record.get("clip_id")
        if clip_id:
            latest[clip_id] = record
    return latest


def audit_record(
    record: dict[str, Any],
    clip_row: dict[str, str],
    min_confidence: float,
    max_event_width: float,
    min_precontact: float,
) -> list[str]:
    errors: list[str] = []
    payload = record.get("label") or {}
    label = payload.get("label")
    expected = clip_row.get("expected_label")
    confidence = float(payload.get("confidence") or 0.0)
    event_start = float(payload.get("event_start") or 0.0)
    event_end = float(payload.get("event_end") or 0.0)

    if record.get("error"):
        errors.append(f"model_error={record['error']}")
    if label not in {"ground_ball", "fly_ball"}:
        errors.append(f"not_accepted_label={label}")
    if confidence < min_confidence:
        errors.append(f"low_confidence={confidence:.2f}")
    if expected in {"ground_ball", "fly_ball"} and label in {"ground_ball", "fly_ball"} and expected != label:
        errors.append(f"expected_label_conflict={expected}!={label}")
    if not (0 <= event_start < event_end):
        errors.append("bad_event_interval")
    elif event_end - event_start > max_event_width + 1e-9:
        errors.append(f"wide_event_interval={event_end - event_start:.3f}")
    if label in {"ground_ball", "fly_ball"} and event_start < min_precontact:
        errors.append(f"insufficient_precontact={event_start:.3f}")
    if payload.get("contact_sound_clear") is not True:
        errors.append("contact_sound_not_clear")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit latest Qwen labels before materializing dataset samples.")
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels.jsonl"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--min-confidence", type=float, default=0.70)
    parser.add_argument("--max-event-width", type=float, default=0.200)
    parser.add_argument("--min-precontact", type=float, default=0.500)
    parser.add_argument("--output", type=Path, default=repo_path("reports", "qwen_label_audit.jsonl"))
    args = parser.parse_args()

    records = latest_by_clip(load_jsonl(args.labels))
    clip_rows = {row["clip_id"]: row for row in read_csv(args.clips_manifest)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    checked = 0
    with args.output.open("w", encoding="utf-8") as fh:
        for clip_id, record in records.items():
            clip_row = clip_rows.get(clip_id)
            if not clip_row:
                continue
            checked += 1
            errors = audit_record(record, clip_row, args.min_confidence, args.max_event_width, args.min_precontact)
            status = "pass" if not errors else "review"
            if errors:
                failures += 1
            fh.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "status": status,
                        "errors": errors,
                        "expected_label": clip_row.get("expected_label", ""),
                        "model_label": (record.get("label") or {}).get("label", ""),
                        "confidence": (record.get("label") or {}).get("confidence", 0),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"{status.upper()} {clip_id}" + (f": {'; '.join(errors)}" if errors else ""))
    print(f"Checked {checked} labels; review={failures}; wrote {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
