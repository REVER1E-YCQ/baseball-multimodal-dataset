from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

from common import load_jsonl, repo_path
from qwen_review_dataset import region_evidence_consistent, substantive_evidence, successful_records


def latest_labels(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        if record.get("clip_id"):
            result[str(record["clip_id"])] = record
    return result


def accepted_ids(path: Path) -> set[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return {
            row["clip_id"]
            for row in csv.DictReader(fh)
            if row.get("clip_id") and row.get("final_status") == "auto_accepted"
        }


def payload_from_adjudication(original: dict[str, Any], semantics: dict[str, Any], adjudication: dict[str, Any]) -> dict[str, Any] | None:
    label = adjudication.get("label")
    try:
        start = float(adjudication["event_start"])
        end = float(adjudication["event_end"])
    except (KeyError, TypeError, ValueError):
        return None
    if label not in {"ground_ball", "fly_ball"} or semantics.get("verified_label") != label or not (0 <= start < end <= start + 0.200001):
        return None
    payload: dict[str, Any] = {
        "label": label,
        "confidence": min(float(adjudication.get("confidence") or 0.0), float(semantics.get("confidence") or 0.0)),
        "contact_sound_clear": adjudication.get("contact_audible") is True,
        "event_start": round(start, 3),
        "event_end": round(end, 3),
        "review_reconciled": True,
    }
    strength = semantics.get("verified_strength")
    if strength not in {"low", "medium", "high"}:
        return None
    if label == "ground_ball":
        original_ground = original.get("ground_ball") or {}
        ground = semantics.get("ground_ball") or {}
        region = original_ground.get("region")
        if ground.get("region_verified") is True and region_evidence_consistent(ground) and substantive_evidence(ground.get("region_evidence")):
            region = ground.get("region")
        bounce = original_ground.get("bounce")
        if ground.get("bounce") in {"yes", "no"} and ground.get("receiving_moment_visible") is True and ground.get("knee_reference_visible") is True and substantive_evidence(ground.get("receiving_height_evidence")):
            bounce = ground.get("bounce")
        if str(region) not in {"1", "2", "3", "4"} or bounce not in {"yes", "no"}:
            return None
        payload["ground_ball"] = {
            "region": int(region),
            "strength": strength,
            "bounce": bounce,
        }
    else:
        original_fly = original.get("fly_ball") or {}
        fly = semantics.get("fly_ball") or {}
        landing_zone = fly.get("landing_zone") if fly.get("landing_zone_verified") is True and substantive_evidence(fly.get("flight_evidence")) else original_fly.get("landing_zone")
        trajectory = fly.get("trajectory_type") if substantive_evidence(fly.get("flight_evidence")) else original_fly.get("trajectory_type")
        if str(landing_zone) not in {str(value) for value in range(1, 10)} or trajectory not in {"fly", "line_drive", "pop_fly"}:
            return None
        payload["fly_ball"] = {
            "landing_zone": int(landing_zone),
            "strength": strength,
            "trajectory_type": trajectory,
        }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Write review-adjudicated labels for auto-accepted candidate clips.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    labels = latest_labels(args.labels)
    reviews = successful_records(args.review)
    output: list[dict[str, Any]] = []
    for clip_id in sorted(accepted_ids(args.summary)):
        original = labels.get(clip_id)
        semantics = (reviews.get((clip_id, "semantics")) or {}).get("result") or {}
        adjudication = (reviews.get((clip_id, "adjudication")) or {}).get("result") or {}
        payload = payload_from_adjudication((original or {}).get("label") or {}, semantics, adjudication)
        if not original or not payload:
            continue
        record = copy.deepcopy(original)
        record["label"] = payload
        record["reconciled_from"] = "qwen_candidate_review_adjudication"
        output.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in output:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"Reconciled {len(output)} auto-accepted candidate labels; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
