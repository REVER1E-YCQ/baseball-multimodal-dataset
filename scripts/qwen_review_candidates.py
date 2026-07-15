from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

from common import get_env_first, load_jsonl, read_csv, repo_path, write_csv
from qwen_omni_label import model_token_cap, model_token_reserve
from qwen_review_dataset import (
    acquire_process_lock,
    adjudication_evidence_passes,
    audio_transient_candidates,
    combined_usage,
    context_prompt,
    invoke,
    load_review_models,
    semantics_evidence_passes,
    semantics_needs_change,
    successful_records,
    timing_evidence_passes,
    timing_needs_change,
)


SUMMARY_FIELDS = [
    "clip_id",
    "source_id",
    "label",
    "timing_status",
    "semantics_status",
    "adjudication_status",
    "final_status",
    "reviewed_stages",
]


def semantics_core_evidence_passes(result: dict[str, Any]) -> bool:
    """Gate candidate admission on hit class, not on every secondary field."""
    return (
        result.get("decision") in {"pass", "correct"}
        and float(result.get("confidence") or 0) >= 0.85
        and result.get("verified_label") in {"ground_ball", "fly_ball"}
        and result.get("verified_strength") in {"low", "medium", "high"}
    )


def adjudication_agrees_with_semantics(semantics: dict[str, Any], adjudication: dict[str, Any]) -> bool:
    """Require agreement on the core hit class; secondary fields are reconciled separately."""
    if not adjudication:
        return True
    label = semantics.get("verified_label")
    return label in {"ground_ball", "fly_ball"} and adjudication.get("label") == label


def latest_labels(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        clip_id = record.get("clip_id")
        if clip_id:
            records[clip_id] = record
    return records


def flat_label_row(clip_id: str, payload: dict[str, Any]) -> dict[str, str]:
    row = {
        "sample_id": clip_id,
        "label": str(payload.get("label", "")),
        "strength": "",
        "event_start": str(payload.get("event_start", "")),
        "event_end": str(payload.get("event_end", "")),
    }
    if row["label"] == "ground_ball":
        ground = payload.get("ground_ball") or {}
        row.update(region=str(ground.get("region", "")), strength=str(ground.get("strength", "")), bounce=str(ground.get("bounce", "")))
    elif row["label"] == "fly_ball":
        fly = payload.get("fly_ball") or {}
        row.update(
            landing_zone=str(fly.get("landing_zone", "")),
            strength=str(fly.get("strength", "")),
            trajectory_type=str(fly.get("trajectory_type", "")),
        )
    return row


def materialized_sources() -> set[str]:
    result: set[str] = set()
    for path in repo_path("dataset").glob("*/*/*/source.txt"):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("source_id:"):
                result.add(line.split(":", 1)[1].strip())
    return result


def write_summary(
    items: list[tuple[dict[str, str], dict[str, Any]]],
    records: dict[tuple[str, str], dict[str, Any]],
    output: Path,
) -> None:
    rows: list[dict[str, str]] = []
    for clip, label_record in items:
        clip_id = clip["clip_id"]
        row = flat_label_row(clip_id, label_record.get("label") or {})
        results = {
            stage: (records.get((clip_id, stage)) or {}).get("result") or {}
            for stage in ("timing", "semantics", "adjudication")
        }
        timing = results["timing"]
        semantics = results["semantics"]
        adjudication = results["adjudication"]
        completed = [stage for stage, result in results.items() if result]
        foundations = timing_evidence_passes(timing) and semantics_core_evidence_passes(semantics)
        if adjudication:
            # Audio decides contact timing.  A later adjudication pass may
            # not see the swing frame in a delayed broadcast cut; do not let
            # that visual-only failure override clear audio timing plus video
            # semantic evidence from the dedicated stages.
            final = "auto_accepted" if foundations and (
                (adjudication_evidence_passes(adjudication) and adjudication_agrees_with_semantics(semantics, adjudication))
                or adjudication.get("decision") == "reject"
            ) else "manual_review"
        elif timing and semantics and foundations and not timing_needs_change(row, timing) and not semantics_needs_change(row, semantics):
            final = "auto_accepted"
        else:
            final = "incomplete" if not timing or not semantics else "manual_review"
        rows.append(
            {
                "clip_id": clip_id,
                "source_id": clip.get("source_id", ""),
                "label": row.get("label", ""),
                "timing_status": str(timing.get("decision", "missing")),
                "semantics_status": str(semantics.get("decision", "missing")),
                "adjudication_status": str(adjudication.get("decision", "not_required" if timing and semantics else "missing")),
                "final_status": final,
                "reviewed_stages": ",".join(completed),
            }
        )
    write_csv(output, rows, SUMMARY_FIELDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged Qwen review before candidate materialization.")
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels_refined.jsonl"))
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--sources-manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--output", type=Path, default=repo_path("reports", "qwen_candidate_review.jsonl"))
    parser.add_argument("--summary", type=Path, default=repo_path("reports", "qwen_candidate_review_summary.csv"))
    parser.add_argument("--statuses", default="labeled")
    parser.add_argument("--clip-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--include-materialized", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    labels = latest_labels(args.labels)
    wanted_statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    wanted_clips = set(args.clip_id)
    existing_sources = materialized_sources() if not args.include_materialized else set()
    items: list[tuple[dict[str, str], dict[str, Any]]] = []
    for clip in read_csv(args.clips_manifest):
        if clip.get("status") not in wanted_statuses or clip.get("clip_id") not in labels:
            continue
        if wanted_clips and clip.get("clip_id") not in wanted_clips:
            continue
        if clip.get("source_id") in existing_sources:
            continue
        items.append((clip, labels[clip["clip_id"]]))
    if args.limit:
        items = items[: args.limit]

    records = successful_records(args.output)
    pending = sum((clip["clip_id"], stage) not in records for clip, _label in items for stage in ("timing", "semantics"))
    if args.dry_run:
        print(f"candidates={len(items)} minimum_pending_calls={pending} resumable_output={args.output}")
        return 0

    api_key = get_env_first(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    if not api_key:
        raise SystemExit("Set QWEN_API_KEY or DASHSCOPE_API_KEY before review.")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    models = load_review_models()
    usage_logs = [repo_path("reports", "qwen_labels.jsonl"), *repo_path("reports").glob("qwen_*review*.jsonl")]
    usage_totals = combined_usage(list(dict.fromkeys(usage_logs)))
    cap = model_token_cap()
    reserve = model_token_reserve()
    blocked: set[str] = set()
    lock = acquire_process_lock(args.output.with_suffix(".lock"))
    prompts = {
        "timing": repo_path("prompts", "contact_timing_review_prompt.md").read_text(encoding="utf-8"),
        "semantics": repo_path("prompts", "field_semantics_review_prompt.md").read_text(encoding="utf-8"),
        "adjudication": repo_path("prompts", "review_adjudication_prompt.md").read_text(encoding="utf-8"),
    }
    sources = {row["source_id"]: row for row in read_csv(args.sources_manifest) if row.get("source_id")}

    for index, (clip, label_record) in enumerate(items, 1):
        clip_id = clip["clip_id"]
        row = flat_label_row(clip_id, label_record.get("label") or {})
        video = Path(clip["clip_path"])
        audio = Path(clip["audio_path"])
        if not video.is_absolute():
            video = repo_path(str(video))
        if not audio.is_absolute():
            audio = repo_path(str(audio))
        source = json.dumps({"clip": clip, "source": sources.get(clip.get("source_id", ""), {})}, ensure_ascii=False)
        candidates = audio_transient_candidates(audio)

        timing_record = records.get((clip_id, "timing"))
        if timing_record is None:
            timing_record = invoke(clip_id, "timing", video, context_prompt(prompts["timing"], row, source, audio_candidates=candidates), args.output, models, usage_totals, blocked, api_key, base_url, cap, reserve)
            if timing_record:
                records[(clip_id, "timing")] = timing_record
        if timing_record is None:
            continue
        timing = timing_record["result"]

        semantics_record = records.get((clip_id, "semantics"))
        if semantics_record is None:
            semantics_record = invoke(clip_id, "semantics", video, context_prompt(prompts["semantics"], row, source, timing), args.output, models, usage_totals, blocked, api_key, base_url, cap, reserve)
            if semantics_record:
                records[(clip_id, "semantics")] = semantics_record
        if semantics_record is None:
            continue
        semantics = semantics_record["result"]

        needs_adjudication = timing_needs_change(row, timing) or semantics_needs_change(row, semantics) or min(float(timing.get("confidence") or 0), float(semantics.get("confidence") or 0)) < 0.85
        if needs_adjudication and (clip_id, "adjudication") not in records:
            prior = {"timing": timing, "semantics": semantics}
            adjudication = invoke(clip_id, "adjudication", video, context_prompt(prompts["adjudication"], row, source, prior), args.output, models, usage_totals, blocked, api_key, base_url, cap, reserve)
            if adjudication:
                records[(clip_id, "adjudication")] = adjudication
        print(f"{index}/{len(items)} {clip_id}: reviewed")
        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            write_summary(items, records, args.summary)

    write_summary(items, records, args.summary)
    lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
