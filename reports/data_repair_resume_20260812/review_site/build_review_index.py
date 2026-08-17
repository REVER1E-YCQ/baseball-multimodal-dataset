from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path


REPORT_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = REPORT_DIR.parents[1]
FIRST_PASS_ROOT = WORKSPACE_ROOT.parent / "baseball-ai-first-pass-20260803"

READY_CSV = REPORT_DIR / "READY_TO_MATERIALIZE.csv"
NEWLY_VERIFIED_CSV = REPORT_DIR / "NEWLY_VERIFIED_MANIFEST.csv"
AUDIT_CSV = REPORT_DIR / "MATERIALIZATION_AUDIT.csv"
JSON_OUT = SITE_DIR / "data" / "newly_verified_index.json"
JS_OUT = SITE_DIR / "data" / "newly_verified_index.js"

REQUIRED_FILES = ("video.mp4", "audio.wav", "label.txt", "sample.csv", "source.txt")
GATE_FIELDS = (
    "schema_gate",
    "media_gate",
    "audio_candidate_gate",
    "video_contact_gate",
    "audio_video_binding_gate",
    "independent_review_gate",
    "label_consistency_gate",
    "source_traceability_gate",
    "media_readable_gate",
    "duration_gate",
    "event_time_gate",
    "contact_audio_gate",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_source_txt(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def parse_sample_csv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def as_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def slash(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def workspace_safe_path(value: str) -> str:
    if not value:
        return ""
    raw = Path(value)
    try:
        resolved = raw.resolve(strict=False)
        if resolved.is_relative_to(WORKSPACE_ROOT):
            return slash(resolved.relative_to(WORKSPACE_ROOT))
        if resolved.is_relative_to(FIRST_PASS_ROOT):
            return "external:first-pass/" + slash(resolved.relative_to(FIRST_PASS_ROOT))
    except Exception:
        pass
    if raw.is_absolute():
        return raw.name
    return slash(value)


def gate_passed(value: str) -> bool:
    return value == "yes" or value == "pass" or value.startswith("pass_")


def file_info(sample_dir: Path) -> dict[str, dict[str, int | bool]]:
    info: dict[str, dict[str, int | bool]] = {}
    for name in REQUIRED_FILES:
        path = sample_dir / name
        info[name] = {
            "exists": path.exists(),
            "sizeBytes": path.stat().st_size if path.exists() else 0,
        }
    return info


def build_record(row: dict[str, str], order: int) -> dict:
    materialized_rel = slash(row.get("materialized_relative_path", ""))
    sample_dir = REPORT_DIR / materialized_rel
    source = parse_source_txt(sample_dir / "source.txt")
    sample_csv = parse_sample_csv(sample_dir / "sample.csv")
    files = file_info(sample_dir)
    final_start = as_float(row.get("final_event_start"))
    final_end = as_float(row.get("final_event_end"))
    before_start = as_float(row.get("event_start_before"))
    before_end = as_float(row.get("event_end_before"))
    final_center = None
    before_center = None
    if final_start is not None and final_end is not None:
        final_center = round((final_start + final_end) / 2, 6)
    if before_start is not None and before_end is not None:
        before_center = round((before_start + before_end) / 2, 6)

    gates = {field: row.get(field, "") for field in GATE_FIELDS}
    failed_gates = [field for field, value in gates.items() if not gate_passed(value)]
    staged_complete = all(item["exists"] and item["sizeBytes"] > 0 for item in files.values())

    return {
        "order": order,
        "sampleId": row.get("sample_id", ""),
        "label": row.get("label", ""),
        "collector": row.get("collector", ""),
        "batchName": row.get("batch_name", ""),
        "batchIndex": row.get("batch_index", ""),
        "reauditIndex": row.get("reaudit_index", ""),
        "sourceCsvRowNumber": row.get("source_csv_row_number", ""),
        "mainRelativePath": slash(row.get("main_relative_path", "")),
        "status": row.get("status", ""),
        "readiness": {
            "materializationReady": row.get("materialization_ready", ""),
            "blockedReason": row.get("blocked_reason", ""),
            "allListedGatesPass": len(failed_gates) == 0,
            "failedListedGates": failed_gates,
            "stagedFiveFilesComplete": staged_complete,
            "formalTrainableDirectStatus": row.get("formal_trainable_direct_status", ""),
            "currentCheckoutCompleteFiveFiles": row.get("current_checkout_complete_five_files", ""),
            "currentCheckoutTimeMatchesFinal": row.get("current_checkout_time_matches_final", ""),
            "firstpassCompleteFiveFiles": row.get("firstpass_complete_five_files", ""),
        },
        "timing": {
            "eventStartBefore": row.get("event_start_before", ""),
            "eventEndBefore": row.get("event_end_before", ""),
            "eventStartAfter": row.get("event_start_after", ""),
            "eventEndAfter": row.get("event_end_after", ""),
            "finalEventStart": row.get("final_event_start", ""),
            "finalEventEnd": row.get("final_event_end", ""),
            "finalCenter": final_center,
            "beforeCenter": before_center,
            "visualContactTime": row.get("visual_contact_time", ""),
            "audioCandidateTime": row.get("audio_candidate_time", ""),
            "materializedAudioPeakTime": row.get("materialized_audio_peak_time", ""),
            "materializedAudioMetrics": row.get("materialized_audio_metrics", ""),
            "videoDurationSec": row.get("video_duration_sec", ""),
            "audioDurationSec": row.get("audio_duration_sec", ""),
        },
        "gates": gates,
        "paths": {
            "materializedRelativePath": materialized_rel,
            "videoUrl": "../" + materialized_rel + "/video.mp4",
            "audioUrl": "../" + materialized_rel + "/audio.wav",
            "sampleCsvUrl": "../" + materialized_rel + "/sample.csv",
            "sourceTxtUrl": "../" + materialized_rel + "/source.txt",
            "reviewMediaVideoPath": workspace_safe_path(row.get("review_media_video_path", "")),
            "reviewMediaAudioPath": workspace_safe_path(row.get("review_media_audio_path", "")),
            "reviewMediaSourceFile": row.get("review_media_source_file", ""),
            "evidencePath": workspace_safe_path(row.get("evidence_path", "")),
            "reviewOutputPath": workspace_safe_path(row.get("review_output_path", "")),
        },
        "source": {
            "videoTitle": source.get("video_title", ""),
            "videoUrl": source.get("video_url", ""),
            "sourceId": source.get("source_id", ""),
            "clipId": source.get("clip_id", ""),
            "sourcePath": slash(source.get("source_path", "")),
            "clipStartTime": source.get("clip_start_time", ""),
            "clipEndTime": source.get("clip_end_time", ""),
        },
        "sampleCsv": sample_csv,
        "files": files,
        "notes": row.get("notes", ""),
    }


def count_by(rows: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = row.get(key, "") or "blank"
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    ready_rows = read_csv(READY_CSV)
    newly_rows = read_csv(NEWLY_VERIFIED_CSV)
    audit_rows = read_csv(AUDIT_CSV)
    records = [build_record(row, index + 1) for index, row in enumerate(ready_rows)]

    stats = {
        "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "sourceFiles": {
            "newlyVerifiedManifest": slash(NEWLY_VERIFIED_CSV.relative_to(WORKSPACE_ROOT)),
            "materializationAudit": slash(AUDIT_CSV.relative_to(WORKSPACE_ROOT)),
            "readyToMaterialize": slash(READY_CSV.relative_to(WORKSPACE_ROOT)),
        },
        "newlyVerifiedRows": len(newly_rows),
        "newlyVerifiedUniqueSampleIds": len({row.get("sample_id", "") for row in newly_rows}),
        "auditRows": len(audit_rows),
        "readyRows": len(ready_rows),
        "readyUniqueSampleIds": len({row.get("sample_id", "") for row in ready_rows}),
        "materializationReadyRows": sum(1 for row in ready_rows if row.get("materialization_ready") == "yes"),
        "blockedMaterializationRows": sum(1 for row in ready_rows if row.get("materialization_ready") != "yes"),
        "formalDirectTrainableRows": sum(
            1 for row in ready_rows if row.get("formal_trainable_direct_status") == "direct_trainable"
        ),
        "currentCheckoutCompleteFiveFiles": sum(
            1 for row in ready_rows if row.get("current_checkout_complete_five_files") == "yes"
        ),
        "currentCheckoutTimeMatchesFinal": sum(
            1 for row in ready_rows if row.get("current_checkout_time_matches_final") == "yes"
        ),
        "stagedFiveFilesComplete": sum(1 for record in records if record["readiness"]["stagedFiveFilesComplete"]),
        "labels": count_by(ready_rows, "label"),
        "batches": count_by(ready_rows, "batch_name"),
        "status": count_by(ready_rows, "status"),
        "policyNotes": [
            "These rows are evidence-verified candidates staged under reports/data_repair_resume_20260812; formal dataset files are not modified.",
            "The committed review site contains metadata and local relative media links, not the staged video/audio payloads.",
            "A row is review-ready only when staged five-file materialization and the listed gates pass.",
        ],
    }

    payload = {
        "schemaVersion": 1,
        "title": "新修复棒球击球数据检索复核页",
        "stats": stats,
        "records": records,
    }

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    JS_OUT.write_text(
        "window.REPAIR_REVIEW_INDEX = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n",
        encoding="utf-8",
    )
    print(f"wrote {JSON_OUT}")
    print(f"wrote {JS_OUT}")
    print(f"records={len(records)} ready={stats['materializationReadyRows']} blocked={stats['blockedMaterializationRows']}")


if __name__ == "__main__":
    main()
