from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

from common import repo_path, write_csv
from detect_contact_audio import window_energies_from_wav
from qwen_review_dataset import (
    adjudication_evidence_passes,
    semantics_evidence_passes,
    semantics_needs_change,
    region_evidence_consistent,
    successful_records,
    substantive_evidence,
    timing_evidence_passes,
    timing_needs_change,
)


REPORT_FIELDS = [
    "sample_id",
    "sample_path",
    "status",
    "audio_ratio",
    "changed_fields",
    "old_values",
    "new_values",
    "reason",
]


def sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("*/*/*") if (path / "sample.csv").exists())


def review_records(paths: list[Path]) -> dict[tuple[str, str], dict[str, Any]]:
    combined: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        combined.update(successful_records(path))
    return combined


def read_sample(path: Path) -> tuple[list[str], dict[str, str]]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), next(reader)


def write_sample(path: Path, fields: list[str], row: dict[str, str]) -> None:
    write_csv(path / "sample.csv", [row], fields)


def manual_decisions(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return {row["sample_id"]: row for row in csv.DictReader(fh) if row.get("sample_id")}


def proposed_row(current: dict[str, str], timing: dict[str, Any], semantics: dict[str, Any]) -> dict[str, str]:
    result = dict(current)
    result["event_start"] = f"{float(timing['corrected_event_start']):.3f}"
    result["event_end"] = f"{float(timing['corrected_event_end']):.3f}"
    result["strength"] = str(semantics["verified_strength"])
    if current.get("label") == "ground_ball":
        ground = semantics.get("ground_ball") or {}
        result["region"] = str(ground["region"])
        result["bounce"] = str(ground["bounce"])
    else:
        fly = semantics.get("fly_ball") or {}
        result["landing_zone"] = str(fly["landing_zone"])
        result["trajectory_type"] = str(fly["trajectory_type"])
    return result


def canonicalize_timing(timing: dict[str, Any]) -> dict[str, Any]:
    """Center legacy edge-anchored intervals on an otherwise verified collision."""
    result = dict(timing)
    try:
        contact = float(result["observed_contact_time"])
        start = float(result["corrected_event_start"])
        end = float(result["corrected_event_end"])
    except (KeyError, TypeError, ValueError):
        return result
    evidence_ok = (
        result.get("decision") in {"pass", "correct"}
        and float(result.get("confidence") or 0) >= 0.85
        and result.get("contact_visible") is True
        and result.get("contact_audible") is True
        and result.get("audio_video_aligned") is True
        and 0.500 <= start < end
        and end - start <= 0.2000001
        and start <= contact <= end
        and contact >= 0.550
    )
    if evidence_ok and not (contact - start >= 0.020 and end - contact >= 0.020 and end - start <= 0.1500001):
        result["corrected_event_start"] = round(contact - 0.050, 3)
        result["corrected_event_end"] = round(contact + 0.050, 3)
    return result


def adjudicated_timing(adjudication: dict[str, Any]) -> dict[str, Any]:
    if not adjudication_evidence_passes(adjudication):
        return {}
    try:
        start = float(adjudication["event_start"])
        end = float(adjudication["event_end"])
    except (KeyError, TypeError, ValueError):
        return {}
    if not (0.500 <= start < end and end - start <= 0.2000001):
        return {}
    midpoint = (start + end) / 2.0
    return {
        "decision": "correct",
        "confidence": adjudication.get("confidence"),
        "contact_visible": adjudication.get("contact_visible"),
        "contact_audible": adjudication.get("contact_audible"),
        "audio_video_aligned": adjudication.get("audio_video_aligned"),
        "observed_contact_time": midpoint,
        "corrected_event_start": round(midpoint - 0.050, 3),
        "corrected_event_end": round(midpoint + 0.050, 3),
    }


def apply_manual(current: dict[str, str], decision: dict[str, str]) -> dict[str, str]:
    result = dict(current)
    for field in current:
        if field in decision and decision[field] != "":
            result[field] = decision[field]
    return result


def changed_fields(old: dict[str, str], new: dict[str, str]) -> list[str]:
    return [field for field in old if old.get(field, "") != new.get(field, "")]


def compact_values(row: dict[str, str], fields: list[str]) -> str:
    return json.dumps({field: row.get(field, "") for field in fields}, ensure_ascii=False, sort_keys=True)


def audio_evidence_ratio(path: Path, start: float, end: float, tolerance: float) -> float:
    energies, diff_energies = window_energies_from_wav(path / "audio.wav")
    event = [energy for timestamp, energy in energies if start - tolerance <= timestamp <= end + tolerance]
    diff_event = [energy for timestamp, energy in diff_energies if start - tolerance <= timestamp <= end + tolerance]
    rms_ratio = max(event, default=0.0) / max(statistics.median(energy for _timestamp, energy in energies), 1e-9)
    diff_ratio = max(diff_event, default=0.0) / max(
        statistics.median(energy for _timestamp, energy in diff_energies), 1e-9
    )
    return max(rms_ratio, diff_ratio)


def adjudication_matches(row: dict[str, str], adjudication: dict[str, Any], time_tolerance: float = 0.150) -> bool:
    if adjudication.get("label") != row.get("label"):
        return False
    try:
        adjudicated_mid = (float(adjudication["event_start"]) + float(adjudication["event_end"])) / 2.0
        row_mid = (float(row["event_start"]) + float(row["event_end"])) / 2.0
    except (KeyError, TypeError, ValueError):
        return False
    if abs(adjudicated_mid - row_mid) > time_tolerance:
        return False
    if adjudication.get("strength") != row.get("strength"):
        return False
    if row.get("label") == "ground_ball":
        return str(adjudication.get("region", "")) == row.get("region") and adjudication.get("bounce") == row.get("bounce")
    return (
        str(adjudication.get("landing_zone", "")) == row.get("landing_zone")
        and adjudication.get("trajectory_type") == row.get("trajectory_type")
    )


def permissive_semantics_row(current: dict[str, str], semantics: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Apply only independently evidenced semantic fields; keep the rest unchanged."""
    result = dict(current)
    applied: list[str] = []
    if semantics.get("decision") not in {"pass", "correct"} or float(semantics.get("confidence") or 0) < 0.75:
        return result, applied
    if semantics.get("verified_strength") in {"low", "medium", "high"}:
        result["strength"] = str(semantics["verified_strength"])
        applied.append("strength")
    if current.get("label") == "ground_ball" and semantics.get("verified_label") == "ground_ball":
        ground = semantics.get("ground_ball") or {}
        if (
            ground.get("region_verified") is True
            and str(ground.get("region")) in {"1", "2", "3", "4"}
            and region_evidence_consistent(ground)
            and substantive_evidence(ground.get("region_evidence"))
        ):
            result["region"] = str(ground["region"])
            applied.append("region")
        if (
            ground.get("bounce") in {"yes", "no"}
            and ground.get("receiving_moment_visible") is True
            and ground.get("knee_reference_visible") is True
            and substantive_evidence(ground.get("receiving_height_evidence"))
        ):
            result["bounce"] = str(ground["bounce"])
            applied.append("bounce")
    elif current.get("label") == "fly_ball" and semantics.get("verified_label") == "fly_ball":
        fly = semantics.get("fly_ball") or {}
        if (
            fly.get("landing_zone_verified") is True
            and str(fly.get("landing_zone")) in {str(value) for value in range(1, 10)}
            and substantive_evidence(fly.get("flight_evidence"))
        ):
            result["landing_zone"] = str(fly["landing_zone"])
            applied.append("landing_zone")
        if fly.get("trajectory_type") in {"fly", "line_drive", "pop_fly"} and substantive_evidence(fly.get("flight_evidence")):
            result["trajectory_type"] = str(fly["trajectory_type"])
            applied.append("trajectory_type")
    return result, applied


def explicit_rejection(timing: dict[str, Any], semantics: dict[str, Any], adjudication: dict[str, Any]) -> bool:
    """A permissive pass excludes only a model's explicit unusable-sample decision."""
    return any(result.get("decision") == "reject" for result in (timing, semantics, adjudication) if result)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile strict Qwen dataset review into sample CSV files.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--review", type=Path, nargs="+", default=[repo_path("reports", "qwen_dataset_review.jsonl")])
    parser.add_argument(
        "--manual-decisions",
        type=Path,
        default=repo_path("reports", "manual_qc_G001_G004_G006_G009_G012_G016", "full_audit_decisions.csv"),
    )
    parser.add_argument("--output", type=Path, default=repo_path("reports", "qwen_dataset_reconciliation.csv"))
    parser.add_argument("--audio-tolerance", type=float, default=0.050)
    parser.add_argument("--min-audio-ratio", type=float, default=2.0)
    parser.add_argument(
        "--permissive-fields",
        action="store_true",
        help="Apply independently evidenced fields and preserve unverified secondary fields; exclude only explicit rejects.",
    )
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    records = review_records(args.review)
    manual = manual_decisions(args.manual_decisions)
    report: list[dict[str, str]] = []
    for path in sample_dirs(args.dataset_root.resolve()):
        fields, current = read_sample(path)
        sample_id = current.get("sample_id", path.name)
        audio_ratio = ""
        decision = manual.get(sample_id)
        if decision and decision.get("decision") == "reject":
            new = current
            status = "manual_reject"
            reason = decision.get("reason", "manual rejection")
        elif decision and decision.get("decision") == "accept":
            new = apply_manual(current, decision)
            status = "manual_applied" if args.apply else "manual_proposed"
            reason = decision.get("reason", "manual evidence")
        else:
            timing = canonicalize_timing((records.get((sample_id, "timing")) or {}).get("result") or {})
            semantics = (records.get((sample_id, "semantics")) or {}).get("result") or {}
            adjudication = (records.get((sample_id, "adjudication")) or {}).get("result") or {}
            if args.permissive_fields:
                if not timing or not semantics:
                    new = current
                    status = "incomplete"
                    reason = "missing timing or semantics stage"
                elif not timing_evidence_passes(timing):
                    new = current
                    status = "manual_review"
                    reason = "audio-candidate timing evidence gate failed"
                elif explicit_rejection(timing, semantics, adjudication):
                    new = current
                    status = "permissive_reject"
                    reason = "one review stage explicitly found the sample unusable"
                elif semantics.get("verified_label") not in {current.get("label"), None, ""}:
                    new = current
                    status = "needs_migration"
                    reason = "usable sample has a class change and needs label-directory migration"
                else:
                    new = dict(current)
                    applied: list[str] = []
                    if timing_evidence_passes(timing):
                        new["event_start"] = f"{float(timing['corrected_event_start']):.3f}"
                        new["event_end"] = f"{float(timing['corrected_event_end']):.3f}"
                        applied.extend(["event_start", "event_end"])
                    semantic_row, semantic_fields = permissive_semantics_row(new, semantics)
                    new = semantic_row
                    applied.extend(semantic_fields)
                    status = "auto_applied" if changed_fields(current, new) else "pass"
                    reason = (
                        "field-level evidence applied; unverified secondary fields retained"
                        if applied
                        else "usable sample retained; no independently evidenced field change"
                    )
            elif timing and not timing_evidence_passes(timing):
                replacement = adjudicated_timing(adjudication)
                if replacement:
                    timing = replacement
            if not args.permissive_fields and (not timing or not semantics):
                new = current
                status = "incomplete"
                reason = "missing timing or semantics stage"
            elif not args.permissive_fields and (not timing_evidence_passes(timing) or not semantics_evidence_passes(semantics)):
                new = current
                status = "manual_review"
                reason = "foundational audio-visual evidence gate failed"
            elif not args.permissive_fields and semantics.get("verified_label") != current.get("label"):
                new = current
                status = "manual_review"
                reason = "class change requires directory and sample-id migration"
            elif not args.permissive_fields:
                candidate = proposed_row(current, timing, semantics)
                ratio = audio_evidence_ratio(
                    path,
                    float(candidate["event_start"]),
                    float(candidate["event_end"]),
                    args.audio_tolerance,
                )
                audio_ratio = f"{ratio:.3f}"
                timing_changed = timing_needs_change(current, timing)
                semantics_changed = semantics_needs_change(current, semantics)
                any_change = timing_changed or semantics_changed
                if ratio < args.min_audio_ratio:
                    new = current
                    status = "manual_review"
                    reason = f"local contact transient ratio {ratio:.2f} is below {args.min_audio_ratio:.2f}"
                elif adjudication:
                    if not adjudication_evidence_passes(adjudication):
                        new = current
                        status = "manual_review"
                        reason = "adjudication evidence gate failed"
                    elif not adjudication_matches(candidate, adjudication):
                        new = current
                        status = "manual_review"
                        reason = "adjudication fields do not match the evidence-backed candidate"
                    else:
                        new = proposed_row(current, timing, semantics)
                        status = "auto_applied" if args.apply and any_change else "auto_proposed" if any_change else "pass"
                        reason = "strict timing, semantics, and adjudication gates passed"
                elif any_change:
                    new = current
                    status = "manual_review"
                    reason = "correction requires adjudication"
                else:
                    new = proposed_row(current, timing, semantics)
                    status = "pass"
                    reason = "strict timing and semantics gates passed"

        changes = changed_fields(current, new)
        if args.apply and changes and status in {"manual_applied", "auto_applied"}:
            write_sample(path, fields, new)
        report.append(
            {
                "sample_id": sample_id,
                "sample_path": str(path.relative_to(repo_path())),
                "status": status,
                "audio_ratio": audio_ratio,
                "changed_fields": ",".join(changes),
                "old_values": compact_values(current, changes),
                "new_values": compact_values(new, changes),
                "reason": reason,
            }
        )

    write_csv(args.output, report, REPORT_FIELDS)
    counts: dict[str, int] = {}
    for row in report:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"checked={len(report)} apply={args.apply} statuses={json.dumps(counts, sort_keys=True)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
