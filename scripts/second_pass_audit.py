from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import wave
from array import array
from pathlib import Path
from typing import Any

from common import load_jsonl, repo_path


MANUAL_CORRECTIONS: dict[str, dict[str, Any]] = {
    "G_006": {
        "bounce": "no",
        "reason": "user_manual_qc: receiving height exceeds knee-height bounce standard",
    },
    "G_009": {
        "region": "1",
        "bounce": "no",
        "event_start": "0.970",
        "event_end": "1.070",
        "reason": "manual_qc: true contact is about 1.02s on third-base-side play",
    },
    "G_012": {
        "region": "1",
        "reason": "source/qwen evidence says third-base-side ground ball",
    },
    "G_016": {
        "region": "2",
        "reason": "source/qwen evidence says shortstop-side ground ball",
    },
}

MANUAL_REVIEW_NOTES: dict[str, str] = {
    "G_001": "user flagged sample but no specific failing field is documented",
    "G_004": "user flagged timing and region; source text is not enough for a safe automatic region correction",
}

REGION_RULES: list[tuple[str, str]] = [
    (r"\b(third baseman|third base|3rd baseman|3rd base|third-base line|hot corner)\b", "1"),
    (r"\b(shortstop|short stop|left side|hole between third and short)\b", "2"),
    (r"\b(second baseman|second base|2nd baseman|2nd base)\b", "3"),
    (r"\b(first baseman|first base|1st baseman|1st base|first-base line)\b", "4"),
]


def sample_dirs(dataset_root: Path) -> list[Path]:
    return sorted(
        [path for path in dataset_root.glob("*/*/*") if path.is_dir()],
        key=lambda path: (path.parents[1].name, path.name),
    )


def read_sample_csv(path: Path) -> tuple[list[str], dict[str, str]]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)
        return list(reader.fieldnames or []), row


def write_sample_csv(path: Path, fieldnames: list[str], row: dict[str, str]) -> None:
    with (path / "sample.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def baseline_sample_row(path: Path) -> dict[str, str]:
    rel_csv = (path / "sample.csv").relative_to(repo_path()).as_posix()
    try:
        proc = subprocess.run(
            ["git", "show", f"HEAD:{rel_csv}"],
            cwd=repo_path(),
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        return {}
    rows = list(csv.DictReader(proc.stdout.splitlines()))
    return rows[0] if rows else {}


def source_map(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    source_path = path / "source.txt"
    if not source_path.exists():
        return result
    for line in source_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def latest_qwen_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(path):
        clip_id = record.get("clip_id")
        if clip_id:
            records[clip_id] = record
    return records


def combined_evidence(source: dict[str, str], qwen_record: dict[str, Any] | None) -> str:
    parts = [
        source.get("video_title", ""),
        source.get("source_id", ""),
        source.get("clip_id", ""),
    ]
    if qwen_record:
        label = qwen_record.get("label") or {}
        parts.append(str(label.get("video_evidence") or ""))
        parts.append(str(label.get("audio_evidence") or ""))
    return " ".join(parts).lower()


def infer_region(evidence: str) -> tuple[str, str]:
    matches: list[tuple[str, str]] = []
    for pattern, region in REGION_RULES:
        if re.search(pattern, evidence, flags=re.IGNORECASE):
            matches.append((region, pattern))
    unique_regions = sorted({region for region, _pattern in matches})
    if len(unique_regions) == 1:
        return unique_regions[0], ";".join(pattern for _region, pattern in matches)
    if len(unique_regions) > 1:
        return "", "ambiguous:" + ",".join(unique_regions)
    return "", ""


def read_wav_first_channel(path: Path) -> tuple[int, array]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"Expected 16-bit WAV, got sample width {width}")
    values = array("h")
    values.frombytes(frames)
    if channels > 1:
        values = values[0::channels]
    return sample_rate, values


def rms(values: list[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(value * value for value in values) / len(values))


def top_diff_peak(path: Path, window_ms: float) -> tuple[float, float]:
    sample_rate, samples = read_wav_first_channel(path)
    window = max(1, int(sample_rate * window_ms / 1000.0))
    best_time = 0.0
    best_energy = -1.0
    for start in range(0, len(samples), window):
        stop = min(len(samples), start + window)
        if stop - start < 2:
            continue
        diff_sum = 0
        diff_count = 0
        previous = samples[start]
        for idx in range(start + 1, stop):
            diff = samples[idx] - previous
            diff_sum += diff * diff
            diff_count += 1
            previous = samples[idx]
        energy = math.sqrt(diff_sum / max(diff_count, 1))
        if energy > best_energy:
            best_energy = energy
            best_time = start / sample_rate
    return best_time, best_energy


def time_status(path: Path, row: dict[str, str], qwen_record: dict[str, Any] | None, tolerance: float) -> tuple[str, str, str]:
    event_start = float(row.get("event_start") or 0.0)
    event_end = float(row.get("event_end") or 0.0)
    reasons: list[str] = []
    status = "pass_audio_only_needs_visual_confirmation"
    peak_time = ""
    try:
        peak, _energy = top_diff_peak(path / "audio.wav", 20.0)
        peak_time = f"{peak:.3f}"
        if not (event_start - tolerance <= peak <= event_end + tolerance):
            status = "needs_visual_review"
            reasons.append(f"global_audio_diff_peak_outside_event_tolerance:{peak:.3f}")
    except Exception as exc:
        status = "needs_review"
        reasons.append(f"audio_peak_error:{type(exc).__name__}")

    if qwen_record:
        refinement = qwen_record.get("event_refinement") or {}
        original_start = refinement.get("original_event_start")
        original_end = refinement.get("original_event_end")
        if original_start is not None and original_end is not None:
            original_mid = (float(original_start) + float(original_end)) / 2.0
            current_mid = (event_start + event_end) / 2.0
            if abs(current_mid - original_mid) > 0.500:
                status = "needs_visual_review"
                reasons.append(f"large_refinement_shift:{original_mid:.3f}->{current_mid:.3f}")
    return status, peak_time, ";".join(reasons)


def audit_sample(path: Path, qwen_records: dict[str, dict[str, Any]], tolerance: float) -> dict[str, str]:
    fieldnames, row = read_sample_csv(path)
    baseline_row = baseline_sample_row(path) or row
    source = source_map(path)
    sample_id = row.get("sample_id") or path.name
    clip_id = source.get("clip_id", "")
    qwen_record = qwen_records.get(clip_id)
    evidence = combined_evidence(source, qwen_record)
    suggested_region, region_hint = ("", "")
    if row.get("label") == "ground_ball":
        suggested_region, region_hint = infer_region(evidence)

    time_state, audio_peak, time_reason = time_status(path, row, qwen_record, tolerance)
    correction = MANUAL_CORRECTIONS.get(sample_id, {})
    review_note = MANUAL_REVIEW_NOTES.get(sample_id, "")

    new_row = dict(row)
    needs_write = False
    for field in ("region", "bounce", "event_start", "event_end"):
        if field in correction and str(new_row.get(field, "")) != str(correction[field]):
            new_row[field] = str(correction[field])
            needs_write = True

    if correction and needs_write:
        write_sample_csv(path, fieldnames, new_row)

    changed_fields = [
        field
        for field in ("region", "bounce", "event_start", "event_end")
        if str(baseline_row.get(field, "")) != str(new_row.get(field, ""))
    ]

    region_status = "not_applicable"
    if row.get("label") == "ground_ball":
        if correction.get("region"):
            region_status = "manual_corrected" if "region" in changed_fields else "manual_checked"
        elif suggested_region and suggested_region != row.get("region"):
            region_status = "suggested_change_needs_visual_review"
        elif suggested_region:
            region_status = "source_hint_matches"
        else:
            region_status = "needs_visual_review"

    bounce_status = "not_applicable"
    if row.get("label") == "ground_ball":
        if correction.get("bounce"):
            bounce_status = "manual_corrected" if "bounce" in changed_fields else "manual_checked"
        else:
            bounce_status = "needs_knee_height_visual_review"

    if correction.get("event_start") or correction.get("event_end"):
        time_state = "manual_corrected" if {"event_start", "event_end"} & set(changed_fields) else "manual_checked"

    return {
        "sample_path": str(path.relative_to(repo_path())),
        "sample_id": sample_id,
        "label": row.get("label", ""),
        "old_region": baseline_row.get("region", row.get("region", "")),
        "new_region": new_row.get("region", ""),
        "suggested_region": suggested_region,
        "old_bounce": baseline_row.get("bounce", row.get("bounce", "")),
        "new_bounce": new_row.get("bounce", ""),
        "old_event_start": baseline_row.get("event_start", row.get("event_start", "")),
        "old_event_end": baseline_row.get("event_end", row.get("event_end", "")),
        "new_event_start": new_row.get("event_start", ""),
        "new_event_end": new_row.get("event_end", ""),
        "changed_fields": ",".join(changed_fields),
        "region_status": region_status,
        "bounce_status": bounce_status,
        "time_status": time_state,
        "audio_diff_peak_time": audio_peak,
        "review_note": review_note,
        "change_reason": correction.get("reason", ""),
        "time_reason": time_reason,
        "region_hint": region_hint,
        "clip_id": clip_id,
    }


def write_report(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_path",
        "sample_id",
        "label",
        "old_region",
        "new_region",
        "suggested_region",
        "old_bounce",
        "new_bounce",
        "old_event_start",
        "old_event_end",
        "new_event_start",
        "new_event_end",
        "changed_fields",
        "region_status",
        "bounce_status",
        "time_status",
        "audio_diff_peak_time",
        "review_note",
        "change_reason",
        "time_reason",
        "region_hint",
        "clip_id",
    ]
    with (output_dir / "second_pass_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    changed = [row for row in rows if row["changed_fields"]]
    needs_review = [
        row
        for row in rows
        if "review" in row["region_status"] or "review" in row["bounce_status"] or "review" in row["time_status"]
    ]
    summary = {
        "checked": len(rows),
        "changed_samples": len(changed),
        "needs_review_samples": len(needs_review),
        "changed_sample_ids": [row["sample_id"] for row in changed],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Second Pass Audit",
        "",
        f"- Checked samples: {summary['checked']}",
        f"- Samples changed: {summary['changed_samples']}",
        f"- Samples still needing visual review: {summary['needs_review_samples']}",
        "",
        "## Changed Samples",
        "",
    ]
    if changed:
        lines.append("| sample | changed_fields | old -> new | reason |")
        lines.append("| --- | --- | --- | --- |")
        for row in changed:
            deltas = []
            for field in row["changed_fields"].split(","):
                if field:
                    old = row.get(f"old_{field}", "")
                    new = row.get(f"new_{field}", "")
                    deltas.append(f"{field}: {old} -> {new}")
            lines.append(
                f"| {row['sample_id']} | {row['changed_fields']} | {'; '.join(deltas)} | {row['change_reason']} |"
            )
    else:
        lines.append("No sample CSV fields were changed.")
    lines.extend(
        [
            "",
            "## Review Policy",
            "",
            "- `region` can be suggested from source/Qwen evidence, but field-side ambiguity still requires visual review.",
            "- `bounce` requires the receiving fielder's knee-height standard and cannot be proven from CSV ranges alone.",
            "- `time_status=pass_audio_only_needs_visual_confirmation` means the audio check passed but visual contact still has not been certified.",
        ]
    )
    (output_dir / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-pass audit for region, bounce, and event timing definitions.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels_refined.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=repo_path("reports", "second_pass_audit_20260712"))
    parser.add_argument("--audio-tolerance", type=float, default=0.350)
    args = parser.parse_args()

    qwen_records = latest_qwen_records(args.labels)
    rows = [audit_sample(path, qwen_records, args.audio_tolerance) for path in sample_dirs(args.dataset_root)]
    write_report(rows, args.output_dir)
    changed = sum(1 for row in rows if row["changed_fields"])
    needs_review = sum(
        1
        for row in rows
        if "review" in row["region_status"] or "review" in row["bounce_status"] or "review" in row["time_status"]
    )
    print(f"Checked {len(rows)} samples; changed={changed}; needs_review={needs_review}; wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
