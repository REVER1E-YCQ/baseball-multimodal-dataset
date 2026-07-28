from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_TRAJECTORIES = {"fly", "pop_fly", "line_drive"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def bool_value(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def float_value(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def update_sample_csv(
    path: Path,
    *,
    event_start: str,
    event_end: str,
    trajectory_type: str,
) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if len(rows) != 1:
        raise ValueError(f"{path} must contain exactly one data row")
    rows[0]["event_start"] = event_start
    rows[0]["event_end"] = event_end
    if trajectory_type:
        rows[0]["trajectory_type"] = trajectory_type
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def update_source_txt(
    path: Path,
    *,
    clip_start: str,
    clip_end: str,
) -> None:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    values: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in values:
            order.append(key)
        values[key] = value.strip()
    values["clip_start_time"] = clip_start
    values["clip_end_time"] = clip_end
    source_id = values.get("source_id", "source")
    values["clip_id"] = (
        f"{source_id}_reclean_{clip_start.replace('.', 'p')}_{clip_end.replace('.', 'p')}"
    )
    for required in ("clip_id", "clip_start_time", "clip_end_time"):
        if required not in order:
            order.append(required)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        "\n".join(f"{key}: {values[key]}" for key in order) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def acceptance_reason(row: dict[str, str], minimum_confidence: float) -> tuple[bool, str]:
    if row.get("binding_status") != "audio_candidate_bound":
        return False, row.get("binding_status") or "missing_qwen_result"
    required_true = [
        "contact_audible",
        "contact_visible",
        "live_play",
        "fly_ball_semantics",
        "full_play_visible",
    ]
    failed = [field for field in required_true if not bool_value(row.get(field))]
    if failed:
        return False, "failed_gate:" + ",".join(failed)
    if bool_value(row.get("replay_or_slow_motion")):
        return False, "replay_or_slow_motion"
    if row.get("trajectory_type") not in VALID_TRAJECTORIES:
        return False, "invalid_qwen_trajectory"
    if float_value(row.get("confidence")) < minimum_confidence:
        return False, "qwen_confidence_below_threshold"
    if not row.get("final_event_start") or not row.get("final_event_end"):
        return False, "missing_bound_event_interval"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize one verified fly-ball repair batch.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--recut-manifest", type=Path, required=True)
    parser.add_argument("--qwen-summary", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--minimum-confidence", type=float, default=0.80)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    queue = read_csv(args.queue)
    recuts = {
        row["main_relative_path"]: row for row in read_csv(args.recut_manifest)
    }
    qwen = {
        row["main_relative_path"]: row for row in read_csv(args.qwen_summary)
    }
    reconciled: list[dict[str, Any]] = []

    for row in queue:
        relative_path = row["main_relative_path"]
        sample_dir = REPO_ROOT / relative_path
        recut = recuts.get(relative_path)
        review = qwen.get(relative_path)
        if review is None:
            accepted, reason = False, "missing_qwen_result"
        else:
            accepted, reason = acceptance_reason(review, args.minimum_confidence)
        before_video = sample_dir / "video.mp4"
        before_audio = sample_dir / "audio.wav"
        before_video_hash = sha256(before_video) if before_video.is_file() else ""
        before_audio_hash = sha256(before_audio) if before_audio.is_file() else ""
        before_trajectory = row.get("trajectory_type", "")
        after_trajectory = (
            review.get("trajectory_type", "") if accepted and review else before_trajectory
        )
        if accepted and recut:
            recut_video = Path(recut["video_path"])
            recut_audio = Path(recut["audio_path"])
            if (
                recut.get("status") not in {"recut_complete", "recut_partial_context"}
                or not recut_video.is_file()
                or not recut_audio.is_file()
            ):
                accepted = False
                reason = "recut_media_not_ready"
        if accepted and args.apply:
            if recut:
                atomic_copy(Path(recut["video_path"]), before_video)
                atomic_copy(Path(recut["audio_path"]), before_audio)
                update_source_txt(
                    sample_dir / "source.txt",
                    clip_start=recut["new_clip_start"],
                    clip_end=recut["new_clip_end"],
                )
            update_sample_csv(
                sample_dir / "sample.csv",
                event_start=review["final_event_start"],
                event_end=review["final_event_end"],
                trajectory_type=after_trajectory,
            )

        if not accepted:
            action = "unchanged_unresolved"
        elif recut:
            action = "recut_and_retime"
        else:
            action = "retime_or_metadata_only"
        changed = accepted and (
            bool(recut)
            or review["final_event_start"] != row["current_event_start"]
            or review["final_event_end"] != row["current_event_end"]
            or after_trajectory != before_trajectory
        )
        after_video_hash = (
            sha256(before_video) if args.apply and accepted and before_video.is_file() else ""
        )
        after_audio_hash = (
            sha256(before_audio) if args.apply and accepted and before_audio.is_file() else ""
        )
        reconciled.append(
            {
                "repair_batch": row["repair_batch"],
                "repair_batch_index": row["repair_batch_index"],
                "collector": row["collector"],
                "sample_id": row["sample_id"],
                "main_relative_path": relative_path,
                "primary_error": row["primary_error"],
                "result": action,
                "changed": "yes" if changed else "no",
                "unresolved_reason": reason,
                "before_event_start": row["current_event_start"],
                "before_event_end": row["current_event_end"],
                "after_event_start": review.get("final_event_start", "") if accepted else "",
                "after_event_end": review.get("final_event_end", "") if accepted else "",
                "selected_candidate_time": (
                    review.get("selected_candidate_time", "") if review else ""
                ),
                "visual_contact_time": (
                    review.get("visual_contact_time", "") if review else ""
                ),
                "audio_visual_offset": (
                    review.get("audio_visual_offset", "") if review else ""
                ),
                "before_duration": row.get("video_duration", ""),
                "after_duration": recut.get("new_duration", "") if accepted and recut else row.get("video_duration", ""),
                "recut_status": recut.get("status", "") if recut else "not_required",
                "source_clip_start_before": row.get("source_clip_start", ""),
                "source_clip_start_after": recut.get("new_clip_start", "") if accepted and recut else "",
                "source_clip_end_after": recut.get("new_clip_end", "") if accepted and recut else "",
                "before_trajectory": before_trajectory,
                "after_trajectory": after_trajectory if accepted else "",
                "trajectory_changed": (
                    "yes" if accepted and after_trajectory != before_trajectory else "no"
                ),
                "qwen_binding_status": review.get("binding_status", "") if review else "",
                "qwen_confidence": review.get("confidence", "") if review else "",
                "qwen_model": review.get("model", "") if review else "",
                "before_video_sha256": before_video_hash,
                "before_audio_sha256": before_audio_hash,
                "after_video_sha256": after_video_hash,
                "after_audio_sha256": after_audio_hash,
                "apply_mode": "applied" if args.apply else "dry_run",
            }
        )

    write_csv(args.output_csv, reconciled)
    result_counts = Counter(row["result"] for row in reconciled)
    error_counts = Counter(row["primary_error"] for row in reconciled)
    changed_count = sum(row["changed"] == "yes" for row in reconciled)
    trajectory_count = sum(row["trajectory_changed"] == "yes" for row in reconciled)
    complete_recut = sum(
        row["result"] == "recut_and_retime" and row["recut_status"] == "recut_complete"
        for row in reconciled
    )
    partial_recut = sum(
        row["result"] == "recut_and_retime"
        and row["recut_status"] == "recut_partial_context"
        for row in reconciled
    )
    lines = [
        "# Fly Ball Batch Materialization Report",
        "",
        f"- Mode: {'applied' if args.apply else 'dry run'}",
        f"- Queue rows: {len(reconciled)}",
        f"- Changed samples: {changed_count}",
        f"- Recut and retimed: {result_counts['recut_and_retime']}",
        f"- Retime or metadata only: {result_counts['retime_or_metadata_only']}",
        f"- Unchanged unresolved: {result_counts['unchanged_unresolved']}",
        f"- Complete-context recuts accepted: {complete_recut}",
        f"- Partial-context recuts accepted after visual gate: {partial_recut}",
        f"- Trajectory corrections: {trajectory_count}",
        "",
        "## Original Error Categories",
        "",
        *[f"- {name}: {count}" for name, count in error_counts.most_common()],
        "",
        "## Changed Samples",
        "",
        "| sample | result | before event | after event | before duration | after duration | trajectory |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    for row in reconciled:
        if row["changed"] == "yes":
            lines.append(
                f"| {row['sample_id']} | {row['result']} | "
                f"{row['before_event_start']}-{row['before_event_end']} | "
                f"{row['after_event_start']}-{row['after_event_end']} | "
                f"{row['before_duration']} | {row['after_duration']} | "
                f"{row['before_trajectory']} -> {row['after_trajectory']} |"
            )
    lines.extend(
        [
            "",
            "## Unresolved Samples",
            "",
            "| sample | primary error | reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in reconciled:
        if row["result"] == "unchanged_unresolved":
            lines.append(
                f"| {row['sample_id']} | {row['primary_error']} | {row['unresolved_reason']} |"
            )
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"mode={'apply' if args.apply else 'dry_run'} queue={len(reconciled)} "
        f"changed={changed_count} unresolved={result_counts['unchanged_unresolved']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
