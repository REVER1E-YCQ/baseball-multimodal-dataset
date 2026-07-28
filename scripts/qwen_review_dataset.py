from __future__ import annotations

import argparse
import csv
import json
import msvcrt
import os
import time
import wave
from array import array
from pathlib import Path
from typing import Any

from common import append_jsonl, get_env_first, load_jsonl, repo_path, write_csv
from qwen_omni_label import (
    AuthError,
    ModelQuotaError,
    call_qwen,
    extract_json,
    filter_models_by_usage,
    load_models,
    model_token_cap,
    model_token_reserve,
    model_usage_totals,
    usage_total_tokens,
)
from qwen_realtime import call_qwen_realtime


SUMMARY_FIELDS = [
    "sample_id",
    "sample_path",
    "label",
    "timing_status",
    "semantics_status",
    "adjudication_status",
    "final_status",
    "reviewed_stages",
    "notes",
]


class ModelPoolExhausted(RuntimeError):
    pass


def acquire_process_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        handle.close()
        raise SystemExit(f"Another dataset review process is already running: {path}") from exc
    return handle


def sample_dirs(dataset_root: Path) -> list[Path]:
    return sorted(path for path in dataset_root.glob("*/*/*") if (path / "sample.csv").exists())


def sample_row(path: Path) -> dict[str, str]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        return next(csv.DictReader(fh))


def audio_transient_candidates(path: Path, limit: int = 8, separation: float = 0.250) -> list[float]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        width = wav.getsampwidth()
        values = array("h")
        values.frombytes(wav.readframes(wav.getnframes()))
    if width != 2:
        return []
    if channels > 1:
        values = values[0::channels]
    window = max(1, int(sample_rate * 0.020))
    energies: list[tuple[float, float]] = []
    for start in range(0, len(values) - 1, window):
        stop = min(len(values), start + window)
        previous = values[start]
        diff_sum = 0
        count = 0
        for index in range(start + 1, stop):
            delta = values[index] - previous
            diff_sum += delta * delta
            count += 1
            previous = values[index]
        energies.append((diff_sum / max(count, 1), start / sample_rate))
    selected: list[float] = []
    for _energy, timestamp in sorted(energies, reverse=True):
        if all(abs(timestamp - existing) >= separation for existing in selected):
            selected.append(round(timestamp, 3))
        if len(selected) >= limit:
            break
    return selected


def source_text(path: Path) -> str:
    source = path / "source.txt"
    return source.read_text(encoding="utf-8", errors="replace") if source.exists() else ""


def successful_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record in load_jsonl(path):
        if record.get("stage") == "reset" and record.get("sample_id"):
            sample_id = record["sample_id"]
            reset_stages = set((record.get("result") or {}).get("stages") or [])
            for key in list(records):
                if key[0] == sample_id and (not reset_stages or key[1] in reset_stages):
                    records.pop(key)
            continue
        if record.get("sample_id") and record.get("stage") and record.get("result") and not record.get("error"):
            records[(record["sample_id"], record["stage"])] = record
    return records


def combined_usage(paths: list[Path]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for path in paths:
        for model, tokens in model_usage_totals(path).items():
            totals[model] = totals.get(model, 0) + tokens
    return totals


def load_review_models() -> list[str]:
    configured = os.getenv("QWEN_REVIEW_MODEL_FALLBACKS")
    if configured:
        return [item.strip() for item in configured.split(",") if item.strip()]
    config_path = repo_path("config", "qwen_models.json")
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        models = config.get("review_fallback_models")
        if models:
            realtime = config.get("review_realtime_models") or []
            return [str(item) for item in [*models, *realtime]]
    return load_models()


def models_for_stage(all_models: list[str], stage: str) -> list[str]:
    if os.getenv("QWEN_REVIEW_MODEL_FALLBACKS"):
        return all_models
    config_path = repo_path("config", "qwen_models.json")
    if not config_path.exists():
        return all_models
    config = json.loads(config_path.read_text(encoding="utf-8"))
    exact_stage_models = (config.get("review_stage_models") or {}).get(stage) or []
    if exact_stage_models:
        allowed = {str(model) for model in exact_stage_models}
        return [model for model in all_models if model in allowed]
    stage_models = (config.get("review_realtime_stage_models") or {}).get(stage) or []
    if not stage_models:
        return all_models
    allowed_realtime = {str(model) for model in stage_models}
    return [model for model in all_models if "-realtime" not in model or model in allowed_realtime]


def invoke(
    sample_id: str,
    stage: str,
    video_path: Path,
    prompt: str,
    output: Path,
    all_models: list[str],
    usage_totals: dict[str, int],
    blocked: set[str],
    api_key: str,
    base_url: str,
    cap: int,
    reserve: int,
) -> dict[str, Any] | None:
    available = [m for m in filter_models_by_usage(all_models, usage_totals, cap, reserve) if m not in blocked]
    if not available:
        raise ModelPoolExhausted("No review model remains below the configured token cap.")
    # Rotate independently by sample and stage so timing, semantics, and adjudication
    # do not inherit one model's systematic visual interpretation.
    stage_offset = {"timing": 0, "semantics": 1, "adjudication": 2}.get(stage, 0)
    rotation = (sum(ord(char) for char in sample_id) + stage_offset) % len(available)
    available = available[rotation:] + available[:rotation]
    last_error = ""
    started = time.time()
    for model in available:
        usage = None
        try:
            if "-realtime" in model:
                realtime_url = os.getenv("QWEN_REALTIME_BASE_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime")
                # Fielding-region review needs the actual receiving moment, so
                # retain a denser cadence than broad semantic classification.
                frame_fps = 4.0 if stage == "fielding" else (2.0 if stage == "timing" else 1.0)
                text, usage = call_qwen_realtime(model, video_path, prompt, api_key, realtime_url, frame_fps)
            else:
                text, usage = call_qwen(model, video_path, prompt, base_url, api_key)
            result = extract_json(text)
            record = {
                "sample_id": sample_id,
                "stage": stage,
                "model": model,
                "usage": usage,
                "elapsed_seconds": round(time.time() - started, 3),
                "result": result,
                "error": "",
            }
            append_jsonl(output, record)
            if usage:
                usage_totals[model] = usage_totals.get(model, 0) + usage_total_tokens(usage)
            # A model that takes several minutes per clip is not viable for a
            # full-batch audit. Keep its completed result, but route later
            # samples to the next enabled model.
            if record["elapsed_seconds"] > 60:
                blocked.add(model)
            return record
        except AuthError as exc:
            # Access can differ by model and endpoint for the same key.  A
            # 401 from one model must not prevent trying the next configured
            # model; if every model rejects the key the final record captures
            # that fact without consuming further quota.
            blocked.add(model)
            last_error = str(exc)
            continue
        except ModelQuotaError as exc:
            blocked.add(model)
            last_error = str(exc)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if "-realtime" in model and any(
                marker in str(exc).lower()
                for marker in ("capacity-limited", "too many requests", "thread pool", "max_workers")
            ):
                blocked.add(model)
    append_jsonl(
        output,
        {
            "sample_id": sample_id,
            "stage": stage,
            "model": "",
            "usage": None,
            "elapsed_seconds": round(time.time() - started, 3),
            "result": None,
            "error": last_error or "no model available below token cap",
        },
    )
    return None


def timing_needs_change(row: dict[str, str], timing: dict[str, Any]) -> bool:
    if timing.get("decision") != "pass":
        return True
    try:
        old_mid = (float(row["event_start"]) + float(row["event_end"])) / 2.0
        new_mid = float(timing["observed_contact_time"])
        return abs(old_mid - new_mid) > 0.050
    except (KeyError, TypeError, ValueError):
        return True


def semantics_needs_change(row: dict[str, str], result: dict[str, Any]) -> bool:
    if result.get("decision") != "pass" or result.get("verified_label") != row.get("label"):
        return True
    if result.get("verified_strength") != row.get("strength"):
        return True
    if row.get("label") == "ground_ball":
        ground = result.get("ground_ball") or {}
        return str(ground.get("region", "")) != row.get("region") or ground.get("bounce") != row.get("bounce")
    fly = result.get("fly_ball") or {}
    return str(fly.get("landing_zone", "")) != row.get("landing_zone") or fly.get("trajectory_type") != row.get("trajectory_type")


def timing_evidence_passes(result: dict[str, Any]) -> bool:
    try:
        start = float(result.get("corrected_event_start"))
        end = float(result.get("corrected_event_end"))
        contact = float(result.get("observed_contact_time"))
    except (TypeError, ValueError):
        return False
    return (
        result.get("decision") in {"pass", "correct"}
        and float(result.get("confidence") or 0) >= 0.85
        and result.get("contact_audible") is True
        and substantive_evidence(result.get("audio_evidence"))
        and 0.0 <= start < end
        and 0.020 <= contact - start
        and 0.020 <= end - contact
        and end - start <= 0.200001
    )


def substantive_evidence(value: Any) -> bool:
    text = str(value or "").strip()
    normalized = text.lower()
    placeholders = {
        "short evidence",
        "short direct evidence",
        "ball-path evidence",
        "short frame-level evidence",
        "unverified",
        "n/a",
    }
    return len(text) >= 24 and normalized not in placeholders


def region_evidence_consistent(ground: dict[str, Any]) -> bool:
    region = str(ground.get("region", ""))
    evidence = str(ground.get("region_evidence", "")).lower()
    third_side = any(term in evidence for term in ("third-base", "third base", "3b side", "3b line"))
    first_side = any(term in evidence for term in ("first-base", "first base", "1b side", "1b line"))
    left_side = "left" in evidence
    right_side = "right" in evidence
    if third_side and region not in {"1", "2"}:
        return False
    if first_side and region not in {"3", "4"}:
        return False
    if left_side and region not in {"1", "2"}:
        return False
    if right_side and region not in {"3", "4"}:
        return False
    return not (third_side and first_side)


def semantics_evidence_passes(result: dict[str, Any]) -> bool:
    if result.get("decision") not in {"pass", "correct"} or float(result.get("confidence") or 0) < 0.85:
        return False
    label = result.get("verified_label")
    if result.get("verified_strength") not in {"low", "medium", "high"}:
        return False
    if label == "ground_ball":
        ground = result.get("ground_ball") or {}
        return (
            ground.get("region_verified") is True
            and str(ground.get("region")) in {"1", "2", "3", "4"}
            and ground.get("bounce") in {"yes", "no"}
            and ground.get("receiving_moment_visible") is True
            and ground.get("knee_reference_visible") is True
            and region_evidence_consistent(ground)
            and substantive_evidence(ground.get("region_evidence"))
            and substantive_evidence(ground.get("receiving_height_evidence"))
        )
    if label == "fly_ball":
        fly = result.get("fly_ball") or {}
        return (
            fly.get("landing_zone_verified") is True
            and str(fly.get("landing_zone")) in {str(value) for value in range(1, 10)}
            and fly.get("trajectory_type") in {"fly", "line_drive", "pop_fly"}
            and substantive_evidence(fly.get("flight_evidence"))
        )
    return False


def adjudication_evidence_passes(result: dict[str, Any]) -> bool:
    return (
        result.get("decision") in {"accept_current", "accept_correction"}
        and float(result.get("confidence") or 0) >= 0.85
        and result.get("contact_audible") is True
        and substantive_evidence(result.get("timing_evidence"))
        and substantive_evidence(result.get("field_evidence"))
    )


def context_prompt(
    base: str,
    row: dict[str, str],
    source: str,
    prior: dict[str, Any] | None = None,
    audio_candidates: list[float] | None = None,
) -> str:
    context: dict[str, Any] = {"current_sample_csv": row, "source_metadata": source}
    if audio_candidates is not None:
        context["local_audio_transient_candidates_seconds"] = audio_candidates
    if prior is not None:
        context["prior_audit"] = prior
    return base + "\n\nAUDIT CONTEXT:\n" + json.dumps(context, ensure_ascii=False, sort_keys=True)


def write_summary(samples: list[Path], records: dict[tuple[str, str], dict[str, Any]], output: Path) -> None:
    rows: list[dict[str, str]] = []
    for path in samples:
        row = sample_row(path)
        sample_id = row.get("sample_id", path.name)
        stage_results = {
            stage: (records.get((sample_id, stage)) or {}).get("result") or {}
            for stage in ("timing", "semantics", "adjudication")
        }
        timing = stage_results["timing"]
        semantics = stage_results["semantics"]
        adjudication = stage_results["adjudication"]
        completed = [stage for stage, result in stage_results.items() if result]
        foundations_pass = timing_evidence_passes(timing) and semantics_evidence_passes(semantics)
        if adjudication:
            final = "auto_accepted" if foundations_pass and adjudication_evidence_passes(adjudication) else "manual_review"
        elif timing and semantics and foundations_pass and not timing_needs_change(row, timing) and not semantics_needs_change(row, semantics):
            final = "auto_accepted"
        else:
            final = "incomplete"
        rows.append(
            {
                "sample_id": sample_id,
                "sample_path": str(path.relative_to(repo_path())),
                "label": row.get("label", ""),
                "timing_status": str(timing.get("decision", "missing")),
                "semantics_status": str(semantics.get("decision", "missing")),
                "adjudication_status": str(adjudication.get("decision", "not_required" if timing and semantics else "missing")),
                "final_status": final,
                "reviewed_stages": ",".join(completed),
                "notes": "Qwen results are proposals until applied by the separate reconciliation step.",
            }
        )
    write_csv(output, rows, SUMMARY_FIELDS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run staged Qwen audio-visual review over materialized samples.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--output", type=Path, default=repo_path("reports", "qwen_dataset_review.jsonl"))
    parser.add_argument("--summary", type=Path, default=repo_path("reports", "qwen_dataset_review_summary.csv"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample-id", action="append", default=[], help="Review only the named sample; repeatable.")
    parser.add_argument("--force", action="store_true", help="Rerun selected samples even when successful stages exist.")
    parser.add_argument("--force-stage", action="append", choices=["timing", "semantics", "adjudication"], default=[])
    parser.add_argument(
        "--only-stage",
        choices=["timing", "semantics", "fielding"],
        help="Run exactly one independent review stage.  This avoids unrelated model calls during focused audits.",
    )
    parser.add_argument("--retry-model", help="Restrict rerun to samples whose selected stage used this model.")
    parser.add_argument(
        "--retry-disallowed-stage-models",
        action="store_true",
        help="Rerun selected stages whose recorded model is outside the current exact stage allowlist.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    samples = sample_dirs(args.dataset_root.resolve())
    if args.sample_id:
        wanted = set(args.sample_id)
        samples = [path for path in samples if sample_row(path).get("sample_id", path.name) in wanted]
    if args.limit:
        samples = samples[: args.limit]
    records = successful_records(args.output)
    forced_stages = {"timing", "semantics", "adjudication"} if args.force else set(args.force_stage)
    configured_models = load_review_models()
    if args.retry_model:
        if not forced_stages:
            raise SystemExit("--retry-model requires --force-stage or --force.")
        retry_ids = {
            sample_id
            for (sample_id, stage), record in records.items()
            if stage in forced_stages and record.get("model") == args.retry_model
        }
        samples = [path for path in samples if sample_row(path).get("sample_id", path.name) in retry_ids]
    if args.retry_disallowed_stage_models:
        if not forced_stages:
            raise SystemExit("--retry-disallowed-stage-models requires --force-stage or --force.")
        retry_ids = {
            sample_id
            for (sample_id, stage), record in records.items()
            if stage in forced_stages and record.get("model") not in models_for_stage(configured_models, stage)
        }
        samples = [path for path in samples if sample_row(path).get("sample_id", path.name) in retry_ids]
    if forced_stages:
        selected_ids = {sample_row(path).get("sample_id", path.name) for path in samples}
        for key in list(records):
            if key[0] in selected_ids and key[1] in forced_stages:
                records.pop(key)
    timing_prompt = repo_path("prompts", "contact_timing_review_prompt.md").read_text(encoding="utf-8")
    semantics_prompt = repo_path("prompts", "field_semantics_review_prompt.md").read_text(encoding="utf-8")
    adjudication_prompt = repo_path("prompts", "review_adjudication_prompt.md").read_text(encoding="utf-8")

    pending_calls = 0
    for path in samples:
        sample_id = sample_row(path).get("sample_id", path.name)
        if args.only_stage:
            pending_calls += int((sample_id, args.only_stage) not in records)
        else:
            pending_calls += int((sample_id, "timing") not in records)
            pending_calls += int((sample_id, "semantics") not in records)
    if args.dry_run:
        print(f"samples={len(samples)} minimum_pending_calls={pending_calls} resumable_output={args.output}")
        return 0

    process_lock = acquire_process_lock(args.output.with_suffix(".lock"))

    api_key = get_env_first(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    if not api_key:
        raise SystemExit("Set QWEN_API_KEY or DASHSCOPE_API_KEY in this process before review.")
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_models = configured_models
    cap = model_token_cap()
    reserve = model_token_reserve()
    # Include this run's own log (and focused-audit logs) in the per-model
    # accounting.  Otherwise a resumed region-only run could ignore tokens it
    # has already spent and cross the configured 90% safety threshold.
    usage_paths = [args.output, repo_path("reports", "qwen_labels.jsonl"), *repo_path("reports").glob("qwen*.jsonl")]
    usage_totals = combined_usage(list(dict.fromkeys(usage_paths)))
    blocked: set[str] = set()

    if forced_stages:
        for path in samples:
            append_jsonl(
                args.output,
                {
                    "sample_id": sample_row(path).get("sample_id", path.name),
                    "stage": "reset",
                    "model": "",
                    "usage": None,
                    "result": {"stages": sorted(forced_stages)},
                    "error": "",
                },
            )

    for index, path in enumerate(samples, 1):
        row = sample_row(path)
        sample_id = row.get("sample_id", path.name)
        video_path = path / "video.mp4"
        audio_candidates = audio_transient_candidates(path / "audio.wav")
        source = source_text(path)

        if args.only_stage == "fielding":
            fielding_prompt = repo_path("prompts", "ground_ball_fielding_region_prompt.md").read_text(encoding="utf-8")
            fielding_record = records.get((sample_id, "fielding"))
            if fielding_record:
                prior = fielding_record.get("result") or {}
                try:
                    valid_region = int(prior.get("region")) in {1, 2, 3, 4}
                except (TypeError, ValueError):
                    valid_region = False
                if prior.get("receiving_moment_visible") is not True or not valid_region:
                    # An unresolved first-control frame is not a completed
                    # review; allow the next model to inspect it.
                    fielding_record = None
            if fielding_record is None:
                # Do not pass the old label, source metadata, or a player name:
                # the model must identify the first receiving fielder directly
                # from video.
                fielding_record = invoke(sample_id, "fielding", video_path, fielding_prompt, args.output, models_for_stage(all_models, "fielding"), usage_totals, blocked, api_key, base_url, cap, reserve)
                if fielding_record:
                    records[(sample_id, "fielding")] = fielding_record
            print(f"{index}/{len(samples)} {sample_id}: fielding region reviewed" if fielding_record else f"{sample_id}: fielding review failed")
            if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                write_summary(samples, records, args.summary)
            continue

        if args.only_stage == "semantics":
            semantics_record = records.get((sample_id, "semantics"))
            if semantics_record is None:
                semantics_record = invoke(sample_id, "semantics", video_path, context_prompt(semantics_prompt, row, source, {}), args.output, models_for_stage(all_models, "semantics"), usage_totals, blocked, api_key, base_url, cap, reserve)
                if semantics_record:
                    records[(sample_id, "semantics")] = semantics_record
            print(f"{index}/{len(samples)} {sample_id}: region reviewed" if semantics_record else f"{sample_id}: semantics failed")
            if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                write_summary(samples, records, args.summary)
            continue

        if args.only_stage == "timing":
            timing_record = records.get((sample_id, "timing"))
            if timing_record is None:
                timing_record = invoke(sample_id, "timing", video_path, context_prompt(timing_prompt, row, source, audio_candidates=audio_candidates), args.output, models_for_stage(all_models, "timing"), usage_totals, blocked, api_key, base_url, cap, reserve)
                if timing_record:
                    records[(sample_id, "timing")] = timing_record
            print(f"{index}/{len(samples)} {sample_id}: timing reviewed" if timing_record else f"{sample_id}: timing failed")
            if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
                write_summary(samples, records, args.summary)
            continue

        timing_record = records.get((sample_id, "timing"))
        if timing_record is None:
            timing_record = invoke(sample_id, "timing", video_path, context_prompt(timing_prompt, row, source, audio_candidates=audio_candidates), args.output, models_for_stage(all_models, "timing"), usage_totals, blocked, api_key, base_url, cap, reserve)
            if timing_record:
                records[(sample_id, "timing")] = timing_record
        if timing_record is None:
            print(f"{sample_id}: timing failed")
            continue

        timing_result = timing_record["result"]
        semantics_record = records.get((sample_id, "semantics"))
        if semantics_record is None:
            semantics_record = invoke(sample_id, "semantics", video_path, context_prompt(semantics_prompt, row, source, timing_result), args.output, models_for_stage(all_models, "semantics"), usage_totals, blocked, api_key, base_url, cap, reserve)
            if semantics_record:
                records[(sample_id, "semantics")] = semantics_record
        if semantics_record is None:
            print(f"{sample_id}: semantics failed")
            continue

        semantics_result = semantics_record["result"]
        needs_adjudication = timing_needs_change(row, timing_result) or semantics_needs_change(row, semantics_result) or min(float(timing_result.get("confidence") or 0), float(semantics_result.get("confidence") or 0)) < 0.85
        if needs_adjudication and (sample_id, "adjudication") not in records:
            prior = {"timing": timing_result, "semantics": semantics_result}
            adjudication_record = invoke(sample_id, "adjudication", video_path, context_prompt(adjudication_prompt, row, source, prior), args.output, models_for_stage(all_models, "adjudication"), usage_totals, blocked, api_key, base_url, cap, reserve)
            if adjudication_record:
                records[(sample_id, "adjudication")] = adjudication_record
        print(f"{index}/{len(samples)} {sample_id}: reviewed")
        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            write_summary(samples, records, args.summary)

    write_summary(samples, records, args.summary)
    process_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
