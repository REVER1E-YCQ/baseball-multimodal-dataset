from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from common import append_jsonl, get_env_first, load_jsonl, read_csv, repo_path, write_csv


class AuthError(RuntimeError):
    pass


CLIP_FIELDS = [
    "clip_id",
    "source_id",
    "source_path",
    "clip_path",
    "audio_path",
    "start_time",
    "end_time",
    "expected_label",
    "status",
    "notes",
]

DEFAULT_MODEL_TOKEN_CAP = 800_000
DEFAULT_MODEL_TOKEN_RESERVE = 10_000


def load_models() -> list[str]:
    env = os.getenv("QWEN_MODEL_FALLBACKS")
    if env:
        return [item.strip() for item in env.split(",") if item.strip()]
    cfg_path = repo_path("config", "qwen_models.json")
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))["fallback_models"]
    return ["qwen3.5-omni-flash", "qwen3-omni-flash", "qwen-omni-turbo-latest", "qwen3.5-omni-plus"]


def usage_total_tokens(usage: dict[str, Any]) -> int:
    total = usage.get("total_tokens")
    if total is not None:
        try:
            return int(total)
        except (TypeError, ValueError):
            return 0
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    try:
        return int(prompt or 0) + int(completion or 0)
    except (TypeError, ValueError):
        return 0


def model_usage_totals(labels_path: Path) -> dict[str, int]:
    totals: dict[str, int] = {}
    for record in load_jsonl(labels_path):
        model = record.get("model")
        usage = record.get("usage") or {}
        if model and usage:
            totals[model] = totals.get(model, 0) + usage_total_tokens(usage)
    return totals


def model_token_cap() -> int:
    raw = os.getenv("QWEN_MODEL_TOKEN_CAP")
    if raw is None:
        return DEFAULT_MODEL_TOKEN_CAP
    try:
        return int(raw)
    except ValueError:
        raise SystemExit("QWEN_MODEL_TOKEN_CAP must be an integer token count, or 0 to disable.")


def model_token_reserve() -> int:
    raw = os.getenv("QWEN_MODEL_TOKEN_RESERVE")
    if raw is None:
        return DEFAULT_MODEL_TOKEN_RESERVE
    try:
        return int(raw)
    except ValueError:
        raise SystemExit("QWEN_MODEL_TOKEN_RESERVE must be an integer token count.")


def filter_models_by_usage(
    models: list[str],
    usage_totals: dict[str, int],
    cap: int,
    reserve: int,
    announced: set[str] | None = None,
) -> list[str]:
    if cap <= 0:
        return models
    available: list[str] = []
    for model in models:
        used = usage_totals.get(model, 0)
        if used + reserve >= cap:
            if announced is not None and model not in announced:
                print(
                    f"Skipping {model}: local_usage_total_tokens={used} "
                    f"+ reserve={reserve} >= cap={cap}"
                )
                announced.add(model)
            continue
        available.append(model)
    return available


def materialized_source_ids(dataset_root: Path = repo_path("dataset")) -> set[str]:
    source_ids: set[str] = set()
    for source_file in dataset_root.glob("*/*/*/source.txt"):
        try:
            for line in source_file.read_text(encoding="utf-8").splitlines():
                if line.startswith("source_id:"):
                    source_id = line.split(":", 1)[1].strip()
                    if source_id:
                        source_ids.add(source_id)
        except OSError:
            continue
    return source_ids


def data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "video/mp4"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if len(encoded) > 10_000_000:
        raise ValueError(f"Base64 payload is too large for local upload: {path} ({len(encoded)} chars)")
    return f"data:{mime};base64,{encoded}"


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_label(payload: dict[str, Any]) -> dict[str, Any]:
    label = str(payload.get("label", "uncertain")).strip().lower()
    if label not in {"ground_ball", "fly_ball", "reject", "uncertain"}:
        label = "uncertain"
    payload["label"] = label
    payload["confidence"] = float(payload.get("confidence") or 0.0)
    payload["event_start"] = float(payload.get("event_start") or 0.0)
    payload["event_end"] = float(payload.get("event_end") or 0.0)
    return payload


def _chunk_text(delta_content: Any) -> str:
    if isinstance(delta_content, str):
        return delta_content
    if isinstance(delta_content, list):
        parts = []
        for item in delta_content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return ""


def call_qwen(model: str, clip_path: Path, prompt: str, base_url: str, api_key: str) -> tuple[str, dict[str, Any] | None]:
    content = [
        {"type": "video_url", "video_url": {"url": data_url(clip_path)}},
        {"type": "text", "text": prompt},
    ]
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["text"],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
    }
    if model == "qwen3-omni-flash":
        body["enable_thinking"] = False

    endpoint = base_url.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    text_parts: list[str] = []
    usage = None
    try:
        response = urllib.request.urlopen(req, timeout=180)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        if exc.code in {401, 403} or "invalid_api_key" in detail or "Incorrect API key" in detail:
            raise AuthError(f"HTTP {exc.code}: authentication failed; check QWEN_API_KEY/DASHSCOPE_API_KEY")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text_parts.append(_chunk_text(delta.get("content")))
    return "".join(text_parts), usage


def main() -> int:
    parser = argparse.ArgumentParser(description="Label pending baseball clips with Qwen-Omni.")
    parser.add_argument("--clips-manifest", type=Path, default=repo_path("manifests", "clips_manifest.csv"))
    parser.add_argument("--prompt", type=Path, default=repo_path("prompts", "baseball_hit_labeling_prompt.md"))
    parser.add_argument("--output", type=Path, default=repo_path("reports", "qwen_labels.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--statuses",
        default="pending",
        help="Comma-separated clip statuses to process, for example pending,label_failed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.clips_manifest)
    wanted_statuses = {item.strip() for item in args.statuses.split(",") if item.strip()}
    materialized_sources = materialized_source_ids()
    skipped_materialized = 0
    pending = []
    for row in rows:
        if not (row.get("status") in wanted_statuses or (not row.get("status") and "" in wanted_statuses)):
            continue
        if row.get("source_id") in materialized_sources:
            skipped_materialized += 1
            continue
        pending.append(row)
    if args.limit:
        pending = pending[: args.limit]

    prompt = args.prompt.read_text(encoding="utf-8")
    api_key = get_env_first(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    all_models = load_models()
    cap = model_token_cap()
    reserve = model_token_reserve()
    usage_totals = model_usage_totals(args.output)
    announced_skips: set[str] = set()
    models = filter_models_by_usage(all_models, usage_totals, cap, reserve, announced_skips)

    if args.dry_run:
        if skipped_materialized:
            print(f"Skipped {skipped_materialized} clips whose source_id is already materialized.")
        print(f"Would label {len(pending)} clips with models: {', '.join(models)}")
        return 0
    if not api_key:
        raise SystemExit("Set QWEN_API_KEY or DASHSCOPE_API_KEY before labeling.")
    if not models:
        raise SystemExit("No Qwen models are below QWEN_MODEL_TOKEN_CAP; set QWEN_MODEL_TOKEN_CAP=0 to override.")

    row_by_clip = {row["clip_id"]: row for row in rows}
    for row in pending:
        models = filter_models_by_usage(all_models, usage_totals, cap, reserve, announced_skips)
        if not models:
            print("No Qwen models are below QWEN_MODEL_TOKEN_CAP; stopping before the next clip.")
            break
        clip_path = Path(row["clip_path"])
        if not clip_path.is_absolute():
            clip_path = repo_path(str(clip_path))
        result: dict[str, Any] | None = None
        last_error = ""
        used_model = ""
        usage = None
        started = time.time()

        for model in models:
            try:
                text, usage = call_qwen(model, clip_path, prompt, base_url, api_key)
                result = normalize_label(extract_json(text))
                used_model = model
                break
            except AuthError as exc:
                last_error = str(exc)
                print(last_error)
                return 2
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                continue

        record: dict[str, Any] = {
            "clip_id": row["clip_id"],
            "source_id": row.get("source_id", ""),
            "clip_path": row.get("clip_path", ""),
            "model": used_model,
            "usage": usage,
            "elapsed_seconds": round(time.time() - started, 3),
            "label": result,
            "error": last_error if result is None else "",
        }
        append_jsonl(args.output, record)
        if used_model and usage:
            usage_totals[used_model] = usage_totals.get(used_model, 0) + usage_total_tokens(usage)

        mutable = row_by_clip[row["clip_id"]]
        if result is None:
            mutable["status"] = "label_failed"
            mutable["notes"] = last_error[-300:]
        elif result["label"] in {"ground_ball", "fly_ball"} and result["confidence"] >= 0.70:
            mutable["status"] = "labeled"
            mutable["notes"] = f"{used_model}; confidence={result['confidence']:.2f}"
        else:
            mutable["status"] = "needs_review"
            mutable["notes"] = f"{used_model}; label={result['label']}; confidence={result['confidence']:.2f}"

        write_csv(args.clips_manifest, rows, CLIP_FIELDS)
        print(f"{row['clip_id']}: {mutable['status']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
