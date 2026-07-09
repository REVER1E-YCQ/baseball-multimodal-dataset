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

from common import append_jsonl, get_env_first, read_csv, repo_path, write_csv


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


def load_models() -> list[str]:
    env = os.getenv("QWEN_MODEL_FALLBACKS")
    if env:
        return [item.strip() for item in env.split(",") if item.strip()]
    cfg_path = repo_path("config", "qwen_models.json")
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))["fallback_models"]
    return ["qwen3.5-omni-plus", "qwen3.5-omni-flash", "qwen3-omni-flash", "qwen-omni-turbo-latest"]


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
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = read_csv(args.clips_manifest)
    pending = [r for r in rows if r.get("status") in {"pending", "dry_run", ""}]
    if args.limit:
        pending = pending[: args.limit]

    prompt = args.prompt.read_text(encoding="utf-8")
    api_key = get_env_first(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    base_url = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    models = load_models()

    if args.dry_run:
        print(f"Would label {len(pending)} clips with models: {', '.join(models)}")
        return 0
    if not api_key:
        raise SystemExit("Set QWEN_API_KEY or DASHSCOPE_API_KEY before labeling.")

    row_by_clip = {row["clip_id"]: row for row in rows}
    for row in pending:
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
