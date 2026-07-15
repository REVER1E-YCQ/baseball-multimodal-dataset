from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from typing import Any

import websockets

from common import get_env_first


async def probe(endpoint: str, model: str, api_key: str) -> tuple[str, dict[str, Any]]:
    url = endpoint.rstrip("/") + f"?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    async with websockets.connect(url, additional_headers=headers, open_timeout=30, close_timeout=10) as ws:
        await ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "event_id": f"probe_session_{int(time.time() * 1000)}",
                    "session": {
                        "modalities": ["text"],
                        "instructions": 'Return only this strict JSON object: {"realtime_probe":"ok"}',
                        "input_audio_format": "pcm",
                        "turn_detection": None,
                    },
                }
            )
        )
        silence = bytes(3200)
        await ws.send(
            json.dumps(
                {
                    "type": "input_audio_buffer.append",
                    "event_id": f"probe_audio_{int(time.time() * 1000)}",
                    "audio": base64.b64encode(silence).decode("ascii"),
                }
            )
        )
        await ws.send(json.dumps({"type": "input_audio_buffer.commit", "event_id": "probe_commit"}))
        await ws.send(json.dumps({"type": "response.create", "event_id": "probe_response"}))
        async with asyncio.timeout(60):
            async for raw in ws:
                event = json.loads(raw)
                event_type = event.get("type")
                if event_type == "error":
                    raise RuntimeError(json.dumps(event.get("error") or event, ensure_ascii=False))
                if event_type == "response.text.delta":
                    text_parts.append(str(event.get("delta") or ""))
                elif event_type == "response.text.done" and event.get("text"):
                    text_parts = [str(event["text"])]
                elif event_type == "response.done":
                    response = event.get("response") or {}
                    usage = response.get("usage") or event.get("usage") or {}
                    break
    return "".join(text_parts), usage


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Qwen Omni Realtime without exposing credentials.")
    parser.add_argument("--endpoint", default=os.getenv("QWEN_REALTIME_BASE_URL", "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"))
    parser.add_argument("--model", default="qwen3.5-omni-plus-realtime-2026-03-15")
    args = parser.parse_args()
    key = get_env_first(["QWEN_API_KEY", "DASHSCOPE_API_KEY"])
    if not key:
        raise SystemExit("QWEN_API_KEY or DASHSCOPE_API_KEY is required.")
    text, usage = asyncio.run(probe(args.endpoint, args.model, key))
    print(json.dumps({"model": args.model, "text": text, "usage": usage}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
