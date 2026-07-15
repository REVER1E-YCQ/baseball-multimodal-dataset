from __future__ import annotations

import asyncio
import base64
import json
import tempfile
import time
import subprocess
from pathlib import Path
from typing import Any

import websockets

from common import tool_path
from qwen_omni_label import AuthError, ModelQuotaError


DEFAULT_ENDPOINT = "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"


def extract_pcm(video_path: Path) -> bytes:
    proc = subprocess.run(
        [
            tool_path("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return proc.stdout


def extract_timestamped_frames(video_path: Path, output_dir: Path, fps: float = 4.0) -> list[Path]:
    filter_graph = (
        f"fps={fps},scale=512:-2,"
        "drawtext=text='%{pts\\:hms}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65"
    )
    subprocess.run(
        [
            tool_path("ffmpeg"),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            filter_graph,
            "-q:v",
            "6",
            str(output_dir / "frame_%04d.jpg"),
        ],
        capture_output=True,
        check=True,
    )
    frames = sorted(output_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("Realtime frame extraction produced no images.")
    oversized = [path for path in frames if path.stat().st_size > 190_000]
    if oversized:
        raise RuntimeError(f"Realtime JPEG exceeds 190 KB recommendation: {oversized[0]}")
    return frames


def realtime_total_tokens(usage: dict[str, Any]) -> int:
    return int(usage.get("total_tokens") or 0)


async def _call(
    model: str,
    video_path: Path,
    prompt: str,
    api_key: str,
    endpoint: str,
    fps: float,
) -> tuple[str, dict[str, Any]]:
    url = endpoint.rstrip("/") + f"?model={model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    pcm = extract_pcm(video_path)
    with tempfile.TemporaryDirectory(prefix="qwen_realtime_frames_") as tmp:
        frames = extract_timestamped_frames(video_path, Path(tmp), fps)
        try:
            async with websockets.connect(
                url,
                additional_headers=headers,
                open_timeout=30,
                close_timeout=10,
                max_size=8 * 1024 * 1024,
            ) as ws:
                now = int(time.time() * 1000)
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "event_id": f"session_{now}",
                            "session": {
                                "modalities": ["text"],
                                "instructions": prompt,
                                "input_audio_format": "pcm",
                                "turn_detection": None,
                            },
                        }
                    )
                )
                await asyncio.sleep(0.15)
                frame_interval = 1.0 / fps
                pcm_bytes_per_frame = int(16000 * 2 * frame_interval)
                pcm_cursor = 0
                for index, frame in enumerate(frames):
                    pcm_stop = min(len(pcm), pcm_cursor + pcm_bytes_per_frame)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": f"audio_{now}_{pcm_cursor}",
                                "audio": base64.b64encode(pcm[pcm_cursor:pcm_stop]).decode("ascii"),
                            }
                        )
                    )
                    pcm_cursor = pcm_stop
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_image_buffer.append",
                                "event_id": f"image_{now}_{index}",
                                "image": base64.b64encode(frame.read_bytes()).decode("ascii"),
                            }
                        )
                    )
                    # Realtime models sample a live video stream by wall-clock time.
                    # Preserve that cadence so later play frames aren't collapsed.
                    await asyncio.sleep(frame_interval)
                for offset in range(pcm_cursor, len(pcm), 6400):
                    await ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": f"audio_{now}_{offset}",
                                "audio": base64.b64encode(pcm[offset : offset + 6400]).decode("ascii"),
                            }
                        )
                    )
                    await asyncio.sleep(0.005)
                await ws.send(json.dumps({"type": "input_audio_buffer.commit", "event_id": f"commit_{now}"}))
                await ws.send(json.dumps({"type": "response.create", "event_id": f"response_{now}"}))
                async with asyncio.timeout(180):
                    async for raw in ws:
                        event = json.loads(raw)
                        event_type = event.get("type")
                        if event_type == "error":
                            detail = json.dumps(event.get("error") or event, ensure_ascii=False)
                            if "AllocationQuota" in detail or "quota" in detail.lower():
                                raise ModelQuotaError(f"{model}: realtime quota unavailable")
                            if "auth" in detail.lower() or "api key" in detail.lower():
                                raise AuthError("Realtime authentication failed; check local API key.")
                            raise RuntimeError(detail)
                        if event_type == "response.text.delta":
                            text_parts.append(str(event.get("delta") or ""))
                        elif event_type == "response.text.done" and event.get("text"):
                            text_parts = [str(event["text"])]
                        elif event_type == "response.done":
                            response = event.get("response") or {}
                            usage = response.get("usage") or event.get("usage") or {}
                            break
        except websockets.exceptions.InvalidStatus as exc:
            if exc.response.status_code in {401, 403}:
                raise AuthError("Realtime authentication failed; check local API key.") from exc
            raise
    return "".join(text_parts), usage


def call_qwen_realtime(
    model: str,
    video_path: Path,
    prompt: str,
    api_key: str,
    endpoint: str = DEFAULT_ENDPOINT,
    fps: float = 4.0,
) -> tuple[str, dict[str, Any]]:
    delays = (0, 3, 10)
    last_error: Exception | None = None
    for delay in delays:
        if delay:
            time.sleep(delay)
        try:
            return asyncio.run(_call(model, video_path, prompt, api_key, endpoint, fps))
        except (TimeoutError, websockets.exceptions.ConnectionClosedError, RuntimeError) as exc:
            message = str(exc).lower()
            transient = any(
                marker in message
                for marker in ("too many requests", "throttl", "capacity", "thread pool", "max_workers", "1011")
            ) or isinstance(exc, TimeoutError)
            if not transient:
                raise
            last_error = exc
    raise RuntimeError(f"{model}: realtime service remained capacity-limited after retries: {last_error}")
