"""
Gemini video analysis with strict multi-session isolation.

CRITICAL RULES (防污染):
  - Each video → NEW genai.Client instance
  - Unique UUID session_id per analysis
  - Uploaded file deleted from Gemini server after analysis
  - Client destroyed (del) after use — never reused
  - All local state cleared between analyses

Model: Gemini 2.5 Pro (supports video understanding natively)
"""

import uuid
import json
import logging
import time
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from src.config import AppConfig

logger = logging.getLogger(__name__)


# ============================================================
# Structured results
# ============================================================
@dataclass
class GroundBallResult:
    sample_id: str
    label: str = "ground_ball"
    region: int = 2
    strength: str = "medium"
    bounce: str = "no"
    event_start: float = 0.0
    event_end: float = 0.0


@dataclass
class FlyBallResult:
    sample_id: str
    label: str = "fly_ball"
    landing_zone: int = 5
    strength: str = "medium"
    trajectory_type: str = "fly"
    event_start: float = 0.0
    event_end: float = 0.0


# ============================================================
# Prompt templates
# ============================================================
def _get_analysis_prompt(label: str) -> str:
    """Full baseball hit-detection prompt (user-provided)."""
    return """请仔细分析整个棒球视频，不要只看开头或只找第一个击球点。这个视频中可能有多个击球点，请从头到尾逐帧/分段检查，找出所有"球棒击中球"的瞬间。

任务：

找出视频中每一个击球点。 对每个击球点判断击球类型。 重点区分：地滚球、平飞球、高飞球、界外球、未击中/擦棒。 如果同一个击球在后面以慢动作或不同角度重播，请标记为"重播"，不要当成新的击球点重复计数。 击球点定义： "击球点"是球棒与棒球发生接触的瞬间。如果看不清接触帧，就用球离开球棒后的第一帧作为近似时间点。

分类规则：

地滚球 Ground ball： 球离开球棒后很快向下，低角度前进，并且很早接触地面或草皮，通常出现弹跳、滚动、穿越内野、被内野手处理等情况。 高飞球 Fly ball： 球离开球棒后明显向上飞，轨迹有较高弧线，停留空中较久，通常外野手抬头追球，球可能被接杀、落到外野深处、或成为本垒打。 平飞球 Line drive： 球速度快、轨迹较平直，离地较高但弧线不明显，不是很快落地滚动，也不是明显高抛弧线。 弹地高飞/小飞球 Pop-up： 球几乎垂直向上或很高但距离不远，通常内野手或捕手抬头等待。 界外球 Foul ball： 如果球明显飞向界外区域，或转播/字幕/裁判动作显示 foul，请标记为界外球，并尽量同时说明它是地滚球界外、飞球界外还是不确定。 未击中/无击球： 挥棒但没有明显接触球，或只是投捕过程，不算击球点。 擦棒 Foul tip / tipped： 如果只看到球轻微碰到球棒后进入捕手方向，且没有形成正常击出轨迹，标记为擦棒或不确定。 判断时请重点观察：

球离开球棒后的最初 1-2 秒轨迹。 球是否很快落地或弹跳。 内野手/外野手的反应方向。 转播镜头是否切到外野高空追球。 慢动作重播是否只是同一次击球的重复视角。 字幕、比分板、解说图形如果有帮助，可以作为辅助证据，但不要只依赖字幕。 请按以下格式输出，不要漏掉可能的击球点：

总共发现：N 个不同击球点

每个击球点：

时间戳：mm:ss 是否重播：否/是，如果是，关联到第几个击球点 击球类型：地滚球 / 高飞球 / 平飞球 / 小飞球 / 界外球 / 擦棒 / 不确定 置信度：高 / 中 / 低 依据：用 1-2 句话说明，例如"球离棒后低角度前进并很快落地弹跳，所以判断为地滚球"。 如果无法确定，请不要强行猜测，标记为"不确定"，并说明缺少什么画面证据"""


def _get_json_extraction_prompt(sample_id: str, label: str) -> str:
    """Compact prompt to also extract event timestamps as JSON.
    Appended after the main analysis prompt; Gemini returns JSON
    in addition to the natural-language analysis.
    """
    if label == "ground_ball":
        return f"""
在分析完以上内容后，请再输出一段纯JSON（不要markdown代码块），用于结构化数据提取：

{{
  "sample_id": "{sample_id}",
  "label": "ground_ball",
  "region": <1-4>,
  "strength": "<low|medium|high>",
  "bounce": "<yes|no>",
  "event_start": <float seconds>,
  "event_end": <float seconds>
}}

event_start: 投球/挥棒准备开始的时间点
event_end: 球被处理或初始运动结束的时间点
region: 1=左内野 2=中间内野 3=右内野 4=触击区
"""
    else:
        return f"""
在分析完以上内容后，请再输出一段纯JSON（不要markdown代码块），用于结构化数据提取：

{{
  "sample_id": "{sample_id}",
  "label": "fly_ball",
  "landing_zone": <1-9>,
  "strength": "<low|medium|high>",
  "trajectory_type": "<fly|line_drive|pop_fly>",
  "event_start": <float seconds>,
  "event_end": <float seconds>
}}
"""


# ============================================================
# Gemini Analyzer
# ============================================================
class GeminiAnalyzer:
    """Performs isolated video analysis using Gemini.

    ISOLATION PROTOCOL:
      1. session_id = uuid4()
      2. client = new genai.Client()
      3. Upload → poll ACTIVE → generate → parse
      4. Delete uploaded file from server
      5. del client
      → Repeat fresh for next video
    """

    def __init__(self, config: AppConfig):
        self.cfg = config
        self.api_key = config.gemini_api_key
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Please add it to .env file.\n"
                "Get a key at: https://aistudio.google.com/apikey"
            )

    def analyze_video(self, video_path: Path, sample_id: str,
                      label: str, source_url: str = "",
                      source_title: str = "") -> tuple:
        """Analyze one video → (analysis_text, structured_result).

        Returns:
            (full_response_text, GroundBallResult | FlyBallResult)
        """
        session_id = str(uuid.uuid4())
        logger.info("[%s] === New analysis session ===", session_id)
        logger.info("[%s] Sample: %s | Video: %s", session_id, sample_id, video_path)

        # STEP 1: Fresh client
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        uploaded_file = None

        try:
            # STEP 2: Upload video
            logger.info("[%s] Uploading video (%s bytes)...",
                         session_id, video_path.stat().st_size)
            uploaded_file = client.files.upload(file=str(video_path))
            logger.info("[%s] Uploaded: %s (state=%s)",
                         session_id, uploaded_file.name, uploaded_file.state)

            # STEP 3: Poll until ACTIVE
            uploaded_file = self._wait_for_active(
                client, uploaded_file.name, session_id
            )

            # STEP 4: Build combined prompt
            main_prompt = _get_analysis_prompt(label)
            json_prompt = _get_json_extraction_prompt(sample_id, label)
            full_prompt = main_prompt + "\n\n" + json_prompt

            logger.info("[%s] Sending to Gemini model=%s...",
                         session_id, self.cfg.gemini.model)

            # STEP 5: Generate
            response = client.models.generate_content(
                model=self.cfg.gemini.model,
                contents=[uploaded_file, full_prompt],
                config=types.GenerateContentConfig(
                    temperature=self.cfg.gemini.temperature,
                    max_output_tokens=self.cfg.gemini.max_output_tokens,
                ),
            )

            response_text = response.text or ""
            logger.info("[%s] Response received (%d chars)",
                         session_id, len(response_text))

            # STEP 6: Parse structured JSON from response
            result = self._parse_response(response_text, sample_id, label, session_id)

            # STEP 7: Cleanup
            if uploaded_file:
                client.files.delete(name=uploaded_file.name)
                logger.info("[%s] Deleted uploaded file from Gemini", session_id)

            return response_text, result

        except Exception as e:
            logger.exception("[%s] Analysis failed: %s", session_id, e)
            raise

        finally:
            # STEP 8: Destroy client
            del client
            logger.info("[%s] Client destroyed. Session complete.", session_id)

    # ================================================================
    # Internal helpers
    # ================================================================
    def _wait_for_active(self, client, file_name: str,
                         session_id: str):
        """Poll until uploaded file is ACTIVE."""
        start = time.time()
        while True:
            f = client.files.get(name=file_name)
            state = str(f.state) if hasattr(f.state, 'name') else str(f.state)

            if "ACTIVE" in state.upper():
                logger.info("[%s] File ACTIVE after %.1fs",
                             session_id, time.time() - start)
                return f

            elapsed = time.time() - start
            if elapsed > self.cfg.gemini.poll_timeout_seconds:
                raise TimeoutError(
                    f"[{session_id}] Video processing timed out "
                    f"after {elapsed:.0f}s. State: {state}"
                )

            logger.info("[%s] File state=%s, waiting %ds...",
                         session_id, state, self.cfg.gemini.poll_interval_seconds)
            time.sleep(self.cfg.gemini.poll_interval_seconds)

    def _parse_response(self, text: str, sample_id: str,
                        label: str, session_id: str):
        """Extract JSON from Gemini response. Falls back to defaults."""
        # Find JSON block in response
        # Look for the last { ... } block in the text
        json_str = None
        brace_start = text.rfind("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            json_str = text[brace_start:brace_end + 1]

        if json_str:
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError:
                # Try cleaning
                clean = json_str.replace("\n", " ").replace("  ", " ")
                try:
                    data = json.loads(clean)
                except json.JSONDecodeError as e:
                    logger.warning("[%s] JSON parse failed: %s", session_id, e)
                    data = {}
        else:
            logger.warning("[%s] No JSON block found in response", session_id)
            data = {}

        if label == "ground_ball":
            return GroundBallResult(
                sample_id=data.get("sample_id", sample_id),
                label="ground_ball",
                region=int(data.get("region", 2)),
                strength=str(data.get("strength", "medium")),
                bounce=str(data.get("bounce", "no")),
                event_start=float(data.get("event_start", 0.0)),
                event_end=float(data.get("event_end", 0.0)),
            )
        else:
            return FlyBallResult(
                sample_id=data.get("sample_id", sample_id),
                label="fly_ball",
                landing_zone=int(data.get("landing_zone", 5)),
                strength=str(data.get("strength", "medium")),
                trajectory_type=str(data.get("trajectory_type", "fly")),
                event_start=float(data.get("event_start", 0.0)),
                event_end=float(data.get("event_end", 0.0)),
            )
