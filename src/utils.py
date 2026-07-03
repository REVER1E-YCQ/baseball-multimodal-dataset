"""
Shared utilities: sample ID generation, audio extraction, file validation.
"""

import re
import subprocess
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================
# Sample ID management
# ============================================================
def get_next_sample_ids(collector_dir: Path, label: str,
                        count: int) -> list[str]:
    """Scan collector_dir and return the next `count` sample IDs.

    Example: If G_001..G_025 exist, returns ['G_026', 'G_027'] for count=2.
    """
    prefix = "G_" if label == "ground_ball" else "F_"
    existing_nums = []

    for item in collector_dir.iterdir():
        if item.is_dir() and item.name.startswith(prefix):
            try:
                num = int(item.name.split("_")[1])
                existing_nums.append(num)
            except (IndexError, ValueError):
                pass

    start = max(existing_nums) + 1 if existing_nums else 1
    return [f"{prefix}{n:03d}" for n in range(start, start + count)]


# ============================================================
# Audio extraction
# ============================================================
def extract_audio(video_path: Path, output_wav_path: Path) -> bool:
    """Extract WAV audio from video using ffmpeg.

    Command: ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 2 audio.wav

    Returns True on success.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "2",
        str(output_wav_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and output_wav_path.exists():
            logger.info("Audio extracted: %s", output_wav_path)
            return True
        logger.error("ffmpeg failed: %s", result.stderr[:200])
        return False
    except FileNotFoundError:
        logger.error("ffmpeg not found — install it or add to PATH")
        return False


# ============================================================
# Sample directory validation
# ============================================================
REQUIRED_FILES = ["video.mp4", "audio.wav", "label.txt",
                  "sample.csv", "source.txt", "gemini_analysis.md"]


def validate_sample_dir(sample_dir: Path) -> list[str]:
    """Check a sample directory has all required files. Returns missing."""
    return [f for f in REQUIRED_FILES if not (sample_dir / f).exists()]


def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
# Output file writers
# ============================================================
def write_sample_csv(sample_dir: Path, result) -> None:
    """Write sample.csv from analysis result (dataclass)."""
    from dataclasses import asdict
    d = asdict(result)
    # Build CSV line matching existing format
    if result.label == "ground_ball":
        line = (
            f"{d['sample_id']},{d['label']},{d['region']},"
            f"{d['strength']},{d['bounce']},"
            f"{d['event_start']},{d['event_end']}\n"
        )
        header = "sample_id,label,region,strength,bounce,event_start,event_end\n"
    else:
        line = (
            f"{d['sample_id']},{d['label']},{d['landing_zone']},"
            f"{d['strength']},{d['trajectory_type']},"
            f"{d['event_start']},{d['event_end']}\n"
        )
        header = "sample_id,label,landing_zone,strength,trajectory_type,event_start,event_end\n"

    with open(sample_dir / "sample.csv", "w", encoding="utf-8") as f:
        f.write(header + line)
    logger.info("Wrote sample.csv for %s", d['sample_id'])


def write_label_txt(sample_dir: Path, result) -> None:
    """Write label.txt in the space-separated format."""
    from dataclasses import asdict
    d = asdict(result)

    if result.label == "ground_ball":
        strength_code = d['strength'][0].upper()  # low→L, medium→M, high→H
        bounce_code = d['bounce'][0].upper()      # yes→Y, no→N
        region = d['region']
        line = (
            f"{d['event_start']:.6f} {d['event_end']:.6f} "
            f"{d['label']}|{region}|{strength_code}|{bounce_code}\n"
        )
    else:
        strength_code = d['strength'][0].upper()
        line = (
            f"{d['event_start']:.6f} {d['event_end']:.6f} "
            f"{d['label']}|{d['landing_zone']}|{strength_code}|{d['trajectory_type']}\n"
        )

    with open(sample_dir / "label.txt", "w", encoding="utf-8") as f:
        f.write(line)
    logger.info("Wrote label.txt for %s", d['sample_id'])


def write_source_txt(sample_dir: Path, title: str, url: str) -> None:
    """Write source.txt in canonical format."""
    content = f"video_title: {title}\nvideo_url: {url}\n"
    with open(sample_dir / "source.txt", "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Wrote source.txt for %s", sample_dir.name)


def write_gemini_analysis(sample_dir: Path, response_text: str,
                          sample_id: str, source_title: str,
                          source_url: str) -> None:
    """Write gemini_analysis.md with full analysis text."""
    content = (
        f"# Gemini Analysis — {sample_id}\n\n"
        f"**Source**: {source_title}\n"
        f"**URL**: {source_url}\n\n"
        f"---\n\n"
        f"{response_text}\n"
    )
    with open(sample_dir / "gemini_analysis.md", "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Wrote gemini_analysis.md for %s", sample_id)
