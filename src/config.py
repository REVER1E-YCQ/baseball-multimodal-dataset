"""
Configuration loader: reads config.yaml and .env, merges them,
provides typed access to all settings via a single AppConfig object.

Usage:
    from src.config import load_config
    config = load_config()
    print(config.gemini.model)  # 'gemini-2.5-pro'
"""

import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml

# Add project root for .env loading
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PipelineConfig:
    collector_name: str = "Zichen_Yang"
    label: str = "ground_ball"
    num_videos: int = 2
    video_duration_min: int = 60
    video_duration_max: int = 120


@dataclass
class PathsConfig:
    dataset_root: Path = Path("dataset")
    log_dir: Path = Path("logs")
    temp_dir: Path = Path("temp")

    def __post_init__(self):
        self.dataset_root = PROJECT_ROOT / self.dataset_root
        self.log_dir = PROJECT_ROOT / self.log_dir
        self.temp_dir = PROJECT_ROOT / self.temp_dir


@dataclass
class ScraperConfig:
    timeout_seconds: int = 120
    retry_count: int = 3
    retry_delay_seconds: int = 10
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
    rate_limit_bytes: str = "2M"
    video_format: str = "best[height<=720]"


@dataclass
class GeminiConfig:
    model: str = "gemini-2.5-pro"
    poll_interval_seconds: int = 5
    poll_timeout_seconds: int = 300
    temperature: float = 0.4
    max_output_tokens: int = 4096


@dataclass
class GitConfig:
    auto_commit: bool = True
    auto_push: bool = True
    commit_message_template: str = (
        "Add {collector} {label} sample {sample_id}"
    )
    branch: str = "zichen-yang"
    remote: str = "origin"


@dataclass
class NetworkConfig:
    check_mlb_reachable: bool = True
    mlb_test_url: str = "https://www.mlb.com"
    mlb_stats_api_base: str = "https://statsapi.mlb.com/api/v1"


@dataclass
class AppConfig:
    """Unified application configuration."""
    pipeline: PipelineConfig
    paths: PathsConfig
    scraper: ScraperConfig
    gemini: GeminiConfig
    git: GitConfig
    network: NetworkConfig
    gemini_api_key: Optional[str] = None
    proxy_url: Optional[str] = None

    @property
    def prompt_file(self) -> Path:
        return PROJECT_ROOT / "test" / "prompts" / "baseball_hit_detection.txt"


def _load_env() -> dict:
    """Load .env file manually (avoid python-dotenv import)."""
    env = {}
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    # Also check os.environ
    env["GEMINI_API_KEY"] = env.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    env["PROXY_URL"] = env.get("PROXY_URL") or os.environ.get("PROXY_URL", "")
    return env


def load_config(config_path: str = None) -> AppConfig:
    """Load config.yaml + .env → AppConfig."""
    if config_path is None:
        config_path = PROJECT_ROOT / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    env = _load_env()

    return AppConfig(
        pipeline=PipelineConfig(**raw.get("pipeline", {})),
        paths=PathsConfig(**raw.get("paths", {})),
        scraper=ScraperConfig(**raw.get("scraper", {})),
        gemini=GeminiConfig(**raw.get("gemini", {})),
        git=GitConfig(**raw.get("git", {})),
        network=NetworkConfig(**raw.get("network", {})),
        gemini_api_key=env.get("GEMINI_API_KEY") or None,
        proxy_url=env.get("PROXY_URL") or None,
    )
