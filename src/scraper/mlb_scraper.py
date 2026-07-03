"""
MLB video scraper using yt-dlp + pybaseball + MLB Stats API.

Priority:
  1. pybaseball Statcast to find plays matching target bb_type
  2. MLB Stats API to get highlight video URLs
  3. yt-dlp MLB extractor to download

Anti-scraping:
  - Rate-limited downloads
  - Realistic User-Agent
  - Retry with backoff
  - 429/403 detection → immediate stop

No YouTube fallback — if MLB fails, stop and report.
"""

import os
import re
import sys
import json
import time
import random
import logging
import subprocess
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from src.config import AppConfig

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Metadata for a downloaded video."""
    title: str
    url: str
    duration_seconds: float
    local_path: Path
    sample_id: str


class MLBDownloadError(Exception):
    """Raised when MLB video download fails."""


class AntiScrapingError(MLBDownloadError):
    """Raised when anti-scraping protection is detected (429/403)."""


class MLBScraper:
    """Downloads MLB hitting videos using yt-dlp + MLB Stats API."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.proxy = self._build_proxy()

    def _build_proxy(self) -> dict:
        proxy_url = self.config.proxy_url
        if not proxy_url:
            return {}
        return {"http": proxy_url, "https": proxy_url}

    # ================================================================
    # Public API
    # ================================================================
    def find_video_urls(self, label: str,
                        count: int) -> list[tuple[str, str]]:
        """Find MLB video URLs matching the target label.

        Returns list of (url, title) tuples. Raises MLBDownloadError
        if not enough videos found.
        """
        logger.info("Searching for %d x '%s' videos on MLB...", count, label)

        urls = self._find_via_statcast(label, count)

        if len(urls) < count:
            raise MLBDownloadError(
                f"MLB: only found {len(urls)} '{label}' videos "
                f"(needed {count}).\n"
                "Possible causes:\n"
                "  - MLB.com blocked from your network → set PROXY_URL in .env\n"
                "  - Not enough recent games with this hit type\n"
                "  - pybaseball/Statcast API unavailable\n"
                "Pipeline STOPPED. No YouTube fallback is used."
            )

        random.shuffle(urls)
        return urls[:count]

    def download_video(self, url: str, sample_dir: Path,
                       sample_id: str, title: str = "") -> VideoMetadata:
        """Download a single video via yt-dlp to sample_dir/video.mp4.

        Raises MLBDownloadError or AntiScrapingError on failure.
        """
        output_path = sample_dir / "video.mp4"

        # Ensure sample directory exists
        sample_dir.mkdir(parents=True, exist_ok=True)

        ydl_opts = {
            "outtmpl": str(output_path),
            "format": self.config.scraper.video_format,
            "match_filter": self._duration_filter,
            "quiet": False,
            "no_warnings": False,
            "user_agent": self.config.scraper.user_agent,
            "socket_timeout": self.config.scraper.timeout_seconds,
            "retries": self.config.scraper.retry_count,
            "sleep_interval": 5,
            "max_sleep_interval": 15,
            "ratelimit": 2 * 1024 * 1024,  # 2 MB/s (must be int, not str)
        }

        if self.proxy:
            ydl_opts["proxy"] = self.proxy.get("https", "")

        logger.info("Downloading %s → %s", url, output_path)

        try:
            import yt_dlp
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                duration = info.get("duration", 0) if info else 0

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise MLBDownloadError(f"Download produced empty file: {output_path}")

            logger.info("Downloaded (%.0fs) → %s", duration, output_path)
            return VideoMetadata(
                title=title or info.get("title", ""),
                url=url,
                duration_seconds=duration,
                local_path=output_path,
                sample_id=sample_id,
            )

        except yt_dlp.DownloadError as e:
            error_str = str(e)
            if "429" in error_str or "403" in error_str:
                raise AntiScrapingError(
                    f"MLB anti-scraping triggered (HTTP 429/403)!\n"
                    f"URL: {url}\n"
                    "Pipeline STOPPED. Wait and try again, or use a proxy."
                ) from e
            raise MLBDownloadError(f"yt-dlp download failed: {e}") from e

    # ================================================================
    # Internal: Statcast + MLB Stats API
    # ================================================================
    def _find_via_statcast(self, label: str,
                           count: int) -> list[tuple[str, str]]:
        """Use pybaseball Statcast to find hit plays, then get video URLs."""
        try:
            from pybaseball import statcast
        except ImportError:
            logger.warning("pybaseball not available, trying MLB Stats API directly")
            return self._find_via_mlb_api(label, count)

        # Map label to Statcast bb_type
        bb_type_map = {
            "ground_ball": "ground_ball",
            "fly_ball": "fly_ball",
        }
        bb_type = bb_type_map.get(label, label)

        urls = []
        today = time.strftime("%Y-%m-%d")

        try:
            # Get recent Statcast data (last 3 days)
            from datetime import datetime, timedelta
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=7)

            logger.info("Querying Statcast: %s → %s, bb_type=%s",
                         start_dt.strftime("%Y-%m-%d"), today, bb_type)

            data = statcast(
                start_dt=start_dt.strftime("%Y-%m-%d"),
                end_dt=today,
            )

            if data is None or data.empty:
                logger.warning("Statcast returned no data")
                return self._find_via_mlb_api(label, count)

            # Filter by bb_type
            hits = data[data["bb_type"] == bb_type] if "bb_type" in data.columns else data
            logger.info("Statcast found %d plays (filtered by %s)", len(hits), bb_type)

            # Get video URLs from game highlights
            seen_games = set()
            for _, row in hits.iterrows():
                game_pk = row.get("game_pk")
                if not game_pk or game_pk in seen_games:
                    continue
                seen_games.add(game_pk)

                video_info = self._get_game_highlight(int(game_pk))
                if video_info:
                    urls.append(video_info)
                    if len(urls) >= count * 3:  # Get extra for random selection
                        break

        except Exception as e:
            logger.warning("Statcast query failed: %s", e)
            return self._find_via_mlb_api(label, count)

        logger.info("Found %d video URLs via Statcast", len(urls))
        return urls

    def _get_game_highlight(self, game_pk: int) -> Optional[tuple[str, str]]:
        """Get a highlight video URL from a specific game.

        Uses the MLB Content API to fetch highlights, filters by duration
        (1-2 min), and prefers hitting-related clips.
        """
        import urllib.request

        api_url = (
            f"{self.config.network.mlb_stats_api_base}/game/{game_pk}/content"
        )

        try:
            req = urllib.request.Request(api_url)
            req.add_header("User-Agent", self.config.scraper.user_agent)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())

            # Navigate: highlights → highlights → items
            items = (
                data.get("highlights", {})
                .get("highlights", {})
                .get("items", [])
            )

            if not items:
                logger.debug("Game %s: no highlight items", game_pk)
                return None

            candidates = []

            for h in items:
                # Parse duration (format: "HH:MM:SS" or "MM:SS")
                dur_str = h.get("duration", "0:00")
                seconds = self._parse_duration(dur_str)

                # Filter by duration range
                if not (self.config.pipeline.video_duration_min <=
                        seconds <= self.config.pipeline.video_duration_max):
                    continue

                # Get playback URL (MLB CDN URL, needs format suffix)
                playbacks = h.get("playbacks", [])
                if not playbacks:
                    continue

                base_url = playbacks[0].get("url", "")
                if not base_url:
                    continue

                # Build MLB.com video page URL (yt-dlp works with this)
                slug = h.get("slug", "")
                if slug:
                    video_url = f"https://www.mlb.com/video/{slug}"
                else:
                    # Fallback: CDN URL with format suffix
                    video_url = f"{base_url}_720p.mp4"

                title = h.get("headline", "") or h.get("blurb", "")

                # Check keywords for hitting relevance
                keywords = [
                    kw.get("value", "") for kw in h.get("keywordsAll", [])
                ]
                kw_str = " ".join(keywords).lower()
                is_hitting = any(k in kw_str for k in [
                    "hitting", "ground", "in-game-highlight",
                    "hit", "single", "double", "triple",
                    "ground-out", "groundout",
                ])

                candidates.append({
                    "url": video_url,
                    "title": title,
                    "duration": seconds,
                    "is_hitting": is_hitting,
                })

            if not candidates:
                logger.debug("Game %s: no 1-2 min highlights", game_pk)
                return None

            # Prefer hitting-related clips
            hitting = [c for c in candidates if c["is_hitting"]]
            if hitting:
                chosen = random.choice(hitting)
            else:
                chosen = random.choice(candidates)

            return (chosen["url"], chosen["title"])

        except Exception as e:
            logger.debug("Game %s highlight error: %s", game_pk, e)
            return None

    @staticmethod
    def _parse_duration(dur_str: str) -> int:
        """Parse 'HH:MM:SS' or 'MM:SS' to total seconds."""
        parts = dur_str.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        return 0

    def _find_via_mlb_api(self, label: str,
                          count: int) -> list[tuple[str, str]]:
        """Fallback: search MLB.com video page with yt-dlp."""
        import urllib.request

        urls = []
        search_term = label.replace("_", " ")

        # Build MLB.com video search URL
        search_url = (
            "https://www.mlb.com/video/search?"
            f"q={search_term.replace(' ', '+')}"
        )

        logger.info("Searching MLB.com: %s", search_url)

        try:
            # Use yt-dlp to extract video URLs from search page
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "extract_flat": True,
                "force_generic_extractor": False,
            }
            if self.proxy:
                ydl_opts["proxy"] = self.proxy.get("https", "")

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(search_url, download=False)
                if info and "entries" in info:
                    for entry in info["entries"]:
                        if entry is None:
                            continue
                        dur = entry.get("duration", 0) or 0
                        if self.config.pipeline.video_duration_min <= dur <= self.config.pipeline.video_duration_max:
                            urls.append((
                                entry.get("webpage_url") or entry.get("url", ""),
                                entry.get("title", ""),
                            ))
        except Exception as e:
            logger.warning("MLB.com search failed: %s", e)

        logger.info("Found %d video URLs via MLB.com search", len(urls))
        return urls

    # ================================================================
    # Helpers
    # ================================================================
    def _duration_filter(self, info: dict) -> Optional[str]:
        """yt-dlp match_filter: only allow videos 1-2 minutes."""
        duration = info.get("duration")
        if duration is None:
            return None  # Allow unknown durations
        min_dur = self.config.pipeline.video_duration_min
        max_dur = self.config.pipeline.video_duration_max
        if min_dur <= duration <= max_dur:
            return None  # Accept
        return f"Duration {duration}s outside {min_dur}-{max_dur}s range"
