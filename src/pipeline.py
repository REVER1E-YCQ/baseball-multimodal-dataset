#!/usr/bin/env python3
"""
MLB Video Pipeline — Automated scraping → Gemini analysis → GitHub sync.

Usage:
    python -m src.pipeline                        # Full pipeline
    python -m src.pipeline --dry-run              # Config + network check only
    python -m src.pipeline --skip-scrape          # Re-analyze existing videos
    python -m src.pipeline --skip-sync            # Local only, no git push
    python -m src.pipeline --label ground_ball --count 2
"""

import sys
import logging
import argparse
from pathlib import Path

from src.config import load_config
from src.network import diagnose_network
from src.utils import (
    get_next_sample_ids, extract_audio, validate_sample_dir,
    ensure_dir, write_sample_csv, write_label_txt,
    write_source_txt, write_gemini_analysis,
)
from src.scraper.mlb_scraper import (
    MLBScraper, MLBDownloadError, AntiScrapingError,
)
from src.analyzer.gemini_analyzer import GeminiAnalyzer
from src.watcher.folder_watcher import wait_for_file_stable
from src.sync.github_sync import GitHubSync

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("pipeline")


def main():
    parser = argparse.ArgumentParser(description="MLB Video Pipeline")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate config and network, then exit")
    parser.add_argument("--skip-scrape", action="store_true",
                        help="Skip downloading, use existing videos")
    parser.add_argument("--skip-sync", action="store_true",
                        help="Skip git push (local only)")
    parser.add_argument("--label", type=str, default=None,
                        help="ground_ball | fly_ball (overrides config)")
    parser.add_argument("--count", type=int, default=None,
                        help="Number of videos (overrides config)")
    args = parser.parse_args()

    # ── Load config ──
    config = load_config()
    label = args.label or config.pipeline.label
    count = args.count or config.pipeline.num_videos
    collector = config.pipeline.collector_name

    print("=" * 60)
    print(f"MLB Pipeline — {label} x{count} → {collector}")
    print("=" * 60)

    # ── Validate API key ──
    if not config.gemini_api_key:
        logger.error("GEMINI_API_KEY not set. Add it to .env")
        logger.error("Get a key: https://aistudio.google.com/apikey")
        sys.exit(1)
    logger.info("API Key: %s...", config.gemini_api_key[:20])

    # ── Network check ──
    diagnosis = diagnose_network(config)
    logger.info("MLB reachable: %s | Proxy: %s",
                diagnosis["mlb_reachable"], diagnosis["proxy_configured"])
    logger.info(diagnosis["recommended_action"])

    if args.dry_run:
        logger.info("Dry run complete.")
        return

    # ── Setup directories ──
    collector_dir = (config.paths.dataset_root / label / collector)
    ensure_dir(collector_dir)
    config.paths.log_dir.mkdir(exist_ok=True)

    sample_ids = get_next_sample_ids(collector_dir, label, count)
    logger.info("Next sample IDs: %s", sample_ids)
    logger.info("Target: %s", collector_dir)

    # ── STAGE 1: Scrape ──
    video_metas = []

    if not args.skip_scrape:
        scraper = MLBScraper(config)

        try:
            urls = scraper.find_video_urls(label, count)
        except MLBDownloadError as e:
            logger.critical(str(e))
            sys.exit(2)

        for i, (url, title) in enumerate(urls):
            sample_id = sample_ids[i]
            sample_dir = collector_dir / sample_id

            print(f"\n{'─'*40}")
            print(f"[{i+1}/{count}] Downloading {sample_id}...")
            print(f"  URL: {url[:80]}...")
            print(f"  Title: {title}")

            try:
                meta = scraper.download_video(url, sample_dir, sample_id, title)
                meta = meta  # VideoMetadata
                # Wait for file to stabilize
                if wait_for_file_stable(sample_dir / "video.mp4", timeout=120):
                    video_metas.append({
                        "local_path": sample_dir / "video.mp4",
                        "sample_id": sample_id,
                        "sample_dir": sample_dir,
                        "url": url,
                        "title": title,
                    })
            except AntiScrapingError as e:
                logger.critical(str(e))
                sys.exit(2)
            except MLBDownloadError as e:
                logger.error("Download failed for %s: %s", sample_id, e)
                # Continue to next video
            except Exception as e:
                logger.exception("Unexpected error for %s", sample_id)
    else:
        # Skip scrape: look for existing videos
        logger.info("--skip-scrape: looking for existing videos...")
        for sample_id in sample_ids:
            sample_dir = collector_dir / sample_id
            video_path = sample_dir / "video.mp4"
            if video_path.exists() and video_path.stat().st_size > 0:
                video_metas.append({
                    "local_path": video_path,
                    "sample_id": sample_id,
                    "sample_dir": sample_dir,
                    "url": "",
                    "title": "",
                })
                logger.info("  Found existing: %s", video_path)
            else:
                logger.warning("  No video for %s", sample_id)

    if not video_metas:
        logger.error("No videos available for analysis. Exiting.")
        sys.exit(1)

    # ── STAGE 2: Gemini Analysis ──
    analyzer = GeminiAnalyzer(config)

    for meta in video_metas:
        sample_id = meta["sample_id"]
        sample_dir = meta["sample_dir"]
        video_path = meta["local_path"]
        source_url = meta["url"]
        source_title = meta["title"]

        print(f"\n{'─'*40}")
        print(f"Analyzing {sample_id} with Gemini {config.gemini.model}...")

        try:
            # Run Gemini (fresh session per video!)
            response_text, result = analyzer.analyze_video(
                video_path=video_path,
                sample_id=sample_id,
                label=label,
                source_url=source_url,
                source_title=source_title,
            )

            # ── STAGE 3: Write output files ──
            write_sample_csv(sample_dir, result)
            write_label_txt(sample_dir, result)
            write_source_txt(sample_dir, source_title, source_url)
            write_gemini_analysis(sample_dir, response_text,
                                  sample_id, source_title, source_url)

            # ── STAGE 4: Extract audio ──
            audio_path = sample_dir / "audio.wav"
            extract_audio(video_path, audio_path)

            # ── Validate ──
            missing = validate_sample_dir(sample_dir)
            if missing:
                logger.warning("%s missing files: %s", sample_id, missing)
            else:
                logger.info("%s — all 6 files present!", sample_id)

            # ── STAGE 5: Git sync ──
            if not args.skip_sync and config.git.auto_commit:
                syncer = GitHubSync(config)
                syncer.sync_sample(sample_dir, sample_id, label)

        except Exception as e:
            logger.exception("Analysis failed for %s: %s", sample_id, e)
            # Continue to next video

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Pipeline complete.")
    print(f"  Processed: {len(video_metas)} videos")
    print(f"  Output: {collector_dir}")
    print(f"  Logs: {config.paths.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
