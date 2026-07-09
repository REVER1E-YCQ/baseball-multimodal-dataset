from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from common import read_csv, safe_slug, write_csv, repo_path


FIELDS = [
    "source_id",
    "source_url",
    "video_title",
    "game_date",
    "teams",
    "batter",
    "event_text",
    "expected_label",
    "rights_note",
    "status",
    "local_path",
    "source_hash",
    "notes",
]


def fetch_json(url: str, retries: int = 3, backoff: float = 1.5) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "baseball-dataset-research/0.1"})
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_error}") from last_error


def schedule_game_pks(start_date: str, end_date: str) -> list[dict[str, str]]:
    params = urllib.parse.urlencode({"sportId": 1, "startDate": start_date, "endDate": end_date})
    url = f"https://statsapi.mlb.com/api/v1/schedule?{params}"
    payload = fetch_json(url)
    games: list[dict[str, str]] = []
    for date_row in payload.get("dates", []):
        game_date = date_row.get("date", "")
        for game in date_row.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {}).get("team", {}).get("name", "")
            home = teams.get("home", {}).get("team", {}).get("name", "")
            games.append({"game_pk": str(game.get("gamePk")), "game_date": game_date, "teams": f"{away} at {home}"})
    return games


def text_for_video(item: dict[str, Any]) -> str:
    fields = [
        item.get("headline"),
        item.get("title"),
        item.get("blurb"),
        item.get("description"),
        item.get("seoTitle"),
        item.get("seoDescription"),
        item.get("callToAction"),
    ]
    return " ".join(str(value) for value in fields if value)


def iter_video_items(node: Any):
    if isinstance(node, dict):
        if "playbacks" in node and isinstance(node.get("playbacks"), list):
            yield node
        for value in node.values():
            yield from iter_video_items(value)
    elif isinstance(node, list):
        for value in node:
            yield from iter_video_items(value)


def best_playback_url(item: dict[str, Any]) -> str:
    playbacks = item.get("playbacks") or []
    mp4s = [p for p in playbacks if str(p.get("url", "")).lower().endswith(".mp4")]
    candidates = mp4s or playbacks
    if not candidates:
        return ""
    def score(playback: dict[str, Any]) -> int:
        name = str(playback.get("name") or playback.get("height") or "")
        numbers = [int(x) for x in re.findall(r"\d+", name)]
        return max(numbers or [0])
    return str(max(candidates, key=score).get("url", ""))


def classify_expected(text: str) -> str:
    lowered = text.lower()
    if any(term in lowered for term in ["ground ball", "grounder", "grounds", "groundout"]):
        return "ground_ball"
    if any(term in lowered for term in ["fly ball", "flies", "line drive", "liner", "pop fly", "pops"]):
        return "fly_ball"
    return "unknown"


def collect_for_game(game: dict[str, str], keywords: list[str]) -> list[dict[str, str]]:
    url = f"https://statsapi.mlb.com/api/v1/game/{game['game_pk']}/content"
    try:
        payload = fetch_json(url)
    except Exception as exc:
        print(f"SKIP gamePk={game['game_pk']}: {exc}")
        return []
    rows: list[dict[str, str]] = []
    for item in iter_video_items(payload):
        text = text_for_video(item)
        lowered = text.lower()
        if keywords and not any(keyword.lower() in lowered for keyword in keywords):
            continue
        playback = best_playback_url(item)
        if not playback:
            continue
        slug = safe_slug(str(item.get("slug") or item.get("id") or text[:40]), "mlb_video")
        rows.append(
            {
                "source_id": f"MLB_{game['game_pk']}_{slug}"[:80],
                "source_url": playback,
                "video_title": text[:300],
                "game_date": game["game_date"],
                "teams": game["teams"],
                "batter": "",
                "event_text": text[:300],
                "expected_label": classify_expected(text),
                "rights_note": "MLB official public video metadata; verify redistribution rights before public release.",
                "status": "discovered",
                "local_path": "",
                "source_hash": "",
                "notes": f"gamePk={game['game_pk']}",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover MLB official highlight videos into sources_manifest.csv.")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--keywords", default="ground ball,grounder,fly ball,line drive,pop fly,liner")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    dt.date.fromisoformat(args.start_date)
    dt.date.fromisoformat(args.end_date)
    keywords = [item.strip() for item in args.keywords.split(",") if item.strip()]
    existing = read_csv(args.manifest)
    seen = {row.get("source_id") for row in existing}
    seen_urls = {row.get("source_url") for row in existing if row.get("source_url")}

    discovered: list[dict[str, str]] = []
    for game in schedule_game_pks(args.start_date, args.end_date):
        for row in collect_for_game(game, keywords):
            if row["source_id"] in seen or row.get("source_url") in seen_urls:
                continue
            discovered.append(row)
            seen.add(row["source_id"])
            seen_urls.add(row.get("source_url", ""))
            if args.limit and len(discovered) >= args.limit:
                break
        if args.limit and len(discovered) >= args.limit:
            break

    write_csv(args.manifest, existing + discovered, FIELDS)
    print(f"Discovered {len(discovered)} new source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
