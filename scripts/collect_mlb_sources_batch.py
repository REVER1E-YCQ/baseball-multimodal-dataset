from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

from collect_mlb_sources import FIELDS, collect_for_game, schedule_game_pks
from common import read_csv, repo_path, write_csv


def date_range(start_date: str, end_date: str):
    current = dt.date.fromisoformat(start_date)
    end = dt.date.fromisoformat(end_date)
    while current <= end:
        yield current.isoformat()
        current += dt.timedelta(days=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resumable MLB source discovery that writes the manifest after each game."
    )
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--manifest", type=Path, default=repo_path("manifests", "sources_manifest.csv"))
    parser.add_argument("--keywords", default="ground ball,grounder,grounds,groundout,fly ball,flies,line drive,liner,pop fly,pops")
    parser.add_argument("--target-new", type=int, default=0, help="Stop after this many new rows; 0 means no cap.")
    parser.add_argument("--per-day-limit", type=int, default=0, help="Stop each day after this many new rows; 0 means no cap.")
    args = parser.parse_args()

    keywords = [item.strip() for item in args.keywords.split(",") if item.strip()]
    existing = read_csv(args.manifest)
    seen_ids = {row.get("source_id") for row in existing}
    seen_urls = {row.get("source_url") for row in existing if row.get("source_url")}
    total_new = 0

    for day in date_range(args.start_date, args.end_date):
        day_new = 0
        print(f"DATE {day}")
        for game in schedule_game_pks(day, day):
            game_new = 0
            for row in collect_for_game(game, keywords):
                if row["source_id"] in seen_ids or row.get("source_url") in seen_urls:
                    continue
                existing.append(row)
                seen_ids.add(row["source_id"])
                seen_urls.add(row.get("source_url", ""))
                day_new += 1
                game_new += 1
                total_new += 1
                if args.per_day_limit and day_new >= args.per_day_limit:
                    break
                if args.target_new and total_new >= args.target_new:
                    break
            if game_new:
                write_csv(args.manifest, existing, FIELDS)
                print(f"  gamePk={game['game_pk']} new={game_new} total_new={total_new}")
            if args.per_day_limit and day_new >= args.per_day_limit:
                break
            if args.target_new and total_new >= args.target_new:
                break
        write_csv(args.manifest, existing, FIELDS)
        print(f"DATE {day} new={day_new}")
        if args.target_new and total_new >= args.target_new:
            break

    print(f"Discovered {total_new} new source rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
