from __future__ import annotations

import argparse
import csv
import html
import random
from pathlib import Path

from common import repo_path


def sample_dirs(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.glob("*/*/*") if p.is_dir()])


def read_csv_row(path: Path) -> dict[str, str]:
    with (path / "sample.csv").open("r", newline="", encoding="utf-8-sig") as fh:
        return next(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an HTML review sheet for manual spot checks.")
    parser.add_argument("--dataset-root", type=Path, default=repo_path("dataset"))
    parser.add_argument("--output", type=Path, default=repo_path("reports", "review_sheet.html"))
    parser.add_argument("--sample-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    paths = sample_dirs(args.dataset_root)
    random.Random(args.seed).shuffle(paths)
    keep = max(1, int(len(paths) * args.sample_rate)) if paths else 0
    paths = sorted(paths[:keep])

    rows = []
    for path in paths:
        rel = path.relative_to(repo_path()).as_posix()
        row = read_csv_row(path)
        source = html.escape((path / "source.txt").read_text(encoding="utf-8"))
        rows.append(
            f"<tr><td>{html.escape(rel)}</td><td>{html.escape(row.get('label', ''))}</td>"
            f"<td>{html.escape(row.get('event_start', ''))}-{html.escape(row.get('event_end', ''))}</td>"
            f"<td><video src='../{rel}/video.mp4' controls width='320'></video>"
            f"<br><audio src='../{rel}/audio.wav' controls></audio></td>"
            f"<td><pre>{source}</pre></td><td>pass / wrong_label / bad_audio / bad_cut / source_issue</td></tr>"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Baseball Dataset Review</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #ccc;padding:8px;vertical-align:top}pre{white-space:pre-wrap}</style>"
        "</head><body><h1>Baseball Dataset Review</h1>"
        f"<p>Samples selected: {len(paths)}</p><table><thead><tr><th>Sample</th><th>Label</th>"
        "<th>Event</th><th>Media</th><th>Source</th><th>Manual result</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></body></html>",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

