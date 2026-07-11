from __future__ import annotations

import argparse
import csv
import html
import subprocess
from pathlib import Path

from common import repo_path, tool_path


DEFAULT_SAMPLES = [
    "G_003",
    "G_007",
    "G_015",
    "G_055",
    "F_004",
    "F_007",
    "F_020",
    "F_033",
    "F_044",
]


def sample_dir(sample_id: str) -> Path:
    if sample_id.startswith("G_"):
        return repo_path("dataset", "ground_ball", "Codex_Workstation", sample_id)
    if sample_id.startswith("F_"):
        return repo_path("dataset", "fly_ball", "Codex_Workstation", sample_id)
    raise SystemExit(f"Unknown sample prefix for {sample_id}")


def read_sample_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 1:
        raise SystemExit(f"Expected one metadata row in {path}, found {len(rows)}")
    return rows[0]


def read_source_txt(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def make_spectrogram(audio_path: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        tool_path("ffmpeg"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-lavfi",
        "showspectrumpic=s=1200x420:legend=1:scale=log",
        str(output_path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode == 0:
        return

    fallback = [
        tool_path("ffmpeg"),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(audio_path),
        "-lavfi",
        "showspectrumpic=s=1200x420:legend=1",
        str(output_path),
    ]
    fallback_proc = subprocess.run(fallback, text=True, capture_output=True)
    if fallback_proc.returncode != 0:
        raise SystemExit(
            f"ffmpeg failed for {audio_path}\n"
            f"first error:\n{proc.stderr}\n"
            f"fallback error:\n{fallback_proc.stderr}"
        )


def attrs(row: dict[str, str]) -> str:
    keep = [
        "sample_id",
        "label",
        "region",
        "landing_zone",
        "strength",
        "bounce",
        "trajectory_type",
        "event_start",
        "event_end",
    ]
    return " | ".join(f"{key}: {row[key]}" for key in keep if row.get(key))


def build_html(items: list[dict[str, str]], output_path: Path) -> None:
    rows = []
    for item in items:
        title = html.escape(item.get("video_title") or "")
        source_id = html.escape(item.get("source_id") or "")
        clip_id = html.escape(item.get("clip_id") or "")
        item_attrs = html.escape(item["attrs"])
        sample_id = html.escape(item["sample_id"])
        label = html.escape(item["label"])
        video_uri = html.escape(item["video_uri"], quote=True)
        audio_uri = html.escape(item["audio_uri"], quote=True)
        spec_uri = html.escape(item["spectrogram_uri"], quote=True)
        rows.append(
            f"""
      <section class="sample" id="{sample_id}">
        <header>
          <div>
            <h2>{sample_id}</h2>
            <p class="meta">{label} | {item_attrs}</p>
          </div>
          <a class="audio-link" href="{audio_uri}">Open audio file</a>
        </header>
        <div class="source">
          <div><strong>source_id</strong>: {source_id}</div>
          <div><strong>clip_id</strong>: {clip_id}</div>
          <div><strong>title</strong>: {title}</div>
        </div>
        <div class="media-grid">
          <video controls preload="metadata" src="{video_uri}"></video>
          <div class="audio-panel">
            <audio controls preload="metadata" src="{audio_uri}"></audio>
            <img src="{spec_uri}" alt="Spectrogram for {sample_id}">
          </div>
        </div>
      </section>
"""
        )

    body = "\n".join(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual audio review</title>
  <style>
    :root {{
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f9;
      color: #18202a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
    }}
    main {{
      max-width: 1340px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 28px;
      line-height: 1.2;
    }}
    .summary {{
      margin: 0 0 20px;
      color: #536173;
      font-size: 14px;
    }}
    .sample {{
      background: #ffffff;
      border: 1px solid #d9dee7;
      border-radius: 8px;
      padding: 16px;
      margin: 0 0 18px;
    }}
    header {{
      display: flex;
      align-items: start;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 10px;
    }}
    h2 {{
      margin: 0 0 4px;
      font-size: 20px;
    }}
    .meta, .source {{
      color: #4d5b69;
      font-size: 13px;
      line-height: 1.45;
    }}
    .meta {{ margin: 0; }}
    .source {{
      border-top: 1px solid #edf0f4;
      padding-top: 10px;
      margin-bottom: 12px;
      overflow-wrap: anywhere;
    }}
    .audio-link {{
      flex: 0 0 auto;
      color: #0f5f99;
      font-size: 13px;
      text-decoration: none;
    }}
    .audio-link:hover {{ text-decoration: underline; }}
    .media-grid {{
      display: grid;
      grid-template-columns: minmax(320px, 0.95fr) minmax(360px, 1.05fr);
      gap: 14px;
      align-items: start;
    }}
    video, img {{
      display: block;
      width: 100%;
      max-width: 100%;
      border: 1px solid #cfd6df;
      border-radius: 6px;
      background: #111820;
    }}
    audio {{
      display: block;
      width: 100%;
      margin: 0 0 10px;
    }}
    @media (max-width: 920px) {{
      body {{ padding: 14px; }}
      .media-grid {{ grid-template-columns: 1fr; }}
      header {{ display: block; }}
      .audio-link {{ display: inline-block; margin-top: 8px; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>Manual audio review</h1>
    <p class="summary">Each sample includes video, direct audio controls, and an ffmpeg-generated spectrogram from audio.wav.</p>
{body}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manual review page with audio spectrograms.")
    parser.add_argument(
        "--samples",
        default=",".join(DEFAULT_SAMPLES),
        help="Comma-separated sample IDs, for example G_003,F_004",
    )
    parser.add_argument(
        "--output",
        default=str(repo_path("reports", "manual_review_20260711", "index.html")),
        help="Output HTML path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate existing spectrogram PNGs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_path(args.output)
    spec_dir = output_path.parent / "spectrograms"

    sample_ids = [value.strip() for value in args.samples.split(",") if value.strip()]
    items: list[dict[str, str]] = []
    for sample_id in sample_ids:
        base = sample_dir(sample_id)
        audio_path = base / "audio.wav"
        video_path = base / "video.mp4"
        csv_path = base / "sample.csv"
        if not audio_path.exists() or not video_path.exists() or not csv_path.exists():
            raise SystemExit(f"Missing sample media or metadata under {base}")

        row = read_sample_csv(csv_path)
        source = read_source_txt(base / "source.txt")
        spec_path = spec_dir / f"{sample_id}.png"
        if args.force or not spec_path.exists():
            make_spectrogram(audio_path, spec_path)

        items.append(
            {
                "sample_id": sample_id,
                "label": row.get("label", ""),
                "attrs": attrs(row),
                "video_title": source.get("video_title", ""),
                "source_id": source.get("source_id", ""),
                "clip_id": source.get("clip_id", ""),
                "video_uri": file_uri(video_path),
                "audio_uri": file_uri(audio_path),
                "spectrogram_uri": file_uri(spec_path),
            }
        )

    build_html(items, output_path)
    print(f"wrote {output_path}")
    print(f"spectrograms {spec_dir}")


if __name__ == "__main__":
    main()
