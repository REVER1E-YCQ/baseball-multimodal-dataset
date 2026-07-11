from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(*parts: str) -> Path:
    return REPO_ROOT.joinpath(*parts)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(5):
        try:
            with path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({name: row.get(name, "") for name in fieldnames})
            return
        except OSError as exc:
            last_error = exc
            if attempt == 4:
                break
            time.sleep(0.25 * (attempt + 1))
    if last_error:
        raise last_error


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_slug(value: str, fallback: str = "item") -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    value = value.strip("._")
    return value or fallback


def require_tool(name: str) -> None:
    exe = tool_path(name)
    try:
        subprocess.run([exe, "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(f"Required tool not found on PATH: {name}") from exc


def tool_path(name: str) -> str:
    env_name = f"{name.upper()}_PATH"
    if os.getenv(env_name):
        return os.environ[env_name]
    found = shutil.which(name)
    if found:
        return found
    candidates = [
        Path("C:/tmp/ffmpeg-essentials").glob(f"**/{name}.exe"),
        Path("/tmp").glob(f"**/{name}"),
    ]
    for matches in candidates:
        for match in matches:
            if match.exists():
                return str(match)
    return name


def ffprobe_duration(path: Path) -> float | None:
    cmd = [
        tool_path("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        return float(proc.stdout.strip())
    except Exception:
        return None


def get_env_first(names: list[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None
