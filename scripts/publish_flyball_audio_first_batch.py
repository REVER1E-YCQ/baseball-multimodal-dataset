from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from common import read_csv, repo_path, write_csv


AUDIT_FIELDS = ["sample_id", "status", "action", "reason", "published_path", "archive_path"]


def archive_once(source: Path, archive: Path) -> None:
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, archive)


def clear_sample(path: Path) -> None:
    for child in path.iterdir():
        if child.name != ".gitkeep":
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    (path / ".gitkeep").write_text("", encoding="ascii")


def publish_approved(source: Path, target: Path) -> None:
    for name in ("video.mp4", "audio.wav", "sample.csv", "label.txt", "source.txt"):
        source_file = source / name
        target_file = target / name
        if target_file.exists() and os.path.samefile(source_file, target_file):
            continue
        shutil.copy2(source_file, target_file)
    gitkeep = target / ".gitkeep"
    if gitkeep.exists():
        gitkeep.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish one reviewed Fly Ball batch or leave an auditable empty placeholder.")
    parser.add_argument("--reconciliation", type=Path, required=True)
    parser.add_argument("--review-manifest", type=Path, required=True)
    parser.add_argument("--work-queue", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    decisions = {row["sample_id"]: row for row in read_csv(args.reconciliation)}
    reviews = {row["sample_id"]: row for row in read_csv(args.review_manifest)}
    queue = {row["sample_id"]: row for row in read_csv(args.work_queue)}
    audit: list[dict[str, str]] = []
    for sample_id, decision in sorted(decisions.items()):
        work = queue[sample_id]
        target = Path(work["sample_path"])
        archive = args.archive_root.resolve() / sample_id
        approved = decision["status"] in {"pass", "auto_applied"}
        action = "publish_approved" if approved else "empty_placeholder"
        if args.apply:
            archive_once(target, archive)
            if approved:
                publish_approved(Path(reviews[sample_id]["review_path"]), target)
            else:
                clear_sample(target)
        audit.append(
            {
                "sample_id": sample_id,
                "status": decision["status"],
                "action": action,
                "reason": decision["reason"],
                "published_path": str(target.relative_to(repo_path())),
                "archive_path": str(archive.relative_to(repo_path())),
            }
        )
    write_csv(args.audit_output, audit, AUDIT_FIELDS)
    counts: dict[str, int] = {}
    for row in audit:
        counts[row["action"]] = counts.get(row["action"], 0) + 1
    print(f"apply={args.apply} actions={json.dumps(counts, sort_keys=True)} audit={args.audit_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
