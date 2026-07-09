from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common import load_jsonl, repo_path


def usage_value(usage: dict[str, Any], *names: str) -> int:
    for name in names:
        value = usage.get(name)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Qwen/DashScope usage from local label logs.")
    parser.add_argument("--labels", type=Path, default=repo_path("reports", "qwen_labels.jsonl"))
    args = parser.parse_args()

    rows = load_jsonl(args.labels)
    model_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    model_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    records_with_usage = 0

    for row in rows:
        model = row.get("model") or "unknown"
        model_counts[model] += 1
        if row.get("error"):
            status_counts["error"] += 1
        else:
            label = (row.get("label") or {}).get("label") or "unknown"
            status_counts[label] += 1

        usage = row.get("usage") or {}
        if not usage:
            continue
        records_with_usage += 1
        prompt_tokens = usage_value(usage, "prompt_tokens", "input_tokens")
        completion_tokens = usage_value(usage, "completion_tokens", "output_tokens")
        total_tokens = usage_value(usage, "total_tokens")
        if not total_tokens:
            total_tokens = prompt_tokens + completion_tokens
        model_tokens[model]["prompt_tokens"] += prompt_tokens
        model_tokens[model]["completion_tokens"] += completion_tokens
        model_tokens[model]["total_tokens"] += total_tokens

    print(f"records={len(rows)}")
    print(f"records_with_usage={records_with_usage}")
    print("models:")
    for model, count in model_counts.most_common():
        tokens = model_tokens[model]
        print(
            f"  {model}: records={count} "
            f"prompt_tokens={tokens['prompt_tokens']} "
            f"completion_tokens={tokens['completion_tokens']} "
            f"total_tokens={tokens['total_tokens']}"
        )
    print("labels_or_statuses:")
    for label, count in status_counts.most_common():
        print(f"  {label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
