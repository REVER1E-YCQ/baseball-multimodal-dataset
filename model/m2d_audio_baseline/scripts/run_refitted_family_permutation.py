from __future__ import annotations

import argparse
import json
from pathlib import Path

from .exploratory_probe_benchmark import ProbeConfig
from .refitted_family_permutation import (
    PermutationFamilyConfig,
    run_refitted_family_permutation,
)


def _load_config(path: Path) -> PermutationFamilyConfig:
    document = json.loads(path.read_text(encoding="utf-8"))
    candidates = tuple(
        ProbeConfig(
            name=str(candidate["name"]),
            estimator_family=str(candidate["estimator_family"]),
            hyperparameter_grid={
                str(name): tuple(values)
                for name, values in candidate["hyperparameter_grid"].items()
            },
            score_output=str(candidate["score_output"]),
            fixed_decision_threshold=candidate.get(
                "fixed_decision_threshold"
            ),
            calibrate_threshold=bool(
                candidate.get("calibrate_threshold", False)
            ),
        )
        for candidate in document["candidates"]
    )
    return PermutationFamilyConfig(
        name=str(document["name"]),
        candidates=candidates,
        n_permutations=int(document.get("n_permutations", 999)),
        seed=int(document.get("seed", 20260805)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refit a complete exploratory probe family under synchronized "
            "groupwise label permutation."
        )
    )
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_refitted_family_permutation(
        args.source_bundle,
        args.output_dir,
        _load_config(args.config),
    )
    print(result.root)
    print(result.path("permutation_summary").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
