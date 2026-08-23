from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.benchmark_artifact_roles import (
    BenchmarkArtifactRole,
    BenchmarkArtifactRoleError,
    resolve_benchmark_bundle,
)


M2D = "m2d_vit_base_80x200p16x4_40ms"
REVISION = "verified-snapshot"


def _write_protocol(
    root: Path,
    artifact_id: str,
    *,
    encoder: str = M2D,
    pooling: str = "attention",
    windows: tuple[str, ...] = ("event_200ms",),
    normalization: str = "snapshot_level",
    controls: bool = True,
    seed: int = 20260805,
    calibrated: bool = False,
    event_window_shift_ms: int = 0,
    attention_control_transform_policy: str | None = (
        "event_fitted_transfer_v1"
    ),
) -> Path:
    bundle = root / artifact_id
    bundle.mkdir(parents=True)
    protocol = {
        "artifact_id": artifact_id,
        "dataset": {"revision": REVISION},
        "encoders": [{"name": encoder}],
        "pooling": pooling,
        "window_conditions": list(windows),
        "normalization": normalization,
        "controls": {"enabled": controls},
        "fold_policy": {
            "group": "lineage_group_id",
            "seed": seed,
        },
        "classifier": {
            "name": "balanced_l2_logistic_regression",
            "C_selection": "inner_grouped_cv",
        },
        "decision_threshold": {
            "calibrate": calibrated,
            "fixed_default": 0.5,
        },
        "event_window_shift_ms": event_window_shift_ms,
        "attention_control_transform_policy": (
            attention_control_transform_policy
        ),
    }
    (bundle / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )
    return bundle


def _attention_role() -> BenchmarkArtifactRole:
    return BenchmarkArtifactRole(
        name="m2d_attention_headline",
        encoder_name=M2D,
        dataset_revision=REVISION,
        pooling="attention",
        window_conditions=("event_200ms",),
        normalization="snapshot_level",
        controls_enabled=True,
        fold_seed=20260805,
        lineage_group="lineage_group_id",
        classifier_name="balanced_l2_logistic_regression",
        c_selection="inner_grouped_cv",
        threshold_calibrated=False,
        fixed_threshold=0.5,
        attention_control_transform_policy="event_fitted_transfer_v1",
    )


class BenchmarkArtifactRoleTest(unittest.TestCase):
    def test_resolves_the_only_complete_protocol_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intended = _write_protocol(root, "intended")
            _write_protocol(root, "wrong-window", windows=("event_050ms",))
            _write_protocol(root, "wrong-normalization", normalization="rms_normalized")
            _write_protocol(root, "wrong-threshold", calibrated=True)
            _write_protocol(root, "wrong-shift", event_window_shift_ms=25)
            _write_protocol(
                root,
                "legacy-transform-policy",
                attention_control_transform_policy=None,
            )

            self.assertEqual(
                resolve_benchmark_bundle(root, _attention_role()), intended
            )

    def test_fails_visibly_when_the_role_has_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_protocol(root, "wrong-window", windows=("event_050ms",))

            with self.assertRaisesRegex(
                BenchmarkArtifactRoleError,
                "m2d_attention_headline.*no matching artifact",
            ):
                resolve_benchmark_bundle(root, _attention_role())

    def test_fails_visibly_when_the_role_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_protocol(root, "first")
            _write_protocol(root, "second")

            with self.assertRaisesRegex(
                BenchmarkArtifactRoleError,
                "m2d_attention_headline.*2 matching artifacts",
            ):
                resolve_benchmark_bundle(root, _attention_role())


if __name__ == "__main__":
    unittest.main()
