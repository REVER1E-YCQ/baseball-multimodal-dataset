from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from scripts.benchmark_artifact_roles import BenchmarkArtifactRoleError
from scripts.secondary_evidence import compute_secondary_evidence
from scripts.short_contact_benchmark import (
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderAdapter,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)
from scripts.statistical_evidence import compute_statistical_evidence
from scripts.validate_and_report import (
    generate_reports,
    validate_complete_run,
)


M2D_ENCODER = "m2d_vit_base_80x200p16x4_40ms"
BEATS_ENCODER = "beats_iter3plus_as2m"


@dataclass
class _FakeEncoder(EncoderAdapter):
    name: str = "fake"

    provenance: EncoderProvenance = field(init=False)

    def __post_init__(self) -> None:
        self.provenance = EncoderProvenance(
            name=self.name,
            upstream_revision="fake-revision-1",
            checkpoint_sha256="fake-checkpoint-sha256",
            precision="fp32",
            token_dimension=4,
        )

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        center = float(waveform[len(waveform) // 2])
        energy = float(np.mean(np.square(waveform)))
        if self.name == BEATS_ENCODER:
            center = 0.0
            energy = 0.0
        token = np.asarray(
            [center, energy, abs(center), float(sample_rate) / 16_000.0]
        )
        return np.stack([token, token])


def _make_snapshot(root: Path) -> DatasetSnapshot:
    samples: list[SnapshotSample] = []
    for game_index in range(24):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"{label}__Collector_A__S{game_index:02d}"
            waveform = np.zeros(16_000, dtype=np.float32)
            waveform[8_000] = polarity
            audio_path = root / "snapshot" / f"{uid}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio_path, 16_000, waveform)
            samples.append(
                SnapshotSample(
                    uid=uid,
                    label=label,
                    lineage_group_id=f"game-{game_index:02d}",
                    audio_path=audio_path,
                    event_start=0.45,
                    event_end=0.55,
                )
            )
    return DatasetSnapshot(revision="validate-snapshot", samples=tuple(samples))


def _fixed_split(root: Path, snapshot: DatasetSnapshot) -> Path:
    rows = []
    for position, uid in enumerate(sorted(s.uid for s in snapshot.samples)):
        label, collector, sample_id = uid.split("__")
        rows.append(
            {
                "dataset_path": f"dataset/{label}/{collector}/{sample_id}",
                "sample_id": sample_id,
                "label": label,
                "source_group": f"src-{position % 4}",
                "split": ["train", "train", "val", "test"][position % 4],
            }
        )
    path = root / "dataset_split.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


class ValidateAndReportTest(unittest.TestCase):
    def _layout(self, root: Path) -> dict[str, Path]:
        snapshot = _make_snapshot(root)
        protocol = BenchmarkProtocol(
                seed=77,
                outer_splits=2,
                inner_splits=2,
                c_grid=(0.001, 0.01, 0.1),
                include_controls=True,
            )
        bundles = {
            M2D_ENCODER: run_short_contact_benchmark(
                protocol,
                snapshot,
                (_FakeEncoder(name=M2D_ENCODER),),
                root / "common",
            ),
            BEATS_ENCODER: run_short_contact_benchmark(
                protocol,
                snapshot,
                (_FakeEncoder(name=BEATS_ENCODER),),
                root / "common",
            ),
        }
        compute_statistical_evidence(
            bundles,
            root / "common" / "statistical_evidence",
            n_bootstrap=49,
            n_permutations=49,
        )
        split_path = _fixed_split(root, snapshot)
        compute_secondary_evidence(
            bundles,
            split_path,
            root / "secondary" / "evidence",
            seed=79,
        )
        def sensitivity_protocol(**overrides: object) -> BenchmarkProtocol:
            c_grid = overrides.pop("c_grid", (0.001, 0.01, 0.1))
            return BenchmarkProtocol(
                seed=77,
                outer_splits=2,
                c_grid=c_grid,
                include_controls=True,
                **overrides,
            )

        sensitivity_protocols = [
            sensitivity_protocol(),
            sensitivity_protocol(
                window_conditions=(50, 100, 200), pooling="attention"
            ),
            sensitivity_protocol(normalization="rms_normalized"),
            sensitivity_protocol(pooling="legacy_mean_std_max"),
            sensitivity_protocol(
                pooling="legacy_mean_std_max", calibrate_threshold=True
            ),
            sensitivity_protocol(pooling="mean_std"),
            sensitivity_protocol(pooling="mean_max"),
            sensitivity_protocol(pooling="energy_weighted"),
            sensitivity_protocol(pooling="attention"),
            sensitivity_protocol(
                pooling="attention", calibrate_threshold=True
            ),
            sensitivity_protocol(pooling="attention_lda", c_grid=None),
            sensitivity_protocol(pooling="attention_multi", c_grid=None),
            sensitivity_protocol(
                pooling="attention_neighbourhood", c_grid=None
            ),
            sensitivity_protocol(
                window_conditions=(50, 200),
                c_grid=None,
                feature_composition=(
                    (50, "valid_final_layer_token_mean"),
                    (200, "attention"),
                ),
            ),
        ]
        for sensitivity_protocol_item in sensitivity_protocols:
            run_short_contact_benchmark(
                sensitivity_protocol_item,
                snapshot,
                (_FakeEncoder(name=M2D_ENCODER),),
                root / "sensitivity",
            )
        audit = {
            "revision": "validate-snapshot",
            "sample_count": 48,
            "label_counts": {"fly_ball": 24, "ground_ball": 24},
            "verification_source_counts": {
                "human_binary_review": 30,
                "local_first_pass_direct": 18,
            },
        }
        audit_path = root / "snapshot_audit.json"
        audit_path.write_text(json.dumps(audit), encoding="utf-8")
        return {
            "common": root / "common",
            "secondary": root / "secondary" / "evidence",
            "sensitivity": root / "sensitivity",
            "audit": audit_path,
            "out": root / "validated",
        }

    def test_validates_complete_run_and_writes_reports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            report = validate_complete_run(
                common_root=layout["common"],
                secondary_evidence_dir=layout["secondary"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                sensitivity_root=layout["sensitivity"],
                output_root=layout["out"],
                expected_snapshot_count=48,
                expected_label_counts={"fly_ball": 24, "ground_ball": 24},
                benchmark_seed=77,
                dataset_revision="validate-snapshot",
            )
            self.assertEqual(report["status"], "pass", report["failures"])

            paths = generate_reports(
                common_root=layout["common"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                secondary_evidence_dir=layout["secondary"],
                sensitivity_root=layout["sensitivity"],
                snapshot_audit_path=layout["audit"],
                output_root=layout["out"],
                benchmark_seed=77,
            )
            report_text = paths["report"].read_text(encoding="utf-8")
            self.assertIn("Balanced Accuracy", report_text)
            self.assertIn("不能由本实验单独证明音频信息可迁移", report_text)
            self.assertIn("human_binary_review", report_text)
            self.assertIn("validate-snapshot", report_text)
            self.assertIn("attention", report_text)
            self.assertIn("event_fitted_transfer_v1", report_text)
            self.assertIn("池化消融", report_text)
            self.assertIn("决策阈值校准", report_text)
            self.assertIn("头条表示", report_text)
            self.assertIn("头条决策", report_text)
            self.assertIn("attention-neighbourhood", report_text)
            self.assertIn("50ms mean + 200ms attention", report_text)
            self.assertIn("峰值 token 选择器", report_text)
            summary_text = paths["summary"].read_text(encoding="utf-8")
            self.assertIn("组会摘要", summary_text)
            self.assertIn("M2D 1.000 / BEATs 0.500", summary_text)
            self.assertIn("attention（1.000）", summary_text)
            self.assertNotIn("attention（0.667）", summary_text)

    def test_report_rejects_an_ambiguous_sensitivity_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            def is_attention_headline(candidate: Path) -> bool:
                protocol = json.loads(
                    (candidate / "protocol.json").read_text(encoding="utf-8")
                )
                return (
                    protocol.get("pooling") == "attention"
                    and protocol.get("decision_threshold", {}).get(
                        "calibrate"
                    )
                    is False
                    and protocol.get("window_conditions")
                    == ["event_200ms"]
                )

            source = next(
                candidate
                for candidate in layout["sensitivity"].glob("*")
                if is_attention_headline(candidate)
            )
            duplicate = layout["sensitivity"] / "duplicate-attention"
            duplicate.mkdir()
            shutil.copy2(source / "protocol.json", duplicate / "protocol.json")

            with self.assertRaisesRegex(
                BenchmarkArtifactRoleError,
                "m2d_attention_headline.*2 matching artifacts",
            ):
                generate_reports(
                    common_root=layout["common"],
                    statistical_evidence_dir=(
                        layout["common"] / "statistical_evidence"
                    ),
                    secondary_evidence_dir=layout["secondary"],
                    sensitivity_root=layout["sensitivity"],
                    snapshot_audit_path=layout["audit"],
                    output_root=layout["out"],
                    benchmark_seed=77,
                )

    def _full_audio_layout(self, root: Path) -> Path:
        full_audio = root / "full_audio"
        full_audio.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "condition": "event_500ms",
                    "window_ms": 500,
                    "event_balanced_accuracy": 0.583,
                    "event_roc_auc": 0.607,
                    "eligible_samples": 785,
                },
                {
                    "condition": "pre_contact_1000ms",
                    "window_ms": 1000,
                    "event_balanced_accuracy": 0.570,
                    "event_roc_auc": 0.595,
                    "eligible_samples": 670,
                },
                {
                    "condition": "post_contact_1000ms",
                    "window_ms": 1000,
                    "event_balanced_accuracy": 0.694,
                    "event_roc_auc": 0.758,
                    "eligible_samples": 802,
                },
                {
                    "condition": "post_contact_4000ms",
                    "window_ms": 4000,
                    "event_balanced_accuracy": 0.705,
                    "event_roc_auc": 0.793,
                    "eligible_samples": 576,
                },
                {
                    "condition": "full_audio",
                    "window_ms": 0,
                    "event_balanced_accuracy": 0.777,
                    "event_roc_auc": 0.875,
                    "eligible_samples": 804,
                },
            ]
        ).to_csv(full_audio / "condition_scan.csv", index=False)
        (full_audio / "attribution_summary.json").write_text(
            json.dumps(
                {
                    "baseline_event_500ms": 0.583,
                    "pre_contact_gain": -0.013,
                    "post_contact_1s_gain": 0.111,
                    "post_contact_4s_gain": 0.122,
                    "full_audio_gain": 0.193,
                    "gain_beyond_4s": 0.072,
                    "conclusion": "gain_lives_after_contact",
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            [
                {
                    "lead_condition": "0.5s_contact",
                    "lead_ungrouped_accuracy": 73,
                    "our_condition": "event_500ms",
                    "our_grouped_balanced_accuracy": 0.583,
                },
                {
                    "lead_condition": "4s_window",
                    "lead_ungrouped_accuracy": 78,
                    "our_condition": "post_contact_4000ms",
                    "our_grouped_balanced_accuracy": 0.705,
                },
                {
                    "lead_condition": "full_audio",
                    "lead_ungrouped_accuracy": 88,
                    "our_condition": "full_audio",
                    "our_grouped_balanced_accuracy": 0.777,
                },
            ]
        ).to_csv(full_audio / "lead_comparison.csv", index=False)
        return full_audio

    def test_reports_full_audio_leakage_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            full_audio = self._full_audio_layout(root)
            paths = generate_reports(
                common_root=layout["common"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                secondary_evidence_dir=layout["secondary"],
                sensitivity_root=layout["sensitivity"],
                snapshot_audit_path=layout["audit"],
                output_root=layout["out"],
                full_audio_conditions_dir=full_audio,
                benchmark_seed=77,
            )
            report_text = paths["report"].read_text(encoding="utf-8")
            self.assertIn("完整音频对照（泄漏验证）", report_text)
            self.assertIn("0.777", report_text)
            self.assertIn("0.705", report_text)
            self.assertIn("gain_lives_after_contact", report_text)
            self.assertIn("击球后", report_text)
            self.assertIn("部署", report_text)
            self.assertIn("88", report_text)

    def _alignment_layout(self, root: Path) -> Path:
        align = root / "alignment"
        align.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"shift_ms": -50, "event_balanced_accuracy": 0.60,
                 "event_roc_auc": 0.63, "delta_vs_0ms": -0.07,
                 "eligible_samples": 790},
                {"shift_ms": 0, "event_balanced_accuracy": 0.67,
                 "event_roc_auc": 0.70, "delta_vs_0ms": 0.0,
                 "eligible_samples": 803},
                {"shift_ms": 50, "event_balanced_accuracy": 0.58,
                 "event_roc_auc": 0.61, "delta_vs_0ms": -0.09,
                 "eligible_samples": 790},
            ]
        ).to_csv(align / "alignment_sensitivity.csv", index=False)
        (align / "alignment_sensitivity_summary.json").write_text(
            json.dumps(
                {
                    "drop_at_50ms": 0.09,
                    "interpretation": "precise_alignment_dependence",
                }
            ),
            encoding="utf-8",
        )
        return align

    def test_reports_alignment_sensitivity_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            align = self._alignment_layout(root)
            paths = generate_reports(
                common_root=layout["common"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                secondary_evidence_dir=layout["secondary"],
                sensitivity_root=layout["sensitivity"],
                snapshot_audit_path=layout["audit"],
                output_root=layout["out"],
                alignment_sensitivity_dir=align,
                benchmark_seed=77,
            )
            report_text = paths["report"].read_text(encoding="utf-8")
            self.assertIn("对齐敏感性", report_text)
            self.assertIn("precise_alignment_dependence", report_text)
            self.assertIn("0.67", report_text)
            self.assertIn("部署诊断", report_text)

    def test_reports_finetune_pilot_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            primary = root / "primary"
            run_short_contact_benchmark(
                BenchmarkProtocol(
                    seed=77,
                    outer_splits=2,
                    c_grid=(0.001, 0.01, 0.1),
                ),
                _make_snapshot(root),
                (_FakeEncoder(name=M2D_ENCODER),),
                primary,
            )
            finetune = root / "finetune"
            finetune.mkdir(parents=True, exist_ok=True)
            (finetune / "pilot_comparison.json").write_text(
                json.dumps(
                    {
                        "fine_tuned_mean_balanced_accuracy": 0.511,
                        "fine_tuned_eligible_samples": 48,
                        "frozen_mean_balanced_accuracy": 0.595,
                        "frozen_mean_eligible_samples": 47,
                        "gain_vs_frozen_mean": -0.084,
                        "conclusion": "fine_tuning_closed",
                        "attention_headline_reference_balanced_accuracy": 0.613,
                        "attention_headline_eligible_samples": 44,
                        "overfitting_signature": {
                            "train_minus_inner_val_mean": -0.004
                        },
                    }
                ),
                encoding="utf-8",
            )
            paths = generate_reports(
                common_root=layout["common"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                secondary_evidence_dir=layout["secondary"],
                sensitivity_root=layout["sensitivity"],
                snapshot_audit_path=layout["audit"],
                output_root=layout["out"],
                finetune_pilot_dir=finetune,
                finetune_primary_root=primary,
                benchmark_seed=77,
            )
            report_text = paths["report"].read_text(encoding="utf-8")
            self.assertIn("微调试点", report_text)
            self.assertIn("fine_tuning_closed", report_text)
            self.assertIn("失败模式", report_text)
            self.assertIn("冻结 mean：1.000（增益 -0.489", report_text)
            self.assertNotIn("冻结 mean：0.595", report_text)

    def test_reports_skip_full_audio_section_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            paths = generate_reports(
                common_root=layout["common"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                secondary_evidence_dir=layout["secondary"],
                sensitivity_root=layout["sensitivity"],
                snapshot_audit_path=layout["audit"],
                output_root=layout["out"],
                benchmark_seed=77,
            )
            report_text = paths["report"].read_text(encoding="utf-8")
            self.assertNotIn("完整音频对照", report_text)

    def test_fails_when_bundle_is_tampered(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = self._layout(root)
            # Tamper: drop one metric row from the first bundle.
            common_root = layout["common"]
            bundles = {}
            for candidate in sorted(common_root.glob("*")):
                protocol_path = candidate / "protocol.json"
                if not protocol_path.is_file():
                    continue
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                if protocol.get("controls", {}).get("enabled"):
                    bundles[str(protocol["encoders"][0]["name"])] = candidate
            first = sorted(bundles)[0]
            metrics_path = bundles[first] / "metrics.csv"
            metrics = pd.read_csv(metrics_path)
            metrics = metrics.iloc[:-1]
            metrics.to_csv(metrics_path, index=False)
            report = validate_complete_run(
                common_root=layout["common"],
                secondary_evidence_dir=layout["secondary"],
                statistical_evidence_dir=layout["common"]
                / "statistical_evidence",
                sensitivity_root=layout["sensitivity"],
                output_root=layout["out"],
                expected_snapshot_count=48,
                expected_label_counts={"fly_ball": 24, "ground_ball": 24},
                benchmark_seed=77,
                dataset_revision="validate-snapshot",
            )
            self.assertEqual(report["status"], "fail")
            self.assertTrue(any("metrics conditions" in item for item in report["failures"]))


if __name__ == "__main__":
    unittest.main()
