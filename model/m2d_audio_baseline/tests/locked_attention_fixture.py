from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from scripts.benchmark_artifact_roles import (
    M2D_ENCODER_NAME,
    VERIFIED_DATASET_REVISION,
)
from scripts.short_contact_benchmark import (
    ArtifactBundle,
    BenchmarkProtocol,
    DatasetSnapshot,
    EncoderProvenance,
    SnapshotSample,
    run_short_contact_benchmark,
)


@dataclass
class SyntheticM2D:
    provenance: EncoderProvenance = field(
        default_factory=lambda: EncoderProvenance(
            name=M2D_ENCODER_NAME,
            upstream_revision="synthetic-m2d-revision",
            checkpoint_sha256="synthetic-m2d-checkpoint",
            precision="fp32",
            token_dimension=4,
            training_epochs=0,
        )
    )

    def encode_tokens(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        chunks = np.array_split(waveform, 2)
        return np.asarray(
            [
                [
                    float(chunk.mean()),
                    float(np.square(chunk).mean()),
                    float(np.max(np.abs(chunk))),
                    float(index),
                ]
                for index, chunk in enumerate(chunks)
            ],
            dtype=np.float64,
        )


def build_locked_attention_source(root: Path) -> ArtifactBundle:
    samples: list[SnapshotSample] = []
    sample_rate = 16_000
    for game_index in range(10):
        for label, polarity in (
            ("fly_ball", -1.0),
            ("ground_ball", 1.0),
        ):
            uid = f"game-{game_index:02d}-{label}"
            waveform = np.zeros(sample_rate, dtype=np.float32)
            waveform[sample_rate // 2] = polarity * (
                1.0 + game_index / 20
            )
            audio_path = root / "snapshot" / f"{uid}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(audio_path, sample_rate, waveform)
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
    return run_short_contact_benchmark(
        BenchmarkProtocol(
            seed=20260805,
            outer_splits=5,
            inner_splits=3,
            c_grid=(0.001, 0.01, 0.1),
            pooling="attention",
            include_controls=True,
        ),
        DatasetSnapshot(
            revision=VERIFIED_DATASET_REVISION,
            samples=tuple(samples),
        ),
        (SyntheticM2D(),),
        root / "source",
    )
