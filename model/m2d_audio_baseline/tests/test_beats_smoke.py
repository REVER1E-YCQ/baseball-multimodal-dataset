from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from scripts.beats_encoder import BEATsEncoderAdapter, TOKEN_DIMENSION

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "data/models/beats_iter3plus_as2m/BEATs_iter3_plus_AS2M.pt"
)
DEFAULT_BEATS_ROOT = REPO_ROOT / "external/unilm/beats"
EXPECTED_SHA256 = (
    "d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34"
)


@unittest.skipUnless(
    os.environ.get("BEATS_SMOKE") == "1",
    "Set BEATS_SMOKE=1 to run the real BEATs adapter smoke test",
)
class BEATsAdapterSmokeTest(unittest.TestCase):
    def test_real_beats_adapter_contract(self) -> None:
        checkpoint = Path(os.environ.get("BEATS_CHECKPOINT", DEFAULT_CHECKPOINT))
        beats_root = Path(os.environ.get("BEATS_ROOT", DEFAULT_BEATS_ROOT))
        if not checkpoint.is_file() or not (beats_root / "BEATs.py").is_file():
            self.skipTest("BEATs checkpoint or upstream source is not available")

        adapter = BEATsEncoderAdapter(
            checkpoint=checkpoint,
            beats_root=beats_root,
            device="auto",
            expected_checkpoint_sha256=EXPECTED_SHA256,
        )

        self.assertEqual(adapter.provenance.token_dimension, TOKEN_DIMENSION)
        self.assertEqual(adapter.provenance.training_epochs, 0)
        self.assertEqual(adapter.provenance.precision, "fp32")
        self.assertEqual(adapter.provenance.checkpoint_sha256, EXPECTED_SHA256)
        self.assertNotEqual(adapter.provenance.upstream_revision, "unknown")

        model, _device = adapter._load_model()
        frozen = all(
            not parameter.requires_grad for parameter in model.parameters()
        )
        self.assertTrue(frozen)
        self.assertFalse(model.training)

        sample_rate = 16_000
        waveform = np.zeros(sample_rate // 5, dtype=np.float32)
        waveform[len(waveform) // 2] = 1.0
        tokens = adapter.encode_tokens(waveform, sample_rate)

        self.assertEqual(tokens.ndim, 2)
        self.assertEqual(tokens.shape[1], TOKEN_DIMENSION)
        self.assertGreater(tokens.shape[0], 0)
        self.assertTrue(np.isfinite(tokens).all())

    def test_long_audio_produces_valid_tokens(self) -> None:
        checkpoint = Path(os.environ.get("BEATS_CHECKPOINT", DEFAULT_CHECKPOINT))
        beats_root = Path(os.environ.get("BEATS_ROOT", DEFAULT_BEATS_ROOT))
        if not checkpoint.is_file() or not (beats_root / "BEATs.py").is_file():
            self.skipTest("BEATs checkpoint or upstream source is not available")

        adapter = BEATsEncoderAdapter(
            checkpoint=checkpoint,
            beats_root=beats_root,
            device="auto",
            expected_checkpoint_sha256=EXPECTED_SHA256,
        )

        # An 8 s clip represents a full-audio condition; the encoder must
        # return finite tokens without tripping the fixed-length padding
        # assertion that guards short inputs.
        sample_rate = 16_000
        seconds = 8
        waveform = np.zeros(sample_rate * seconds, dtype=np.float32)
        waveform[sample_rate // 2] = 1.0
        tokens = adapter.encode_tokens(waveform, sample_rate)

        self.assertEqual(tokens.ndim, 2)
        self.assertEqual(tokens.shape[1], TOKEN_DIMENSION)
        self.assertGreater(tokens.shape[0], 100)
        self.assertTrue(np.isfinite(tokens).all())
        embedding = tokens.mean(axis=0)
        self.assertEqual(embedding.shape, (TOKEN_DIMENSION,))
        self.assertTrue(np.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
