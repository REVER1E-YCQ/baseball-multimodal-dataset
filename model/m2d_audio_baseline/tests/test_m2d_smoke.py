from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from scripts.m2d_encoder import M2DEncoderAdapter, TOKEN_DIMENSION

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "data/models/m2d_40ms/m2d_vit_base-80x200p16x4-230529/checkpoint-300.pth"
)
DEFAULT_M2D_ROOT = REPO_ROOT / "external/m2d"
EXPECTED_SHA256 = (
    "63578974bc004ef57a8e5456bac8c684f62c9285537a7b2ddef13b442386786f"
)


@unittest.skipUnless(
    os.environ.get("M2D_SMOKE") == "1",
    "Set M2D_SMOKE=1 to run the real M2D adapter smoke test",
)
class M2DAdapterSmokeTest(unittest.TestCase):
    def test_real_m2d_adapter_contract(self) -> None:
        checkpoint = Path(os.environ.get("M2D_CHECKPOINT", DEFAULT_CHECKPOINT))
        m2d_root = Path(os.environ.get("M2D_ROOT", DEFAULT_M2D_ROOT))
        if not checkpoint.is_file() or not (m2d_root / "examples").is_dir():
            self.skipTest("M2D checkpoint or upstream source is not available")

        adapter = M2DEncoderAdapter(
            checkpoint=checkpoint,
            m2d_root=m2d_root,
            device="auto",
            precision="fp32",
            expected_checkpoint_sha256=EXPECTED_SHA256,
        )

        self.assertEqual(adapter.provenance.token_dimension, TOKEN_DIMENSION)
        self.assertEqual(adapter.provenance.training_epochs, 0)
        self.assertEqual(
            adapter.provenance.checkpoint_sha256, EXPECTED_SHA256
        )
        self.assertNotEqual(adapter.provenance.upstream_revision, "unknown")

        model, device = adapter._load_model()
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
        embedding = tokens.mean(axis=0)
        self.assertEqual(embedding.shape, (TOKEN_DIMENSION,))
        self.assertTrue(np.isfinite(embedding).all())


if __name__ == "__main__":
    unittest.main()
