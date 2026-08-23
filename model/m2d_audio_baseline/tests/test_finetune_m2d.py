from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile

from scripts.finetune_m2d import (
    FinetuneConfig,
    FinetuneProvenance,
    build_trainable_model,
    run_finetune_pilot,
)


class FakeAttention(torch.nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        return self.proj(torch.matmul(weights, v))


class FakeBlock(torch.nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d)
        self.attn = FakeAttention(d)
        self.norm2 = torch.nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x


class FakeBackbone(torch.nn.Module):
    """A tiny differentiable ViT-like encoder: waveform -> [B, T, D]."""

    def __init__(self, d: int = 8, tokens: int = 4) -> None:
        super().__init__()
        self.d = d
        self.tokens = tokens
        # Per-chunk mean energy is a linearly separable signal.
        self.projection = torch.nn.Linear(1, d)
        self.backbone = torch.nn.Module()
        self.backbone.blocks = torch.nn.ModuleList(
            [FakeBlock(d) for _ in range(2)]
        )

    def encode(self, waveform: torch.Tensor) -> torch.Tensor:
        # [B, samples] -> [B, T, samples/T] -> chunk means -> [B, T, d]
        batch_size = waveform.shape[0]
        samples = waveform.shape[-1]
        if samples % self.tokens != 0:
            raise AssertionError(
                f"FakeBackbone needs samples divisible by {self.tokens}; "
                f"got {samples}"
            )
        chunks = waveform.reshape(batch_size, self.tokens, -1)
        x = self.projection(chunks.mean(dim=2, keepdim=True))
        for block in self.backbone.blocks:
            x = block(x)
        return x


def _make_snapshot(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Twelve games, two labels, 200 ms windows at 16 kHz with a learnable
    signal: the pulse sits in the first chunk for fly and the last chunk
    for ground, so mean-pooled tokens are linearly separable."""
    sample_rate = 16_000
    windows: list[dict[str, object]] = []
    for game_index in range(12):
        for label, polarity in (("fly_ball", 1.0), ("ground_ball", -1.0)):
            uid = f"game-{game_index:02d}-{label}"
            waveform = np.zeros(sample_rate // 5, dtype=np.float32)
            chunk = sample_rate // 5 // 4
            position = 0 if label == "fly_ball" else 3
            waveform[position * chunk : (position + 1) * chunk] = polarity
            window_path = root / "windows" / f"{uid}.wav"
            window_path.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(window_path, sample_rate, waveform)
            windows.append(
                {
                    "uid": uid,
                    "label": label,
                    "lineage_group_id": f"game-{game_index:02d}",
                    "window_path": window_path,
                }
            )
    window_frame = pd.DataFrame(windows)
    folds = pd.DataFrame(
        [
            {"uid": uid, "outer_fold": game_index % 3}
            for game_index in range(12)
            for uid in (
                f"game-{game_index:02d}-fly_ball",
                f"game-{game_index:02d}-ground_ball",
            )
        ]
    )
    return window_frame, folds


class FinetunePilotTest(unittest.TestCase):
    def _run(self, root: Path, seed: int = 7):
        windows, folds = _make_snapshot(root)
        return run_finetune_pilot(
            model_factory=lambda: (FakeBackbone(), torch.device("cpu")),
            token_dimension=8,
            provenance=FinetuneProvenance(
                backbone="fake-vit",
                upstream_revision="fake-revision-1",
                checkpoint_sha256="fake-checkpoint-sha256",
                mode="lora",
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
                unfreeze_layers=0,
                lr=1e-3,
                head_lr=1e-2,
                max_epochs=20,
                inner_splits=2,
            ),
            config=FinetuneConfig(
                lr=1e-2, head_lr=1e-2, max_epochs=50, inner_splits=2
            ),
            windows=windows,
            manifest_root=root,
            folds=folds,
            seed=seed,
            output_dir=root / "pilot",
        )

    def test_pilot_learns_the_signal_and_early_stops(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self._run(root)
            predictions = result["oof_predictions"]
            self.assertEqual(len(predictions), 24)
            self.assertGreaterEqual(result["event_balanced_accuracy"], 0.9)
            trace = result["trace"]
            self.assertGreaterEqual(
                len(trace), len(trace["outer_fold"].unique())
            )
            self.assertTrue((trace["epoch"] <= 50).all())
            # Early stopping triggered: the best epoch is below max and
            # the last logged epoch is below max for at least one fold.
            per_fold_last = trace.groupby("outer_fold")["epoch"].max()
            self.assertTrue((per_fold_last < 50).any())
            # The train/val gap is small on the synthetic signal: the best
            # inner-validation BA per fold is near-perfect.
            best_per_fold = trace.groupby("outer_fold")[
                "inner_val_balanced_accuracy"
            ].max()
            self.assertGreater(float(best_per_fold.mean()), 0.9)

    def test_test_fold_never_in_training_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            windows, folds = _make_snapshot(root)
            from scripts.finetune_m2d import WindowDataset

            outer_fold_by_uid = dict(
                zip(folds["uid"], folds["outer_fold"])
            )
            for outer_fold in sorted(set(folds["outer_fold"])):
                train_uids = [
                    uid
                    for uid in folds["uid"]
                    if outer_fold_by_uid[uid] != outer_fold
                ]
                dataset = WindowDataset(windows, root, train_uids)
                seen = {str(dataset[index]["uid"]) for index in range(len(dataset))}
                test_uids = {
                    uid
                    for uid in folds["uid"]
                    if outer_fold_by_uid[uid] == outer_fold
                }
                self.assertTrue(seen.isdisjoint(test_uids))

    def test_deterministic_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._run(root, seed=7)
            second = self._run(root, seed=7)
            self.assertEqual(
                first["event_balanced_accuracy"],
                second["event_balanced_accuracy"],
            )

    def test_lora_attaches_to_attention_projections(self) -> None:
        model = FakeBackbone()
        peft_model, head = build_trainable_model(
            model, FinetuneConfig(), token_dimension=8
        )
        lora_names = [
            name
            for name, _parameter in peft_model.named_parameters()
            if "lora" in name
        ]
        self.assertTrue(any("attn.qkv" in name for name in lora_names))
        self.assertTrue(any("attn.proj" in name for name in lora_names))
        trainable = sum(
            parameter.numel()
            for parameter in peft_model.parameters()
            if parameter.requires_grad
        )
        self.assertGreater(trainable, 0)

    def test_unfreeze_top_mode_trainable_blocks(self) -> None:
        model = FakeBackbone()
        trainable_model, head = build_trainable_model(
            model,
            FinetuneConfig(mode="unfreeze_top", unfreeze_layers=1),
            token_dimension=8,
        )
        trainable_names = {
            name
            for name, parameter in trainable_model.named_parameters()
            if parameter.requires_grad
        }
        self.assertTrue(
            any("blocks.1" in name for name in trainable_names)
        )
        self.assertFalse(
            any("blocks.0" in name for name in trainable_names)
        )
        self.assertGreater(len(trainable_names), 0)


if __name__ == "__main__":
    unittest.main()
