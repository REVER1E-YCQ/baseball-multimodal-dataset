#!/usr/bin/env python3
"""Small CNN on full-clip Log-Mel audio with source-grouped evaluation."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset

from audio_baseline import read_manifest
from temporal_ablation_experiment import source_groups


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = Path(__file__).resolve().parent / "results"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SpectrogramDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool):
        self.x = torch.from_numpy(x[:, None]).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        values = self.x[index].clone()
        if self.augment:
            shift = int(torch.randint(-5, 6, (1,)).item())
            values = torch.roll(values, shift, dims=2)
            if torch.rand(()) < 0.6:
                width = int(torch.randint(2, 10, (1,)).item())
                start = int(torch.randint(0, values.shape[2] - width + 1, (1,)).item())
                values[:, :, start : start + width] = 0
            if torch.rand(()) < 0.4:
                height = int(torch.randint(2, 7, (1,)).item())
                start = int(torch.randint(0, values.shape[1] - height + 1, (1,)).item())
                values[:, start : start + height, :] = 0
            values += 0.015 * torch.randn_like(values)
        return values, self.y[index]


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(96, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(values))


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "roc_auc": float(roc_auc_score(y, probability)),
    }


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    loader = DataLoader(
        SpectrogramDataset(x, np.zeros(len(x), dtype=np.int64), augment=False),
        batch_size=batch_size,
        shuffle=False,
    )
    output = []
    model.eval()
    with torch.inference_mode():
        for values, _labels in loader:
            output.append(torch.softmax(model(values), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(output)


def train(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    *,
    seed: int,
    epochs: int,
    val_idx: np.ndarray | None = None,
):
    set_seed(seed)
    model = AudioCNN()
    counts = np.bincount(y[train_idx], minlength=2)
    weights = torch.tensor(len(train_idx) / (2.0 * counts), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.04)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    loader = DataLoader(
        SpectrogramDataset(x[train_idx], y[train_idx], augment=True),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best = None
    best_score = (-1.0, -1.0)
    best_epoch = epochs
    patience = 10
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for values, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(values), labels)
            loss.backward()
            optimizer.step()
        if val_idx is None:
            continue
        probability = predict(model, x[val_idx])
        current = metrics(y[val_idx], probability)
        rank = (current["balanced_accuracy"], current["roc_auc"])
        if rank > best_score:
            best_score = rank
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, best_epoch, best_score


def main() -> None:
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    rows = read_manifest(
        ROOT,
        ROOT / "reports/verified_dataset_20260804/VERIFIED_DATASET_MANIFEST.csv",
    )
    y = np.asarray([1 if row["label"] == "fly_ball" else 0 for row in rows])
    groups = source_groups(rows)
    cache = np.load(RESULT_DIR / "audio_context_features.npz")
    x = cache["full_clip"].astype(np.float32)
    predictions = np.full(len(y), np.nan)
    fold_reports = []
    indices = np.arange(len(y))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260805)
    for fold, (outer_train_idx, test_idx) in enumerate(outer.split(indices, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=20261200 + fold)
        train_rel, val_rel = next(
            inner.split(outer_train_idx, y[outer_train_idx], groups[outer_train_idx])
        )
        inner_train = outer_train_idx[train_rel]
        val_idx = outer_train_idx[val_rel]
        mean = x[inner_train].mean(axis=(0, 2), keepdims=True)
        std = x[inner_train].std(axis=(0, 2), keepdims=True) + 1e-5
        normalized = (x - mean) / std
        _selection_model, best_epoch, validation_score = train(
            normalized,
            y,
            inner_train,
            val_idx=val_idx,
            seed=20260805 + fold,
            epochs=80,
        )
        mean = x[outer_train_idx].mean(axis=(0, 2), keepdims=True)
        std = x[outer_train_idx].std(axis=(0, 2), keepdims=True) + 1e-5
        normalized = (x - mean) / std
        model, _epoch, _score = train(
            normalized,
            y,
            outer_train_idx,
            seed=20261800 + fold,
            epochs=best_epoch,
        )
        predictions[test_idx] = predict(model, normalized[test_idx])
        fold_result = {
            "fold": fold,
            "selected_epoch": best_epoch,
            "validation_balanced_accuracy": validation_score[0],
            **metrics(y[test_idx], predictions[test_idx]),
        }
        fold_reports.append(fold_result)
        print(json.dumps(fold_result), flush=True)
    output = {
        "model": "small_2d_cnn",
        "input": "full-clip 64x96 Log-Mel map",
        "protocol": "5-fold source-grouped outer test with inner epoch selection",
        **metrics(y, predictions),
        "fold_balanced_accuracy_mean": float(
            np.mean([row["balanced_accuracy"] for row in fold_reports])
        ),
        "fold_balanced_accuracy_std": float(
            np.std([row["balanced_accuracy"] for row in fold_reports])
        ),
        "folds": fold_reports,
    }
    (RESULT_DIR / "audio_context_cnn_results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in output.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
