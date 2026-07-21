from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import librosa
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")


LABEL_TO_ID = {"ground_ball": 0, "fly_ball": 1}
ID_TO_LABEL = {value: key for key, value in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    sample_rate: int = 48_000
    clip_seconds: float = 0.5
    pre_event_seconds: float = 0.1
    n_fft: int = 2048
    win_length: int = 1200
    hop_length: int = 480
    n_mels: int = 64
    n_mfcc: int = 20
    batch_size: int = 32
    max_epochs: int = 60
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.15


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)


def parse_source_id(source_text: str) -> str:
    for line in source_text.splitlines():
        if line.startswith("source_id:"):
            return line.split(":", 1)[1].strip()
    return ""


def build_manifest(dataset_root: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for audio_path in sorted(dataset_root.rglob("audio.wav")):
        sample_dir = audio_path.parent
        with (sample_dir / "sample.csv").open(encoding="utf-8-sig", newline="") as handle:
            metadata = next(csv.DictReader(handle))
        label = metadata["label"].strip()
        if label not in LABEL_TO_ID:
            continue
        source_text = (sample_dir / "source.txt").read_text(encoding="utf-8")
        source_id = parse_source_id(source_text)
        if not source_id:
            source_id = f"path:{audio_path.relative_to(dataset_root).as_posix()}"
        with audio_path.open("rb") as handle:
            audio_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        rows.append(
            {
                "sample_id": metadata["sample_id"].strip(),
                "label": label,
                "label_id": LABEL_TO_ID[label],
                "event_start": float(metadata["event_start"]),
                "event_end": float(metadata["event_end"]),
                "source_id": source_id,
                "audio_sha256": audio_sha256,
                "audio_path": str(audio_path.resolve()),
            }
        )
    manifest = pd.DataFrame(rows).sort_values(["label_id", "sample_id"]).reset_index(drop=True)
    if len(manifest) == 0:
        raise RuntimeError(f"No samples found under {dataset_root}")
    if manifest["source_id"].duplicated().any():
        duplicates = manifest.loc[manifest["source_id"].duplicated(False), "source_id"].tolist()
        raise RuntimeError(f"Repeated source IDs require grouped splitting: {duplicates[:5]}")
    return manifest


def create_split(manifest: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    test_total = int(math.ceil(len(manifest) * 0.30))
    label_counts = manifest["label_id"].value_counts().sort_index()
    exact_targets = label_counts * 0.30
    test_targets = np.floor(exact_targets).astype(int)
    remaining = test_total - int(test_targets.sum())
    for label_id in (exact_targets - test_targets).sort_values(ascending=False).index[:remaining]:
        test_targets.loc[label_id] += 1

    test_indices: list[int] = []
    for label_id, target in test_targets.items():
        label_rows = manifest.index[manifest["label_id"] == label_id].to_numpy()
        hash_groups = [
            group.index.to_numpy()
            for _, group in manifest.loc[label_rows].groupby("audio_sha256", sort=False)
        ]
        rng.shuffle(hash_groups)
        needed = int(target)
        deferred: list[np.ndarray] = []
        for group in hash_groups:
            if len(group) <= needed:
                test_indices.extend(group.tolist())
                needed -= len(group)
            else:
                deferred.append(group)
        if needed != 0:
            raise RuntimeError(f"Could not build an exact grouped test split for label {label_id}")

    test_indices = np.asarray(sorted(test_indices), dtype=int)
    train_indices = np.setdiff1d(np.arange(len(manifest)), test_indices)
    split = manifest.copy()
    split["split"] = ""
    split.loc[train_indices, "split"] = "train"
    split.loc[test_indices, "split"] = "test"
    return split.sort_values(["split", "label_id", "sample_id"]).reset_index(drop=True)


def read_event_clip(row: pd.Series, cfg: Config) -> tuple[np.ndarray, float, float]:
    audio, sample_rate = librosa.load(row["audio_path"], sr=None, mono=True)
    if sample_rate != cfg.sample_rate:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=cfg.sample_rate)
        sample_rate = cfg.sample_rate

    event_center = (float(row["event_start"]) + float(row["event_end"])) / 2.0
    clip_start = event_center - cfg.pre_event_seconds
    clip_end = clip_start + cfg.clip_seconds
    target_length = int(round(cfg.clip_seconds * cfg.sample_rate))

    source_start = max(0, int(round(clip_start * sample_rate)))
    source_end = min(len(audio), int(round(clip_end * sample_rate)))
    target_start = max(0, -int(round(clip_start * sample_rate)))

    clip = np.zeros(target_length, dtype=np.float32)
    available = max(0, source_end - source_start)
    if available:
        copy_length = min(available, target_length - target_start)
        clip[target_start : target_start + copy_length] = audio[source_start : source_start + copy_length]
    return clip, clip_start, clip_end


def peak_normalize(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-8:
        return audio.astype(np.float32, copy=True)
    return (audio * (target_peak / peak)).astype(np.float32)


def frame_statistics(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    flat = values.reshape(values.shape[0], -1) if values.ndim > 1 else values.reshape(1, -1)
    mean = flat.mean(axis=1)
    std = flat.std(axis=1)
    maximum = flat.max(axis=1)
    minimum = flat.min(axis=1)
    median = np.median(flat, axis=1)
    centered = flat - mean[:, None]
    safe_std = np.maximum(std, 1e-10)
    skewness = np.mean(centered**3, axis=1) / safe_std**3
    kurtosis = np.mean(centered**4, axis=1) / safe_std**4
    return np.concatenate([mean, std, maximum, minimum, median, skewness, kurtosis]).astype(np.float32)


def traditional_features(audio: np.ndarray, cfg: Config) -> np.ndarray:
    original_peak = float(np.max(np.abs(audio)))
    original_rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
    original_energy = float(np.sum(audio**2))
    normalized = peak_normalize(audio)

    stft = librosa.stft(
        normalized,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hamming",
        center=True,
    )
    magnitude = np.abs(stft)
    power = magnitude**2

    mel = librosa.feature.melspectrogram(
        S=power,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        fmin=20,
        fmax=cfg.sample_rate / 2,
    )
    log_mel = librosa.power_to_db(mel + 1e-12, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=cfg.n_mfcc)

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=cfg.sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=cfg.sample_rate)
    rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=cfg.sample_rate, roll_percent=0.85)
    zcr = librosa.feature.zero_crossing_rate(
        normalized, frame_length=cfg.win_length, hop_length=cfg.hop_length
    )
    rms = librosa.feature.rms(S=magnitude, frame_length=cfg.n_fft, hop_length=cfg.hop_length)

    frequencies = librosa.fft_frequencies(sr=cfg.sample_rate, n_fft=cfg.n_fft)
    band_edges = [0, 500, 1000, 2000, 4000, 8000, 12000, 16000, 24000]
    total_power = np.maximum(power.sum(axis=0), 1e-12)
    band_ratios = []
    for low, high in zip(band_edges[:-1], band_edges[1:]):
        mask = (frequencies >= low) & (frequencies < high)
        band_ratios.append(power[mask].sum(axis=0) / total_power)
    band_ratios_array = np.asarray(band_ratios)

    frame_rms = rms.reshape(-1)
    peak_frame = int(np.argmax(frame_rms))
    tail = frame_rms[peak_frame:]
    if len(tail) >= 2:
        x = np.arange(len(tail), dtype=np.float64)
        decay_slope = float(np.polyfit(x, np.log(tail + 1e-8), 1)[0])
    else:
        decay_slope = 0.0

    feature_blocks = [
        frame_statistics(log_mel),
        frame_statistics(mfcc),
        frame_statistics(centroid),
        frame_statistics(bandwidth),
        frame_statistics(rolloff),
        frame_statistics(zcr),
        frame_statistics(rms),
        frame_statistics(band_ratios_array),
        np.asarray([original_peak, original_rms, math.log1p(original_energy), decay_slope], dtype=np.float32),
    ]
    feature_vector = np.concatenate(feature_blocks)
    return np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def linear_log_spectrogram(audio: np.ndarray, cfg: Config) -> np.ndarray:
    normalized = peak_normalize(audio)
    stft = librosa.stft(
        normalized,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        window="hamming",
        center=True,
    )
    return np.log1p(np.abs(stft) ** 2).astype(np.float32)


def prepare_arrays(split: pd.DataFrame, cache_dir: Path, cfg: Config) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "prepared_arrays.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        if (
            len(cached["waveforms"]) == len(split)
            and int(cached["sample_rate"]) == cfg.sample_rate
            and float(cached["clip_seconds"]) == cfg.clip_seconds
        ):
            return {name: cached[name] for name in cached.files}

    waveforms = []
    traditional = []
    spectrograms = []
    clip_starts = []
    clip_ends = []
    for index, row in split.iterrows():
        clip, clip_start, clip_end = read_event_clip(row, cfg)
        waveforms.append(peak_normalize(clip))
        traditional.append(traditional_features(clip, cfg))
        spectrograms.append(linear_log_spectrogram(clip, cfg))
        clip_starts.append(clip_start)
        clip_ends.append(clip_end)
        if (index + 1) % 50 == 0 or index + 1 == len(split):
            print(f"Prepared {index + 1}/{len(split)} samples", flush=True)

    arrays = {
        "waveforms": np.stack(waveforms),
        "traditional": np.stack(traditional),
        "spectrograms": np.stack(spectrograms),
        "labels": split["label_id"].to_numpy(dtype=np.int64),
        "clip_starts": np.asarray(clip_starts, dtype=np.float32),
        "clip_ends": np.asarray(clip_ends, dtype=np.float32),
        "sample_rate": np.asarray(cfg.sample_rate),
        "clip_seconds": np.asarray(cfg.clip_seconds),
    }
    np.savez_compressed(cache_file, **arrays)
    return arrays


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=[ID_TO_LABEL[0], ID_TO_LABEL[1]],
            output_dict=True,
            zero_division=0,
        ),
    }


def save_predictions(
    method: str,
    split: pd.DataFrame,
    test_indices: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
) -> None:
    predictions = split.iloc[test_indices][["sample_id", "label", "source_id", "audio_path"]].copy()
    predictions["predicted_label"] = [ID_TO_LABEL[int(value)] for value in y_pred]
    predictions["fly_ball_probability"] = probabilities
    predictions["correct"] = predictions["label"] == predictions["predicted_label"]
    predictions.to_csv(output_dir / f"{method}_test_predictions.csv", index=False, encoding="utf-8-sig")


def train_traditional(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced", probability=True, random_state=seed)),
        ]
    )
    search = GridSearchCV(
        pipeline,
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
        },
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=-1,
        refit=True,
    )
    start = time.time()
    search.fit(features[train_indices], labels[train_indices])
    y_pred = search.predict(features[test_indices])
    probabilities = search.predict_proba(features[test_indices])[:, 1]
    metrics = calculate_metrics(labels[test_indices], y_pred, probabilities)
    metrics["best_parameters"] = search.best_params_
    metrics["cross_validation_macro_f1"] = float(search.best_score_)
    metrics["training_seconds"] = time.time() - start
    metrics["feature_dimension"] = int(features.shape[1])
    joblib.dump(search.best_estimator_, output_dir / "traditional_svm.joblib")
    save_predictions("traditional", split, test_indices, y_pred, probabilities, output_dir)
    return metrics


class AudioArrayDataset(Dataset):
    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        indices: np.ndarray,
        mode: str,
        augment: bool,
        seed: int,
    ):
        self.inputs = inputs
        self.labels = labels
        self.indices = np.asarray(indices)
        self.mode = mode
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        index = int(self.indices[item])
        x = self.inputs[index].copy()
        if self.augment:
            if self.mode == "waveform":
                gain = self.rng.uniform(0.85, 1.15)
                shift = int(self.rng.integers(-240, 241))
                x = np.roll(x * gain, shift)
                noise_scale = self.rng.uniform(0.0, 0.005)
                x = x + self.rng.normal(0.0, noise_scale, size=x.shape).astype(np.float32)
                x = np.clip(x, -1.0, 1.0)
            else:
                if self.rng.random() < 0.5:
                    width = int(self.rng.integers(1, max(2, x.shape[0] // 16)))
                    start = int(self.rng.integers(0, max(1, x.shape[0] - width)))
                    x[start : start + width, :] = 0
                if self.rng.random() < 0.5:
                    width = int(self.rng.integers(1, max(2, x.shape[1] // 8)))
                    start = int(self.rng.integers(0, max(1, x.shape[1] - width)))
                    x[:, start : start + width] = 0
        tensor = torch.from_numpy(x).float().unsqueeze(0)
        return tensor, torch.tensor(self.labels[index], dtype=torch.long)


class SpectrumCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(7, 3), padding=(3, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),
            nn.Conv2d(16, 32, kernel_size=(5, 3), padding=(2, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((4, 2)),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(64, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class WaveformCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=63, stride=4, padding=31),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(16, 32, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.35), nn.Linear(128, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels = []
    predictions = []
    probabilities = []
    for inputs, target in loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probability = torch.softmax(logits, dim=1)[:, 1]
        prediction = torch.argmax(logits, dim=1)
        labels.extend(target.numpy().tolist())
        predictions.extend(prediction.cpu().numpy().tolist())
        probabilities.extend(probability.cpu().numpy().tolist())
    return np.asarray(labels), np.asarray(predictions), np.asarray(probabilities)


def train_cnn(
    method: str,
    inputs: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    split: pd.DataFrame,
    output_dir: Path,
    cfg: Config,
    device: torch.device,
) -> dict[str, object]:
    inner_train, validation = train_test_split(
        train_indices,
        test_size=cfg.validation_fraction,
        random_state=cfg.seed,
        stratify=labels[train_indices],
    )

    if method == "spectrum_cnn":
        train_mean = float(inputs[inner_train].mean())
        train_std = float(inputs[inner_train].std() + 1e-8)
        normalized_inputs = ((inputs - train_mean) / train_std).astype(np.float32)
        model = SpectrumCNN()
        mode = "spectrum"
        normalization = {"mean": train_mean, "std": train_std}
    else:
        normalized_inputs = inputs.astype(np.float32)
        model = WaveformCNN()
        mode = "waveform"
        normalization = {"peak_normalized": True}

    train_dataset = AudioArrayDataset(normalized_inputs, labels, inner_train, mode, True, cfg.seed)
    validation_dataset = AudioArrayDataset(normalized_inputs, labels, validation, mode, False, cfg.seed + 1)
    test_dataset = AudioArrayDataset(normalized_inputs, labels, test_indices, mode, False, cfg.seed + 2)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    class_counts = np.bincount(labels[inner_train], minlength=2)
    class_weights = len(inner_train) / (2.0 * np.maximum(class_counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    checkpoint_path = output_dir / f"{method}.pt"
    start = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch_inputs, target in train_loader:
            batch_inputs = batch_inputs.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(target)
            seen += len(target)

        validation_true, validation_pred, validation_prob = evaluate_loader(model, validation_loader, device)
        validation_f1 = f1_score(validation_true, validation_pred, average="macro")
        validation_balanced = balanced_accuracy_score(validation_true, validation_pred)
        scheduler.step(validation_f1)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(seen, 1),
                "validation_macro_f1": float(validation_f1),
                "validation_balanced_accuracy": float(validation_balanced),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"{method} epoch {epoch:02d}: loss={history[-1]['train_loss']:.4f} "
            f"val_f1={validation_f1:.4f} val_bal_acc={validation_balanced:.4f}",
            flush=True,
        )

        if validation_f1 > best_score + 1e-5:
            best_score = float(validation_f1)
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "normalization": normalization,
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    y_true, y_pred, probabilities = evaluate_loader(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred, probabilities)
    metrics["best_epoch"] = best_epoch
    metrics["best_validation_macro_f1"] = best_score
    metrics["training_seconds"] = time.time() - start
    metrics["device"] = str(device)
    metrics["inner_train_samples"] = int(len(inner_train))
    metrics["validation_samples"] = int(len(validation))
    pd.DataFrame(history).to_csv(output_dir / f"{method}_training_history.csv", index=False)
    save_predictions(method, split, test_indices, y_pred, probabilities, output_dir)
    return metrics


def save_confusion_plot(results: dict[str, dict[str, object]], output_path: Path) -> None:
    methods = list(results)
    figure, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]
    for axis, method in zip(axes, methods):
        matrix = np.asarray(results[method]["confusion_matrix"])
        image = axis.imshow(matrix, cmap="Blues")
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="black")
        axis.set_title(method.replace("_", " "))
        axis.set_xticks([0, 1], [ID_TO_LABEL[0], ID_TO_LABEL[1]], rotation=20)
        axis.set_yticks([0, 1], [ID_TO_LABEL[0], ID_TO_LABEL[1]])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_comparison_plot(results: dict[str, dict[str, object]], output_path: Path) -> None:
    methods = list(results)
    metric_names = ["accuracy", "balanced_accuracy", "macro_f1", "roc_auc"]
    values = np.asarray([[float(results[method][metric]) for metric in metric_names] for method in methods])
    x = np.arange(len(methods))
    width = 0.18
    figure, axis = plt.subplots(figsize=(10, 5))
    for index, metric in enumerate(metric_names):
        axis.bar(x + (index - 1.5) * width, values[:, index], width, label=metric)
    axis.set_ylim(0, 1)
    axis.set_xticks(x, [method.replace("_", "\n") for method in methods])
    axis.set_ylabel("Score")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    cfg = Config(max_epochs=args.epochs)
    set_seed(cfg.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(args.dataset_root)
    split = create_split(manifest, cfg.seed)
    split.to_csv(args.output_dir / "dataset_split.csv", index=False, encoding="utf-8-sig")

    arrays = prepare_arrays(split, args.cache_dir, cfg)
    split["clip_start"] = arrays["clip_starts"]
    split["clip_end"] = arrays["clip_ends"]
    split.to_csv(args.output_dir / "dataset_split.csv", index=False, encoding="utf-8-sig")

    train_indices = np.flatnonzero(split["split"].to_numpy() == "train")
    test_indices = np.flatnonzero(split["split"].to_numpy() == "test")
    labels = arrays["labels"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summary = {
        "dataset_samples": int(len(split)),
        "train_samples": int(len(train_indices)),
        "test_samples": int(len(test_indices)),
        "class_counts_total": split["label"].value_counts().to_dict(),
        "class_counts_train": split.iloc[train_indices]["label"].value_counts().to_dict(),
        "class_counts_test": split.iloc[test_indices]["label"].value_counts().to_dict(),
        "config": asdict(cfg),
        "device": str(device),
        "torch_version": torch.__version__,
    }
    (args.output_dir / "experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    results: dict[str, dict[str, object]] = {}
    print("Training traditional feature SVM...", flush=True)
    results["traditional_svm"] = train_traditional(
        arrays["traditional"], labels, train_indices, test_indices, split, args.output_dir, cfg.seed
    )
    print("Training spectrum CNN...", flush=True)
    results["spectrum_cnn"] = train_cnn(
        "spectrum_cnn",
        arrays["spectrograms"],
        labels,
        train_indices,
        test_indices,
        split,
        args.output_dir,
        cfg,
        device,
    )
    print("Training waveform CNN...", flush=True)
    results["waveform_cnn"] = train_cnn(
        "waveform_cnn",
        arrays["waveforms"],
        labels,
        train_indices,
        test_indices,
        split,
        args.output_dir,
        cfg,
        device,
    )

    (args.output_dir / "metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison_rows = []
    for method, metrics in results.items():
        comparison_rows.append(
            {
                "method": method,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "roc_auc": metrics["roc_auc"],
                "training_seconds": metrics["training_seconds"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(args.output_dir / "method_comparison.csv", index=False)
    save_confusion_plot(results, args.output_dir / "confusion_matrices.png")
    save_comparison_plot(results, args.output_dir / "method_comparison.png")
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
