from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import resample_poly
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset

from .short_contact_benchmark import LABEL_TO_INT

TARGET_SAMPLE_RATE = 16_000


@dataclass(frozen=True)
class FinetuneConfig:
    mode: str = "lora"
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    unfreeze_layers: int = 2
    lr: float = 3e-4
    head_lr: float = 1e-3
    batch_size: int = 32
    max_epochs: int = 60
    inner_splits: int = 3
    patience: int = 6


@dataclass(frozen=True)
class FinetuneProvenance:
    backbone: str
    upstream_revision: str
    checkpoint_sha256: str
    mode: str
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    unfreeze_layers: int
    lr: float
    head_lr: float
    max_epochs: int
    inner_splits: int
    pretraining_epochs_on_backbone: int = 0


def build_trainable_model(
    model: torch.nn.Module,
    config: FinetuneConfig,
    token_dimension: int,
) -> tuple[torch.nn.Module, torch.nn.Linear]:
    """Freeze the backbone and attach the training adapter.

    ``mode="lora"`` adds LoRA to the attention projections (rank 8 by
    default); ``mode="unfreeze_top"`` unfreezes the last ``N`` blocks
    entirely. The head is a fresh mean-pooled linear layer either way.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if config.mode == "unfreeze_top":
        blocks = model.backbone.blocks
        for block in blocks[-config.unfreeze_layers:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        trainable_model = model
    elif config.mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["attn.qkv", "attn.proj"],
        )
        trainable_model = get_peft_model(model, lora_config)
    else:
        raise ValueError(
            f"Unknown fine-tuning mode {config.mode!r}; expected "
            "'lora' or 'unfreeze_top'"
        )
    head = torch.nn.Linear(token_dimension, 2)
    return trainable_model, head


def to_model_waveform(
    waveform: np.ndarray, sample_rate: int
) -> np.ndarray:
    """Mirror the locked encoder's input preprocessing (int16 scale,
    mono, resample to 16 kHz)."""
    audio = waveform.astype(np.float32, copy=False)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / float(
            max(abs(info.min), info.max)
        )
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        audio = resample_poly(
            audio,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )
    return np.nan_to_num(audio).astype(np.float32, copy=False)


class WindowDataset(Dataset):
    """Event-window wavs keyed by uid with labels and groups."""

    def __init__(
        self,
        windows: pd.DataFrame,
        manifest_root: Path,
        uids: list[str],
    ) -> None:
        self._rows: list[tuple[str, Path, int, str]] = []
        by_uid = windows.set_index("uid")
        for uid in uids:
            row = by_uid.loc[uid]
            path = Path(str(row["window_path"]))
            if not path.is_absolute():
                path = manifest_root / path
            self._rows.append(
                (
                    str(uid),
                    path,
                    int(LABEL_TO_INT[str(row["label"])]),
                    str(row["lineage_group_id"]),
                )
            )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        uid, path, label, group = self._rows[index]
        return {
            "uid": uid,
            "group": group,
            "label": label,
            "waveform": np.load(path.with_suffix(".npy"), allow_pickle=False)
            if path.with_suffix(".npy").is_file()
            else self._read_wav(path),
        }

    @staticmethod
    def _read_wav(path: Path) -> np.ndarray:
        from scipy.io import wavfile

        sample_rate, raw = wavfile.read(path)
        return to_model_waveform(raw, int(sample_rate))


class WaveformCollator:
    def __call__(
        self, items: list[dict[str, object]]
    ) -> dict[str, object]:
        uids = [str(item["uid"]) for item in items]
        groups = [str(item["group"]) for item in items]
        labels = torch.as_tensor(
            [int(item["label"]) for item in items], dtype=torch.long
        )
        lengths = [len(np.asarray(item["waveform"])) for item in items]
        max_length = max(lengths)
        stacked = np.zeros((len(items), max_length), dtype=np.float32)
        for position, item in enumerate(items):
            waveform = np.asarray(item["waveform"])
            stacked[position, : len(waveform)] = waveform
        return {
            "uid": uids,
            "group": groups,
            "label": labels,
            "waveform": torch.from_numpy(stacked),
        }


def mean_pooled_logits(
    peft_model: torch.nn.Module,
    head: torch.nn.Linear,
    waveform_batch: torch.Tensor,
) -> torch.Tensor:
    tokens = peft_model.encode(waveform_batch)  # [B, T, D]
    pooled = tokens.mean(dim=1)
    return head(pooled)


def _balanced_accuracy_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> float:
    predictions = torch.argmax(logits, dim=1).cpu().numpy()
    return float(
        balanced_accuracy_score(labels.cpu().numpy(), predictions)
    )


def run_finetune_pilot(
    model_factory,
    token_dimension: int,
    provenance: FinetuneProvenance,
    config: FinetuneConfig,
    windows: pd.DataFrame,
    manifest_root: Path,
    folds: pd.DataFrame,
    seed: int,
    output_dir: Path,
) -> dict[str, object]:
    """Per-outer-fold LoRA fine-tuning with inner grouped early stopping.

    The test fold never touches fitting, early stopping, or any
    hyper-parameter choice: the LoRA weights, the early-stopping decision
    and the final epoch come from the training fold and its inner grouped
    validation split only.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = folds.sort_values("uid").reset_index(drop=True)
    all_uids = [str(uid) for uid in ordered["uid"]]
    label_by_uid = dict(
        zip(
            windows["uid"].astype(str),
            windows["label"].astype(str),
        )
    )
    group_by_uid = dict(
        zip(
            windows["uid"].astype(str),
            windows["lineage_group_id"].astype(str),
        )
    )
    outer_fold_by_uid = dict(
        zip(ordered["uid"].astype(str), ordered["outer_fold"].astype(int))
    )

    model, device = model_factory()
    del model

    prediction_rows: list[dict[str, object]] = []
    trace_rows: list[dict[str, object]] = []
    for outer_fold in sorted(set(outer_fold_by_uid.values())):
        train_uids = [
            uid
            for uid in all_uids
            if outer_fold_by_uid[uid] != outer_fold
        ]
        test_uids = [
            uid
            for uid in all_uids
            if outer_fold_by_uid[uid] == outer_fold
        ]
        fold_seed = seed + int(outer_fold)
        inner_labels = np.asarray(
            [LABEL_TO_INT[label_by_uid[uid]] for uid in train_uids],
            dtype=int,
        )
        inner_groups = np.asarray(
            [group_by_uid[uid] for uid in train_uids], dtype=object
        )
        splitter = StratifiedGroupKFold(
            n_splits=config.inner_splits,
            shuffle=True,
            random_state=fold_seed,
        )
        inner_folds = list(
            splitter.split(np.zeros(len(train_uids)), inner_labels, inner_groups)
        )
        val_positions, train_positions = inner_folds[-1]
        inner_train_uids = [train_uids[i] for i in train_positions]
        inner_val_uids = [train_uids[i] for i in val_positions]

        model, device = model_factory()
        peft_model, head = build_trainable_model(
            model, config, token_dimension
        )
        peft_model = peft_model.to(device)
        head = head.to(device)
        parameters = [
            parameter
            for parameter in list(peft_model.parameters())
            + list(head.parameters())
            if parameter.requires_grad
        ]
        head_parameters = [
            parameter for parameter in head.parameters()
        ]
        lora_parameters = [
            parameter
            for parameter in peft_model.parameters()
            if parameter.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": lora_parameters, "lr": config.lr},
                {"params": head_parameters, "lr": config.head_lr},
            ]
        )
        criterion = torch.nn.CrossEntropyLoss()

        def epoch_evaluate(uid_list: list[str]) -> float:
            dataset = WindowDataset(windows, manifest_root, uid_list)
            loader = DataLoader(
                dataset,
                batch_size=config.batch_size,
                shuffle=False,
                collate_fn=WaveformCollator(),
            )
            logits_all: list[torch.Tensor] = []
            labels_all: list[torch.Tensor] = []
            with torch.inference_mode():
                for batch in loader:
                    waveform = batch["waveform"].to(device)
                    logits = mean_pooled_logits(peft_model, head, waveform)
                    logits_all.append(logits.cpu())
                    labels_all.append(batch["label"])
            logits = torch.cat(logits_all)
            labels = torch.cat(labels_all)
            return _balanced_accuracy_from_logits(logits, labels)

        best_val_ba = -1.0
        patience_left = config.patience
        final_epoch = 0
        best_state: dict[str, object] = {}
        for epoch in range(1, config.max_epochs + 1):
            train_dataset = WindowDataset(
                windows, manifest_root, inner_train_uids
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                collate_fn=WaveformCollator(),
            )
            peft_model.train()
            head.train()
            for batch in train_loader:
                waveform = batch["waveform"].to(device)
                labels = batch["label"].to(device)
                optimizer.zero_grad()
                logits = mean_pooled_logits(peft_model, head, waveform)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
            peft_model.eval()
            head.eval()
            train_ba = epoch_evaluate(inner_train_uids)
            val_ba = epoch_evaluate(inner_val_uids)
            trace_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "epoch": epoch,
                    "train_balanced_accuracy": train_ba,
                    "inner_val_balanced_accuracy": val_ba,
                }
            )
            if val_ba > best_val_ba + 1e-9:
                best_val_ba = val_ba
                patience_left = config.patience
                final_epoch = epoch
                best_state = {
                    "peft": {
                        key: value.detach().cpu().clone()
                        for key, value in peft_model.state_dict().items()
                    },
                    "head": {
                        key: value.detach().cpu().clone()
                        for key, value in head.state_dict().items()
                    },
                }
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        # Restore the weights of the best inner-validation epoch so the
        # test-fold prediction comes from the early-stopped model.
        if best_state:
            peft_model.load_state_dict(best_state["peft"])
            head.load_state_dict(best_state["head"])

        # Final model: the one at final_epoch (weights at the best val
        # epoch). We re-train to the best epoch from scratch to get the
        # cleanest weights is too costly for a pilot; the trace records
        # the gap honestly. Predict the test fold with the current model.
        test_dataset = WindowDataset(windows, manifest_root, test_uids)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=WaveformCollator(),
        )
        peft_model.eval()
        head.eval()
        with torch.inference_mode():
            for batch in test_loader:
                waveform = batch["waveform"].to(device)
                logits = mean_pooled_logits(peft_model, head, waveform)
                probabilities = torch.softmax(logits, dim=1)[:, 1]
                for position, uid in enumerate(batch["uid"]):
                    prediction_rows.append(
                        {
                            "encoder": provenance.backbone,
                            "outer_fold": int(outer_fold),
                            "uid": str(uid),
                            "label": label_by_uid[str(uid)],
                            "lineage_group_id": group_by_uid[str(uid)],
                            "y_true": int(LABEL_TO_INT[label_by_uid[str(uid)]]),
                            "score_ground_ball": float(
                                probabilities[position]
                            ),
                            "y_pred": int(
                                (probabilities[position] >= 0.5).item()
                            ),
                            "final_epoch": final_epoch,
                            "best_inner_val_ba": best_val_ba,
                        }
                    )
        del peft_model, head, optimizer

    predictions = pd.DataFrame(prediction_rows).sort_values("uid")
    trace = pd.DataFrame(trace_rows)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    trace.to_csv(output_dir / "finetune_trace.csv", index=False)
    event_ba = float(
        balanced_accuracy_score(
            predictions["y_true"].to_numpy(),
            predictions["y_pred"].to_numpy(),
        )
    )
    return {
        "oof_predictions": predictions,
        "trace": trace,
        "event_balanced_accuracy": event_ba,
        "provenance": provenance.__dict__,
        "output_dir": output_dir,
    }
