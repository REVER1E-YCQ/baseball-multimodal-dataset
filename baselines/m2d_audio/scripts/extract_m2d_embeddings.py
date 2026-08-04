from __future__ import annotations

import argparse
import csv
import hashlib
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.io import wavfile
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 16_000
TOKEN_DIMENSION = 768
POOLING_STATISTICS = ("mean", "std", "max")
REQUIRED_COLUMNS = {
    "uid",
    "label",
    "source_id",
    "protocol_role",
    "window_name",
    "window_path",
}


def read_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Manifest has no header: {path}")
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Window manifest is missing columns: {sorted(missing)}")
        return list(reader), list(reader.fieldnames)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def resolve_window_path(value: str, manifest_path: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def load_wave(path: Path) -> np.ndarray:
    sample_rate, data = wavfile.read(path)
    original_dtype = data.dtype
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        data = data.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if data.ndim != 1:
        raise ValueError(f"Unexpected waveform shape for {path}: {data.shape}")
    if int(sample_rate) != TARGET_SAMPLE_RATE:
        divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        data = resample_poly(
            data,
            TARGET_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        )
    return np.nan_to_num(data).astype(np.float32, copy=False)


def batched(rows: list[dict[str, str]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def load_model(checkpoint: Path, m2d_root: Path, device: torch.device):
    examples_root = m2d_root / "examples"
    portable_loader = examples_root / "portable_m2d.py"
    if not portable_loader.is_file():
        raise FileNotFoundError(
            f"Cannot find M2D portable loader at {portable_loader}. "
            "Clone the pinned upstream M2D repository first."
        )
    sys.path.insert(0, str(examples_root.resolve()))
    try:
        from portable_m2d import PortableM2D
        model = PortableM2D(weight_file=str(checkpoint), flat_features=True)
    finally:
        sys.path.pop(0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model = model.eval().to(device)
    if int(model.cfg.sample_rate) != TARGET_SAMPLE_RATE:
        raise AssertionError(f"Unexpected M2D sample rate: {model.cfg.sample_rate}")
    if list(model.cfg.patch_size) != [16, 4]:
        raise AssertionError(f"Expected the 40 ms M2D patch size [16, 4], got {model.cfg.patch_size}")
    return model


def make_batch(
    rows: list[dict[str, str]],
    manifest_path: Path,
    device: torch.device,
) -> torch.Tensor:
    waves = [
        torch.from_numpy(load_wave(resolve_window_path(row["window_path"], manifest_path)))
        for row in rows
    ]
    lengths = {wave.numel() for wave in waves}
    if len(lengths) != 1:
        raise ValueError(f"Batch contains unequal waveform lengths: {sorted(lengths)}")
    return torch.stack(waves).to(device)


def pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3 or tokens.shape[-1] != TOKEN_DIMENSION:
        raise ValueError(f"Unexpected M2D timestamp embedding shape: {tuple(tokens.shape)}")
    return torch.cat(
        [
            tokens.mean(dim=1),
            tokens.std(dim=1, unbiased=False),
            tokens.amax(dim=1),
        ],
        dim=1,
    )


def extract_embeddings(
    windows_manifest: Path,
    output_path: Path,
    checkpoint: Path,
    m2d_root: Path,
    windows: set[str],
    device_name: str = "auto",
    batch_size: int = 16,
    limit_per_window: int = 0,
    amp: bool = False,
    expected_checkpoint_sha256: str = "",
) -> int:
    windows_manifest = windows_manifest.resolve()
    output_path = output_path.resolve()
    checkpoint = checkpoint.resolve()
    m2d_root = m2d_root.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    checkpoint_hash = file_sha256(checkpoint)
    if (
        expected_checkpoint_sha256
        and checkpoint_hash.lower() != expected_checkpoint_sha256.lower()
    ):
        raise ValueError(
            "Checkpoint SHA256 mismatch: "
            f"expected {expected_checkpoint_sha256.lower()}, got {checkpoint_hash}"
        )

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    rows, _ = read_manifest(windows_manifest)
    if windows:
        rows = [row for row in rows if row["window_name"] in windows]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["window_name"]].append(row)
    grouped = {
        name: sorted(group, key=lambda item: item["uid"])[
            : limit_per_window or None
        ]
        for name, group in grouped.items()
    }
    if not grouped or not sum(map(len, grouped.values())):
        raise ValueError("No matching windows were found")

    model = load_model(checkpoint, m2d_root, device)
    feature_fields = [
        f"feat_m2d_40ms_last_{statistic}_{index:04d}"
        for statistic in POOLING_STATISTICS
        for index in range(TOKEN_DIMENSION)
    ]
    metadata_fields = [
        "uid",
        "label",
        "source_id",
        "protocol_role",
        "window_name",
        "encoder",
        "encoder_training_epochs",
        "embedding_pooling",
        "checkpoint_sha256",
        "m2d_git_commit",
        "inference_precision",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    commit = git_commit(m2d_root)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=metadata_fields + feature_fields)
        writer.writeheader()
        for window_name in sorted(grouped):
            for batch_rows in batched(grouped[window_name], batch_size):
                audio = make_batch(batch_rows, windows_manifest, device)
                amp_enabled = bool(amp and device.type == "cuda")
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    tokens, _timestamps = model.get_timestamp_embeddings(audio)
                    pooled = pool_tokens(tokens)
                values = pooled.float().cpu().numpy()
                expected_shape = (len(batch_rows), len(feature_fields))
                if values.shape != expected_shape:
                    raise AssertionError(
                        f"Expected embedding shape {expected_shape}, got {values.shape}"
                    )
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Non-finite embedding in {window_name}")
                for row, embedding in zip(batch_rows, values, strict=True):
                    output = {
                        "uid": row["uid"],
                        "label": row["label"],
                        "source_id": row["source_id"],
                        "protocol_role": row["protocol_role"],
                        "window_name": row["window_name"],
                        "encoder": "m2d_vit_base_80x200_patch16x4_40ms",
                        "encoder_training_epochs": 0,
                        "embedding_pooling": "timestamp_tokens_mean_std_max",
                        "checkpoint_sha256": checkpoint_hash,
                        "m2d_git_commit": commit,
                        "inference_precision": "amp_fp16" if amp_enabled else "fp32",
                    }
                    output.update(
                        {
                            field: float(value)
                            for field, value in zip(feature_fields, embedding, strict=True)
                        }
                    )
                    writer.writerow(output)
                processed += len(batch_rows)
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frozen M2D 40 ms timestamp-token statistics."
    )
    parser.add_argument("--windows-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--m2d-root", type=Path, required=True)
    parser.add_argument(
        "--window",
        action="append",
        default=["event_200ms", "pre_200ms"],
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-per-window", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--expected-checkpoint-sha256", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = extract_embeddings(
        args.windows_manifest,
        args.out,
        args.checkpoint,
        args.m2d_root,
        set(args.window),
        args.device,
        args.batch_size,
        args.limit_per_window,
        args.amp,
        args.expected_checkpoint_sha256,
    )
    print(f"Wrote {count} embeddings to {args.out.resolve()}")


if __name__ == "__main__":
    main()
