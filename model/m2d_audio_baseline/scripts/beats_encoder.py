from __future__ import annotations

import hashlib
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from .short_contact_benchmark import EncoderAdapter, EncoderProvenance


TARGET_SAMPLE_RATE = 16_000
TOKEN_DIMENSION = 768


class BEATsEncoderAdapter(EncoderAdapter):
    """Frozen BEATs iter3+ AS2M encoder behind the shared adapter seam."""

    def __init__(
        self,
        checkpoint: Path,
        beats_root: Path,
        device: str = "auto",
        expected_checkpoint_sha256: str = "",
    ) -> None:
        self._checkpoint = Path(checkpoint).resolve()
        self._beats_root = Path(beats_root).resolve()
        self._device_name = device
        self._expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._model = None
        self._device: torch.device | None = None

        if not self._checkpoint.is_file():
            raise FileNotFoundError(f"BEATs checkpoint is missing: {self._checkpoint}")
        checkpoint_hash = self._file_sha256(self._checkpoint)
        if (
            expected_checkpoint_sha256
            and checkpoint_hash.lower() != expected_checkpoint_sha256.lower()
        ):
            raise ValueError(
                "BEATs checkpoint SHA256 mismatch: "
                f"expected {expected_checkpoint_sha256.lower()}, "
                f"got {checkpoint_hash}"
            )
        upstream_revision = self._git_revision(self._beats_root)
        self.provenance = EncoderProvenance(
            name="beats_iter3plus_as2m",
            upstream_revision=upstream_revision,
            checkpoint_sha256=checkpoint_hash,
            precision="fp32",
            token_dimension=TOKEN_DIMENSION,
            training_epochs=0,
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _git_revision(root: Path) -> str:
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

    def _resolve_device(self) -> torch.device:
        if self._device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device_name = self._device_name
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(device_name)

    def _load_model(self) -> tuple[torch.nn.Module, torch.device]:
        if self._model is None:
            package_dir = (
                self._beats_root
                if (self._beats_root / "BEATs.py").is_file()
                else self._beats_root / "beats"
            )
            if not (package_dir / "BEATs.py").is_file():
                raise FileNotFoundError(
                    f"Cannot find the BEATs source under {self._beats_root}. "
                    "Clone the pinned upstream unilm repository first."
                )
            sys.path.insert(0, str(package_dir.resolve()))
            try:
                from BEATs import BEATs, BEATsConfig

                checkpoint = torch.load(
                    self._checkpoint,
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                model = BEATs(BEATsConfig(checkpoint["cfg"]))
                model.load_state_dict(checkpoint["model"], strict=True)
                del checkpoint
            finally:
                sys.path.pop(0)
            if model.predictor is not None:
                raise AssertionError(
                    "Expected a pre-trained BEATs encoder without predictor"
                )
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model = model.eval()
            device = self._resolve_device()
            self._model = model.to(device)
            self._device = device
        return self._model, self._device

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = waveform.astype(np.float32, copy=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if int(sample_rate) != TARGET_SAMPLE_RATE:
            divisor = math.gcd(int(sample_rate), TARGET_SAMPLE_RATE)
            audio = resample_poly(
                audio,
                TARGET_SAMPLE_RATE // divisor,
                int(sample_rate) // divisor,
            )
        audio = np.nan_to_num(audio).astype(np.float32, copy=False)

        model, device = self._load_model()
        tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
        # BEATs inference is forced to FP32 because FP16 previously produced
        # non-finite short-input embeddings.
        with torch.inference_mode():
            tokens, padding_mask = model.extract_features(tensor)
        if padding_mask is not None and bool(padding_mask.any()):
            raise AssertionError("Unexpected BEATs padding in a fixed-length input")
        result = tokens.float().cpu().numpy()
        if result.ndim != 3 or result.shape[-1] != TOKEN_DIMENSION:
            raise AssertionError(f"Unexpected BEATs token shape: {result.shape}")
        if not np.isfinite(result).all():
            raise FloatingPointError("BEATs returned non-finite tokens")
        return result[0]
