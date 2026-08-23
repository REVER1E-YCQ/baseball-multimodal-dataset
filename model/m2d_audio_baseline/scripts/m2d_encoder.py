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
PATCH_SIZE = [16, 4]


class M2DEncoderAdapter(EncoderAdapter):
    """Frozen M2D 40 ms encoder behind the shared benchmark adapter seam."""

    def __init__(
        self,
        checkpoint: Path,
        m2d_root: Path,
        device: str = "auto",
        precision: str = "amp_fp16",
        expected_checkpoint_sha256: str = "",
    ) -> None:
        if precision not in {"fp32", "amp_fp16"}:
            raise ValueError(f"Unsupported M2D precision: {precision!r}")
        self._checkpoint = Path(checkpoint).resolve()
        self._m2d_root = Path(m2d_root).resolve()
        self._device_name = device
        self._precision = precision
        self._expected_checkpoint_sha256 = expected_checkpoint_sha256
        self._model = None
        self._device: torch.device | None = None

        if not self._checkpoint.is_file():
            raise FileNotFoundError(f"M2D checkpoint is missing: {self._checkpoint}")
        checkpoint_hash = self._file_sha256(self._checkpoint)
        if (
            expected_checkpoint_sha256
            and checkpoint_hash.lower() != expected_checkpoint_sha256.lower()
        ):
            raise ValueError(
                "M2D checkpoint SHA256 mismatch: "
                f"expected {expected_checkpoint_sha256.lower()}, "
                f"got {checkpoint_hash}"
            )
        upstream_revision = self._git_revision(self._m2d_root)
        self.provenance = EncoderProvenance(
            name="m2d_vit_base_80x200p16x4_40ms",
            upstream_revision=upstream_revision,
            checkpoint_sha256=checkpoint_hash,
            precision=precision,
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
            examples_root = self._m2d_root / "examples"
            loader = examples_root / "portable_m2d.py"
            if not loader.is_file():
                raise FileNotFoundError(
                    f"Cannot find the M2D portable loader at {loader}. "
                    "Clone the pinned upstream M2D repository first."
                )
            sys.path.insert(0, str(examples_root.resolve()))
            try:
                from portable_m2d import PortableM2D

                model = PortableM2D(
                    weight_file=str(self._checkpoint), flat_features=True
                )
            finally:
                sys.path.pop(0)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            model = model.eval()
            if int(model.cfg.sample_rate) != TARGET_SAMPLE_RATE:
                raise AssertionError(
                    f"Unexpected M2D sample rate: {model.cfg.sample_rate}"
                )
            if list(model.cfg.patch_size) != PATCH_SIZE:
                raise AssertionError(
                    f"Expected the 40 ms M2D patch size {PATCH_SIZE}, "
                    f"got {list(model.cfg.patch_size)}"
                )
            device = self._resolve_device()
            self._model = model.to(device)
            self._device = device
        return self._model, self._device

    def encode_tokens(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        audio = waveform.astype(np.float32, copy=False)
        if np.issubdtype(audio.dtype, np.integer):
            info = np.iinfo(audio.dtype)
            audio = audio.astype(np.float32) / float(max(abs(info.min), info.max))
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
        amp_enabled = bool(
            self._precision == "amp_fp16" and device.type == "cuda"
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            tokens, _timestamps = model.get_timestamp_embeddings(tensor)
        result = tokens.float().cpu().numpy()
        if result.ndim != 3 or result.shape[-1] != TOKEN_DIMENSION:
            raise AssertionError(
                f"Unexpected M2D token shape: {result.shape}"
            )
        if not np.isfinite(result).all():
            raise FloatingPointError("M2D returned non-finite tokens")
        return result[0]

    def encode_layer_tokens(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Return per-block token outputs as [layer, token, feature].

        Each layer is the output after its transformer block, normalised with
        the backbone final norm, CLS dropped, and averaged per patch frame —
        the identical post-processing the last layer receives, so layer k and
        the last layer are directly comparable.
        """
        audio = self._prepare_audio(waveform, sample_rate)
        model, device = self._load_model()
        tensor = torch.from_numpy(audio).unsqueeze(0).to(device)
        patch_fbins = int(model.backbone.grid_size()[0])
        embed_d = int(model.backbone.patch_embed.proj.out_channels)
        unit_frames = int(model.cfg.input_size[1])
        patch_frames = int(model.backbone.patch_size()[1])
        from einops import rearrange

        x = model.to_normalized_feature(tensor)
        n_chunk = (x.shape[-1] + unit_frames - 1) // unit_frames
        if n_chunk != 1:
            raise NotImplementedError(
                "layer-wise extraction currently supports single-chunk "
                f"inputs, got {n_chunk} chunks"
            )
        # Mirror encode_lms: pad the frame axis to the patch boundary so the
        # token count matches the last layer exactly.
        pad_frames = (
            patch_frames - (x.shape[-1] % unit_frames % patch_frames)
        ) % patch_frames
        if pad_frames > 0:
            x = torch.nn.functional.pad(x, (0, pad_frames))

        layer_outputs: dict[int, torch.Tensor] = {}
        hooks = []
        for index, block in enumerate(model.backbone.blocks):
            hooks.append(
                block.register_forward_hook(
                    lambda _module, _input, output, index=index: (
                        layer_outputs.__setitem__(index, output.detach())
                    )
                )
            )
        try:
            amp_enabled = bool(
                self._precision == "amp_fp16" and device.type == "cuda"
            )
            with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                model.backbone.forward_encoder(x)
        finally:
            for hook in hooks:
                hook.remove()

        layers: list[np.ndarray] = []
        for index in sorted(layer_outputs):
            hidden = model.backbone.norm(layer_outputs[index])
            hidden = hidden[..., 1:, :]
            hidden = rearrange(
                hidden, "b (f t) d -> b t d f", f=patch_fbins, d=embed_d
            ).mean(-1)
            result = hidden.float().cpu().numpy()
            if result.ndim != 3 or result.shape[-1] != TOKEN_DIMENSION:
                raise AssertionError(
                    f"Unexpected layer {index} token shape: {result.shape}"
                )
            if not np.isfinite(result).all():
                raise FloatingPointError(
                    f"M2D layer {index} returned non-finite tokens"
                )
            layers.append(result[0])
        return np.stack(layers)

    def _prepare_audio(
        self, waveform: np.ndarray, sample_rate: int
    ) -> np.ndarray:
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
