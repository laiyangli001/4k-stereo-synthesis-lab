from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .output import ensure_bchw, ensure_b1hw, match_depth

DISTILL_ANY_DEPTH_BASE_NAME = "Distill-Any-Depth-Base"
DISTILL_ANY_DEPTH_BASE_MODEL_ID = "lc700x/Distill-Any-Depth-Base-hf"
DISTILL_ANY_DEPTH_BASE_RESOLUTION = 518
DISTILL_ANY_DEPTH_PATCH_SIZE = 14


@dataclass(frozen=True)
class DepthProviderInfo:
    provider: str
    model_name: str
    model_id: str
    depth_resolution: int
    cache_dir: str
    load_mode: str = "online"

    def to_report(self) -> dict[str, str | int | bool]:
        return asdict(self)


def default_lab_cache_dir() -> Path:
    root = Path(__file__).resolve().parents[2]
    return root / "models"


def distill_base_518_info(cache_dir: str | Path | None = None) -> DepthProviderInfo:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return DepthProviderInfo(
        provider="transformers.AutoModelForDepthEstimation",
        model_name=DISTILL_ANY_DEPTH_BASE_NAME,
        model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
        depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
        cache_dir=str(cache),
    )


def _nearest_multiple(value: int, patch: int) -> int:
    down = (value // patch) * patch
    up = down + patch
    return max(1, up if abs(up - value) <= abs(value - down) else down)


def _model_input_size(height: int, width: int, target: int, patch: int) -> tuple[int, int]:
    longest = max(height, width)
    scale = target / float(longest) if longest != target else 1.0
    resized_h = max(1, int(round(height * scale)))
    resized_w = max(1, int(round(width * scale)))
    return _nearest_multiple(resized_h, patch), _nearest_multiple(resized_w, patch)


def _normalize_depth(depth: torch.Tensor) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    flat = depth.flatten(start_dim=2)
    amin = flat.amin(dim=-1).view(depth.shape[0], 1, 1, 1)
    amax = flat.amax(dim=-1).view(depth.shape[0], 1, 1, 1)
    return ((depth - amin) / (amax - amin).clamp_min(1e-6)).clamp(0, 1)


class DistillAnyDepthBase518:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
        force_download: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.dtype = dtype or (torch.float16 if self.device.type == "cuda" else torch.float32)
        self.local_files_only = bool(local_files_only)
        self.force_download = bool(force_download)
        self.info = DepthProviderInfo(
            provider="transformers.AutoModelForDepthEstimation",
            model_name=DISTILL_ANY_DEPTH_BASE_NAME,
            model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            cache_dir=str(self.cache_dir),
            load_mode="local_files_only" if self.local_files_only else "online_force_download" if self.force_download else "online",
        )
        self._model = None

    def load(self):
        if self._model is not None:
            return self._model

        from transformers import AutoModelForDepthEstimation

        kwargs = {
            "cache_dir": str(self.cache_dir),
            "dtype": self.dtype,
            "weights_only": True,
            "local_files_only": self.local_files_only,
            "force_download": self.force_download,
        }
        model = AutoModelForDepthEstimation.from_pretrained(
            DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            **kwargs,
        )

        self._model = model.to(self.device).eval()
        return self._model

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = ensure_bchw(rgb, name="rgb").to(self.device).float().clamp(0, 1)
        _, _, height, width = rgb.shape
        input_h, input_w = _model_input_size(
            height,
            width,
            DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            DISTILL_ANY_DEPTH_PATCH_SIZE,
        )

        tensor = F.interpolate(
            rgb,
            size=(input_h, input_w),
            mode="bicubic" if self.device.type == "cuda" else "bilinear",
            align_corners=False,
            antialias=True if self.device.type == "cuda" else False,
        ).to(self.dtype)

        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        model = self.load()
        use_autocast = self.device.type == "cuda" and self.dtype == torch.float16
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=use_autocast):
            predicted = model(pixel_values=tensor).predicted_depth

        depth = ensure_b1hw(predicted)
        depth = _normalize_depth(depth)
        return match_depth(depth, height, width)


def estimate_distill_any_depth_base_518(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    provider = DistillAnyDepthBase518(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    return provider.predict(rgb), provider.info
