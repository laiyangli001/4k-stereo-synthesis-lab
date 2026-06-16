from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from .depth_provider import (
    DISTILL_ANY_DEPTH_BASE_MODEL_ID,
    DISTILL_ANY_DEPTH_BASE_NAME,
    DISTILL_ANY_DEPTH_BASE_RESOLUTION,
    DISTILL_ANY_DEPTH_PATCH_SIZE,
    DepthProviderInfo,
    DistillAnyDepthBase518,
    _model_input_size,
    _normalize_depth,
    default_lab_cache_dir,
)
from .output import ensure_b1hw, ensure_bchw, match_depth


def default_distill_base_onnx_path(cache_dir: str | Path | None = None) -> Path:
    cache = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
    return cache / "models--lc700x--Distill-Any-Depth-Base-hf" / "model_fp16_294x518.onnx"


def _preprocess_distill_rgb(rgb: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    rgb = ensure_bchw(rgb, name="rgb").to(device).float().clamp(0, 1)
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
        mode="bicubic" if device.type == "cuda" else "bilinear",
        align_corners=False,
        antialias=True if device.type == "cuda" else False,
    ).to(dtype)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


class DistillAnyDepthBaseOnnxCuda:
    def __init__(
        self,
        *,
        device: str | torch.device = "cuda",
        cache_dir: str | Path | None = None,
        onnx_path: str | Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else default_lab_cache_dir()
        self.onnx_path = Path(onnx_path) if onnx_path is not None else default_distill_base_onnx_path(self.cache_dir)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.info = DepthProviderInfo(
            provider="onnxruntime.InferenceSession",
            model_name=DISTILL_ANY_DEPTH_BASE_NAME,
            model_id=DISTILL_ANY_DEPTH_BASE_MODEL_ID,
            depth_resolution=DISTILL_ANY_DEPTH_BASE_RESOLUTION,
            cache_dir=str(self.cache_dir),
            load_mode="local_onnx",
            depth_backend="onnx_cuda" if self.device.type == "cuda" else "onnx_cpu",
            runtime="onnxruntime",
            onnx_path=str(self.onnx_path),
        )
        self._session = None

    def load(self):
        if self._session is not None:
            return self._session
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"ONNX file not found: {self.onnx_path}")

        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(directory="")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        active = session.get_providers()
        if self.device.type == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(f"ONNX Runtime CUDA provider unavailable; active providers: {active}")
        self.info = replace(self.info, execution_provider=active[0] if active else None)
        self._session = session
        return session

    def predict(self, rgb: torch.Tensor) -> torch.Tensor:
        rgb = ensure_bchw(rgb, name="rgb")
        _, _, height, width = rgb.shape
        tensor = _preprocess_distill_rgb(rgb, device=self.device, dtype=self.dtype)
        ort_input = tensor.detach().cpu().numpy()

        session = self.load()
        predicted = session.run(["predicted_depth"], {"pixel_values": ort_input})[0]
        depth = torch.from_numpy(predicted).float()
        depth = ensure_b1hw(depth)
        depth = _normalize_depth(depth)
        return match_depth(depth, height, width)


def estimate_distill_any_depth_base_518_nvidia(
    rgb: torch.Tensor,
    *,
    device: str | torch.device = "cuda",
    cache_dir: str | Path | None = None,
    onnx_path: str | Path | None = None,
    prefer_onnx: bool = True,
    allow_pytorch_fallback: bool = True,
    local_files_only: bool = False,
    force_download: bool = False,
) -> tuple[torch.Tensor, DepthProviderInfo]:
    fallback_reason = None
    if prefer_onnx:
        try:
            provider = DistillAnyDepthBaseOnnxCuda(device=device, cache_dir=cache_dir, onnx_path=onnx_path)
            return provider.predict(rgb), provider.info
        except Exception as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
            if not allow_pytorch_fallback:
                raise

    provider = DistillAnyDepthBase518(
        device=device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        force_download=force_download,
    )
    depth = provider.predict(rgb)
    info = replace(
        provider.info,
        fallback_reason=fallback_reason,
        onnx_path=str(onnx_path or default_distill_base_onnx_path(cache_dir)),
    )
    return depth, info
