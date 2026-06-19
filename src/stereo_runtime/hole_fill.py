from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .output import ensure_bchw, ensure_b1hw


def box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=radius, count_include_pad=True)


def edge_aware_fill(image: torch.Tensor, mask: torch.Tensor, radius: int = 3, strength: float = 1.0, fused: bool = True) -> torch.Tensor:
    image = ensure_bchw(image, name="image").float()
    mask = ensure_b1hw(mask).to(device=image.device, dtype=image.dtype).clamp(0, 1)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    backend = edge_aware_fill_backend(image, mask, radius=radius, strength=strength, fused=fused)
    if backend == "triton_radius1":
        from .hole_fill_triton import edge_aware_fill_radius1_strength060

        return edge_aware_fill_radius1_strength060(image.contiguous(), mask.contiguous())
    if backend == "triton_radius3":
        from .hole_fill_triton import edge_aware_fill_radius3

        return edge_aware_fill_radius3(image.contiguous(), mask.contiguous())
    blurred = box_blur(image, radius=radius)
    blend = (mask * strength).clamp(0, 1)
    return torch.lerp(image, blurred, blend)


def edge_aware_fill_backend(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float, fused: bool = True) -> str:
    if not fused or _triton_disabled_by_env():
        return "torch_avg_pool"
    if _can_use_triton_fill_radius1(image, mask, radius=radius, strength=strength):
        return "triton_radius1"
    if _can_use_triton_fill_radius3(image, mask, radius=radius, strength=strength):
        return "triton_radius3"
    return "torch_avg_pool"


def _can_use_triton_fill_radius1(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    try:
        from .hole_fill_triton import can_use_triton_radius1
    except Exception:
        return False
    return can_use_triton_radius1(image, mask, radius=radius, strength=strength)


def _can_use_triton_fill_radius3(image: torch.Tensor, mask: torch.Tensor, *, radius: int, strength: float) -> bool:
    try:
        from .hole_fill_triton import can_use_triton_radius3
    except Exception:
        return False
    return can_use_triton_radius3(image, mask, radius=radius, strength=strength)


def _triton_disabled_by_env() -> bool:
    return (
        os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
    )