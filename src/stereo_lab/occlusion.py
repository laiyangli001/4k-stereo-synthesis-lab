from __future__ import annotations

import os

import torch
import torch.nn.functional as F

from .layers import depth_edges
from .output import ensure_b1hw


def dilate_mask(mask: torch.Tensor, radius: int = 2) -> torch.Tensor:
    mask = ensure_b1hw(mask).float()
    if radius <= 0:
        return mask
    k = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=k, stride=1, padding=radius)


def make_occlusion_mask(
    depth: torch.Tensor,
    shift_px: torch.Tensor,
    edge_threshold: float = 0.04,
    dilation: int = 2,
    fused: bool = True,
) -> torch.Tensor:
    depth = ensure_b1hw(depth).float()
    shift_px = ensure_b1hw(shift_px).float()
    if occlusion_backend(depth, shift_px, edge_threshold=edge_threshold, dilation=dilation, fused=fused) == "triton_occlusion_radius2":
        from .occlusion_triton import make_occlusion_mask_radius2

        return make_occlusion_mask_radius2(depth, shift_px)
    edge_mask = depth_edges(depth, threshold=edge_threshold)
    shift_grad = depth_edges(shift_px.abs() / shift_px.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6), threshold=0.05)
    return dilate_mask(torch.maximum(edge_mask, shift_grad), radius=dilation).clamp(0, 1)


def occlusion_backend(
    depth: torch.Tensor,
    shift_px: torch.Tensor,
    *,
    edge_threshold: float,
    dilation: int,
    fused: bool = True,
) -> str:
    if not fused or _triton_disabled_by_env():
        return "torch_max_pool"
    try:
        from .occlusion_triton import can_use_triton_occlusion_radius2
    except Exception:
        return "torch_max_pool"
    if can_use_triton_occlusion_radius2(depth, shift_px, edge_threshold=edge_threshold, dilation=dilation):
        return "triton_occlusion_radius2"
    return "torch_max_pool"


def _triton_disabled_by_env() -> bool:
    return os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
