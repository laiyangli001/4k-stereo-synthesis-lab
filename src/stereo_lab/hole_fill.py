from __future__ import annotations

import torch
import torch.nn.functional as F

from .output import ensure_bchw, ensure_b1hw

_BOX_KERNEL_CACHE: dict[tuple[int, int, str, torch.dtype, str], torch.Tensor] = {}


def _box_kernel(channels: int, k: int, device: torch.device, dtype: torch.dtype, axis: str) -> torch.Tensor:
    key = (channels, k, str(device), dtype, axis)
    weight = _BOX_KERNEL_CACHE.get(key)
    if weight is not None:
        return weight

    if axis == "horizontal":
        shape = (channels, 1, 1, k)
    elif axis == "vertical":
        shape = (channels, 1, k, 1)
    else:
        raise ValueError(f"unknown box blur axis: {axis}")

    weight = torch.ones(shape, device=device, dtype=dtype) / float(k)
    _BOX_KERNEL_CACHE[key] = weight
    return weight


def box_blur(x: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return x
    k = radius * 2 + 1
    channels = x.shape[1]
    horizontal = _box_kernel(channels, k, x.device, x.dtype, "horizontal")
    vertical = _box_kernel(channels, k, x.device, x.dtype, "vertical")
    x = F.conv2d(x, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(x, vertical, padding=(radius, 0), groups=channels)


def edge_aware_fill(image: torch.Tensor, mask: torch.Tensor, radius: int = 3, strength: float = 1.0) -> torch.Tensor:
    image = ensure_bchw(image, name="image").float()
    mask = ensure_b1hw(mask).to(device=image.device, dtype=image.dtype).clamp(0, 1)
    if mask.shape[-2:] != image.shape[-2:]:
        mask = F.interpolate(mask, size=image.shape[-2:], mode="bilinear", align_corners=False)
    blurred = box_blur(image, radius=radius)
    blend = (mask * strength).clamp(0, 1)
    return image * (1.0 - blend) + blurred * blend
