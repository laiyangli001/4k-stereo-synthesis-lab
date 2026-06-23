from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .output import ensure_bchw, match_depth


@dataclass(frozen=True)
class ShiftParams:
    depth_strength: float = 2.0
    convergence: float = 0.0
    ipd: float = 0.064
    max_shift_ratio: float = 0.05
    ipd_mm: float | None = 64.0
    stereo_scale: float = 0.5


_GRID_CACHE: dict[tuple[int, int, int, str, torch.dtype], torch.Tensor] = {}
_GRID_COMPONENT_CACHE: dict[tuple[int, int, str, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}


def make_base_grid(batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (batch, height, width, str(device), dtype)
    cached = _GRID_CACHE.get(key)
    if cached is not None:
        return cached
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)
    grid = grid.unsqueeze(0).expand(batch, height, width, 2)
    _GRID_CACHE[key] = grid
    return grid


def make_base_grid_components(height: int, width: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    key = (height, width, str(device), dtype)
    cached = _GRID_COMPONENT_CACHE.get(key)
    if cached is not None:
        return cached
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    _GRID_COMPONENT_CACHE[key] = (xx, yy)
    return xx, yy


def _effective_ipd_m(params: ShiftParams) -> float:
    if params.ipd_mm is None:
        return max(0.0, float(params.ipd))
    return max(0.0, float(params.ipd_mm)) / 1000.0 * max(0.0, float(params.stereo_scale))


def compute_shift_px(depth: torch.Tensor, width: int, params: ShiftParams) -> torch.Tensor:
    depth = depth.clamp(0, 1)
    centered = depth - params.convergence
    max_px = width * _effective_ipd_m(params) * params.max_shift_ratio
    return -centered * params.depth_strength * max_px


def warp_horizontal(rgb: torch.Tensor, shift_px: torch.Tensor, eye_sign: float) -> torch.Tensor:
    rgb = ensure_bchw(rgb, name="rgb").float()
    b, _, h, w = rgb.shape
    shift_px = match_depth(shift_px, h, w)
    xx, yy = make_base_grid_components(h, w, rgb.device, rgb.dtype)
    shift_norm = (2.0 * shift_px.squeeze(1) / max(w - 1, 1)) * eye_sign
    grid_x = xx.unsqueeze(0) + shift_norm
    grid_y = yy.expand(b, h, w)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(rgb, grid, mode="bilinear", padding_mode="reflection", align_corners=True)


def synthesize_baseline(rgb: torch.Tensor, depth: torch.Tensor, params: ShiftParams) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = match_depth(depth, rgb.shape[-2], rgb.shape[-1])
    shift_px = compute_shift_px(depth, rgb.shape[-1], params)
    left = warp_horizontal(rgb, shift_px, eye_sign=1.0)
    right = warp_horizontal(rgb, shift_px, eye_sign=-1.0)
    return left, right, shift_px
