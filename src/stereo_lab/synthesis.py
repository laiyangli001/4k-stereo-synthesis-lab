from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

import torch

from .baseline_shift import ShiftParams, compute_shift_px, synthesize_baseline, warp_horizontal
from .hole_fill import edge_aware_fill, edge_aware_fill_backend
from .layers import composite_layers, make_depth_layers
from .occlusion import make_occlusion_mask, occlusion_backend
from .output import OutputFormat, ensure_bchw, make_sbs, match_depth
from .refine import refine_local
from .temporal import TemporalState, apply_temporal

Backend = Literal["fast", "quality_4k", "hq_4k"]
HoleFill = Literal["none", "fast", "edge_aware"]


@dataclass
class StereoConfig:
    backend: Backend = "quality_4k"
    layers: int = 2
    occlusion: bool = True
    symmetric: bool = True
    hole_fill: HoleFill = "edge_aware"
    temporal: bool = True
    output_format: OutputFormat = "half_sbs"
    debug_output: bool = False
    depth_strength: float = 2.0
    convergence: float = 0.0
    ipd: float = 0.064
    max_shift_ratio: float = 0.05
    temporal_strength: float = 0.85
    refine: bool = False
    fused: bool = True


@dataclass
class StereoResult:
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    sbs: torch.Tensor
    debug_info: dict[str, torch.Tensor | float | int | str] = field(default_factory=dict)


def _layered_synthesis(rgb: torch.Tensor, depth: torch.Tensor, config: StereoConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    params = ShiftParams(
        depth_strength=config.depth_strength,
        convergence=config.convergence,
        ipd=config.ipd,
        max_shift_ratio=config.max_shift_ratio,
    )
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = match_depth(depth, rgb.shape[-2], rgb.shape[-1])
    base_shift = compute_shift_px(depth, rgb.shape[-1], params)

    layer_count = max(1, int(config.layers))
    fused = _try_fused_warp_composite2(
        rgb,
        depth,
        base_shift,
        layers=layer_count,
        symmetric=config.symmetric,
        enabled=config.fused,
    )
    warp_composite_backend = "triton_warp_composite2" if fused is not None else "torch_grid_sample"
    if fused is not None:
        left, right = fused
    else:
        weights = make_depth_layers(depth, layers=layer_count)
        left_layers: list[torch.Tensor] = []
        right_layers: list[torch.Tensor] = []
        for idx in range(layer_count):
            layer_shift = base_shift * (0.75 + 0.25 * (idx + 1) / layer_count)
            left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=-1.0))
            sign = 1.0 if config.symmetric else 0.9
            right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=sign))

        left = composite_layers(left_layers, weights)
        right = composite_layers(right_layers, weights)
    if config.occlusion:
        occlusion_mask_backend = occlusion_backend(depth, base_shift, edge_threshold=0.04, dilation=2, fused=config.fused)
        mask = make_occlusion_mask(depth, base_shift, fused=config.fused)
    else:
        occlusion_mask_backend = "none"
        mask = torch.zeros_like(depth)

    hole_fill_backend = "none"
    if config.hole_fill != "none":
        radius = 2 if config.hole_fill == "fast" else 3
        strength = 0.65 if config.hole_fill == "fast" else 1.0
        eyes = torch.cat([left, right], dim=0)
        fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
        hole_fill_backend = edge_aware_fill_backend(eyes, fill_mask, radius=radius, strength=strength, fused=config.fused)
        eyes = edge_aware_fill(eyes, fill_mask, radius=radius, strength=strength, fused=config.fused)
        left, right = eyes.chunk(2, dim=0)

    left = refine_local(left, mask, enabled=config.refine)
    right = refine_local(right, mask, enabled=config.refine)
    return left, right, mask, {
        "layers": layer_count,
        "shift_px": base_shift,
        "occlusion_mask": mask,
        "warp_composite_backend": warp_composite_backend,
        "occlusion_mask_backend": occlusion_mask_backend,
        "hole_fill_backend": hole_fill_backend,
    }


def _try_fused_warp_composite2(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    base_shift: torch.Tensor,
    *,
    layers: int,
    symmetric: bool,
    enabled: bool = True,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not enabled or _triton_disabled_by_env():
        return None
    try:
        from .warp_composite_triton import can_use_triton_warp_composite2, warp_composite2
    except Exception:
        return None
    if not can_use_triton_warp_composite2(rgb, depth, base_shift, layers=layers, symmetric=symmetric):
        return None
    return warp_composite2(rgb, depth, base_shift)


def _triton_disabled_by_env() -> bool:
    return os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}


def synthesize_stereo(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: StereoConfig | None = None,
    temporal_state: TemporalState | None = None,
) -> StereoResult:
    config = config or StereoConfig()
    if config.backend == "fast":
        params = ShiftParams(
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            ipd=config.ipd,
            max_shift_ratio=config.max_shift_ratio,
        )
        left, right, shift_px = synthesize_baseline(rgb, depth, params)
        mask = None
        debug = {"backend": config.backend, "shift_px": shift_px}
    else:
        if config.backend == "hq_4k" and config.layers < 3:
            config = StereoConfig(**{**config.__dict__, "layers": 3})
        left, right, mask, debug = _layered_synthesis(rgb, depth, config)
        debug["backend"] = config.backend

    if config.temporal:
        left, right = apply_temporal(left, right, mask, temporal_state, strength=config.temporal_strength)

    sbs = make_sbs(left, right, config.output_format)
    if not config.debug_output:
        debug = {k: v for k, v in debug.items() if isinstance(v, (float, int, str))}
    return StereoResult(left_eye=left, right_eye=right, sbs=sbs, debug_info=debug)
