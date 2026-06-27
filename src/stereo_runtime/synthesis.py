from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from .baseline_shift import ShiftParams, compute_shift_px, shift_debug_info, synthesize_baseline, warp_horizontal
from .depth_postprocess import postprocess_depth
from .hole_fill import (
    directional_edge_aware_fill,
    directional_edge_aware_fill_backend,
    edge_aware_fill,
    edge_aware_fill_backend,
)
from .layers import composite_layers, make_depth_layers
from .occlusion import make_occlusion_mask, occlusion_backend
from .output import AnaglyphMethod, OutputFormat, ensure_bchw, make_sbs, match_depth, sbs_backend
from .refine import refine_local
from .temporal import TemporalState, apply_temporal, detect_scene_change

Backend = Literal["fast", "fast_plus", "quality_4k", "hq_4k"]
HoleFill = Literal["none", "fast", "edge_aware"]
HoleFillMode = Literal["balanced", "soft_low_ghost", "sharp_test", "quality", "content_aware", "directional"]


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
    ipd_mm: float | None = 32.0
    stereo_scale: float = 0.4
    max_disparity_px: float | None = None
    parallax_preset: str = "legacy"
    temporal_strength: float = 0.85
    auto_reset_temporal: bool = False
    scene_reset_threshold: float = 0.22
    reset_cooldown_frames: int = 3
    foreground_scale: float = 0.0
    depth_antialias_strength: float = 0.0
    edge_dilation: int = 2
    edge_threshold: float = 0.04
    mask_feather_radius: int = 3
    hole_fill_mode: HoleFillMode = "balanced"
    hole_fill_radius: int = 3
    hole_fill_strength: float = 1.0
    screen_edge_mask_suppression: int = 0
    cross_eyed: bool = False
    anaglyph_method: AnaglyphMethod = "red_cyan"
    refine: bool = False
    fused: bool = True


@dataclass
class StereoResult:
    left_eye: torch.Tensor
    right_eye: torch.Tensor
    sbs: torch.Tensor
    debug_info: dict[str, torch.Tensor | float | int | str] = field(default_factory=dict)


def _layered_synthesis(rgb: torch.Tensor, depth: torch.Tensor, config: StereoConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    stage_times: dict[str, float] = {}
    stage_start = time.perf_counter()
    params = ShiftParams(
        depth_strength=config.depth_strength,
        convergence=config.convergence,
        ipd=config.ipd,
        max_shift_ratio=config.max_shift_ratio,
        ipd_mm=config.ipd_mm,
        stereo_scale=config.stereo_scale,
        max_disparity_px=config.max_disparity_px,
        parallax_preset=config.parallax_preset,
    )
    rgb = ensure_bchw(rgb, name="rgb").float()
    depth = postprocess_depth(
        match_depth(depth, rgb.shape[-2], rgb.shape[-1]),
        foreground_scale=config.foreground_scale,
        antialias_strength=config.depth_antialias_strength,
    )
    base_shift = compute_shift_px(depth, rgb.shape[-1], params)
    parallax_debug = shift_debug_info(depth, rgb.shape[-1], params)
    stage_times["depth_postprocess_shift_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()

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
            left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=1.0))
            sign = -1.0 if config.symmetric else -0.9
            right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=sign))

        left = composite_layers(left_layers, weights)
        right = composite_layers(right_layers, weights)
    stage_times["warp_composite_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()
    if config.occlusion:
        occlusion_mask_backend = occlusion_backend(
            depth,
            base_shift,
            edge_threshold=config.edge_threshold,
            dilation=config.edge_dilation,
            fused=config.fused,
        )
        mask = make_occlusion_mask(
            depth,
            base_shift,
            edge_threshold=config.edge_threshold,
            dilation=config.edge_dilation,
            fused=config.fused,
            screen_edge_suppression=config.screen_edge_mask_suppression,
        )
    else:
        occlusion_mask_backend = "none"
        mask = torch.zeros_like(depth)

    stage_times["occlusion_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()
    hole_fill_backend = "none"
    if config.hole_fill != "none":
        radius = int(config.hole_fill_radius)
        strength = float(config.hole_fill_strength)
        if config.hole_fill == "fast" and config.hole_fill_mode == "balanced":
            radius = 2
            strength = 0.65
        eyes = torch.cat([left, right], dim=0)
        fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
        use_directional_fill = str(config.hole_fill_mode).strip().lower() in {
            "quality",
            "content_aware",
            "directional",
        }
        if use_directional_fill:
            hole_fill_backend = directional_edge_aware_fill_backend()
            eyes = directional_edge_aware_fill(
                eyes,
                fill_mask,
                depth=depth,
                shift_px=base_shift,
                radius=radius,
                strength=strength,
                mask_feather_radius=config.mask_feather_radius,
                depth_edge_threshold=config.edge_threshold,
            )
        else:
            hole_fill_backend = edge_aware_fill_backend(
                eyes,
                fill_mask,
                radius=radius,
                strength=strength,
                fused=config.fused,
            )
            eyes = edge_aware_fill(
                eyes,
                fill_mask,
                radius=radius,
                strength=strength,
                fused=config.fused,
                mask_feather_radius=config.mask_feather_radius,
            )
        left, right = eyes.chunk(2, dim=0)

    stage_times["hole_fill_ms"] = (time.perf_counter() - stage_start) * 1000.0
    stage_start = time.perf_counter()
    left = refine_local(left, mask, enabled=config.refine)
    right = refine_local(right, mask, enabled=config.refine)
    stage_times["refine_ms"] = (time.perf_counter() - stage_start) * 1000.0
    return left, right, mask, {
        "layers": layer_count,
        "shift_px": base_shift,
        "occlusion_mask": mask,
        "warp_composite_backend": warp_composite_backend,
        "occlusion_mask_backend": occlusion_mask_backend,
        "hole_fill_backend": hole_fill_backend,
        "mask_feather_radius": int(config.mask_feather_radius),
        "hole_fill_mode": str(config.hole_fill_mode),
        "hole_fill_radius": int(radius) if config.hole_fill != "none" else 0,
        "hole_fill_strength": float(strength) if config.hole_fill != "none" else 0.0,
        **parallax_debug,
        **stage_times,
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
    return (
        os.environ.get("STEREO_RUNTIME_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
        or os.environ.get("STEREO_LAB_DISABLE_TRITON", "").lower() in {"1", "true", "yes", "on"}
    )


def synthesize_stereo(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    config: StereoConfig | None = None,
    temporal_state: TemporalState | None = None,
) -> StereoResult:
    config = config or StereoConfig()
    temporal_reset = False
    if config.temporal and config.auto_reset_temporal and temporal_state is not None:
        temporal_reset = detect_scene_change(
            rgb,
            temporal_state,
            threshold=config.scene_reset_threshold,
            cooldown_frames=config.reset_cooldown_frames,
        )
        if temporal_reset:
            temporal_state.reset_stereo()
    if config.backend in {"fast", "fast_plus"}:
        params = ShiftParams(
            depth_strength=config.depth_strength,
            convergence=config.convergence,
            ipd=config.ipd,
            max_shift_ratio=config.max_shift_ratio,
            ipd_mm=config.ipd_mm,
            stereo_scale=config.stereo_scale,
            max_disparity_px=config.max_disparity_px,
            parallax_preset=config.parallax_preset,
        )
        depth = postprocess_depth(
            depth,
            foreground_scale=config.foreground_scale,
            antialias_strength=config.depth_antialias_strength,
        )
        left, right, shift_px = synthesize_baseline(rgb, depth, params)
        if config.backend == "fast_plus":
            depth_for_mask = match_depth(depth, left.shape[-2], left.shape[-1])
            mask = make_occlusion_mask(
                depth_for_mask,
                shift_px,
                edge_threshold=0.03,
                dilation=1,
                fused=config.fused,
                screen_edge_suppression=config.screen_edge_mask_suppression,
            )
            eyes = torch.cat([left, right], dim=0)
            fill_mask = mask.expand(eyes.shape[0], -1, -1, -1)
            hole_fill_backend = directional_edge_aware_fill_backend()
            eyes = directional_edge_aware_fill(
                eyes,
                fill_mask,
                depth=depth_for_mask,
                shift_px=shift_px,
                radius=1,
                strength=0.60,
                mask_feather_radius=config.mask_feather_radius,
                depth_edge_threshold=0.03,
            )
            left, right = eyes.chunk(2, dim=0)
            debug = {
                "backend": config.backend,
                "shift_px": shift_px,
                "occlusion_mask": mask,
                "occlusion_mask_backend": occlusion_backend(
                    depth_for_mask,
                    shift_px,
                    edge_threshold=0.03,
                    dilation=1,
                    fused=config.fused,
                ),
                "hole_fill_backend": hole_fill_backend,
                "fast_plus_edge_threshold": 0.03,
                "fast_plus_edge_dilation": 1,
                "fast_plus_hole_fill_radius": 1,
                "fast_plus_hole_fill_strength": 0.60,
                "mask_feather_radius": int(config.mask_feather_radius),
                **shift_debug_info(depth, left.shape[-1], params),
            }
        else:
            mask = None
            debug = {"backend": config.backend, "shift_px": shift_px, **shift_debug_info(depth, left.shape[-1], params)}
    else:
        if config.backend == "hq_4k" and config.layers < 3:
            config = StereoConfig(**{**config.__dict__, "layers": 3})
        left, right, mask, debug = _layered_synthesis(rgb, depth, config)
        debug["backend"] = config.backend

    if config.temporal:
        left, right = apply_temporal(left, right, mask, temporal_state, strength=config.temporal_strength)

    output_depth = postprocess_depth(
        match_depth(depth, left.shape[-2], left.shape[-1]),
        foreground_scale=config.foreground_scale,
        antialias_strength=config.depth_antialias_strength,
    )
    if config.cross_eyed:
        left, right = right, left
    if config.debug_output:
        debug["output_depth"] = output_depth
        debug["temporal_reset"] = int(temporal_reset)
        if temporal_state is not None:
            debug["scene_delta"] = float(temporal_state.last_scene_delta)
            debug["temporal_reset_count"] = int(temporal_state.reset_count)
    if temporal_reset:
        debug["temporal_reset_reason"] = "scene_reset"
    debug["cross_eyed"] = int(config.cross_eyed)
    debug["anaglyph_method"] = config.anaglyph_method
    debug["convergence"] = float(config.convergence)
    debug["temporal_enabled"] = int(bool(config.temporal))
    debug["temporal_strength"] = float(config.temporal_strength)
    debug["hole_fill_mode"] = str(config.hole_fill_mode)
    debug["hole_fill_radius"] = int(config.hole_fill_radius)
    debug["hole_fill_strength"] = float(config.hole_fill_strength)
    debug["edge_threshold"] = float(config.edge_threshold)
    debug["edge_dilation"] = int(config.edge_dilation)
    debug["mask_feather_radius"] = int(config.mask_feather_radius)
    debug["sbs_backend"] = sbs_backend(
        left,
        right,
        config.output_format,
        fused=config.fused,
        depth=output_depth,
        anaglyph_method=config.anaglyph_method,
    )
    sbs = make_sbs(
        left,
        right,
        config.output_format,
        fused=config.fused,
        depth=output_depth,
        anaglyph_method=config.anaglyph_method,
    )
    if not config.debug_output:
        debug = {k: v for k, v in debug.items() if isinstance(v, (float, int, str))}
    return StereoResult(left_eye=left, right_eye=right, sbs=sbs, debug_info=debug)
