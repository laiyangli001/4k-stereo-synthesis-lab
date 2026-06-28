from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DepthResponseFn = Callable[[Any], Any]

PARALLAX_RESOLVER_VERSION = 1
DEPTH_RESPONSE_NAME = "linear_clamp_convergence_v1"

PARALLAX_BUDGET_TABLE: dict[str, dict[int, float]] = {
    "comfort": {720: 24.0, 1080: 32.0, 1440: 48.0, 2160: 64.0},
    "standard": {720: 36.0, 1080: 48.0, 1440: 64.0, 2160: 96.0},
    "strong": {720: 48.0, 1080: 64.0, 1440: 88.0, 2160: 128.0},
    "extreme": {720: 64.0, 1080: 80.0, 1440: 112.0, 2160: 160.0},
}

_STRENGTH_ALIASES = {
    "comfortable": "comfort",
    "soft": "comfort",
    "low": "comfort",
    "normal": "standard",
    "balanced": "standard",
    "default": "standard",
    "std": "standard",
    "high": "strong",
    "enhanced": "strong",
    "very_strong": "extreme",
    "max": "extreme",
    "maximum": "extreme",
}


@dataclass(frozen=True)
class ParallaxBudget:
    max_disparity_px: float
    depth_response: DepthResponseFn
    preset: str
    depth_response_name: str = DEPTH_RESPONSE_NAME
    resolver_version: int = PARALLAX_RESOLVER_VERSION


def resolve_parallax_budget(
    render_width: int,
    render_height: int,
    preset: str,
    depth_strength: float = 1.0,
    stereo_scale: float = 1.0,
    convergence: float = 0.0,
    ipd_mm: float | None = None,
    max_shift_ratio: float = 0.05,
    *,
    ipd: float = 0.064,
    max_disparity_px: float | None = None,
) -> ParallaxBudget:
    normalized_preset = _normalize_strength_preset(preset)
    width = max(1, int(render_width))
    height = max(1, int(render_height))

    if max_disparity_px is not None:
        resolved_max_disparity = max(0.0, float(max_disparity_px))
    elif normalized_preset == "legacy":
        resolved_max_disparity = _legacy_max_disparity_px(
            width=width,
            depth_strength=depth_strength,
            stereo_scale=stereo_scale,
            ipd=ipd,
            ipd_mm=ipd_mm,
            max_shift_ratio=max_shift_ratio,
        )
    else:
        resolved_max_disparity = _resolve_table_budget(width, height, normalized_preset)

    def depth_response(depth):
        return depth.clamp(0, 1) - float(convergence)

    return ParallaxBudget(
        max_disparity_px=float(resolved_max_disparity),
        depth_response=depth_response,
        preset=normalized_preset,
        depth_response_name=DEPTH_RESPONSE_NAME,
    )


def parallax_debug_info(budget: ParallaxBudget) -> dict[str, float | int | str]:
    return {
        "resolved_max_disparity_px": float(budget.max_disparity_px),
        "parallax_budget_preset": str(budget.preset),
        "depth_response": str(budget.depth_response_name),
        "parallax_resolver_version": int(budget.resolver_version),
    }


def _normalize_strength_preset(preset: str | None) -> str:
    key = str(preset or "standard").strip().lower().replace("-", "_").replace(" ", "_")
    if key == "legacy":
        return "legacy"
    key = _STRENGTH_ALIASES.get(key, key)
    if key in PARALLAX_BUDGET_TABLE:
        return key
    raise ValueError(f"unknown parallax strength preset: {preset!r}")


def _resolve_table_budget(width: int, height: int, preset: str) -> float:
    short_side = float(min(width, height))
    table = PARALLAX_BUDGET_TABLE[preset]
    levels = sorted(table)
    if short_side <= levels[0]:
        base = table[levels[0]] * short_side / float(levels[0])
    elif short_side >= levels[-1]:
        base = table[levels[-1]] * short_side / float(levels[-1])
    else:
        lower = levels[0]
        upper = levels[-1]
        for idx in range(len(levels) - 1):
            if levels[idx] <= short_side <= levels[idx + 1]:
                lower = levels[idx]
                upper = levels[idx + 1]
                break
        t = (short_side - lower) / float(upper - lower)
        base = table[lower] + (table[upper] - table[lower]) * t
    return float(base * _aspect_protection_factor(width, height))


def _aspect_protection_factor(width: int, height: int) -> float:
    short_side = max(1.0, float(min(width, height)))
    aspect = max(float(width), float(height)) / short_side
    if aspect <= 2.0:
        return 1.0
    return max(0.70, min(1.0, 2.0 / aspect))


def _legacy_max_disparity_px(
    *,
    width: int,
    depth_strength: float,
    stereo_scale: float,
    ipd: float,
    ipd_mm: float | None,
    max_shift_ratio: float,
) -> float:
    if ipd_mm is None:
        effective_ipd_m = max(0.0, float(ipd))
    else:
        effective_ipd_m = max(0.0, float(ipd_mm)) / 1000.0 * max(0.0, float(stereo_scale))
    legacy_eye_shift_px = float(width) * effective_ipd_m * max(0.0, float(max_shift_ratio)) * max(
        0.0,
        float(depth_strength),
    )
    return 2.0 * legacy_eye_shift_px
