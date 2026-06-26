from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RenderSizePolicy(Enum):
    NATIVE = "native"
    SCALED = "scaled"
    FIXED = "fixed"
    DYNAMIC = "dynamic"


_RENDER_SIZE_POLICY_ALIASES = {
    "native": RenderSizePolicy.NATIVE,
    "scaled": RenderSizePolicy.SCALED,
    "fixed": RenderSizePolicy.FIXED,
    "dynamic": RenderSizePolicy.DYNAMIC,
}


@dataclass(frozen=True)
class RenderSizeConfig:
    policy: RenderSizePolicy = RenderSizePolicy.NATIVE
    scale_factor: float = 1.0
    fixed_width: int = 1920
    fixed_height: int = 1080
    max_pixels: int = 3840 * 2160
    min_dimension: int = 480
    align: int = 16


def render_size_config_from_settings(settings: dict | None) -> RenderSizeConfig:
    settings = settings or {}
    return RenderSizeConfig(
        policy=_normalize_policy(settings.get("Render Size Policy", RenderSizePolicy.NATIVE.value)),
        scale_factor=_float_setting(settings, "Render Scale", 1.0),
        fixed_width=_int_setting(settings, "Render Fixed Width", 1920),
        fixed_height=_int_setting(settings, "Render Fixed Height", 1080),
        max_pixels=_int_setting(settings, "Render Max Pixels", 3840 * 2160),
        min_dimension=_int_setting(settings, "Render Min Dimension", 480),
        align=_int_setting(settings, "Render Align", 16),
    )


def resolve_render_size(
    capture_size: tuple[int, int],
    config: RenderSizeConfig | None = None,
) -> tuple[int, int]:
    """Resolve runtime render size from a capture size and policy."""
    config = config or RenderSizeConfig()
    capture_width, capture_height = _valid_size(capture_size, name="capture_size")
    align = max(1, int(config.align))

    if config.policy is RenderSizePolicy.NATIVE:
        return _align_size(capture_width, capture_height, align)

    if config.policy is RenderSizePolicy.SCALED:
        scale = max(0.01, float(config.scale_factor))
        return _align_size(capture_width * scale, capture_height * scale, align)

    if config.policy is RenderSizePolicy.FIXED:
        return _align_size(config.fixed_width, config.fixed_height, align)

    if config.policy is RenderSizePolicy.DYNAMIC:
        return _resolve_dynamic_size(capture_width, capture_height, config, align)

    raise ValueError(f"unknown render size policy: {config.policy!r}")


def runtime_output_size_text(size: tuple[int, int] | None) -> str:
    if size is None:
        return "unknown"
    width, height = _valid_size(size, name="size")
    return f"{width}x{height}"


def _normalize_policy(value) -> RenderSizePolicy:
    if isinstance(value, RenderSizePolicy):
        return value
    key = str(value or "native").strip().lower().replace(" ", "_").replace("-", "_")
    return _RENDER_SIZE_POLICY_ALIASES.get(key, RenderSizePolicy.NATIVE)


def _int_setting(settings: dict, key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _float_setting(settings: dict, key: str, default: float) -> float:
    try:
        return float(settings.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _resolve_dynamic_size(width: int, height: int, config: RenderSizeConfig, align: int) -> tuple[int, int]:
    max_pixels = max(1, int(config.max_pixels))
    min_dimension = max(1, int(config.min_dimension))
    pixels = width * height
    scale = 1.0
    if pixels > max_pixels:
        scale = (max_pixels / float(pixels)) ** 0.5
    scaled_width = width * scale
    scaled_height = height * scale
    short_side = min(scaled_width, scaled_height)
    if short_side < min_dimension:
        boost = min_dimension / max(short_side, 1.0)
        scaled_width *= boost
        scaled_height *= boost
    return _align_size(scaled_width, scaled_height, align)


def _align_size(width: float, height: float, align: int) -> tuple[int, int]:
    return _align_dimension(width, align), _align_dimension(height, align)


def _align_dimension(value: float, align: int) -> int:
    value = max(1, int(round(float(value))))
    if align <= 1:
        return value
    return max(align, (value // align) * align)


def _valid_size(size: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(size) != 2:
        raise ValueError(f"{name} must be a (width, height) tuple")
    width = int(size[0])
    height = int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return width, height
