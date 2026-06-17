from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from .openxr_render import OpenXRRenderConfig
from .output import OutputFormat
from .synthesis import StereoConfig

StereoModePreset = Literal["auto", "cinema", "game_low_latency", "still_image_hq", "debug_export"]

PRESET_CHOICES: tuple[StereoModePreset, ...] = (
    "auto",
    "cinema",
    "game_low_latency",
    "still_image_hq",
    "debug_export",
)

_PRESET_ALIASES: dict[str, StereoModePreset] = {
    "auto": "auto",
    "cinema": "cinema",
    "movie": "cinema",
    "film": "cinema",
    "game": "game_low_latency",
    "game_low_latency": "game_low_latency",
    "game / low latency": "game_low_latency",
    "low_latency": "game_low_latency",
    "still": "still_image_hq",
    "still_image": "still_image_hq",
    "still_image_hq": "still_image_hq",
    "still image / hq": "still_image_hq",
    "hq": "still_image_hq",
    "debug": "debug_export",
    "export": "debug_export",
    "debug_export": "debug_export",
    "debug / export": "debug_export",
}


@dataclass(frozen=True)
class AutoModeSignals:
    frame_motion_score: float = 0.0
    scene_cut_score: float = 0.0
    still_duration_s: float = 0.0
    foreground_process: str = ""
    fullscreen: bool = False
    openxr_active: bool = False
    user_export_action: bool = False
    latency_pressure: float = 0.0
    target_fps: float = 60.0


@dataclass(frozen=True)
class AutoModeDecision:
    preset: StereoModePreset
    reason: str
    hold_seconds: float = 3.0
    blend_seconds: float = 0.35
    require_consecutive_frames: int = 8


def normalize_preset(preset: str | StereoModePreset) -> StereoModePreset:
    key = str(preset).strip().lower().replace("-", "_")
    try:
        return _PRESET_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unknown stereo mode preset: {preset!r}") from exc


def stereo_config_for_preset(
    preset: str | StereoModePreset,
    *,
    output_format: OutputFormat = "half_sbs",
    overrides: dict[str, Any] | None = None,
) -> StereoConfig:
    normalized = normalize_preset(preset)
    if normalized == "auto":
        normalized = "cinema"

    config = _STEREO_PRESETS[normalized]
    config = replace(config, output_format=output_format)
    if overrides:
        config = _replace_checked(config, overrides)
    return config


def openxr_config_for_preset(
    preset: str | StereoModePreset,
    *,
    screen_roll: float = 0.0,
    overrides: dict[str, Any] | None = None,
) -> OpenXRRenderConfig:
    normalized = normalize_preset(preset)
    if normalized == "auto":
        normalized = "cinema"

    config = _OPENXR_PRESETS[normalized]
    config = replace(config, screen_roll=screen_roll)
    if overrides:
        config = _replace_checked(config, overrides)
    return config


def classify_auto_mode(signals: AutoModeSignals) -> AutoModeDecision:
    process = signals.foreground_process.lower()
    if signals.user_export_action:
        return AutoModeDecision("debug_export", "user export/debug action", hold_seconds=2.0, blend_seconds=0.2, require_consecutive_frames=1)

    if signals.still_duration_s >= 1.5 and signals.frame_motion_score <= 0.03:
        return AutoModeDecision("still_image_hq", "stable still frame", hold_seconds=3.0, blend_seconds=0.5, require_consecutive_frames=12)

    game_like_process = any(token in process for token in ("game", "unreal", "unity", "steam", "dx", "vulkan"))
    fast_motion = signals.frame_motion_score >= 0.35 or signals.scene_cut_score >= 0.35
    latency_pressure = signals.latency_pressure >= 0.65 or signals.target_fps >= 90.0
    if game_like_process or (signals.fullscreen and latency_pressure) or fast_motion:
        return AutoModeDecision("game_low_latency", "fast motion or latency pressure", hold_seconds=2.0, blend_seconds=0.2, require_consecutive_frames=4)

    if signals.openxr_active:
        return AutoModeDecision("cinema", "openxr active conservative cinema defaults", hold_seconds=3.0, blend_seconds=0.35, require_consecutive_frames=8)

    return AutoModeDecision("cinema", "default stable video mode", hold_seconds=3.0, blend_seconds=0.35, require_consecutive_frames=8)


def stereo_config_for_auto_mode(
    signals: AutoModeSignals,
    *,
    output_format: OutputFormat = "half_sbs",
    overrides: dict[str, Any] | None = None,
) -> tuple[AutoModeDecision, StereoConfig]:
    decision = classify_auto_mode(signals)
    return decision, stereo_config_for_preset(decision.preset, output_format=output_format, overrides=overrides)


def openxr_config_for_auto_mode(
    signals: AutoModeSignals,
    *,
    screen_roll: float = 0.0,
    overrides: dict[str, Any] | None = None,
) -> tuple[AutoModeDecision, OpenXRRenderConfig]:
    decision = classify_auto_mode(signals)
    return decision, openxr_config_for_preset(decision.preset, screen_roll=screen_roll, overrides=overrides)


def preset_summary() -> dict[str, dict[str, Any]]:
    return {
        name: {
            "stereo": asdict(_STEREO_PRESETS[name]),
            "openxr": asdict(_OPENXR_PRESETS[name]),
        }
        for name in ("cinema", "game_low_latency", "still_image_hq", "debug_export")
    }


def _replace_checked(config: StereoConfig | OpenXRRenderConfig, overrides: dict[str, Any]):
    allowed = {field.name for field in fields(config)}
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"unknown config override fields: {unknown}")
    return replace(config, **overrides)


_STEREO_PRESETS: dict[StereoModePreset, StereoConfig] = {
    "auto": StereoConfig(),
    "cinema": StereoConfig(
        backend="quality_4k",
        layers=2,
        temporal=True,
        temporal_strength=0.80,
        auto_reset_temporal=True,
        scene_reset_threshold=0.22,
        reset_cooldown_frames=3,
        depth_strength=2.0,
        edge_dilation=2,
        edge_threshold=0.04,
        depth_antialias_strength=0.4,
        foreground_scale=0.0,
        hole_fill="edge_aware",
        fused=True,
    ),
    "game_low_latency": StereoConfig(
        backend="quality_4k",
        layers=2,
        temporal=True,
        temporal_strength=0.60,
        auto_reset_temporal=True,
        scene_reset_threshold=0.18,
        reset_cooldown_frames=2,
        depth_strength=1.6,
        edge_dilation=1,
        edge_threshold=0.04,
        depth_antialias_strength=0.0,
        foreground_scale=0.0,
        hole_fill="fast",
        fused=True,
    ),
    "still_image_hq": StereoConfig(
        backend="hq_4k",
        layers=3,
        temporal=False,
        auto_reset_temporal=False,
        temporal_strength=0.0,
        depth_strength=2.4,
        edge_dilation=3,
        edge_threshold=0.04,
        depth_antialias_strength=0.6,
        foreground_scale=0.2,
        hole_fill="edge_aware",
        fused=True,
    ),
    "debug_export": StereoConfig(
        backend="quality_4k",
        layers=2,
        temporal=True,
        temporal_strength=0.80,
        auto_reset_temporal=True,
        scene_reset_threshold=0.22,
        reset_cooldown_frames=3,
        depth_strength=2.4,
        edge_dilation=2,
        edge_threshold=0.04,
        depth_antialias_strength=0.0,
        foreground_scale=0.0,
        hole_fill="edge_aware",
        debug_output=True,
        fused=True,
    ),
}

_OPENXR_PRESETS: dict[StereoModePreset, OpenXRRenderConfig] = {
    "auto": OpenXRRenderConfig(),
    "cinema": OpenXRRenderConfig(depth_strength=1.8, convergence=0.0, ipd=0.064, max_shift_ratio=0.045),
    "game_low_latency": OpenXRRenderConfig(depth_strength=1.5, convergence=0.0, ipd=0.064, max_shift_ratio=0.04),
    "still_image_hq": OpenXRRenderConfig(depth_strength=2.1, convergence=0.0, ipd=0.064, max_shift_ratio=0.05),
    "debug_export": OpenXRRenderConfig(depth_strength=2.0, convergence=0.0, ipd=0.064, max_shift_ratio=0.05),
}
