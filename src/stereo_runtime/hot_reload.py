from __future__ import annotations

import os
import time
from dataclasses import replace
from typing import Callable

from stereo_runtime import stereo_config_for_preset
from stereo_runtime.adapter import preset_for_runtime_mode

def read_yaml(path: str) -> dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def runtime_stereo_overrides(runtime) -> dict:
    config = runtime.config
    return {
        "backend": config.stereo_quality,
        "depth_strength": config.depth_strength,
        "convergence": config.convergence,
        "ipd": config.ipd,
        "ipd_mm": config.ipd_mm,
        "stereo_scale": config.stereo_scale,
        "max_shift_ratio": config.max_shift_ratio,
        "temporal": config.temporal,
        "temporal_strength": config.temporal_strength,
        "auto_reset_temporal": config.auto_reset_temporal,
        "scene_reset_threshold": config.scene_reset_threshold,
        "reset_cooldown_frames": config.reset_cooldown_frames,
        "foreground_scale": config.foreground_scale,
        "depth_antialias_strength": config.depth_antialias_strength,
        "edge_threshold": config.edge_threshold,
        "edge_dilation": config.edge_dilation,
        "screen_edge_mask_suppression": config.screen_edge_mask_suppression,
        "cross_eyed": config.cross_eyed,
        "anaglyph_method": config.anaglyph_method,
        "fused": config.fused,
    }


def to_bool_hot_reload(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def clamp_foreground_scale_hot_reload(value) -> float:
    return max(-0.9, min(5.0, float(value)))


def _is_fast_quality(settings_dict: dict, config) -> bool:
    raw = settings_dict.get("Stereo Quality", settings_dict.get("Synthetic View", getattr(config, "stereo_quality", "")))
    key = str(raw).strip().lower().replace("-", "_").replace("+", "_plus").replace(" ", "_")
    return key == "fast"


def hot_reload_value_snapshot(settings_dict: dict, config) -> dict:
    ipd_raw = settings_dict.get(
        "IPD mm",
        settings_dict.get("IPD (mm)", settings_dict.get("IPD", config.ipd_mm or 64.0)),
    )
    ipd_mm = float(ipd_raw)
    if ipd_mm <= 1.0:
        ipd_mm *= 1000.0
    values = {
        "depth_strength": float(settings_dict.get("Depth Strength", config.depth_strength)),
        "convergence": float(settings_dict.get("Convergence", config.convergence)),
        "ipd": ipd_mm / 1000.0,
        "ipd_mm": max(1.0, ipd_mm),
        "stereo_scale": float(
            settings_dict.get(
                "Stereo Scale",
                settings_dict.get("Stereo Strength Scale", config.stereo_scale),
            )
        ),
        "max_shift_ratio": float(settings_dict.get("Max Shift Ratio", config.max_shift_ratio)),
        "temporal": float(settings_dict.get("Temporal Strength", config.temporal_strength)) > 0.0,
        "temporal_strength": float(settings_dict.get("Temporal Strength", config.temporal_strength)),
        "auto_reset_temporal": float(
            settings_dict.get("Scene Reset Threshold", config.scene_reset_threshold)
        )
        > 0.0,
        "scene_reset_threshold": float(
            settings_dict.get("Scene Reset Threshold", config.scene_reset_threshold)
        ),
        "reset_cooldown_frames": int(
            settings_dict.get("Reset Cooldown Frames", config.reset_cooldown_frames)
        ),
        "foreground_scale": clamp_foreground_scale_hot_reload(
            settings_dict.get("Foreground Scale", config.foreground_scale)
        ),
        "depth_antialias_strength": float(
            settings_dict.get(
                "Depth Antialias Strength",
                settings_dict.get("Anti-aliasing", config.depth_antialias_strength),
            )
        ),
        "edge_dilation": int(settings_dict.get("Edge Dilation", config.edge_dilation)),
        "edge_threshold": float(settings_dict.get("Edge Threshold", config.edge_threshold)),
        "anaglyph_method": str(settings_dict.get("Anaglyph Method", config.anaglyph_method)),
        "cross_eyed": to_bool_hot_reload(settings_dict.get("Cross Eyed", config.cross_eyed)),
    }
    if _is_fast_quality(settings_dict, config):
        values.update(
            {
                "temporal": False,
                "temporal_strength": 0.0,
                "auto_reset_temporal": False,
                "scene_reset_threshold": 0.0,
                "reset_cooldown_frames": 0,
                "foreground_scale": 0.0,
                "depth_antialias_strength": 0.0,
            }
        )
    return values


class StereoHotReloader:
    def __init__(
        self,
        *,
        settings_path: str,
        interval_s: float = 0.25,
        read_settings: Callable[[str], dict] = read_yaml,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.settings_path = settings_path
        self.interval_s = interval_s
        self.read_settings = read_settings
        self.clock = clock
        self.last_check = 0.0
        self.last_mtime = os.path.getmtime(settings_path) if os.path.exists(settings_path) else 0.0
        self.last_values = None

    def apply_if_needed(
        self,
        *,
        runtime,
        active_preset,
        on_openxr_config_update: Callable[..., None],
        on_mode_log: Callable[[str], None],
    ) -> bool:
        now = self.clock()
        if now - self.last_check < self.interval_s:
            return False
        self.last_check = now
        try:
            mtime = os.path.getmtime(self.settings_path)
        except OSError:
            return False
        if mtime <= self.last_mtime and self.last_values is not None:
            return False
        try:
            settings_dict = self.read_settings(self.settings_path)
            values = hot_reload_value_snapshot(settings_dict, runtime.config)
        except Exception as exc:
            print(f"[Main] Stereo hot reload skipped: {type(exc).__name__}: {exc}", flush=True)
            self.last_mtime = mtime
            return False
        if values == self.last_values:
            self.last_mtime = mtime
            return False

        runtime.config = replace(runtime.config, **values)
        current = runtime.stereo_config
        runtime.configure_stereo(
            stereo_config_for_preset(
                active_preset or runtime.config.stereo_preset or preset_for_runtime_mode(runtime.config.mode),
                output_format=current.output_format,
                overrides=runtime_stereo_overrides(runtime),
            ),
            reset_temporal=False,
        )
        on_openxr_config_update(
            ipd=values["ipd"],
            depth_ratio=values["depth_strength"],
            convergence=values["convergence"],
        )
        self.last_values = values
        self.last_mtime = mtime
        print(
            "[Main] Stereo hot reload:"
            f" ipd_mm={values['ipd_mm']:.1f}"
            f" stereo_scale={values['stereo_scale']:.3f}"
            f" depth_strength={values['depth_strength']:.3f}"
            f" convergence={values['convergence']:.3f}"
            f" max_shift_ratio={values['max_shift_ratio']:.3f}"
            f" temporal_strength={values['temporal_strength']:.3f}"
            f" scene_reset={values['scene_reset_threshold']:.3f}"
            f" reset_cooldown={values['reset_cooldown_frames']}"
            f" foreground_scale={values['foreground_scale']:.3f}"
            f" antialias={values['depth_antialias_strength']:.3f}"
            f" edge_dilation={values['edge_dilation']}"
            f" edge_threshold={values['edge_threshold']:.3f}"
            f" anaglyph={values['anaglyph_method']}"
            f" cross_eyed={int(values['cross_eyed'])}",
            flush=True,
        )
        on_mode_log("hot-reload")
        return True
