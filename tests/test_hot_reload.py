from __future__ import annotations

from types import SimpleNamespace

from stereo_runtime.adapter import StereoRuntimeConfig
from stereo_runtime.hot_reload import (
    StereoHotReloader,
    clamp_foreground_scale_hot_reload,
    hot_reload_runtime_settings_snapshot,
    hot_reload_value_snapshot,
    runtime_stereo_overrides,
    to_bool_hot_reload,
)
from stereo_runtime.settings_snapshot import SnapshotChangeClass


def make_config(**overrides):
    values = {
        "stereo_quality": "fast_plus",
        "depth_strength": 1.0,
        "convergence": 0.5,
        "ipd": 0.032,
        "ipd_mm": 32.0,
        "stereo_scale": 1.0,
        "max_shift_ratio": 0.03,
        "output_format": "half_sbs",
        "max_disparity_px": None,
        "parallax_preset": "standard",
        "temporal": True,
        "foreground_scale": 1.0,
        "depth_antialias_strength": 0.25,
        "edge_threshold": 0.1,
        "edge_dilation": 2,
        "mask_feather_radius": 3,
        "hole_fill_mode": "balanced",
        "hole_fill_radius": 3,
        "hole_fill_strength": 1.0,
        "screen_edge_mask_suppression": 0.0,
        "cross_eyed": False,
        "anaglyph_method": "dubois",
        "debug_output": False,
        "fused": True,
        "temporal_strength": 0.5,
        "auto_reset_temporal": True,
        "scene_reset_threshold": 0.2,
        "reset_cooldown_frames": 4,
        "stereo_preset": "cinema",
        "mode": "cinema",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_hot_reload_value_snapshot_parses_expected_fields():
    config = make_config()
    settings = {
        "Depth Strength": "1.25",
        "Convergence": "0.75",
        "IPD mm": "65",
        "Stereo Scale": "1.5",
        "Max Shift Ratio": "0.04",
        "Display Mode": "Full-SBS",
        "Max Disparity Px": "88",
        "Parallax Preset": "strong",
        "Temporal Strength": "0",
        "Scene Reset Threshold": "0.3",
        "Reset Cooldown Frames": "7",
        "Foreground Scale": "9.0",
        "Depth Antialias Strength": "0.8",
        "Edge Dilation": "3",
        "Mask Feather Radius": "4",
        "Hole Fill Mode": "Soft / Low Ghost",
        "Hole Fill Radius": "5",
        "Hole Fill Strength": "0.4",
        "Edge Threshold": "0.2",
        "Anaglyph Method": "gray",
        "Cross Eyed": "yes",
        "Screen Edge Mask Suppression": "2",
        "Debug Stereo Output": "yes",
    }

    values = hot_reload_value_snapshot(settings, config)

    assert values["depth_strength"] == 1.25
    assert values["convergence"] == 0.75
    assert values["ipd"] == 0.065
    assert values["ipd_mm"] == 65.0
    assert values["stereo_scale"] == 1.5
    assert values["max_shift_ratio"] == 0.04
    assert values["output_format"] == "full_sbs"
    assert values["max_disparity_px"] == 88.0
    assert values["parallax_preset"] == "strong"
    assert values["temporal"] is False
    assert values["scene_reset_threshold"] == 0.3
    assert values["reset_cooldown_frames"] == 7
    assert values["foreground_scale"] == 5.0
    assert values["depth_antialias_strength"] == 0.8
    assert values["edge_dilation"] == 3
    assert values["mask_feather_radius"] == 4
    assert values["hole_fill_mode"] == "soft_low_ghost"
    assert values["hole_fill_radius"] == 1
    assert values["hole_fill_strength"] == 0.6
    assert values["edge_threshold"] == 0.2
    assert values["anaglyph_method"] == "gray"
    assert values["cross_eyed"] is True
    assert values["screen_edge_mask_suppression"] == 2
    assert values["debug_output"] is True
    assert values["debug_flags"] == {"debug_output": True}


def test_runtime_stereo_overrides_maps_runtime_config():
    runtime = SimpleNamespace(config=make_config())

    overrides = runtime_stereo_overrides(runtime)

    assert overrides["backend"] == "fast_plus"
    assert overrides["depth_strength"] == 1.0
    assert overrides["temporal"] is True
    assert overrides["temporal_strength"] == 0.5
    assert overrides["auto_reset_temporal"] is True
    assert overrides["scene_reset_threshold"] == 0.2
    assert overrides["reset_cooldown_frames"] == 4
    assert overrides["mask_feather_radius"] == 3
    assert overrides["hole_fill_mode"] == "balanced"
    assert overrides["hole_fill_radius"] == 3
    assert overrides["hole_fill_strength"] == 1.0
    assert overrides["cross_eyed"] is False
    assert overrides["debug_output"] is False
    assert overrides["fused"] is True


def test_hot_reload_bool_and_foreground_helpers():
    assert to_bool_hot_reload(True) is True
    assert to_bool_hot_reload("on") is True
    assert to_bool_hot_reload(None) is False
    assert clamp_foreground_scale_hot_reload(-5.0) == -0.9
    assert clamp_foreground_scale_hot_reload(10.0) == 5.0


def test_hot_reload_fast_quality_disables_temporal_and_postprocess():
    config = make_config(stereo_quality="fast", temporal_strength=0.7, foreground_scale=0.5, depth_antialias_strength=2.0)
    settings = {
        "Stereo Quality": "fast",
        "Synthetic View": "fast",
        "Temporal Strength": "0.7",
        "Scene Reset Threshold": "0.22",
        "Reset Cooldown Frames": "3",
        "Foreground Scale": "0.5",
        "Depth Antialias Strength": "2.0",
    }

    values = hot_reload_value_snapshot(settings, config)

    assert values["temporal"] is False
    assert values["temporal_strength"] == 0.0
    assert values["auto_reset_temporal"] is False
    assert values["scene_reset_threshold"] == 0.0
    assert values["reset_cooldown_frames"] == 0
    assert values["foreground_scale"] == 0.0
    assert values["depth_antialias_strength"] == 0.0


def test_hot_reload_builds_runtime_settings_snapshot():
    config = make_config()
    settings = {
        "Depth Strength": "1.25",
        "IPD mm": "65",
        "Display Mode": "Half-TAB",
        "Max Disparity PX": "96",
        "Parallax Budget Preset": "comfort",
        "Temporal Strength": "0.3",
        "Scene Reset Threshold": "0.4",
        "Reset Cooldown Frames": "8",
        "Screen Edge Mask Suppression": "3",
        "Debug Stereo Output": "true",
    }

    snapshot = hot_reload_runtime_settings_snapshot(
        settings,
        config,
        version=12,
        timestamp=3.5,
    )

    assert snapshot.version == 12
    assert snapshot.timestamp == 3.5
    assert snapshot.source == "settings_yaml_hot_reload"
    assert snapshot.depth_strength == 1.25
    assert snapshot.ipd_mm == 65.0
    assert snapshot.output_format == "half_tab"
    assert snapshot.max_disparity_px == 96.0
    assert snapshot.parallax_preset == "comfort"
    assert snapshot.temporal is True
    assert snapshot.temporal_strength == 0.3
    assert snapshot.auto_reset_temporal is True
    assert snapshot.scene_reset_threshold == 0.4
    assert snapshot.reset_cooldown_frames == 8
    assert snapshot.screen_edge_mask_suppression == 3
    assert snapshot.debug_output is True
    assert snapshot.debug_flags == {"debug_output": True}
    assert snapshot.classify() is SnapshotChangeClass.HOT_RELOAD


def test_hot_reload_pushes_all_openxr_stereo_controls(tmp_path):
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text("", encoding="utf-8")
    runtime_config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        stereo_quality="fast",
        stereo_preset="cinema",
    )
    runtime = SimpleNamespace(
        config=runtime_config,
        stereo_config=SimpleNamespace(output_format="half_sbs"),
        apply_settings_snapshot=lambda snapshot, active_preset=None: setattr(runtime, "applied_snapshot", snapshot),
        configure_stereo=lambda stereo_config, reset_temporal=False: setattr(runtime, "stereo_config", stereo_config),
    )
    pushed = {}
    reloader = StereoHotReloader(
        settings_path=str(settings_path),
        interval_s=0.0,
        read_settings=lambda _path: {
            "Stereo Quality": "fast",
            "Depth Strength": "2.0",
            "Convergence": "0.25",
            "IPD mm": "64",
            "Stereo Scale": "0.35",
            "Max Shift Ratio": "0.05",
            "Display Mode": "Full-TAB",
            "Max Disparity Px": "72",
            "Parallax Preset": "standard",
            "Screen Edge Mask Suppression": "4",
            "Debug Stereo Output": "on",
        },
        clock=lambda: 1.0,
    )

    assert reloader.apply_if_needed(
        runtime=runtime,
        active_preset="cinema",
        on_openxr_config_update=lambda **kwargs: pushed.update(kwargs),
        on_mode_log=lambda _reason: None,
    )

    assert pushed["snapshot"] is runtime.applied_snapshot
    assert runtime.applied_snapshot.source == "settings_yaml_hot_reload"
    assert runtime.applied_snapshot.ipd_mm == 64.0
    assert runtime.applied_snapshot.depth_strength == 2.0
    assert runtime.applied_snapshot.convergence == 0.25
    assert runtime.applied_snapshot.stereo_scale == 0.35
    assert runtime.applied_snapshot.max_shift_ratio == 0.05
    assert runtime.applied_snapshot.output_format == "full_tab"
    assert runtime.applied_snapshot.max_disparity_px == 72.0
    assert runtime.applied_snapshot.parallax_preset == "standard"
    assert runtime.applied_snapshot.screen_edge_mask_suppression == 4
    assert runtime.applied_snapshot.debug_output is True
    assert runtime.applied_snapshot.debug_flags == {"debug_output": True}
