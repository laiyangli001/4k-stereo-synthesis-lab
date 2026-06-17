import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab.openxr_render import OpenXRRenderConfig
from stereo_lab.presets import (
    AutoModeSignals,
    PRESET_CHOICES,
    classify_auto_mode,
    openxr_config_for_auto_mode,
    openxr_config_for_preset,
    preset_summary,
    stereo_config_for_auto_mode,
    stereo_config_for_preset,
)
from stereo_lab.synthesis import StereoConfig


def test_preset_choices_are_public_and_stable():
    assert PRESET_CHOICES == ("auto", "cinema", "game_low_latency", "still_image_hq", "debug_export")


def test_stereo_presets_map_to_expected_modes():
    cinema = stereo_config_for_preset("cinema")
    game = stereo_config_for_preset("game")
    still = stereo_config_for_preset("still image / hq")
    debug = stereo_config_for_preset("debug")

    assert isinstance(cinema, StereoConfig)
    assert cinema.backend == "quality_4k"
    assert cinema.temporal is True
    assert cinema.auto_reset_temporal is True

    assert game.backend == "quality_4k"
    assert game.hole_fill == "fast"
    assert game.temporal_strength < cinema.temporal_strength
    assert game.depth_strength < cinema.depth_strength

    assert still.backend == "hq_4k"
    assert still.layers == 3
    assert still.temporal is False
    assert still.auto_reset_temporal is False

    assert debug.debug_output is True
    assert debug.depth_strength >= cinema.depth_strength


def test_preset_output_format_and_overrides():
    config = stereo_config_for_preset("cinema", output_format="full_sbs", overrides={"depth_strength": 2.25})
    assert config.output_format == "full_sbs"
    assert config.depth_strength == 2.25

    with pytest.raises(ValueError, match="unknown config override"):
        stereo_config_for_preset("cinema", overrides={"not_a_field": 1})


def test_openxr_presets_map_shared_stereo_params():
    openxr = openxr_config_for_preset("game_low_latency", screen_roll=0.5)
    cinema = openxr_config_for_preset("cinema")

    assert isinstance(openxr, OpenXRRenderConfig)
    assert openxr.screen_roll == 0.5
    assert openxr.depth_strength < cinema.depth_strength
    assert openxr.max_shift_ratio <= cinema.max_shift_ratio


def test_auto_mode_classifier_priority():
    export = classify_auto_mode(AutoModeSignals(user_export_action=True, frame_motion_score=1.0))
    still = classify_auto_mode(AutoModeSignals(still_duration_s=2.0, frame_motion_score=0.01))
    game = classify_auto_mode(AutoModeSignals(frame_motion_score=0.5))
    cinema = classify_auto_mode(AutoModeSignals(frame_motion_score=0.05))

    assert export.preset == "debug_export"
    assert still.preset == "still_image_hq"
    assert game.preset == "game_low_latency"
    assert cinema.preset == "cinema"


def test_auto_mode_config_helpers_return_decision_and_config():
    decision, stereo = stereo_config_for_auto_mode(AutoModeSignals(foreground_process="SteamGame.exe"), output_format="half_tab")
    openxr_decision, openxr = openxr_config_for_auto_mode(AutoModeSignals(openxr_active=True), screen_roll=0.25)

    assert decision.preset == "game_low_latency"
    assert stereo.output_format == "half_tab"
    assert stereo.temporal_strength <= 0.7
    assert openxr_decision.preset == "cinema"
    assert openxr.screen_roll == 0.25


def test_preset_summary_is_serializable_shape():
    summary = preset_summary()
    assert set(summary) == {"cinema", "game_low_latency", "still_image_hq", "debug_export"}
    assert summary["cinema"]["stereo"]["backend"] == "quality_4k"
    assert "depth_strength" in summary["cinema"]["openxr"]


def test_presets_do_not_control_depth_provider_or_model_paths():
    forbidden = {
        "cache_dir",
        "depth_backend",
        "depth_resolution",
        "engine_path",
        "model_id",
        "model_name",
        "onnx_path",
        "trt_cache_dir",
    }
    summary = preset_summary()
    for preset in summary.values():
        assert forbidden.isdisjoint(preset["stereo"])
        assert forbidden.isdisjoint(preset["openxr"])
