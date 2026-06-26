from __future__ import annotations

import pytest

from stereo_runtime.render_size import (
    RenderSizeConfig,
    RenderSizePolicy,
    render_size_config_from_settings,
    resolve_render_size,
    runtime_output_size_text,
)


def test_render_size_config_from_settings_parses_gui_fields():
    config = render_size_config_from_settings(
        {
            "Render Size Policy": "scaled",
            "Render Scale": "0.75",
            "Render Fixed Width": "1600",
            "Render Fixed Height": "900",
            "Render Max Pixels": "2073600",
            "Render Min Dimension": "540",
            "Render Align": "8",
        }
    )

    assert config == RenderSizeConfig(
        policy=RenderSizePolicy.SCALED,
        scale_factor=0.75,
        fixed_width=1600,
        fixed_height=900,
        max_pixels=2073600,
        min_dimension=540,
        align=8,
    )


def test_render_size_config_from_settings_defaults_invalid_policy_to_native():
    config = render_size_config_from_settings({"Render Size Policy": "unknown"})

    assert config.policy is RenderSizePolicy.NATIVE


def test_resolve_render_size_native_aligns_capture_size():
    config = RenderSizeConfig(policy=RenderSizePolicy.NATIVE, align=16)

    assert resolve_render_size((1919, 1079), config) == (1904, 1072)


def test_resolve_render_size_scaled_keeps_aspect_and_alignment():
    config = RenderSizeConfig(policy=RenderSizePolicy.SCALED, scale_factor=0.5, align=8)

    assert resolve_render_size((1920, 1080), config) == (960, 536)


def test_resolve_render_size_fixed_uses_configured_size():
    config = RenderSizeConfig(policy=RenderSizePolicy.FIXED, fixed_width=1280, fixed_height=720, align=16)

    assert resolve_render_size((3840, 2160), config) == (1280, 720)


def test_resolve_render_size_dynamic_caps_pixels():
    config = RenderSizeConfig(
        policy=RenderSizePolicy.DYNAMIC,
        max_pixels=1280 * 720,
        min_dimension=360,
        align=16,
    )

    width, height = resolve_render_size((3840, 2160), config)

    assert width <= 1280
    assert height <= 720
    assert width % 16 == 0
    assert height % 16 == 0


def test_runtime_output_size_text_validates_size():
    assert runtime_output_size_text((1920, 1080)) == "1920x1080"
    assert runtime_output_size_text(None) == "unknown"
    with pytest.raises(ValueError):
        runtime_output_size_text((0, 1080))
