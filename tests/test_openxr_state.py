from __future__ import annotations

from types import SimpleNamespace

from stereo_runtime.adapter import openxr_render_config_from_snapshot
from stereo_runtime.openxr_state import OpenXRStateController
from stereo_runtime.settings_snapshot import RuntimeSettingsSnapshot
from xr_viewer.core_source_state import CoreSourceStateMixin


def make_runtime():
    return SimpleNamespace(
        stereo_config=SimpleNamespace(
            ipd_mm=32.0,
            stereo_scale=0.4,
            max_shift_ratio=0.04,
            max_disparity_px=None,
            parallax_preset="standard",
        )
    )


def test_openxr_source_paused_depends_on_bootstrap_and_source_active(capsys):
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_strength=1.0,
        convergence=0.5,
    )

    assert state.source_paused() is False
    state.bootstrap_done.set()
    assert state.source_paused() is True
    state.source_active.set()
    assert state.source_paused() is False

    output = capsys.readouterr().out
    assert "OpenXR source inference paused" in output
    assert "OpenXR source inference resumed" in output


def test_openxr_hard_idle_on_enter_runs_once_per_transition(capsys):
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_strength=1.0,
        convergence=0.5,
    )
    calls = []
    state.bootstrap_done.set()
    state.wait_idle_active.set()

    assert state.hard_idle_active(on_enter=lambda: calls.append("enter")) is True
    assert state.hard_idle_active(on_enter=lambda: calls.append("enter")) is True
    state.wait_idle_active.clear()
    assert state.hard_idle_active(on_enter=lambda: calls.append("enter")) is False

    assert calls == ["enter"]
    output = capsys.readouterr().out
    assert "OpenXR hard idle entered" in output
    assert "OpenXR hard idle exited" in output


def test_openxr_render_config_from_snapshot_resolves_normalized_fields():
    snapshot = RuntimeSettingsSnapshot(
        version=1,
        timestamp=1.0,
        depth_strength=1.5,
        convergence=0.2,
        ipd_mm=64.0,
        stereo_scale=0.5,
        max_shift_ratio=0.04,
        parallax_preset="standard",
    )

    config = openxr_render_config_from_snapshot(snapshot, render_size=(1920, 1080), screen_roll=0.15)

    assert config.ipd == 0.064
    assert config.ipd_mm == 64.0
    assert config.stereo_scale == 0.5
    assert config.depth_strength == 1.5
    assert config.convergence == 0.2
    assert config.max_shift_ratio == 0.04
    assert config.max_disparity_px is not None
    assert config.parallax_preset == "standard"
    assert config.screen_roll == 0.15


def test_openxr_runtime_config_update_and_render_config():
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_strength=1.0,
        convergence=0.5,
    )

    state.update_runtime_config(
        ipd=0.065,
        depth_strength=2.0,
        convergence=0.7,
        stereo_scale=0.35,
        max_shift_ratio=0.055,
        screen_roll=0.1,
    )
    config = state.current_render_config(make_runtime())

    assert config.ipd == 0.065
    assert config.ipd_mm == 32.0
    assert config.stereo_scale == 0.35
    assert config.depth_strength == 2.0
    assert config.convergence == 0.7
    assert config.max_shift_ratio == 0.055
    assert config.screen_roll == 0.1


def test_openxr_runtime_config_accepts_runtime_settings_snapshot():
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_strength=1.0,
        convergence=0.5,
    )

    state.update_runtime_config(
        snapshot=RuntimeSettingsSnapshot(
            version=2,
            timestamp=2.0,
            depth_strength=1.75,
            convergence=0.25,
            ipd_mm=63.0,
            stereo_scale=0.45,
            max_shift_ratio=0.035,
            max_disparity_px=24.0,
            parallax_preset="comfort",
        ),
        screen_roll=0.2,
    )
    config = state.current_render_config(make_runtime())

    assert config.ipd == 0.063
    assert config.ipd_mm == 63.0
    assert config.stereo_scale == 0.45
    assert config.depth_strength == 1.75
    assert config.convergence == 0.25
    assert config.max_shift_ratio == 0.035
    assert config.max_disparity_px == 24.0
    assert config.parallax_preset == "comfort"
    assert config.screen_roll == 0.2


def test_openxr_runtime_config_falls_back_to_runtime_stereo_config():
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_strength=1.0,
        convergence=0.5,
    )

    config = state.current_render_config(make_runtime())

    assert config.stereo_scale == 0.4
    assert config.max_shift_ratio == 0.04


def test_openxr_runtime_config_can_initialize_stereo_scale_and_max_shift():
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_strength=1.0,
        convergence=0.5,
        stereo_scale=0.35,
        max_shift_ratio=0.05,
    )

    config = state.current_render_config(make_runtime())

    assert config.stereo_scale == 0.35
    assert config.max_shift_ratio == 0.05


class RuntimeConfigPublisher(CoreSourceStateMixin):
    def __init__(self, callback):
        self._runtime_config_callback = callback
        self.screen_roll = 0.25
        self.ipd_uv = 0.064
        self.depth_strength = 2.5
        self.convergence = 0.1

    def _quad_layer_can_replace_projection_screen(self):
        return False


def test_viewer_runtime_config_publish_defaults_to_screen_roll_only():
    calls = []
    publisher = RuntimeConfigPublisher(lambda **kwargs: calls.append(kwargs))

    publisher._publish_runtime_config()
    publisher._publish_runtime_config(include_stereo=True)

    assert calls[0] == {"screen_roll": 0.25}
    assert calls[1] == {
        "screen_roll": 0.25,
        "ipd": 0.064,
        "depth_strength": 2.5,
        "convergence": 0.1,
    }
