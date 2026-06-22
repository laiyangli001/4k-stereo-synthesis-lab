from __future__ import annotations

from types import SimpleNamespace

from stereo_runtime.openxr_state import OpenXRStateController


def make_runtime():
    return SimpleNamespace(
        stereo_config=SimpleNamespace(
            ipd_mm=64.0,
            stereo_scale=1.2,
            max_shift_ratio=0.04,
        )
    )


def test_openxr_source_paused_depends_on_bootstrap_and_source_active(capsys):
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_ratio=1.0,
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
        depth_ratio=1.0,
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


def test_openxr_runtime_config_update_and_render_config():
    state = OpenXRStateController(
        run_mode="OpenXR",
        ipd=0.064,
        depth_ratio=1.0,
        convergence=0.5,
    )

    state.update_runtime_config(ipd=0.065, depth_ratio=2.0, convergence=0.7, screen_roll=0.1)
    config = state.current_render_config(make_runtime())

    assert config.ipd == 0.065
    assert config.ipd_mm == 64.0
    assert config.stereo_scale == 1.2
    assert config.depth_strength == 2.0
    assert config.convergence == 0.7
    assert config.max_shift_ratio == 0.04
    assert config.screen_roll == 0.1
