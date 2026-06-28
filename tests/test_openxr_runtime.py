import os
import queue
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from xr_viewer.openxr_runtime import (
    OpenXRRuntimeCallbacks,
    OpenXRRuntimeConfig,
    frame_size_from_eye,
    frame_size_from_runtime_result,
    load_openxr_viewer,
    run_openxr_mode,
    use_environment_viewer,
)


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_openxr_runtime_import_does_not_load_xr_implementation(tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    code = "from xr_viewer.openxr_runtime import use_environment_viewer; print(use_environment_viewer('none'))"

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_use_environment_viewer_requires_real_environment_name():
    assert not use_environment_viewer(None)
    assert not use_environment_viewer("")
    assert not use_environment_viewer(" none ")
    assert use_environment_viewer("Default")
    assert use_environment_viewer("studio")


def test_frame_size_from_eye_uses_height_width_shape_for_array_like():
    eye = SimpleNamespace(shape=(720, 1280, 4))

    assert frame_size_from_eye(eye) == (1280, 720)


def test_frame_size_from_runtime_result_prefers_display_size_metadata():
    result = SimpleNamespace(
        left_eye=SimpleNamespace(shape=(1, 3, 2160, 1920)),
        output_display_size=(3840, 2160),
        debug_info={"runtime_output_display_size": "1920x1080"},
    )

    assert frame_size_from_runtime_result(result) == (3840, 2160)


def test_frame_size_from_runtime_result_supports_legacy_debug_display_size():
    result = SimpleNamespace(
        left_eye=SimpleNamespace(shape=(1, 3, 2160, 1920)),
        debug_info={"runtime_output_display_size": "3840x2160"},
    )

    assert frame_size_from_runtime_result(result) == (3840, 2160)


def test_load_openxr_viewer_uses_environment_split_for_named_environment(monkeypatch):
    fake_base_viewer = object()
    fake_environment_viewer = object()
    monkeypatch.setitem(
        sys.modules,
        "xr_viewer.base",
        types.SimpleNamespace(OPENXR_AVAILABLE=True, OpenXRViewer=fake_base_viewer),
    )
    monkeypatch.setitem(
        sys.modules,
        "xr_viewer.environment",
        types.SimpleNamespace(OPENXR_AVAILABLE=True, OpenXRViewer=fake_environment_viewer),
    )

    assert load_openxr_viewer("Cinema") is fake_environment_viewer
    assert load_openxr_viewer("Default") is fake_environment_viewer
    assert load_openxr_viewer("none") is fake_base_viewer


def test_load_openxr_viewer_raises_when_runtime_unavailable(monkeypatch):
    fake_base = types.SimpleNamespace(OPENXR_AVAILABLE=False, OpenXRViewer=object)
    monkeypatch.setitem(sys.modules, "xr_viewer.base", fake_base)

    with pytest.raises(ImportError, match="pyopenxr not installed"):
        load_openxr_viewer(None)


def test_run_openxr_mode_passes_depth_strength_to_viewer(monkeypatch):
    calls = []

    class FakeViewer:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def run(self, **kwargs):
            calls.append({"run": kwargs})

    monkeypatch.setitem(
        sys.modules,
        "xr_viewer.base",
        types.SimpleNamespace(OPENXR_AVAILABLE=True, OpenXRViewer=FakeViewer),
    )
    runtime_q = queue.Queue()
    runtime_q.put(
        (
            SimpleNamespace(
                left_eye=SimpleNamespace(shape=(1, 3, 2160, 1920)),
                output_display_size=(3840, 2160),
                debug_info={"runtime_output_display_size": "1920x1080"},
            ),
            123.0,
        )
    )
    config = OpenXRRuntimeConfig(
        ipd=0.064,
        depth_strength=2.4,
        convergence=0.1,
        fps=72,
        show_fps=True,
        controller_model="pico",
        environment_model="none",
        show_preview_window=False,
        capture_mode="Monitor",
        monitor_index=1,
    )
    callbacks = OpenXRRuntimeCallbacks(
        update_runtime_config=lambda *args, **kwargs: None,
        render_active_set=lambda: None,
        render_active_clear=lambda: None,
        source_active_set=lambda: None,
        wait_idle_clear=lambda: None,
        bootstrap_done_set=lambda: None,
    )

    viewer = run_openxr_mode(runtime_q, config, callbacks)

    assert isinstance(viewer, FakeViewer)
    assert calls[0]["depth_strength"] == 2.4
    assert "depth_ratio" not in calls[0]
    assert calls[0]["frame_size"] == (3840, 2160)


def test_runtime_eye_tensor_hwc_u8_scales_near_normalized_float_range(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    frame = torch.full((1, 3, 2, 2), 0.5, dtype=torch.float32)
    frame[..., 0, 0] = 1.0001

    out = CoreRuntimeEyeMixin()._runtime_eye_tensor_hwc_u8(torch, frame)

    assert out.dtype == torch.uint8
    assert tuple(out.shape) == (2, 2, 3)
    assert int(out[1, 1, 0]) == 127
    assert int(out[0, 0, 0]) == 255


def test_runtime_eye_stats_log_prefers_structured_output_fields(monkeypatch, capsys):
    torch = pytest.importorskip("torch")
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    left_eye = torch.zeros(1, 3, 2, 4, dtype=torch.float32)
    right_eye = torch.ones(1, 3, 2, 4, dtype=torch.float32)
    result = SimpleNamespace(
        left_eye=left_eye,
        right_eye=right_eye,
        output_format="openxr_full_synthesis_eyes",
        output_dtype="uint8",
        output_eye_size=(3840, 2160),
        output_pack_backend="none",
        debug_info={
            "runtime_output_format": "legacy_format",
            "runtime_output_dtype": "legacy_dtype",
            "runtime_output_eye_size": "1x1",
            "runtime_output_pack_backend": "legacy_pack",
        },
    )

    viewer._log_runtime_eye_stats_once(result, upload_path="cpu")

    output = capsys.readouterr().out
    assert "format=openxr_full_synthesis_eyes" in output
    assert "runtime_dtype=uint8" in output
    assert "eye_size=(3840, 2160)" in output
    assert "pack=none" in output
    assert "legacy_format" not in output
    assert "legacy_dtype" not in output
    assert "eye_size=1x1" not in output
    assert "legacy_pack" not in output


def test_d3d11_rgb_depth_shader_uses_screen_roll_for_parallax_direction(monkeypatch):
    monkeypatch.chdir(SRC)
    source = (SRC / "xr_viewer" / "d3d11_native_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "#define roll params.w" in source
    assert "float2 shiftedUv = uv - float2(shift * cos(roll), shift * sin(roll));" in source
    assert "def render_eye(self, swapchain_texture, width, height, eye_index, ipd, depth_strength, convergence, mvp, roll=0.0):" in source
    assert "constants[16:20] = np.array([eye_sign * ipd * 0.5, depth_strength, convergence, roll]" in source
    assert "self.runtime_eye_srv[eye_index], 0.0, 0.0, 0.0, mvp, roll=0.0" in source
    assert "roll=self.screen_roll" in implementation


def test_runtime_rgb_depth_config_prefers_structured_legacy_shader_uniforms(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    viewer._apply_runtime_rgb_depth_config(
        {
            "openxr_legacy_shader_uniforms": {
                "convergence": 9.0,
                "ipd": 9.0,
                "stereo_scale": 9.0,
                "max_shift_ratio": 9.0,
            },
            "openxr_convergence": 8.0,
            "openxr_ipd": 8.0,
            "openxr_stereo_scale": 8.0,
            "openxr_max_shift_ratio": 8.0,
        },
        legacy_shader_uniforms={
            "convergence": 0.25,
            "ipd": 0.061,
            "stereo_scale": 0.42,
            "max_shift_ratio": 0.07,
        },
    )

    assert viewer.convergence == 0.25
    assert viewer.ipd_uv == 0.061
    assert viewer._runtime_rgb_depth_stereo_scale == 0.42
    assert viewer._runtime_rgb_depth_max_shift_ratio == 0.07


def test_runtime_rgb_depth_config_keeps_debug_uniform_fallback(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    viewer._apply_runtime_rgb_depth_config(
        {
            "openxr_legacy_shader_uniforms": {
                "convergence": 0.25,
                "ipd": 0.061,
                "stereo_scale": 0.42,
                "max_shift_ratio": 0.07,
            },
            "openxr_convergence": 9.0,
            "openxr_ipd": 9.0,
            "openxr_stereo_scale": 9.0,
            "openxr_max_shift_ratio": 9.0,
        }
    )

    assert viewer.convergence == 0.25
    assert viewer.ipd_uv == 0.061
    assert viewer._runtime_rgb_depth_stereo_scale == 0.42
    assert viewer._runtime_rgb_depth_max_shift_ratio == 0.07
