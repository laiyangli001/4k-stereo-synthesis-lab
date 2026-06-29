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
    assert "ipd" not in calls[0]
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


def test_cpu_fallback_paths_emit_red_console_warnings(monkeypatch):
    monkeypatch.chdir(SRC)
    runtime_eye = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    frame_upload = (SRC / "xr_viewer" / "core_frame_upload.py").read_text(encoding="utf-8")
    d3d11 = (SRC / "xr_viewer" / "d3d11_native_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    viewer = (SRC / "viewer" / "viewer.py").read_text(encoding="utf-8")
    gl_uploader = (SRC / "viewer" / "gl_texture_uploader.py").read_text(encoding="utf-8")
    metal = (SRC / "viewer" / "metal_viewer.py").read_text(encoding="utf-8")
    output_convert = (SRC / "stereo_runtime" / "output_convert.py").read_text(encoding="utf-8")
    depth_onnx = (SRC / "stereo_runtime" / "depth_onnx_provider.py").read_text(encoding="utf-8")
    tensorrt_ort = (SRC / "stereo_runtime" / "providers" / "nvidia" / "tensorrt_ort.py").read_text(encoding="utf-8")
    legacy_sbs = (SRC / "streaming" / "legacy_sbs.py").read_text(encoding="utf-8")
    warnings = (SRC / "utils" / "cpu_warnings.py").read_text(encoding="utf-8")

    assert "\\033[91m" in warnings
    assert "warn_cpu_fallback" in runtime_eye
    assert "warn_cpu_transfer" in runtime_eye
    assert "runtime_eye_not_cuda" in runtime_eye
    assert "texture_image=fallback" in runtime_eye
    assert "CudaGlTextureUploader" in runtime_eye
    assert "upload_rgba" in runtime_eye
    assert "runtime_eye_tensor" in runtime_eye
    assert "runtime_eye_sync" in runtime_eye
    assert "runtime_eye_image" in runtime_eye
    assert "runtime_eye_total" in runtime_eye
    assert "glGenerateMipmap" in gl_uploader
    assert "_upload_pbo" in gl_uploader
    assert "memcpy_2d_to_array" in gl_uploader
    assert "row_bytes, row_bytes" in gl_uploader
    assert "_runtime_eye_tensor_rgba_u8" in runtime_eye
    assert "D2S_OPENXR_RUNTIME_EYE_TEXTURE_GPU_UPLOAD" in implementation
    assert "os.environ.get('D2S_OPENXR_RUNTIME_EYE_TEXTURE_GPU_UPLOAD', '1')" in implementation
    assert "_runtime_eye_texture_components = 4" in implementation
    assert "warn_cpu_fallback" in frame_upload
    assert "OpenXR RGB+depth texture upload" in frame_upload
    assert "OpenXR depth texture upload" in frame_upload
    assert "OpenXR D3D11 RGB+depth texture upload" in d3d11
    assert "using_cpu_update_subresource" in d3d11
    assert "StereoWindow runtime texture upload" in viewer
    assert "StereoWindow RGB+depth texture upload" in viewer
    assert "CudaGlTextureUploader" in viewer
    assert "CUDA/GL image texture upload failed" in gl_uploader
    assert "using PBO fallback" in gl_uploader
    assert "requires {name} support" in gl_uploader
    assert "metal_rgb_cpu_transfer" in metal
    assert "metal_depth_cpu_transfer" in metal
    assert "runtime_output_to_numpy_cpu_transfer" in output_convert
    assert "depth_onnx_input_cpu_transfer" in depth_onnx
    assert "tensorrt_ort_input_cpu_transfer" in tensorrt_ort
    assert "tensorrt_ort_output_numpy_transfer" in tensorrt_ort
    assert "legacy_sbs_output_cpu_transfer" in legacy_sbs


def test_openxr_rgb_depth_shaders_use_consistent_parallax_formula(monkeypatch):
    monkeypatch.chdir(SRC)
    source = (SRC / "xr_viewer" / "d3d11_native_renderer.py").read_text(encoding="utf-8")
    implementation = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    viewer_source = (SRC / "viewer" / "viewer.py").read_text(encoding="utf-8")

    assert "#define roll params.w" in source
    assert "float2 shiftedUv = uv - float2(shift * cos(roll), shift * sin(roll));" in source
    assert "#define parallaxOffset params.x" in source
    assert "float depthResponse = depth - convergence;" in source
    assert "float shift = depthResponse * parallaxOffset * depthStrength;" in source
    assert "depthInv" not in source
    assert "def render_eye(self, swapchain_texture, width, height, eye_index, eye_offset, depth_strength, convergence, mvp, roll=0.0):" in source
    assert "constants[16:20] = np.array([eye_offset, depth_strength, convergence, roll]" in source
    assert "eye_sign * ipd * 0.5" not in source
    assert "self.runtime_eye_srv[eye_index], 0.0, 0.0, 0.0, mvp, roll=0.0" in source
    assert "screen_disparity_uv = max(0.0, runtime_rgb_depth_max_disparity_px) / float(runtime_rgb_depth_render_width)" in implementation
    assert "roll=self.screen_roll" in implementation
    assert "float depth_response = depth - u_convergence;" in viewer_source
    assert "float shift = depth_response;" in viewer_source
    assert "float px = u_eye_offset * shift * u_depth_strength * edge_falloff;" in viewer_source
    assert "float shift_amount = (depth - u_convergence) * u_depth_strength;" in viewer_source
    assert "float shift_amount = (depth - u_convergence);" in viewer_source
    assert "depth_shaped" not in viewer_source
    assert "float depth_inv = -depth;" not in viewer_source
    assert "depth_inv + u_convergence" not in viewer_source


def test_runtime_rgb_depth_config_prefers_structured_shader_uniforms(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    viewer._apply_runtime_rgb_depth_config(
        {
            "openxr_shader_uniforms": {
                "convergence": 9.0,
                "depth_strength": 9.0,
                "max_disparity_px": 9.0,
                "render_size": (9, 9),
            },
            "openxr_convergence": 8.0,
            "resolved_max_disparity_px": 8.0,
            "runtime_output_eye_size": "8x8",
        },
        shader_uniforms={
            "convergence": 0.25,
            "depth_strength": 2.4,
            "max_disparity_px": 18.0,
            "render_size": (1920, 1080),
        },
    )

    assert viewer.convergence == 0.25
    assert viewer._runtime_rgb_depth_depth_strength == 2.4
    assert viewer._runtime_rgb_depth_max_disparity_px == 18.0
    assert viewer._runtime_rgb_depth_render_width == 1920


def test_runtime_rgb_depth_config_keeps_debug_uniform_fallback(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_runtime_eye import CoreRuntimeEyeMixin

    viewer = CoreRuntimeEyeMixin()
    viewer._apply_runtime_rgb_depth_config(
        {
            "openxr_shader_uniforms": {
                "convergence": 0.25,
                "depth_strength": 1.7,
                "max_disparity_px": 18.0,
                "render_size": (1920, 1080),
            },
            "openxr_convergence": 9.0,
            "resolved_max_disparity_px": 9.0,
            "runtime_output_eye_size": "9x9",
        }
    )

    assert viewer.convergence == 0.25
    assert viewer._runtime_rgb_depth_depth_strength == 1.7
    assert viewer._runtime_rgb_depth_max_disparity_px == 18.0
    assert viewer._runtime_rgb_depth_render_width == 1920
