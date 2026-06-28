import inspect
import sys
from dataclasses import fields
from pathlib import Path
from types import ModuleType, SimpleNamespace

from streaming.encoder_profile import EncoderProfile
from viewer.viewer_runtime import (
    ViewerRuntimeCallbacks,
    ViewerRuntimeConfig,
    frame_size_from_output,
    frame_size_from_runtime_result,
    _select_stereo_window_class,
    start_viewer_streaming,
)


def _config(**overrides):
    values = dict(
        capture_mode="monitor",
        monitor_index=0,
        initial_depth_strength=2.5,
        convergence=0.0,
        display_mode="sbs",
        fill_16_9=False,
        show_fps=True,
        use_3d_monitor=False,
        fix_viewer_aspect=False,
        stream_mode="",
        lossless_scaling_support=False,
        stereo_display_selection=False,
        stereo_display_index=0,
        use_cudart=False,
        device_id=0,
        local_vsync=False,
        upscaler="none",
        upscaler_sharpness=0.0,
        os_name="Linux",
        fps=60,
        stream_port=8000,
        stream_quality=80,
        time_sleep=1 / 60,
    )
    values.update(overrides)
    return ViewerRuntimeConfig(**values)


def _module_with_stereo_window(name, stereo_window):
    module = ModuleType(name)
    module.StereoWindow = stereo_window
    return module


def _callbacks():
    return ViewerRuntimeCallbacks(
        shutdown_is_set=lambda: False,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        log_fps_breakdown=lambda *args, **kwargs: None,
        rtmp_stream=lambda *args, **kwargs: None,
        is_window_visible_on_screen=lambda *args, **kwargs: True,
        set_rtmp_thread=lambda thread: None,
        update_depth_strength=lambda value: None,
    )


def test_viewer_runtime_config_does_not_expose_legacy_parallax_fields():
    names = {field.name for field in fields(ViewerRuntimeConfig)}

    assert "ipd" not in names
    assert "initial_depth_strength" in names
    assert "depth_strength" not in names


def test_opengl_viewer_constructor_does_not_expose_legacy_parallax_fields():
    source = Path("src/viewer/viewer.py").read_text(encoding="utf-8")
    init_line = next(line for line in source.splitlines() if "def __init__(self, capture_mode=" in line)

    assert "ipd=" not in init_line
    assert " depth_strength=" not in init_line
    assert "runtime_depth_strength=" in init_line
    assert "show_runtime_depth_strength" in source
    assert "Depth Strength:" in source


def test_frame_size_from_output_scales_local_viewer_width():
    frame = SimpleNamespace(shape=(720, 1280, 3))

    assert frame_size_from_output(frame, stream_mode="") == (1280, 720)
    assert frame_size_from_output(frame, stream_mode="MJPEG") == (1280, 720)


def test_frame_size_from_runtime_result_prefers_structured_display_size():
    result = SimpleNamespace(
        sbs=SimpleNamespace(shape=(720, 1280, 3)),
        output_display_size=(3840, 2160),
        debug_info={"runtime_output_display_size": "1920x1080"},
    )

    assert frame_size_from_runtime_result(result, stream_mode="MJPEG") == (3840, 2160)
    assert frame_size_from_runtime_result(result, stream_mode="") == (1280, 720)


def test_frame_size_from_runtime_result_supports_legacy_debug_display_size():
    result = SimpleNamespace(
        sbs=SimpleNamespace(shape=(720, 1280, 3)),
        debug_info={"runtime_output_display_size": "3840x2160"},
    )

    assert frame_size_from_runtime_result(result, stream_mode="MJPEG") == (3840, 2160)
    assert frame_size_from_runtime_result(result, stream_mode="") == (1280, 720)


def test_frame_size_from_runtime_result_falls_back_to_sbs_shape():
    result = SimpleNamespace(
        sbs=SimpleNamespace(shape=(720, 1280, 3)),
        debug_info={"runtime_output_display_size": "invalid"},
    )

    assert frame_size_from_runtime_result(result, stream_mode="MJPEG") == (1280, 720)


def test_select_stereo_window_uses_metal_only_for_darwin_non_mjpeg(monkeypatch):
    class OpenGLWindow:
        pass

    class MetalWindow:
        uses_metal = True

    monkeypatch.setitem(sys.modules, "viewer.viewer", _module_with_stereo_window("viewer.viewer", OpenGLWindow))
    monkeypatch.setitem(
        sys.modules,
        "viewer.metal_viewer",
        _module_with_stereo_window("viewer.metal_viewer", MetalWindow),
    )

    assert _select_stereo_window_class(_config(os_name="Darwin", stream_mode="")) is MetalWindow
    assert _select_stereo_window_class(_config(os_name="Darwin", stream_mode="MJPEG")) is OpenGLWindow
    assert _select_stereo_window_class(_config(os_name="Windows", stream_mode="")) is OpenGLWindow


def test_select_stereo_window_falls_back_when_metal_import_fails(monkeypatch, capsys):
    class OpenGLWindow:
        pass

    monkeypatch.setitem(sys.modules, "viewer.viewer", _module_with_stereo_window("viewer.viewer", OpenGLWindow))
    monkeypatch.setitem(sys.modules, "viewer.metal_viewer", None)

    assert _select_stereo_window_class(_config(os_name="Darwin", stream_mode="")) is OpenGLWindow
    assert "Metal viewer unavailable" in capsys.readouterr().out


def test_metal_viewer_exposes_current_runtime_contract():
    from viewer import metal_viewer
    from viewer.metal_viewer import StereoWindow

    signature = inspect.signature(StereoWindow)
    update_source = inspect.getsource(StereoWindow.update_runtime_frame)
    render_source = inspect.getsource(StereoWindow.render)

    assert "ipd" not in signature.parameters
    assert "depth_strength" not in signature.parameters
    assert "depth_ratio" not in signature.parameters
    assert "kwargs" not in signature.parameters
    assert hasattr(StereoWindow, "update_runtime_frame")
    assert not hasattr(StereoWindow, "update_frame")
    assert "runtime_result.sbs" in update_source
    assert "self.depth_strength" not in update_source
    assert "self.depth_ratio" not in update_source
    assert "_runtime_direct_output = True" in update_source
    assert "runtime_direct = bool(self._runtime_direct_output)" in render_source
    assert "uniform_bytes = self._uniform_bytes(viewport, 0, 0.0)" in render_source
    assert "self.depth_strength" not in render_source
    assert "self.depth_ratio" not in render_source
    assert "if runtime_direct:" in render_source
    assert render_source.index("if runtime_direct:") < render_source.index(
        'elif self.display_mode in ("Full-SBS", "Half-SBS", "Half-TAB", "Full-TAB")'
    )
    assert "displaced_uv(uv, u.parallaxOffset" in metal_viewer.METAL_SHADER
    assert "float shift = (d - u.convergence) * eye * u.depthStrength;" in metal_viewer.METAL_SHADER
    assert "uv.x - shift" in metal_viewer.METAL_SHADER
    assert "u.convergence - d" not in metal_viewer.METAL_SHADER
    assert "eyeOffset" not in metal_viewer.METAL_SHADER


def test_start_viewer_streaming_returns_none_for_local_mode(capsys):
    window = SimpleNamespace(window="handle")

    streamer = start_viewer_streaming(window, _config(stream_mode=""), _callbacks())

    assert streamer is None
    assert "Local Viewer Started" in capsys.readouterr().out


def test_start_viewer_streaming_passes_encoder_profile_to_mjpeg(monkeypatch):
    created = {}

    class FakeStreamer:
        def __init__(self, **kwargs):
            created.update(kwargs)

        def start(self):
            created["started"] = True

    monkeypatch.setattr("streaming.mjpeg_streamer.MJPEGStreamer", FakeStreamer)
    profile = EncoderProfile(codec="mjpeg", quality=72, target_fps=24, resize_width=640, resize_height=360)

    streamer = start_viewer_streaming(
        SimpleNamespace(window="handle"),
        _config(stream_mode="MJPEG", encoder_profile=profile),
        _callbacks(),
    )

    assert streamer is not None
    assert created["port"] == 8000
    assert created["profile"] is profile
    assert created["started"] is True
