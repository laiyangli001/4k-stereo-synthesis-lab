from types import SimpleNamespace

from streaming.encoder_profile import EncoderProfile
from viewer.viewer_runtime import ViewerRuntimeCallbacks, ViewerRuntimeConfig, frame_size_from_output, start_viewer_streaming


def _config(**overrides):
    values = dict(
        capture_mode="monitor",
        monitor_index=0,
        ipd=0.06,
        depth_strength=1.0,
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


def _callbacks():
    return ViewerRuntimeCallbacks(
        shutdown_is_set=lambda: False,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        log_fps_breakdown=lambda *args, **kwargs: None,
        rtmp_stream=lambda *args, **kwargs: None,
        is_window_visible_on_screen=lambda *args, **kwargs: True,
        set_rtmp_thread=lambda thread: None,
    )


def test_frame_size_from_output_scales_local_viewer_width():
    frame = SimpleNamespace(shape=(720, 1280, 3))

    assert frame_size_from_output(frame, stream_mode="") == (1280, 720)
    assert frame_size_from_output(frame, stream_mode="MJPEG") == (1280, 720)


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
