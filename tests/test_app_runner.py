from app_runtime.app_runner import AppModeCallbacks, AppModeSettings, build_app_mode_callbacks, build_app_mode_settings, build_current_app_mode_settings, run_app_mode


def _settings():
    return AppModeSettings(
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
        controller_model="controller",
        environment_model="none",
        xr_preview_window=False,
    )


def _callbacks():
    return AppModeCallbacks(
        shutdown_is_set=lambda: True,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        log_fps_breakdown=lambda *args, **kwargs: None,
        is_window_visible_on_screen=lambda *args, **kwargs: True,
        set_rtmp_thread=lambda thread: None,
        rtmp_stream=lambda *args, **kwargs: None,
        update_openxr_runtime_config=lambda **kwargs: None,
        render_active_event=object(),
        source_active_event=object(),
        idle_active_event=object(),
        render_active_clear=lambda: None,
        source_active_set=lambda: None,
        wait_idle_clear=lambda: None,
        bootstrap_done_set=lambda: None,
    )


def test_run_app_mode_dispatches_legacy(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "app_runtime.app_runner.run_legacy_stream_mode",
        lambda runtime_q, config, callbacks, stats: calls.append((config, stats)) or "streamer",
    )

    result = run_app_mode(
        "Legacy",
        runtime_q=object(),
        thread_latencies={},
        settings=_settings(),
        callbacks=_callbacks(),
    )

    assert result.streamer == "streamer"
    assert result.window is None
    assert calls
    assert calls[0][0].stream_port == 8000


def test_build_app_mode_settings_maps_core_fields():
    settings = build_app_mode_settings(
        capture_mode="monitor",
        monitor_index=1,
        ipd=0.06,
        depth_strength=1.1,
        convergence=0.2,
        display_mode="sbs",
        fill_16_9=True,
        show_fps=True,
        use_3d_monitor=False,
        fix_viewer_aspect=True,
        stream_mode="MJPEG",
        lossless_scaling_support=False,
        stereo_display_selection=True,
        stereo_display_index=2,
        use_cudart=True,
        device_id=0,
        local_vsync=False,
        upscaler="none",
        upscaler_sharpness=0.4,
        os_name="Windows",
        fps=60,
        stream_port=8000,
        stream_quality=80,
        time_sleep=1 / 60,
        controller_model="controller",
        environment_model="none",
        xr_preview_window=False,
    )

    assert settings.stream_mode == "MJPEG"
    assert settings.use_cudart is True
    assert settings.stereo_display_index == 2


def test_build_app_mode_callbacks_maps_callables():
    def shutdown():
        return True

    callbacks = build_app_mode_callbacks(
        shutdown_is_set=shutdown,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        log_fps_breakdown=lambda *args, **kwargs: None,
        is_window_visible_on_screen=lambda *args, **kwargs: True,
        set_rtmp_thread=lambda thread: None,
        rtmp_stream=lambda *args, **kwargs: None,
        update_openxr_runtime_config=lambda **kwargs: None,
        render_active_event=object(),
        source_active_event=object(),
        idle_active_event=object(),
        render_active_clear=lambda: None,
        source_active_set=lambda: None,
        wait_idle_clear=lambda: None,
        bootstrap_done_set=lambda: None,
    )

    assert callbacks.shutdown_is_set is shutdown
    assert callbacks.is_window_visible_on_screen() is True

def test_build_current_app_mode_settings_reads_utils_lazily(monkeypatch):
    import sys
    import types

    fake_utils = types.SimpleNamespace(
        CAPTURE_MODE="monitor",
        CONTROLLER_MODEL="controller",
        CONVERGENCE=0.0,
        DEPTH_STRENGTH=1.0,
        DEVICE_ID=0,
        DISPLAY_MODE="sbs",
        ENVIRONMENT_MODEL="none",
        FILL_16_9=False,
        FIX_VIEWER_ASPECT=False,
        FPS=60,
        IPD=0.06,
        LOCAL_VSYNC=False,
        LOSSLESS_SCALING_SUPPORT=False,
        MONITOR_INDEX=0,
        OS_NAME="Linux",
        SHOW_FPS=True,
        STEREO_DISPLAY_INDEX=0,
        STEREO_DISPLAY_SELECTION=False,
        STREAM_MODE="",
        STREAM_PORT=8000,
        STREAM_QUALITY=80,
        UPSCALER="none",
        UPSCALER_SHARPNESS=0.0,
        USE_3D_MONITOR=False,
        XR_PREVIEW_WINDOW=False,
    )
    monkeypatch.setitem(sys.modules, "utils", fake_utils)

    settings = build_current_app_mode_settings(use_cudart=True, time_sleep=1 / 60)

    assert settings.use_cudart is True
    assert settings.stream_port == 8000
    assert settings.environment_model == "none"