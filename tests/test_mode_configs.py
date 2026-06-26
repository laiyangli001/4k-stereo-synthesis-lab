from app_runtime.mode_configs import (
    build_legacy_stream_config,
    build_openxr_runtime_config,
    build_viewer_runtime_config,
)


def test_build_viewer_runtime_config_maps_expected_fields():
    config = build_viewer_runtime_config(
        capture_mode="monitor",
        monitor_index=1,
        ipd=0.06,
        depth_strength=1.2,
        convergence=0.1,
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
        upscaler_sharpness=0.5,
        os_name="Windows",
        fps=60,
        stream_port=8000,
        stream_quality=80,
        time_sleep=1 / 60,
    )

    assert config.stream_mode == "MJPEG"
    assert config.use_cudart is True
    assert config.stereo_display_index == 2
    assert config.encoder_profile.codec == "mjpeg"
    assert config.encoder_profile.quality == 80
    assert config.encoder_profile.target_fps == 60


def test_build_openxr_runtime_config_maps_expected_fields():
    config = build_openxr_runtime_config(
        ipd=0.06,
        depth_strength=1.2,
        convergence=0.1,
        fps=72,
        show_fps=False,
        controller_model="controller",
        environment_model="none",
        show_preview_window=True,
        capture_mode="window",
        monitor_index=0,
    )

    assert config.fps == 72
    assert config.environment_model == "none"
    assert config.show_preview_window is True


def test_build_legacy_stream_config_maps_expected_fields():
    config = build_legacy_stream_config(
        stream_port=9000,
        fps=30,
        stream_quality=70,
        time_sleep=1 / 30,
    )

    assert config.stream_port == 9000
    assert config.fps == 30
    assert config.stream_quality == 70
    assert config.encoder_profile.codec == "mjpeg"
    assert config.encoder_profile.quality == 70
    assert config.encoder_profile.target_fps == 30
