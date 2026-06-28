from __future__ import annotations

from streaming.encoder_profile import EncoderProfile
from streaming.legacy_runtime import LegacyStreamConfig
from viewer.viewer_runtime import ViewerRuntimeConfig
from xr_viewer.openxr_runtime import OpenXRRuntimeConfig


def build_viewer_runtime_config(
    *,
    capture_mode,
    monitor_index,
    depth_strength,
    convergence,
    display_mode,
    fill_16_9,
    show_fps,
    use_3d_monitor,
    fix_viewer_aspect,
    stream_mode,
    lossless_scaling_support,
    stereo_display_selection,
    stereo_display_index,
    use_cudart,
    device_id,
    local_vsync,
    upscaler,
    upscaler_sharpness,
    os_name,
    fps,
    stream_port,
    stream_quality,
    time_sleep,
):
    encoder_profile = EncoderProfile(
        codec="mjpeg",
        quality=stream_quality,
        target_fps=fps,
    )
    return ViewerRuntimeConfig(
        capture_mode=capture_mode,
        monitor_index=monitor_index,
        convergence=convergence,
        initial_depth_strength=depth_strength,
        display_mode=display_mode,
        fill_16_9=fill_16_9,
        show_fps=show_fps,
        use_3d_monitor=use_3d_monitor,
        fix_viewer_aspect=fix_viewer_aspect,
        stream_mode=stream_mode,
        lossless_scaling_support=lossless_scaling_support,
        stereo_display_selection=stereo_display_selection,
        stereo_display_index=stereo_display_index,
        use_cudart=use_cudart,
        device_id=device_id,
        local_vsync=local_vsync,
        upscaler=upscaler,
        upscaler_sharpness=upscaler_sharpness,
        os_name=os_name,
        fps=fps,
        stream_port=stream_port,
        stream_quality=stream_quality,
        time_sleep=time_sleep,
        encoder_profile=encoder_profile,
    )


def build_openxr_runtime_config(
    *,
    depth_strength,
    convergence,
    fps,
    show_fps,
    controller_model,
    environment_model,
    show_preview_window,
    capture_mode,
    monitor_index,
):
    return OpenXRRuntimeConfig(
        depth_strength=depth_strength,
        convergence=convergence,
        fps=fps,
        show_fps=show_fps,
        controller_model=controller_model,
        environment_model=environment_model,
        show_preview_window=show_preview_window,
        capture_mode=capture_mode,
        monitor_index=monitor_index,
    )


def build_legacy_stream_config(*, stream_port, fps, stream_quality, time_sleep):
    encoder_profile = EncoderProfile(
        codec="mjpeg",
        quality=stream_quality,
        target_fps=fps,
    )
    return LegacyStreamConfig(
        stream_port=stream_port,
        fps=fps,
        stream_quality=stream_quality,
        time_sleep=time_sleep,
        encoder_profile=encoder_profile,
    )
