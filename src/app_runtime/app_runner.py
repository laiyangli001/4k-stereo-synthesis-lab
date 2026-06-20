from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app_runtime.mode_configs import (
    build_legacy_stream_config,
    build_openxr_runtime_config,
    build_viewer_runtime_config,
)
from stereo_runtime.frame_stats import FrameStats
from streaming.legacy_runtime import LegacyStreamCallbacks, run_legacy_stream_mode
from viewer.viewer_runtime import ViewerRuntimeCallbacks, run_viewer_mode
from xr_viewer.openxr_runtime import OpenXRRuntimeCallbacks, run_openxr_mode


@dataclass
class AppModeSettings:
    capture_mode: str
    monitor_index: int
    ipd: float
    depth_strength: float
    convergence: float
    display_mode: str
    fill_16_9: bool
    show_fps: bool
    use_3d_monitor: bool
    fix_viewer_aspect: bool
    stream_mode: str
    lossless_scaling_support: bool
    stereo_display_selection: bool
    stereo_display_index: int
    use_cudart: bool
    device_id: int
    local_vsync: bool
    upscaler: str
    upscaler_sharpness: float
    os_name: str
    fps: int
    stream_port: int
    stream_quality: int
    time_sleep: float
    controller_model: str
    environment_model: str
    xr_preview_window: bool


@dataclass
class AppModeCallbacks:
    shutdown_is_set: Callable
    breakdown_inc: Callable
    breakdown_add_time: Callable
    log_fps_breakdown: Callable
    is_window_visible_on_screen: Callable
    set_rtmp_thread: Callable
    rtmp_stream: Callable
    update_openxr_runtime_config: Callable
    render_active_event: object
    source_active_event: object
    idle_active_event: object
    render_active_clear: Callable
    source_active_set: Callable
    wait_idle_clear: Callable
    bootstrap_done_set: Callable




def build_app_mode_settings(
    *,
    capture_mode,
    monitor_index,
    ipd,
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
    controller_model,
    environment_model,
    xr_preview_window,
):
    return AppModeSettings(
        capture_mode=capture_mode,
        monitor_index=monitor_index,
        ipd=ipd,
        depth_strength=depth_strength,
        convergence=convergence,
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
        controller_model=controller_model,
        environment_model=environment_model,
        xr_preview_window=xr_preview_window,
    )




def build_current_app_mode_settings(*, use_cudart, time_sleep):
    from utils import (
        CAPTURE_MODE,
        CONTROLLER_MODEL,
        CONVERGENCE,
        DEPTH_STRENGTH,
        DEVICE_ID,
        DISPLAY_MODE,
        ENVIRONMENT_MODEL,
        FILL_16_9,
        FIX_VIEWER_ASPECT,
        FPS,
        IPD,
        LOCAL_VSYNC,
        LOSSLESS_SCALING_SUPPORT,
        MONITOR_INDEX,
        OS_NAME,
        SHOW_FPS,
        STEREO_DISPLAY_INDEX,
        STEREO_DISPLAY_SELECTION,
        STREAM_MODE,
        STREAM_PORT,
        STREAM_QUALITY,
        UPSCALER,
        UPSCALER_SHARPNESS,
        USE_3D_MONITOR,
        XR_PREVIEW_WINDOW,
    )

    return build_app_mode_settings(
        capture_mode=CAPTURE_MODE,
        monitor_index=MONITOR_INDEX,
        ipd=IPD,
        depth_strength=DEPTH_STRENGTH,
        convergence=CONVERGENCE,
        display_mode=DISPLAY_MODE,
        fill_16_9=FILL_16_9,
        show_fps=SHOW_FPS,
        use_3d_monitor=USE_3D_MONITOR,
        fix_viewer_aspect=FIX_VIEWER_ASPECT,
        stream_mode=STREAM_MODE,
        lossless_scaling_support=LOSSLESS_SCALING_SUPPORT,
        stereo_display_selection=STEREO_DISPLAY_SELECTION,
        stereo_display_index=STEREO_DISPLAY_INDEX,
        use_cudart=use_cudart,
        device_id=DEVICE_ID,
        local_vsync=LOCAL_VSYNC,
        upscaler=UPSCALER,
        upscaler_sharpness=UPSCALER_SHARPNESS,
        os_name=OS_NAME,
        fps=FPS,
        stream_port=STREAM_PORT,
        stream_quality=STREAM_QUALITY,
        time_sleep=time_sleep,
        controller_model=CONTROLLER_MODEL,
        environment_model=ENVIRONMENT_MODEL,
        xr_preview_window=XR_PREVIEW_WINDOW,
    )

def build_app_mode_callbacks(
    *,
    shutdown_is_set,
    breakdown_inc,
    breakdown_add_time,
    log_fps_breakdown,
    is_window_visible_on_screen,
    set_rtmp_thread,
    rtmp_stream,
    update_openxr_runtime_config,
    render_active_event,
    source_active_event,
    idle_active_event,
    render_active_clear,
    source_active_set,
    wait_idle_clear,
    bootstrap_done_set,
):
    return AppModeCallbacks(
        shutdown_is_set=shutdown_is_set,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=breakdown_add_time,
        log_fps_breakdown=log_fps_breakdown,
        is_window_visible_on_screen=is_window_visible_on_screen,
        set_rtmp_thread=set_rtmp_thread,
        rtmp_stream=rtmp_stream,
        update_openxr_runtime_config=update_openxr_runtime_config,
        render_active_event=render_active_event,
        source_active_event=source_active_event,
        idle_active_event=idle_active_event,
        render_active_clear=render_active_clear,
        source_active_set=source_active_set,
        wait_idle_clear=wait_idle_clear,
        bootstrap_done_set=bootstrap_done_set,
    )

@dataclass
class AppRunResult:
    stats: FrameStats
    streamer: object = None
    window: object = None


def run_app_mode(mode, *, runtime_q, thread_latencies, settings: AppModeSettings, callbacks: AppModeCallbacks):
    stats = FrameStats(low_percentile=0.1).start(time.perf_counter())
    if mode == "Viewer":
        viewer_config = build_viewer_runtime_config(
            capture_mode=settings.capture_mode,
            monitor_index=settings.monitor_index,
            ipd=settings.ipd,
            depth_strength=settings.depth_strength,
            convergence=settings.convergence,
            display_mode=settings.display_mode,
            fill_16_9=settings.fill_16_9,
            show_fps=settings.show_fps,
            use_3d_monitor=settings.use_3d_monitor,
            fix_viewer_aspect=settings.fix_viewer_aspect,
            stream_mode=settings.stream_mode,
            lossless_scaling_support=settings.lossless_scaling_support,
            stereo_display_selection=settings.stereo_display_selection,
            stereo_display_index=settings.stereo_display_index,
            use_cudart=settings.use_cudart,
            device_id=settings.device_id,
            local_vsync=settings.local_vsync,
            upscaler=settings.upscaler,
            upscaler_sharpness=settings.upscaler_sharpness,
            os_name=settings.os_name,
            fps=settings.fps,
            stream_port=settings.stream_port,
            stream_quality=settings.stream_quality,
            time_sleep=settings.time_sleep,
        )
        viewer_callbacks = ViewerRuntimeCallbacks(
            shutdown_is_set=callbacks.shutdown_is_set,
            breakdown_inc=callbacks.breakdown_inc,
            breakdown_add_time=callbacks.breakdown_add_time,
            log_fps_breakdown=callbacks.log_fps_breakdown,
            rtmp_stream=callbacks.rtmp_stream,
            is_window_visible_on_screen=callbacks.is_window_visible_on_screen,
            set_rtmp_thread=callbacks.set_rtmp_thread,
        )
        stats, streamer, window = run_viewer_mode(
            runtime_q,
            viewer_config,
            viewer_callbacks,
            thread_latencies,
        )
        return AppRunResult(stats=stats, streamer=streamer, window=window)

    if mode == "OpenXR":
        openxr_config = build_openxr_runtime_config(
            ipd=settings.ipd,
            depth_strength=settings.depth_strength,
            convergence=settings.convergence,
            fps=settings.fps,
            show_fps=settings.show_fps,
            controller_model=settings.controller_model,
            environment_model=settings.environment_model,
            show_preview_window=settings.xr_preview_window,
            capture_mode=settings.capture_mode,
            monitor_index=settings.monitor_index,
        )
        openxr_callbacks = OpenXRRuntimeCallbacks(
            update_runtime_config=callbacks.update_openxr_runtime_config,
            render_active_event=callbacks.render_active_event,
            source_active_event=callbacks.source_active_event,
            idle_active_event=callbacks.idle_active_event,
            render_active_clear=callbacks.render_active_clear,
            source_active_set=callbacks.source_active_set,
            wait_idle_clear=callbacks.wait_idle_clear,
            bootstrap_done_set=callbacks.bootstrap_done_set,
        )
        return AppRunResult(
            stats=stats,
            window=run_openxr_mode(runtime_q, openxr_config, openxr_callbacks),
        )

    legacy_config = build_legacy_stream_config(
        stream_port=settings.stream_port,
        fps=settings.fps,
        stream_quality=settings.stream_quality,
        time_sleep=settings.time_sleep,
    )
    legacy_callbacks = LegacyStreamCallbacks(
        shutdown_is_set=callbacks.shutdown_is_set,
    )
    return AppRunResult(
        stats=stats,
        streamer=run_legacy_stream_mode(runtime_q, legacy_config, legacy_callbacks, stats),
    )
