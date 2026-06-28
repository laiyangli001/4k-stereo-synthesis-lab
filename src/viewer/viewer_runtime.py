from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import glfw

from stereo_runtime.frame_stats import FrameStats, LatencyStats, format_viewer_title
from streaming.encoder_profile import EncoderProfile


@dataclass
class ViewerRuntimeConfig:
    capture_mode: str
    monitor_index: int
    convergence: float
    initial_depth_strength: float
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
    encoder_profile: EncoderProfile | None = None


@dataclass
class ViewerRuntimeCallbacks:
    shutdown_is_set: Callable
    breakdown_inc: Callable
    breakdown_add_time: Callable
    log_fps_breakdown: Callable
    rtmp_stream: Callable
    is_window_visible_on_screen: Callable
    set_rtmp_thread: Callable
    update_depth_strength: Callable


def frame_size_from_output(output_frame, *, stream_mode):
    import torch

    if isinstance(output_frame, torch.Tensor):
        if output_frame.ndim == 4:
            width, height = output_frame.shape[3], output_frame.shape[2]
        else:
            width, height = output_frame.shape[2], output_frame.shape[1]
    else:
        width, height = output_frame.shape[1], output_frame.shape[0]
    if not stream_mode:
        height = int(1280 / width * height)
        width = 1280
    return width, height


def frame_size_from_runtime_result(runtime_result, *, stream_mode):
    display_size = getattr(runtime_result, "output_display_size", None)
    if display_size is None:
        debug = getattr(runtime_result, "debug_info", None) or {}
        display_size = _parse_size_text(debug.get("runtime_output_display_size"))
    if display_size is None:
        display_size = frame_size_from_output(runtime_result.sbs, stream_mode=True)
    width, height = display_size
    if not stream_mode:
        height = int(1280 / width * height)
        width = 1280
    return width, height


def _parse_size_text(value):
    if value is None:
        return None
    parts = str(value).strip().lower().split("x", 1)
    if len(parts) != 2:
        return None
    try:
        width = int(parts[0])
        height = int(parts[1])
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height


def start_viewer_streaming(window, config: ViewerRuntimeConfig, callbacks: ViewerRuntimeCallbacks):
    if config.stream_mode == "RTMP":
        if config.os_name == "Windows":
            from viewer.window_control import set_window_to_bottom

            def bottom_loop():
                while True:
                    set_window_to_bottom(window.window)
                    time.sleep(0.01)

            threading.Thread(target=bottom_loop, daemon=True).start()
        rtmp_thread = threading.Thread(
            target=callbacks.rtmp_stream,
            args=(window.window, callbacks.is_window_visible_on_screen),
            daemon=True,
        )
        callbacks.set_rtmp_thread(rtmp_thread)
        rtmp_thread.start()
        print("[Main] RTMP Streamer Started (auto-restart on resize)")
        return None

    if config.stream_mode == "MJPEG":
        from streaming.mjpeg_streamer import MJPEGStreamer

        profile = config.encoder_profile or EncoderProfile(
            codec="mjpeg",
            quality=config.stream_quality,
            target_fps=config.fps,
        )
        streamer = MJPEGStreamer(
            port=config.stream_port,
            profile=profile,
        )
        streamer.start()
        print("[Main] MJPEG Streamer Started")
        return streamer

    print("[Main] Local Viewer Started")
    return None


def _opengl_stereo_window_class():
    from viewer.viewer import StereoWindow

    return StereoWindow


def _select_stereo_window_class(config: ViewerRuntimeConfig):
    if config.os_name == "Darwin" and config.stream_mode != "MJPEG":
        try:
            from viewer.metal_viewer import StereoWindow as MetalStereoWindow
        except Exception as exc:
            print(f"[Main] Metal viewer unavailable, falling back to OpenGL viewer: {exc}")
        else:
            return MetalStereoWindow
    return _opengl_stereo_window_class()


def run_viewer_mode(runtime_q, config: ViewerRuntimeConfig, callbacks: ViewerRuntimeCallbacks, thread_latencies):
    StereoWindow = _select_stereo_window_class(config)

    stats = FrameStats(low_percentile=0.1).start(time.perf_counter())
    latency_stats = LatencyStats()
    runtime_result, capture_start_time = runtime_q.get()
    width, height = frame_size_from_runtime_result(runtime_result, stream_mode=config.stream_mode)

    window_kwargs = dict(
        capture_mode=config.capture_mode,
        monitor_index=config.monitor_index,
        convergence=config.convergence,
        runtime_depth_strength=config.initial_depth_strength,
        on_runtime_depth_strength_change=callbacks.update_depth_strength,
        display_mode=config.display_mode,
        fill_16_9=config.fill_16_9,
        show_fps=config.show_fps,
        use_3d=config.use_3d_monitor,
        fix_aspect=config.fix_viewer_aspect,
        stream_mode=config.stream_mode,
        lossless_scaling=config.lossless_scaling_support,
        specify_display=config.stereo_display_selection,
        stereo_display_index=config.stereo_display_index,
        frame_size=(width, height),
        use_cuda=config.use_cudart,
        cuda_device_id=config.device_id,
        local_vsync=config.local_vsync,
        upscaler=config.upscaler,
        upscaler_sharpness=config.upscaler_sharpness,
    )
    try:
        window = StereoWindow(**window_kwargs)
    except Exception as exc:
        if not getattr(StereoWindow, "uses_metal", False):
            raise
        print(f"[Main] Metal viewer initialization failed, falling back to OpenGL viewer: {exc}")
        window = _opengl_stereo_window_class()(**window_kwargs)

    streamer = start_viewer_streaming(window, config, callbacks)

    render_start_time = time.perf_counter()
    window.update_runtime_frame(runtime_result, stats.current_fps, 0.0)
    render_latency = time.perf_counter() - render_start_time
    total_latency = (render_start_time - capture_start_time) + render_latency
    thread_latencies["render"] = render_latency
    thread_latencies["total"] = total_latency

    next_render_time = time.perf_counter()
    while not glfw.window_should_close(window.window) and not callbacks.shutdown_is_set():
        try:
            runtime_result, capture_start_time = runtime_q.get(timeout=0.001)
            callbacks.breakdown_inc("viewer_get")

            current_time = time.perf_counter()
            total_latency = current_time - capture_start_time
            latency_stats.record(total_latency)

            if stats.record_frame(current_time):
                title_text = format_viewer_title(
                    stats,
                    latency_stats,
                    thread_latencies,
                    render_latency,
                    show_fps=config.show_fps,
                )
                if config.stream_mode == "MJPEG":
                    print(title_text)
                glfw.set_window_title(window.window, f"Stereo Viewer {title_text}")

                update_start_time = time.perf_counter()
                window.update_runtime_frame(
                    runtime_result,
                    stats.current_fps,
                    latency_stats.last_display_latency,
                )
                callbacks.breakdown_add_time("update", time.perf_counter() - update_start_time)
            else:
                update_start_time = time.perf_counter()
                window.update_runtime_frame(runtime_result)
                callbacks.breakdown_add_time("update", time.perf_counter() - update_start_time)

            render_start_time = time.perf_counter()
            if config.stream_mode == "MJPEG":
                frame = window.capture_glfw_image()
                streamer.set_frame(frame)

            render_latency = time.perf_counter() - render_start_time
            thread_latencies["render"] = render_latency
            thread_latencies["total"] = total_latency

        except queue.Empty:
            pass

        now = time.perf_counter()
        if not config.use_3d_monitor and now < next_render_time:
            wait_duration = next_render_time - now
            time.sleep(wait_duration)
            callbacks.breakdown_add_time("wait", wait_duration)
        if not config.use_3d_monitor:
            next_render_time += config.time_sleep

        callbacks.breakdown_inc("loops")
        render_loop_start = time.perf_counter()
        window.render()
        callbacks.breakdown_add_time("render", time.perf_counter() - render_loop_start)
        post_ms = getattr(window, "last_postprocess_ms", 0.0)
        if post_ms:
            callbacks.breakdown_add_time("post", post_ms / 1000.0)
        swap_start = time.perf_counter()
        if not getattr(window, "uses_metal", False):
            glfw.swap_buffers(window.window)
        callbacks.breakdown_add_time("swap", time.perf_counter() - swap_start)
        glfw.poll_events()
        callbacks.log_fps_breakdown()

    glfw.terminate()
    return stats, streamer, window
