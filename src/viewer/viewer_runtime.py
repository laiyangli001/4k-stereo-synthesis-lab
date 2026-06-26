from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

import glfw

from stereo_runtime.frame_stats import FrameStats, LatencyStats, format_viewer_title


@dataclass
class ViewerRuntimeConfig:
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


@dataclass
class ViewerRuntimeCallbacks:
    shutdown_is_set: Callable
    breakdown_inc: Callable
    breakdown_add_time: Callable
    log_fps_breakdown: Callable
    rtmp_stream: Callable
    is_window_visible_on_screen: Callable
    set_rtmp_thread: Callable


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


def start_viewer_streaming(window, config: ViewerRuntimeConfig, callbacks: ViewerRuntimeCallbacks):
    if config.stream_mode == "RTMP":
        if config.os_name == "Windows":
            from utils import set_window_to_bottom

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

        streamer = MJPEGStreamer(
            port=config.stream_port,
            fps=config.fps,
            quality=config.stream_quality,
        )
        streamer.start()
        print("[Main] MJPEG Streamer Started")
        return streamer

    print("[Main] Local Viewer Started")
    return None


def run_viewer_mode(runtime_q, config: ViewerRuntimeConfig, callbacks: ViewerRuntimeCallbacks, thread_latencies):
    from viewer.viewer import StereoWindow

    stats = FrameStats(low_percentile=0.1).start(time.perf_counter())
    latency_stats = LatencyStats()
    runtime_result, capture_start_time = runtime_q.get()
    width, height = frame_size_from_output(runtime_result.sbs, stream_mode=config.stream_mode)

    window = StereoWindow(
        capture_mode=config.capture_mode,
        monitor_index=config.monitor_index,
        ipd=config.ipd,
        depth_strength=config.depth_strength,
        convergence=config.convergence,
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
        glfw.swap_buffers(window.window)
        callbacks.breakdown_add_time("swap", time.perf_counter() - swap_start)
        glfw.poll_events()
        callbacks.log_fps_breakdown()

    glfw.terminate()
    return stats, streamer, window
