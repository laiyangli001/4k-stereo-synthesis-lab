# main.py
import threading
import time
import subprocess
import os

from utils import OS_NAME, OUTPUT_RESOLUTION, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, CONVERGENCE, RUN_MODE, STEREOMIX_DEVICE, STREAM_KEY, AUDIO_DELAY, CRF, DEVICE_INFO, DEVICE, CACHE_PATH, settings, shutdown_event, SHOW_FPS
from capture import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime
from capture.session import CaptureSessionLoop
from stereo_runtime.pipeline import RuntimePipelineLoop
from app_runtime.app_runner import build_app_mode_callbacks, build_current_app_mode_settings, run_app_mode
from app_runtime.shutdown import build_cleanup_handler, build_signal_handler, register_signal_handlers
from app_runtime.runtime_context import build_capture_callbacks, build_runtime_pipeline_context, create_runtime_context
from app_runtime.runtime_callbacks import RuntimeCallbacks
from streaming.rtmp import global_processes, rtmp_stream
from viewer.window_utils import is_window_visible_on_screen, list_windows

context = create_runtime_context(
    file_path=__file__,
    settings=settings,
    cache_path=CACHE_PATH,
    device=DEVICE,
    device_info=DEVICE_INFO,
    output_resolution=OUTPUT_RESOLUTION,
    fps=FPS,
    window_title=WINDOW_TITLE,
    capture_mode=CAPTURE_MODE,
    monitor_index=MONITOR_INDEX,
    capture_tool=CAPTURE_TOOL,
    os_name=OS_NAME,
    run_mode=RUN_MODE,
    ipd=IPD,
    depth_strength=DEPTH_STRENGTH,
    convergence=CONVERGENCE,
)
runtime_callbacks = RuntimeCallbacks(context)

def capture_loop():
    callbacks = build_capture_callbacks(
        raw_q=context.raw_q,
        shutdown_event=shutdown_event,
        queue_clear=runtime_callbacks.queue_clear_nonblocking,
        inc_source_stat=runtime_callbacks.source_stat_inc,
        inc_breakdown=runtime_callbacks.breakdown_inc,
        put_raw_latest=runtime_callbacks.put_raw_latest,
        is_paused=runtime_callbacks.openxr_source_paused,
        is_hard_idle=runtime_callbacks.openxr_hard_idle_active,
        on_session_update=runtime_callbacks.capture_session_update,
        on_tick=runtime_callbacks.log_source_health,
    )
    CaptureSessionLoop(context.capture_config, callbacks).run(shutdown_event)

# Combined capture-to-runtime processing thread (replaces process_loop and runtime_loop)
def process_runtime_loop():
    pipeline_context = build_runtime_pipeline_context(
        shutdown_event=shutdown_event,
        app_context=context,
        run_mode=RUN_MODE,
        device=DEVICE,
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=prepare_rgb_for_stereo_runtime,
        current_openxr_render_config=runtime_callbacks.current_openxr_render_config,
        is_hard_idle=runtime_callbacks.openxr_hard_idle_active,
        is_source_paused=runtime_callbacks.openxr_source_paused,
        log_source_health=runtime_callbacks.log_source_health,
        source_stat_inc=runtime_callbacks.source_stat_inc,
        breakdown_inc=runtime_callbacks.breakdown_inc,
        breakdown_add_time=runtime_callbacks.breakdown_add_time,
        breakdown_add_runtime_timing=runtime_callbacks.breakdown_add_runtime_timing,
        set_preprocess_backend=runtime_callbacks.set_runtime_preprocess_backend,
        queue_clear=runtime_callbacks.queue_clear_nonblocking,
        queue_drain_latest=runtime_callbacks.queue_drain_latest,
        queue_put_latest=runtime_callbacks.queue_put_latest,
        log_stereo_runtime_mode_once=runtime_callbacks.log_stereo_runtime_mode_once,
        apply_stereo_hot_reload_if_needed=runtime_callbacks.apply_stereo_hot_reload_if_needed,
        warmup_stereo_once_for_frame=runtime_callbacks.warmup_stereo_once_for_frame,
        log_fast_plus_fused_runtime_state=runtime_callbacks.log_fast_plus_fused_runtime_state,
    )
    RuntimePipelineLoop(pipeline_context).run()

cleanup_all_resources = build_cleanup_handler(
    global_processes=global_processes,
    stop_capture=runtime_callbacks.stop_active_capture_session,
    get_streamer=lambda: globals().get("streamer"),
    queues=[context.raw_q, context.runtime_q],
    queue_timeout=context.time_sleep,
    get_rtmp_thread=lambda: globals().get("rtmp_thread"),
)
signal_handler = build_signal_handler(
    shutdown_event=shutdown_event,
    cleanup_all_resources=cleanup_all_resources,
)
register_signal_handlers(os_name=OS_NAME, signal_handler=signal_handler)

def _set_rtmp_thread(thread):
    global rtmp_thread
    rtmp_thread = thread

def main(mode="Viewer"):
    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    # Replace separate process_loop and depth_loop with combined thread
    threading.Thread(target=process_runtime_loop, daemon=True).start()

    stats = None

    try:
        app_settings = build_current_app_mode_settings(
            use_cudart=context.use_cudart,
            time_sleep=context.time_sleep,
        )
        app_callbacks = build_app_mode_callbacks(
            shutdown_is_set=shutdown_event.is_set,
            breakdown_inc=runtime_callbacks.breakdown_inc,
            breakdown_add_time=runtime_callbacks.breakdown_add_time,
            log_fps_breakdown=runtime_callbacks.log_fps_breakdown,
            is_window_visible_on_screen=is_window_visible_on_screen,
            set_rtmp_thread=_set_rtmp_thread,
            rtmp_stream=rtmp_stream,
            update_openxr_runtime_config=runtime_callbacks.update_openxr_runtime_config,
            render_active_event=context.openxr_state.render_active,
            source_active_event=context.openxr_state.source_active,
            idle_active_event=context.openxr_state.wait_idle_active,
            render_active_clear=context.openxr_state.render_active.clear,
            source_active_set=context.openxr_state.source_active.set,
            wait_idle_clear=context.openxr_state.wait_idle_active.clear,
            bootstrap_done_set=context.openxr_state.bootstrap_done.set,
        )
        result = run_app_mode(
            mode,
            runtime_q=context.runtime_q,
            thread_latencies=context.thread_latencies,
            settings=app_settings,
            callbacks=app_callbacks,
        )
        stats = result.stats
        globals()["streamer"] = result.streamer
        globals()["window"] = result.window

    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received, shutting down...")
    # except Exception as e:
    #     print(f"[Main] Error: {e}")
    finally:
        # Ensure cleanup happens
        shutdown_event.set()
        cleanup_all_resources()

        if SHOW_FPS and stats is not None:
            print(f"Overall Average FPS: {stats.overall_avg_fps(time.perf_counter()):.2f}")
            if stats.fps_values:
                print(f"Recent Average FPS: {stats.avg_fps:.1f}")
                print(f"Recent 1% Low Average FPS: {stats.low_fps_avg:.1f}")
        print(f"[Main] Stopped")

if __name__ == "__main__":
    main(mode=RUN_MODE)
