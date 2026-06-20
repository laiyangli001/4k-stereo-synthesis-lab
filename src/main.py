# main.py
import threading
import time
import signal
import sys
import subprocess
import os

from utils import OS_NAME, OUTPUT_RESOLUTION, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, CONVERGENCE, RUN_MODE, STEREOMIX_DEVICE, STREAM_KEY, AUDIO_DELAY, CRF, DEVICE_INFO, DEVICE, CACHE_PATH, settings, shutdown_event, SHOW_FPS
from capture import capture_frame_to_rgb, prepare_rgb_for_stereo_runtime
from capture.session import CaptureSessionLoop
from stereo_runtime.pipeline import RuntimePipelineLoop
from utils.queue_utils import clear_nonblocking, drain_latest, put_latest
from app_support.cleanup import cleanup_resources
from app_support.app_runner import build_app_mode_callbacks, build_current_app_mode_settings, run_app_mode
from app_support.runtime_context import build_capture_callbacks, build_runtime_pipeline_context, create_runtime_context
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
BASE_DIR = context.base_dir
USE_CUDART = context.use_cudart
TIME_SLEEP = context.time_sleep
OPENXR_RUNTIME_DIRECT = context.openxr_runtime_direct
raw_q = context.raw_q
runtime_q = context.runtime_q
runtime_config = context.runtime_config
stereo_runtime = context.stereo_runtime
stereo_auto_enabled = context.stereo_auto_enabled
stereo_active_preset = context.stereo_active_preset
stereo_still_duration_s = context.stereo_still_duration_s
stereo_last_auto_ts = context.stereo_last_auto_ts
stereo_hot_reloader = context.stereo_hot_reloader
stereo_warmup_tracker = context.stereo_warmup_tracker
stereo_runtime_logger = context.stereo_runtime_logger
openxr_state = context.openxr_state
openxr_render_active = openxr_state.render_active
openxr_source_active = openxr_state.source_active
openxr_wait_idle_active = openxr_state.wait_idle_active
openxr_bootstrap_done = openxr_state.bootstrap_done
capture_control = None
capture_session = None
FPS_BREAKDOWN_LOG = context.fps_breakdown_log
_source_health = context.source_health
_fps_breakdown = context.fps_breakdown
capture_config = context.capture_config
thread_latencies = context.thread_latencies

def _stereo_warmup_key(rgb_frame):
    return stereo_warmup_tracker.key_for_frame(rgb_frame)


def _warmup_stereo_once_for_frame(rgb_frame):
    stereo_warmup_tracker.warmup_once_for_frame(rgb_frame)

def _breakdown_inc(name, amount=1):
    _fps_breakdown.inc(name, amount)


def _breakdown_add_time(name, seconds):
    _fps_breakdown.add_time(name, seconds)


def _breakdown_add_runtime_timing(runtime_result):
    _fps_breakdown.add_runtime_timing(runtime_result)


def _log_fps_breakdown(now=None):
    _fps_breakdown.log(now)


def _source_stat_inc(name, amount=1, **values):
    _source_health.inc(name, amount, **values)


def _source_stat_set(**values):
    _source_health.set(**values)


def _log_source_health(now=None, force=False):
    _source_health.log(now, force)


def _openxr_source_paused():
    return openxr_state.source_paused()


def _stop_active_capture_session():
    global capture_control, capture_session
    stopped = False
    try:
        if capture_control is not None:
            capture_control.stop()
            stopped = True
    except Exception:
        pass
    try:
        if not stopped and capture_session is not None and hasattr(capture_session, "stop"):
            capture_session.stop()
            stopped = True
    except Exception:
        pass
    return stopped


def _on_openxr_hard_idle_enter():
    _queue_clear_nonblocking(raw_q)
    _queue_clear_nonblocking(runtime_q)
    _stop_active_capture_session()


def _openxr_hard_idle_active():
    return openxr_state.hard_idle_active(on_enter=_on_openxr_hard_idle_enter)


def _queue_put_latest(q, item):
    put_latest(q, item)


def _queue_clear_nonblocking(q):
    clear_nonblocking(q)


def _queue_drain_latest(q, first_item):
    def on_drop():
        _source_stat_inc("raw_dropped_stale")
        _breakdown_inc("raw_dropped_stale")

    return drain_latest(q, first_item, on_drop=on_drop)


def _update_openxr_runtime_config(*, ipd=None, depth_ratio=None, convergence=None, screen_roll=None):
    openxr_state.update_runtime_config(
        ipd=ipd,
        depth_ratio=depth_ratio,
        convergence=convergence,
        screen_roll=screen_roll,
    )


def _current_openxr_render_config():
    return openxr_state.current_render_config(stereo_runtime)


def _apply_stereo_hot_reload_if_needed():
    stereo_hot_reloader.apply_if_needed(
        runtime=stereo_runtime,
        active_preset=stereo_active_preset,
        on_openxr_config_update=_update_openxr_runtime_config,
        on_mode_log=_log_stereo_runtime_mode_once,
    )

def _log_stereo_runtime_mode(reason, decision=None, samples=None, motion=None):
    stereo_runtime_logger.log_mode(reason, decision=decision, samples=samples, motion=motion)


def _log_stereo_runtime_mode_once(reason="active"):
    stereo_runtime_logger.log_mode_once(reason)


def _log_fast_plus_fused_runtime_state(runtime_result):
    stereo_runtime_logger.log_fast_plus_fused_runtime_state(runtime_result)



def _capture_session_update(session, control):
    global capture_control, capture_session
    capture_session = session
    capture_control = control


def _put_raw_latest(item):
    was_full = raw_q.full()
    _queue_put_latest(raw_q, item)
    return was_full


def capture_loop():
    callbacks = build_capture_callbacks(
        raw_q=raw_q,
        shutdown_event=shutdown_event,
        queue_clear=_queue_clear_nonblocking,
        inc_source_stat=_source_stat_inc,
        inc_breakdown=_breakdown_inc,
        put_raw_latest=_put_raw_latest,
        is_paused=_openxr_source_paused,
        is_hard_idle=_openxr_hard_idle_active,
        on_session_update=_capture_session_update,
        on_tick=_log_source_health,
    )
    CaptureSessionLoop(capture_config, callbacks).run(shutdown_event)

# Combined capture-to-runtime processing thread (replaces process_loop and runtime_loop)
def _set_runtime_preprocess_backend(backend):
    if FPS_BREAKDOWN_LOG:
        _fps_breakdown.set_latest("rt_preprocess_backend", backend)


def process_runtime_loop():
    pipeline_context = build_runtime_pipeline_context(
        shutdown_event=shutdown_event,
        app_context=context,
        run_mode=RUN_MODE,
        device=DEVICE,
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=prepare_rgb_for_stereo_runtime,
        current_openxr_render_config=_current_openxr_render_config,
        is_hard_idle=_openxr_hard_idle_active,
        is_source_paused=_openxr_source_paused,
        log_source_health=_log_source_health,
        source_stat_inc=_source_stat_inc,
        breakdown_inc=_breakdown_inc,
        breakdown_add_time=_breakdown_add_time,
        breakdown_add_runtime_timing=_breakdown_add_runtime_timing,
        set_preprocess_backend=_set_runtime_preprocess_backend,
        queue_clear=_queue_clear_nonblocking,
        queue_drain_latest=_queue_drain_latest,
        queue_put_latest=_queue_put_latest,
        log_stereo_runtime_mode_once=_log_stereo_runtime_mode_once,
        apply_stereo_hot_reload_if_needed=_apply_stereo_hot_reload_if_needed,
        warmup_stereo_once_for_frame=_warmup_stereo_once_for_frame,
        log_fast_plus_fused_runtime_state=_log_fast_plus_fused_runtime_state,
    )
    RuntimePipelineLoop(pipeline_context).run()

def cleanup_all_resources():
    """Global cleanup function"""
    cleanup_resources(
        global_processes=global_processes,
        stop_capture=_stop_active_capture_session,
        streamer=globals().get("streamer"),
        queues=[raw_q, runtime_q],
        queue_timeout=TIME_SLEEP,
        rtmp_thread=globals().get("rtmp_thread"),
    )
def signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals"""
    print(f"\n[Signal] Received signal {signum}, shutting down...")
    shutdown_event.set()
    cleanup_all_resources()
    sys.exit(0)

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)
if OS_NAME != "Windows":
    signal.signal(signal.SIGQUIT, signal_handler)


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
            use_cudart=USE_CUDART,
            time_sleep=TIME_SLEEP,
        )
        app_callbacks = build_app_mode_callbacks(
            shutdown_is_set=shutdown_event.is_set,
            breakdown_inc=_breakdown_inc,
            breakdown_add_time=_breakdown_add_time,
            log_fps_breakdown=_log_fps_breakdown,
            is_window_visible_on_screen=is_window_visible_on_screen,
            set_rtmp_thread=_set_rtmp_thread,
            rtmp_stream=rtmp_stream,
            update_openxr_runtime_config=_update_openxr_runtime_config,
            render_active_event=openxr_render_active,
            source_active_event=openxr_source_active,
            idle_active_event=openxr_wait_idle_active,
            render_active_clear=openxr_render_active.clear,
            source_active_set=openxr_source_active.set,
            wait_idle_clear=openxr_wait_idle_active.clear,
            bootstrap_done_set=openxr_bootstrap_done.set,
        )
        result = run_app_mode(
            mode,
            runtime_q=runtime_q,
            thread_latencies=thread_latencies,
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
