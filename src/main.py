# main.py
import threading
import queue
import glfw
import time
import signal
import sys
import subprocess
import os
from collections import deque

from utils import OS_NAME, OUTPUT_RESOLUTION, DISPLAY_MODE, CAPTURE_MODE, CAPTURE_TOOL, MONITOR_INDEX, SHOW_FPS, FPS, WINDOW_TITLE, IPD, DEPTH_STRENGTH, CONVERGENCE, RUN_MODE, STREAM_MODE, STREAM_PORT, STREAM_QUALITY, STEREOMIX_DEVICE, STREAM_KEY, AUDIO_DELAY, CRF, LOSSLESS_SCALING_SUPPORT, USE_3D_MONITOR, FILL_16_9, LOCAL_VSYNC, UPSCALER, UPSCALER_SHARPNESS, FIX_VIEWER_ASPECT, CAPTURE_MODE, STEREO_DISPLAY_SELECTION, STEREO_DISPLAY_INDEX, shutdown_event, DEVICE_ID, DEVICE_INFO, DEVICE, CONTROLLER_MODEL, ENVIRONMENT_MODEL, XR_PREVIEW_WINDOW, CACHE_PATH, settings
from capture import CaptureConfig, capture_frame_to_rgb, create_capture_runner, prepare_rgb_for_stereo_runtime
from stereo_runtime import AutoModeRuntime, AutoModeSignals, OpenXRRenderConfig, StereoRuntime, runtime_config_from_d2s_settings, stereo_config_for_preset
from stereo_runtime.adapter import stereo_config_from_runtime
from stereo_runtime.adapter import preset_for_runtime_mode
from stereo_runtime.presets import normalize_preset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RTMP_DIR = os.path.join(BASE_DIR, "streaming", "rtmp")

if "CUDA" in DEVICE_INFO and "ZLUDA" not in DEVICE_INFO:
    USE_CUDART = True
else:
    USE_CUDART = False
# Global process references
global_processes = {
    'ffmpeg': None,
    'rtmp_server': None
}

# Track current stream size + restart lock
current_stream_size = None
ffmpeg_restart_lock = threading.Lock()

# Use precise frame interval
TIME_SLEEP = 1.0 / FPS

# Queues with size=1 (latest-frame-only logic)
raw_q = queue.Queue(maxsize=1)
runtime_q = queue.Queue(maxsize=1)
runtime_config = runtime_config_from_d2s_settings(
    settings,
    cache_dir=CACHE_PATH,
    device=str(DEVICE),
    depth_only=False,
)
stereo_runtime = StereoRuntime(runtime_config)

def _initial_stereo_preset_state(config):
    raw_preset = config.stereo_preset
    if raw_preset is not None:
        preset = normalize_preset(raw_preset)
        return preset == "auto", preset
    if config.mode == "auto":
        return True, "auto"
    return False, preset_for_runtime_mode(config.mode)

stereo_auto_enabled, stereo_active_preset = _initial_stereo_preset_state(runtime_config)
if not stereo_auto_enabled:
    stereo_runtime.configure_stereo(stereo_config_from_runtime(runtime_config), reset_temporal=True)
stereo_auto_runtime = AutoModeRuntime(initial_preset="cinema" if stereo_active_preset == "auto" else stereo_active_preset)
stereo_last_logged_mode_state = None
stereo_last_logged_fused_state = None
stereo_last_motion_frame = None
stereo_still_duration_s = 0.0
stereo_last_auto_ts = time.perf_counter()
stereo_signal_lock = threading.Lock()
stereo_signal_state = {
    "gpu_3d_util": 0.0,
    "gpu_video_decode_util": 0.0,
    "input_activity": 0.0,
    "idle_seconds": 0.0,
    "audio_active": False,
    "maximized": False,
    "foreground_process": "",
    "fullscreen": False,
}
stereo_signal_thread_started = False
openxr_render_active = threading.Event()
openxr_source_active = threading.Event()
openxr_wait_idle_active = threading.Event()
openxr_bootstrap_done = threading.Event()
openxr_runtime_config_lock = threading.Lock()
openxr_runtime_config_state = {
    "ipd": float(IPD),
    "depth_ratio": float(DEPTH_STRENGTH),
    "convergence": float(CONVERGENCE),
    "screen_roll": 0.0,
}
OPENXR_RUNTIME_DIRECT = str(
    os.environ.get("D2S_OPENXR_RUNTIME_DIRECT", "1") or "1"
).strip().lower() not in ("0", "false", "no", "off")
_openxr_source_pause_notice_lock = threading.Lock()
_openxr_source_pause_noticed = None
_openxr_wait_idle_notice_lock = threading.Lock()
_openxr_wait_idle_noticed = None
capture_control = None
capture_session = None
_source_stats_lock = threading.Lock()
_source_stats = {
    "capture_frames": 0,
    "capture_errors": 0,
    "capture_dropped_paused": 0,
    "raw_put": 0,
    "raw_get": 0,
    "raw_queue_empty": 0,
    "runtime_frames": 0,
    "runtime_none": 0,
    "runtime_errors": 0,
    "runtime_dropped_paused": 0,
    "last_capture_ts": 0.0,
    "last_raw_get_ts": 0.0,
    "last_runtime_ts": 0.0,
    "last_process_latency": 0.0,
    "last_runtime_latency": 0.0,
    "last_error": "",
}
_last_source_health_log = 0.0
SOURCE_HEALTH_LOG = str(
    os.environ.get("D2S_SOURCE_HEALTH_LOG", os.environ.get("D2S_OPENXR_DEBUG", "0")) or "0"
).strip().lower() in ("1", "true", "yes", "on")
FPS_BREAKDOWN_LOG = str(
    os.environ.get("D2S_FPS_BREAKDOWN", "0") or "0"
).strip().lower() in ("1", "true", "yes", "on")
_BREAKDOWN_LATEST_KEYS = {
    "rt_backend",
    "rt_depth_backend",
    "rt_output_format",
    "rt_output_dtype",
    "rt_output_pack",
    "rt_sbs_backend",
    "rt_occ_backend",
    "rt_fill_backend",
    "rt_depth_total_ms",
    "rt_depth_model_ms",
    "rt_synthesis_ms",
    "rt_total_ms",
}

_breakdown_lock = threading.Lock()
_breakdown_stats = {
    "capture": 0,
    "raw_get": 0,
    "runtime": 0,
    "viewer_get": 0,
    "loops": 0,
    "update_ms": 0.0,
    "render_ms": 0.0,
    "swap_ms": 0.0,
    "wait_ms": 0.0,
    "update_count": 0,
    "render_count": 0,
    "swap_count": 0,
    "wait_count": 0,
}
_breakdown_last = time.perf_counter()


def _breakdown_inc(name, amount=1):
    if not FPS_BREAKDOWN_LOG:
        return
    with _breakdown_lock:
        _breakdown_stats[name] = _breakdown_stats.get(name, 0) + amount


def _breakdown_add_time(name, seconds):
    if not FPS_BREAKDOWN_LOG:
        return
    with _breakdown_lock:
        _breakdown_stats[f"{name}_ms"] = _breakdown_stats.get(f"{name}_ms", 0.0) + seconds * 1000.0
        _breakdown_stats[f"{name}_count"] = _breakdown_stats.get(f"{name}_count", 0) + 1



def _breakdown_add_runtime_timing(runtime_result):
    if not FPS_BREAKDOWN_LOG:
        return
    timing = getattr(runtime_result, "timing", None) or {}
    debug = getattr(runtime_result, "debug_info", None) or {}
    with _breakdown_lock:
        for key in ("depth_total_ms", "depth_model_ms", "synthesis_ms", "total_ms"):
            value = timing.get(key)
            if value is not None:
                _breakdown_stats[f"rt_{key}"] = float(value)
        _breakdown_stats["rt_backend"] = str(debug.get("backend", "unknown"))
        _breakdown_stats["rt_depth_backend"] = str(debug.get("runtime_depth_backend", "unknown"))
        _breakdown_stats["rt_output_format"] = str(debug.get("runtime_output_format", "unknown"))
        _breakdown_stats["rt_output_dtype"] = str(debug.get("runtime_output_dtype", "unknown"))
        _breakdown_stats["rt_output_pack"] = str(debug.get("runtime_output_pack_backend", "n/a"))
        if "sbs_backend" in debug:
            _breakdown_stats["rt_sbs_backend"] = str(debug.get("sbs_backend"))
        if "occlusion_mask_backend" in debug:
            _breakdown_stats["rt_occ_backend"] = str(debug.get("occlusion_mask_backend"))
        if "hole_fill_backend" in debug:
            _breakdown_stats["rt_fill_backend"] = str(debug.get("hole_fill_backend"))
        if "fast_plus_fused_backend" in debug:
            _breakdown_stats["rt_fast_plus_fused_backend"] = str(debug.get("fast_plus_fused_backend"))
        if "fast_plus_fused_skip" in debug:
            _breakdown_stats["rt_fast_plus_fused_skip"] = str(debug.get("fast_plus_fused_skip"))
        if "fast_plus_fused_temporal_bypass" in debug:
            _breakdown_stats["rt_fast_plus_fused_temporal_bypass"] = str(debug.get("fast_plus_fused_temporal_bypass"))

def _log_fps_breakdown(now=None):
    global _breakdown_last
    if not FPS_BREAKDOWN_LOG:
        return
    now = time.perf_counter() if now is None else now
    elapsed = now - _breakdown_last
    if elapsed < 1.0:
        return
    with _breakdown_lock:
        stats = dict(_breakdown_stats)
        for key in list(_breakdown_stats.keys()):
            if key in _BREAKDOWN_LATEST_KEYS:
                continue
            _breakdown_stats[key] = 0.0 if key.endswith("_ms") else 0
        _breakdown_last = now

    def rate(name):
        return stats.get(name, 0) / elapsed

    def avg_ms(name):
        count = stats.get(f"{name}_count", 0)
        return stats.get(f"{name}_ms", 0.0) / count if count else 0.0

    print(
        "[FPSBreakdown] "
        f"target={FPS}Hz "
        f"cap={rate('capture'):.1f} raw={rate('raw_get'):.1f} "
        f"overwrite={rate('raw_overwritten'):.1f} drain_drop={rate('raw_dropped_stale'):.1f} "
        f"runtime={rate('runtime'):.1f} viewer_get={rate('viewer_get'):.1f} "
        f"loop={rate('loops'):.1f} "
        f"update={avg_ms('update'):.2f}ms "
        f"render={avg_ms('render'):.2f}ms "
        f"post={avg_ms('post'):.2f}ms "
        f"swap={avg_ms('swap'):.2f}ms "
        f"wait={avg_ms('wait'):.2f}ms "
        f"rt_loop={avg_ms('rt_loop'):.2f}ms "
        f"rt_cap2rgb={avg_ms('rt_cap2rgb'):.2f}ms "
        f"rt_prepare={avg_ms('rt_prepare'):.2f}ms "
        f"pre={stats.get('rt_preprocess_backend', 'unknown')} "
        f"rt_call={avg_ms('rt_call'):.2f}ms "
        f"rt_put={avg_ms('rt_put'):.2f}ms "
        f"rt_backend={stats.get('rt_backend', 'unknown')} "
        f"rt_depth={stats.get('rt_depth_total_ms', 0.0):.2f}ms "
        f"rt_model={stats.get('rt_depth_model_ms', 0.0):.2f}ms "
        f"rt_synth={stats.get('rt_synthesis_ms', 0.0):.2f}ms "
        f"rt_total={stats.get('rt_total_ms', 0.0):.2f}ms "
        f"rt_depth_backend={stats.get('rt_depth_backend', 'unknown')} "
        f"rt_out={stats.get('rt_output_dtype', 'unknown')} "
        f"rt_pack={stats.get('rt_output_pack', 'n/a')} "
        f"rt_sbs={stats.get('rt_sbs_backend', 'unknown')} "
        f"rt_occ={stats.get('rt_occ_backend', 'n/a')} "
        f"rt_fill={stats.get('rt_fill_backend', 'n/a')} "
        f"rt_fused={stats.get('rt_fast_plus_fused_backend', 'n/a')} "
        f"rt_fused_skip={stats.get('rt_fast_plus_fused_skip', 'n/a')} "
        f"rt_fused_temporal_bypass={stats.get('rt_fast_plus_fused_temporal_bypass', 'n/a')}",
        flush=True,
    )


def _source_stat_inc(name, amount=1, **values):
    with _source_stats_lock:
        _source_stats[name] = _source_stats.get(name, 0) + amount
        _source_stats.update(values)


def _source_stat_set(**values):
    with _source_stats_lock:
        _source_stats.update(values)


def _safe_qsize(q):
    try:
        return q.qsize()
    except Exception:
        return -1


def _format_age(seconds):
    if seconds < 0:
        return "n/a"
    return f"{seconds:.2f}s"


def _log_source_health(now=None, force=False):
    global _last_source_health_log
    if RUN_MODE != "OpenXR":
        return
    if not SOURCE_HEALTH_LOG:
        return
    now = time.perf_counter() if now is None else now
    if not force and (now - _last_source_health_log) < 5.0:
        return
    _last_source_health_log = now
    with _source_stats_lock:
        stats = dict(_source_stats)

    last_capture = stats.get("last_capture_ts", 0.0)
    last_runtime = stats.get("last_runtime_ts", 0.0)
    raw_age = now - last_capture if last_capture > 0.0 else -1.0
    runtime_age = now - last_runtime if last_runtime > 0.0 else -1.0
    last_error = stats.get("last_error") or "none"
    print(
        "[Main] Source health: "
        f"cap={stats.get('capture_frames', 0)} raw_put={stats.get('raw_put', 0)} "
        f"raw_get={stats.get('raw_get', 0)} runtime={stats.get('runtime_frames', 0)} "
        f"empty={stats.get('raw_queue_empty', 0)} none={stats.get('runtime_none', 0)} "
        f"cap_err={stats.get('capture_errors', 0)} runtime_err={stats.get('runtime_errors', 0)} "
        f"raw_age={_format_age(raw_age)} runtime_age={_format_age(runtime_age)} "
        f"raw_q={_safe_qsize(raw_q)} runtime_q={_safe_qsize(runtime_q)} "
        f"resize_ms={stats.get('last_process_latency', 0.0) * 1000.0:.1f} "
        f"runtime_ms={stats.get('last_runtime_latency', 0.0) * 1000.0:.1f} "
        f"source={openxr_source_active.is_set()} render={openxr_render_active.is_set()} "
        f"idle={openxr_wait_idle_active.is_set()} err={last_error}",
        flush=True,
    )


def _openxr_source_paused():
    paused = RUN_MODE == "OpenXR" and openxr_bootstrap_done.is_set() and not openxr_source_active.is_set()
    global _openxr_source_pause_noticed
    with _openxr_source_pause_notice_lock:
        if _openxr_source_pause_noticed is not paused:
            _openxr_source_pause_noticed = paused
            if paused:
                print("[Main] OpenXR source inference paused")
            else:
                print("[Main] OpenXR source inference resumed")
    return paused


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


def _openxr_hard_idle_active():
    idle = RUN_MODE == "OpenXR" and openxr_bootstrap_done.is_set() and openxr_wait_idle_active.is_set()
    global _openxr_wait_idle_noticed
    with _openxr_wait_idle_notice_lock:
        if _openxr_wait_idle_noticed is not idle:
            _openxr_wait_idle_noticed = idle
            if idle:
                _queue_clear_nonblocking(raw_q)
                _queue_clear_nonblocking(runtime_q)
                _stop_active_capture_session()
                print("[Main] OpenXR hard idle entered")
            else:
                print("[Main] OpenXR hard idle exited")
    return idle


def _queue_put_latest(q, item):
    """Keep only the newest item without blocking producer threads."""
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def _queue_clear_nonblocking(q):
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return


def _queue_drain_latest(q, first_item):
    """Drop stale queued items and return the newest available frame."""
    latest = first_item
    while True:
        try:
            latest = q.get_nowait()
            _source_stat_inc("raw_dropped_stale")
            _breakdown_inc("raw_dropped_stale")
        except queue.Empty:
            return latest


def _update_openxr_runtime_config(*, ipd=None, depth_ratio=None, convergence=None, screen_roll=None):
    with openxr_runtime_config_lock:
        if ipd is not None:
            openxr_runtime_config_state["ipd"] = float(ipd)
        if depth_ratio is not None:
            openxr_runtime_config_state["depth_ratio"] = float(depth_ratio)
        if convergence is not None:
            openxr_runtime_config_state["convergence"] = float(convergence)
        if screen_roll is not None:
            openxr_runtime_config_state["screen_roll"] = float(screen_roll)


def _current_openxr_render_config():
    with openxr_runtime_config_lock:
        state = dict(openxr_runtime_config_state)
    return OpenXRRenderConfig(
        ipd=state["ipd"],
        ipd_mm=stereo_runtime.stereo_config.ipd_mm,
        stereo_scale=stereo_runtime.stereo_config.stereo_scale,
        depth_strength=0.1 * state["depth_ratio"],
        convergence=state["convergence"],
        max_shift_ratio=stereo_runtime.stereo_config.max_shift_ratio,
        screen_roll=state["screen_roll"],
    )


def _runtime_stereo_overrides():
    config = stereo_runtime.config
    return {
        "backend": config.stereo_quality,
        "depth_strength": config.depth_strength,
        "convergence": config.convergence,
        "ipd": config.ipd,
        "ipd_mm": config.ipd_mm,
        "stereo_scale": config.stereo_scale,
        "max_shift_ratio": config.max_shift_ratio,
        "foreground_scale": config.foreground_scale,
        "depth_antialias_strength": config.depth_antialias_strength,
        "edge_threshold": config.edge_threshold,
        "edge_dilation": config.edge_dilation,
        "screen_edge_mask_suppression": config.screen_edge_mask_suppression,
        "cross_eyed": config.cross_eyed,
        "anaglyph_method": config.anaglyph_method,
        "fused": config.fused,
    }


def _clamp01(value):
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def _sample_runtime_motion(rgb_frame):
    global stereo_last_motion_frame
    try:
        import torch

        frame = rgb_frame.detach()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3:
            return 0.0
        if frame.shape[0] in (3, 4):
            frame = frame[:3]
        else:
            frame = frame[..., :3].permute(2, 0, 1)
        frame = torch.nn.functional.interpolate(
            frame.unsqueeze(0).float(),
            size=(32, 32),
            mode="bilinear",
            align_corners=False,
        )[0]
        if stereo_last_motion_frame is None:
            stereo_last_motion_frame = frame.detach()
            return 0.0
        motion = float((frame - stereo_last_motion_frame).abs().mean().item())
        stereo_last_motion_frame = frame.detach()
        return _clamp01(motion * 4.0)
    except Exception:
        return 0.0


def _query_process_name(pid):
    if OS_NAME != "Windows" or not pid:
        return ""
    try:
        import ctypes
        from ctypes import wintypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return ""
        try:
            size = wintypes.DWORD(32768)
            buffer = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return os.path.basename(buffer.value)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return ""
    return ""


def _sample_window_input_context():
    result = {
        "input_activity": 0.0,
        "idle_seconds": 0.0,
        "maximized": False,
        "foreground_process": "",
        "fullscreen": False,
    }
    if OS_NAME != "Windows":
        return result
    try:
        import ctypes
        from ctypes import wintypes

        class LASTINPUTINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(info)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            tick = ctypes.windll.kernel32.GetTickCount()
            idle_s = max(0.0, float((tick - info.dwTime) & 0xFFFFFFFF) / 1000.0)
            result["idle_seconds"] = idle_s
            if idle_s < 0.25:
                result["input_activity"] = 1.0
            elif idle_s < 1.0:
                result["input_activity"] = 0.7
            elif idle_s < 3.0:
                result["input_activity"] = 0.35

        try:
            import win32api
            import win32gui
            import win32process

            hwnd = win32gui.GetForegroundWindow()
            if hwnd:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                result["foreground_process"] = _query_process_name(pid)
                result["maximized"] = bool(win32gui.IsZoomed(hwnd))
                rect = win32gui.GetWindowRect(hwnd)
                monitor = win32api.MonitorFromWindow(hwnd, 2)
                monitor_info = win32api.GetMonitorInfo(monitor)
                mx1, my1, mx2, my2 = monitor_info.get("Monitor", monitor_info.get("Work"))
                result["fullscreen"] = (
                    rect[0] <= mx1 + 2
                    and rect[1] <= my1 + 2
                    and rect[2] >= mx2 - 2
                    and rect[3] >= my2 - 2
                )
        except Exception:
            pass
    except Exception:
        pass
    return result


def _sample_gpu_engine_utilization():
    if OS_NAME != "Windows":
        return {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0}
    command = (
        "$samples=(Get-Counter '\\GPU Engine(*)\\Utilization Percentage' -ErrorAction Stop).CounterSamples; "
        "$samples | Select-Object Path,CookedValue | ConvertTo-Json -Compress"
    )
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0}
        import json

        rows = json.loads(proc.stdout)
        if isinstance(rows, dict):
            rows = [rows]
        gpu_3d = 0.0
        video_decode = 0.0
        for row in rows:
            path = str(row.get("Path", "")).lower()
            value = float(row.get("CookedValue", 0.0)) / 100.0
            if "engtype_3d" in path:
                gpu_3d += value
            elif "engtype_videodecode" in path or "engtype_video decode" in path:
                video_decode += value
        return {
            "gpu_3d_util": _clamp01(gpu_3d),
            "gpu_video_decode_util": _clamp01(video_decode),
        }
    except Exception:
        return {"gpu_3d_util": 0.0, "gpu_video_decode_util": 0.0}


def _auto_signal_sampler_loop():
    while not shutdown_event.is_set():
        samples = {}
        samples.update(_sample_gpu_engine_utilization())
        samples.update(_sample_window_input_context())
        process = samples.get("foreground_process", "").lower()
        samples["audio_active"] = bool(samples.get("gpu_video_decode_util", 0.0) > 0.05 or any(
            token in process for token in ("vlc", "mpv", "potplayer", "player", "chrome", "edge", "firefox")
        ))
        with stereo_signal_lock:
            stereo_signal_state.update(samples)
        time.sleep(2.0)


def _ensure_auto_signal_sampler():
    global stereo_signal_thread_started
    if not stereo_auto_enabled or stereo_signal_thread_started:
        return
    stereo_signal_thread_started = True
    threading.Thread(target=_auto_signal_sampler_loop, daemon=True, name="StereoAutoSignalSampler").start()


def _log_stereo_runtime_mode(reason, decision=None, samples=None, motion=None):
    config = stereo_runtime.stereo_config
    runtime_cfg = stereo_runtime.config
    preset = stereo_active_preset or runtime_cfg.stereo_preset or runtime_cfg.mode
    fused_candidate = (
        config.backend == "fast_plus"
        and config.output_format == "half_sbs"
        and str(os.environ.get("D2S_RUNTIME_OUTPUT_UINT8", "0") or "0").strip().lower() in {"1", "true", "yes", "on"}
        and str(os.environ.get("D2S_FAST_PLUS_FUSED", "1") or "1").strip().lower() in {"1", "true", "yes", "on"}
    )
    parts = [
        f"[Main] Stereo mode {reason}:",
        f"auto={'on' if stereo_auto_enabled else 'off'}",
        f"preset={preset}",
        f"synthetic_view={config.backend}",
        f"quality_setting={runtime_cfg.stereo_quality}",
        f"output={config.output_format}",
        f"hole_fill={config.hole_fill}",
        f"temporal={config.temporal}",
        f"fast_plus_fused_candidate={int(fused_candidate)}",
        f"runtime_uint8={os.environ.get('D2S_RUNTIME_OUTPUT_UINT8', '0')}",
    ]
    if decision is not None:
        parts.append(f"decision={decision.preset}")
        parts.append(f"reason={decision.reason}")
    if motion is not None:
        parts.append(f"motion={float(motion):.3f}")
    if samples:
        parts.append(f"gpu3d={float(samples.get('gpu_3d_util', 0.0)):.2f}")
        parts.append(f"video={float(samples.get('gpu_video_decode_util', 0.0)):.2f}")
        parts.append(f"input={float(samples.get('input_activity', 0.0)):.2f}")
        parts.append(f"idle={float(samples.get('idle_seconds', 0.0)):.1f}s")
    print(" ".join(parts), flush=True)


def _log_stereo_runtime_mode_once(reason="active"):
    global stereo_last_logged_mode_state
    config = stereo_runtime.stereo_config
    runtime_cfg = stereo_runtime.config
    state = (
        stereo_active_preset,
        config.backend,
        runtime_cfg.stereo_quality,
        config.output_format,
        config.hole_fill,
        config.temporal,
    )
    if state == stereo_last_logged_mode_state:
        return
    stereo_last_logged_mode_state = state
    _log_stereo_runtime_mode(reason)


def _log_fast_plus_fused_runtime_state(runtime_result):
    global stereo_last_logged_fused_state
    debug = getattr(runtime_result, "debug_info", None) or {}
    state = (
        str(debug.get("backend", "unknown")),
        str(debug.get("runtime_output_format", "unknown")),
        str(debug.get("runtime_output_dtype", "unknown")),
        str(debug.get("runtime_output_pack_backend", "n/a")),
        str(debug.get("fast_plus_fused_backend", "n/a")),
        str(debug.get("fast_plus_fused_skip", "n/a")),
        str(debug.get("fast_plus_fused_temporal_bypass", "n/a")),
    )
    if state == stereo_last_logged_fused_state:
        return
    stereo_last_logged_fused_state = state
    print(
        "[Main] Stereo runtime output:"
        f" backend={state[0]}"
        f" output={state[1]}"
        f" dtype={state[2]}"
        f" pack={state[3]}"
        f" fast_plus_fused={state[4]}"
        f" fast_plus_fused_skip={state[5]}"
        f" fast_plus_fused_temporal_bypass={state[6]}",
        flush=True,
    )


def _update_auto_stereo_mode(rgb_frame):
    global stereo_active_preset, stereo_still_duration_s, stereo_last_auto_ts
    if not stereo_auto_enabled:
        return
    now = time.perf_counter()
    dt_s = max(0.0, now - stereo_last_auto_ts)
    stereo_last_auto_ts = now
    motion = _sample_runtime_motion(rgb_frame)
    stereo_still_duration_s = stereo_still_duration_s + dt_s if motion < 0.03 else 0.0
    with stereo_signal_lock:
        samples = dict(stereo_signal_state)
    decision = stereo_auto_runtime.update(
        AutoModeSignals(
            frame_motion_score=motion,
            still_duration_s=stereo_still_duration_s,
            gpu_3d_util=samples.get("gpu_3d_util", 0.0),
            gpu_video_decode_util=samples.get("gpu_video_decode_util", 0.0),
            input_activity=samples.get("input_activity", 0.0),
            idle_seconds=samples.get("idle_seconds", 0.0),
            audio_active=bool(samples.get("audio_active", False)),
            maximized=bool(samples.get("maximized", False)),
            foreground_process=str(samples.get("foreground_process", "")),
            fullscreen=bool(samples.get("fullscreen", False) or RUN_MODE in {"OpenXR", "3D Monitor"}),
            openxr_active=RUN_MODE == "OpenXR",
            latency_pressure=1.0 if FPS >= 90 else 0.0,
            target_fps=float(FPS),
        ),
        dt_s=dt_s,
    )
    if decision.preset == stereo_active_preset:
        return
    stereo_active_preset = decision.preset
    stereo_runtime.configure_stereo(
        stereo_config_for_preset(
            decision.preset,
            output_format=stereo_runtime.config.output_format,
            overrides=_runtime_stereo_overrides(),
        ),
        reset_temporal=True,
    )
    _log_stereo_runtime_mode("auto-switch", decision=decision, samples=samples, motion=motion)


def _runtime_output_to_numpy(frame):
    import numpy as np
    import torch

    if isinstance(frame, torch.Tensor):
        frame = frame.detach()
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim == 3 and frame.shape[0] in (3, 4):
            frame = frame[:3].permute(1, 2, 0)
        elif frame.ndim == 3 and frame.shape[-1] >= 3:
            frame = frame[..., :3]
        else:
            raise RuntimeError(f"Unsupported runtime output shape: {tuple(frame.shape)}")
        if frame.is_floating_point():
            frame = frame.clamp(0.0, 1.0).mul(255.0)
        return frame.contiguous().to(torch.uint8).cpu().numpy()

    frame_np = np.asarray(frame)
    if frame_np.ndim == 4:
        frame_np = frame_np[0]
    if frame_np.ndim == 3 and frame_np.shape[0] in (3, 4):
        frame_np = np.transpose(frame_np[:3], (1, 2, 0))
    elif frame_np.ndim == 3 and frame_np.shape[-1] >= 3:
        frame_np = frame_np[..., :3]
    else:
        raise RuntimeError(f"Unsupported runtime output shape: {tuple(frame_np.shape)}")
    if np.issubdtype(frame_np.dtype, np.floating):
        frame_np = np.clip(frame_np, 0.0, 1.0) * 255.0
    return frame_np.astype("uint8", copy=False)

# Thread latency tracking dictionaries
thread_latencies = {
    'capture': 0.0,
    'resize': 0.0,
    'runtime': 0.0,
    'render': 0.0,
    'total': 0.0
}

# Initialize capture
capture_config = CaptureConfig(
    output_resolution=OUTPUT_RESOLUTION,
    fps=FPS,
    window_title=WINDOW_TITLE,
    capture_mode=CAPTURE_MODE,
    monitor_index=MONITOR_INDEX,
    capture_tool=CAPTURE_TOOL,
    os_name=OS_NAME,
)


def _capture_session_update(session, control):
    global capture_control, capture_session
    capture_session = session
    capture_control = control


def _capture_paused(reason):
    _queue_clear_nonblocking(raw_q)
    if reason == "paused":
        _source_stat_inc("capture_dropped_paused")


def _capture_frame_arrived(frame_raw, size, capture_start_time):
    _source_stat_inc("capture_frames", last_capture_ts=capture_start_time)
    _breakdown_inc("capture")
    if shutdown_event.is_set():
        return
    if raw_q.full():
        _source_stat_inc("raw_overwritten")
        _breakdown_inc("raw_overwritten")
    _queue_put_latest(raw_q, (frame_raw, size, capture_start_time))
    _source_stat_inc("raw_put")


def _capture_error(exc):
    _source_stat_inc(
        "capture_errors",
        last_error=f"capture_loop {type(exc).__name__}: {exc}",
    )
    print(f"[capture_loop] Capture session error: {type(exc).__name__}: {exc}", flush=True)


def _capture_closed():
    if not shutdown_event.is_set():
        print("[capture_loop] Capture session closed")


def capture_loop():
    runner = create_capture_runner(capture_config)
    runner.run(
        shutdown_event=shutdown_event,
        on_frame=_capture_frame_arrived,
        on_error=_capture_error,
        on_closed=_capture_closed,
        is_paused=_openxr_source_paused,
        is_hard_idle=_openxr_hard_idle_active,
        on_paused=_capture_paused,
        on_session_update=_capture_session_update,
        on_tick=_log_source_health,
    )

# Combined capture-to-runtime processing thread (replaces process_loop and runtime_loop)
def process_runtime_loop():
    while not shutdown_event.is_set():
        _log_source_health()
        try:
            if shutdown_event.is_set():
                break
            if _openxr_hard_idle_active():
                _queue_clear_nonblocking(raw_q)
                _queue_clear_nonblocking(runtime_q)
                time.sleep(0.1)
                continue
            if _openxr_source_paused():
                _queue_clear_nonblocking(raw_q)
                _queue_clear_nonblocking(runtime_q)
                _source_stat_inc("runtime_dropped_paused")
                time.sleep(0.01)
                continue

            # Wait briefly for a frame, then drain stale frames so inference always works on latest input.
            frame_raw, size, capture_start_time = _queue_drain_latest(
                raw_q,
                raw_q.get(timeout=min(TIME_SLEEP, 0.01)),
            )
            _source_stat_inc("raw_get", last_raw_get_ts=time.perf_counter())
            _breakdown_inc("raw_get")

            if _openxr_source_paused():
                _queue_clear_nonblocking(raw_q)
                _queue_clear_nonblocking(runtime_q)
                _source_stat_inc("runtime_dropped_paused")
                time.sleep(0.01)
                continue

            loop_start_time = time.perf_counter()

            # Process: resize / color conversion
            process_start_time = time.perf_counter()
            frame_rgb = capture_frame_to_rgb(
                frame_raw,
                size,
                device=DEVICE,
                use_torch=USE_CUDART,
                output="tensor",
            )
            _breakdown_add_time("rt_cap2rgb", time.perf_counter() - process_start_time)
            if FPS_BREAKDOWN_LOG:
                with _breakdown_lock:
                    _breakdown_stats["rt_preprocess_backend"] = str(getattr(frame_rgb, "_d2s_preprocess_backend", "unknown"))
            process_latency = process_start_time - capture_start_time
            thread_latencies['capture'] = process_latency  # capture latency

            if _openxr_source_paused():
                _queue_clear_nonblocking(raw_q)
                _queue_clear_nonblocking(runtime_q)
                _source_stat_inc("runtime_dropped_paused")
                time.sleep(0.01)
                continue

            # Runtime inference + stereo synthesis
            runtime_start_time = time.perf_counter()
            prepare_start_time = time.perf_counter()
            runtime_rgb = prepare_rgb_for_stereo_runtime(frame_rgb, device=DEVICE)
            _breakdown_add_time("rt_prepare", time.perf_counter() - prepare_start_time)
            _ensure_auto_signal_sampler()
            _log_stereo_runtime_mode_once()
            _update_auto_stereo_mode(runtime_rgb)
            runtime_call_start_time = time.perf_counter()
            if RUN_MODE == "OpenXR" and OPENXR_RUNTIME_DIRECT:
                runtime_result = stereo_runtime.process_openxr_frame(
                    runtime_rgb,
                    _current_openxr_render_config(),
                )
            else:
                runtime_result = stereo_runtime.process_rgb_frame(runtime_rgb)
            _breakdown_add_time("rt_call", time.perf_counter() - runtime_call_start_time)
            _breakdown_add_runtime_timing(runtime_result)
            _log_fast_plus_fused_runtime_state(runtime_result)
            if runtime_result.depth is None:
                _queue_clear_nonblocking(runtime_q)
                _source_stat_inc("runtime_none")
                continue
            runtime_latency = time.perf_counter() - runtime_start_time
            thread_latencies['resize'] = process_latency   # resize latency
            thread_latencies['runtime'] = runtime_latency    # runtime latency

            # Send to render queue
            queue_put_start_time = time.perf_counter()
            if RUN_MODE == "OpenXR" and not OPENXR_RUNTIME_DIRECT:
                fallback_depth = runtime_result.depth
                if hasattr(fallback_depth, "detach") and fallback_depth.ndim == 4:
                    fallback_depth = fallback_depth[0, 0]
                _queue_put_latest(runtime_q, ((frame_rgb, fallback_depth), capture_start_time))
            else:
                _queue_put_latest(runtime_q, (runtime_result, capture_start_time))
            _breakdown_add_time("rt_put", time.perf_counter() - queue_put_start_time)
            _breakdown_add_time("rt_loop", time.perf_counter() - loop_start_time)
            _source_stat_inc(
                "runtime_frames",
                last_runtime_ts=time.perf_counter(),
                last_process_latency=process_latency,
                last_runtime_latency=runtime_latency,
            )
            _breakdown_inc("runtime")

        except queue.Empty:
            _source_stat_inc("raw_queue_empty")
            continue
        except Exception as e:
            _source_stat_inc(
                "runtime_errors",
                last_error=f"process_runtime_loop {type(e).__name__}: {e}",
            )
            print(f"[process_runtime_loop] Error: {type(e).__name__}: {e}", flush=True)
            time.sleep(0.05)
            continue
def cleanup_all_resources():
    """Global cleanup function"""
    print("[Cleanup] Shutting down all resources...")
    
    # Kill all processes
    for proc_name, process in global_processes.items():
        if process and hasattr(process, 'poll'):
            try:
                print(f"[Cleanup] Stopping {proc_name}...")
                process.terminate()
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    print(f"[Cleanup] Force killing {proc_name}...")
                    process.kill()
                    process.wait()
                except:
                    pass
            except Exception as e:
                print(f"[Cleanup] Error stopping {proc_name}: {e}")
            finally:
                global_processes[proc_name] = None
    
    # Stop capture
    try:
        if _stop_active_capture_session():
            print("[Cleanup] Capture stopped")
    except Exception as e:
        print(f"[Cleanup] Error stopping capture: {e}")
    
    # Stop streamer if exists
    try:
        if 'streamer' in globals() and streamer:
            streamer.stop()
            print("[Cleanup] Streamer stopped")
    except Exception as e:
        print(f"[Cleanup] Error stopping streamer: {e}")
    
    # Clear all queues to unblock threads
    queues = [raw_q, runtime_q]
    
    for q in queues:
        while not q.empty():
            try:
                q.get(timeout=TIME_SLEEP)
            except:
                pass

    # Wait for RTMP thread
    if 'rtmp_thread' in globals() and rtmp_thread.is_alive():
        rtmp_thread.join(timeout=3)
    
    print("[Cleanup] All resources cleaned up")

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

# get ffmpeg command

# Get window lists
if OS_NAME == "Windows":
    try:
        import win32gui
    except ImportError:
        win32gui = None

    def list_windows():
        windows = []
        def callback(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append((title, hwnd))
            return True

        win32gui.EnumWindows(callback, None)
        return windows
elif OS_NAME == "Darwin":
    try:
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGWindowListExcludeDesktopElements,
            kCGNullWindowID,
        )
    except ImportError:
        CGWindowListCopyWindowInfo = None

    def list_windows():
        windows = []
        options = kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements
        window_info = CGWindowListCopyWindowInfo(options, kCGNullWindowID)
        # System UI processes we want to ignore
        blacklist = {
            "Window Server",
            "ControlCenter",
            "NotificationCenter",
            "Spotlight",
            "Dock",
            "SystemUIServer",
            "CoreServicesUIAgent",
            "TextInputMenuAgent",
        }
        for win in window_info:
            title = win.get("kCGWindowName", "") or ""
            owner = win.get("kCGWindowOwnerName", "")
            layer = win.get("kCGWindowLayer", 0)
            bounds = win.get("kCGWindowBounds", {})
            # Filtering rules
            if not title.strip():
                continue
            if owner in blacklist:
                continue
            if title.strip().lower().startswith("item-"):
                continue
            if bounds.get("Y", 1) == 0:
                continue
            windows.append((title.strip(), win["kCGWindowNumber"]))
        return windows
else:
    import subprocess
    def list_windows():
        windows = []
        try:
            result = subprocess.check_output(["wmctrl", "-l"]).decode("utf-8").splitlines()
            for line in result:
                parts = line.split(None, 3)
                if len(parts) >= 4:
                    _, _, _, title = parts
                    if title.strip():
                        windows.append((title.strip(), None))
        except Exception as e:
            print("Linux window listing error:", e)
        return windows

def is_window_visible_on_screen(window_title_search, partial_match=True, timeout=2.0):
    """
    Check if a window with the given title is actually visible on screen.
    
    Args:
        window_title_search: Title or partial title to search for
        partial_match: If True, search for windows containing the search string
        timeout: How long to keep trying (seconds)
    
    Returns:
        tuple: (found, window_title, window_id_or_handle)
    """
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            windows = list_windows()
            
            for title, window_id in windows:
                if partial_match:
                    if window_title_search.lower() in title.lower():
                        return True
                    if title == window_title_search:
                        return True
            
            time.sleep(0.1)
        except Exception as e:
            print(f"[Window Check] Error listing windows: {e}")
            time.sleep(0.5)
    
    return False

def get_rtmp_cmd(os_name=OS_NAME, window=None):
    if not window:
        raise ValueError("GLFW window required for size-aware streaming")
    
    time.sleep(0.5) # wait 0.5 seconds to get correct window position
    width, height = glfw.get_framebuffer_size(window)
    if width > 0 and height > 0:
        width = (width // 2) * 2  # make even
        height = (height // 2) * 2  # make even
    else:
        time.sleep(3) # retry after 3 seconds
        get_rtmp_cmd()
        
    if os_name == "Windows":
        server_cmd = [os.path.join(RTMP_DIR, "mediamtx", "mediamtx.exe"), os.path.join(RTMP_DIR, "mediamtx", "mediamtx.yml")]
        if not LOSSLESS_SCALING_SUPPORT: 
            ffmpeg_cmd = [
                os.path.join(RTMP_DIR, "ffmpeg", "bin", "ffmpeg.exe"),
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '64',
                '-analyzeduration', '0',
                '-filter_complex', f"gfxcapture=window_title='(?i)Stereo Viewer':capture_cursor=0:max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",  # Label video output [v], fix odd height
                '-itsoffset', f'{AUDIO_DELAY}',  # Audio delay (applies to next input)
                '-f', 'dshow',
                '-rtbufsize', '256M',
                '-i', f'audio={STEREOMIX_DEVICE}',
                '-map', '[v]',
                '-map', '0:a',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-bf', '0',
                '-g', f'{FPS}',
                '-force_key_frames', f'expr:gte(t,n_forced*1)',  # Force keyframes every second
                '-r', f'{FPS}',  # Force constant output framerate
                '-crf', f'{CRF}', # 18-24 smaller better quality
                '-c:a', 'libopus',
                # '-ar', '44100',
                '-b:a', '96k',
                '-muxdelay', '0',
                '-muxpreload', '0',
                '-flush_packets', '1',
                '-f', 'mpegts',
                f'srt://localhost:8890?streamid=publish:{STREAM_KEY}&pkt_size=1316'
            ]
        else:
            import win32gui
            import win32api

            def find_window_by_prefix(prefix):
                target_hwnd = None
                def enum_handler(hwnd, _):
                    nonlocal target_hwnd
                    title = win32gui.GetWindowText(hwnd)
                    if title.lower().startswith(prefix.lower()) and win32gui.IsWindowVisible(hwnd):
                        target_hwnd = hwnd
                win32gui.EnumWindows(enum_handler, None)
                return target_hwnd

            def get_monitor_index_for_window(hwnd):
                """Returns the 0-based monitor index (matching gfxcapture order) that contains the center of the window"""
                if not hwnd:
                    raise ValueError("Invalid window handle")
                
                left, top, right, bottom = win32gui.GetWindowRect(hwnd)
                center_x = (left + right) // 2
                center_y = (top + bottom) // 2
                
                monitors = []
                for hMonitor, hdcMonitor, (left, top, right, bottom) in win32api.EnumDisplayMonitors():
                    info = win32api.GetMonitorInfo(hMonitor)
                    monitors.append({
                        'index': len(monitors),  # This order matches gfxcapture monitor_idx
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom,
                        'width': right - left,
                        'height': bottom - top,
                    })
                
                for mon in monitors:
                    if mon['left'] <= center_x < mon['right'] and mon['top'] <= center_y < mon['bottom']:
                        return mon['index'], mon
                
                raise RuntimeError("Could not determine monitor for window")

            # Main logic
            hwnd = find_window_by_prefix("Stereo Viewer")
            if not hwnd:
                raise RuntimeError("Window starting with 'Stereo Viewer' not found")

            # Get monitor index and full monitor rect
            monitor_idx, monitor = get_monitor_index_for_window(hwnd)

            # Get window client area (recommended: excludes title bar and borders)
            window_left, window_top = win32gui.ClientToScreen(hwnd, (0, 0))
            client_rect = win32gui.GetClientRect(hwnd)
            window_right = window_left + client_rect[2]
            window_bottom = window_top + client_rect[3]

            # Alternatively: full window including borders/title bar
            # window_left, window_top, window_right, window_bottom = win32gui.GetWindowRect(hwnd)

            print(f"[OK]Stereo Viewer window found on monitor {monitor_idx}")
            print(f"Monitor bounds: X={monitor['left']} Y={monitor['top']} W={monitor['width']} H={monitor['height']}")
            print(f"Window client area: X={window_left} Y={window_top} -> {window_right}x{window_bottom}")

            # Calculate crop values so that the captured monitor region is cropped exactly to the window
            crop_left   = window_left - monitor['left']
            crop_top    = window_top - monitor['top']
            crop_right  = monitor['right'] - window_right
            crop_bottom = monitor['bottom'] - window_bottom

            # Ensure non-negative crops (in case window is partially off-screen or miscalculated)
            crop_left   = max(0, crop_left)
            crop_top    = max(0, crop_top)
            crop_right  = max(0, crop_right)
            crop_bottom = max(0, crop_bottom)

            print("Calculated crop values (to capture only the window area):")
            print(f"crop_left   = {crop_left}")
            print(f"crop_top    = {crop_top}")
            print(f"crop_right  = {crop_right}")
            print(f"crop_bottom = {crop_bottom}")

            # Build gfxcapture options
            gfxcapture_options = f"monitor_idx={monitor_idx}:crop_left={crop_left}:crop_top={crop_top}:crop_right={crop_right}:crop_bottom={crop_bottom}"

            ffmpeg_cmd = [
                os.path.join(RTMP_DIR, "ffmpeg", "bin", "ffmpeg.exe"),
                '-fflags', 'nobuffer',
                '-flags', 'low_delay',
                '-probesize', '64',
                '-analyzeduration', '0',
                '-filter_complex', f"gfxcapture={gfxcapture_options}:capture_cursor=0:max_framerate={FPS},hwdownload,format=bgra,scale={width}:{height},format=yuv420p[v]",
                '-itsoffset', f'{AUDIO_DELAY}',
                '-f', 'dshow',
                '-rtbufsize', '256M',
                '-i', f'audio={STEREOMIX_DEVICE}',
                '-map', '[v]',
                '-map', '0:a',
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-bf', '0',
                '-g', f'{FPS}',
                '-force_key_frames', f'expr:gte(t,n_forced*1)',
                '-r', f'{FPS}',
                '-crf', f'{CRF}',
                '-c:a', 'libopus',
                '-b:a', '96k',
                '-muxdelay', '0',
                '-muxpreload', '0',
                '-flush_packets', '1',
                '-f', 'mpegts',
                f'srt://localhost:8890?streamid=publish:{STREAM_KEY}&pkt_size=1316'
            ]
            
    elif os_name == "Darwin":
        
        from AppKit import NSScreen

        def get_scale(monitor_index):
            """Get the Retina scale factor for a specific monitor"""
            screens = NSScreen.screens()
            if monitor_index < len(screens):
                return screens[monitor_index].backingScaleFactor()
            return 2.0  # Default to 2x for Retina displays
        
        def get_monitor_index_for_glfw(window):
            window_x, window_y = glfw.get_window_pos(window)
            monitors = glfw.get_monitors()

            for i, monitor in enumerate(monitors):
                mx, my = glfw.get_monitor_pos(monitor)
                mode = glfw.get_video_mode(monitor)
                mw, mh = mode.size.width, mode.size.height

                if mx <= window_x < mx + mw and my <= window_y < my + mh:
                    return i
            return 0
        import re
        def get_device_index(target_name, device_type="video"):
            cmd = [os.path.join(RTMP_DIR, "mac", "ffmpeg"), "-f", "avfoundation", "-list_devices", "true", "-i", ""]
            result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
            output = result.stderr

            found_audio = False
            for line in output.splitlines():
                if "AVFoundation audio devices:" in line:
                    found_audio = True
                if "AVFoundation video devices:" in line:
                    found_audio = False

                match = re.search(r'\[(\d+)\](.+)', line)
                if match:
                    index = int(match.group(1))
                    name = match.group(2).strip()
                    if name == target_name and ((device_type == "audio" and found_audio) or (device_type == "video" and not found_audio)):
                        return index
            return None
        monitor_index = get_monitor_index_for_glfw(window)
        monitors = glfw.get_monitors()
        monitor = monitors[monitor_index]
        scale_factor = get_scale(monitor_index)
        screen_name = f"Capture screen {monitor_index}"
        screen_index = get_device_index(screen_name, "video")
        audio_index = get_device_index(STEREOMIX_DEVICE, "audio")
        win_x, win_y = glfw.get_window_pos(window)
        mon_x, mon_y = glfw.get_monitor_pos(monitor)
        x = win_x - mon_x
        y = win_y - mon_y
        x = int(x * scale_factor)
        y = int(y * scale_factor)
        width = int(width * scale_factor)
        height = int(height * scale_factor)
        
        server_cmd = [os.path.join(RTMP_DIR, "mac", "mediamtx"), os.path.join(RTMP_DIR, "mac", "mediamtx.yml")]
        ffmpeg_cmd = [
            os.path.join(RTMP_DIR, "mac", "ffmpeg"),
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "1024",
            "-analyzeduration", "0",
            "-itsoffset", str(AUDIO_DELAY),
            "-pixel_format", "uyvy422",
            "-f", "avfoundation",
            "-rtbufsize", "256M",
            "-framerate", "60",
            "-i", f"{screen_index}:{audio_index}",
            "-filter_complex",
            f"[0:v]fps={FPS},crop={width}:{height}:{x}:{y},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=uyvy422[v];[0:a]aresample=async=1[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "h264_videotoolbox",
            "-profile:v", "high",
            "-pix_fmt", "yuv420p",
            "-b:v", "10M",
            "-maxrate", "12M",
            "-bufsize", "24M",
            "-g", str(FPS),
            "-r", str(FPS),
            "-realtime", "true",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-c:a", "libopus",
            "-b:a", "96k",
            "-ar", "48000",
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}",
        ]


    elif os_name == "Linux":
        import re

        def run(cmd):
            """Run a shell command and return output as string."""
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            return result.stdout.strip()

        def get_window_geometry(window_id, entire=True):
            """
            Parse xwininfo to get full window geometry and decorations (border + title bar).
            Returns: (x, y, width, height, border, titlebar)
            """
            output = run(f"xwininfo -id {window_id}")
            x = y = w = h = b = t = 0

            # Parse lines similarly to the sed version
            for line in output.splitlines():
                line = line.strip()
                if m := re.match(r"Absolute upper-left X:\s+(\d+)", line):
                    x = int(m.group(1))
                elif m := re.match(r"Absolute upper-left Y:\s+(\d+)", line):
                    y = int(m.group(1))
                elif m := re.match(r"Width:\s+(\d+)", line):
                    w = int(m.group(1))
                elif m := re.match(r"Height:\s+(\d+)", line):
                    h = int(m.group(1))
                elif m := re.match(r"Relative upper-left X:\s+(\d+)", line):
                    b = int(m.group(1))
                elif m := re.match(r"Relative upper-left Y:\s+(\d+)", line):
                    t = int(m.group(1))

            # Adjust if user wanted entire window including borders/titlebar
            if entire:
                x -= b
                y -= t
                w += 2 * b
                h += t + b

            return x, y, w, h, b, t

        def drag_window_offscreen(window_id, dx=10000, dy=10000, steps=1, delay=0.01):
            """
            Simulate a mouse drag on a window by ID using xdotool.
            Uses xwininfo to compute exact title bar position (no scale factor needed).
            """
            # Get window geometry + decoration info
            x, y, w, h, b, t = get_window_geometry(window_id)
            print(f"Window pos=({x},{y}), size={w}x{h}, border={b}, title={t}")

            # Activate the window
            run(f"xdotool windowactivate {window_id}")

            # Compute title bar click position
            # title_x = x + w // 2
            title_x = x + 20
            title_y = y + t // 2  # halfway down the title bar for reliable click

            # Move mouse to title bar and start drag
            run(f"xdotool mousemove {title_x} {title_y}")
            run("xdotool mousedown 1")

            # Smooth drag motion
            step_x = dx / steps
            step_y = dy / steps
            for _ in range(steps):
                run(f"xdotool mousemove_relative -- {step_x:.2f} {step_y:.2f}")
                time.sleep(delay)

            # Release mouse button
            run("xdotool mouseup 1")



        def get_display_env(window_id: str) -> str:
            """
            Get the DISPLAY environment variable from the process that owns the window.
            Falls back to ':0.0' if not found or on error.
            """
            try:
                # First, get the PID of the window
                pid_result = subprocess.run(
                    ["xprop", "-id", window_id, "_NET_WM_PID"],
                    capture_output=True, text=True, check=True
                )
                
                # Parse PID from output (e.g., '_NET_WM_PID(CARDINAL) = 1234')
                for line in pid_result.stdout.splitlines():
                    if "=" in line:
                        pid = line.split("=")[-1].strip()
                        if pid.isdigit():
                            # Get environment variables from the process
                            env_result = subprocess.run(
                                ["cat", f"/proc/{pid}/environ"],
                                capture_output=True, text=True, check=True
                            )
                            
                            # Parse DISPLAY from environment variables
                            for env_var in env_result.stdout.split('\x00'):
                                if env_var.startswith("DISPLAY="):
                                    return env_var.split("=", 1)[1]
                                    
            except (subprocess.CalledProcessError, FileNotFoundError, IndexError, ValueError):
                pass
            
            return ":0.0"

        def get_device_index(target_name, device_type="video"):
            """Return a device identifier suitable for ffmpeg on Linux.
            For video: returns a display/input string (we'll use it later as the ffmpeg input).
            For audio: tries to locate a PulseAudio/ALSA name that matches target_name; returns 'default' if not found.
            """
            # AUDIO: try PulseAudio device listing via ffmpeg
            if device_type == "audio":
                # Use the bundled or system ffmpeg binary; fallback to system
                cmd = ["ffmpeg", "-f", "pulse", "-list_devices", "true", "-i", ""]  # PulseAudio list (printed to stderr)
                try:
                    result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                    output = result.stderr + result.stdout
                    # ffmpeg prints devices with lines like: "    name 'alsa_input.pci-0000_00_1b.0.analog-stereo'": device names can appear
                    # Simpler: find lines containing the target_name or the friendly name
                    for line in output.splitlines():
                        if target_name in line:
                            # extract quoted name if present
                            m = re.search(r"'([^']+)'", line)
                            if m:
                                return m.group(1)
                            # otherwise attempt to extract a token
                            tokens = line.strip().split()
                            if tokens:
                                return tokens[-1].strip()
                except Exception:
                    pass

                # Try listing via pactl as fallback
                try:
                    pactl_out = subprocess.check_output(["pactl", "list", "sources"], text=True)
                    # parse "Name: <name>" and "Description: <desc>"
                    name = None
                    for line in pactl_out.splitlines():
                        line = line.strip()
                        if line.startswith("Name:"):
                            name = line.split(":", 1)[1].strip()
                        if line.startswith("Description:") and target_name in line:
                            return name or "default"
                except Exception:
                    pass

                # fallback
                return "default"
            return target_name or ":0.0"
        
        def find_window_id_by_title(title_pattern: str) -> str | None:
            """
            Returns the hex window id (e.g. '0x4a0000b') of the first window whose
            title contains *title_pattern* (case-insensitive).  Uses `xwininfo -tree -root`.
            """
            try:
                out = subprocess.check_output(
                    ["xwininfo", "-root", "-tree"], text=True, stderr=subprocess.DEVNULL
                )
                # Example line:
                #     0x4a0000b "Stereo Viewer - Left Eye": ("Stereo Viewer - Left Eye" ...
                pat = re.compile(
                    rf'^\s+(0x[0-9a-fA-F]+)\s.*?"[^"]*{re.escape(title_pattern)}[^"]*"',
                    re.IGNORECASE,
                )
                for line in out.splitlines():
                    m = pat.match(line)
                    if m:
                        return m.group(1)
            except Exception as e:
                print(f"[window_id] search failed: {e}")
            return None

        # Make sure the GLFW window has a recognizable title:
        glfw_title = glfw.get_window_title(window) or ""
        search_title = glfw_title if "Stereo Viewer" in glfw_title else "Stereo Viewer"

        window_id = find_window_id_by_title(search_title)
        display_env = get_display_env(window_id)+".0"
        
        if not window_id:
            raise RuntimeError(
                f"Could not locate a window with title containing '{search_title}'. "
                "Check glfw.set_window_title() or run `xwininfo -tree -root | grep -i stereo`."
            )
        print(f"[info] Capturing window id {window_id}")
        drag_window_offscreen(window_id)

        # audio_index = get_device_index(STEREOMIX_DEVICE, "audio")
        server_cmd = [os.path.join(RTMP_DIR, "linux", "mediamtx"), os.path.join(RTMP_DIR, "linux", "mediamtx.yml")]

        ffmpeg_cmd = [
            "ffmpeg",
            "-fflags", "+genpts+nobuffer+flush_packets",
            "-flags", "low_delay",
            "-avioflags", "direct",
            "-probesize", "1024",
            "-analyzeduration", "0",
            "-draw_mouse", "0",
            "-itsoffset", str(AUDIO_DELAY),
            "-f", "x11grab",
            "-framerate", "60", 
            "-vsync", "1",
            "-window_id", window_id,
            "-use_wallclock_as_timestamps", "1",
            "-thread_queue_size", "2048",
            "-i", display_env,
            "-f", "pulse",
            "-thread_queue_size", "512",
            "-i", STEREOMIX_DEVICE,
            "-ac", "2",
            "-filter_complex", f"[0:v]fps={FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,setpts=PTS-STARTPTS[v];[1:a]aresample=async=1:first_pts=0,apad[a]",
            "-map", "[v]",
            "-map", "[a]",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-x264-params", f"keyint={FPS}:min-keyint={FPS}:scenecut=0:rc-lookahead=0",
            "-g", str(FPS),
            "-r", str(FPS),
            "-crf", str(CRF),
            "-c:a", "aac", 
            "-ar", "44100", 
            "-b:a", "96k", 
            "-threads", "2",
            "-f", "rtsp",
            f"rtsp://localhost:8554/{STREAM_KEY}",
        ]

    return server_cmd, ffmpeg_cmd

# ffmpeg based rtmp streamer
def rtmp_stream(window):
    global current_stream_size, global_processes

    # Wait for GLFW window to be fully initialized
    print("[RTMP] Waiting for window to be ready...")
    max_attempts = 100  # 5 seconds with 0.1s intervals
    for _ in range(max_attempts):
        if shutdown_event.is_set():
            return
        
        # Check if window is valid and has a size
        try:
            width, height = glfw.get_framebuffer_size(window)
            if width > 0 and height > 0 and is_window_visible_on_screen("Stereo Viewer"):
                print(f"[RTMP] Window ready: {width}x{height}")
                time.sleep(2)
                break
        except Exception as e:
            print(f"[RTMP] Window check error: {e}")
        
        time.sleep(0.5)
    
    if shutdown_event.is_set():
        return

    while not shutdown_event.is_set():
        try:
            width, height = glfw.get_framebuffer_size(window)
            new_size = (width, height)

            # Debounce: ignore tiny changes
            if current_stream_size and abs(current_stream_size[0] - new_size[0]) < 8 and abs(current_stream_size[1] - new_size[1]) < 8:
                time.sleep(0.1)
                continue

            with ffmpeg_restart_lock:
                if current_stream_size == new_size:
                    time.sleep(0.1)
                    continue
                current_stream_size = new_size

            # Terminate old processes
            for name in ['ffmpeg', 'rtmp_server']:
                proc = global_processes.get(name)
                if proc and proc.poll() is None:
                    print(f"[RTMP] Stopping {name} for resize...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()

            # Start new
            server_cmd, ffmpeg_cmd = get_rtmp_cmd(OS_NAME, window=window)
            print(f"[RTMP] Restarting stream at {width}x{height}")

            rtmp_server = subprocess.Popen(server_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            ffmpeg = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)

            global_processes['rtmp_server'] = rtmp_server
            global_processes['ffmpeg'] = ffmpeg

            print(f"[RTMP] Stream active: {width}x{height}")

        except Exception as e:
            print(f"[RTMP] Error: {e}")
            time.sleep(1)

        time.sleep(0.2)

    print("[RTMP] Stream thread exited.")

def main(mode="Viewer"):
    # Start capture and processing threads
    threading.Thread(target=capture_loop, daemon=True).start()
    # Replace separate process_loop and depth_loop with combined thread
    threading.Thread(target=process_runtime_loop, daemon=True).start()
    
    # FPS tracking variables
    frame_count = 0
    start_time = time.perf_counter()
    last_time = time.perf_counter()
    last_fps_update_time = time.perf_counter()
    current_fps = 0.0
    total_frames = 0
    streamer, window = None, None
    
    # FPS statistics tracking
    fps_values = deque(maxlen=300)  # Recent FPS values (O(1) bounded; for 1% low)
    max_fps_history = 300  # Keep last 300 FPS values (5 seconds at 60 FPS)
    avg_fps = 0.0
    low_fps_1_percent_avg = float('inf')  # Average of FPS below 1% percentile
    fps_update_interval = 5.0  # Update statistics every 5 seconds

    # Latency statistics tracking.
    # Use a deque + running sum so per-frame collection is O(1) (the old list
    # used .append()+.pop(0), and .pop(0) is O(n) on every single frame), and the
    # sliding-window average is O(1) instead of summing the whole history.
    max_latency_history = 300  # Keep same amount as FPS
    latency_history = deque()   # bounded manually to keep the running sum in sync
    latency_sum = 0.0           # running sum of latency_history for O(1) average
    avg_total_latency = 0.0
    
    try:
        if mode == "Viewer":
            from viewer.viewer import StereoWindow
            # Get initial frame to determine window size (block until first frame arrives)
            runtime_result, capture_start_time = runtime_q.get()
            import torch
            output_frame = runtime_result.sbs
            if isinstance(output_frame, torch.Tensor):
                if output_frame.ndim == 4:
                    w, h = output_frame.shape[3], output_frame.shape[2]
                else:
                    w, h = output_frame.shape[2], output_frame.shape[1]
            else:
                w, h = output_frame.shape[1], output_frame.shape[0] # (H, W, C)
            if not STREAM_MODE:
                # For local viewer only
                h = int(1280 / w * h)
                w = 1280
                
            window = StereoWindow(
                capture_mode=CAPTURE_MODE, 
                monitor_index=MONITOR_INDEX, 
                ipd=IPD, depth_ratio=DEPTH_STRENGTH, 
                convergence=CONVERGENCE, 
                display_mode=DISPLAY_MODE, 
                fill_16_9=FILL_16_9, 
                show_fps=SHOW_FPS, 
                use_3d=USE_3D_MONITOR, 
                fix_aspect=FIX_VIEWER_ASPECT, 
                stream_mode=STREAM_MODE, 
                lossless_scaling=LOSSLESS_SCALING_SUPPORT, 
                specify_display=STEREO_DISPLAY_SELECTION, 
                stereo_display_index=STEREO_DISPLAY_INDEX, 
                frame_size=(w,h),
                use_cuda=USE_CUDART,
                cuda_device_id=DEVICE_ID,
                local_vsync=LOCAL_VSYNC,
                upscaler=UPSCALER,
                upscaler_sharpness=UPSCALER_SHARPNESS)

            if STREAM_MODE == "RTMP":
                if OS_NAME == "Windows":
                    from utils import set_window_to_bottom
                    def bottom_loop():
                        while True:
                            set_window_to_bottom(window.window)
                            time.sleep(0.01)
                    threading.Thread(target=bottom_loop, daemon=True).start()
                global rtmp_thread
                rtmp_thread = threading.Thread(target=rtmp_stream, args=(window.window,), daemon=True)
                rtmp_thread.start()
                print(f"[Main] RTMP Streamer Started (auto-restart on resize)")
            elif STREAM_MODE == "MJPEG":
                from streaming.mjpeg_streamer import MJPEGStreamer
                streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
                streamer.start()
                print(f"[Main] MJPEG Streamer Started")
            else:
                print(f"[Main] Local Viewer Started")
            
            # Process the first frame we already retrieved
            render_start_time = time.perf_counter()
            window.update_runtime_frame(runtime_result, current_fps, 0.0)  # initial latency unknown
            render_latency = time.perf_counter() - render_start_time
            total_latency = (render_start_time - capture_start_time) + render_latency
            thread_latencies['render'] = render_latency
            thread_latencies['total'] = total_latency
            
            # Main render loop
            next_render_time = time.perf_counter()
            # Variables for latency update at 1Hz
            last_latency_display = 0.0    # last latency value shown
            last_fps_time = time.perf_counter()
            while (not glfw.window_should_close(window.window) and 
                   not shutdown_event.is_set()):

                try:
                    # Get next frame (already processed + depth)
                    runtime_result, capture_start_time = runtime_q.get(timeout=0.001)
                    _breakdown_inc("viewer_get")
                    
                    # Calculate total latency for this frame
                    current_time = time.perf_counter()
                    total_latency = current_time - capture_start_time

                    # Update latencies for statistics (O(1) sliding window via
                    # running sum; popleft() is O(1), unlike list.pop(0)).
                    latency_history.append(total_latency)
                    latency_sum += total_latency
                    if len(latency_history) > max_latency_history:
                        latency_sum -= latency_history.popleft()

                    # Update FPS every second
                    frame_count += 1
                    total_frames += 1
                    elapsed = current_time - last_time
                    if elapsed >= 1.0:
                        current_fps = frame_count / elapsed
                        frame_count = 0
                        last_time = current_time
                        
                        # Store FPS value for statistics (deque auto-evicts oldest)
                        fps_values.append(current_fps)
                        
                        # Update FPS and latency statistics every 5 seconds
                        if current_time - last_fps_update_time >= fps_update_interval:
                            # Calculate average FPS
                            if fps_values:
                                avg_fps = sum(fps_values) / len(fps_values)
                            if fps_values and len(fps_values) >= 20:
                                # Calculate 1% low average
                                sorted_fps = sorted(fps_values)
                                one_percent_index = int(len(sorted_fps) * 0.1)
                                if one_percent_index == 0 and len(sorted_fps) > 0:
                                    one_percent_index = 1
                                fps_below_1_percent = sorted_fps[:one_percent_index]
                                if fps_below_1_percent:
                                    low_fps_1_percent_avg = sum(fps_below_1_percent) / len(fps_below_1_percent)
                                else:
                                    low_fps_1_percent_avg = sorted_fps[0] if sorted_fps else 0.0
                            
                            # Calculate average latency (O(1) from running sum)
                            if latency_history:
                                avg_total_latency = latency_sum / len(latency_history)
                            
                            last_fps_update_time = current_time
                        
                        # Display current latency (update once per second)
                        last_latency_display = total_latency
                        
                        # Create window title with detailed FPS and latency statistics
                        if SHOW_FPS:
                            title_text = (
                                f"{current_fps:.1f}FPS | "
                                f"Avg: {avg_fps:.1f} | "
                                f"1% Low Avg: {low_fps_1_percent_avg:.1f} | "
                                f"Latency: {last_latency_display*1000:.0f}ms | "
                                f"Avg Latency: {avg_total_latency*1000:.0f}ms "
                                f"(Capture:{thread_latencies['capture']*1000:.0f}ms "
                                f"Resize:{thread_latencies['resize']*1000:.0f}ms "
                                f"Runtime:{thread_latencies['runtime']*1000:.0f}ms "
                                f"Render:{render_latency*1000:.0f}ms)"
                            )
                        else:
                            title_text = f"{current_fps:.0f}FPS {last_latency_display*1000:.0f}ms"
                        
                        if STREAM_MODE == "MJPEG":
                            print(title_text)
                        
                        glfw.set_window_title(window.window, f"Stereo Viewer {title_text}")
                        
                        # Update the viewer OSD with new FPS and latency (once per second)
                        update_start_time = time.perf_counter()
                        window.update_runtime_frame(runtime_result, current_fps, last_latency_display)
                        _breakdown_add_time("update", time.perf_counter() - update_start_time)
                    else:
                        # Update only frame, keep previous stats (no FPS/latency change)
                        update_start_time = time.perf_counter()
                        window.update_runtime_frame(runtime_result)
                        _breakdown_add_time("update", time.perf_counter() - update_start_time)

                    # Render latency and MJPEG frame capture
                    render_start_time = time.perf_counter()
                    if STREAM_MODE == "MJPEG":
                        frame = window.capture_glfw_image()
                        streamer.set_frame(frame)
                    
                    render_latency = time.perf_counter() - render_start_time
                    thread_latencies['render'] = render_latency
                    thread_latencies['total'] = total_latency
                    
                except queue.Empty:
                    pass
                
                now = time.perf_counter()
                if not USE_3D_MONITOR and now < next_render_time:
                    wait_duration = next_render_time - now
                    time.sleep(wait_duration)
                    _breakdown_add_time("wait", wait_duration)
                if not USE_3D_MONITOR:
                    next_render_time += TIME_SLEEP
                
                _breakdown_inc("loops")
                render_loop_start = time.perf_counter()
                window.render()
                _breakdown_add_time("render", time.perf_counter() - render_loop_start)
                post_ms = getattr(window, "last_postprocess_ms", 0.0)
                if post_ms:
                    _breakdown_add_time("post", post_ms / 1000.0)
                swap_start = time.perf_counter()
                glfw.swap_buffers(window.window)
                _breakdown_add_time("swap", time.perf_counter() - swap_start)
                glfw.poll_events()
                _log_fps_breakdown()
            
            glfw.terminate()

        elif mode == "OpenXR":
            env_name = str(ENVIRONMENT_MODEL or "").strip()
            use_environment_viewer = bool(env_name) and env_name.lower() != "none"
            if use_environment_viewer:
                from xr_viewer.environment import OpenXRViewer, OPENXR_AVAILABLE
            else:
                from xr_viewer.base import OpenXRViewer, OPENXR_AVAILABLE
            if not OPENXR_AVAILABLE:
                raise ImportError("pyopenxr not installed -run: pip install pyopenxr")
            runtime_result, capture_start_time = runtime_q.get()
            import torch
            first_eye = runtime_result.left_eye
            if isinstance(first_eye, torch.Tensor):
                if first_eye.ndim == 4:
                    w, h = first_eye.shape[3], first_eye.shape[2]
                else:
                    w, h = first_eye.shape[2], first_eye.shape[1]
            else:
                w, h = first_eye.shape[1], first_eye.shape[0]
            try:
                viewer = OpenXRViewer(
                    ipd=IPD,
                    depth_ratio=DEPTH_STRENGTH,
                    convergence=CONVERGENCE,
                    frame_size=(w, h),
                    fps=FPS,
                    depth_q=runtime_q,
                    show_fps=SHOW_FPS,
                    controller_model=CONTROLLER_MODEL,
                    environment_model=ENVIRONMENT_MODEL,
                    breath_enabled=False,
                    show_preview_window=XR_PREVIEW_WINDOW,
                    capture_mode=CAPTURE_MODE,
                    monitor_index=MONITOR_INDEX,
                    render_active_event=openxr_render_active,
                    source_active_event=openxr_source_active,
                    idle_active_event=openxr_wait_idle_active,
                    runtime_config_callback=_update_openxr_runtime_config,
                )
                openxr_source_active.set()
                openxr_render_active.clear()
                openxr_wait_idle_active.clear()
                openxr_bootstrap_done.set()
                viewer.run(first_runtime_result=runtime_result, first_frame_ts=capture_start_time)
            except Exception as e:
                print(f"[Main] OpenXR Link error: {e}")

        else:
            from streaming.mjpeg_streamer import MJPEGStreamer

            streamer = MJPEGStreamer(port=STREAM_PORT, fps=FPS, quality=STREAM_QUALITY)
            streamer.start()
            
            print(f"[Main] Legacy Streamer Started")
            
            # FPS and latency tracking for legacy mode
            fps_values = deque(maxlen=300)  # O(1) bounded history
            max_fps_history = 300
            avg_fps = 0.0
            low_fps_1_percent_avg = float('inf')
            last_fps_update_time = time.perf_counter()
            
            while not shutdown_event.is_set():
                try:
                    # Fix for unstable dml runtime error
                    runtime_result, _ = runtime_q.get(timeout=TIME_SLEEP)
                    streamer.set_frame(_runtime_output_to_numpy(runtime_result.sbs))
                    
                    # Calculate FPS
                    frame_count += 1
                    total_frames += 1
                    current_time = time.perf_counter()
                    if current_time - last_time >= 1.0:
                        current_fps = frame_count / (current_time - last_time)
                        
                        # Store FPS value for statistics (deque auto-evicts oldest)
                        fps_values.append(current_fps)
                        
                        # Update FPS statistics every 5 seconds
                        if current_time - last_fps_update_time >= fps_update_interval:
                            if fps_values and len(fps_values) >= 10:
                                # Calculate average FPS
                                avg_fps = sum(fps_values) / len(fps_values)
                                
                                # Calculate average of FPS below 1% percentile
                                sorted_fps = sorted(fps_values)
                                one_percent_index = int(len(sorted_fps) * 0.01)
                                
                                if one_percent_index == 0 and len(sorted_fps) > 0:
                                    one_percent_index = 1
                                
                                fps_below_1_percent = sorted_fps[:one_percent_index]
                                
                                if fps_below_1_percent:
                                    low_fps_1_percent_avg = sum(fps_below_1_percent) / len(fps_below_1_percent)
                                else:
                                    low_fps_1_percent_avg = sorted_fps[0] if sorted_fps else 0.0
                            
                            last_fps_update_time = current_time
                        
                        frame_count = 0
                        last_time = current_time
                        print(f"{current_fps:.1f} FPS | Avg: {avg_fps:.1f} | 1% Low Avg: {low_fps_1_percent_avg:.1f}")
                            
                except queue.Empty:
                    continue
                except Exception as e:
                    if not shutdown_event.is_set():
                        print(f"Streamer error: {e}")
                    break
                    
    except KeyboardInterrupt:
        print("\n[Main] Keyboard interrupt received, shutting down...")
    # except Exception as e:
    #     print(f"[Main] Error: {e}")
    finally:
        # Ensure cleanup happens
        shutdown_event.set()
        cleanup_all_resources()
        
        if SHOW_FPS:
            total_time = time.perf_counter() - start_time
            overall_avg_fps = total_frames / total_time if total_time > 0 else 0
            print(f"Overall Average FPS: {overall_avg_fps:.2f}")
            if fps_values:
                print(f"Recent Average FPS: {avg_fps:.1f}")
                print(f"Recent 1% Low Average FPS: {low_fps_1_percent_avg:.1f}")
        print(f"[Main] Stopped")

if __name__ == "__main__":
    main(mode=RUN_MODE)
