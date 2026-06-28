from __future__ import annotations

import os
import queue
import time
from dataclasses import dataclass
from typing import Callable

from capture.types import CapturedFrame

from .render_size import RenderSizeConfig, resolve_render_size, runtime_output_size_text
from .runtime import openxr_result_from_stereo_result
from .settings_snapshot import RuntimeSettingsPipelineRebuildRequired, RuntimeSettingsRestartRequired

_OPENXR_FULL_SYNTHESIS_PRESETS = {"cinema", "game_low_latency", "still_image_hq", "debug_export"}


@dataclass(frozen=True)
class RuntimePipelineContext:
    shutdown_event: object
    raw_q: object
    runtime_q: object
    time_sleep: float
    run_mode: str
    openxr_runtime_direct: bool
    stereo_active_preset: str | None
    device: object
    use_cudart: bool
    thread_latencies: dict
    stereo_runtime: object
    capture_frame_to_rgb: Callable
    prepare_rgb_for_stereo_runtime: Callable
    current_openxr_render_config: Callable[[], object]
    is_hard_idle: Callable[[], bool]
    is_source_paused: Callable[[], bool]
    log_source_health: Callable[[], None]
    source_stat_inc: Callable[..., None]
    breakdown_inc: Callable[..., None]
    breakdown_add_time: Callable[..., None]
    breakdown_add_runtime_timing: Callable[..., None]
    set_preprocess_backend: Callable[[str], None]
    queue_clear: Callable[[object], None]
    queue_drain_latest: Callable[[object, object], object]
    queue_put_latest: Callable[[object, object], None]
    log_stereo_runtime_mode_once: Callable[[], None]
    apply_stereo_hot_reload_if_needed: Callable[[], None]
    warmup_stereo_once_for_frame: Callable[[object], None]
    log_fast_plus_fused_runtime_state: Callable[[object], None]
    application_runtime_target: str | None = None
    output_transport: str | None = None
    settings_update_q: object | None = None
    render_size_config: RenderSizeConfig | None = None


def _drain_latest_nowait(q: object | None):
    if q is None:
        return None
    latest = None
    while True:
        try:
            latest = q.get_nowait()
        except queue.Empty:
            return latest


def _openxr_full_synthesis_enabled(ctx: RuntimePipelineContext) -> bool:
    if ctx.run_mode != "OpenXR":
        return False
    if not ctx.openxr_runtime_direct:
        return True
    return str(ctx.stereo_active_preset or "").strip().lower() in _OPENXR_FULL_SYNTHESIS_PRESETS


def _active_preset_for_snapshot(settings_snapshot, active_preset):
    snapshot_preset = getattr(settings_snapshot, "stereo_preset", None)
    if snapshot_preset == "auto":
        return active_preset
    return snapshot_preset or active_preset


def _apply_latest_settings_snapshot(ctx: RuntimePipelineContext):
    settings_snapshot = _drain_latest_nowait(ctx.settings_update_q)
    if settings_snapshot is None:
        return None
    change_class = ctx.stereo_runtime.apply_settings_snapshot(
        settings_snapshot,
        active_preset=_active_preset_for_snapshot(settings_snapshot, ctx.stereo_active_preset),
    )
    ctx.source_stat_inc(
        "settings_updates",
        last_settings_version=int(settings_snapshot.version),
        last_settings_change_class=change_class.value,
    )
    return change_class


def _unpack_raw_queue_item(item):
    if isinstance(item, CapturedFrame):
        return item.frame, item.target_height, item.timestamp, item
    frame_raw, size, capture_start_time = item
    return frame_raw, size, capture_start_time, None


def _resolve_pipeline_render_size(size, config: RenderSizeConfig | None):
    if config is None:
        return size
    if isinstance(size, (tuple, list)) and len(size) == 2:
        return resolve_render_size((int(size[0]), int(size[1])), config)
    return size


def _reset_runtime_temporal_state(runtime) -> None:
    temporal_state = getattr(runtime, "temporal_state", None)
    reset_stereo = getattr(temporal_state, "reset_stereo", None)
    if callable(reset_stereo):
        reset_stereo()


def _source_target_key(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return ("legacy",)
    metadata = captured_frame.metadata if isinstance(captured_frame.metadata, dict) else {}
    metadata_parts = tuple(
        (key, str(metadata[key]))
        for key in ("source_id", "source_key", "capture_source", "capture_target", "target_id", "window_handle", "hwnd")
        if key in metadata and metadata[key] is not None
    )
    return (
        "captured",
        str(captured_frame.capture_mode or ""),
        int(captured_frame.monitor_index or 0),
        str(captured_frame.window_title or ""),
        metadata_parts,
    )


def _append_temporal_reset_reason(debug_info: dict, reason: str) -> None:
    current = debug_info.get("temporal_reset_reason")
    if not current:
        debug_info["temporal_reset_reason"] = reason
        return
    reasons = [part.strip() for part in str(current).split(",") if part.strip()]
    if reason not in reasons:
        reasons.append(reason)
    debug_info["temporal_reset_reason"] = ",".join(reasons)


def _rgb_size_text(frame) -> str:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        return f"{int(shape[3])}x{int(shape[2])}"
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"


def _capture_zero_copy(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return None
    if "zero_copy" in captured_frame.metadata:
        return bool(captured_frame.metadata["zero_copy"])
    return captured_frame.copy_mode.value == "none"


def _capture_copy_mode(captured_frame: CapturedFrame | None):
    if captured_frame is None:
        return None
    return captured_frame.copy_mode.value


def _capture_debug_fields(captured_frame: CapturedFrame | None, frame_rgb) -> dict:
    fields = {
        "preprocess_device_origin": getattr(frame_rgb, "_d2s_preprocess_device_origin", None),
        "preprocess_device_output": getattr(frame_rgb, "_d2s_preprocess_device_output", None),
        "preprocess_device_transfer": getattr(frame_rgb, "_d2s_preprocess_device_transfer", None),
        "preprocess_input_kind": getattr(frame_rgb, "_d2s_preprocess_input_kind", None),
        "capture_copy_mode": getattr(frame_rgb, "_d2s_capture_copy_mode", _capture_copy_mode(captured_frame)),
        "capture_zero_copy": getattr(frame_rgb, "_d2s_capture_zero_copy", _capture_zero_copy(captured_frame)),
    }
    if captured_frame is not None:
        fields.update(
            capture_tool=captured_frame.capture_tool,
            capture_frame_raw_device=captured_frame.frame_raw_device,
            capture_frame_raw_type=captured_frame.frame_raw_type,
            capture_frame_raw_dtype=captured_frame.frame_raw_dtype,
        )
    return {key: value for key, value in fields.items() if value is not None}


def _attach_capture_debug(runtime_result, captured_frame: CapturedFrame | None, frame_rgb) -> None:
    debug_info = getattr(runtime_result, "debug_info", None)
    if isinstance(debug_info, dict):
        debug_info.update(_capture_debug_fields(captured_frame, frame_rgb))


def _attach_pipeline_debug(runtime_result, *, capture_size, render_size, run_mode, render_size_config, output_transport=None) -> None:
    debug_info = getattr(runtime_result, "debug_info", None)
    if not isinstance(debug_info, dict):
        return
    debug_info["capture_size"] = _size_debug_text(capture_size)
    debug_info["render_size"] = _size_debug_text(render_size)
    debug_info["transport"] = _transport_debug_label(run_mode, output_transport=output_transport)
    if render_size_config is not None:
        debug_info["render_size_policy"] = render_size_config.policy.value
        debug_info["stereo_render_scale"] = render_size_config.scale_factor


def _transport_debug_label(run_mode, *, output_transport=None) -> str:
    if output_transport:
        return str(output_transport)
    return "openxr_swapchain" if run_mode == "OpenXR" else "local_window"


def _size_debug_text(size) -> str:
    if isinstance(size, (tuple, list)) and len(size) == 2:
        try:
            return runtime_output_size_text((int(size[0]), int(size[1])))
        except (TypeError, ValueError):
            pass
    return str(size)


class RuntimePipelineLoop:
    def __init__(self, context: RuntimePipelineContext):
        self.context = context
        self._logged_rgb_shape = False
        self._last_render_size = None
        self._last_source_target_key = None
        self._has_source_target_key = False

    def run(self) -> None:
        ctx = self.context
        while not ctx.shutdown_event.is_set():
            ctx.log_source_health()
            try:
                if ctx.shutdown_event.is_set():
                    break
                if ctx.is_hard_idle():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    time.sleep(0.1)
                    continue
                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                _apply_latest_settings_snapshot(ctx)

                frame_raw, size, capture_start_time, captured_frame = _unpack_raw_queue_item(
                    ctx.queue_drain_latest(
                        ctx.raw_q,
                        ctx.raw_q.get(timeout=min(ctx.time_sleep, 0.01)),
                    )
                )
                ctx.source_stat_inc("raw_get", last_raw_get_ts=time.perf_counter())
                ctx.breakdown_inc("raw_get")

                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                loop_start_time = time.perf_counter()

                process_start_time = time.perf_counter()
                render_size = _resolve_pipeline_render_size(size, ctx.render_size_config)
                render_size_changed = self._last_render_size is not None and render_size != self._last_render_size
                source_target_key = _source_target_key(captured_frame)
                source_target_changed = (
                    self._has_source_target_key
                    and source_target_key != self._last_source_target_key
                )
                if render_size_changed:
                    _reset_runtime_temporal_state(ctx.stereo_runtime)
                if source_target_changed:
                    _reset_runtime_temporal_state(ctx.stereo_runtime)
                self._last_render_size = render_size
                self._last_source_target_key = source_target_key
                self._has_source_target_key = True
                frame_rgb = ctx.capture_frame_to_rgb(
                    frame_raw,
                    render_size,
                    device=ctx.device,
                    use_torch=ctx.use_cudart,
                    output="tensor",
                    frame_raw_device=captured_frame.frame_raw_device if captured_frame else None,
                    capture_copy_mode=_capture_copy_mode(captured_frame),
                    capture_zero_copy=_capture_zero_copy(captured_frame),
                )
                if not self._logged_rgb_shape and os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                    self._logged_rgb_shape = True
                    print(
                        f"[process_runtime_loop] rgb={_rgb_size_text(frame_rgb)}",
                        flush=True,
                    )
                ctx.breakdown_add_time("rt_cap2rgb", time.perf_counter() - process_start_time)
                ctx.set_preprocess_backend(
                    str(getattr(frame_rgb, "_d2s_preprocess_backend", "unknown"))
                )
                process_latency = process_start_time - capture_start_time
                ctx.thread_latencies["capture"] = process_latency

                if ctx.is_source_paused():
                    ctx.queue_clear(ctx.raw_q)
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_dropped_paused")
                    time.sleep(0.01)
                    continue

                runtime_start_time = time.perf_counter()
                prepare_start_time = time.perf_counter()
                runtime_rgb = ctx.prepare_rgb_for_stereo_runtime(frame_rgb, device=ctx.device)
                ctx.breakdown_add_time("rt_prepare", time.perf_counter() - prepare_start_time)
                ctx.log_stereo_runtime_mode_once()
                ctx.apply_stereo_hot_reload_if_needed()
                _apply_latest_settings_snapshot(ctx)
                ctx.warmup_stereo_once_for_frame(runtime_rgb)
                runtime_call_start_time = time.perf_counter()
                if ctx.run_mode == "OpenXR" and not _openxr_full_synthesis_enabled(ctx):
                    runtime_result = ctx.stereo_runtime.process_openxr_frame(
                        runtime_rgb,
                        ctx.current_openxr_render_config(),
                    )
                else:
                    runtime_result = ctx.stereo_runtime.process_rgb_frame(runtime_rgb)
                    if ctx.run_mode == "OpenXR":
                        runtime_result = openxr_result_from_stereo_result(runtime_result)
                ctx.breakdown_add_time("rt_call", time.perf_counter() - runtime_call_start_time)
                _attach_pipeline_debug(
                    runtime_result,
                    capture_size=size,
                    render_size=render_size,
                    run_mode=ctx.run_mode,
                    render_size_config=ctx.render_size_config,
                    output_transport=ctx.output_transport,
                )
                debug_info = getattr(runtime_result, "debug_info", None)
                if isinstance(debug_info, dict):
                    if render_size_changed:
                        _append_temporal_reset_reason(debug_info, "render_size_changed")
                    if source_target_changed:
                        _append_temporal_reset_reason(debug_info, "source_target_changed")
                    if ctx.application_runtime_target:
                        debug_info["application_runtime_target"] = ctx.application_runtime_target
                    if ctx.output_transport:
                        debug_info["output_transport"] = ctx.output_transport
                _attach_capture_debug(runtime_result, captured_frame, frame_rgb)
                ctx.breakdown_add_runtime_timing(runtime_result)
                ctx.log_fast_plus_fused_runtime_state(runtime_result)
                if runtime_result.depth is None:
                    ctx.queue_clear(ctx.runtime_q)
                    ctx.source_stat_inc("runtime_none")
                    continue
                runtime_latency = time.perf_counter() - runtime_start_time
                ctx.thread_latencies["resize"] = process_latency
                ctx.thread_latencies["runtime"] = runtime_latency

                queue_put_start_time = time.perf_counter()
                ctx.queue_put_latest(ctx.runtime_q, (runtime_result, capture_start_time))
                ctx.breakdown_add_time("rt_put", time.perf_counter() - queue_put_start_time)
                ctx.breakdown_add_time("rt_loop", time.perf_counter() - loop_start_time)
                ctx.source_stat_inc(
                    "runtime_frames",
                    last_runtime_ts=time.perf_counter(),
                    last_process_latency=process_latency,
                    last_runtime_latency=runtime_latency,
                )
                ctx.breakdown_inc("runtime")

            except queue.Empty:
                ctx.source_stat_inc("raw_queue_empty")
                continue
            except (RuntimeSettingsPipelineRebuildRequired, RuntimeSettingsRestartRequired):
                raise
            except Exception as exc:
                ctx.source_stat_inc(
                    "runtime_errors",
                    last_error=f"process_runtime_loop {type(exc).__name__}: {exc}",
                )
                print(f"[process_runtime_loop] Error: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.05)
                continue
