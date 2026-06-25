from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Callable

from .runtime import openxr_result_from_stereo_result

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



def _openxr_full_synthesis_enabled(ctx: RuntimePipelineContext) -> bool:
    if ctx.run_mode != "OpenXR":
        return False
    if not ctx.openxr_runtime_direct:
        return True
    return str(ctx.stereo_active_preset or "").strip().lower() in _OPENXR_FULL_SYNTHESIS_PRESETS


def _rgb_size_text(frame) -> str:
    shape = tuple(getattr(frame, "shape", ()))
    if len(shape) == 4:
        return f"{int(shape[3])}x{int(shape[2])}"
    if len(shape) == 3 and shape[0] in (3, 4):
        return f"{int(shape[2])}x{int(shape[1])}"
    if len(shape) == 3:
        return f"{int(shape[1])}x{int(shape[0])}"
    return "unknown"
class RuntimePipelineLoop:
    def __init__(self, context: RuntimePipelineContext):
        self.context = context
        self._logged_rgb_shape = False

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

                frame_raw, size, capture_start_time = ctx.queue_drain_latest(
                    ctx.raw_q,
                    ctx.raw_q.get(timeout=min(ctx.time_sleep, 0.01)),
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
                frame_rgb = ctx.capture_frame_to_rgb(
                    frame_raw,
                    size,
                    device=ctx.device,
                    use_torch=ctx.use_cudart,
                    output="tensor",
                )
                if not self._logged_rgb_shape:
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
            except Exception as exc:
                ctx.source_stat_inc(
                    "runtime_errors",
                    last_error=f"process_runtime_loop {type(exc).__name__}: {exc}",
                )
                print(f"[process_runtime_loop] Error: {type(exc).__name__}: {exc}", flush=True)
                time.sleep(0.05)
                continue
