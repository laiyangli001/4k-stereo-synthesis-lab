from __future__ import annotations

import queue
import sys
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from capture.types import CapturedFrame, FrameCopyMode
from stereo_runtime.pipeline import (
    RuntimePipelineContext,
    RuntimePipelineLoop,
    _attach_cuda_ready_event,
    _attach_pipeline_debug,
    _add_cuda_event_timings,
    _is_fatal_runtime_preparation_error,
    _runtime_sync_after_frame_enabled,
)
from stereo_runtime.render_size import RenderSizeConfig, RenderSizePolicy
from stereo_runtime.settings_snapshot import (
    RuntimeSettingsPipelineRebuildRequired,
    SnapshotChangeClass,
    RuntimeSettingsSnapshot,
)


class OneShotShutdown:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 2


class NFrameShutdown:
    def __init__(self, frame_count):
        self.calls = 0
        self.frame_count = int(frame_count)

    def is_set(self):
        self.calls += 1
        return self.calls > self.frame_count * 2


class FlagShutdown:
    def __init__(self):
        self.flag = False

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True


class CountingTemporalState:
    def __init__(self):
        self.reset_count = 0

    def reset_stereo(self):
        self.reset_count += 1


@dataclass(frozen=True)
class FakeStereoConfig:
    temporal: bool = True
    temporal_strength: float = 0.85
    auto_reset_temporal: bool = True
    mask_feather_radius: int = 1


class FakeRuntime:
    def __init__(self):
        self.openxr_config = None
        self.rgb_calls = 0
        self.openxr_calls = 0
        self.depth_calls = 0
        self.load_calls = 0
        self.snapshots = []
        self.skip_sbs_output = None
        self.temporal_state = CountingTemporalState()
        self.stereo_config = FakeStereoConfig()
        self.seen_stereo_config = None
        self.config = SimpleNamespace(depth_backend="tensorrt_native", use_cuda_graph=False)

    def load(self):
        self.load_calls += 1

    def _predict_depth_profile(self, runtime_rgb):
        self.depth_calls += 1
        return SimpleNamespace(depth="depth", preprocess_ms=0.1, model_ms=0.2, postprocess_ms=0.1)

    def apply_settings_snapshot(self, snapshot, *, active_preset=None):
        self.snapshots.append((snapshot, active_preset))
        if snapshot.use_cuda_graph is not None:
            self.config.use_cuda_graph = snapshot.use_cuda_graph
        return snapshot.classify()

    def process_rgb_frame(self, runtime_rgb, **_kwargs):
        self.rgb_calls += 1
        self.skip_sbs_output = _kwargs.get("skip_sbs_output")
        self.seen_stereo_config = self.stereo_config
        return SimpleNamespace(
            depth="depth",
            left_eye="left-eye",
            right_eye="right-eye",
            sbs="sbs",
            timing={"total_ms": 1.0},
            debug_info={"backend": "fake"},
            provider_info={"provider": "fake"},
        )

    def process_openxr_frame(self, runtime_rgb, openxr_config):
        self.openxr_calls += 1
        self.openxr_config = openxr_config
        return SimpleNamespace(
            depth="depth",
            timing={"total_ms": 1.0},
            debug_info={"backend": "openxr_viewer_shader_dibr", "runtime_output_format": "openxr_rgb_depth"},
        )


class PipelineRebuildRuntime(FakeRuntime):
    def apply_settings_snapshot(self, snapshot, *, active_preset=None):
        raise RuntimeSettingsPipelineRebuildRequired(snapshot, ("render_size_policy",))


def test_attach_cuda_ready_event_records_on_cuda_result(monkeypatch):
    events = []
    stream = object()

    class FakeEvent:
        def __init__(self, blocking=False):
            self.blocking = blocking
            self.recorded_stream = None
            events.append(self)

        def record(self, recorded_stream):
            self.recorded_stream = recorded_stream

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            Event=FakeEvent,
            current_stream=lambda device: stream,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = SimpleNamespace(left_eye=SimpleNamespace(device=SimpleNamespace(type="cuda")))

    updated = _attach_cuda_ready_event(result)

    assert updated is result
    assert result.cuda_ready_event is events[0]
    assert result.cuda_ready_event.blocking is False
    assert result.cuda_ready_event.recorded_stream is stream


def test_add_cuda_event_timings_records_elapsed_segments():
    class FakeEvent:
        def __init__(self, ms):
            self.ms = float(ms)

        def elapsed_time(self, other):
            return other.ms - self.ms

    events = {
        "start": FakeEvent(0.0),
        "depth": FakeEvent(10.0),
        "synthesis": FakeEvent(25.0),
        "pack": FakeEvent(30.0),
        "openxr_pack_start": FakeEvent(30.0),
        "openxr_pack": FakeEvent(32.0),
        "end": FakeEvent(32.0),
    }
    recorded = {}
    ctx = SimpleNamespace(
        breakdown_add_time=lambda name, seconds: recorded.setdefault(name, seconds),
    )

    _add_cuda_event_timings(ctx, SimpleNamespace(cuda_timing_events=events))

    assert recorded["rt_gpu_depth"] == 0.01
    assert recorded["rt_gpu_synth"] == 0.015
    assert recorded["rt_gpu_pack"] == 0.005
    assert recorded["rt_gpu_openxr_pack"] == 0.002
    assert recorded["rt_gpu_total"] == 0.032


def test_pipeline_debug_transport_prefers_explicit_output_transport():
    result = SimpleNamespace(debug_info={})

    _attach_pipeline_debug(
        result,
        capture_size=(3840, 2160),
        render_size=(1920, 1080),
        run_mode="Viewer",
        render_size_config=None,
        application_runtime_target="network_stream",
        output_transport="encoded_stream",
    )

    assert result.debug_info["application_runtime_target"] == "network_stream"
    assert result.debug_info["transport"] == "encoded_stream"
    assert result.debug_info["output_transport"] == "encoded_stream"
    assert result.debug_info["capture_size"] == "3840x2160"
    assert result.debug_info["render_size"] == "1920x1080"


def test_pipeline_debug_transport_falls_back_for_openxr_and_local_window():
    openxr_result = SimpleNamespace(debug_info={})
    local_result = SimpleNamespace(debug_info={})

    _attach_pipeline_debug(
        openxr_result,
        capture_size=(2, 2),
        render_size=(2, 2),
        run_mode="OpenXR",
        render_size_config=None,
    )
    _attach_pipeline_debug(
        local_result,
        capture_size=(2, 2),
        render_size=(2, 2),
        run_mode="Viewer",
        render_size_config=None,
    )

    assert openxr_result.debug_info["application_runtime_target"] == "openxr"
    assert openxr_result.debug_info["transport"] == "openxr_swapchain"
    assert openxr_result.debug_info["output_transport"] == "openxr_swapchain"
    assert local_result.debug_info["application_runtime_target"] == "local_viewer"
    assert local_result.debug_info["transport"] == "local_window"
    assert local_result.debug_info["output_transport"] == "local_window"


def test_runtime_sync_after_frame_defaults_off_and_keeps_explicit_override(monkeypatch):
    monkeypatch.delenv("D2S_RUNTIME_SYNC_AFTER_FRAME", raising=False)
    ctx = SimpleNamespace(run_mode="Viewer", use_cudart=True)

    assert _runtime_sync_after_frame_enabled(ctx) is False

    ctx.use_cudart = False
    assert _runtime_sync_after_frame_enabled(ctx) is False

    monkeypatch.setenv("D2S_RUNTIME_SYNC_AFTER_FRAME", "1")
    assert _runtime_sync_after_frame_enabled(ctx) is True
    monkeypatch.setenv("D2S_RUNTIME_SYNC_AFTER_FRAME", "0")
    assert _runtime_sync_after_frame_enabled(ctx) is False
    monkeypatch.setenv("D2S_RUNTIME_SYNC_AFTER_FRAME", "auto")
    assert _runtime_sync_after_frame_enabled(ctx) is False


def test_fatal_runtime_preparation_error_detection():
    assert _is_fatal_runtime_preparation_error(RuntimeError("unable to resolve InfiniDepth weights for 'x'")) is True
    assert _is_fatal_runtime_preparation_error(FileNotFoundError("model directory not found")) is True
    assert _is_fatal_runtime_preparation_error(RuntimeError("random transient failure")) is False


def test_runtime_pipeline_processes_one_frame():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    shutdown = OneShotShutdown()
    stats = {}
    breakdown = {}
    latencies = {}
    calls = []

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    def breakdown_add_time(name, seconds):
        breakdown[f"{name}_count"] = breakdown.get(f"{name}_count", 0) + 1

    runtime = FakeRuntime()
    original_process_rgb_frame = runtime.process_rgb_frame

    def process_rgb_frame(runtime_rgb, **kwargs):
        calls.append(("runtime", runtime_rgb))
        return original_process_rgb_frame(runtime_rgb, **kwargs)

    runtime.process_rgb_frame = process_rgb_frame

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies=latencies,
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: calls.append("health"),
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=breakdown_add_time,
        breakdown_add_runtime_timing=lambda result: calls.append(("timing", result.timing)),
        set_preprocess_backend=lambda backend: calls.append(("pre", backend)),
        queue_clear=lambda q: calls.append(("clear", q)),
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: calls.append("mode"),
        apply_stereo_hot_reload_if_needed=lambda: calls.append("hot-reload"),
        warmup_stereo_once_for_frame=lambda frame: calls.append(("warmup", frame)),
        log_fast_plus_fused_runtime_state=lambda result: calls.append(("fused", result.debug_info)),
    )

    RuntimePipelineLoop(context).run()

    runtime_result, capture_start_time = runtime_q.get_nowait()
    assert runtime_result.sbs == "sbs"
    assert capture_start_time == 10.0
    assert stats["raw_get"] == 1
    assert stats["runtime_frames"] == 1
    assert breakdown["raw_get"] == 1
    assert breakdown["runtime"] == 1
    assert latencies["capture"] > 0
    assert latencies["runtime"] >= 0
    assert ("pre", "fake-preprocess") in calls
    assert "mode" in calls
    assert "hot-reload" in calls
    assert ("warmup", "runtime-rgb") in calls
    assert calls.index(("runtime", "runtime-rgb")) < calls.index(("warmup", "runtime-rgb"))


def test_runtime_pipeline_stops_after_fatal_model_preparation_error(capsys):
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    shutdown = FlagShutdown()
    stats = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    runtime = FakeRuntime()

    def fail_process_rgb_frame(runtime_rgb, **kwargs):
        raise RuntimeError("unable to resolve InfiniDepth weights for 'lc700x/InfiniDepth-Large'")

    runtime.process_rgb_frame = fail_process_rgb_frame

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(_d2s_preprocess_backend="fake-preprocess"),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    output = capsys.readouterr().out
    assert shutdown.flag is True
    assert stats["runtime_errors"] == 1
    assert stats["runtime_fatal_errors"] == 1
    assert "Fatal: RuntimeError: unable to resolve InfiniDepth weights" in output
    assert runtime_q.empty()


def test_runtime_pipeline_drops_raw_when_runtime_queue_is_backpressured():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    runtime_q.put(("old-runtime", 9.0))
    stats = {}
    breakdown = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    def fail_gpu_work(*args, **kwargs):
        raise AssertionError("backpressure drop must not run GPU work")

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=fail_gpu_work,
        prepare_rgb_for_stereo_runtime=fail_gpu_work,
        current_openxr_render_config=fail_gpu_work,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda *args, **kwargs: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    assert runtime_q.get_nowait() == ("old-runtime", 9.0)
    assert raw_q.empty()
    assert stats["raw_get"] == 1
    assert stats["runtime_drop_backpressure"] == 1
    assert breakdown["runtime_drop_backpressure"] == 1


def test_runtime_pipeline_holds_pending_result_until_cuda_event_ready(monkeypatch):
    import stereo_runtime.pipeline as pipeline

    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    stats = {}
    breakdown = {}

    class DelayedEvent:
        def __init__(self):
            self.calls = 0

        def query(self):
            self.calls += 1
            return self.calls >= 2

    event = DelayedEvent()

    def attach_ready_event(result):
        result.cuda_ready_event = event
        return result

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    context = RuntimePipelineContext(
        shutdown_event=NFrameShutdown(2),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda *args, **kwargs: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )
    monkeypatch.setattr(pipeline, "_attach_cuda_ready_event", attach_ready_event)

    RuntimePipelineLoop(context).run()

    runtime_result, capture_start_time = runtime_q.get_nowait()
    assert runtime_result.cuda_ready_event is event
    assert capture_start_time == 10.0
    assert stats["runtime_pending_cuda"] == 1
    assert stats["runtime_frames"] == 1
    assert breakdown["runtime_pending_cuda"] == 1
    assert breakdown["runtime"] == 1


def test_runtime_pipeline_continues_until_pending_cuda_depth_limit(monkeypatch):
    import stereo_runtime.pipeline as pipeline

    raw_q = queue.Queue(maxsize=2)
    runtime_q = queue.Queue(maxsize=2)
    raw_q.put(("raw-a", (2, 2), 10.0))
    raw_q.put(("raw-b", (2, 2), 11.0))
    stats = {}
    breakdown = {}
    runtime = FakeRuntime()

    class NeverReadyEvent:
        def query(self):
            return False

    def attach_ready_event(result):
        result.cuda_ready_event = NeverReadyEvent()
        return result

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    context = RuntimePipelineContext(
        shutdown_event=NFrameShutdown(2),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda *args, **kwargs: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )
    monkeypatch.setattr(pipeline, "_attach_cuda_ready_event", attach_ready_event)

    RuntimePipelineLoop(context).run()

    assert runtime.rgb_calls == 2
    assert raw_q.empty()
    assert runtime_q.empty()
    assert stats["runtime_pending_cuda"] == 2
    assert breakdown["runtime_pending_cuda"] == 2


def test_runtime_pipeline_drops_raw_when_pending_cuda_depth_is_full():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    stats = {}
    breakdown = {}

    class InflightEvent:
        def query(self):
            return False

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    def fail_gpu_work(*args, **kwargs):
        raise AssertionError("full pending CUDA depth must not allow new GPU work")

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=fail_gpu_work,
        prepare_rgb_for_stereo_runtime=fail_gpu_work,
        current_openxr_render_config=fail_gpu_work,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda *args, **kwargs: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )
    loop = RuntimePipelineLoop(context)
    loop._last_cuda_ready_event = InflightEvent()
    loop._pending_runtime_items = [
        (SimpleNamespace(cuda_ready_event=InflightEvent()), 8.0, 0.0, 0.0, 1.0),
        (SimpleNamespace(cuda_ready_event=InflightEvent()), 9.0, 0.0, 0.0, 1.0),
    ]

    loop.run()

    assert runtime_q.empty()
    assert raw_q.empty()
    assert stats["raw_get"] == 1
    assert stats["runtime_drop_cuda_inflight"] == 1
    assert breakdown["runtime_drop_cuda_inflight"] == 1


def test_runtime_pipeline_drop_only_drains_raw_without_gpu_work(monkeypatch):
    monkeypatch.setenv("D2S_RUNTIME_DROP_ONLY", "1")
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    stats = {}
    breakdown = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def breakdown_inc(name, amount=1):
        breakdown[name] = breakdown.get(name, 0) + amount

    def fail_gpu_work(*args, **kwargs):
        raise AssertionError("drop-only mode must not run GPU work")

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=fail_gpu_work,
        prepare_rgb_for_stereo_runtime=fail_gpu_work,
        current_openxr_render_config=fail_gpu_work,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=breakdown_inc,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda *args, **kwargs: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    assert runtime_q.empty()
    assert stats["raw_get"] == 1
    assert stats["runtime_diag_raw"] == 1
    assert breakdown["raw_get"] == 1
    assert breakdown["runtime_diag_raw"] == 1


@pytest.mark.parametrize(
    ("stage", "expected_stat"),
    [
        ("preprocess", "runtime_diag_preprocess"),
        ("prepare", "runtime_diag_prepare"),
        ("depth", "runtime_diag_depth"),
        ("runtime", "runtime_diag_runtime"),
    ],
)
def test_runtime_pipeline_diag_stage_stops_after_selected_stage(monkeypatch, stage, expected_stat):
    monkeypatch.setenv("D2S_RUNTIME_DIAG_STAGE", stage)
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    stats = {}
    calls = []

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    def fail_runtime_work(*args, **kwargs):
        raise AssertionError("diag stage must stop before runtime work")

    runtime = FakeRuntime()
    original_process_openxr_frame = runtime.process_openxr_frame

    def process_openxr_frame(runtime_rgb, openxr_config):
        calls.append("openxr")
        return original_process_openxr_frame(runtime_rgb, openxr_config)

    runtime.process_openxr_frame = process_openxr_frame

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: calls.append("preprocess") or SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: calls.append("prepare") or "runtime-rgb",
        current_openxr_render_config=lambda: "openxr-config",
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda *args, **kwargs: None,
        set_preprocess_backend=lambda *args, **kwargs: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: calls.append("mode"),
        apply_stereo_hot_reload_if_needed=lambda: calls.append("hot-reload"),
        warmup_stereo_once_for_frame=lambda frame: calls.append("warmup"),
        log_fast_plus_fused_runtime_state=lambda result: calls.append("fused"),
    )

    RuntimePipelineLoop(context).run()

    assert runtime_q.empty()
    assert stats[expected_stat] == 1
    if stage == "preprocess":
        assert calls == ["preprocess"]
    elif stage == "prepare":
        assert calls == ["preprocess", "prepare"]
    elif stage == "depth":
        assert calls == ["preprocess", "prepare"]
        assert runtime.load_calls == 1
        assert runtime.depth_calls == 1
    else:
        assert "mode" in calls
        assert "hot-reload" in calls
        assert "warmup" in calls
        assert "fused" in calls
        assert runtime.openxr_calls == 1
        assert calls.index("openxr") < calls.index("warmup")


def test_runtime_pipeline_resets_temporal_state_when_render_size_changes():
    raw_q = queue.Queue(maxsize=2)
    runtime_q = queue.Queue(maxsize=2)
    raw_q.put(("raw-a", (2, 2), 10.0))
    raw_q.put(("raw-b", (4, 4), 11.0))
    runtime = FakeRuntime()

    context = RuntimePipelineContext(
        shutdown_event=NFrameShutdown(2),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    first_result, _first_ts = runtime_q.get_nowait()
    second_result, _second_ts = runtime_q.get_nowait()
    assert "temporal_reset_reason" not in first_result.debug_info
    assert second_result.debug_info["temporal_reset_reason"] == "render_size_changed"
    assert runtime.temporal_state.reset_count == 1


def test_runtime_pipeline_resets_temporal_state_when_source_target_changes():
    raw_q = queue.Queue(maxsize=2)
    runtime_q = queue.Queue(maxsize=2)
    raw_q.put(
        CapturedFrame(
            frame="raw-a",
            target_height=(4, 4),
            timestamp=10.0,
            capture_mode="Monitor",
            monitor_index=1,
        )
    )
    raw_q.put(
        CapturedFrame(
            frame="raw-b",
            target_height=(4, 4),
            timestamp=11.0,
            capture_mode="Monitor",
            monitor_index=2,
        )
    )
    runtime = FakeRuntime()

    context = RuntimePipelineContext(
        shutdown_event=NFrameShutdown(2),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    first_result, _first_ts = runtime_q.get_nowait()
    second_result, _second_ts = runtime_q.get_nowait()
    assert "temporal_reset_reason" not in first_result.debug_info
    assert second_result.debug_info["temporal_reset_reason"] == "source_target_changed"
    assert runtime.temporal_state.reset_count == 1


def test_runtime_pipeline_applies_latest_settings_snapshot_before_frame():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    settings_update_q = queue.Queue(maxsize=2)
    raw_q.put(("raw", (2, 2), 10.0))
    settings_update_q.put(RuntimeSettingsSnapshot(version=1, timestamp=1.0, depth_strength=1.0))
    settings_update_q.put(RuntimeSettingsSnapshot(version=2, timestamp=2.0, depth_strength=2.0))
    shutdown = OneShotShutdown()
    runtime = FakeRuntime()
    stats = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset="cinema",
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
        settings_update_q=settings_update_q,
    )

    RuntimePipelineLoop(context).run()

    assert [(snapshot.version, preset) for snapshot, preset in runtime.snapshots] == [(2, "cinema")]
    assert stats["settings_updates"] == 1
    assert stats["last_settings_version"] == 2
    assert stats["last_settings_change_class"] == SnapshotChangeClass.HOT_RELOAD.value


def test_runtime_pipeline_propagates_pipeline_rebuild_required():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    settings_update_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    settings_update_q.put(RuntimeSettingsSnapshot(version=1, timestamp=1.0, render_size_policy="scaled"))

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset="cinema",
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=PipelineRebuildRuntime(),
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
        settings_update_q=settings_update_q,
    )

    with pytest.raises(RuntimeSettingsPipelineRebuildRequired):
        RuntimePipelineLoop(context).run()


def test_runtime_pipeline_accepts_captured_frame_queue_item():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(
        CapturedFrame(
            frame="captured-raw",
            target_height=(4, 4),
            timestamp=12.5,
            capture_tool="FakeCapture",
            frame_raw_type="torch.Tensor",
            frame_raw_device="cuda",
            frame_raw_dtype="torch.uint8",
            copy_mode=FrameCopyMode.CLONE,
            metadata={"zero_copy": False},
        )
    )
    shutdown = OneShotShutdown()
    seen = {}

    def capture_frame_to_rgb(frame, size, **kwargs):
        seen["frame"] = frame
        seen["size"] = size
        seen["kwargs"] = kwargs
        return SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess",
            _d2s_preprocess_device_origin="cuda",
            _d2s_preprocess_device_output="cpu",
            _d2s_preprocess_device_transfer="cuda->cpu",
            _d2s_preprocess_input_kind="torch.Tensor",
            _d2s_capture_copy_mode="clone",
            _d2s_capture_zero_copy=False,
        )

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    assert seen["frame"] == "captured-raw"
    assert seen["size"] == (4, 4)
    assert seen["kwargs"]["frame_raw_device"] == "cuda"
    assert seen["kwargs"]["capture_copy_mode"] == "clone"
    assert seen["kwargs"]["capture_zero_copy"] is False
    runtime_result, capture_start_time = runtime_q.get_nowait()
    assert capture_start_time == 12.5
    assert runtime_result.debug_info["capture_copy_mode"] == "clone"
    assert runtime_result.debug_info["capture_zero_copy"] is False
    assert runtime_result.debug_info["capture_frame_raw_device"] == "cuda"
    assert runtime_result.debug_info["preprocess_device_origin"] == "cuda"
    assert runtime_result.debug_info["preprocess_device_transfer"] == "cuda->cpu"


def test_runtime_pipeline_resolves_4k_render_size_before_preprocess():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (3840, 2160), 10.0))
    shutdown = OneShotShutdown()
    seen = {}

    def capture_frame_to_rgb(frame, size, **kwargs):
        seen["frame"] = frame
        seen["size"] = size
        return SimpleNamespace(_d2s_preprocess_backend="fake-preprocess")

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=capture_frame_to_rgb,
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
        application_runtime_target="local_display",
        output_transport="local_window",
        render_size_config=RenderSizeConfig(
            policy=RenderSizePolicy.SCALED,
            scale_factor="1K / 50%",
            align=8,
        ),
    )

    RuntimePipelineLoop(context).run()

    assert seen == {"frame": "raw", "size": (1920, 1080)}
    runtime_result, capture_start_time = runtime_q.get_nowait()
    assert capture_start_time == 10.0
    assert runtime_result.debug_info["capture_size"] == "3840x2160"
    assert runtime_result.debug_info["render_size"] == "1920x1080"
    assert runtime_result.debug_info["render_size_policy"] == "scaled"
    assert runtime_result.debug_info["stereo_render_scale"] == "1K / 50%"
    assert runtime_result.debug_info["transport"] == "local_window"
    assert runtime_result.debug_info["application_runtime_target"] == "local_display"
    assert runtime_result.debug_info["output_transport"] == "local_window"


def test_runtime_pipeline_propagates_polling_capture_zero_copy_debug():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(
        CapturedFrame(
            frame="raw",
            target_height=(8, 4),
            timestamp=10.0,
            capture_tool="DesktopDuplication",
            capture_mode="Monitor",
            monitor_index=1,
            capture_size=(8, 4),
            frame_raw_device="",
            frame_raw_dtype="uint8",
            copy_mode=FrameCopyMode.COPY,
            metadata={"backend": "FakePollingSource", "zero_copy": False},
        )
    )

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=FakeRuntime(),
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    runtime_result, _capture_start_time = runtime_q.get_nowait()
    assert runtime_result.debug_info["capture_copy_mode"] == "copy"
    assert runtime_result.debug_info["capture_zero_copy"] is False
    assert runtime_result.debug_info["capture_tool"] == "DesktopDuplication"
    assert runtime_result.debug_info["capture_frame_raw_dtype"] == "uint8"


def test_runtime_pipeline_applies_hot_reload_snapshot_from_queue_before_runtime_call():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    settings_update_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    runtime = FakeRuntime()
    snapshot = RuntimeSettingsSnapshot(
        version=9,
        timestamp=12.0,
        source="settings_yaml_hot_reload",
        stereo_preset="game_low_latency",
        depth_strength=2.0,
    )
    stats = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="Viewer",
        openxr_runtime_direct=False,
        stereo_active_preset="cinema",
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: None,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: settings_update_q.put_nowait(snapshot),
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
        settings_update_q=settings_update_q,
    )

    RuntimePipelineLoop(context).run()

    assert runtime.snapshots == [(snapshot, "game_low_latency")]
    assert stats["settings_updates"] == 1
    assert stats["last_settings_version"] == 9
    assert stats["last_settings_change_class"] == SnapshotChangeClass.HOT_RELOAD.value
    assert runtime.rgb_calls == 1
    runtime_q.get_nowait()


def test_runtime_pipeline_passes_current_openxr_config_to_runtime():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    shutdown = OneShotShutdown()
    runtime = FakeRuntime()
    openxr_config = object()

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset="traditional_fastest",
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: openxr_config,
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    assert runtime.openxr_config is openxr_config
    runtime_result, _capture_start_time = runtime_q.get_nowait()
    assert runtime_result.debug_info["runtime_output_format"] == "openxr_rgb_depth"
    assert runtime_result.debug_info["transport"] == "openxr_swapchain"
    assert runtime.openxr_calls == 1
    assert runtime.rgb_calls == 0


def test_runtime_pipeline_openxr_non_direct_queues_full_synthesis_runtime_result():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    shutdown = OneShotShutdown()
    runtime = FakeRuntime()

    context = RuntimePipelineContext(
        shutdown_event=shutdown,
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=False,
        stereo_active_preset=None,
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: object(),
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=lambda *args, **kwargs: None,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    runtime_result, capture_start_time = runtime_q.get_nowait()
    assert runtime.rgb_calls == 1
    assert runtime.openxr_calls == 0
    assert runtime_result.depth == "depth"
    assert runtime_result.left_eye == "left-eye"
    assert runtime_result.right_eye == "right-eye"
    assert not isinstance(runtime_result, tuple)
    assert runtime_result.debug_info["runtime_output_format"] == "openxr_full_synthesis_eyes"
    assert capture_start_time == 10.0


def test_runtime_pipeline_openxr_quality_presets_use_full_synthesis_even_when_direct_enabled():
    for preset in ("cinema", "game_low_latency", "still_image_hq"):
        raw_q = queue.Queue(maxsize=1)
        runtime_q = queue.Queue(maxsize=1)
        raw_q.put(("raw", (2, 2), 10.0))
        shutdown = OneShotShutdown()
        runtime = FakeRuntime()

        context = RuntimePipelineContext(
            shutdown_event=shutdown,
            raw_q=raw_q,
            runtime_q=runtime_q,
            time_sleep=0.01,
            run_mode="OpenXR",
            openxr_runtime_direct=True,
            stereo_active_preset=preset,
            device="cpu",
            use_cudart=False,
            thread_latencies={},
            stereo_runtime=runtime,
            capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
                _d2s_preprocess_backend="fake-preprocess"
            ),
            prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
            current_openxr_render_config=lambda: object(),
            is_hard_idle=lambda: False,
            is_source_paused=lambda: False,
            log_source_health=lambda: None,
            source_stat_inc=lambda *args, **kwargs: None,
            breakdown_inc=lambda *args, **kwargs: None,
            breakdown_add_time=lambda *args, **kwargs: None,
            breakdown_add_runtime_timing=lambda result: None,
            set_preprocess_backend=lambda backend: None,
            queue_clear=lambda q: None,
            queue_drain_latest=lambda q, first_item: first_item,
            queue_put_latest=lambda q, item: q.put_nowait(item),
            log_stereo_runtime_mode_once=lambda: None,
            apply_stereo_hot_reload_if_needed=lambda: None,
            warmup_stereo_once_for_frame=lambda frame: None,
            log_fast_plus_fused_runtime_state=lambda result: None,
        )

        RuntimePipelineLoop(context).run()

        runtime_result, capture_start_time = runtime_q.get_nowait()
        assert runtime.rgb_calls == 1
        assert runtime.openxr_calls == 0
        assert runtime.skip_sbs_output is True
        assert runtime.seen_stereo_config.temporal is False
        assert runtime.seen_stereo_config.temporal_strength == 0.0
        assert runtime.seen_stereo_config.auto_reset_temporal is False
        assert runtime.seen_stereo_config.mask_feather_radius == 1
        assert runtime.stereo_config == FakeStereoConfig()
        assert runtime_result.debug_info["runtime_output_format"] == "openxr_full_synthesis_eyes"
        assert capture_start_time == 10.0


def test_runtime_pipeline_openxr_full_synthesis_enables_depth_cuda_graph_once():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (2, 2), 10.0))
    runtime = FakeRuntime()
    stats = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset="cinema",
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: object(),
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    assert runtime.config.use_cuda_graph is True
    assert [snapshot.use_cuda_graph for snapshot, _preset in runtime.snapshots] == [True]
    assert stats["openxr_depth_cuda_graph_enabled"] == 1


def test_runtime_pipeline_skips_auto_cuda_graph_for_windows_capture_cuda():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(
        CapturedFrame(
            frame="captured-raw",
            target_height=(2, 2),
            timestamp=10.0,
            capture_tool="WindowsCaptureCUDA",
            frame_raw_device="cuda",
            frame_raw_dtype="torch.uint8",
            copy_mode=FrameCopyMode.CLONE,
        )
    )
    runtime = FakeRuntime()
    stats = {}

    def source_stat_inc(name, amount=1, **values):
        stats[name] = stats.get(name, 0) + amount
        stats.update(values)

    context = RuntimePipelineContext(
        shutdown_event=OneShotShutdown(),
        raw_q=raw_q,
        runtime_q=runtime_q,
        time_sleep=0.01,
        run_mode="OpenXR",
        openxr_runtime_direct=True,
        stereo_active_preset="cinema",
        device="cpu",
        use_cudart=False,
        thread_latencies={},
        stereo_runtime=runtime,
        capture_frame_to_rgb=lambda frame, size, **kwargs: SimpleNamespace(
            _d2s_preprocess_backend="fake-preprocess"
        ),
        prepare_rgb_for_stereo_runtime=lambda frame, **kwargs: "runtime-rgb",
        current_openxr_render_config=lambda: object(),
        is_hard_idle=lambda: False,
        is_source_paused=lambda: False,
        log_source_health=lambda: None,
        source_stat_inc=source_stat_inc,
        breakdown_inc=lambda *args, **kwargs: None,
        breakdown_add_time=lambda *args, **kwargs: None,
        breakdown_add_runtime_timing=lambda result: None,
        set_preprocess_backend=lambda backend: None,
        queue_clear=lambda q: None,
        queue_drain_latest=lambda q, first_item: first_item,
        queue_put_latest=lambda q, item: q.put_nowait(item),
        log_stereo_runtime_mode_once=lambda: None,
        apply_stereo_hot_reload_if_needed=lambda: None,
        warmup_stereo_once_for_frame=lambda frame: None,
        log_fast_plus_fused_runtime_state=lambda result: None,
    )

    RuntimePipelineLoop(context).run()

    assert runtime.config.use_cuda_graph is False
    assert runtime.snapshots == []
    assert stats["openxr_depth_cuda_graph_skipped_cuda_capture"] == 1
