from __future__ import annotations

import queue
from types import SimpleNamespace

from capture.types import CapturedFrame, FrameCopyMode
from stereo_runtime.pipeline import RuntimePipelineContext, RuntimePipelineLoop
from stereo_runtime.render_size import RenderSizeConfig, RenderSizePolicy
from stereo_runtime.settings_snapshot import SnapshotChangeClass, RuntimeSettingsSnapshot


class OneShotShutdown:
    def __init__(self):
        self.calls = 0

    def is_set(self):
        self.calls += 1
        return self.calls > 2


class FakeRuntime:
    def __init__(self):
        self.openxr_config = None
        self.rgb_calls = 0
        self.openxr_calls = 0
        self.snapshots = []

    def apply_settings_snapshot(self, snapshot, *, active_preset=None):
        self.snapshots.append((snapshot, active_preset))
        return snapshot.classify()

    def process_rgb_frame(self, runtime_rgb):
        self.rgb_calls += 1
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
        stereo_runtime=FakeRuntime(),
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


def test_runtime_pipeline_resolves_render_size_before_preprocess():
    raw_q = queue.Queue(maxsize=1)
    runtime_q = queue.Queue(maxsize=1)
    raw_q.put(("raw", (1920, 1080), 10.0))
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
        render_size_config=RenderSizeConfig(
            policy=RenderSizePolicy.SCALED,
            scale_factor=0.5,
            align=8,
        ),
    )

    RuntimePipelineLoop(context).run()

    assert seen == {"frame": "raw", "size": (960, 536)}
    _runtime_result, capture_start_time = runtime_q.get_nowait()
    assert capture_start_time == 10.0


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
        assert runtime_result.debug_info["runtime_output_format"] == "openxr_full_synthesis_eyes"
        assert capture_start_time == 10.0
