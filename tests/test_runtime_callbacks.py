from __future__ import annotations

import queue
from types import SimpleNamespace

from app_runtime.runtime_callbacks import RuntimeCallbacks
from stereo_runtime.settings_snapshot import RuntimeSettingsSnapshot


class FakeCounter:
    def __init__(self):
        self.calls = []

    def inc(self, *args, **kwargs):
        self.calls.append(("inc", args, kwargs))

    def add_time(self, *args, **kwargs):
        self.calls.append(("add_time", args, kwargs))

    def add_runtime_timing(self, result):
        self.calls.append(("timing", result))

    def log(self, now=None):
        self.calls.append(("log", now))

    def set_latest(self, key, value):
        self.calls.append(("latest", key, value))


class FakeHotReloader:
    def __init__(self, snapshot):
        self.snapshot = snapshot
        self.logged_values = None

    def poll_settings_snapshot_if_needed(self, *, runtime, active_preset):
        return self.snapshot, "game_low_latency", {"stereo_preset": "game_low_latency"}

    def log_settings_snapshot(self, values, *, on_mode_log):
        self.logged_values = values
        on_mode_log("hot-reload")


class FakeOpenXRState:
    def __init__(self):
        self.updated_snapshot = None

    def update_runtime_config(self, **kwargs):
        self.updated_snapshot = kwargs.get("snapshot")


class FakeRuntime:
    def apply_settings_snapshot(self, *args, **kwargs):
        raise AssertionError("RuntimeCallbacks must enqueue snapshots, not apply them directly")


def _context():
    return SimpleNamespace(
        raw_q=queue.Queue(),
        runtime_q=queue.Queue(),
        settings_update_q=queue.Queue(maxsize=1),
        fps_breakdown=FakeCounter(),
        fps_breakdown_log=True,
        source_health=FakeCounter(),
        openxr_state=SimpleNamespace(
            source_paused=lambda: False,
            hard_idle_active=lambda on_enter: False,
            update_runtime_config=lambda **kwargs: None,
            current_render_config=lambda runtime: ("config", runtime),
        ),
        stereo_runtime="runtime",
        stereo_hot_reloader=SimpleNamespace(
            apply_if_needed=lambda **kwargs: None,
        ),
        stereo_active_preset="cinema",
        stereo_runtime_logger=SimpleNamespace(
            log_mode=lambda *args, **kwargs: None,
            log_mode_once=lambda reason="active": None,
            log_fast_plus_fused_runtime_state=lambda result: None,
        ),
        stereo_warmup_tracker=SimpleNamespace(
            key_for_frame=lambda frame: ("key", frame),
            warmup_once_for_frame=lambda frame: None,
        ),
    )


def test_queue_drain_latest_records_stale_drop():
    ctx = _context()
    callbacks = RuntimeCallbacks(ctx)
    q = queue.Queue()
    q.put("newer")

    assert callbacks.queue_drain_latest(q, "old") == "newer"
    assert ("inc", ("raw_dropped_stale", 1), {}) in ctx.source_health.calls
    assert ("inc", ("raw_dropped_stale", 1), {}) in ctx.fps_breakdown.calls


def test_stop_active_capture_session_prefers_control():
    ctx = _context()
    callbacks = RuntimeCallbacks(ctx)
    calls = []
    callbacks.capture_control = SimpleNamespace(stop=lambda: calls.append("control"))
    callbacks.capture_session = SimpleNamespace(stop=lambda: calls.append("session"))

    assert callbacks.stop_active_capture_session() is True
    assert calls == ["control"]


def test_runtime_callbacks_hot_reload_enqueues_settings_snapshot():
    snapshot = RuntimeSettingsSnapshot(
        version=41,
        timestamp=2.0,
        source="settings_yaml_hot_reload",
        stereo_preset="game_low_latency",
    )
    settings_update_q = queue.Queue(maxsize=1)
    hot_reloader = FakeHotReloader(snapshot)
    openxr_state = FakeOpenXRState()
    mode_reasons = []
    context = SimpleNamespace(
        stereo_hot_reloader=hot_reloader,
        stereo_runtime=FakeRuntime(),
        stereo_active_preset="cinema",
        settings_update_q=settings_update_q,
        openxr_state=openxr_state,
        stereo_runtime_logger=SimpleNamespace(log_mode_once=lambda reason: mode_reasons.append(reason)),
    )
    callbacks = RuntimeCallbacks(context)

    assert callbacks.apply_stereo_hot_reload_if_needed() is True

    assert settings_update_q.get_nowait() is snapshot
    assert openxr_state.updated_snapshot is snapshot
    assert hot_reloader.logged_values == {"stereo_preset": "game_low_latency"}
    assert mode_reasons == ["hot-reload"]
