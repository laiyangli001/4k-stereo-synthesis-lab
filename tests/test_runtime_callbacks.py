import queue
from types import SimpleNamespace

from app_runtime.runtime_callbacks import RuntimeCallbacks


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


def _context():
    return SimpleNamespace(
        raw_q=queue.Queue(),
        runtime_q=queue.Queue(),
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
