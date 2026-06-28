from types import SimpleNamespace

from app_runtime.shutdown import build_cleanup_handler, build_signal_handler, register_signal_handlers


def test_build_cleanup_handler_passes_current_resources(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "app_runtime.shutdown.cleanup_resources",
        lambda **kwargs: calls.append(kwargs),
    )

    cleanup = build_cleanup_handler(
        global_processes={"ffmpeg": None},
        stop_capture=lambda: True,
        get_streamer=lambda: "streamer",
        queues=["raw", "runtime"],
        queue_timeout=0.1,
        get_rtmp_thread=lambda: "rtmp",
    )
    cleanup()

    assert calls[0]["streamer"] == "streamer"
    assert calls[0]["rtmp_thread"] == "rtmp"
    assert calls[0]["queues"] == ["raw", "runtime"]


def test_signal_handler_sets_shutdown_and_starts_watchdog(monkeypatch):
    calls = []
    shutdown_event = SimpleNamespace(set=lambda: calls.append("set"))

    class FakeThread:
        def __init__(self, target, args, name, daemon):
            calls.append(("thread", target.__name__, args, name, daemon))

        def start(self):
            calls.append("start")

    monkeypatch.setattr("app_runtime.shutdown.threading.Thread", FakeThread)
    handler = build_signal_handler(
        shutdown_event=shutdown_event,
        cleanup_all_resources=lambda: calls.append("cleanup"),
        watchdog_delay=3.0,
    )

    handler(2, None)
    handler(2, None)

    assert calls == [
        "set",
        ("thread", "_force_exit_watchdog", (3.0,), "ShutdownWatchdog", True),
        "start",
        "set",
    ]


def test_signal_handler_can_preserve_immediate_exit_compatibility():
    calls = []
    shutdown_event = SimpleNamespace(set=lambda: calls.append("set"))
    handler = build_signal_handler(
        shutdown_event=shutdown_event,
        cleanup_all_resources=lambda: calls.append("cleanup"),
        exit_fn=lambda code: calls.append(("exit", code)),
    )

    handler(2, None)

    assert calls == ["set", "cleanup", ("exit", 0)]


def test_register_signal_handlers_handles_windows_sigbreak(monkeypatch):
    calls = []
    fake_signal = SimpleNamespace(
        SIGINT="INT",
        SIGTERM="TERM",
        SIGQUIT="QUIT",
        SIGBREAK="BREAK",
        signal=lambda sig, handler: calls.append((sig, handler)),
    )
    monkeypatch.setattr("app_runtime.shutdown.signal", fake_signal)

    register_signal_handlers(os_name="Windows", signal_handler="handler")

    assert calls == [("INT", "handler"), ("TERM", "handler"), ("BREAK", "handler")]
