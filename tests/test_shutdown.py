from types import SimpleNamespace

from app_support.shutdown import build_cleanup_handler, build_signal_handler


def test_build_cleanup_handler_passes_current_resources(monkeypatch):
    calls = []

    monkeypatch.setattr(
        "app_support.shutdown.cleanup_resources",
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


def test_signal_handler_sets_shutdown_cleans_up_and_exits():
    calls = []
    shutdown_event = SimpleNamespace(set=lambda: calls.append("set"))
    handler = build_signal_handler(
        shutdown_event=shutdown_event,
        cleanup_all_resources=lambda: calls.append("cleanup"),
        exit_fn=lambda code: calls.append(("exit", code)),
    )

    handler(2, None)

    assert calls == ["set", "cleanup", ("exit", 0)]
