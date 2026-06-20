from __future__ import annotations

import signal
import sys

from app_support.cleanup import cleanup_resources


def build_cleanup_handler(
    *,
    global_processes,
    stop_capture,
    get_streamer,
    queues,
    queue_timeout,
    get_rtmp_thread,
):
    def cleanup_all_resources():
        cleanup_resources(
            global_processes=global_processes,
            stop_capture=stop_capture,
            streamer=get_streamer(),
            queues=queues,
            queue_timeout=queue_timeout,
            rtmp_thread=get_rtmp_thread(),
        )

    return cleanup_all_resources


def build_signal_handler(*, shutdown_event, cleanup_all_resources, exit_fn=sys.exit):
    def signal_handler(signum, frame):
        print(f"\n[Signal] Received signal {signum}, shutting down...")
        shutdown_event.set()
        cleanup_all_resources()
        exit_fn(0)

    return signal_handler


def register_signal_handlers(*, os_name, signal_handler):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if os_name != "Windows":
        signal.signal(signal.SIGQUIT, signal_handler)
