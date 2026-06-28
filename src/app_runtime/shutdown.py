from __future__ import annotations

import os
import signal
import sys
import threading
import time

from app_runtime.cleanup import cleanup_resources


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


def _force_exit_watchdog(delay: float) -> None:
    time.sleep(delay)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(0)


def build_signal_handler(
    *,
    shutdown_event,
    cleanup_all_resources=None,
    exit_fn=None,
    watchdog_delay: float = 8.0,
):
    watchdog_started = False

    def signal_handler(signum, frame):
        nonlocal watchdog_started
        print(f"\n[Signal] Received signal {signum}, shutting down gracefully...")
        shutdown_event.set()
        if exit_fn is not None:
            if cleanup_all_resources is not None:
                cleanup_all_resources()
            exit_fn(0)
            return
        if not watchdog_started:
            watchdog_started = True
            threading.Thread(
                target=_force_exit_watchdog,
                args=(watchdog_delay,),
                name="ShutdownWatchdog",
                daemon=True,
            ).start()

    return signal_handler


def register_signal_handlers(*, os_name, signal_handler):
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if os_name == "Windows":
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, signal_handler)
    else:
        signal.signal(signal.SIGQUIT, signal_handler)
