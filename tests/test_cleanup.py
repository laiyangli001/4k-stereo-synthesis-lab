from __future__ import annotations

import queue
import subprocess

from app_runtime.cleanup import cleanup_resources


class FakeProcess:
    def __init__(self, *, timeout=False):
        self.timeout = timeout
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        if self.timeout and self.wait_calls == 1:
            raise subprocess.TimeoutExpired("fake", timeout)


class FakeStreamer:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeThread:
    def __init__(self):
        self.joined = False

    def is_alive(self):
        return True

    def join(self, timeout=None):
        self.joined = timeout


def test_cleanup_resources_stops_processes_streamer_queues_and_thread():
    processes = {
        "ffmpeg": FakeProcess(),
        "rtmp_server": FakeProcess(timeout=True),
    }
    streamer = FakeStreamer()
    rtmp_thread = FakeThread()
    q = queue.Queue()
    q.put("item")
    capture_calls = []

    cleanup_resources(
        global_processes=processes,
        stop_capture=lambda: capture_calls.append("stop") or True,
        streamer=streamer,
        queues=[q],
        queue_timeout=0.001,
        rtmp_thread=rtmp_thread,
    )

    assert processes["ffmpeg"] is None
    assert processes["rtmp_server"] is None
    assert streamer.stopped is True
    assert q.empty()
    assert capture_calls == ["stop"]
    assert rtmp_thread.joined == 3
