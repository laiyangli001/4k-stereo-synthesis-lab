import sys
import threading
import types

from capture import CaptureConfig, CapturedFrame, FrameCopyMode
from capture.backends import windows_capture_event


class FakeControl:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class FakeFrame:
    def __init__(self, buffer):
        self.frame_buffer = buffer


class CopyBuffer:
    def __init__(self):
        self.copied = False

    def copy(self):
        self.copied = True
        return "copied-buffer"


class CloneBuffer:
    def __init__(self):
        self.cloned = False

    def clone(self):
        self.cloned = True
        return "cloned-buffer"


def _install_capture_module(monkeypatch, module_name):
    module = types.ModuleType(module_name)

    class FakeWindowsCapture:
        last_instance = None

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.handlers = []
            FakeWindowsCapture.last_instance = self

        def event(self, handler):
            self.handlers.append(handler)
            return handler

        def start(self):
            raise RuntimeError("stop fake capture")

        def stop(self):
            self.stopped = True

    module.WindowsCapture = FakeWindowsCapture
    module.Frame = FakeFrame
    module.InternalCaptureControl = FakeControl
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


def test_load_windows_capture_selects_cuda_and_rocm_modules(monkeypatch):
    cuda = _install_capture_module(monkeypatch, "wc_cuda")
    rocm = _install_capture_module(monkeypatch, "wc_rocm")
    base = _install_capture_module(monkeypatch, "windows_capture")

    assert windows_capture_event._load_windows_capture("WindowsCaptureCUDA")[0] is cuda.WindowsCapture
    assert windows_capture_event._load_windows_capture("WindowsCaptureROCm")[0] is rocm.WindowsCapture
    assert windows_capture_event._load_windows_capture("WindowsCapture")[0] is base.WindowsCapture


def test_windows_capture_runner_uses_copy_or_clone_buffers(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_cuda")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureCUDA",
            capture_mode="Monitor",
            monitor_index=3,
            output_resolution=(3840, 2160),
        )
    )
    received = []
    shutdown_event = threading.Event()

    def on_frame(captured_frame):
        received.append(captured_frame)

    def on_error(exc):
        shutdown_event.set()

    runner.run(shutdown_event=shutdown_event, on_frame=on_frame, on_error=on_error)

    capture = module.WindowsCapture.last_instance
    assert capture.kwargs == {"monitor_index": 3}
    assert len(capture.handlers) == 2

    shutdown_event.clear()
    copy_buffer = CopyBuffer()
    capture.handlers[0](FakeFrame(copy_buffer), FakeControl())
    assert copy_buffer.copied is True
    assert isinstance(received[-1], CapturedFrame)
    assert received[-1].frame == "copied-buffer"
    assert received[-1].target_height == (3840, 2160)
    assert received[-1].copy_mode is FrameCopyMode.COPY
    assert received[-1].frame_raw_device == "cuda"
    assert received[-1].metadata["zero_copy"] is False
    assert received[-1].capture_tool == "WindowsCaptureCUDA"
    assert received[-1].capture_mode == "Monitor"
    assert received[-1].monitor_index == 3
    assert received[-1].original_format == "CopyBuffer"
    assert received[-1].metadata["backend"] == "windows_capture_event"

    shutdown_event.clear()
    clone_buffer = CloneBuffer()
    capture.handlers[0](FakeFrame(clone_buffer), FakeControl())
    assert clone_buffer.cloned is True
    assert received[-1].frame == "cloned-buffer"
    assert received[-1].copy_mode is FrameCopyMode.CLONE
    assert received[-1].frame_raw_device == "cuda"
    assert received[-1].original_format == "CloneBuffer"


def test_windows_capture_runner_marks_rocm_clone_device(monkeypatch):
    module = _install_capture_module(monkeypatch, "wc_rocm")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCaptureROCm",
            capture_mode="Monitor",
            monitor_index=2,
            output_resolution=(1920, 1080),
        )
    )
    received = []
    shutdown_event = threading.Event()

    runner.run(
        shutdown_event=shutdown_event,
        on_frame=received.append,
        on_error=lambda exc: shutdown_event.set(),
    )

    capture = module.WindowsCapture.last_instance
    clone_buffer = CloneBuffer()
    shutdown_event.clear()
    capture.handlers[0](FakeFrame(clone_buffer), FakeControl())

    assert clone_buffer.cloned is True
    assert received[-1].frame == "cloned-buffer"
    assert received[-1].copy_mode is FrameCopyMode.CLONE
    assert received[-1].frame_raw_device == "rocm"
    assert received[-1].metadata["zero_copy"] is False


def test_windows_capture_runner_uses_window_name_for_window_capture(monkeypatch):
    module = _install_capture_module(monkeypatch, "windows_capture")
    monkeypatch.setattr(windows_capture_event, "_setup_dpi_awareness", lambda: None)
    monkeypatch.setattr(windows_capture_event.WindowsCaptureEventRunner, "_start_keyboard_worker", lambda self, event: None)

    runner = windows_capture_event.WindowsCaptureEventRunner(
        CaptureConfig(
            os_name="Windows",
            capture_tool="WindowsCapture",
            capture_mode="Window",
            window_title="Stereo Viewer",
        )
    )
    shutdown_event = threading.Event()

    def on_error(exc):
        shutdown_event.set()

    runner.run(
        shutdown_event=shutdown_event,
        on_frame=lambda captured_frame: None,
        on_error=on_error,
    )

    assert module.WindowsCapture.last_instance.kwargs == {"window_name": "Stereo Viewer"}
