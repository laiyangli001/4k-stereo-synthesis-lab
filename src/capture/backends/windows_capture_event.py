from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes


CAPTURE_CURSOR_DELAY_S = 0.2


def _load_windows_capture(capture_tool):
    if capture_tool == "WindowsCaptureROCm":
        from wc_rocm import WindowsCapture, Frame, InternalCaptureControl
    elif capture_tool == "WindowsCaptureCUDA":
        from wc_cuda import WindowsCapture, Frame, InternalCaptureControl
    else:
        from windows_capture import WindowsCapture, Frame, InternalCaptureControl
    return WindowsCapture, Frame, InternalCaptureControl



def _setup_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        ctypes.windll.user32.SetProcessDPIAware()


class WindowsCaptureEventRunner:
    def __init__(self, config):
        self.config = config
        self.capture_tool = config.capture_tool or "WindowsCapture"
        self._capture_started_event = threading.Event()
        self._keyboard_thread = None
        self._session = None
        self._control = None

    @property
    def session(self):
        return self._session

    @property
    def control(self):
        return self._control

    def stop(self):
        if self._control is not None:
            self._control.stop()
        elif self._session is not None and hasattr(self._session, "stop"):
            self._session.stop()

    def _start_keyboard_worker(self, shutdown_event):
        user32 = ctypes.windll.user32
        user32.ShowCursor.argtypes = [wintypes.BOOL]
        user32.ShowCursor.restype = ctypes.c_int
        user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_ulonglong]
        user32.keybd_event.restype = None

        vk_menu = 0x12
        vk_tab = 0x09
        keyeventf_keyup = 0x0002

        def simulate_alt_tab():
            user32.keybd_event(vk_menu, 0, 0, 0)
            user32.keybd_event(vk_tab, 0, 0, 0)
            time.sleep(0.01)
            user32.keybd_event(vk_tab, 0, keyeventf_keyup, 0)
            user32.keybd_event(vk_menu, 0, keyeventf_keyup, 0)
            return True

        def keyboard_worker():
            while not shutdown_event.is_set():
                triggered = self._capture_started_event.wait(timeout=0.1)
                if shutdown_event.is_set():
                    break
                if not triggered:
                    continue
                try:
                    simulate_alt_tab()
                    time.sleep(0.2)
                    simulate_alt_tab()
                    if CAPTURE_CURSOR_DELAY_S:
                        time.sleep(CAPTURE_CURSOR_DELAY_S)
                except Exception as exc:
                    print(f"[keyboard] Exception during action: {exc}")
                finally:
                    break

        self._keyboard_thread = threading.Thread(target=keyboard_worker, name="CursorWorker", daemon=True)
        self._keyboard_thread.start()

    def run(
        self,
        *,
        shutdown_event,
        on_frame,
        on_error=None,
        on_closed=None,
        is_paused=None,
        is_hard_idle=None,
        on_paused=None,
        on_session_update=None,
        on_tick=None,
    ):
        _setup_dpi_awareness()
        WindowsCapture, Frame, InternalCaptureControl = _load_windows_capture(self.capture_tool)
        self._start_keyboard_worker(shutdown_event)

        while not shutdown_event.is_set():
            if on_tick is not None:
                on_tick()
            if is_hard_idle is not None and is_hard_idle():
                if on_paused is not None:
                    on_paused("hard_idle")
                time.sleep(0.1)
                continue

            if self.config.capture_mode == "Window":
                cap = WindowsCapture(window_name=self.config.window_title)
            else:
                if os.environ.get('D2S_DEBUG', '0') in ('1', 'true', 'yes', 'on'):
                    print(
                        f"[capture_loop] WindowsCapture monitor_index={self.config.monitor_index} "
                        f"tool={self.capture_tool}",
                        flush=True,
                    )
                cap = WindowsCapture(monitor_index=self.config.monitor_index)
            self._session = cap
            self._control = None
            if on_session_update is not None:
                on_session_update(self._session, self._control)

            @cap.event
            def on_frame_arrived(frame: Frame, internal_capture_control: InternalCaptureControl):
                self._control = internal_capture_control
                if on_session_update is not None:
                    on_session_update(self._session, self._control)
                capture_start_time = time.perf_counter()
                if shutdown_event.is_set():
                    return
                if (is_hard_idle is not None and is_hard_idle()) or (is_paused is not None and is_paused()):
                    if on_paused is not None:
                        on_paused("paused")
                    return
                if hasattr(frame.frame_buffer, "copy"):
                    raw = frame.frame_buffer.copy()
                else:
                    raw = frame.frame_buffer.clone()
                on_frame(raw, self.config.output_resolution, capture_start_time)

            @cap.event
            def closed():
                if on_closed is not None:
                    on_closed()

            try:
                cap.start()
            except Exception as exc:
                if on_error is not None:
                    on_error(exc)
                else:
                    raise
                time.sleep(0.5)
            finally:
                self._control = None
                self._session = None
                if on_session_update is not None:
                    on_session_update(None, None)

            if shutdown_event.is_set():
                break
            time.sleep(0.1)
