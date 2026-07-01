from __future__ import annotations

from ctypes import windll

import mss
import numpy as np
import win32gui

from capture.dxgi import DxgiDuplicationFrameSource

try:
    windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    windll.user32.SetProcessDPIAware()


def get_window_client_bounds(hwnd):
    rc = win32gui.GetClientRect(hwnd)
    if rc is None:
        raise Exception(f"Window not found {hwnd}")

    left, top, right, bottom = rc
    w = right - left
    h = bottom - top
    left, top = win32gui.ClientToScreen(hwnd, (left, top))
    return left, top, w, h


class DesktopGrabber:
    """
    Desktop grabber using Windows DXGI Desktop Duplication API.
    It is a capture source; PollingCaptureRunner owns CapturedFrame wrapping.
    """

    def __init__(
        self,
        output_resolution=1080,
        fps=60,
        window_title=None,
        capture_mode="Monitor",
        monitor_index=0,
    ):
        self.scaled_height = output_resolution
        self.fps = fps
        self.monitor_index = monitor_index
        self.window_title = window_title
        self.capture_mode = capture_mode
        self._dxgi_source = None
        self.last_frame = None
        self._frame_count = 0
        self.prev_rect = None
        self._mss = mss.mss()
        self.hwnd = None

        if self.capture_mode == "Monitor":
            self._create_dxgi_source(monitor_index, f"for monitor {monitor_index}")
        else:
            self.hwnd = win32gui.FindWindow(None, self.window_title)
            if not self.hwnd:
                raise RuntimeError(f"Window '{self.window_title}' not found")
            self._create_dxgi_source(1, "")

    @property
    def session(self):
        return self._dxgi_source.session if self._dxgi_source is not None else None

    def _create_dxgi_source(self, monitor_index, context):
        try:
            self._dxgi_source = DxgiDuplicationFrameSource(monitor_index=monitor_index)
            self.monitor_index = int(monitor_index)
        except RuntimeError as e:
            suffix = f" {context}" if context else ""
            raise RuntimeError(f"Failed to create Desktop Duplication session{suffix}: {e}")

    def _monitor_contains(self, mon, rect):
        """Check whether a rectangle is completely inside a monitor's bounds."""
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon["left"], mon["top"]
        mon_right, mon_bottom = mon_left + mon["width"], mon_top + mon["height"]
        return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

    def _monitor_intersection_area(self, mon, rect):
        """Compute the area of overlap between a rectangle and a monitor."""
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon["left"], mon["top"]
        mon_right, mon_bottom = mon_left + mon["width"], mon_top + mon["height"]
        inter_w = max(0, min(mon_right, right) - max(mon_left, left))
        inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
        return inter_w * inter_h

    def _choose_monitor_and_rect(self, rect):
        """Select the best monitor for the window and clamp the rectangle to fit."""
        left, top, w, h = rect
        right, bottom = left + w, top + h

        for mon in self._mss.monitors[1:]:
            if self._monitor_contains(mon, rect):
                return mon, rect

        best_mon, best_area = None, -1
        for mon in self._mss.monitors[1:]:
            area = self._monitor_intersection_area(mon, rect)
            if area > best_area:
                best_area = area
                best_mon = mon

        if best_mon is None or best_area <= 0:
            best_mon = self._mss.monitors[1]

        mon_left, mon_top = best_mon["left"], best_mon["top"]
        mon_right, mon_bottom = mon_left + best_mon["width"], mon_top + best_mon["height"]
        new_left = max(left, mon_left)
        new_top = max(top, mon_top)
        new_right = min(right, mon_right)
        new_bottom = min(bottom, mon_bottom)
        new_w = max(0, new_right - new_left)
        new_h = max(0, new_bottom - new_top)

        if new_w == 0 or new_h == 0:
            return best_mon, (mon_left, mon_top, best_mon["width"], best_mon["height"])

        return best_mon, (new_left, new_top, new_w, new_h)

    def _get_monitor_for_window(self):
        """Get the monitor index that the window is primarily on."""
        try:
            bounds = get_window_client_bounds(self.hwnd)
            if bounds is None:
                return 0

            _, rect = self._choose_monitor_and_rect(bounds)

            for idx, mon in enumerate(self._mss.monitors[1:], 1):
                if self._monitor_contains(mon, rect):
                    return idx - 1

            return 0
        except Exception:
            return 0

    def _ensure_session_matches_window(self):
        """Ensure the DXGI source is on the correct monitor for window capture."""
        if self.capture_mode == "Monitor":
            return

        try:
            bounds = get_window_client_bounds(self.hwnd)
            if bounds is None:
                self.prev_rect = None
                return

            if bounds == self.prev_rect:
                return

            self.prev_rect = bounds
            monitor_idx = self._get_monitor_for_window()

            if self._dxgi_source is not None and monitor_idx != self.monitor_index:
                try:
                    self._dxgi_source.switch_monitor(monitor_idx)
                    self.monitor_index = monitor_idx
                except Exception:
                    try:
                        self._dxgi_source.recreate()
                    except Exception:
                        pass
        except Exception:
            pass

    def grab(self):
        """
        Capture a single frame from the DXGI desktop duplication source.

        Returns:
            tuple: (image_array, scaled_height) where image_array is the captured frame.
        """
        if self._dxgi_source is None:
            raise RuntimeError("DXGI source is not initialized")

        if self.capture_mode != "Monitor":
            self._ensure_session_matches_window()

        try:
            timeout_ms = max(16, int(1000 / self.fps))
            frame = self._dxgi_source.acquire_frame(timeout_ms=timeout_ms)

            if frame is not None:
                image_rgb = frame.to_numpy(copy=True)
                self.last_frame = image_rgb.copy()
                self._frame_count += 1
                return image_rgb, self.scaled_height

            if self.last_frame is not None:
                return self.last_frame, self.scaled_height
            return (
                np.zeros((self.scaled_height, int(self.scaled_height * 16 / 9), 3), dtype=np.uint8),
                self.scaled_height,
            )

        except RuntimeError as e:
            error_str = str(e).lower()
            if any(x in error_str for x in ["access loss", "access denied", "device lost"]):
                try:
                    self._dxgi_source.recreate()
                    frame = self._dxgi_source.acquire_frame(timeout_ms=33)
                    if frame is not None:
                        image_rgb = frame.to_numpy(copy=False)
                        return image_rgb, self.scaled_height
                except Exception as recreate_error:
                    raise RuntimeError(f"Failed to recreate DXGI source: {recreate_error}")
            raise

    def stop(self):
        """Clean up and release the Desktop Duplication source."""
        if self._dxgi_source is not None:
            try:
                self._dxgi_source.close()
            except Exception:
                pass
            finally:
                self._dxgi_source = None

        self.last_frame = None
        if self._mss:
            try:
                self._mss.close()
            except Exception:
                pass
