from __future__ import annotations

import mss
import numpy as np
import win32gui
from ctypes import windll

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
    Provides direct access to the desktop frame buffer with hardware acceleration.
    """
    def __init__(self, output_resolution=1080, fps=60, window_title=None, 
                 capture_mode="Monitor", monitor_index=0):
        """
        Initialize the DXGI desktop grabber for either a window or monitor.

        Args:
            output_resolution (int): Output image height (used for scaling).
            fps (int): Frames per second (informational).
            window_title (str): Title of the window to capture (for Window mode).
            capture_mode (str): 'Window' to capture an app window, 'Monitor' to capture a screen.
            monitor_index (int): Monitor index to capture (0 = primary monitor).
        """
        try:
            from windows_capture import DxgiDuplicationSession
        except ImportError:
            raise RuntimeError(
                "windows_capture module not found. "
                "Install it with: pip install windows-capture"
            )

        self.scaled_height = output_resolution
        self.fps = fps
        self.monitor_index = monitor_index
        self.window_title = window_title
        self.capture_mode = capture_mode
        self.session = None
        self.last_frame = None
        self._frame_count = 0
        self.prev_rect = None
        self._mss = mss.mss()
        self.hwnd = None

        # Initialize based on capture mode
        if self.capture_mode == "Monitor":
            # For monitor capture, use the specified monitor index
            try:
                self.session = DxgiDuplicationSession(monitor_index=monitor_index)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to create Desktop Duplication session for monitor {monitor_index}: {e}")
        else:
            # For window capture, we need to get the window bounds
            self.hwnd = win32gui.FindWindow(None, self.window_title)
            if not self.hwnd:
                raise RuntimeError(f"Window '{self.window_title}' not found")
            # Initialize with primary monitor for window capture
            try:
                self.session = DxgiDuplicationSession(monitor_index=1)
            except RuntimeError as e:
                raise RuntimeError(f"Failed to create Desktop Duplication session: {e}")

    def _monitor_contains(self, mon, rect):
        """
        Check whether a rectangle is completely inside a monitor's bounds.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon['left'], mon['top']
        mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
        return left >= mon_left and top >= mon_top and right <= mon_right and bottom <= mon_bottom

    def _monitor_intersection_area(self, mon, rect):
        """
        Compute the area of overlap between a rectangle and a monitor.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h
        mon_left, mon_top = mon['left'], mon['top']
        mon_right, mon_bottom = mon_left + mon['width'], mon_top + mon['height']
        inter_w = max(0, min(mon_right, right) - max(mon_left, left))
        inter_h = max(0, min(mon_bottom, bottom) - max(mon_top, top))
        return inter_w * inter_h

    def _choose_monitor_and_rect(self, rect):
        """
        Select the best monitor for the window and clamp the rectangle to fit.
        """
        left, top, w, h = rect
        right, bottom = left + w, top + h

        # Check if the window is fully inside any secondary monitor (index >= 1)
        for mon in self._mss.monitors[1:]:
            if self._monitor_contains(mon, rect):
                return mon, rect

        # Find monitor with largest overlapping area
        best_mon, best_area = None, -1
        for mon in self._mss.monitors[1:]:
            area = self._monitor_intersection_area(mon, rect)
            if area > best_area:
                best_area = area
                best_mon = mon

        # Fallback to first non-primary monitor if no overlap
        if best_mon is None or best_area <= 0:
            best_mon = self._mss.monitors[1]

        # Clamp rectangle to monitor bounds
        mon_left, mon_top = best_mon['left'], best_mon['top']
        mon_right, mon_bottom = mon_left + best_mon['width'], mon_top + best_mon['height']
        new_left = max(left, mon_left)
        new_top = max(top, mon_top)
        new_right = min(right, mon_right)
        new_bottom = min(bottom, mon_bottom)
        new_w = max(0, new_right - new_left)
        new_h = max(0, new_bottom - new_top)

        # Default to full monitor if clamping results in empty area
        if new_w == 0 or new_h == 0:
            return best_mon, (mon_left, mon_top, best_mon['width'], best_mon['height'])

        return best_mon, (new_left, new_top, new_w, new_h)

    def _get_monitor_for_window(self):
        """
        Get the monitor index that the window is primarily on.
        """
        try:
            bounds = get_window_client_bounds(self.hwnd)
            if bounds is None:
                return 0

            _, rect = self._choose_monitor_and_rect(bounds)
            left, top, w, h = rect

            # Find which monitor this rectangle is on
            for idx, mon in enumerate(self._mss.monitors[1:], 1):
                if self._monitor_contains(mon, rect):
                    return idx - 1  # Convert back to DXGI monitor index

            return 0  # Default to primary monitor
        except Exception:
            return 0

    def _ensure_session_matches_window(self):
        """
        Ensure the DXGI session is on the correct monitor for window capture.
        """
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

            # Determine which monitor the window is on
            monitor_idx = self._get_monitor_for_window()

            # Switch to that monitor if needed
            if self.session and monitor_idx != self.monitor_index:
                try:
                    self.session.switch_monitor(monitor_idx)
                    self.monitor_index = monitor_idx
                except Exception as e:
                    # If switch fails, try to recreate the session
                    try:
                        self.session.recreate()
                    except Exception:
                        pass
        except Exception:
            pass

    def grab(self):
        """
        Capture a single frame from the DXGI desktop duplication session.

        Returns:
            tuple: (image_array, scaled_height) where image_array is the captured frame.
        """
        if self.session is None:
            raise RuntimeError("DXGI session is not initialized")

        # For window capture, ensure we're capturing from the right monitor
        if self.capture_mode != "Monitor":
            self._ensure_session_matches_window()

        try:
            # Attempt to acquire a frame with timeout based on FPS
            timeout_ms = max(16, int(1000 / self.fps))
            frame = self.session.acquire_frame(timeout_ms=timeout_ms)

            if frame is not None:
                # Use to_bgr() method for proper color format handling
                image_rgb = frame.to_numpy(copy=True)  # Returns BGR uint8 format

                # Cache the frame
                self.last_frame = image_rgb.copy()
                self._frame_count += 1
                return image_rgb, self.scaled_height
            else:
                # No new frame available, return last cached frame if available
                if self.last_frame is not None:
                    return self.last_frame, self.scaled_height
                else:
                    # Return a black frame as fallback
                    return np.zeros((self.scaled_height, int(self.scaled_height * 16/9), 3), dtype=np.uint8), self.scaled_height

        except RuntimeError as e:
            # Handle DXGI access loss by recreating the session
            error_str = str(e).lower()
            if any(x in error_str for x in ["access loss", "access denied", "device lost"]):
                try:
                    self.session.recreate()
                    # Retry acquiring frame after recreation
                    frame = self.session.acquire_frame(timeout_ms=33)
                    if frame is not None:
                        image_rgb = frame.to_numpy(copy=False)
                        return image_rgb, self.scaled_height
                except Exception as recreate_error:
                    raise RuntimeError(f"Failed to recreate DXGI session: {recreate_error}")
            raise

    def stop(self):
        """
        Clean up and release the Desktop Duplication session.
        """
        if self.session is not None:
            try:
                # Try to close the session if it has a close method
                if hasattr(self.session, 'close'):
                    self.session.close()
            except Exception:
                pass
            finally:
                self.session = None

        self.last_frame = None
        if self._mss:
            try:
                self._mss.close()
            except Exception:
                pass

