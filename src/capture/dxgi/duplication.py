from __future__ import annotations


class DxgiDuplicationFrameSource:
    """Project-facing adapter around the project-bundled DXGI windows_capture build."""

    def __init__(self, monitor_index: int):
        try:
            from .windows_capture import DxgiDuplicationSession
        except ImportError:
            raise RuntimeError(
                "capture.dxgi.windows_capture module not found. "
                "Build the DXGI-enabled native package from src/capture/dxgi/native/lib.rs."
            )

        self.monitor_index = int(monitor_index)
        self._session = DxgiDuplicationSession(monitor_index=self.monitor_index)

    @property
    def session(self):
        return self._session

    @property
    def supports_gpu_frames(self) -> bool:
        return callable(getattr(self._session, "acquire_gpu_frame", None))

    def acquire_frame(self, timeout_ms: int):
        return self._session.acquire_frame(timeout_ms=timeout_ms)

    def acquire_gpu_frame(self, timeout_ms: int = 16, shared: bool = True):
        acquire = getattr(self._session, "acquire_gpu_frame", None)
        if not callable(acquire):
            raise RuntimeError(
                "capture.dxgi.windows_capture.DxgiDuplicationSession does not expose "
                "acquire_gpu_frame(); rebuild the local DXGI-enabled native package."
            )
        return acquire(timeout_ms=timeout_ms, shared=shared)

    def switch_monitor(self, monitor_index: int) -> None:
        monitor_index = int(monitor_index)
        if monitor_index == self.monitor_index:
            return
        self._session.switch_monitor(monitor_index)
        self.monitor_index = monitor_index

    def recreate(self) -> None:
        self._session.recreate()

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()
