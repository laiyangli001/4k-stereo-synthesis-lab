from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol


@dataclass(frozen=True)
class CaptureConfig:
    output_resolution: int = 1080
    fps: int = 60
    window_title: str | None = None
    capture_mode: str = "Monitor"
    monitor_index: int = 1
    capture_tool: str | None = None
    os_name: str | None = None


@dataclass(frozen=True)
class CapturedFrame:
    frame: Any
    target_height: int
    timestamp: float


class CaptureSource(Protocol):
    def grab(self): ...
    def stop(self) -> None: ...


FrameCallback = Callable[[Any, int, float], None]
ErrorCallback = Callable[[BaseException], None]
StateCallback = Callable[[Any | None, Any | None], None]
Predicate = Callable[[], bool]
PausedCallback = Callable[[str], None]


class CaptureRunner(Protocol):
    def run(
        self,
        *,
        shutdown_event: Any,
        on_frame: FrameCallback,
        on_error: ErrorCallback | None = None,
        on_closed: Callable[[], None] | None = None,
        is_paused: Predicate | None = None,
        is_hard_idle: Predicate | None = None,
        on_paused: PausedCallback | None = None,
        on_session_update: StateCallback | None = None,
        on_tick: Callable[[], None] | None = None,
    ) -> None: ...
