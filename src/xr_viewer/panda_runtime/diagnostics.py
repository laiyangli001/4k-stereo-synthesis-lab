"""Diagnostics helpers for the optional Panda3D renderer adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class PandaRuntimeEvent:
    name: str
    detail: str
    timestamp_seconds: float


@dataclass
class PandaRuntimeDiagnostics:
    events: list[PandaRuntimeEvent] = field(default_factory=list)

    def record_event(self, name: str, detail: str = "") -> None:
        self.events.append(
            PandaRuntimeEvent(
                name=str(name),
                detail=str(detail),
                timestamp_seconds=time.monotonic(),
            )
        )

    def summary(self) -> dict[str, object]:
        return {
            "event_count": len(self.events),
            "events": [event.name for event in self.events],
        }
