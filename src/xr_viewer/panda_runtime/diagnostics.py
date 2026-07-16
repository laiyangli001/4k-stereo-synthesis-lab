"""Diagnostics helpers for the optional Panda3D renderer adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class PandaRuntimeEvent:
    name: str
    detail: str
    timestamp_seconds: float


@dataclass(frozen=True)
class PandaRuntimeSnapshot:
    released: bool
    scene_assets: tuple[Mapping[str, object], ...]
    target_generation: int
    target_ready: bool
    target_refs: tuple[Mapping[str, object], ...]
    bridge_mode: str
    bridge_resource_count: int
    bridge_resource_keys: tuple[str, ...]
    render_success_count: int
    render_failure_count: int
    last_render_bridge_mode: str
    last_render_left_rendered: bool
    last_render_right_rendered: bool
    last_render_error: str
    last_render_cpu_seconds: float
    frame_predicted_display_time: float | None
    frame_animation_time_seconds: float | None
    frame_index: int | None
    frame_eye_view_count: int
    frame_controller_count: int
    animation_playback_speed: float
    animation_paused: bool
    animation_fixed_time_seconds: float | None
    animation_loop: bool
    scene_animation_time_seconds: float | None
    scene_animation_sample_count: int
    scene_animation_applied_player_count: int
    scene_animation_player_count: int
    scene_animation_channel_count: int
    scene_animation_bound_node_count: int
    scene_controller_hands: tuple[str, ...]
    scene_eye_view_count: int
    scene_applied_controller_hands: tuple[str, ...]
    event_count: int
    events: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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

    def snapshot(self, renderer: Any) -> PandaRuntimeSnapshot:
        scene = getattr(renderer, "scene", None)
        targets = getattr(renderer, "targets", None)
        bridge = getattr(renderer, "bridge", None)
        frame_state = getattr(renderer, "_last_frame_state", None)
        return PandaRuntimeSnapshot(
            released=bool(getattr(renderer, "released", False)),
            scene_assets=_scene_asset_summary(scene),
            target_generation=int(getattr(targets, "generation", 0) or 0),
            target_ready=bool(getattr(targets, "ready", False)),
            target_refs=_target_ref_summary(targets),
            bridge_mode=str(getattr(bridge, "bridge_mode", "")),
            bridge_resource_count=len(getattr(bridge, "resources", {}) or {}),
            bridge_resource_keys=_bridge_resource_key_summary(bridge),
            render_success_count=int(getattr(renderer, "_render_success_count", 0) or 0),
            render_failure_count=int(getattr(renderer, "_render_failure_count", 0) or 0),
            last_render_bridge_mode=str(getattr(renderer, "_last_render_bridge_mode", "") or ""),
            last_render_left_rendered=bool(getattr(renderer, "_last_render_left_rendered", False)),
            last_render_right_rendered=bool(getattr(renderer, "_last_render_right_rendered", False)),
            last_render_error=str(getattr(renderer, "_last_render_error", "") or ""),
            last_render_cpu_seconds=float(getattr(renderer, "_last_render_cpu_seconds", 0.0) or 0.0),
            frame_predicted_display_time=_optional_float(
                getattr(frame_state, "predicted_display_time", None)
            ),
            frame_animation_time_seconds=_optional_float(
                getattr(frame_state, "animation_time_seconds", None)
            ),
            frame_index=_optional_int(getattr(frame_state, "frame_index", None)),
            frame_eye_view_count=_eye_view_count(frame_state),
            frame_controller_count=len(getattr(frame_state, "controller_poses", {}) or {}),
            animation_playback_speed=_animation_playback_speed(renderer),
            animation_paused=_animation_paused(renderer),
            animation_fixed_time_seconds=_animation_fixed_time_seconds(renderer),
            animation_loop=bool(getattr(renderer, "_animation_loop", True)),
            scene_animation_time_seconds=_scene_optional_float(scene, "animation_time_seconds"),
            scene_animation_sample_count=_scene_int(scene, "animation_sample_count"),
            scene_animation_applied_player_count=_scene_int(scene, "animation_applied_player_count"),
            scene_animation_player_count=_scene_int(scene, "animation_player_count"),
            scene_animation_channel_count=_scene_int(scene, "animation_channel_count"),
            scene_animation_bound_node_count=_scene_int(scene, "animation_bound_node_count"),
            scene_controller_hands=_scene_controller_hands(scene),
            scene_eye_view_count=_scene_int(scene, "eye_view_count"),
            scene_applied_controller_hands=_scene_applied_controller_hands(scene),
            event_count=len(self.events),
            events=tuple(event.name for event in self.events),
        )

    def snapshot_json(self, renderer: Any) -> str:
        return json.dumps(self.snapshot(renderer).to_dict(), ensure_ascii=False, indent=2, sort_keys=True)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _eye_view_count(frame_state: Any) -> int:
    eye_views = getattr(frame_state, "eye_views", ()) or ()
    return sum(1 for eye_view in eye_views if eye_view is not None)


def _scene_snapshot(scene: Any) -> Any:
    return getattr(scene, "snapshot", None)


def _animation_playback_state(renderer: Any) -> Any:
    clock = getattr(renderer, "animation_clock", None)
    return getattr(clock, "playback_state", None)


def _animation_playback_speed(renderer: Any) -> float:
    return float(getattr(_animation_playback_state(renderer), "playback_speed", 1.0) or 0.0)


def _animation_paused(renderer: Any) -> bool:
    return bool(getattr(_animation_playback_state(renderer), "paused", False))


def _animation_fixed_time_seconds(renderer: Any) -> float | None:
    fixed = getattr(_animation_playback_state(renderer), "fixed_time_seconds", None)
    if fixed is None:
        return None
    return float(fixed)


def _scene_controller_hands(scene: Any) -> tuple[str, ...]:
    return tuple(getattr(_scene_snapshot(scene), "controller_hands", ()) or ())


def _scene_applied_controller_hands(scene: Any) -> tuple[str, ...]:
    return tuple(getattr(_scene_snapshot(scene), "applied_controller_hands", ()) or ())


def _scene_int(scene: Any, name: str) -> int:
    return int(getattr(_scene_snapshot(scene), name, 0) or 0)


def _scene_optional_float(scene: Any, name: str) -> float | None:
    value = getattr(_scene_snapshot(scene), name, None)
    if value is None:
        return None
    return float(value)


def _scene_asset_summary(scene: Any) -> tuple[Mapping[str, object], ...]:
    loaded_assets = getattr(scene, "loaded_assets", None)
    if not callable(loaded_assets):
        return ()
    return tuple(
        {
            "role": getattr(asset, "role", ""),
            "path": getattr(asset, "path", ""),
            "loaded_with_panda": bool(getattr(asset, "loaded_with_panda", False)),
            "node_count": int(getattr(asset, "node_count", 0) or 0),
            "geom_count": int(getattr(asset, "geom_count", 0) or 0),
            "animation_channel_count": int(
                getattr(asset, "animation_channel_count", 0) or 0
            ),
            "animation_target_node_count": int(
                getattr(asset, "animation_target_node_count", 0) or 0
            ),
            "animation_bound_node_count": int(
                getattr(asset, "animation_bound_node_count", 0) or 0
            ),
            "animation_duration_seconds": float(
                getattr(asset, "animation_duration_seconds", 0.0) or 0.0
            ),
        }
        for asset in loaded_assets()
    )


def _target_ref_summary(targets: Any) -> tuple[Mapping[str, object], ...]:
    target_refs = getattr(targets, "target_refs", None)
    if not callable(target_refs):
        return ()
    refs = target_refs()
    return tuple(
        {
            "eye_index": int(getattr(ref, "eye_index", -1)),
            "width": int(getattr(getattr(ref, "spec", None), "width", 0) or 0),
            "height": int(getattr(getattr(ref, "spec", None), "height", 0) or 0),
            "format": getattr(getattr(ref, "spec", None), "format", ""),
            "created_with_panda": bool(getattr(ref, "created_with_panda", False)),
            "texture_native_id_available": bool(
                getattr(ref, "texture_native_id_available", False)
            ),
            "buffer_name": getattr(ref, "buffer_name", ""),
        }
        for ref in refs
    )


def _bridge_resource_key_summary(bridge: Any) -> tuple[str, ...]:
    resources = getattr(bridge, "resources", {}) or {}
    keys = []
    for key in resources:
        keys.append(
            f"session={getattr(key, 'session_generation', '')}:"
            f"eye={getattr(key, 'eye_index', '')}:"
            f"image={getattr(key, 'image_index', '')}:"
            f"size={getattr(key, 'width', '')}x{getattr(key, 'height', '')}:"
            f"format={getattr(key, 'format', '')}"
        )
    return tuple(sorted(keys))
