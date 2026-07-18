"""Optional Panda3D frame-state sampling from the active OpenXR viewer."""

from __future__ import annotations

from typing import Any

from .panda_runtime.frame_source import (
    PandaFrameSourceInput,
    build_panda_frame_state,
)
from .panda_runtime.scene_bindings import sync_panda_scene_assets_from_viewer
from .xr_math import xr_pose_to_model_mat4

_XR_TIME_TO_SECONDS = 1e-9


def update_panda_frame_state_from_viewer(
    viewer: Any,
    *,
    predicted_display_time: float,
    views: Any,
    screen_frame_uploaded: bool,
) -> Any | None:
    """Cache a PandaFrameState on the viewer when the optional path is requested."""
    config = getattr(viewer, "_gltf_renderer_config", None)
    if not bool(getattr(config, "panda3d_requested", False)):
        return None
    try:
        frame_state = build_panda_frame_state(
            PandaFrameSourceInput(
                # XrTime is nanoseconds; Panda animation and shader clocks use seconds.
                predicted_display_time=float(predicted_display_time) * _XR_TIME_TO_SECONDS,
                frame_index=getattr(viewer, "_frame_count", None),
                projection_near=float(getattr(viewer, "_xr_projection_near", 0.05)),
                projection_far=float(getattr(viewer, "_xr_projection_far", 100.0)),
                eye_pose_mats=_eye_pose_mats(viewer, views),
                eye_fovs=_eye_fovs(views),
                controller_pose_mats=_controller_pose_mats(viewer),
            )
        )
    except Exception as exc:
        viewer._panda_frame_state_error = f"{type(exc).__name__}: {exc}"
        if not getattr(viewer, "_panda_frame_state_error_logged", False):
            print(f"[OpenXRViewer] Panda3D frame-state failed: {viewer._panda_frame_state_error}", flush=True)
            viewer._panda_frame_state_error_logged = True
        return None
    viewer._panda_frame_state = frame_state
    viewer._panda_frame_state_error = ""
    if not getattr(viewer, "_panda_frame_state_logged", False):
        print(
            "[OpenXRViewer] Panda3D frame-state active "
            f"frame_index={getattr(frame_state, 'frame_index', None)} "
            f"eye_views={len(getattr(frame_state, 'eye_views', ()) or ())} "
            f"eye_poses={sum(getattr(eye, 'pose', None) is not None for eye in (getattr(frame_state, 'eye_views', ()) or ())) } "
            f"clip={frame_state.projection_near:.3f}/{frame_state.projection_far:.1f}",
            flush=True,
        )
        viewer._panda_frame_state_logged = True
    renderer = getattr(viewer, "_panda_scene_renderer", None)
    if renderer is None:
        _log_panda_scene_binding_pending(viewer, "renderer is not initialized")
    else:
        try:
            binding = sync_panda_scene_assets_from_viewer(viewer)
            if binding is None:
                _log_panda_scene_binding_pending(viewer, getattr(viewer, "_panda_scene_binding_error", ""))
            else:
                _log_panda_scene_binding_active(viewer, binding)
            renderer.update_frame_state(frame_state)
        except Exception as exc:
            viewer._panda_frame_state_error = f"{type(exc).__name__}: {exc}"
            if not getattr(viewer, "_panda_frame_state_error_logged", False):
                print(f"[OpenXRViewer] Panda3D frame-state failed: {viewer._panda_frame_state_error}", flush=True)
                viewer._panda_frame_state_error_logged = True
    return frame_state


def _log_panda_scene_binding_pending(viewer: Any, reason: str) -> None:
    pending_key = (
        getattr(viewer, "_env_model_path", None),
        reason,
    )
    if getattr(viewer, "_panda_scene_binding_pending_key", None) == pending_key:
        return
    if reason == "renderer is not initialized":
        print("[OpenXRViewer] Panda3D scene binding pending: renderer is not initialized", flush=True)
    else:
        print(
            "[OpenXRViewer] Panda3D scene binding pending "
            f"env_path={getattr(viewer, '_env_model_path', None)!r} "
            f"error={reason!r}",
            flush=True,
        )
    viewer._panda_scene_binding_pending_key = pending_key


def _log_panda_scene_binding_active(viewer: Any, binding: Any) -> None:
    ambient_color = tuple(getattr(binding, "ambient_color", ()) or ())
    head_light_color = tuple(getattr(binding, "head_light_color", ()) or ())
    fill_light_count = int(getattr(binding, "fill_light_count", 0) or 0)
    active_key = (
        bool(getattr(binding, "loaded", False)),
        getattr(binding, "environment_path", None),
        getattr(binding, "controller_paths", None),
        ambient_color,
        head_light_color,
        fill_light_count,
    )
    if getattr(viewer, "_panda_scene_binding_active_key", None) == active_key:
        return
    print(
        "[OpenXRViewer] Panda3D scene binding active "
        f"loaded={active_key[0]} "
        f"lights=ambient:{_format_vec3(ambient_color)}/head:{_format_vec3(head_light_color)}/fill:{fill_light_count}",
        flush=True,
    )
    viewer._panda_scene_binding_active_key = active_key


def _format_vec3(value: tuple[Any, ...]) -> str:
    if len(value) < 3:
        return "0,0,0"
    try:
        return f"{float(value[0]):.3g},{float(value[1]):.3g},{float(value[2]):.3g}"
    except (TypeError, ValueError):
        return "0,0,0"


def _eye_pose_mats(viewer: Any, views: Any) -> tuple[Any | None, Any | None]:
    if not views:
        return (None, None)
    converter = getattr(viewer, "_view_pose_mat4", None)
    converted = []
    for index in range(2):
        view = views[index] if index < len(views) else None
        if view is None:
            converted.append(None)
        elif callable(converter):
            converted.append(converter(view))
        else:
            pose = getattr(view, "pose", None)
            converted.append(xr_pose_to_model_mat4(pose) if pose is not None else None)
    return (converted[0], converted[1])


def _eye_fovs(views: Any) -> tuple[Any | None, Any | None]:
    if not views:
        return (None, None)
    converted = []
    for index in range(2):
        view = views[index] if index < len(views) else None
        converted.append(getattr(view, "fov", None) if view is not None else None)
    return (converted[0], converted[1])


def _controller_pose_mats(viewer: Any) -> dict[str, Any]:
    poses = {}
    for hand, attr in (("left", "_grip_mat_l"), ("right", "_grip_mat_r")):
        pose = getattr(viewer, attr, None)
        if pose is not None:
            poses[hand] = pose
    return poses
