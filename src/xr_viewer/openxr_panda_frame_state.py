"""Optional Panda3D frame-state sampling from the active OpenXR viewer."""

from __future__ import annotations

from typing import Any

from .panda_runtime.frame_source import (
    PandaFrameSourceInput,
    build_panda_frame_state,
    controller_ray_from_vectors,
)
from .panda_runtime.scene_bindings import sync_panda_scene_assets_from_viewer


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
                predicted_display_time=predicted_display_time,
                frame_index=getattr(viewer, "_frame_count", None),
                eye_pose_mats=_eye_pose_mats(viewer, views),
                eye_fovs=_eye_fovs(views),
                controller_pose_mats=_controller_pose_mats(viewer),
                controller_rays=_controller_rays(viewer),
                screen_pose_mat=viewer._screen_pose_mat4()
                if callable(getattr(viewer, "_screen_pose_mat4", None))
                else None,
                screen_texture=_screen_texture_frame(viewer, screen_frame_uploaded),
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
            f"eye_views={len(getattr(frame_state, 'eye_views', ()) or ())}",
            flush=True,
        )
        viewer._panda_frame_state_logged = True
    ensure_screen_dimensions = getattr(viewer, "_ensure_screen_dimensions", None)
    if callable(ensure_screen_dimensions):
        try:
            ensure_screen_dimensions()
        except Exception as exc:
            viewer._panda_scene_binding_error = f"screen dimensions failed: {type(exc).__name__}: {exc}"
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
        getattr(viewer, "screen_width", None),
        getattr(viewer, "screen_height", None),
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
            f"screen={getattr(viewer, 'screen_width', None)!r}x{getattr(viewer, 'screen_height', None)!r} "
            f"error={reason!r}",
            flush=True,
        )
    viewer._panda_scene_binding_pending_key = pending_key


def _log_panda_scene_binding_active(viewer: Any, binding: Any) -> None:
    active_key = (
        bool(getattr(binding, "loaded", False)),
        bool(getattr(binding, "screen_target_bound", False)),
        getattr(binding, "screen_size", None),
        getattr(binding, "environment_path", None),
    )
    if getattr(viewer, "_panda_scene_binding_active_key", None) == active_key:
        return
    print(
        "[OpenXRViewer] Panda3D scene binding active "
        f"loaded={active_key[0]} "
        f"screen_bound={active_key[1]}",
        flush=True,
    )
    viewer._panda_scene_binding_active_key = active_key


def _eye_pose_mats(viewer: Any, views: Any) -> tuple[Any | None, Any | None]:
    if not views:
        return (None, None)
    converter = getattr(viewer, "_view_pose_mat4", None)
    if not callable(converter):
        return (None, None)
    converted = []
    for index in range(2):
        view = views[index] if index < len(views) else None
        converted.append(converter(view) if view is not None else None)
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


def _controller_rays(viewer: Any) -> dict[str, Any]:
    get_ray = getattr(viewer, "_get_smoothed_ray", None)
    if not callable(get_ray):
        return {}
    rays = {}
    for hand, is_left in (("left", True), ("right", False)):
        origin, direction = get_ray(is_left)
        if origin is None or direction is None:
            continue
        rays[hand] = controller_ray_from_vectors(origin, direction)
    return rays


def _screen_texture_frame(viewer: Any, screen_frame_uploaded: bool) -> Any | None:
    if not screen_frame_uploaded:
        return None
    return getattr(viewer, "_panda_screen_texture_frame", None)
