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
        return None
    viewer._panda_frame_state = frame_state
    viewer._panda_frame_state_error = ""
    renderer = getattr(viewer, "_panda_scene_renderer", None)
    if renderer is not None:
        try:
            sync_panda_scene_assets_from_viewer(viewer)
            renderer.update_frame_state(frame_state)
        except Exception as exc:
            viewer._panda_frame_state_error = f"{type(exc).__name__}: {exc}"
    return frame_state


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
