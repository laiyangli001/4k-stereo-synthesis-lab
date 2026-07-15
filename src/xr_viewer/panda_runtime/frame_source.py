"""Adapters from existing OpenXR viewer state values into Panda frame snapshots."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

from .runtime import PandaControllerRay, PandaEyeView, PandaFrameState, PandaPose


class PandaFrameSourceError(RuntimeError):
    """Raised when existing viewer state cannot form a Panda frame snapshot."""


@dataclass(frozen=True)
class PandaFrameSourceInput:
    """Import-light source values captured from one existing OpenXR frame."""

    predicted_display_time: float
    frame_index: int | None = None
    eye_pose_mats: tuple[Any | None, Any | None] = (None, None)
    eye_fovs: tuple[Any | None, Any | None] = (None, None)
    controller_pose_mats: Mapping[str, Any] = field(default_factory=dict)
    controller_rays: Mapping[str, PandaControllerRay] = field(default_factory=dict)
    screen_pose_mat: Any | None = None
    screen_texture: Any | None = None


def build_panda_frame_state(source: PandaFrameSourceInput) -> PandaFrameState:
    """Convert one existing OpenXR frame snapshot into Panda runtime contracts."""
    eye_views = tuple(
        PandaEyeView(
            index,
            pose=mat4_to_panda_pose(pose_mat) if pose_mat is not None else None,
            fov=source.eye_fovs[index] if index < len(source.eye_fovs) else None,
        )
        for index, pose_mat in enumerate(source.eye_pose_mats)
    )
    if len(eye_views) != 2:
        raise PandaFrameSourceError("Panda frame source must contain exactly two eye pose slots")
    return PandaFrameState(
        predicted_display_time=float(source.predicted_display_time),
        frame_index=source.frame_index,
        eye_views=(eye_views[0], eye_views[1]),
        controller_poses={
            str(hand).strip().lower(): mat4_to_panda_pose(pose_mat)
            for hand, pose_mat in source.controller_pose_mats.items()
            if pose_mat is not None
        },
        controller_rays={
            str(hand).strip().lower(): ray
            for hand, ray in source.controller_rays.items()
            if ray is not None
        },
        screen_pose=mat4_to_panda_pose(source.screen_pose_mat)
        if source.screen_pose_mat is not None
        else None,
        screen_texture=source.screen_texture,
    )


def mat4_to_panda_pose(matrix: Any) -> PandaPose:
    """Convert a row-major 4x4 transform matrix to PandaPose xyz + xyzw."""
    rows = _matrix_rows(matrix)
    position = (float(rows[0][3]), float(rows[1][3]), float(rows[2][3]))
    rotation = _mat3_to_quat_xyzw(rows)
    return PandaPose(position=position, orientation=rotation)


def controller_ray_from_vectors(
    origin: Any,
    direction: Any,
    *,
    length: float = 30.0,
    visible: bool = True,
    hit_target: str = "",
) -> PandaControllerRay:
    """Create a Panda controller ray from existing smoothed ray vectors."""
    return PandaControllerRay(
        origin=_vector3(origin, "origin"),
        direction=_vector3(direction, "direction"),
        length=float(length),
        visible=bool(visible),
        hit_target=str(hit_target or ""),
    )


def _matrix_rows(matrix: Any) -> tuple[tuple[float, float, float, float], ...]:
    rows = tuple(tuple(float(value) for value in row[:4]) for row in matrix[:4])
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise PandaFrameSourceError("pose matrix must be 4x4")
    return rows


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if value is None or len(value) != 3:
        raise PandaFrameSourceError(f"controller ray {name} must be a 3D vector")
    return tuple(float(component) for component in value)


def _mat3_to_quat_xyzw(rows: tuple[tuple[float, float, float, float], ...]) -> tuple[float, float, float, float]:
    m00, m01, m02 = rows[0][0], rows[0][1], rows[0][2]
    m10, m11, m12 = rows[1][0], rows[1][1], rows[1][2]
    m20, m21, m22 = rows[2][0], rows[2][1], rows[2][2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    length = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if length <= 1e-8:
        raise PandaFrameSourceError("pose matrix rotation cannot be converted to a quaternion")
    return (qx / length, qy / length, qz / length, qw / length)
