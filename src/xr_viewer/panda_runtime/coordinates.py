"""Shared coordinate conversions at the glTF/Panda3D renderer boundary."""

from __future__ import annotations

import math
from typing import Sequence


def gltf_position_to_panda(position: Sequence[float]) -> tuple[float, float, float]:
    """Convert glTF/OpenXR Y-up position to Panda3D Z-up coordinates."""
    x, y, z = (float(value) for value in position)
    return x, -z, y


def gltf_scale_to_panda(scale: Sequence[float]) -> tuple[float, float, float]:
    """Reorder glTF axis scale for Panda3D without introducing reflection."""
    x, y, z = (float(value) for value in scale)
    return x, z, y


def gltf_rotation_to_panda_hpr_degrees(
    rotation_radians: Sequence[float],
) -> tuple[float, float, float]:
    """Convert the profile yaw/pitch/roll convention to Panda heading/pitch/roll."""
    yaw, pitch, roll = (float(value) for value in rotation_radians)
    return math.degrees(yaw), math.degrees(pitch), -math.degrees(roll)


def panda_geometry_to_gltf(
    positions,
    normals,
    tangents,
    uv0,
    uv1,
    world_matrix,
):
    """Convert Panda-loaded geometry back to the renderer's glTF convention.

    panda3d-gltf changes glTF (x, y, z) to Panda (x, -z, y) and flips texture
    V. This function is the single inverse boundary used when Panda geometry is
    handed to a renderer that still consumes the native glTF/OpenGL convention.
    """
    import numpy as np

    positions = np.asarray(positions, dtype=np.float32)
    normals = np.asarray(normals, dtype=np.float32)
    tangents = np.asarray(tangents, dtype=np.float32)
    uv0 = np.asarray(uv0, dtype=np.float32).copy()
    uv1 = np.asarray(uv1, dtype=np.float32).copy()
    world = np.asarray(world_matrix, dtype=np.float32)

    position4 = np.c_[positions, np.ones(len(positions), dtype=np.float32)]
    panda_positions = (position4 @ world)[:, :3]
    linear = world[:3, :3]
    try:
        panda_normals = normals @ np.linalg.inv(linear).T
    except np.linalg.LinAlgError:
        panda_normals = normals @ linear
    normal_lengths = np.linalg.norm(panda_normals, axis=1, keepdims=True)
    panda_normals /= np.maximum(normal_lengths, 1e-8)

    panda_tangents = tangents[:, :3] @ linear
    tangent_lengths = np.linalg.norm(panda_tangents, axis=1, keepdims=True)
    panda_tangents /= np.maximum(tangent_lengths, 1e-8)

    def vectors_to_gltf(values):
        return np.column_stack((values[:, 0], values[:, 2], -values[:, 1])).astype(
            np.float32
        )

    gltf_positions = vectors_to_gltf(panda_positions)
    gltf_normals = vectors_to_gltf(panda_normals)
    gltf_tangents = np.column_stack(
        (vectors_to_gltf(panda_tangents), tangents[:, 3])
    ).astype(np.float32)
    uv0[:, 1] = 1.0 - uv0[:, 1]
    uv1[:, 1] = 1.0 - uv1[:, 1]
    return gltf_positions, gltf_normals, gltf_tangents, uv0, uv1


__all__ = [
    "gltf_position_to_panda",
    "gltf_rotation_to_panda_hpr_degrees",
    "gltf_scale_to_panda",
    "panda_geometry_to_gltf",
]
