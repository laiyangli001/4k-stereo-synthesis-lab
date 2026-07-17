import numpy as np
import pytest

from xr_viewer.panda_runtime.coordinates import (
    gltf_position_to_panda,
    gltf_rotation_to_panda_hpr_degrees,
    gltf_scale_to_panda,
    panda_geometry_to_gltf,
)


def test_gltf_profile_coordinates_use_one_panda_axis_mapping():
    assert gltf_position_to_panda((1.0, 2.0, 3.0)) == pytest.approx(
        (1.0, -3.0, 2.0)
    )
    assert gltf_scale_to_panda((4.0, 5.0, 6.0)) == pytest.approx(
        (4.0, 6.0, 5.0)
    )
    assert gltf_rotation_to_panda_hpr_degrees(
        (np.pi / 2.0, np.pi / 4.0, np.pi / 6.0)
    ) == pytest.approx((90.0, 45.0, -30.0))


def test_panda_geometry_conversion_restores_gltf_axes_and_uvs():
    positions = np.array([[1.0, -3.0, 2.0]], dtype="f4")
    normals = np.array([[0.0, 0.0, 1.0]], dtype="f4")
    tangents = np.array([[1.0, 0.0, 0.0, -1.0]], dtype="f4")
    uv0 = np.array([[0.25, 0.75]], dtype="f4")
    uv1 = np.array([[0.5, 0.9]], dtype="f4")

    result = panda_geometry_to_gltf(
        positions,
        normals,
        tangents,
        uv0,
        uv1,
        np.eye(4, dtype="f4"),
    )

    gltf_positions, gltf_normals, gltf_tangents, gltf_uv0, gltf_uv1 = result
    np.testing.assert_allclose(gltf_positions, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(gltf_normals, [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(gltf_tangents, [[1.0, 0.0, 0.0, -1.0]])
    np.testing.assert_allclose(gltf_uv0, [[0.25, 0.25]], atol=1e-7)
    np.testing.assert_allclose(gltf_uv1, [[0.5, 0.1]], atol=1e-7)
