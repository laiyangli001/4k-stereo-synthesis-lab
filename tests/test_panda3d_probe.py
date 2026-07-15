from pathlib import Path

import pytest

from xr_viewer.panda3d_node_animation import GltfNodeAnimationRuntime
from xr_viewer.panda3d_probe import (
    _runtime_status,
    inspect_panda3d_asset,
    panda3d_probe_available,
    report_as_json,
)


ROOT = Path(__file__).resolve().parents[1]
ARTEMIS = ROOT / "src" / "xr_viewer" / "environments" / "Artemis" / "environment.glb"


def test_panda_probe_marks_animated_asset_without_runtime_nodes_as_blocked():
    ready, reason = _runtime_status(19, 0, 0)

    assert not ready
    assert "no runtime animation nodes" in reason


def test_panda_probe_accepts_static_assets_without_animation_runtime():
    ready, reason = _runtime_status(0, 0, 0)

    assert ready
    assert reason == "asset has no glTF animations"


def test_panda_probe_accepts_custom_node_animation_runtime():
    ready, reason = _runtime_status(19, 0, 0, 19, 19)

    assert ready
    assert "custom glTF node animation runtime" in reason


def test_panda_probe_json_is_stable_for_phase_zero_diagnostics():
    pytest.importorskip("gltf")
    if not panda3d_probe_available():
        pytest.skip("Panda3D probe dependencies are not installed")

    report = inspect_panda3d_asset(ARTEMIS)

    assert report.gltf_animation_count == 19
    assert report.gltf_animation_channel_count == 38
    assert report.gltf_animation_target_node_count == 19
    assert report.gltf_animation_targets_in_active_scene == 19
    assert report.panda_node_count > 0
    assert report.panda_geom_count > 0
    assert report.custom_node_animation_channel_count == 38
    assert report.custom_node_animation_bound_count == 19
    assert report.custom_node_animation_duration_seconds == 15.0
    assert report.animation_runtime_ready
    assert "custom glTF node animation runtime" in report.animation_runtime_reason
    probe_json = report_as_json(report)
    assert '"gltf_animation_count": 19' in probe_json
    assert '"gltf_animation_targets_in_active_scene": 19' in probe_json
    assert '"custom_node_animation_bound_count": 19' in probe_json


def test_custom_node_animation_runtime_samples_artemis_transforms():
    pytest.importorskip("gltf")
    if not panda3d_probe_available():
        pytest.skip("Panda3D probe dependencies are not installed")

    import gltf
    from panda3d.core import NodePath

    root = NodePath(gltf.load_model(str(ARTEMIS)))
    runtime = GltfNodeAnimationRuntime.from_asset(ARTEMIS, root)
    node = runtime.get_bound_node_path(1)

    assert runtime.channel_count == 38
    assert runtime.target_node_count == 19
    assert runtime.bound_node_count == 19
    assert runtime.duration_seconds == 15.0
    assert node is not None

    def matrix_values():
        matrix = node.get_mat()
        return tuple(round(matrix.get_cell(row, col), 5) for row in range(4) for col in range(4))

    runtime.apply_sample(0.0, loop=False)
    matrix_0 = matrix_values()
    runtime.apply_sample(7.5, loop=False)
    matrix_75 = matrix_values()
    runtime.apply_sample(15.0, loop=False)
    matrix_15 = matrix_values()

    assert matrix_75 != matrix_0
    assert matrix_15 == matrix_0
