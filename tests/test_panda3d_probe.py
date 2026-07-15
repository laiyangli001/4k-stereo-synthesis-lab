from pathlib import Path

import pytest

from pygltflib import GLTF2

from xr_viewer.panda3d_node_animation import GltfNodeAnimationRuntime
from xr_viewer.panda3d_probe import (
    _runtime_status,
    inspect_panda3d_asset,
    panda3d_probe_available,
    report_as_json,
)


ROOT = Path(__file__).resolve().parents[1]
ARTEMIS = ROOT / "src" / "xr_viewer" / "environments" / "Artemis" / "environment.glb"
CONTROLLER_ASSETS = tuple(sorted((ROOT / "src" / "xr_viewer" / "controllers").glob("*/*.glb")))
XR_GLB_ASSETS = tuple(sorted((ROOT / "src" / "xr_viewer").glob("**/*.glb")))


def _invalid_node_refs(asset: Path):
    gltf = GLTF2().load(str(asset))
    node_count = len(gltf.nodes or [])
    invalid = []
    for scene_index, scene in enumerate(gltf.scenes or []):
        for node in scene.nodes or []:
            if node < 0 or node >= node_count:
                invalid.append(("scene", scene_index, node))
    for node_index, node in enumerate(gltf.nodes or []):
        for child in node.children or []:
            if child < 0 or child >= node_count:
                invalid.append(("child", node_index, child))
    for skin_index, skin in enumerate(gltf.skins or []):
        if skin.skeleton is not None and (skin.skeleton < 0 or skin.skeleton >= node_count):
            invalid.append(("skin_skeleton", skin_index, skin.skeleton))
        for joint in skin.joints or []:
            if joint < 0 or joint >= node_count:
                invalid.append(("skin_joint", skin_index, joint))
    for animation_index, animation in enumerate(gltf.animations or []):
        for channel_index, channel in enumerate(animation.channels or []):
            target = channel.target
            if target and target.node is not None and (target.node < 0 or target.node >= node_count):
                invalid.append(("animation_target", animation_index, channel_index, target.node))
    return invalid


def test_xr_glb_assets_do_not_reference_missing_nodes():
    assert XR_GLB_ASSETS
    failures = {}
    for asset in XR_GLB_ASSETS:
        invalid = _invalid_node_refs(asset)
        if invalid:
            failures[asset.relative_to(ROOT).as_posix()] = invalid

    assert failures == {}


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


def test_panda_probe_loads_all_controller_fixtures():
    pytest.importorskip("gltf")
    if not panda3d_probe_available():
        pytest.skip("Panda3D probe dependencies are not installed")

    assert len(CONTROLLER_ASSETS) == 12
    reports = [inspect_panda3d_asset(asset) for asset in CONTROLLER_ASSETS]

    assert {asset.parent.name for asset in CONTROLLER_ASSETS} == {
        "HP",
        "INDEX",
        "PICO",
        "QUEST",
        "VIVE",
        "YVR",
    }
    for report in reports:
        assert report.gltf_animation_count == 0
        assert report.gltf_animation_channel_count == 0
        assert report.panda_node_count > 0
        assert report.panda_geom_count > 0
        assert report.animation_runtime_ready
        assert report.animation_runtime_reason == "asset has no glTF animations"
