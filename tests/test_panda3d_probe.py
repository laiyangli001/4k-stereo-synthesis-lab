from pathlib import Path

import pytest

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
    assert not report.animation_runtime_ready
    probe_json = report_as_json(report)
    assert '"gltf_animation_count": 19' in probe_json
    assert '"gltf_animation_targets_in_active_scene": 19' in probe_json
