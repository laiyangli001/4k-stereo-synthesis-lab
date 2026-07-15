from pathlib import Path

import pytest

from xr_viewer.panda3d_animation_screenshot_probe import (
    animation_screenshot_report_as_json,
    inspect_panda3d_animation_screenshots,
    panda3d_animation_screenshot_probe_available,
)


ROOT = Path(__file__).resolve().parents[1]
ARTEMIS = ROOT / "src" / "xr_viewer" / "environments" / "Artemis" / "environment.glb"


def test_panda3d_animation_screenshot_probe_writes_phase_zero_frames(tmp_path):
    pytest.importorskip("gltf")
    if not panda3d_animation_screenshot_probe_available():
        pytest.skip("Panda3D animation screenshot probe dependencies are not installed")

    report = inspect_panda3d_animation_screenshots(
        ARTEMIS,
        tmp_path,
        width=128,
        height=128,
    )

    assert report.asset_path == str(ARTEMIS.resolve())
    assert report.width == 128
    assert report.height == 128
    assert report.duration_seconds == 15.0
    assert report.sample_times_seconds == (0.0, 7.5, 15.0)
    assert report.sampled_node_name
    assert report.transform_changed
    assert report.frame_count == 3
    assert len(report.frames) == 3
    for frame in report.frames:
        screenshot = Path(frame.screenshot_path)
        assert screenshot.is_file()
        assert screenshot.suffix == ".png"
        assert frame.screenshot_byte_length == screenshot.stat().st_size
        assert frame.screenshot_byte_length > 0
        assert len(frame.screenshot_sha256) == 64

    report_json = animation_screenshot_report_as_json(report)
    assert '"frame_count": 3' in report_json
    assert '"sample_times_seconds": [' in report_json
    assert '"transform_changed": true' in report_json
