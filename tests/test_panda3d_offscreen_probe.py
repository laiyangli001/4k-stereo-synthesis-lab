from xr_viewer.panda3d_offscreen_probe import (
    inspect_panda3d_offscreen,
    offscreen_report_as_json,
    panda3d_offscreen_probe_available,
)

import pytest


def test_panda3d_offscreen_probe_reports_render_target():
    pytest.importorskip("panda3d")
    if not panda3d_offscreen_probe_available():
        pytest.skip("Panda3D offscreen probe dependencies are not installed")

    report = inspect_panda3d_offscreen(64, 64)

    assert report.window_created
    assert report.buffer_created
    assert report.buffer_width == 64
    assert report.buffer_height == 64
    assert report.texture_width == 64
    assert report.texture_height == 64
    assert report.texture_has_ram_image
    assert report.driver_vendor
    assert report.driver_renderer
    assert report.driver_version

    report_json = offscreen_report_as_json(report)
    assert '"buffer_created": true' in report_json
    assert '"texture_has_ram_image": true' in report_json
