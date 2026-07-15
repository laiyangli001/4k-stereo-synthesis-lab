import sys

import pytest

from xr_viewer.panda3d_d3d11_interop_probe import (
    d3d11_interop_report_as_json,
    inspect_panda3d_d3d11_interop,
    panda3d_d3d11_interop_probe_available,
)


@pytest.mark.skipif(sys.platform != "win32", reason="D3D11 interop is Windows-only")
def test_panda3d_d3d11_interop_probe_opens_nv_dx_device():
    pytest.importorskip("panda3d")
    if not panda3d_d3d11_interop_probe_available():
        pytest.skip("Panda3D D3D11 interop probe dependencies are not installed")

    report = inspect_panda3d_d3d11_interop()

    assert report.panda_window_created
    assert report.d3d11_device_created
    assert report.d3d11_feature_level == "0xb000"
    assert report.nv_dx_interop_loaded
    assert report.nv_dx_device_opened
    assert report.nv_dx_device_closed
    assert report.d3d11_texture_created
    assert report.d3d11_texture_width == 64
    assert report.d3d11_texture_height == 64
    assert report.d3d11_texture_format == 28
    assert report.nv_dx_texture_registered
    assert report.nv_dx_texture_locked
    assert report.nv_dx_fbo_complete
    assert not report.swapchain_texture_registration_tested
    assert report.readiness_status == "ready_for_openxr_swapchain_texture_probe"

    report_json = d3d11_interop_report_as_json(report)
    assert '"nv_dx_device_opened": true' in report_json
    assert '"nv_dx_texture_registered": true' in report_json
    assert '"nv_dx_fbo_complete": true' in report_json
    assert '"swapchain_texture_registration_tested": false' in report_json
