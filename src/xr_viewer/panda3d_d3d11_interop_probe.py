"""Phase-0 Panda3D OpenGL to D3D11 NV_DX_interop readiness probe."""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import importlib.util
import json
import sys
from typing import Any


@dataclass(frozen=True)
class Panda3DD3D11InteropProbeReport:
    platform: str
    panda_window_created: bool
    driver_vendor: str
    driver_renderer: str
    driver_version: str
    d3d11_device_created: bool
    d3d11_feature_level: str
    nv_dx_interop_loaded: bool
    nv_dx_device_opened: bool
    nv_dx_device_closed: bool
    swapchain_texture_registration_tested: bool
    readiness_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Panda3DD3D11InteropProbeError(RuntimeError):
    """Raised when the interop readiness probe cannot run."""


def panda3d_d3d11_interop_probe_available() -> bool:
    return sys.platform == "win32" and bool(importlib.util.find_spec("panda3d"))


def _release_com_ptr(ptr: Any) -> None:
    value = getattr(ptr, "value", ptr)
    if not value:
        return
    vtbl = ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p)).contents.value
    release = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
        ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)).contents.value
    )
    release(value)


def inspect_panda3d_d3d11_interop() -> Panda3DD3D11InteropProbeReport:
    if not panda3d_d3d11_interop_probe_available():
        raise Panda3DD3D11InteropProbeError("Panda3D D3D11 interop probe requires Windows")

    try:
        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import load_prc_file_data
        from xr_viewer import d3d_interop
    except ImportError as exc:  # pragma: no cover - guarded above for diagnostics
        raise Panda3DD3D11InteropProbeError("Panda3D D3D11 interop imports failed") from exc

    load_prc_file_data(
        "d2s-panda-d3d11-interop-probe",
        "\n".join(
            [
                "window-type offscreen",
                "load-display pandagl",
                "audio-library-name null",
                "sync-video false",
                "show-frame-rate-meter false",
                "notify-level-display error",
            ]
        ),
    )

    base: Any | None = None
    device: Any | None = None
    context: Any | None = None
    nv_handle: Any | None = None
    nv_closed = False
    try:
        base = ShowBase(windowType="offscreen")
        if not base.win:
            raise Panda3DD3D11InteropProbeError("Panda3D did not create an offscreen GL window")
        gsg = base.win.get_gsg()
        if not gsg:
            raise Panda3DD3D11InteropProbeError("Panda3D offscreen window has no GSG")

        device, context, feature_level = d3d_interop._create_d3d11_device()
        nv_loaded = bool(d3d_interop._load_nv_dx_interop())
        if nv_loaded:
            nv_handle = d3d_interop._wglDXOpenDeviceNV(device)
            if nv_handle:
                nv_closed = bool(d3d_interop._wglDXCloseDeviceNV(nv_handle))
                nv_handle = None

        nv_opened = nv_closed
        readiness = "ready_for_swapchain_texture_registration" if nv_loaded and nv_opened else "blocked"
        return Panda3DD3D11InteropProbeReport(
            platform=sys.platform,
            panda_window_created=True,
            driver_vendor=str(gsg.get_driver_vendor()),
            driver_renderer=str(gsg.get_driver_renderer()),
            driver_version=str(gsg.get_driver_version()),
            d3d11_device_created=True,
            d3d11_feature_level=f"0x{feature_level:04x}",
            nv_dx_interop_loaded=nv_loaded,
            nv_dx_device_opened=nv_opened,
            nv_dx_device_closed=nv_closed,
            swapchain_texture_registration_tested=False,
            readiness_status=readiness,
        )
    except Panda3DD3D11InteropProbeError:
        raise
    except Exception as exc:
        raise Panda3DD3D11InteropProbeError(f"Panda3D D3D11 interop probe failed: {exc}") from exc
    finally:
        if nv_handle:
            try:
                from xr_viewer import d3d_interop

                d3d_interop._wglDXCloseDeviceNV(nv_handle)
            except Exception:
                pass
        for ptr in (context, device):
            try:
                _release_com_ptr(ptr)
            except Exception:
                pass
        if base is not None:
            base.destroy()


def d3d11_interop_report_as_json(report: Panda3DD3D11InteropProbeReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
