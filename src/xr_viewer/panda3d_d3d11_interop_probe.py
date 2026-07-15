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
    d3d11_adapter_description: str
    d3d11_adapter_vendor_id: str
    d3d11_adapter_device_id: str
    d3d11_adapter_luid: str
    d3d11_adapter_dedicated_video_memory: int
    gl_d3d_adapter_name_match: bool
    nv_dx_interop_loaded: bool
    nv_dx_device_opened: bool
    nv_dx_device_closed: bool
    d3d11_texture_created: bool
    d3d11_texture_width: int
    d3d11_texture_height: int
    d3d11_texture_format: int
    nv_dx_texture_registered: bool
    nv_dx_texture_locked: bool
    nv_dx_fbo_complete: bool
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


def _com_fn(obj: Any, index: int, restype: Any, *argtypes: Any) -> Any:
    value = getattr(obj, "value", obj)
    vtbl = ctypes.cast(value, ctypes.POINTER(ctypes.c_void_p)).contents.value
    fn_ptr = ctypes.cast(
        vtbl + index * ctypes.sizeof(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ).contents.value
    return ctypes.CFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn_ptr)


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_uint32),
        ("Data2", ctypes.c_uint16),
        ("Data3", ctypes.c_uint16),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _DXGIAdapterDesc(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint),
        ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint),
        ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID),
    ]


_GUID_DATA4 = ctypes.c_ubyte * 8
_IID_IDXGI_DEVICE = _GUID(
    0x54EC77FA,
    0x1377,
    0x44E6,
    _GUID_DATA4(0x8C, 0x32, 0x88, 0xFD, 0x5F, 0x44, 0xC8, 0x4C),
)


def _adapter_luid_text(luid: _LUID) -> str:
    return f"{luid.HighPart & 0xFFFFFFFF:08x}:{luid.LowPart:08x}"


def _d3d11_adapter_desc(device: Any) -> _DXGIAdapterDesc:
    idxgi_device = ctypes.c_void_p()
    adapter = ctypes.c_void_p()
    try:
        query_interface = _com_fn(
            device,
            0,
            ctypes.c_long,
            ctypes.POINTER(_GUID),
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = query_interface(
            getattr(device, "value", device),
            ctypes.byref(_IID_IDXGI_DEVICE),
            ctypes.byref(idxgi_device),
        )
        if hr != 0:
            raise Panda3DD3D11InteropProbeError(
                f"ID3D11Device QueryInterface(IDXGIDevice) failed: hr=0x{hr & 0xFFFFFFFF:08x}"
            )
        get_adapter = _com_fn(
            idxgi_device,
            7,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_void_p),
        )
        hr = get_adapter(idxgi_device.value, ctypes.byref(adapter))
        if hr != 0:
            raise Panda3DD3D11InteropProbeError(
                f"IDXGIDevice.GetAdapter failed: hr=0x{hr & 0xFFFFFFFF:08x}"
            )
        desc = _DXGIAdapterDesc()
        get_desc = _com_fn(
            adapter,
            8,
            ctypes.c_long,
            ctypes.POINTER(_DXGIAdapterDesc),
        )
        hr = get_desc(adapter.value, ctypes.byref(desc))
        if hr != 0:
            raise Panda3DD3D11InteropProbeError(
                f"IDXGIAdapter.GetDesc failed: hr=0x{hr & 0xFFFFFFFF:08x}"
            )
        return desc
    finally:
        for ptr in (adapter, idxgi_device):
            try:
                _release_com_ptr(ptr)
            except Exception:
                pass


def _adapter_name_matches_gl_renderer(adapter_description: str, gl_renderer: str) -> bool:
    adapter_tokens = [token for token in adapter_description.lower().replace("/", " ").split() if token]
    renderer = gl_renderer.lower()
    return bool(adapter_tokens) and all(token in renderer for token in adapter_tokens)


def _create_probe_texture(device: Any, width: int, height: int) -> Any:
    from xr_viewer.d3d11_native_renderer import (
        D3D11_BIND_RENDER_TARGET,
        D3D11_BIND_SHADER_RESOURCE,
        D3D11_USAGE_DEFAULT,
        D3D11Texture2DDesc,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGISampleDesc,
    )

    texture = ctypes.c_void_p()
    desc = D3D11Texture2DDesc(
        width,
        height,
        1,
        1,
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGISampleDesc(1, 0),
        D3D11_USAGE_DEFAULT,
        D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
        0,
        0,
    )
    create_texture = _com_fn(
        device,
        5,
        ctypes.c_long,
        ctypes.POINTER(D3D11Texture2DDesc),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    )
    hr = create_texture(getattr(device, "value", device), ctypes.byref(desc), None, ctypes.byref(texture))
    if hr != 0:
        raise Panda3DD3D11InteropProbeError(f"CreateTexture2D failed: hr=0x{hr & 0xFFFFFFFF:08x}")
    return texture


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
    texture: Any | None = None
    dx_object: Any | None = None
    gl_texture = 0
    gl_fbo = 0
    nv_closed = False
    texture_registered = False
    texture_locked = False
    fbo_complete = False
    width = 64
    height = 64
    texture_format = 28
    try:
        from OpenGL.raw.WGL._types import HANDLE
        from OpenGL.GL import (
            GL_COLOR_ATTACHMENT0,
            GL_FRAMEBUFFER,
            GL_FRAMEBUFFER_COMPLETE,
            GL_LINEAR,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            glBindFramebuffer,
            glBindTexture,
            glCheckFramebufferStatus,
            glDeleteFramebuffers,
            glDeleteTextures,
            glFramebufferTexture2D,
            glGenFramebuffers,
            glGenTextures,
            glTexParameteri,
        )

        base = ShowBase(windowType="offscreen")
        if not base.win:
            raise Panda3DD3D11InteropProbeError("Panda3D did not create an offscreen GL window")
        gsg = base.win.get_gsg()
        if not gsg:
            raise Panda3DD3D11InteropProbeError("Panda3D offscreen window has no GSG")

        device, context, feature_level = d3d_interop._create_d3d11_device()
        adapter_desc = _d3d11_adapter_desc(device)
        adapter_description = str(adapter_desc.Description).rstrip("\x00")
        gl_renderer = str(gsg.get_driver_renderer())
        nv_loaded = bool(d3d_interop._load_nv_dx_interop())
        if nv_loaded:
            nv_handle = d3d_interop._wglDXOpenDeviceNV(device)
            if nv_handle:
                texture = _create_probe_texture(device, width, height)
                gl_texture = int(glGenTextures(1))
                glBindTexture(GL_TEXTURE_2D, gl_texture)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glBindTexture(GL_TEXTURE_2D, 0)

                raw_dx_object = d3d_interop._wglDXRegisterObjectNV(
                    nv_handle,
                    texture,
                    gl_texture,
                    GL_TEXTURE_2D,
                    0x0002,  # WGL_ACCESS_WRITE_DISCARD_NV
                )
                dx_object = HANDLE(raw_dx_object) if raw_dx_object else None
                texture_registered = bool(dx_object)
                if texture_registered:
                    texture_locked = bool(
                        d3d_interop._wglDXLockObjectsNV(nv_handle, 1, ctypes.byref(dx_object))
                    )
                    if texture_locked:
                        gl_fbo = int(glGenFramebuffers(1))
                        glBindFramebuffer(GL_FRAMEBUFFER, gl_fbo)
                        glFramebufferTexture2D(
                            GL_FRAMEBUFFER,
                            GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D,
                            gl_texture,
                            0,
                        )
                        fbo_complete = glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE
                        glBindFramebuffer(GL_FRAMEBUFFER, 0)
                        d3d_interop._wglDXUnlockObjectsNV(nv_handle, 1, ctypes.byref(dx_object))
                if dx_object:
                    d3d_interop._wglDXUnregisterObjectNV(nv_handle, dx_object)
                    dx_object = None
                nv_closed = bool(d3d_interop._wglDXCloseDeviceNV(nv_handle))
                nv_handle = None

        nv_opened = bool(nv_loaded and (texture is not None or nv_closed))
        readiness = "ready_for_openxr_swapchain_texture_probe" if fbo_complete else "blocked"
        return Panda3DD3D11InteropProbeReport(
            platform=sys.platform,
            panda_window_created=True,
            driver_vendor=str(gsg.get_driver_vendor()),
            driver_renderer=str(gsg.get_driver_renderer()),
            driver_version=str(gsg.get_driver_version()),
            d3d11_device_created=True,
            d3d11_feature_level=f"0x{feature_level:04x}",
            d3d11_adapter_description=adapter_description,
            d3d11_adapter_vendor_id=f"0x{adapter_desc.VendorId:04x}",
            d3d11_adapter_device_id=f"0x{adapter_desc.DeviceId:04x}",
            d3d11_adapter_luid=_adapter_luid_text(adapter_desc.AdapterLuid),
            d3d11_adapter_dedicated_video_memory=int(adapter_desc.DedicatedVideoMemory),
            gl_d3d_adapter_name_match=_adapter_name_matches_gl_renderer(
                adapter_description,
                gl_renderer,
            ),
            nv_dx_interop_loaded=nv_loaded,
            nv_dx_device_opened=nv_opened,
            nv_dx_device_closed=nv_closed,
            d3d11_texture_created=texture is not None,
            d3d11_texture_width=width if texture is not None else 0,
            d3d11_texture_height=height if texture is not None else 0,
            d3d11_texture_format=texture_format if texture is not None else 0,
            nv_dx_texture_registered=texture_registered,
            nv_dx_texture_locked=texture_locked,
            nv_dx_fbo_complete=fbo_complete,
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

                if dx_object:
                    d3d_interop._wglDXUnregisterObjectNV(nv_handle, dx_object)
                d3d_interop._wglDXCloseDeviceNV(nv_handle)
            except Exception:
                pass
        if gl_fbo:
            try:
                from OpenGL.GL import glDeleteFramebuffers

                glDeleteFramebuffers(1, [gl_fbo])
            except Exception:
                pass
        if gl_texture:
            try:
                from OpenGL.GL import glDeleteTextures

                glDeleteTextures(1, [gl_texture])
            except Exception:
                pass
        if texture:
            try:
                _release_com_ptr(texture)
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
