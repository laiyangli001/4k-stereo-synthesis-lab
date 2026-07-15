# Desktop2Stereo OpenXR viewer: D3D11 GPU interop setup and cleanup.

import ctypes
import sys

from OpenGL.GL import (
    glBindFramebuffer,
    glCheckFramebufferStatus,
    glDeleteFramebuffers,
    glDeleteTextures,
    glFramebufferTexture2D,
    glGenFramebuffers,
    glGenTextures,
    GL_COLOR_ATTACHMENT0,
    GL_FRAMEBUFFER,
    GL_FRAMEBUFFER_COMPLETE,
    GL_TEXTURE_2D,
)

from . import d3d_interop as _d3d_interop


class CoreD3DInteropMixin:
    """D3D11 GPU interop setup and cleanup."""

    @staticmethod
    def _is_nvidia_gpu():
        """Detect NVIDIA GPU via OpenGL renderer string."""
        try:
            from OpenGL.GL import glGetString, GL_RENDERER
            r = glGetString(GL_RENDERER)
            if r:
                return b'NVIDIA' in r.upper() if isinstance(r, bytes) else 'NVIDIA' in r.upper()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                return 'NVIDIA' in torch.cuda.get_device_name(0)
        except Exception:
            pass
        return False

    def _setup_gpu_interop_d3d11(self):
        """Attempt GPU interop for D3D11 projection layers.

        NV_DX_interop2 is the only D3D11 projection path; without it the
        projection layer is skipped instead of falling back to PBO readback.
        """
        if not sys.platform == "win32":
            return

        if self._swapchain_is_bgra:
            print("[OpenXRViewer] BGRA swapchain - D3D11 projection interop disabled")
            return

        is_nv = self._is_nvidia_gpu()

        if is_nv and _d3d_interop._load_nv_dx_interop():
            try:
                self._init_interop_nv()
                self._interop_mode = 'nv_dx'
                print("[OpenXRViewer] GPU interop active: NV_DX_interop2 (zero-copy)")
                return
            except Exception as e:
                print(f"[OpenXRViewer] NV_DX_interop2 setup failed: {e}")

        self._interop_mode = None
        print("[OpenXRViewer] D3D11 GPU interop unavailable - projection layer will be skipped")

    def _disable_nv_interop_after_failure(self, reason):
        print(f"[OpenXRViewer] NV_DX_interop2 disabled after swapchain registration failure: {reason}")
        self._cleanup_interop()
        self._interop_mode = None
        print("[OpenXRViewer] D3D11 GPU interop unavailable - projection layer will be skipped")

    def _init_interop_nv(self):
        """Set up WGL_NV_DX_interop2: register the D3D11 device with GL.

        Individual swapchain textures are registered per-frame the first time
        each image index is seen (see _get_or_create_nv_interop_fbo).
        """
        self._nv_dx_device = _d3d_interop._wglDXOpenDeviceNV(self._d3d11_device)
        if not self._nv_dx_device:
            raise RuntimeError("wglDXOpenDeviceNV returned NULL")

    def _get_or_create_nv_interop_fbo(self, eye_index, img_index, d3d11_tex, w, h):
        """Register a swapchain D3D11 texture with GL via NV_DX_interop2.

        Each unique (eye, img_index) pair is registered once and cached.
        Returns (mgl_fbo, raw_fbo_id) for direct rendering into the D3D11 texture.
        """
        key = (eye_index, img_index)
        if key in self._nv_dx_objects:
            gl_tex, raw_fbo, _dx_obj = self._nv_dx_objects[key]
            return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

        gl_tex = glGenTextures(1)
        dx_obj = None
        raw_fbo = None
        # Register the D3D11 texture as a GL texture
        register_errors = []
        for access, access_name in (
            (0x0002, "WRITE_DISCARD"),  # WGL_ACCESS_WRITE_DISCARD_NV
            (0x0001, "READ_WRITE"),     # WGL_ACCESS_READ_WRITE_NV
        ):
            try:
                dx_obj = _d3d_interop._wglDXRegisterObjectNV(
                    self._nv_dx_device,
                    d3d11_tex,
                    gl_tex,
                    GL_TEXTURE_2D,
                    access,
                )
                if dx_obj:
                    if access_name != "WRITE_DISCARD":
                        print(f"[OpenXRViewer] NV_DX_interop registered with {access_name}")
                    break
                register_errors.append(f"{access_name}: returned NULL")
            except Exception as e:
                register_errors.append(f"{access_name}: {type(e).__name__}: {e}")
        if not dx_obj:
            try:
                glDeleteTextures(1, [gl_tex])
            except Exception:
                pass
            detail = "; ".join(register_errors) if register_errors else "no detail"
            raise RuntimeError(
                f"wglDXRegisterObjectNV failed for eye {eye_index} img {img_index} "
                f"(format={self._d3d11_swapchain_fmt}, {detail})"
            )

        try:
            # Set up FBO attached to the registered texture
            raw_fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, raw_fbo)
            locked = _d3d_interop._wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
            if not locked:
                raise RuntimeError("wglDXLockObjectsNV returned false")
            try:
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gl_tex, 0)
                status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
                if status != GL_FRAMEBUFFER_COMPLETE:
                    raise RuntimeError(
                        f"NV_DX_interop FBO incomplete for eye {eye_index} "
                        f"img {img_index}: {status:#x}"
                    )
            finally:
                _d3d_interop._wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
                glBindFramebuffer(GL_FRAMEBUFFER, 0)
        except Exception:
            try:
                if raw_fbo:
                    glDeleteFramebuffers(1, [raw_fbo])
            except Exception:
                pass
            try:
                _d3d_interop._wglDXUnregisterObjectNV(self._nv_dx_device, dx_obj)
            except Exception:
                pass
            try:
                glDeleteTextures(1, [gl_tex])
            except Exception:
                pass
            raise

        self._nv_dx_objects[key] = (gl_tex, raw_fbo, dx_obj)
        return self.ctx.detect_framebuffer(raw_fbo), raw_fbo

    def _lock_panda_nv_dx_image(self, eye_index, img_index):
        key = (eye_index, img_index)
        if key not in self._nv_dx_objects:
            raise RuntimeError(f"NV_DX object is not registered for eye {eye_index} img {img_index}")
        _gl_tex, _raw_fbo, dx_obj = self._nv_dx_objects[key]
        locked = _d3d_interop._wglDXLockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))
        if not locked:
            raise RuntimeError("wglDXLockObjectsNV returned false")

    def _unlock_panda_nv_dx_image(self, eye_index, img_index):
        key = (eye_index, img_index)
        if key not in self._nv_dx_objects:
            raise RuntimeError(f"NV_DX object is not registered for eye {eye_index} img {img_index}")
        _gl_tex, _raw_fbo, dx_obj = self._nv_dx_objects[key]
        _d3d_interop._wglDXUnlockObjectsNV(self._nv_dx_device, 1, ctypes.byref(dx_obj))

    def _cleanup_interop(self):
        """Release all GPU interop resources."""
        if self._interop_mode == 'nv_dx' and self._nv_dx_device:
            for (gl_tex, raw_fbo, dx_obj) in self._nv_dx_objects.values():
                try:
                    _d3d_interop._wglDXUnregisterObjectNV(self._nv_dx_device, dx_obj)
                except Exception:
                    pass
                try:
                    glDeleteFramebuffers(1, [raw_fbo])
                except Exception:
                    pass
                try:
                    glDeleteTextures(1, [gl_tex])
                except Exception:
                    pass
            self._nv_dx_objects.clear()
            try:
                _d3d_interop._wglDXCloseDeviceNV(self._nv_dx_device)
            except Exception:
                pass
            self._nv_dx_device = None

        self._interop_mode = None
