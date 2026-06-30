# Desktop2Stereo OpenXR viewer: OpenXR and GPU resource cleanup helpers.

import ctypes

from OpenGL.GL import glDeleteBuffers, glDeleteFramebuffers, glDeleteRenderbuffers

try:
    import xr
except ImportError:
    xr = None


class CoreCleanupMixin:
    """Release OpenXR, swapchain, interop, and controller action resources."""

    def _reset_controller_actions(self):
        self._xr_actions_sync_info = None
        self._action_set = None
        self._act_left_stick = None
        self._act_right_stick = None
        self._act_menu_btn = None
        self._act_left_grip = None
        self._act_right_grip = None
        self._act_a_btn = None
        self._act_b_btn = None
        self._act_x_btn = None
        self._act_y_btn = None
        self._act_left_stick_click = None
        self._act_right_stick_click = None
        self._act_left_trigger = None
        self._act_right_trigger = None
        self._act_haptic = None
        self._act_aim_left = None
        self._act_aim_right = None
        self._act_grip_left = None
        self._act_grip_right = None
        self._path_left = None
        self._path_right = None

    def _release_openxr_swapchain_resources(self):
        self._cleanup_interop()

        for _, mgl_fbo in self._fbo_cache.values():
            try:
                mgl_fbo.release()
            except Exception:
                pass
        raw_ids = [raw_id for raw_id, _ in self._fbo_cache.values()]
        if raw_ids:
            try:
                glDeleteFramebuffers(len(raw_ids), raw_ids)
            except Exception:
                pass
        self._fbo_cache.clear()

        quad_raw_ids = []
        for mgl_fbo, raw_id, _w, _h in self._quad_fbo_cache.values():
            try:
                mgl_fbo.release()
            except Exception:
                pass
            quad_raw_ids.append(raw_id)
        if quad_raw_ids:
            try:
                glDeleteFramebuffers(len(quad_raw_ids), quad_raw_ids)
            except Exception:
                pass
        self._quad_fbo_cache.clear()

        depth_rbs = list(self._depth_rb_cache.values())
        if depth_rbs:
            try:
                glDeleteRenderbuffers(len(depth_rbs), depth_rbs)
            except Exception:
                pass
        self._depth_rb_cache.clear()

        if self._d3d11_pbo_cache:
            try:
                glDeleteBuffers(len(self._d3d11_pbo_cache), [v[0] for v in self._d3d11_pbo_cache.values()])
            except Exception:
                pass
            self._d3d11_pbo_cache.clear()

        offscreen_raw_ids = [entry[1] for entry in self._offscreen_fbo_cache.values()]
        if offscreen_raw_ids:
            try:
                glDeleteFramebuffers(len(offscreen_raw_ids), offscreen_raw_ids)
            except Exception:
                pass
        for entry in self._offscreen_fbo_cache.values():
            try:
                entry[0].release()
            except Exception:
                pass
            try:
                entry[2].release()
            except Exception:
                pass
            try:
                glDeleteRenderbuffers(1, [entry[5]])
            except Exception:
                pass
        self._offscreen_fbo_cache.clear()

    def _release_d3d11_device(self):
        if self._d3d11_native_renderer is not None:
            try:
                self._d3d11_native_renderer.cleanup()
            except Exception:
                pass
            self._d3d11_native_renderer = None
        for d3d_obj in (self._d3d11_context, self._d3d11_device):
            if d3d_obj is not None:
                try:
                    vtbl = ctypes.cast(d3d_obj, ctypes.POINTER(ctypes.c_void_p)).contents.value
                    release_fn = ctypes.CFUNCTYPE(ctypes.c_ulong, ctypes.c_void_p)(
                        ctypes.cast(vtbl + 2 * ctypes.sizeof(ctypes.c_void_p),
                                    ctypes.POINTER(ctypes.c_void_p)).contents.value
                    )
                    release_fn(d3d_obj.value)
                except Exception:
                    pass
        self._d3d11_device = None
        self._d3d11_context = None

    def _cleanup_partial_openxr(self, destroy_instance=True):
        """Tear down OpenXR session state; optionally destroy the instance too."""
        self._session_running = False
        self._session_ready_pending = False
        self._release_openxr_swapchain_resources()

        for swapchain in self._xr_swapchains.values():
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._xr_swapchains.clear()
        seen_quad_swapchains = set()
        for swapchain in self._quad_swapchains.values():
            key = int(swapchain) if isinstance(swapchain, int) else id(swapchain)
            if key in seen_quad_swapchains:
                continue
            seen_quad_swapchains.add(key)
            try:
                xr.destroy_swapchain(swapchain)
            except Exception:
                pass
        self._quad_swapchains.clear()
        self._quad_swapchain_images.clear()
        self._quad_swapchain_sizes.clear()
        self._quad_swapchain_array_size.clear()
        self._swapchain_images.clear()
        self._swapchain_sizes.clear()

        for attr in ("_xr_space", "_aim_space_l", "_aim_space_r", "_grip_space_l", "_grip_space_r"):
            sp = getattr(self, attr, None)
            if sp:
                try:
                    xr.destroy_space(sp)
                except Exception:
                    pass
                setattr(self, attr, None)

        if self._xr_session:
            try:
                xr.destroy_session(self._xr_session)
            except Exception:
                pass
            self._xr_session = None

        self._release_d3d11_device()

        if destroy_instance:
            if self._xr_instance:
                try:
                    xr.destroy_instance(self._xr_instance)
                except Exception:
                    pass
                self._xr_instance = None
            self._xr_system_id = None
            self._xr_backend = None
            self._use_d3d11 = False
            self._reset_controller_actions()