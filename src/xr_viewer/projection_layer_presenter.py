import ctypes
import time

import glfw
import numpy as np
from OpenGL.GL import (
    GL_COLOR_ATTACHMENT0,
    GL_COLOR_BUFFER_BIT,
    GL_DRAW_FRAMEBUFFER,
    GL_LINEAR,
    GL_READ_FRAMEBUFFER,
    glBindFramebuffer,
    glBlitFramebuffer,
    glReadBuffer,
)

from . import d3d_interop as _d3d_interop
from .panda_runtime.bridge import SwapchainImageRef
from .panda_runtime.opengl_bridge import PandaOpenGLBridge
from .panda_runtime.stereo_targets import StereoTargetSpec
from .xr_math import _fov_to_proj_mat4, _fov_to_proj_mat4_d3d, _pose_to_view_mat4

try:
    import xr
except ImportError:
    xr = None


def _record_panda_bridge_timing(viewer, timing):
    snapshot = {str(name): float(seconds) for name, seconds in timing.items()}
    viewer._panda_render_last_timing = snapshot
    recorder = getattr(viewer, '_breakdown_add_time', None)
    if not callable(recorder):
        return
    for name, seconds in snapshot.items():
        recorder(f'openxr_projection_panda_{name}', seconds)


def _log_panda_bridge_diagnostics(viewer):
    _log_panda_render_path_status(viewer, status='bridge-active')


def _panda_opengl_bridge_backoff_active(viewer, now=None):
    if now is None:
        now = time.perf_counter()
    until = float(getattr(viewer, '_panda_opengl_bridge_disabled_until', 0.0) or 0.0)
    return now < until


def _make_viewer_gl_context_current(viewer):
    window = getattr(viewer, 'window', None)
    if window is not None:
        glfw.make_context_current(window)


def _log_panda_render_path_status(viewer, *, status, reason=''):
    renderer = getattr(viewer, '_panda_scene_renderer', None)
    if renderer is None:
        return
    timing = getattr(viewer, '_panda_render_last_timing', {}) or {}
    timing_text = ','.join(f'{name}={seconds * 1000.0:.2f}ms' for name, seconds in timing.items())
    snapshot = None
    snapshot_error = ''
    try:
        snapshot = renderer.diagnostics_snapshot()
    except Exception as exc:
        snapshot_error = f'{type(exc).__name__}: {exc}'
    viewer_success = int(getattr(viewer, '_panda_render_success_count', 0) or 0)
    viewer_failure = int(getattr(viewer, '_panda_render_failure_count', 0) or 0)
    bridge_mode = getattr(viewer, '_panda_render_last_bridge_mode', '')
    renderer_success = getattr(snapshot, 'render_success_count', 0) if snapshot is not None else 0
    renderer_failure = getattr(snapshot, 'render_failure_count', 0) if snapshot is not None else 0
    animation_samples = getattr(snapshot, 'scene_animation_sample_count', 0) if snapshot is not None else 0
    key = (
        status,
        reason,
        bridge_mode,
        getattr(viewer, '_panda_render_error', ''),
        getattr(viewer, '_panda_render_last_target_size', None),
    )
    if getattr(viewer, '_panda_render_path_status_key', None) == key:
        return
    log_count = int(getattr(viewer, '_panda_render_diagnostic_log_count', 0) or 0)
    if log_count >= 8:
        return
    viewer._panda_render_path_status_key = key
    viewer._panda_render_diagnostic_log_count = log_count + 1
    print(
        '[OpenXRViewer] Panda3D render path status '
        f'status={status} '
        f'reason={reason!r} '
        f'backend={"d3d11" if bool(getattr(viewer, "_use_d3d11", False)) else "opengl"} '
        f'viewer_success={viewer_success} '
        f'viewer_failure={viewer_failure} '
        f'bridge={bridge_mode!r} '
        f'renderer_success={renderer_success} '
        f'renderer_failure={renderer_failure} '
        f'animation_samples={animation_samples} '
        f'target_size={getattr(viewer, "_panda_render_last_target_size", None)!r} '
        f'image_indices={getattr(viewer, "_panda_render_last_image_indices", None)!r} '
        f'error={getattr(viewer, "_panda_render_error", "")!r} '
        f'snapshot_error={snapshot_error!r} '
        f'timing={{{timing_text}}}',
        flush=True,
    )


class ProjectionLayerPresenter:
    def __init__(self, viewer):
        self.viewer = viewer

    def _projection_clip_planes(self):
        near = max(0.01, float(getattr(self.viewer, '_xr_projection_near', 0.05) or 0.05))
        far = max(near + 1.0, float(getattr(self.viewer, '_xr_projection_far', 100.0) or 100.0))
        return near, far

    def render_projection(self, *, enabled, views, default_fov, default_proj, default_proj_d3d, updated_quad_eyes=()):
        viewer = self.viewer
        if not enabled:
            return []
        if not viewer._use_d3d11:
            ensure_swapchains = getattr(viewer, '_ensure_projection_swapchains', None)
            if callable(ensure_swapchains) and not ensure_swapchains():
                return []
            if self._panda_opengl_bridge_enabled() and not _panda_opengl_bridge_backoff_active(viewer):
                viewer._panda_opengl_bridge_failed_this_frame = False
                panda_views = self.render_panda_opengl_bridge(
                    views,
                    default_fov,
                    updated_quad_eyes=updated_quad_eyes,
                )
                if panda_views:
                    return panda_views
                if bool(getattr(viewer, '_panda_opengl_bridge_failed_this_frame', False)):
                    return []
            return self.render_opengl(
                views,
                default_fov,
                default_proj,
                updated_quad_eyes=updated_quad_eyes,
            )
        phase0_probe = bool(getattr(viewer, '_panda3d_phase0_swapchain_probe_enabled', False))
        if phase0_probe and viewer._interop_mode == 'nv_dx':
            return self.render_nv_dx_interop(
                views,
                default_fov,
                default_proj,
                phase0_probe=True,
            )
        if self._panda_bridge_enabled():
            panda_views = self.render_panda_bridge(
                views,
                default_fov,
            )
            if panda_views:
                return panda_views
        if viewer._d3d11_native_renderer is not None:
            return self.render_d3d11_native(
                views,
                default_fov,
                default_proj_d3d,
            )
        if viewer._interop_mode == 'nv_dx':
            return self.render_nv_dx_interop(
                views,
                default_fov,
                default_proj,
            )
        viewer._breakdown_inc('openxr_projection_d3d11_no_interop_skip')
        return []

    def _panda_opengl_bridge_enabled(self):
        viewer = self.viewer
        config = getattr(viewer, '_gltf_renderer_config', None)
        return bool(
            getattr(config, 'panda3d_enabled', False)
            and getattr(viewer, '_panda_scene_renderer', None) is not None
            and not getattr(viewer, '_use_d3d11', False)
        )

    def _ensure_panda_opengl_bridge(self, renderer):
        if not isinstance(getattr(renderer, 'bridge', None), PandaOpenGLBridge):
            renderer.bridge = PandaOpenGLBridge(
                make_target_context_current=lambda: _make_viewer_gl_context_current(self.viewer)
            )
        else:
            renderer.bridge.make_target_context_current = lambda: _make_viewer_gl_context_current(self.viewer)
        targets = getattr(renderer, 'targets', None)
        if targets is not None and not bool(getattr(targets, 'create_panda_targets', False)):
            targets.create_panda_targets = True
    def _panda_bridge_enabled(self):
        viewer = self.viewer
        config = getattr(viewer, '_gltf_renderer_config', None)
        return bool(
            getattr(config, 'panda3d_enabled', False)
            and getattr(viewer, '_panda_scene_renderer', None) is not None
            and getattr(viewer, '_use_d3d11', False)
        )

    def render_panda_opengl_bridge(self, views, default_fov, *, updated_quad_eyes=()):
        viewer = self.viewer
        renderer = getattr(viewer, '_panda_scene_renderer', None)
        if renderer is None:
            return []
        self._ensure_panda_opengl_bridge(renderer)
        _make_viewer_gl_context_current(viewer)
        acquired = []
        released = set()
        total_start = time.perf_counter()
        timing = {
            'acquire_wait': 0.0,
            'target_rebuild': 0.0,
            'bridge_render': 0.0,
            'release': 0.0,
            'total': 0.0,
        }
        try:
            for eye_index in range(2):
                acquire_start = time.perf_counter()
                swapchain = viewer._xr_swapchains[eye_index]
                img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
                viewer._wait_swapchain_image(swapchain)
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                raw_fbo, mgl_fbo = viewer._get_or_create_fbo(
                    eye_index, img_index, sc_image.image, sc_w, sc_h
                )
                view = views[eye_index] if views and views[eye_index] else None
                acquired.append((eye_index, swapchain, img_index, raw_fbo, mgl_fbo, sc_w, sc_h, view))
                timing['acquire_wait'] += time.perf_counter() - acquire_start

            left = acquired[0]
            right = acquired[1]
            fmt = getattr(viewer, '_opengl_swapchain_fmt', 'rgba8')
            generation = int(getattr(viewer, '_panda_swapchain_session_generation', 0) or 0)
            rebuild_start = time.perf_counter()
            renderer.rebuild_targets(
                StereoTargetSpec(left[5], left[6], fmt),
                StereoTargetSpec(right[5], right[6], fmt),
            )
            timing['target_rebuild'] = time.perf_counter() - rebuild_start
            left_image = SwapchainImageRef(left[0], left[2], left[4], left[5], left[6], fmt, generation)
            right_image = SwapchainImageRef(right[0], right[2], right[4], right[5], right[6], fmt, generation)
            bridge_start = time.perf_counter()
            result = renderer.render_eyes(left_image, right_image)
            _make_viewer_gl_context_current(viewer)
            timing['bridge_render'] = time.perf_counter() - bridge_start
            if not result.rendered:
                raise RuntimeError(f"Panda OpenGL bridge did not render both eyes: {result.bridge_mode}")
            viewer._panda_render_last_bridge_mode = str(result.bridge_mode)
            viewer._panda_render_last_target_size = ((left[5], left[6]), (right[5], right[6]))
            viewer._panda_render_last_image_indices = (left[2], right[2])
            viewer._panda_render_success_count = int(getattr(viewer, '_panda_render_success_count', 0) or 0) + 1
            viewer._panda_render_error = ""

            eye_layer_views = []
            release_start = time.perf_counter()
            for eye_index, swapchain, _img_index, raw_fbo, _mgl_fbo, sc_w, sc_h, view in acquired:
                if viewer._preview_active and eye_index == 0 and not updated_quad_eyes:
                    self._mirror_preview(raw_fbo, sc_w, sc_h)
                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released.add(eye_index)
                eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
            timing['release'] += time.perf_counter() - release_start
            timing['total'] = time.perf_counter() - total_start
            _record_panda_bridge_timing(viewer, timing)
            _log_panda_bridge_diagnostics(viewer)
            viewer._breakdown_inc('openxr_projection_panda_present')
            viewer._record_projection_screen_presented()
            return eye_layer_views
        except Exception as exc:
            _make_viewer_gl_context_current(viewer)
            release_start = time.perf_counter()
            for eye_index, swapchain, *_rest in acquired:
                if eye_index in released:
                    continue
                try:
                    xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                except Exception:
                    pass
            timing['release'] += time.perf_counter() - release_start
            timing['total'] = time.perf_counter() - total_start
            _record_panda_bridge_timing(viewer, timing)
            viewer._breakdown_inc('openxr_projection_panda_failed')
            viewer._panda_render_failure_count = int(getattr(viewer, '_panda_render_failure_count', 0) or 0) + 1
            viewer._panda_render_error = f"{type(exc).__name__}: {exc}"
            viewer._panda_opengl_bridge_failed_this_frame = True
            viewer._panda_opengl_bridge_disabled_until = time.perf_counter() + 2.0
            if not getattr(viewer, '_panda_render_error_logged', False):
                print(f"[OpenXRViewer] Panda3D OpenGL projection bridge failed; temporarily using native projection on later frames: {type(exc).__name__}: {exc}", flush=True)
                viewer._panda_render_error_logged = True
            _log_panda_render_path_status(viewer, status='bridge-failed', reason=viewer._panda_render_error)
            return []
    def render_panda_bridge(self, views, default_fov):
        viewer = self.viewer
        renderer = getattr(viewer, '_panda_scene_renderer', None)
        if renderer is None:
            return []
        acquired = []
        released = set()
        total_start = time.perf_counter()
        timing = {
            'acquire_wait': 0.0,
            'target_rebuild': 0.0,
            'bridge_render': 0.0,
            'release': 0.0,
            'total': 0.0,
        }
        try:
            for eye_index in range(2):
                acquire_start = time.perf_counter()
                swapchain = viewer._xr_swapchains[eye_index]
                img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
                viewer._wait_swapchain_image(swapchain)
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                acquired.append((eye_index, swapchain, img_index, sc_image, sc_w, sc_h, view))
                timing['acquire_wait'] += time.perf_counter() - acquire_start

            left = acquired[0]
            right = acquired[1]
            fmt = getattr(viewer, '_d3d11_swapchain_fmt', 'rgba8')
            generation = int(getattr(viewer, '_panda_swapchain_session_generation', 0) or 0)
            rebuild_start = time.perf_counter()
            renderer.rebuild_targets(
                StereoTargetSpec(left[4], left[5], fmt),
                StereoTargetSpec(right[4], right[5], fmt),
            )
            timing['target_rebuild'] = time.perf_counter() - rebuild_start
            left_image = SwapchainImageRef(left[0], left[2], left[3].texture, left[4], left[5], fmt, generation)
            right_image = SwapchainImageRef(right[0], right[2], right[3].texture, right[4], right[5], fmt, generation)
            bridge_start = time.perf_counter()
            result = renderer.render_eyes(left_image, right_image)
            timing['bridge_render'] = time.perf_counter() - bridge_start
            if not result.rendered:
                raise RuntimeError(f"Panda bridge did not render both eyes: {result.bridge_mode}")
            viewer._panda_render_last_bridge_mode = str(result.bridge_mode)
            viewer._panda_render_last_target_size = ((left[4], left[5]), (right[4], right[5]))
            viewer._panda_render_last_image_indices = (left[2], right[2])
            viewer._panda_render_success_count = int(getattr(viewer, '_panda_render_success_count', 0) or 0) + 1
            viewer._panda_render_error = ""

            eye_layer_views = []
            release_start = time.perf_counter()
            for eye_index, swapchain, _img_index, _sc_image, sc_w, sc_h, view in acquired:
                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released.add(eye_index)
                eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
            timing['release'] += time.perf_counter() - release_start
            timing['total'] = time.perf_counter() - total_start
            _record_panda_bridge_timing(viewer, timing)
            _log_panda_bridge_diagnostics(viewer)
            viewer._breakdown_inc('openxr_projection_panda_present')
            viewer._record_projection_screen_presented()
            return eye_layer_views
        except Exception as exc:
            release_start = time.perf_counter()
            for eye_index, swapchain, *_rest in acquired:
                if eye_index in released:
                    continue
                try:
                    xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                except Exception:
                    pass
            timing['release'] += time.perf_counter() - release_start
            timing['total'] = time.perf_counter() - total_start
            _record_panda_bridge_timing(viewer, timing)
            viewer._breakdown_inc('openxr_projection_panda_failed')
            viewer._panda_render_failure_count = int(getattr(viewer, '_panda_render_failure_count', 0) or 0) + 1
            viewer._panda_render_error = f"{type(exc).__name__}: {exc}"
            if not getattr(viewer, '_panda_render_error_logged', False):
                print(f"[OpenXRViewer] Panda3D projection bridge failed, falling back to native: {type(exc).__name__}: {exc}")
                viewer._panda_render_error_logged = True
            return []

    def render_d3d11_native(self, views, default_fov, default_proj_d3d):
        viewer = self.viewer
        renderer = viewer._d3d11_native_renderer
        near, far = self._projection_clip_planes()
        if renderer is None or not getattr(renderer, "has_frame", False):
            viewer._breakdown_inc('openxr_projection_d3d11_no_frame')
            return []
        update_panorama_background = getattr(renderer, "update_panorama_background", None)
        if callable(update_panorama_background):
            update_panorama_background(getattr(viewer, "_panorama_background_path", None))

        model = viewer._build_model_mat4()
        eye_layer_views = []
        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4_d3d(view.fov, near=near, far=far) if view else default_proj_d3d
                mvp = proj_mat @ view_mat @ model

                if viewer._runtime_direct_source:
                    renderer.render_runtime_eye(
                        sc_image.texture,
                        sc_w,
                        sc_h,
                        eye_index,
                        mvp,
                        view_mat=view_mat,
                        proj_mat=proj_mat,
                        overlay_viewer=viewer,
                    )
                else:
                    render_width = int(
                        getattr(viewer, '_runtime_rgb_depth_render_width', 0) or 0
                    )
                    if render_width <= 0:
                        render_width = int((viewer._texture_size or (0, 0))[0] or 0)
                    max_disparity = max(
                        0.0,
                        float(getattr(viewer, '_runtime_rgb_depth_max_disparity_px', 0.0) or 0.0),
                    )
                    disparity_uv = max_disparity / float(render_width) if render_width > 0 else 0.0
                    eye_sign = -1.0 if eye_index == 0 else 1.0
                    renderer.render_eye(
                        sc_image.texture,
                        sc_w,
                        sc_h,
                        eye_index,
                        eye_sign * disparity_uv / 2.0,
                        max(0.0, float(getattr(viewer, '_runtime_rgb_depth_depth_strength', viewer.depth_strength) or 0.0)),
                        float(viewer.convergence),
                        mvp,
                        roll=getattr(viewer, 'screen_roll', 0.0),
                        view_mat=view_mat,
                        proj_mat=proj_mat,
                        overlay_viewer=viewer,
                    )

                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
                eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                viewer._breakdown_inc('openxr_projection_render_failed')
                print(f"[OpenXRViewer] D3D11 projection render failed: {type(exc).__name__}: {exc}")
                return []
        viewer._breakdown_inc('openxr_projection_screen_present')
        viewer._record_projection_screen_presented()
        return eye_layer_views

    def _render_phase0_swapchain_probe(self, mgl_fbo, eye_index, sc_w, sc_h):
        viewer = self.viewer
        if not getattr(viewer, '_panda3d_phase0_swapchain_probe_logged', False):
            print("[OpenXRViewer] Panda3D Phase-0 probe rendering into acquired D3D11 swapchain")
            viewer._panda3d_phase0_swapchain_probe_logged = True

        ctx = viewer.ctx
        previous_viewport = ctx.viewport
        mgl_fbo.use()
        ctx.viewport = (0, 0, int(sc_w), int(sc_h))
        try:
            clear = (0.08, 0.04, 0.14, 1.0) if eye_index == 0 else (0.04, 0.08, 0.14, 1.0)
            ctx.clear(*clear)
            if viewer._panda3d_phase0_probe_prog is None:
                viewer._panda3d_phase0_probe_prog = ctx.program(
                    vertex_shader="""
                    #version 330
                    in vec2 in_pos;
                    in vec3 in_color;
                    out vec3 v_color;
                    void main() {
                        gl_Position = vec4(in_pos, 0.0, 1.0);
                        v_color = in_color;
                    }
                    """,
                    fragment_shader="""
                    #version 330
                    in vec3 v_color;
                    out vec4 fragColor;
                    void main() {
                        fragColor = vec4(v_color, 1.0);
                    }
                    """,
                )
                vertices = np.array(
                    [
                        -0.72, -0.62, 1.0, 0.20, 0.12,
                         0.72, -0.62, 0.10, 0.85, 1.0,
                         0.00,  0.64, 1.0, 0.95, 0.10,
                    ],
                    dtype='f4',
                )
                viewer._panda3d_phase0_probe_vbo = ctx.buffer(vertices.tobytes())
                viewer._panda3d_phase0_probe_vao = ctx.vertex_array(
                    viewer._panda3d_phase0_probe_prog,
                    [(viewer._panda3d_phase0_probe_vbo, '2f 3f', 'in_pos', 'in_color')],
                )
            viewer._panda3d_phase0_probe_vao.render()
        finally:
            ctx.viewport = previous_viewport

    def render_nv_dx_interop(self, views, default_fov, default_proj, *, phase0_probe=False):
        viewer = self.viewer
        near, far = self._projection_clip_planes()
        eye_layer_views = []
        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4(view.fov, near=near, far=far) if view else default_proj

                mgl_fbo, _raw_fbo = viewer._get_or_create_nv_interop_fbo(
                    eye_index, img_index, sc_image.texture, sc_w, sc_h,
                )
                _, _, dx_obj = viewer._nv_dx_objects[(eye_index, img_index)]
                _d3d_interop._wglDXLockObjectsNV(viewer._nv_dx_device, 1, ctypes.byref(dx_obj))
                try:
                    if phase0_probe:
                        self._render_phase0_swapchain_probe(mgl_fbo, eye_index, sc_w, sc_h)
                    else:
                        viewer._render_eye(eye_index, mgl_fbo, view_mat, proj_mat, flip_y=True)
                finally:
                    _d3d_interop._wglDXUnlockObjectsNV(viewer._nv_dx_device, 1, ctypes.byref(dx_obj))

                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
                eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                viewer._disable_nv_interop_after_failure(exc)
                return []
        viewer._breakdown_inc('openxr_projection_screen_present')
        viewer._record_projection_screen_presented()
        return eye_layer_views

    def render_opengl(self, views, default_fov, default_proj, *, updated_quad_eyes=()):
        viewer = self.viewer
        near, far = self._projection_clip_planes()
        eye_layer_views = []
        for eye_index in range(2):
            swapchain = viewer._xr_swapchains[eye_index]
            img_index = xr.acquire_swapchain_image(swapchain, viewer._xr_sc_acquire_info)
            viewer._wait_swapchain_image(swapchain)
            released = False
            try:
                sc_image = viewer._swapchain_images[eye_index][img_index]
                sc_w, sc_h = viewer._swapchain_sizes[eye_index]
                view = views[eye_index] if views and views[eye_index] else None
                view_mat = _pose_to_view_mat4(view.pose) if view else np.eye(4, dtype=np.float32)
                proj_mat = _fov_to_proj_mat4(view.fov, near=near, far=far) if view else default_proj

                raw_fbo, mgl_fbo = viewer._get_or_create_fbo(
                    eye_index, img_index, sc_image.image, sc_w, sc_h
                )
                viewer._render_eye(eye_index, mgl_fbo, view_mat, proj_mat)

                if viewer._preview_active and eye_index == 0 and not updated_quad_eyes:
                    self._mirror_preview(raw_fbo, sc_w, sc_h)

                xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                released = True
            except Exception as exc:
                if not released:
                    try:
                        xr.release_swapchain_image(swapchain, viewer._xr_sc_release_info)
                    except Exception:
                        pass
                viewer._breakdown_inc('openxr_projection_render_failed')
                print(f"[OpenXRViewer] OpenGL projection render failed: {type(exc).__name__}: {exc}")
                return []

            eye_layer_views.append(self._projection_view(swapchain, sc_w, sc_h, view, default_fov))
        viewer._breakdown_inc('openxr_projection_screen_present')
        viewer._record_projection_screen_presented()
        return eye_layer_views

    def _projection_view(self, swapchain, sc_w, sc_h, view, default_fov):
        return xr.CompositionLayerProjectionView(
            pose=view.pose if view else xr.Posef(),
            fov=view.fov if view else default_fov,
            sub_image=xr.SwapchainSubImage(
                swapchain=swapchain,
                image_rect=xr.Rect2Di(
                    offset=xr.Offset2Di(x=0, y=0),
                    extent=xr.Extent2Di(width=sc_w, height=sc_h),
                ),
            ),
        )

    def _mirror_preview(self, raw_fbo, sc_w, sc_h):
        viewer = self.viewer
        pw, ph = glfw.get_window_size(viewer.window)
        if pw <= 0 or ph <= 0:
            return
        glBindFramebuffer(GL_READ_FRAMEBUFFER, raw_fbo)
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)
        glBlitFramebuffer(0, 0, sc_w, sc_h, 0, 0, pw, ph, GL_COLOR_BUFFER_BIT, GL_LINEAR)
        glfw.swap_buffers(viewer.window)
