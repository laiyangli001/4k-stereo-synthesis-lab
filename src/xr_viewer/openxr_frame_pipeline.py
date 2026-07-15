import time

import glfw

from .xr_math import _fov_to_proj_mat4, _fov_to_proj_mat4_d3d
from .effect_submitter import EffectSubmitter
from .frame_submitter import FrameSubmitter
from .openxr_frame_gate import OpenXRFrameGate
from .openxr_frame_input import OpenXRFrameInput
from .openxr_frame_renderer import OpenXRFrameRenderer
from .openxr_frame_timing import OpenXRFrameTiming

try:
    import xr
except ImportError:
    xr = None


class OpenXRFramePipeline:
    def __init__(self, viewer):
        self.viewer = viewer
        self.timing = OpenXRFrameTiming(viewer)
        self.input = OpenXRFrameInput(viewer)
        self.frame_submitter = FrameSubmitter(viewer)
        self.gate = OpenXRFrameGate(viewer, self.frame_submitter)
        self.renderer = OpenXRFrameRenderer(viewer)
        self.effect_submitter = EffectSubmitter(viewer)
        self.last_input_t = time.perf_counter()
        self.default_fov = xr.Fovf(
            angle_left=-0.785, angle_right=0.785,
            angle_up=0.785,   angle_down=-0.785,
        )
        self._projection_clip_key = None
        self.default_proj = None
        self.default_proj_d3d = None
        self._refresh_default_projection()

    def _projection_clip_planes(self):
        viewer = self.viewer
        near = max(0.01, float(getattr(viewer, '_xr_projection_near', 0.05) or 0.05))
        far = max(near + 1.0, float(getattr(viewer, '_xr_projection_far', 100.0) or 100.0))
        return near, far

    def _refresh_default_projection(self):
        near, far = self._projection_clip_planes()
        key = (near, far)
        if self._projection_clip_key == key:
            return
        self._projection_clip_key = key
        self.default_proj = _fov_to_proj_mat4(self.default_fov, near=near, far=far)
        self.default_proj_d3d = _fov_to_proj_mat4_d3d(self.default_fov, near=near, far=far)

    def seed_first_frame(self, *, first_rgb=None, first_depth=None, first_runtime_result=None, first_frame_ts=None):
        viewer = self.viewer
        first_source_frame = None
        if first_runtime_result is not None:
            effect_source_rgb = viewer._update_runtime_frame(first_runtime_result)
            viewer._queue_runtime_effect_submit(effect_source_rgb)
            first_source_frame = (first_runtime_result, first_frame_ts)
            if first_frame_ts is not None:
                viewer.total_latency = (time.perf_counter() - first_frame_ts) * 1000.0
        elif first_rgb is not None and first_depth is not None:
            viewer._update_frame(first_rgb, first_depth)
            first_source_frame = (first_rgb, first_depth, first_frame_ts)

        if first_source_frame is None:
            return

        bridge = viewer._screen_frame_bridge()
        bridge.latest_frame = first_source_frame
        bridge.frame_id += 1
        bridge.latest_frame_id = bridge.frame_id
        bridge.source_timestamp = first_frame_ts
        viewer._mark_source_frame_received()

    def begin_loop_frame(self):
        viewer = self.viewer
        now = time.perf_counter()
        dt = now - self.last_input_t
        self.last_input_t = now
        viewer._frame_now = now
        viewer._last_frame_dt = dt
        viewer._frame_count += 1
        viewer._publish_runtime_config()
        glfw.poll_events()
        viewer._poll_frosted_glow_hotkeys()
        return now, dt

    def handle_preview_only(self, now):
        viewer = self.viewer
        if not viewer._preview_only_mode:
            return False
        viewer._refresh_headset_wait_inference_timeout(now)
        viewer._ensure_env_model_initialized("Preview-only")
        if viewer._waiting_retry_notice_pending:
            print(
                f"[OpenXRViewer] Waiting for VR headset connect... "
                f"(retry in {viewer._openxr_no_headset_retry_interval:.1f}s)"
            )
            viewer._waiting_retry_notice_pending = False
        viewer._try_restore_openxr(now)
        time.sleep(viewer._headset_wait_idle_sleep if viewer._hard_idle_active else 0.1)
        return True

    def begin_active_session_frame(self):
        viewer = self.viewer
        viewer._poll_xr_events()
        if viewer._session_running:
            return True
        time.sleep(0.01)
        return False

    def render_frame(self, *, now, dt):
        viewer = self.viewer
        perf_log_enabled = bool(getattr(viewer, '_openxr_perf_log', False))
        breakdown_enabled = callable(getattr(viewer, '_fps_breakdown_add_time', None))
        trace_enabled = perf_log_enabled or breakdown_enabled
        t0 = time.perf_counter() if trace_enabled else 0.0
        last_mark = t0
        marks = []
        viewer._breakdown_inc('openxr_loop')

        def mark(label):
            nonlocal last_mark
            if not trace_enabled:
                return
            t_mark = time.perf_counter()
            elapsed_s = t_mark - last_mark
            if perf_log_enabled:
                marks.append((label, elapsed_s * 1000.0))
            if breakdown_enabled:
                viewer._breakdown_add_time(f'openxr_{label}', elapsed_s)
            last_mark = t_mark

        if viewer._session_ready_pending or not viewer._has_fresh_source_frame(now):
            viewer._poll_source_frame(upload=False)
            mark('poll_no_upload')

        frame_state, submit_start = self.timing.begin_frame(
            breakdown_enabled=breakdown_enabled
        )
        mark('wait_frame')
        mark('begin_frame')

        self.input.sync_actions()
        mark('sync_actions')
        controller_mark = self.input.update_controller_frame(
            display_time=frame_state.predicted_display_time,
            dt=dt,
        )
        mark('controller_pose')
        mark(controller_mark)

        composition_layers = []
        skip_render, session_idle_timeout = self.gate.handle_ready_or_stall(
            frame_state=frame_state,
            now=now,
            composition_layers=composition_layers,
            submit_start=submit_start,
        )
        if skip_render:
            mark('end_frame')
            self.flush_background_after_submit()
            return False

        screen_frame_uploaded = False
        if frame_state.should_render:
            self._refresh_default_projection()
            screen_frame_uploaded, view_pose_adjusted, rendered_projection = self.renderer.render_frame(
                composition_layers=composition_layers,
                display_time=frame_state.predicted_display_time,
                default_fov=self.default_fov,
                default_proj=self.default_proj,
                default_proj_d3d=self.default_proj_d3d,
            )
            mark('poll_upload')
            mark('locate_views')
            if view_pose_adjusted:
                mark('view_pose_adjust')
            mark('quad_update')
            mark('render_eyes' if rendered_projection else 'render_no_layers')
            mark('layers')

        self.frame_submitter.submit(
            composition_layers,
            display_time=frame_state.predicted_display_time,
            submit_start=submit_start,
        )
        mark('end_frame')
        self.flush_background_after_submit()
        self.effect_submitter.flush_after_submit(
            should_render=frame_state.should_render,
            screen_frame_uploaded=screen_frame_uploaded,
        )
        if perf_log_enabled:
            self._log_perf(frame_state, marks, t0)

        if session_idle_timeout:
            self.gate.enter_idle_if_needed(session_idle_timeout)
            return False
        self.record_presented_frame()
        return True

    def flush_background_after_submit(self):
        background_renderer = getattr(self.viewer, '_background_layer_renderer', None)
        if background_renderer is None:
            return
        try:
            background_renderer.flush_pending_upload_after_submit()
        except Exception as exc:
            print(f"[OpenXRViewer] Background upload flush failed: {type(exc).__name__}: {exc}")
            self.viewer._breakdown_inc('openxr_background_layer_upload_failed')

    def record_presented_frame(self):
        viewer = self.viewer
        t_now = time.perf_counter()
        viewer._frame_ts_ring.append(t_now)
        n = len(viewer._frame_ts_ring)
        if n < 2:
            return
        span = t_now - viewer._frame_ts_ring[0]
        if span > 0:
            viewer.actual_fps = (n - 1) / span

    def _log_perf(self, frame_state, marks, t0):
        viewer = self.viewer
        loop_total_ms = (time.perf_counter() - t0) * 1000.0
        loop_log_now = time.perf_counter()
        loop_last_log = float(getattr(viewer, '_xr_loop_perf_last_log', 0.0) or 0.0)
        if loop_total_ms < 25.0 and (loop_log_now - loop_last_log) < 2.0:
            return
        loop_log_count = int(getattr(viewer, '_xr_loop_perf_log_count', 0) or 0)
        if loop_log_count >= 5:
            return
        viewer._xr_loop_perf_last_log = loop_log_now
        viewer._xr_loop_perf_log_count = loop_log_count + 1
        loop_parts = ' '.join(f'{label}={ms:.1f}' for label, ms in marks if ms >= 0.05)
        print(
            '[OpenXRViewer] loop segments '
            f'total_ms={loop_total_ms:.1f} '
            f'fps={float(getattr(viewer, "actual_fps", 0.0)):.1f} '
            f'should_render={bool(getattr(frame_state, "should_render", False))} '
            f'runtime_direct={bool(getattr(viewer, "_runtime_direct_source", False))} '
            f'fresh={bool(viewer._has_fresh_source_frame(time.perf_counter()))} '
            f'renderable={bool(viewer._has_renderable_source_frame())} '
            f'{loop_parts}'
        )
