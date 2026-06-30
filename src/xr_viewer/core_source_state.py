# Desktop2Stereo OpenXR viewer: source-frame and idle/render state helpers.

import queue as _queue
import time

from OpenGL.GL import glDeleteBuffers


class CoreSourceStateMixin:
    """Source frame polling plus idle/render state transitions."""

    def _set_source_active(self, active: bool):
        if self._source_active_event is None:
            return
        try:
            if active:
                self._source_active_event.set()
            else:
                self._source_active_event.clear()
        except Exception:
            pass

    def _set_idle_active(self, active: bool):
        if self._idle_active_event is None:
            return
        try:
            if active:
                self._idle_active_event.set()
            else:
                self._idle_active_event.clear()
        except Exception:
            pass

    def _release_idle_gpu_resources(self):
        self._release_runtime_eye_pbos()
        if self._pbo_color is not None and self._cuda_gl:
            try:
                self._cuda_gl.unregister_resource(self._cuda_res_color)
                self._cuda_gl.unregister_resource(self._cuda_res_depth)
                glDeleteBuffers(2, [self._pbo_color, self._pbo_depth])
            except Exception:
                pass
        self._pbo_color = None
        self._pbo_depth = None
        self._cuda_res_color = None
        self._cuda_res_depth = None
        self._pbo_texture_size = None
        if self._cuda_gl is not False:
            self._cuda_gl = None

        for tex_name in ("color_tex", "depth_tex"):
            tex = getattr(self, tex_name, None)
            if tex is not None:
                try:
                    tex.release()
                except Exception:
                    pass
                setattr(self, tex_name, None)
        self._texture_size = None

    def _arm_headset_wait_inference_timeout(self, now=None):
        now = time.perf_counter() if now is None else now
        timeout = self._headset_wait_inference_timeout
        if timeout <= 0.0:
            self._headset_wait_inference_deadline = now
            self._headset_wait_inference_paused = True
            self._set_source_active(False)
            return
        self._headset_wait_inference_deadline = now + timeout
        self._headset_wait_inference_paused = False
        self._set_source_active(True)

    def _refresh_headset_wait_inference_timeout(self, now=None):
        if self._headset_wait_inference_paused:
            return
        if not self._waiting_for_headset:
            return
        if self._headset_wait_inference_deadline <= 0.0:
            return
        now = time.perf_counter() if now is None else now
        if now < self._headset_wait_inference_deadline:
            return
        self._headset_wait_inference_paused = True
        self._set_source_active(False)
        self._enter_hard_idle_wait()
        print(
            f"[OpenXRViewer] No headset detected for {self._headset_wait_inference_timeout:.0f}s; "
            "stopping source inference (source_active_event cleared)"
        )
        print(
            f"[OpenXRViewer] Waiting for VR headset connect... "
            f"(retry in {self._openxr_no_headset_retry_interval:.1f}s)"
        )

    def _resume_source_inference(self):
        self._headset_wait_inference_deadline = 0.0
        self._headset_wait_inference_paused = False
        self._leave_hard_idle_wait()
        self._set_source_active(True)
        print("[OpenXRViewer] Source inference enabled")

    def _enter_hard_idle_wait(self):
        if self._hard_idle_active:
            return
        self._hard_idle_active = True
        self._set_render_active(False)
        self._set_idle_active(True)
        self._release_idle_gpu_resources()
        if self._preview_active and self._show_preview_window:
            try:
                glfw.hide_window(self.window)
                self._preview_active = False
            except Exception:
                pass
        print("[OpenXRViewer] Enter hard idle: source/render paused")

    def _leave_hard_idle_wait(self):
        if not self._hard_idle_active:
            return
        self._hard_idle_active = False
        self._set_idle_active(False)
        if (not self._preview_active) and self._show_preview_window:
            try:
                glfw.show_window(self.window)
                self._preview_active = True
            except Exception:
                pass
        print("[OpenXRViewer] Exit hard idle")

    def _try_restore_openxr(self, now=None):
        if not self._preview_only_mode:
            return False
        now = time.perf_counter() if now is None else now
        if now < self._openxr_retry_cooldown_until:
            return False
        if (now - self._last_openxr_retry_time) < self._openxr_retry_interval:
            return False
        self._last_openxr_retry_time = now
        try:
            self._debug_openxr_trace("_try_restore_openxr attempt", now)
            if self._xr_backend == 'd3d11':
                # Re-probe from a clean instance so restore attempts still try
                # OpenGL first before falling back to D3D11 again.
                self._cleanup_partial_openxr(destroy_instance=True)
                self._ensure_preview_swapchain_size()
            self._debug_openxr_trace(
                f"_try_restore_openxr attempt probes={self._openxr_restore_opengl_probe_attempts}x{self._openxr_restore_opengl_probe_interval:.1f}s",
                now,
            )
            self._init_openxr(
                quiet=True,
                opengl_probe_attempts=self._openxr_restore_opengl_probe_attempts,
                opengl_probe_interval=self._openxr_restore_opengl_probe_interval,
            )
            self._preview_only_mode = False
            self._debug_openxr_trace("_try_restore_openxr success", now)
            print("[OpenXRViewer] OpenXR session created, waiting for render confirmation")
            return True
        except Exception as exc:
            self._cleanup_partial_openxr(destroy_instance=False)
            self._ensure_preview_swapchain_size()
            if self._is_no_headset_error(exc):
                self._defer_openxr_retry(self._openxr_no_headset_retry_interval)
            self._debug_openxr_trace(f"_try_restore_openxr failed: {type(exc).__name__}", now)
            return False

    def _mark_source_frame_received(self, now=None):
        self._last_source_frame_time = time.perf_counter() if now is None else now
        self._source_resume_grace_until = 0.0
        self._source_stalled = False
        self._source_stall_count = 0
        if self._session_running and not self._session_ready_pending:
            self._set_render_active(True)

    def _has_fresh_source_frame(self, now=None):
        now = time.perf_counter() if now is None else now
        if now <= self._source_resume_grace_until:
            return True
        if self._last_source_frame_time <= 0.0:
            return False
        return (now - self._last_source_frame_time) <= self._source_frame_timeout

    def _pause_xr_output_for_source_stall(self):
        now = time.perf_counter()
        last_notice = getattr(self, "_last_source_stall_notice_time", 0.0)
        self._source_stall_count += 1
        if not self._source_stalled or (now - last_notice) >= 5.0:
            age = -1.0
            if self._last_source_frame_time > 0.0:
                age = now - self._last_source_frame_time
            try:
                queued = self.depth_q.qsize()
            except Exception:
                queued = -1
            if getattr(self, "_openxr_debug", False):
                print(
                    "[OpenXRViewer][debug] Source stale: "
                    f"age={age:.2f}s timeout={self._source_frame_timeout:.2f}s "
                    f"q={queued} count={self._source_stall_count}; keeping previous frame"
                )
            self._last_source_stall_notice_time = now
        self._source_stalled = True

    def _set_render_active(self, active: bool):
        if self._render_active_event is None:
            return
        try:
            if active:
                self._render_active_event.set()
            else:
                self._render_active_event.clear()
        except Exception:
            pass

    def _breakdown_inc(self, name, amount=1):
        callback = getattr(self, "_fps_breakdown_inc", None)
        if not callable(callback):
            return
        try:
            callback(name, amount)
        except Exception:
            pass

    def _breakdown_add_time(self, name, seconds):
        callback = getattr(self, "_fps_breakdown_add_time", None)
        if not callable(callback):
            return
        try:
            callback(name, seconds)
        except Exception:
            pass

    def _poll_source_frame(self, upload=False):
        poll_start = time.perf_counter()
        latest = None
        dequeued = 0
        try:
            while True:
                latest = self.depth_q.get_nowait()
                dequeued += 1
        except _queue.Empty:
            pass

        if dequeued:
            self._breakdown_inc("viewer_get", dequeued)
            if dequeued > 1:
                self._breakdown_inc("viewer_drop", dequeued - 1)

        if latest is not None:
            self._pending_source_frame = latest
            self._mark_source_frame_received()

        if not upload:
            self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return latest is not None

        if self._pending_source_frame is None:
            self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
            return False

        source_frame, frame_ts = self._normalize_source_frame(self._pending_source_frame)
        self._pending_source_frame = None

        upload_start = time.perf_counter()
        if self._is_runtime_result(source_frame):
            self._update_runtime_frame(source_frame)
        else:
            rgb, depth = source_frame
            self._update_frame(rgb, depth)
        self._breakdown_add_time("openxr_upload", time.perf_counter() - upload_start)
        if frame_ts is not None:
            self.total_latency = (time.perf_counter() - frame_ts) * 1000.0
        sbs_now = time.perf_counter()
        self._sbs_ts_ring.append(sbs_now)
        m = len(self._sbs_ts_ring)
        if m >= 2:
            sbs_span = sbs_now - self._sbs_ts_ring[0]
            if sbs_span > 0:
                self.sbs_fps = (m - 1) / sbs_span
        self._breakdown_add_time("openxr_poll", time.perf_counter() - poll_start)
        return True

    def _normalize_source_frame(self, item):
        if self._is_runtime_result(item):
            return item, None
        if isinstance(item, tuple):
            if len(item) == 2 and self._is_runtime_result(item[0]):
                return item[0], item[1]
            if len(item) == 3:
                rgb, depth, frame_ts = item
                return (rgb, depth), frame_ts
        raise RuntimeError(f"Unsupported OpenXR source frame: {type(item).__name__}")

    def _is_runtime_result(self, item):
        return (
            hasattr(item, "left_eye")
            and hasattr(item, "right_eye")
            and hasattr(item, "depth")
        )

    def _publish_runtime_config(self, *, include_stereo=False):
        callback = self._runtime_config_callback
        if not callable(callback):
            return
        try:
            payload = {"screen_roll": self.screen_roll}
            if include_stereo:
                depth_strength = float(self.depth_strength)
                if self._quad_layer_can_replace_projection_screen():
                    depth_strength *= float(getattr(self, '_xr_quad_layer_stereo_boost', 1.0))
                payload.update(
                    depth_strength=depth_strength,
                    convergence=self.convergence,
                )
            callback(**payload)
        except Exception:
            pass

    def _has_renderable_source_frame(self):
        if self._runtime_direct_source:
            return all(self._runtime_eye_textures) and self._runtime_depth_texture is not None
        return self.color_tex is not None and self.depth_tex is not None

    def _should_show_source_border(self, now=None):
        if getattr(self, "_hard_idle_active", False):
            return False
        source_active_event = getattr(self, "_source_active_event", None)
        if source_active_event is not None:
            try:
                if not source_active_event.is_set():
                    return False
            except Exception:
                return False
        if not self._has_renderable_source_frame():
            return False
        return self._has_fresh_source_frame(now)
