from __future__ import annotations

from utils.queue_utils import clear_nonblocking, drain_latest, put_latest


class RuntimeCallbacks:
    def __init__(self, context):
        self.context = context
        self.capture_control = None
        self.capture_session = None

    def stereo_warmup_key(self, rgb_frame):
        return self.context.stereo_warmup_tracker.key_for_frame(rgb_frame)

    def warmup_stereo_once_for_frame(self, rgb_frame):
        self.context.stereo_warmup_tracker.warmup_once_for_frame(rgb_frame)

    def breakdown_inc(self, name, amount=1):
        self.context.fps_breakdown.inc(name, amount)

    def breakdown_add_time(self, name, seconds):
        self.context.fps_breakdown.add_time(name, seconds)

    def breakdown_add_runtime_timing(self, runtime_result):
        self.context.fps_breakdown.add_runtime_timing(runtime_result)

    def log_fps_breakdown(self, now=None):
        self.context.fps_breakdown.log(now)

    def source_stat_inc(self, name, amount=1, **values):
        self.context.source_health.inc(name, amount, **values)

    def source_stat_set(self, **values):
        self.context.source_health.set(**values)

    def log_source_health(self, now=None, force=False):
        self.context.source_health.log(now, force)

    def openxr_source_paused(self):
        return self.context.openxr_state.source_paused()

    def stop_active_capture_session(self):
        stopped = False
        try:
            if self.capture_control is not None:
                self.capture_control.stop()
                stopped = True
        except Exception:
            pass
        try:
            if not stopped and self.capture_session is not None and hasattr(self.capture_session, "stop"):
                self.capture_session.stop()
                stopped = True
        except Exception:
            pass
        return stopped

    def on_openxr_hard_idle_enter(self):
        self.queue_clear_nonblocking(self.context.raw_q)
        self.queue_clear_nonblocking(self.context.runtime_q)
        self.stop_active_capture_session()

    def openxr_hard_idle_active(self):
        return self.context.openxr_state.hard_idle_active(
            on_enter=self.on_openxr_hard_idle_enter
        )

    def queue_put_latest(self, q, item):
        put_latest(q, item)

    def queue_clear_nonblocking(self, q):
        clear_nonblocking(q)

    def queue_drain_latest(self, q, first_item):
        def on_drop():
            self.source_stat_inc("raw_dropped_stale")
            self.breakdown_inc("raw_dropped_stale")

        return drain_latest(q, first_item, on_drop=on_drop)

    def update_openxr_runtime_config(self, *, ipd=None, depth_ratio=None, convergence=None, screen_roll=None):
        self.context.openxr_state.update_runtime_config(
            ipd=ipd,
            depth_ratio=depth_ratio,
            convergence=convergence,
            screen_roll=screen_roll,
        )

    def current_openxr_render_config(self):
        return self.context.openxr_state.current_render_config(self.context.stereo_runtime)

    def apply_stereo_hot_reload_if_needed(self):
        self.context.stereo_hot_reloader.apply_if_needed(
            runtime=self.context.stereo_runtime,
            active_preset=self.context.stereo_active_preset,
            on_openxr_config_update=self.update_openxr_runtime_config,
            on_mode_log=self.log_stereo_runtime_mode_once,
        )

    def log_stereo_runtime_mode(self, reason, decision=None, samples=None, motion=None):
        self.context.stereo_runtime_logger.log_mode(
            reason,
            decision=decision,
            samples=samples,
            motion=motion,
        )

    def log_stereo_runtime_mode_once(self, reason="active"):
        self.context.stereo_runtime_logger.log_mode_once(reason)

    def log_fast_plus_fused_runtime_state(self, runtime_result):
        self.context.stereo_runtime_logger.log_fast_plus_fused_runtime_state(runtime_result)

    def capture_session_update(self, session, control):
        self.capture_session = session
        self.capture_control = control

    def put_raw_latest(self, item):
        was_full = self.context.raw_q.full()
        self.queue_put_latest(self.context.raw_q, item)
        return was_full

    def set_runtime_preprocess_backend(self, backend):
        if self.context.fps_breakdown_log:
            self.context.fps_breakdown.set_latest("rt_preprocess_backend", backend)
