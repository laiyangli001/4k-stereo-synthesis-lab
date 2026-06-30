import ctypes
import time

import numpy as np

try:
    import xr
except ImportError:
    xr = None

from .constants import _VIVE_TB_Y


class CoreOpenXRInputMixin:
    def _poll_xr_events(self):
        """Drain the OpenXR event queue and handle session state transitions."""
        from utils import shutdown_event
        while True:
            try:
                event_buf = xr.poll_event(self._xr_instance)
            except xr.EventUnavailable:
                break

            event_type = event_buf.type

            if event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED:
                event = ctypes.cast(
                    ctypes.byref(event_buf),
                    ctypes.POINTER(xr.EventDataSessionStateChanged),
                ).contents
                state = xr.SessionState(event.state)
                if state == xr.SessionState.READY:
                    now = time.perf_counter()
                    self._debug_openxr_trace("event READY", now)
                    if (
                        now < self._openxr_retry_cooldown_until
                        or self._xr_session is None
                    ):
                        self._debug_openxr_trace("event READY ignored", now)
                        continue
                    xr.begin_session(
                        self._xr_session,
                        xr.SessionBeginInfo(
                            primary_view_configuration_type=
                                xr.ViewConfigurationType.PRIMARY_STEREO
                        ),
                    )
                    self._source_resume_grace_until = (
                        time.perf_counter() + self._source_resume_grace
                    )
                    self._session_running = True
                    self._session_ready_pending = True
                    self._openxr_ready_since = now
                    self._session_idle_since = 0.0
                    self._session_idle_notice_emitted = False
                    self._debug_openxr_trace("event READY begin_session", now)
                    print("[OpenXRViewer] Session READY - awaiting render confirmation")

                elif state in (
                    xr.SessionState.STOPPING,
                    xr.SessionState.LOSS_PENDING,
                    xr.SessionState.EXITING,
                ):
                    if state == xr.SessionState.STOPPING:
                        stop_now = time.perf_counter()
                        retry_delay = self._compute_standby_retry_delay(stop_now)
                        self._debug_openxr_trace(
                            f"event STOPPING retry={retry_delay:.1f}s count={self._openxr_standby_retry_count}",
                            stop_now,
                        )
                        try:
                            xr.end_session(self._xr_session)
                        except Exception:
                            pass
                        self._cleanup_partial_openxr(destroy_instance=True)
                        self._openxr_ready_since = 0.0
                        self._session_idle_since = 0.0
                        self._session_idle_notice_emitted = False
                        self._defer_openxr_retry(retry_delay)
                        self._enter_preview_only_wait()
                        self._debug_openxr_trace("event STOPPING handled", time.perf_counter())
                        print("[OpenXRViewer] Session STOPPING; waiting for headset wake")
                        continue
                    try:
                        xr.end_session(self._xr_session)
                    except Exception:
                        pass
                    print(f"[OpenXRViewer] Session state -> {state.name}; rendering paused")
                    self._cleanup_partial_openxr(destroy_instance=True)
                    self._openxr_ready_since = 0.0
                    self._session_idle_since = 0.0
                    self._session_idle_notice_emitted = False
                    self._defer_openxr_retry(self._openxr_standby_retry_interval)
                    self._enter_preview_only_wait()

            elif event_type == xr.StructureType.EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING:
                view = getattr(self, '_view_pose_profile', {}) or {}
                if isinstance(view, dict) and view.get('auto_center_on_screen', False):
                    if self._xr_session is not None and self._xr_ref_space_type is not None:
                        try:
                            old_space = self._xr_space
                            self._xr_space = xr.create_reference_space(
                                self._xr_session,
                                xr.ReferenceSpaceCreateInfo(
                                    reference_space_type=self._xr_ref_space_type,
                                    pose_in_reference_space=xr.Posef(),
                                ),
                            )
                            self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)
                            if old_space is not None:
                                xr.destroy_space(old_space)
                        except Exception:
                            pass
                    self._xr_profile_space_applied = False
                else:
                    self._reset_screen_to_default(show_border=False)

            elif event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                print("[OpenXRViewer] Instance loss pending - waiting for runtime recovery")
                self._cleanup_partial_openxr(destroy_instance=True)
                self._openxr_ready_since = 0.0
                self._session_idle_since = 0.0
                self._session_idle_notice_emitted = False
                self._defer_openxr_retry(self._openxr_standby_retry_interval)
                self._enter_preview_only_wait()
                break

    def _read_bool_action_raw(self, action, hand_path_str="/user/hand/left"):
        """Return the raw OpenXR boolean action state without trackpad emulation."""
        if action is None:
            return False
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return bool(state.is_active and state.current_state)
        except Exception:
            return False

    def _read_bool_action(self, action, hand_path_str="/user/hand/left"):
        """Return True if the boolean action is pressed, including trackpad emulation."""
        pressed = self._read_bool_action_raw(action, hand_path_str)
        if action is self._act_y_btn and hand_path_str == "/user/hand/left":
            pressed = pressed or self._emu_y
        elif action is self._act_x_btn and hand_path_str == "/user/hand/left":
            pressed = pressed or self._emu_x
        elif action is self._act_b_btn and hand_path_str == "/user/hand/right":
            pressed = pressed or self._emu_b
        elif action is self._act_a_btn and hand_path_str == "/user/hand/right":
            pressed = pressed or self._emu_a
        elif action is self._act_left_stick_click and hand_path_str == "/user/hand/left":
            pressed = False if (self._emu_x or self._emu_y) else (pressed or self._emu_lsc)
        elif action is self._act_right_stick_click and hand_path_str == "/user/hand/right":
            pressed = False if (self._emu_a or self._emu_b) else (pressed or self._emu_rsc)
        return pressed

    def _read_bool_edge(self, action, hand_path_str, prev_state):
        """Return True on the rising edge of a boolean action.

        Tries to use the OpenXR runtime's `changed` flag via the raw ctypes struct
        (pyopenxr may not expose it as a Python attribute).  Falls back to manual
        frame-to-frame comparison if the ctypes path fails.
        """
        if action is None:
            return False
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_boolean(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            pressed = self._read_bool_action(action, hand_path_str)

            # pyopenxr wraps XrActionStateBoolean. Try the Python attribute first,
            # then fall back to reading the underlying ctypes struct.
            changed = False
            if hasattr(state, 'changed'):
                changed = bool(state.changed)
            else:
                # The struct is [isActive:i4, currentState:i4, changed:i4, ...]
                # changed is at byte offset 8 (after two 4-byte fields).
                try:
                    ptr = ctypes.cast(ctypes.byref(state), ctypes.POINTER(ctypes.c_int32))
                    changed = bool(ptr[2])  # offset 2 x 4 bytes
                except Exception:
                    pass

            if changed:
                return pressed   # runtime-confirmed edge
            # Fallback: manual rising-edge detection
            return pressed and not prev_state
        except Exception:
            return False

    def _update_trackpad_button_emu(self):
        """Compute per-frame Vive/WMR trackpad button emulation flags."""
        for hand, stick_act, click_act, attr_top, attr_bot, attr_ctr in [
            ("/user/hand/left", self._act_left_stick, self._act_left_stick_click,
             '_emu_y', '_emu_x', '_emu_lsc'),
            ("/user/hand/right", self._act_right_stick, self._act_right_stick_click,
             '_emu_b', '_emu_a', '_emu_rsc'),
        ]:
            clicked = self._read_bool_action_raw(click_act, hand)
            if not clicked:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, False)
                continue
            try:
                path = self._path_left if hand == "/user/hand/left" else self._path_right
                state = xr.get_action_state_vector2f(
                    self._xr_session,
                    xr.ActionStateGetInfo(action=stick_act, subaction_path=path),
                )
                py = float(state.current_state.y) if state.is_active else 0.0
            except Exception:
                py = 0.0
            if py > _VIVE_TB_Y:
                setattr(self, attr_top, True)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, False)
            elif py < -_VIVE_TB_Y:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, True)
                setattr(self, attr_ctr, False)
            else:
                setattr(self, attr_top, False)
                setattr(self, attr_bot, False)
                setattr(self, attr_ctr, True)

    def _read_float_action(self, action, hand_path_str="/user/hand/left"):
        """Return the float value [0,1] of a trigger/squeeze action."""
        if action is None:
            return 0.0
        try:
            path = (self._path_left
                    if hand_path_str == "/user/hand/left" else self._path_right)
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            state = xr.get_action_state_float(
                self._xr_session,
                xr.ActionStateGetInfo(action=action, subaction_path=path),
            )
            return float(state.current_state) if state.is_active else 0.0
        except Exception:
            return 0.0

    def _update_controller_press_animation_state(self, dt=0.0):
        """Cache per-hand button pressure used by controller render feedback."""
        def _hand_pressure(hand_path, trigger_action, grip_action, face_actions):
            pressure = max(0.0, min(1.0, self._read_float_action(trigger_action, hand_path)))
            if self._read_bool_action(grip_action, hand_path):
                pressure = max(pressure, 1.0)
            for action in face_actions:
                if self._read_bool_action(action, hand_path):
                    pressure = max(pressure, 1.0)
                    break
            return pressure

        left_target = _hand_pressure(
            "/user/hand/left",
            self._act_left_trigger,
            self._act_left_grip,
            (self._act_x_btn, self._act_y_btn, self._act_left_stick_click, self._act_menu_btn),
        )
        right_target = _hand_pressure(
            "/user/hand/right",
            self._act_right_trigger,
            self._act_right_grip,
            (self._act_a_btn, self._act_b_btn, self._act_right_stick_click),
        )
        dt = max(0.0, min(0.050, float(dt or 0.0)))
        alpha = 1.0 if dt <= 0.0 else min(1.0, dt * 24.0)
        left_current = float(getattr(self, "_ctrl_press_l", 0.0) or 0.0)
        right_current = float(getattr(self, "_ctrl_press_r", 0.0) or 0.0)
        self._ctrl_press_l = left_current + (left_target - left_current) * alpha
        self._ctrl_press_r = right_current + (right_target - right_current) * alpha

    def _pulse_haptic(
        self,
        hand_path_str="/user/hand/right",
        *,
        amplitude=0.18,
        duration_s=0.018,
        min_interval_s=0.045,
    ):
        """Send a short controller haptic pulse; failures are non-fatal."""
        action = getattr(self, "_act_haptic", None)
        if action is None or self._xr_session is None:
            return False
        now = time.perf_counter()
        last_attr = "_haptic_last_l" if hand_path_str == "/user/hand/left" else "_haptic_last_r"
        if now - float(getattr(self, last_attr, 0.0) or 0.0) < float(min_interval_s):
            return False
        try:
            path = self._path_left if hand_path_str == "/user/hand/left" else self._path_right
            if path is None:
                path = xr.string_to_path(self._xr_instance, hand_path_str)
            duration_ns = max(1, int(float(duration_s) * 1_000_000_000))
            vibration = xr.HapticVibration(
                duration=duration_ns,
                frequency=xr.FREQUENCY_UNSPECIFIED,
                amplitude=max(0.0, min(1.0, float(amplitude))),
            )
            xr.apply_haptic_feedback(
                self._xr_session,
                xr.HapticActionInfo(action=action, subaction_path=path),
                vibration,
            )
            setattr(self, last_attr, now)
            return True
        except Exception:
            return False
