# Desktop2Stereo OpenXR viewer: environment view and screen layout helpers.

from .implementation import *


class EnvironmentLayoutMixin:
    """Profile-driven XR space, view pose, and screen layout behavior."""

    def _environment_screen_locked(self):
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return False
        return not self._profile_bool(
            screen,
            ('allow_controller_move', 'controller_movable', 'unlock_controller_move'),
            False,
        )


    def _view_pose_has_explicit_layout(self, view=None):
        if view is None:
            view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return False
        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        return (
            any(key in view for key in pos_keys)
            or all(key in view for key in ('x', 'y', 'z'))
            or any(key in view for key in rot_deg_keys + rot_rad_keys)
            or 'angle' in view
            or bool(view.get('auto_center_on_screen', False))
        )


    def _profile_view_position(self, view, default):
        for key in ('position', 'camera_position', 'viewer_position'):
            value = view.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    return [float(value[0]), float(value[1]), float(value[2])]
                except (TypeError, ValueError):
                    pass
        if all(key in view for key in ('x', 'y', 'z')):
            try:
                return [float(view['x']), float(view['y']), float(view['z'])]
            except (TypeError, ValueError):
                pass
        return list(default)


    def _profile_view_rotation_rad(self, view, deg_keys, rad_keys, default):
        if any(key in view for key in deg_keys + rad_keys):
            return self._profile_rotation_rad(view, deg_keys, rad_keys, default)
        if 'angle' in view:
            try:
                return [math.radians(float(view.get('angle', 0.0))), 0.0, 0.0]
            except (TypeError, ValueError):
                pass
        return list(default)


    def _head_model_mat4_from_views(self, views):
        if not views or len(views) < 2 or views[0] is None or views[1] is None:
            return None
        head_mat = xr_pose_to_model_mat4(views[0].pose)
        try:
            p0 = views[0].pose.position
            p1 = views[1].pose.position
            head_mat[:3, 3] = np.array([
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            ], dtype=np.float32)
        except Exception:
            return None
        return head_mat


    def _level_head_model_mat4(self, head_mat):
        """Keep head position and yaw, dropping pitch/roll so the room stays level."""
        if head_mat is None:
            return None
        pos = head_mat[:3, 3].copy()
        forward = -head_mat[:3, 2].astype(np.float32)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            yaw = 0.0
        else:
            forward = forward / norm
            yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        leveled = euler_to_mat4(yaw, 0.0, 0.0).astype(np.float32)
        leveled[:3, 3] = pos
        return leveled


    def _apply_profile_view_pose_to_xr_space(self, views):
        if self._xr_profile_space_applied:
            return False

        self._xr_profile_space_applied = True
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return False

        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        has_pos = any(key in view for key in pos_keys) or all(key in view for key in ('x', 'y', 'z'))
        has_rot = any(key in view for key in rot_deg_keys + rot_rad_keys) or 'angle' in view
        auto_center = bool(view.get('auto_center_on_screen', False))
        if not (has_pos or has_rot or auto_center):
            return False
        if self._xr_session is None or self._xr_space is None or self._xr_ref_space_type is None:
            return False

        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            self._xr_profile_space_applied = False
            return False

        desired_head = self._level_head_model_mat4(raw_head)
        if desired_head is None:
            desired_head = raw_head.copy()
        if auto_center:
            auto_pos = self._auto_view_position_from_screen(view, has_rot, rot_deg_keys, rot_rad_keys)
            if auto_pos is None and has_pos:
                auto_pos = self._profile_view_position(view, raw_head[:3, 3].tolist())
            if auto_pos is not None:
                desired_head[:3, 3] = np.array(auto_pos, dtype=np.float32)
        elif has_pos:
            desired_head[:3, 3] = np.array(
                self._profile_view_position(view, raw_head[:3, 3].tolist()),
                dtype=np.float32,
            )
        if has_rot:
            rot = self._profile_view_rotation_rad(view, rot_deg_keys, rot_rad_keys, [0.0, 0.0, 0.0])
            desired_head[:3, :3] = euler_to_mat4(*rot)[:3, :3]

        try:
            current_space_in_ref = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
            reference_head = self._level_head_model_mat4(current_space_in_ref @ raw_head)
            if reference_head is None:
                self._xr_profile_space_applied = False
                return False
            space_in_ref = reference_head @ np.linalg.inv(desired_head)
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=mat4_to_xr_posef(space_in_ref.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to apply profile view_pose to XrSpace: {exc}")
            return False

        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = space_in_ref.astype(np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass

        pos = [round(float(v), 4) for v in desired_head[:3, 3]]
        print(f"[OpenXRViewer] Applied profile view_pose to XrSpace: position={pos}")
        return True


    def _reset_xr_space_to_identity(self):
        """Reset XR reference space offset to identity."""
        if self._xr_session is None or self._xr_ref_space_type is None:
            return
        current = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        if np.allclose(current, np.eye(4)):
            return
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=xr.Posef(),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to reset XR space: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass
        print("[OpenXRViewer] XR space reset to identity")


    def _recenter_profile_view_pose(self):
        """Re-apply profile view_pose, used by Home/Menu long press."""
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view.get('auto_center_on_screen', False):
            return False
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return False
        self._xr_profile_space_applied = False
        applied = self._apply_profile_view_pose_to_xr_space(views)
        if applied:
            print("[OpenXRViewer] Home recentered view_pose to screen center")
        return applied


    def _auto_view_position_from_screen(self, view, has_view_rot, rot_deg_keys, rot_rad_keys):
        """Compute a viewer position centered on the configured screen."""
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return None

        position = screen.get('position', screen.get('screen_position'))
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return None
        try:
            screen_pos = np.array(
                [float(position[0]), float(position[1]), float(position[2])],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return None

        width = self._profile_float(screen, ('width', 'screen_width'), self.screen_width)
        ratio = self._profile_float(view, ('distance_width_ratio', 'view_distance_width_ratio'), 0.6)
        distance = self._profile_float(view, ('distance', 'view_distance'), width * ratio)
        distance = max(0.05, float(distance))

        if has_view_rot:
            yaw, pitch, _roll = self._profile_rotation_rad(
                view, rot_deg_keys, rot_rad_keys, [0.0, 0.0, 0.0]
            )
            cp = math.cos(pitch)
            forward = np.array(
                [-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp],
                dtype=np.float32,
            )
        else:
            screen_rot = self._profile_rotation_rad(
                screen,
                ('rotation_deg', 'screen_rotation_deg'),
                ('rotation', 'screen_rotation'),
                [self.screen_yaw, self.screen_pitch, self.screen_roll],
            )
            screen_normal = euler_to_mat4(*screen_rot)[:3, 2].astype(np.float32)
            forward = -screen_normal

        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return None
        forward = forward / norm

        pos = screen_pos - forward * distance
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right_norm = float(np.linalg.norm(right))
        if right_norm > 1e-6:
            right = right / right_norm
            pos += right * self._profile_float(view, ('horizontal_offset', 'x_offset'), 0.0)
        pos[1] += self._profile_float(view, ('vertical_offset', 'y_offset'), 0.0)
        return [float(pos[0]), float(pos[1]), float(pos[2])]


    def _apply_profile_screen_layout(self, show_border=False):
        """Apply fixed room-specific screen layout from profile.json.

        This does not lock the physical headset pose.  OpenXR head tracking stays
        active; the profile only defines where the virtual screen is placed in
        the room.
        """
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return False

        self._reset_orientation_offsets()
        width = self._profile_float(screen, ('width', 'screen_width'), self.screen_width)
        self.screen_width = max(0.05, width)
        self._screen_ref_size = self.screen_width
        self.screen_height = None


        rotation = self._profile_rotation_rad(
            screen,
            ('rotation_deg', 'screen_rotation_deg'),
            ('rotation', 'screen_rotation'),
            [self.screen_yaw, self.screen_pitch, self.screen_roll],
        )

        position = screen.get('position', screen.get('screen_position'))
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            try:
                x, y, z = float(position[0]), float(position[1]), float(position[2])
                self.screen_pan_x = x
                self.screen_pan_y = y
                self.screen_distance = max(0.05, -z)
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            except (TypeError, ValueError):
                return False
        else:
            view = getattr(self, '_view_pose_profile', {}) or {}
            view_pos = self._profile_vec3(
                view,
                ('position', 'camera_position', 'viewer_position'),
                [0.0, float(self._initial_head_y), 0.0],
            )
            view_rot = self._profile_rotation_rad(
                view,
                ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg'),
                ('rotation', 'camera_rotation', 'viewer_rotation'),
                [0.0, 0.0, 0.0],
            )
            yaw, pitch, _roll = view_rot
            distance = self._profile_float(screen, ('distance', 'screen_distance'), self.screen_distance)
            cp = math.cos(pitch)
            fx = -math.sin(yaw) * cp
            fy = math.sin(pitch)
            fz = -math.cos(yaw) * cp
            self.screen_pan_x = float(view_pos[0] + fx * distance)
            self.screen_pan_y = float(view_pos[1] + fy * distance)
            self.screen_distance = max(0.05, -(float(view_pos[2]) + fz * distance))
            if 'rotation_deg' in screen or 'screen_rotation_deg' in screen or 'rotation' in screen or 'screen_rotation' in screen:
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            else:
                self.screen_yaw = math.atan2(-fx, -fz)
                self.screen_pitch = 0.0
                self.screen_roll = 0.0

        self._last_overlay_update = 0.0
        self._border_alpha = 0.0
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()
        return True