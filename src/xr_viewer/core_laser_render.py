import math

import moderngl
import numpy as np
from OpenGL.GL import GL_CCW, GL_CW, glFrontFace


class CoreLaserRenderMixin:
    def _cursor_ring_specs(self, distance_m):
        scale = float(np.clip(float(distance_m) / 2.0, 1.0, 50.0))
        return (
            (0.0096 * scale, (0.2, 0.6, 1.0, 0.75)),
            (0.0056 * scale, (1.0, 1.0, 1.0, 0.75)),
        )

    @staticmethod
    def _mat3_to_quat(m33):
        """ 3x3 rotation matrix to (x,y,z,w) quaternion. """
        t = m33[0, 0] + m33[1, 1] + m33[2, 2]
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (m33[2, 1] - m33[1, 2]) / s
            y = (m33[0, 2] - m33[2, 0]) / s
            z = (m33[1, 0] - m33[0, 1]) / s
        elif m33[0, 0] > m33[1, 1] and m33[0, 0] > m33[2, 2]:
            s = np.sqrt(1.0 + m33[0, 0] - m33[1, 1] - m33[2, 2]) * 2.0
            w = (m33[2, 1] - m33[1, 2]) / s
            x = 0.25 * s
            y = (m33[0, 1] + m33[1, 0]) / s
            z = (m33[0, 2] + m33[2, 0]) / s
        elif m33[1, 1] > m33[2, 2]:
            s = np.sqrt(1.0 + m33[1, 1] - m33[0, 0] - m33[2, 2]) * 2.0
            w = (m33[0, 2] - m33[2, 0]) / s
            x = (m33[0, 1] + m33[1, 0]) / s
            y = 0.25 * s
            z = (m33[1, 2] + m33[2, 1]) / s
        else:
            s = np.sqrt(1.0 + m33[2, 2] - m33[0, 0] - m33[1, 1]) * 2.0
            w = (m33[1, 0] - m33[0, 1]) / s
            x = (m33[0, 2] + m33[2, 0]) / s
            y = (m33[1, 2] + m33[2, 1]) / s
            z = 0.25 * s
        q = np.array([x, y, z, w], dtype='f8')
        return q / np.linalg.norm(q)

    def _slerp_quat(self, q1, q2, t):
        """Spherical linear interpolation: t=0 -> q1, t=1 -> q2. Input/output as (x,y,z,w) numpy arrays."""
        dot = np.dot(q1, q2)
        if dot < 0.0:
            q2 = -q2
            dot = -dot
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
        theta_0 = math.acos(min(dot, 1.0))
        theta = theta_0 * t
        sin_t = math.sin(theta)
        sin_t0 = math.sin(theta_0)
        s1 = math.cos(theta) - dot * sin_t / sin_t0
        s2 = sin_t / sin_t0
        return s1 * q1 + s2 * q2

    def _smooth_controller_poses(self):
        """Pre-smooth both controller poses once per frame.

        Must be called after _update_aim_poses / _update_grip_poses and
        before any consumer (grip-to-move, cursor, laser rendering).
        Stores smoothed position + forward in _smooth_ray_* attrs.
        """
        for is_left, aim_mat_attr, grip_mat_attr, pos_attr, quat_attr in [
            (True, '_aim_mat_l', '_grip_mat_l', '_smooth_ray_origin_l', '_smooth_ray_quat_l'),
            (False, '_aim_mat_r', '_grip_mat_r', '_smooth_ray_origin_r', '_smooth_ray_quat_r'),
        ]:
            aim_mat = getattr(self, aim_mat_attr)
            if aim_mat is None:
                continue
            grip_mat = getattr(self, grip_mat_attr)
            if grip_mat is not None:
                raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                raw_pos = aim_mat[:3, 3].astype('f8')
            self._apply_ray_smoothing(raw_pos, aim_mat, pos_attr, quat_attr)
            sm_quat = getattr(self, quat_attr)
            if sm_quat is not None:
                x, y, z, w = sm_quat
                fwd = np.array([
                    -(2 * x * z + 2 * w * y),
                    -(2 * y * z - 2 * w * x),
                    -(1 - 2 * x * x - 2 * y * y),
                ], dtype='f8')
                if is_left:
                    self._smooth_ray_fwd_l = fwd
                else:
                    self._smooth_ray_fwd_r = fwd

    def _get_smoothed_ray(self, is_left):
        """Return (smoothed_pos, smoothed_fwd) from pre-computed attrs."""
        pos_attr = '_smooth_ray_origin_l' if is_left else '_smooth_ray_origin_r'
        fwd_attr = '_smooth_ray_fwd_l' if is_left else '_smooth_ray_fwd_r'
        sm_pos = getattr(self, pos_attr)
        sm_fwd = getattr(self, fwd_attr)
        if sm_pos is None or sm_fwd is None:
            return None, None
        return sm_pos.copy(), sm_fwd.copy()

    def _apply_ray_smoothing(self, raw_pos, aim_mat, smooth_pos_attr, smooth_quat_attr):
        """Position EMA + quaternion SLERP smoothing (with dead zone). Returns (smoothed_pos, smoothed_fwd_world)."""
        raw_quat = self._mat3_to_quat(aim_mat[:3, :3].astype('f8'))

        prev_pos = getattr(self, smooth_pos_attr)
        prev_quat = getattr(self, smooth_quat_attr)

        _filter = self._ray_filter_l if smooth_pos_attr.endswith('_l') else self._ray_filter_r
        if prev_pos is None:
            _filter.reset()
        sm_pos = _filter.filter(raw_pos, self._last_frame_dt)
        setattr(self, smooth_pos_attr, sm_pos.copy())

        if prev_quat is not None:
            _dot = abs(np.dot(raw_quat, prev_quat))
            _dot = min(_dot, 1.0)
            _ang = 2.0 * math.acos(_dot) if _dot < 1.0 else 0.0
            if _ang < self._ray_deadzone_rad:
                sm_quat = prev_quat
            else:
                _adaptive = self._rot_smooth * (1.0 + min(_ang * 30.0, 2.0))
                _adaptive = min(_adaptive, 0.30)
                sm_quat = self._slerp_quat(prev_quat, raw_quat, _adaptive)
        else:
            sm_quat = raw_quat
        setattr(self, smooth_quat_attr, sm_quat.copy())
        x, y, z, w = sm_quat
        r33 = np.array([
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ], dtype='f8')
        fwd_w = -r33[:, 2]
        return sm_pos, fwd_w

    def _laser_beam_setup(self):
        """Ray sharing: Quaternion SLERP (direction) + Position EMA ->simulates VD damping."""
        now = self._frame_now
        beams = []
        for aim_mat, grip_mat, last_move_attr, ctrl_name, smooth_pos_attr, smooth_quat_attr in [
            (self._aim_mat_l, self._grip_mat_l, "_laser_last_move_l", 'left', "_smooth_ray_origin_l", "_smooth_ray_quat_l"),
            (self._aim_mat_r, self._grip_mat_r, "_laser_last_move_r", 'right', "_smooth_ray_origin_r", "_smooth_ray_quat_r"),
        ]:
            if aim_mat is None:
                continue
            if (now - getattr(self, last_move_attr)) > self._LASER_HIDE_AFTER:
                setattr(self, smooth_pos_attr, None)
                setattr(self, smooth_quat_attr, None)
                continue

            is_left = (ctrl_name == 'left')
            ctrl_pos, fwd_w = self._get_smoothed_ray(is_left)
            if ctrl_pos is None:
                continue

            if grip_mat is not None:
                raw_pos = (grip_mat[:3, 3] + grip_mat[:3, 1] * 0.020).astype('f8')
            else:
                raw_pos = aim_mat[:3, 3].astype('f8')

            right_w = aim_mat[:3, 0].astype('f8')
            _ang = math.radians(12)
            _ca, _sa = math.cos(_ang), math.sin(_ang)
            _k = right_w / (np.linalg.norm(right_w) + 1e-10)
            fwd_w = fwd_w * _ca + np.cross(_k, fwd_w) * _sa + _k * np.dot(_k, fwd_w) * (1 - _ca)

            if self._laser_screen_hit_uv(raw_pos, fwd_w) is None:
                _raw_fwd = -aim_mat[:3, 2].astype('f8')
                _raw_fwd = _raw_fwd * _ca + np.cross(_k, _raw_fwd) * _sa + _k * np.dot(_k, _raw_fwd) * (1 - _ca)
                if self._laser_screen_hit_uv(raw_pos, _raw_fwd) is None:
                    _plane_uv = self._laser_plane_uv(raw_pos, fwd_w)
                    if _plane_uv is not None:
                        _cu = max(0.0, min(1.0, _plane_uv[0]))
                        _cv = max(0.0, min(1.0, _plane_uv[1]))
                        _clamped_wp = self._screen_uv_to_world(_cu, _cv)
                        _edge_dir = _clamped_wp - raw_pos
                        _norm = np.linalg.norm(_edge_dir)
                        if _norm > 1e-6:
                            _edge_dir /= _norm
                            _dot2 = np.dot(_raw_fwd, _edge_dir)
                            _dot2 = max(-1.0, min(1.0, _dot2))
                            _ang2 = math.acos(_dot2)
                            if _ang2 < self._ray_edge_deadzone_rad:
                                fwd_w = _edge_dir
            if ctrl_name == 'left':
                self._smooth_ray_prev_fwd_l = fwd_w.copy()
            else:
                self._smooth_ray_prev_fwd_r = fwd_w.copy()

            ctrl_pos = raw_pos + fwd_w * 0.11

            right = aim_mat[:3, 0].astype('f4')
            fwd = fwd_w.astype('f4')
            up = np.cross(right, fwd)
            up = up / (np.linalg.norm(up) + 1e-10)
            right2 = np.cross(fwd, up)
            beams.append((now, ctrl_name, aim_mat, ctrl_pos, fwd_w, right2, fwd, up))
        return beams

    def _render_lasers(self, mgl_fbo, vp_mat, blend=False):
        """blend=False: opaque rainbow beam; blend=True: semi-transparent hit circles."""
        if getattr(self, '_beams_frame', -1) != self._frame_count:
            self._cached_beams = self._laser_beam_setup()
            self._beams_frame = self._frame_count
        beams = self._cached_beams
        if not beams:
            return
        if blend:
            self._render_laser_hit_circles(mgl_fbo, vp_mat, beams)
            return
        mgl_fbo.use()
        beam_max_len = 0.4
        for now, ctrl_name, _aim_mat, ctrl_pos, fwd_w, right2, fwd, up in beams:
            cursor_uv = self._cursor_uv_l if ctrl_name == 'left' else self._cursor_uv_r
            if self._cursor_ctrl == ctrl_name and cursor_uv is not None:
                hit_dist = max(0.01, float(cursor_uv[2]))
            else:
                kb_dist = self._keyboard_laser_hit_dist(ctrl_pos, fwd_w)
                sc_dist = self._laser_screen_hit_dist(ctrl_pos, fwd_w)
                ov_dist = self._overlay_panel_hit_dist(ctrl_pos, fwd_w)
                if self._keyboard_visible and kb_dist < 5.0:
                    hit_dist = kb_dist
                else:
                    hit_dist = min(sc_dist, kb_dist, ov_dist)
            draw_len = min(beam_max_len, max(0.01, hit_dist))
            beam_r = 0.006
            scale = np.diag([beam_r, draw_len, beam_r, 1.0]).astype('f4')
            rot = np.eye(4, dtype='f4')
            rot[:3, 0] = right2
            rot[:3, 1] = fwd
            rot[:3, 2] = up
            trans = np.eye(4, dtype='f4')
            trans[:3, 3] = ctrl_pos.astype('f4')
            beam_mvp = vp_mat @ trans @ rot @ scale
            self._beam_prog['u_mvp'].write(beam_mvp.T.tobytes())
            self._beam_prog['u_time'].value = float(now)
            self._beam_vao.render(moderngl.TRIANGLE_STRIP)

    def _render_laser_hit_circles(self, mgl_fbo, vp_mat, beams):
        mgl_fbo.use()
        for _now, ctrl_name, _aim_mat, ctrl_pos, fwd_w, _right2, _fwd, _up in beams:
            kb_dist = self._keyboard_laser_hit_dist(ctrl_pos, fwd_w)
            sc_dist = self._laser_screen_hit_dist(ctrl_pos, fwd_w)
            ov_dist = self._overlay_panel_hit_dist(ctrl_pos, fwd_w)

            beam_len = 30.0
            hit_target = None
            if self._keyboard_visible and kb_dist < 5.0:
                beam_len = kb_dist
                hit_target = 'keyboard'
            if sc_dist < beam_len:
                beam_len = sc_dist
                hit_target = 'screen'
            if ov_dist < beam_len:
                beam_len = ov_dist
                hit_target = 'overlay'

            if hit_target is None or beam_len >= 29.0:
                continue

            hit_pos = ctrl_pos + fwd_w * beam_len
            if hit_target == 'keyboard':
                _sk = '_kb_smooth_l' if ctrl_name == 'left' else '_kb_smooth_r'
                _smooth_pos = getattr(self, _sk, None)
                if _smooth_pos is not None:
                    _cp = math.cos(self._keyboard_pitch)
                    _sp = math.sin(self._keyboard_pitch)
                    _cy = math.cos(self._keyboard_yaw)
                    _sy = math.sin(self._keyboard_yaw)
                    _kb_x = np.array([_cy, 0.0, -_sy], dtype='f8')
                    _kb_y = np.array([_sy * _sp, _cp, _cy * _sp], dtype='f8')
                    _kb_pos = np.array([self._keyboard_pan_x, self._keyboard_pan_y, -self._keyboard_distance], dtype='f8')
                    hit_pos = (_kb_pos + _kb_x * float(_smooth_pos[0]) + _kb_y * float(_smooth_pos[1])).astype('f4')
            for radius, color in self._cursor_ring_specs(beam_len):
                model = np.eye(4, dtype='f4')
                if hit_target == 'screen':
                    _sh, _screen_pos, r_ax, u_ax, screen_n = self._screen_basis()
                    model[:3, 0] = (r_ax * radius).astype('f4')
                    model[:3, 1] = (u_ax * radius).astype('f4')
                    model[:3, 2] = screen_n.astype('f4')
                elif hit_target == 'keyboard':
                    _cp = math.cos(self._keyboard_pitch)
                    _sp = math.sin(self._keyboard_pitch)
                    _cy = math.cos(self._keyboard_yaw)
                    _sy = math.sin(self._keyboard_yaw)
                    _kb_r = np.array([_cy, 0.0, -_sy], dtype='f4')
                    _kb_u = np.array([_sy * _sp, _cp, _cy * _sp], dtype='f4')
                    _kb_nv = np.array([_sy * _cp, -_sp, _cy * _cp], dtype='f4')
                    model[:3, 0] = _kb_r * radius
                    model[:3, 1] = _kb_u * radius
                    model[:3, 2] = _kb_nv
                else:
                    model[0, 0] = radius
                    model[1, 1] = radius
                model[:3, 3] = hit_pos.astype('f4')
                circle_mvp = vp_mat @ model
                self._border_prog['u_mvp'].write(circle_mvp.T.tobytes())
                self._border_prog['u_color'].value = color
                self._circle_vao.render(moderngl.TRIANGLE_FAN)

    def _render_controllers(self, mgl_fbo, vp_mat, view_mat):
        """Render PICO 4 Ultra 3D controller models with Blinn-Phong lighting."""
        now = self._frame_now
        controllers = []
        for grip_mat, prims, last_move_attr, press_attr in [
            (self._grip_mat_l, self._ctrl_prims_l, "_laser_last_move_l", "_ctrl_press_l"),
            (self._grip_mat_r, self._ctrl_prims_r, "_laser_last_move_r", "_ctrl_press_r"),
        ]:
            if (now - getattr(self, last_move_attr)) > self._LASER_HIDE_AFTER:
                continue
            if grip_mat is None or not prims:
                continue
            r_t = view_mat[:3, :3].T
            eye_pos = -r_t @ view_mat[:3, 3]
            dist = float(np.linalg.norm(grip_mat[:3, 3].astype(np.float64) - eye_pos.astype(np.float64)))
            press_amount = max(0.0, min(1.0, float(getattr(self, press_attr, 0.0) or 0.0)))
            controllers.append((dist, grip_mat, prims, press_amount))

        if not controllers:
            return

        controllers.sort(key=lambda x: x[0], reverse=True)
        mgl_fbo.use()
        view_inv = np.linalg.inv(view_mat)
        cam_pos = view_inv[:3, 3].astype(np.float32)

        for _dist, grip_mat, prims, press_amount in controllers:
            t_mat = np.eye(4, dtype=np.float32)
            _off = self._calibration_temp_offset if self._calibration_mode else self._ctrl_model_offset
            _rot = self._calibration_temp_rot if self._calibration_mode else self._ctrl_model_rot_deg
            t_mat[0, 3] = _off[0]
            t_mat[1, 3] = _off[1]
            t_mat[2, 3] = _off[2]

            _ang = math.radians(_rot)
            _ca, _sa = math.cos(_ang), math.sin(_ang)
            r_mat = np.eye(4, dtype=np.float32)
            r_mat[1, 1] = _ca
            r_mat[1, 2] = -_sa
            r_mat[2, 1] = _sa
            r_mat[2, 2] = _ca

            press_mat = np.eye(4, dtype=np.float32)
            if press_amount > 0.001:
                press_mat[1, 3] = -0.0045 * press_amount
                press_mat[2, 3] = -0.0020 * press_amount
                press_mat[0, 0] = 1.0 + 0.0040 * press_amount
                press_mat[1, 1] = 1.0 - 0.0075 * press_amount
                press_mat[2, 2] = 1.0 + 0.0030 * press_amount

            _corr = (r_mat @ t_mat @ press_mat).astype(np.float32)
            model_mat = (grip_mat @ _corr).astype(np.float32)

            press_glow = 0.14 * press_amount
            self._controller_prog['u_mvp'].write(vp_mat.astype(np.float32).T.tobytes())
            self._controller_prog['u_model'].write(model_mat.T.tobytes())
            self._controller_prog['u_light_color'].value = (0.60 + press_glow, 0.60 + press_glow, 0.65 + press_glow)
            self._controller_prog['u_ambient_color'].value = (0.22 + press_glow * 0.35, 0.22 + press_glow * 0.35, 0.24 + press_glow * 0.35)
            self._controller_prog['u_camera_pos'].write(cam_pos.tobytes())

            sorted_prims = sorted(prims, key=lambda p: p['tri_count'], reverse=True)

            if self._use_d3d11:
                glFrontFace(GL_CW)

            for prim in sorted_prims:
                tex = self._ctrl_tex_cache.get(prim['tex_key'])
                if tex is not None:
                    tex.use(location=3)
                    self._controller_prog['u_use_texture'].value = 1
                    self._controller_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
                else:
                    self._controller_prog['u_use_texture'].value = 0
                    self._controller_prog['u_base_color_factor'].value = (0.7, 0.7, 0.7)
                prim['vao'].render(prim.get('render_mode', moderngl.TRIANGLES))

            if self._use_d3d11:
                glFrontFace(GL_CCW)
