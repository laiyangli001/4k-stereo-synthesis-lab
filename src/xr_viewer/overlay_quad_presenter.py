import ctypes
import math
import os
import time

import numpy as np
from OpenGL.GL import (
    GL_RGBA,
    GL_TEXTURE_2D,
    GL_UNPACK_ALIGNMENT,
    GL_UNSIGNED_BYTE,
    glBindTexture,
    glPixelStorei,
    glTexSubImage2D,
)

try:
    import xr
except ImportError:
    xr = None

from .overlay_textures import (
    build_cursor_rgba,
    build_fps_overlay_rgba,
    build_help_rgba,
    build_keyboard_rgba,
    build_short_osd_rgba,
    build_team_help_rgba,
    build_team_status_rgba,
)


def _quat_from_matrix(mat):
    m = np.asarray(mat, dtype=np.float64)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        return ((m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s)
    if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return (0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s)
    if m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return ((m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s)
    s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return ((m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s)


def _layer_flags():
    flags = getattr(xr, "CompositionLayerFlags", None)
    value = getattr(flags, "BLEND_TEXTURE_SOURCE_ALPHA_BIT", 0) if flags is not None else 0
    return value


def _normalized(v):
    arr = np.asarray(v, dtype=np.float64)
    return arr / (np.linalg.norm(arr) + 1e-10)


def _pose_from_basis(right, up, fwd, pos):
    mat = np.eye(3, dtype=np.float64)
    mat[:, 0] = _normalized(right)
    mat[:, 1] = _normalized(up)
    mat[:, 2] = _normalized(fwd)
    qx, qy, qz, qw = _quat_from_matrix(mat)
    return (qx, qy, qz, qw, float(pos[0]), float(pos[1]), float(pos[2]))


class QuadOverlayPresenter:
    def __init__(self, viewer):
        self.viewer = viewer
        self._entries = {}
        self._layers = []
        self._rgba_cache = {}

    def prepare_layers(self):
        self._layers = []
        viewer = self.viewer
        viewer._overlay_quad_presented_keys = set()
        if xr is None:
            return []
        if getattr(viewer, "_use_d3d11", False) and getattr(viewer, "_d3d11_native_renderer", None) is None:
            return []
        headers = []
        for spec in self._active_specs():
            try:
                layer = self._update_layer(spec)
            except Exception as exc:
                viewer._breakdown_inc("openxr_overlay_quad_failed")
                backend = "D3D11" if getattr(viewer, "_use_d3d11", False) else "OpenGL"
                print(f"[OpenXRViewer] {backend} overlay quad failed: {type(exc).__name__}: {exc}")
                continue
            if layer is not None:
                self._layers.append(layer)
                viewer._overlay_quad_presented_keys.add(spec["key"])
                headers.append(ctypes.cast(ctypes.pointer(layer), ctypes.POINTER(xr.CompositionLayerBaseHeader)))
        return headers

    def _active_specs(self):
        specs = []
        keyboard = self._keyboard_spec()
        if keyboard is not None:
            specs.append(keyboard)
            specs.extend(self._keyboard_cursor_specs())
        osd = self._osd_spec()
        if osd is not None:
            specs.append(osd)
        hand_fps = self._hand_fps_spec()
        if hand_fps is not None:
            specs.append(hand_fps)
        hand_help = self._hand_help_spec()
        if hand_help is not None:
            specs.append(hand_help)
        team_status = self._team_status_spec()
        if team_status is not None:
            specs.append(team_status)
        team_help = self._team_help_spec()
        if team_help is not None:
            specs.append(team_help)
        return specs

    def _keyboard_spec(self):
        v = self.viewer
        if not getattr(v, "_keyboard_visible", False):
            return None
        if not getattr(v, "_keyboard_keys", None):
            init_keyboard = getattr(v, "_init_keyboard", None)
            if callable(init_keyboard):
                init_keyboard()
        show_shift = bool(getattr(v, "_kb_show_shifted", False))
        hover_indices = tuple(i for i in (getattr(v, "_kb_hover_l", None), getattr(v, "_kb_hover_r", None)) if i is not None)
        held_indices = tuple(i for i in (getattr(v, "_kb_held_key_l", None), getattr(v, "_kb_held_key_r", None)) if i is not None)
        content_key = (
            "keyboard",
            show_shift,
            round(float(v._keyboard_width), 4),
            round(float(v._keyboard_height), 4),
            hover_indices,
            held_indices,
        )
        if getattr(v, "_keyboard_content_key", None) != content_key:
            refresh = getattr(v, "_refresh_keyboard_content", None)
            if callable(refresh):
                refresh(hover_indices=hover_indices, held_indices=held_indices)
        rgba = getattr(v, "_keyboard_rgba", None)
        if rgba is None:
            rgba, keys = build_keyboard_rgba(
                show_shift,
                float(v._keyboard_width),
                float(v._keyboard_height),
                getattr(v, "font_type", None),
                hover_indices=hover_indices,
                held_indices=held_indices,
            )
            v._keyboard_keys = keys
        world = v._kb_world_mat()
        qx, qy, qz, qw = _quat_from_matrix(world[:3, :3])
        return {
            "key": "keyboard",
            "rgba": rgba,
            "pose": (qx, qy, qz, qw, float(world[0, 3]), float(world[1, 3]), float(world[2, 3])),
            "size": (float(v._keyboard_width), float(v._keyboard_height)),
            "content_key": content_key,
        }

    def _keyboard_cursor_specs(self):
        v = self.viewer
        specs = []
        world = v._kb_world_mat()
        qx, qy, qz, qw = _quat_from_matrix(world[:3, :3])
        normal = np.asarray(world[:3, 2], dtype=np.float64)
        rgba = self._cached_rgba(("keyboard_cursor_rgba",), lambda: build_cursor_rgba(64))
        for hand, smooth_attr, hover_attr in (("l", "_kb_smooth_l", "_kb_hover_l"), ("r", "_kb_smooth_r", "_kb_hover_r")):
            if getattr(v, hover_attr, None) is None:
                continue
            pos_local = getattr(v, smooth_attr, None)
            if pos_local is None:
                continue
            try:
                local = np.array([float(pos_local[0]), float(pos_local[1]), 0.0, 1.0], dtype=np.float64)
            except Exception:
                continue
            pos = (world.astype(np.float64) @ local)[:3] + normal * 0.003
            dist = max(0.1, float(getattr(v, "_keyboard_distance", 1.0) or 1.0))
            radius = 0.012 * float(np.clip(dist / 2.0, 0.35, 50.0))
            specs.append({
                "key": f"keyboard_cursor_{hand}",
                "rgba": rgba,
                "pose": (qx, qy, qz, qw, float(pos[0]), float(pos[1]), float(pos[2])),
                "size": (radius * 2.0, radius * 2.0),
                "content_key": ("keyboard_cursor", hand),
            })
        return specs

    def _cached_rgba(self, content_key, builder):
        cached = self._rgba_cache.get(content_key)
        if cached is None:
            cached = builder()
            self._rgba_cache[content_key] = cached
        return cached

    def _overlay_lang(self):
        lang = str(os.environ.get("DESKTOP2STEREO_LOCALE", "EN") or "EN").strip().upper()
        return "CN" if lang.startswith("CN") or lang.startswith("ZH") else "EN"

    def _osd_spec(self):
        v = self.viewer
        now = float(getattr(v, "_frame_now", time.perf_counter()) or 0.0)
        lines = []
        if getattr(v, "_preset_name_overlay", None) and now - float(getattr(v, "_preset_osd_show_t", -999.0) or -999.0) < 5.0:
            lines.append(str(v._preset_name_overlay))
        if now - float(getattr(v, "_depth_osd_show_t", -999.0) or -999.0) < 2.5:
            lines.append(f"Depth Strength {float(getattr(v, 'depth_strength', 0.0) or 0.0):.2f}")
        if now - float(getattr(v, "_screen_osd_show_t", -999.0) or -999.0) < 2.5:
            lines.append(f"Screen {float(getattr(v, 'screen_width', 0.0) or 0.0):.2f}m @ {float(getattr(v, 'screen_distance', 0.0) or 0.0):.2f}m")
        if not lines:
            return None
        content_key = ("osd", tuple(lines))
        rgba = self._cached_rgba(content_key, lambda: build_short_osd_rgba(lines, getattr(v, "font_type", None)))
        qx, qy, qz, qw = v._screen_pose_quat_xyzw()
        screen_h = float(getattr(v, "screen_height", 0.0) or (float(getattr(v, "screen_width", 1.6)) * 9.0 / 16.0))
        pos = np.array([float(v.screen_pan_x), float(v.screen_pan_y) + screen_h * 0.5 + 0.08, -float(v.screen_distance)], dtype=np.float32)
        return {
            "key": "osd",
            "rgba": rgba,
            "pose": (qx, qy, qz, qw, float(pos[0]), float(pos[1]), float(pos[2])),
            "size": (0.65, 0.08),
            "content_key": content_key,
        }

    def _refresh_fps_cache(self):
        v = self.viewer
        now = float(getattr(v, "_frame_now", time.perf_counter()) or 0.0)
        if now - float(getattr(v, "_last_overlay_update", 0.0) or 0.0) >= 1.0:
            v._cached_actual_fps = float(getattr(v, "actual_fps", 0.0) or 0.0)
            v._cached_sbs_fps = float(getattr(v, "sbs_fps", 0.0) or 0.0)
            v._cached_latency = float(getattr(v, "total_latency", 0.0) or 0.0)
            v._cached_screen_width = float(getattr(v, "screen_width", 0.0) or 0.0)
            v._cached_screen_height = float(getattr(v, "screen_width", 0.0) or 0.0) * 9.0 / 16.0
            screen_dist = getattr(v, "_screen_view_distance", None)
            v._cached_screen_dist = float(screen_dist() if callable(screen_dist) else getattr(v, "screen_distance", 0.0))
            v._cached_depth_strength = float(getattr(v, "depth_strength", 0.0) or 0.0)
            v._cached_help_visible = bool(getattr(v, "_team_help_visible", False) or getattr(v, "_help_panel_visible", False))
            v._cached_vr_res = tuple(getattr(v, "_swapchain_sizes", {}).get(0, (0, 0)))
            v._cached_sbs_res = tuple(getattr(v, "frame_size", (0, 0)))
            v._last_overlay_update = now

    def _fps_values_key(self):
        v = self.viewer
        return (
            round(float(getattr(v, "_cached_actual_fps", 0.0) or 0.0), 1),
            round(float(getattr(v, "_cached_sbs_fps", 0.0) or 0.0), 1),
            round(float(getattr(v, "_cached_latency", 0.0) or 0.0), 1),
            round(float(getattr(v, "_cached_screen_width", 0.0) or 0.0), 3),
            round(float(getattr(v, "_cached_screen_height", 0.0) or 0.0), 3),
            round(float(getattr(v, "_cached_screen_dist", 0.0) or 0.0), 3),
            round(float(getattr(v, "_cached_depth_strength", 0.0) or 0.0), 3),
            tuple(getattr(v, "_cached_vr_res", (0, 0))),
            tuple(getattr(v, "_cached_sbs_res", (0, 0))),
            str(getattr(v, "_current_brand", "") or ""),
            bool(getattr(v, "_env_model_visible", False)),
        )

    def _hand_fps_pose_size(self, tex_size):
        v = self.viewer
        overlay_h = 0.075
        ow, oh = tex_size
        overlay_w = overlay_h * (float(ow) / max(1.0, float(oh)))
        panel_pos = panel_fwd = panel_up = None
        if getattr(v, "_grip_mat_l", None) is not None and getattr(v, "_aim_mat_l", None) is not None:
            grip_up = _normalized(v._grip_mat_l[:3, 1])
            fwd_w = -np.asarray(v._aim_mat_l[:3, 2], dtype=np.float64)
            right_w = np.asarray(v._aim_mat_l[:3, 0], dtype=np.float64)
            ang = math.radians(12)
            ca, sa = math.cos(ang), math.sin(ang)
            axis = _normalized(right_w)
            laser_fwd = fwd_w * ca + np.cross(axis, fwd_w) * sa + axis * np.dot(axis, fwd_w) * (1 - ca)
            laser_fwd = _normalized(laser_fwd)
            grip_pos = np.asarray(v._grip_mat_l[:3, 3], dtype=np.float64)
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11
            panel_fwd = _normalized(grip_up - laser_fwd)
            panel_up = grip_up
            panel_right = _normalized(np.cross(panel_up, panel_fwd))
            panel_up2 = _normalized(np.cross(panel_fwd, panel_right))
            panel_pos = laser_origin + panel_fwd * 0.05 + panel_up2 * (0.10 - overlay_h / 2.0)
        if panel_pos is None and getattr(v, "_head_pos_w", None) is not None and getattr(v, "_head_fwd_w", None) is not None:
            head = np.asarray(v._head_pos_w, dtype=np.float64)
            fwd = np.asarray(v._head_fwd_w, dtype=np.float64)
            panel_pos = head + fwd * 1.0 + np.array([0.0, -0.15, 0.0], dtype=np.float64)
            panel_fwd = -fwd
            panel_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if panel_pos is None:
            return None
        panel_right = _normalized(np.cross(panel_up, panel_fwd))
        panel_up2 = _normalized(np.cross(panel_fwd, panel_right))
        return _pose_from_basis(panel_right, panel_up2, panel_fwd, panel_pos), (overlay_w, overlay_h)

    def _hand_fps_spec(self):
        v = self.viewer
        if not getattr(v, "_hand_fps_visible", False) or getattr(v, "screen_height", None) is None:
            return None
        self._refresh_fps_cache()
        tex_size = tuple(getattr(v, "_overlay_tex_size", (768, 224)))
        pose_size = self._hand_fps_pose_size(tex_size)
        if pose_size is None:
            return None
        pose, size = pose_size
        content_key = ("hand_fps", self._fps_values_key())
        rgba = self._cached_rgba(content_key, lambda: build_fps_overlay_rgba(
            actual_fps=getattr(v, "_cached_actual_fps", 0.0),
            sbs_fps=getattr(v, "_cached_sbs_fps", 0.0),
            latency_ms=getattr(v, "_cached_latency", 0.0),
            screen_width=getattr(v, "_cached_screen_width", 0.0),
            screen_height=getattr(v, "_cached_screen_height", 0.0),
            screen_distance=getattr(v, "_cached_screen_dist", 0.0),
            depth_strength=getattr(v, "_cached_depth_strength", 0.0),
            vr_res=getattr(v, "_cached_vr_res", (0, 0)),
            sbs_res=getattr(v, "_cached_sbs_res", (0, 0)),
            controller_brand=getattr(v, "_current_brand", None),
            environment_visible=getattr(v, "_env_model_visible", False),
            font_type=getattr(v, "font_type", None),
            size=tex_size,
        ))
        return {"key": "hand_fps", "rgba": rgba, "pose": pose, "size": size, "content_key": content_key}

    def _hand_help_spec(self):
        v = self.viewer
        if not getattr(v, "_fps_overlay_visible", False):
            return None
        content_key = ("hand_help", bool(getattr(v, "ENVIRONMENT_MODE", False)), self._overlay_lang())
        rgba = self._cached_rgba(content_key, lambda: build_help_rgba(
            environment_mode=getattr(v, "ENVIRONMENT_MODE", False),
            font_type=getattr(v, "font_type", None),
            lang=self._overlay_lang(),
        ))
        tex_size = (int(rgba.shape[1]), int(rgba.shape[0]))
        pose_size = self._hand_help_pose_size(tex_size)
        if pose_size is None:
            return None
        pose, size = pose_size
        return {"key": "hand_help", "rgba": rgba, "pose": pose, "size": size, "content_key": content_key}

    def _hand_help_pose_size(self, tex_size):
        v = self.viewer
        panel_h = 0.2
        panel_w = panel_h * (float(tex_size[0]) / max(1.0, float(tex_size[1])))
        panel_pos = panel_fwd = panel_up = None
        if getattr(v, "_grip_mat_r", None) is not None and getattr(v, "_aim_mat_r", None) is not None:
            grip_up = _normalized(v._grip_mat_r[:3, 1])
            fwd_w = -np.asarray(v._aim_mat_r[:3, 2], dtype=np.float64)
            right_w = np.asarray(v._aim_mat_r[:3, 0], dtype=np.float64)
            ang = math.radians(12)
            ca, sa = math.cos(ang), math.sin(ang)
            axis = _normalized(right_w)
            laser_fwd = _normalized(fwd_w * ca + np.cross(axis, fwd_w) * sa + axis * np.dot(axis, fwd_w) * (1 - ca))
            grip_pos = np.asarray(v._grip_mat_r[:3, 3], dtype=np.float64)
            laser_origin = grip_pos + grip_up * 0.020 + laser_fwd * 0.11
            panel_fwd = _normalized(grip_up - laser_fwd)
            panel_up = grip_up
            panel_right = _normalized(np.cross(panel_up, panel_fwd))
            panel_up2 = _normalized(np.cross(panel_fwd, panel_right))
            panel_pos = laser_origin + panel_fwd * 0.05 + panel_up2 * (panel_h + 0.025 - panel_h / 2.0)
        if panel_pos is None and getattr(v, "_head_pos_w", None) is not None and getattr(v, "_head_fwd_w", None) is not None:
            head = np.asarray(v._head_pos_w, dtype=np.float64)
            fwd = np.asarray(v._head_fwd_w, dtype=np.float64)
            panel_pos = head + fwd * 1.2 + np.array([0.0, -0.3, 0.0], dtype=np.float64)
            panel_fwd = -fwd
            panel_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if getattr(v, "_panel_mode", 2) == 0 and panel_pos is not None and getattr(v, "_head_pos_w", None) is not None:
            panel_fwd = _normalized(np.asarray(v._head_pos_w, dtype=np.float64) - panel_pos)
            panel_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        if panel_pos is None:
            return None
        panel_right = _normalized(np.cross(panel_up, panel_fwd))
        panel_up2 = _normalized(np.cross(panel_fwd, panel_right))
        return _pose_from_basis(panel_right, panel_up2, panel_fwd, panel_pos), (panel_w, panel_h)

    def _team_status_spec(self):
        v = self.viewer
        if not getattr(v, "_team_fps_visible", False) or getattr(v, "screen_height", None) is None:
            return None
        self._refresh_fps_cache()
        gap, panel_h = v._team_status_panel_metrics()
        tex_size = tuple(getattr(v, "_team_status_tex_size", (768, 224)))
        panel_w = panel_h * (float(tex_size[0]) / max(1.0, float(tex_size[1])))
        pose = self._screen_local_pose(-float(v.screen_width) / 2.0 + panel_w / 2.0, -float(v.screen_height) / 2.0 - gap - panel_h / 2.0)
        trigger_held = bool(getattr(v, "_team_ov_ltrig_held", False) or getattr(v, "_team_ov_rtrig_held", False))
        if trigger_held:
            dt = max(0.001, float(getattr(v, "_last_frame_dt", 0.016) or 0.016))
            v._team_status_alpha = max(0.15, float(getattr(v, "_team_status_alpha", 1.0) or 1.0) - 8.0 * dt)
        else:
            v._team_status_alpha = 1.0
        content_key = (
            "team_status",
            self._fps_values_key(),
            bool(getattr(v, "_cached_help_visible", False)),
            str(getattr(v, "_active_environment", None) or getattr(v, "_environment_model", None) or "Default"),
        )
        rgba = self._cached_rgba(content_key, lambda: build_team_status_rgba(
            actual_fps=getattr(v, "_cached_actual_fps", 0.0),
            sbs_fps=getattr(v, "_cached_sbs_fps", 0.0),
            latency_ms=getattr(v, "_cached_latency", 0.0),
            screen_width=getattr(v, "_cached_screen_width", 0.0),
            screen_height=getattr(v, "_cached_screen_height", 0.0),
            screen_distance=getattr(v, "_cached_screen_dist", 0.0),
            depth_strength=getattr(v, "_cached_depth_strength", 0.0),
            vr_res=getattr(v, "_cached_vr_res", (0, 0)),
            sbs_res=getattr(v, "_cached_sbs_res", (0, 0)),
            environment_name=(getattr(v, "_active_environment", None) or getattr(v, "_environment_model", None) or "Default"),
            controller_brand=getattr(v, "_current_brand", None),
            shortcuts_visible=getattr(v, "_cached_help_visible", False),
            font_type=getattr(v, "font_type", None),
            size=tex_size,
        ))
        return {"key": "team_status", "rgba": rgba, "pose": pose, "size": (panel_w, panel_h), "content_key": content_key}

    def _team_help_spec(self):
        v = self.viewer
        if not (getattr(v, "_team_status_visible", False) and getattr(v, "_team_help_visible", False)) or getattr(v, "screen_height", None) is None:
            return None
        content_key = ("team_help", self._overlay_lang())
        rgba = self._cached_rgba(content_key, lambda: build_team_help_rgba(
            font_type=getattr(v, "font_type", None),
            lang=self._overlay_lang(),
        ))
        tex_w, tex_h = int(rgba.shape[1]), int(rgba.shape[0])
        panel_h = float(v.screen_height)
        panel_w = panel_h * (float(tex_w) / max(1.0, float(tex_h)))
        sx = float(v.screen_width) / 2.0
        gap = panel_h * 0.02
        head_w = np.asarray(getattr(v, "_head_pos_w", None), dtype=np.float32) if getattr(v, "_head_pos_w", None) is not None else np.zeros(3, dtype=np.float32)
        screen_c_w = np.array([float(v.screen_pan_x), float(v.screen_pan_y), -float(v.screen_distance)], dtype=np.float32)
        screen_mat = v._screen_pose_mat4()
        r3 = screen_mat[:3, :3].astype(np.float32)
        head_local = r3.T @ (head_w - screen_c_w)
        hinge_local = np.array([-sx - gap, 0.0, 0.0], dtype=np.float32)
        to_user = _normalized(head_local - hinge_local)
        theta = math.atan2(float(to_user[0]), float(to_user[2]))
        ct, st = math.cos(theta), math.sin(theta)
        ry = np.eye(4, dtype=np.float32)
        ry[0, 0] = ct
        ry[0, 2] = st
        ry[2, 0] = -st
        ry[2, 2] = ct
        t_hinge = np.eye(4, dtype=np.float32)
        t_hinge[0, 3] = -sx - gap
        t_offset = np.eye(4, dtype=np.float32)
        t_offset[0, 3] = -panel_w / 2.0
        model = screen_mat @ t_hinge @ ry @ t_offset
        pose = _pose_from_basis(model[:3, 0], model[:3, 1], model[:3, 2], model[:3, 3])
        return {"key": "team_help", "rgba": rgba, "pose": pose, "size": (panel_w, panel_h), "content_key": content_key}

    def _screen_local_pose(self, local_x, local_y):
        v = self.viewer
        model = v._screen_pose_mat4()
        local = np.array([float(local_x), float(local_y), 0.0, 1.0], dtype=np.float32)
        pos = model @ local
        return _pose_from_basis(model[:3, 0], model[:3, 1], model[:3, 2], pos[:3])

    def _update_layer(self, spec):
        entry = self._ensure_entry(spec)
        swapchain = entry["swapchain"]
        img_index = xr.acquire_swapchain_image(swapchain, self.viewer._xr_sc_acquire_info)
        self.viewer._wait_swapchain_image(swapchain)
        released = False
        try:
            image = entry["images"][img_index]
            content_keys = entry["content_keys"]
            if content_keys.get(int(img_index)) != spec["content_key"]:
                rgba = spec["rgba"]
                if getattr(self.viewer, "_use_d3d11", False):
                    if getattr(self.viewer, "_swapchain_is_bgra", False):
                        rgba = rgba.copy()
                        rgba[..., [0, 2]] = rgba[..., [2, 0]]
                    self.viewer._d3d11_native_renderer._update_subresource(
                        image.texture,
                        rgba.ctypes.data,
                        int(rgba.shape[1]) * 4,
                    )
                else:
                    self._upload_opengl_rgba(image.image, rgba)
                content_keys[int(img_index)] = spec["content_key"]
                self.viewer._breakdown_inc("openxr_overlay_quad_upload")
            xr.release_swapchain_image(swapchain, self.viewer._xr_sc_release_info)
            released = True
        finally:
            if not released:
                try:
                    xr.release_swapchain_image(swapchain, self.viewer._xr_sc_release_info)
                except Exception:
                    pass
        qx, qy, qz, qw, px, py, pz = spec["pose"]
        width, height = spec["size"]
        layer_kwargs = dict(
            space=self.viewer._xr_space,
            eye_visibility=xr.EyeVisibility.BOTH,
            sub_image=xr.SwapchainSubImage(
                swapchain=swapchain,
                image_rect=xr.Rect2Di(
                    offset=xr.Offset2Di(x=0, y=0),
                    extent=xr.Extent2Di(width=entry["size"][0], height=entry["size"][1]),
                ),
                image_array_index=0,
            ),
            pose=xr.Posef(
                orientation=xr.Quaternionf(x=float(qx), y=float(qy), z=float(qz), w=float(qw)),
                position=xr.Vector3f(x=float(px), y=float(py), z=float(pz)),
            ),
            size=xr.Extent2Df(width=float(width), height=float(height)),
        )
        flags = _layer_flags()
        if flags:
            layer_kwargs["layer_flags"] = flags
        try:
            return xr.CompositionLayerQuad(**layer_kwargs)
        except TypeError:
            layer_kwargs.pop("layer_flags", None)
            return xr.CompositionLayerQuad(**layer_kwargs)

    def _ensure_entry(self, spec):
        key = spec["key"]
        h, w = spec["rgba"].shape[:2]
        entry = self._entries.get(key)
        if entry is not None and entry.get("size") == (w, h):
            return entry
        if entry is not None:
            try:
                xr.destroy_swapchain(entry["swapchain"])
            except Exception:
                pass
        sc_info = xr.SwapchainCreateInfo(
            usage_flags=xr.SwapchainUsageFlags.COLOR_ATTACHMENT_BIT | xr.SwapchainUsageFlags.SAMPLED_BIT,
            format=self._swapchain_format(),
            sample_count=1,
            width=int(w),
            height=int(h),
            face_count=1,
            array_size=1,
            mip_count=1,
        )
        swapchain = xr.create_swapchain(self.viewer._xr_session, sc_info)
        entry = {
            "swapchain": swapchain,
            "images": xr.enumerate_swapchain_images(swapchain, self.viewer._quad_swapchain_image_type),
            "size": (int(w), int(h)),
            "content_keys": {},
        }
        self._entries[key] = entry
        return entry

    def _swapchain_format(self):
        fmt = getattr(self.viewer, "_quad_swapchain_format", None)
        if fmt is not None:
            return fmt
        if getattr(self.viewer, "_use_d3d11", False):
            return self.viewer._d3d11_swapchain_fmt
        formats = tuple(getattr(self.viewer, "_quad_swapchain_formats", ()) or ())
        if formats:
            return formats[0]
        return getattr(self.viewer, "_xr_opengl_swapchain_format")

    def _upload_opengl_rgba(self, texture, rgba):
        # OpenGL interprets the first uploaded row as texture-space bottom.
        # Overlay RGBA builders produce top-first PIL/NumPy rows, so flip only
        # on the OpenGL swapchain upload path. D3D11 keeps the original order.
        rgba = np.ascontiguousarray(np.flipud(rgba))
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glBindTexture(GL_TEXTURE_2D, int(texture))
        try:
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                int(rgba.shape[1]),
                int(rgba.shape[0]),
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                rgba,
            )
        finally:
            glBindTexture(GL_TEXTURE_2D, 0)
