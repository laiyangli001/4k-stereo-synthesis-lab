#!/usr/bin/env python3
"""Preview a room profile's view_pose and screen layout without OpenXR."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

import glfw
import moderngl
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))
os.chdir(APP_DIR)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from xrviewer_core import load_glb_model  # noqa: E402


ENV_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_normal;
out vec3 v_position;
out vec2 v_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_uv = in_uv;
    gl_Position = u_mvp * world_pos;
}
"""

ENV_FRAG = """
#version 330
in vec3 v_normal;
in vec3 v_position;
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_tex;
uniform int u_use_texture;
uniform vec3 u_base_color;
uniform vec3 u_camera_pos;
uniform vec3 u_ambient_color;
uniform vec3 u_light_color;
uniform float u_alpha;
void main() {
    vec3 base = u_base_color;
    if (u_use_texture == 1) {
        base *= texture(u_tex, v_uv).rgb;
    }
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_camera_pos + vec3(0.0, 0.2, 0.0) - v_position);
    float diff = max(abs(dot(N, L)), 0.12);
    vec3 color = base * (u_ambient_color + u_light_color * diff);
    fragColor = vec4(color, u_alpha);
}
"""

SCREEN_VERT = """
#version 330
in vec3 in_position;
in vec2 in_uv;
out vec2 v_uv;
uniform mat4 u_mvp;
void main() {
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

SCREEN_FRAG = """
#version 330
in vec2 v_uv;
out vec4 fragColor;
uniform vec4 u_color;
void main() {
    vec2 g = abs(fract(v_uv * vec2(16.0, 9.0)) - 0.5);
    float line = step(0.47, max(g.x, g.y));
    vec3 grid = mix(u_color.rgb, vec3(1.0), line * 0.35);
    fragColor = vec4(grid, u_color.a);
}
"""


def _vec3(data, default):
    if isinstance(data, (list, tuple)) and len(data) >= 3:
        try:
            return [float(data[0]), float(data[1]), float(data[2])]
        except (TypeError, ValueError):
            pass
    return list(default)


def _rot_deg(data, default=(0.0, 0.0, 0.0)):
    return [math.radians(v) for v in _vec3(data, default)]


def _load_profile(room: str):
    room_dir = APP_DIR / "environment" / room
    profile_path = room_dir / "profile.json"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile.json not found: {profile_path}")
    with profile_path.open("r", encoding="utf-8") as f:
        profile = json.load(f)
    if not isinstance(profile, dict):
        raise ValueError(f"profile.json root must be object: {profile_path}")

    glb_name = str(profile.get("glb", "environment.glb") or "environment.glb")
    glb_path = Path(glb_name)
    if not glb_path.is_absolute():
        glb_path = room_dir / glb_name
    if not glb_path.exists():
        raise FileNotFoundError(f"GLB not found: {glb_path}")
    return room_dir, profile_path, profile, glb_path


def _save_profile(path: Path, profile: dict):
    # Runtime reads GLB-embedded KHR_lights_punctual lights, not profile.gltf_lights.
    # Keep saved room profiles aligned with xrviewer_env.py's profile schema.
    profile.pop("gltf_lights", None)
    profile.setdefault("env_fill_lights", [])
    with path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _mat_from_trs(pos, rot_rad, scale=(1.0, 1.0, 1.0)):
    yaw, pitch, roll = rot_rad
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0, sy, 0], [0, 1, 0, 0], [-sy, 0, cy, 0], [0, 0, 0, 1]], dtype="f4")
    rx = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype="f4")
    rz = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype="f4")
    sm = np.diag([float(scale[0]), float(scale[1]), float(scale[2]), 1.0]).astype("f4")
    tm = np.eye(4, dtype="f4")
    tm[:3, 3] = np.array(pos, dtype="f4")
    return tm @ ry @ rx @ rz @ sm


def _view_matrix(pos, rot_rad):
    yaw, pitch, roll = rot_rad
    model = _mat_from_trs(pos, (yaw, pitch, roll), (1.0, 1.0, 1.0))
    return np.linalg.inv(model).astype("f4")


def _projection(aspect, fov_deg=80.0, near=0.03, far=200.0):
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0],
    ], dtype="f4")


def _screen_vertices(screen):
    width = float(screen.get("width", 2.4))
    height = float(screen.get("height", width * 9.0 / 16.0))
    pos = _vec3(screen.get("position"), [0.0, 1.2, -2.0])
    rot = _rot_deg(screen.get("rotation_deg", screen.get("rotation")), [0.0, 0.0, 0.0])
    model = _mat_from_trs(pos, rot, (1.0, 1.0, 1.0))
    corners = np.array([
        [-width / 2, -height / 2, 0, 0, 0],
        [ width / 2, -height / 2, 0, 1, 0],
        [-width / 2,  height / 2, 0, 0, 1],
        [ width / 2,  height / 2, 0, 1, 1],
    ], dtype="f4")
    p = np.c_[corners[:, :3], np.ones(4, dtype="f4")]
    corners[:, :3] = (model @ p.T).T[:, :3]
    return corners


def _make_env_resources(ctx, prog, glb_path: Path):
    prims_data, textures, _lights = load_glb_model(str(glb_path))
    tex_cache = {}
    for tid, arr in enumerate(textures):
        if arr is None:
            continue
        h, w = arr.shape[:2]
        tex = ctx.texture((w, h), 4, arr.tobytes())
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.build_mipmaps()
        tex.anisotropy = 8.0
        tex_cache[tid] = tex

    prims = []
    for pd in prims_data:
        vbo = ctx.buffer(pd["vertices"].astype("f4").tobytes())
        ibo = ctx.buffer(pd["indices"].astype("u4").tobytes())
        vao = ctx.vertex_array(prog, [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")], ibo)
        prims.append({
            "vao": vao,
            "tex_id": int(pd.get("tex_id", -1)),
            "base_color": np.array(pd.get("base_color", [1.0, 1.0, 1.0]), dtype="f4"),
            "base_alpha": float(pd.get("base_alpha", 1.0)),
        })
    return prims, tex_cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("room", nargs="?", default="bedroom")
    args = parser.parse_args()

    os.chdir(APP_DIR)
    room_dir, profile_path, profile, glb_path = _load_profile(args.room)
    view_pose = profile.setdefault("view_pose", {})
    screen = profile.setdefault("screen", {})
    screen.setdefault("name", "Preview Screen")
    screen.setdefault("width", 2.4)
    screen.setdefault("position", [0.0, 1.2, -2.0])
    screen.setdefault("rotation_deg", [0.0, 0.0, 0.0])

    if not glfw.init():
        raise RuntimeError("GLFW init failed")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(1280, 720, f"Room Layout Preview - {args.room}", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW window creation failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST)
    env_prog = ctx.program(vertex_shader=ENV_VERT, fragment_shader=ENV_FRAG)
    screen_prog = ctx.program(vertex_shader=SCREEN_VERT, fragment_shader=SCREEN_FRAG)

    env_prims, tex_cache = _make_env_resources(ctx, env_prog, glb_path)
    screen_vbo = ctx.buffer(reserve=4 * 5 * 4)
    screen_vao = ctx.vertex_array(screen_prog, [(screen_vbo, "3f 2f", "in_position", "in_uv")])

    model_pos = _vec3(profile.get("model_position"), [0.0, -1.0, -3.0])
    model_rot = _rot_deg(profile.get("model_rotation_deg", profile.get("model_rotation")), [0.0, 0.0, 0.0])
    model_scale = _vec3(profile.get("model_scale"), [1.0, 1.0, 1.0])
    env_model = _mat_from_trs(model_pos, model_rot, model_scale)

    view_pos = _vec3(view_pose.get("position"), [0.0, 1.2, 0.0])
    view_rot = _rot_deg(view_pose.get("rotation_deg", view_pose.get("rotation")), [0.0, 0.0, 0.0])
    speed = 0.75
    rot_speed = 45.0
    size_speed = 0.8
    saved_flash = 0.0
    edit_target = "SCREEN"
    tab_was_down = False
    mouse_look = False
    last_mouse = (0.0, 0.0)

    print(f"Room: {args.room}")
    print(f"Profile: {profile_path}")
    print("Controls:")
    print("  Tab: switch edit target SCREEN/VIEW")
    print("  SCREEN: Arrow=screen X/Y, PageUp/PageDown=screen Z, +/-=width")
    print("  SCREEN: 1=27in monitor, 2=65in TV, 3=100in projector, 4=cinema")
    print("  VIEW:   A/D=seat X, Up/Down or Space/LeftShift=seat Y, W/S=seat Z")
    print("  Mouse:  hold right button and drag to rotate VIEW yaw/pitch")
    print("  Both:   Q/E=yaw, T/G=pitch, Z/C=roll")
    print("  P: save profile, R: reload profile, Esc: exit")

    def mouse_button_cb(_window, button, action, _mods):
        nonlocal mouse_look, last_mouse
        if button == glfw.MOUSE_BUTTON_RIGHT:
            mouse_look = action == glfw.PRESS
            last_mouse = glfw.get_cursor_pos(window)

    def cursor_pos_cb(_window, x, y):
        nonlocal last_mouse, view_rot, view_pose
        if not mouse_look:
            last_mouse = (x, y)
            return
        dx = x - last_mouse[0]
        dy = y - last_mouse[1]
        last_mouse = (x, y)
        view_rot_deg = _vec3(view_pose.get("rotation_deg"), [math.degrees(v) for v in view_rot])
        view_rot_deg[0] -= dx * 0.12
        view_rot_deg[1] = max(-89.0, min(89.0, view_rot_deg[1] - dy * 0.12))
        view_pose["rotation_deg"] = [round(float(v), 3) for v in view_rot_deg]
        view_rot = [math.radians(v) for v in view_rot_deg]

    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)

    def key_down(key):
        return glfw.get_key(window, key) in (glfw.PRESS, glfw.REPEAT)

    last_time = glfw.get_time()
    while not glfw.window_should_close(window):
        now = glfw.get_time()
        dt = max(0.001, min(0.05, now - last_time))
        last_time = now
        glfw.poll_events()

        tab_down = glfw.get_key(window, glfw.KEY_TAB) == glfw.PRESS
        if tab_down and not tab_was_down:
            edit_target = "VIEW" if edit_target == "SCREEN" else "SCREEN"
        tab_was_down = tab_down

        pos = _vec3(screen.get("position"), [0.0, 1.2, -2.0])
        rot = _vec3(screen.get("rotation_deg"), [0.0, 0.0, 0.0])
        view_pos = _vec3(view_pose.get("position"), view_pos)
        view_rot_deg = _vec3(view_pose.get("rotation_deg"), [math.degrees(v) for v in view_rot])
        changed_screen = False
        changed_view = False

        step = speed * dt
        rstep = rot_speed * dt

        if edit_target == "SCREEN":
            size_presets = {
                glfw.KEY_1: ("Desk Monitor", 0.62),
                glfw.KEY_2: ("65in TV", 1.44),
                glfw.KEY_3: ("Default Projector", 2.4),
                glfw.KEY_4: ("Cinema Screen", 8.0),
            }
            for preset_key, (preset_name, preset_width) in size_presets.items():
                if key_down(preset_key):
                    screen["name"] = preset_name
                    screen["width"] = preset_width
                    changed_screen = True
            if key_down(glfw.KEY_LEFT):
                pos[0] -= step; changed_screen = True
            if key_down(glfw.KEY_RIGHT):
                pos[0] += step; changed_screen = True
            if key_down(glfw.KEY_UP):
                pos[1] += step; changed_screen = True
            if key_down(glfw.KEY_DOWN):
                pos[1] -= step; changed_screen = True
            if key_down(glfw.KEY_PAGE_UP):
                pos[2] += step; changed_screen = True
            if key_down(glfw.KEY_PAGE_DOWN):
                pos[2] -= step; changed_screen = True
            if key_down(glfw.KEY_EQUAL) or key_down(glfw.KEY_KP_ADD):
                screen["width"] = round(max(0.05, float(screen.get("width", 2.4)) + size_speed * dt), 4)
                changed_screen = True
            if key_down(glfw.KEY_MINUS) or key_down(glfw.KEY_KP_SUBTRACT):
                screen["width"] = round(max(0.05, float(screen.get("width", 2.4)) - size_speed * dt), 4)
                changed_screen = True
            if key_down(glfw.KEY_Q):
                rot[0] += rstep; changed_screen = True
            if key_down(glfw.KEY_E):
                rot[0] -= rstep; changed_screen = True
            if key_down(glfw.KEY_T):
                rot[1] += rstep; changed_screen = True
            if key_down(glfw.KEY_G):
                rot[1] -= rstep; changed_screen = True
            if key_down(glfw.KEY_Z):
                rot[2] += rstep; changed_screen = True
            if key_down(glfw.KEY_C):
                rot[2] -= rstep; changed_screen = True
        else:
            yaw_rad = math.radians(view_rot_deg[0])
            forward = np.array([-math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype="f4")
            right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)], dtype="f4")
            if key_down(glfw.KEY_W):
                view_pos = (np.array(view_pos) + forward * step).tolist(); changed_view = True
            if key_down(glfw.KEY_S):
                view_pos = (np.array(view_pos) - forward * step).tolist(); changed_view = True
            if key_down(glfw.KEY_A):
                view_pos = (np.array(view_pos) - right * step).tolist(); changed_view = True
            if key_down(glfw.KEY_D):
                view_pos = (np.array(view_pos) + right * step).tolist(); changed_view = True
            if key_down(glfw.KEY_SPACE) or key_down(glfw.KEY_UP):
                view_pos[1] += step; changed_view = True
            if key_down(glfw.KEY_LEFT_SHIFT) or key_down(glfw.KEY_RIGHT_SHIFT) or key_down(glfw.KEY_DOWN):
                view_pos[1] -= step; changed_view = True
            if key_down(glfw.KEY_Q):
                view_rot_deg[0] += rstep; changed_view = True
            if key_down(glfw.KEY_E):
                view_rot_deg[0] -= rstep; changed_view = True
            if key_down(glfw.KEY_T):
                view_rot_deg[1] += rstep; changed_view = True
            if key_down(glfw.KEY_G):
                view_rot_deg[1] -= rstep; changed_view = True
            if key_down(glfw.KEY_Z):
                view_rot_deg[2] += rstep; changed_view = True
            if key_down(glfw.KEY_C):
                view_rot_deg[2] -= rstep; changed_view = True

        if changed_screen:
            screen["position"] = [round(v, 4) for v in pos]
            screen["rotation_deg"] = [round(v, 3) for v in rot]
        if changed_view:
            view_pose["position"] = [round(float(v), 4) for v in view_pos]
            view_pose["rotation_deg"] = [round(float(v), 3) for v in view_rot_deg]
            view_rot = [math.radians(v) for v in view_rot_deg]

        if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
            _save_profile(profile_path, profile)
            saved_flash = 1.0
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            _room_dir, _profile_path, profile, _glb_path = _load_profile(args.room)
            view_pose = profile.setdefault("view_pose", {})
            screen = profile.setdefault("screen", {})
            view_pos = _vec3(view_pose.get("position"), [0.0, 1.2, 0.0])
            view_rot = _rot_deg(view_pose.get("rotation_deg", view_pose.get("rotation")), [0.0, 0.0, 0.0])
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        title = (
            f"{args.room} | {edit_target} | {screen.get('name', 'Screen')} | "
            f"view={view_pose.get('position', view_pos)} {view_pose.get('rotation_deg', view_rot_deg)} | "
            f"pos={screen.get('position')} rot={screen.get('rotation_deg')} "
            f"w={float(screen.get('width', 2.4)):.3f}m"
        )
        if saved_flash > 0:
            title += " | SAVED"
            saved_flash -= dt
        glfw.set_window_title(window, title)

        ww, wh = glfw.get_window_size(window)
        ctx.viewport = (0, 0, ww, wh)
        aspect = ww / max(1, wh)
        proj = _projection(aspect)
        view = _view_matrix(view_pos, view_rot)
        vp = proj @ view
        cam_pos = np.array(view_pos, dtype="f4")

        ctx.clear(0.035, 0.04, 0.045, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        env_prog["u_mvp"].write(vp.T.astype("f4").tobytes())
        env_prog["u_model"].write(env_model.T.astype("f4").tobytes())
        env_prog["u_camera_pos"].write(cam_pos.tobytes())
        env_prog["u_ambient_color"].value = (0.24, 0.24, 0.26)
        env_prog["u_light_color"].value = (0.70, 0.70, 0.72)
        for prim in env_prims:
            tid = prim["tex_id"]
            if tid in tex_cache:
                tex_cache[tid].use(location=0)
                env_prog["u_use_texture"].value = 1
            else:
                env_prog["u_use_texture"].value = 0
            bc = prim["base_color"]
            env_prog["u_base_color"].value = (float(bc[0]), float(bc[1]), float(bc[2]))
            env_prog["u_alpha"].value = min(max(float(prim["base_alpha"]), 0.15), 1.0)
            prim["vao"].render(moderngl.TRIANGLES)

        # Render the configured screen as a translucent blue grid.
        sv = _screen_vertices(screen)
        screen_vbo.write(sv.astype("f4").tobytes())
        screen_prog["u_mvp"].write(vp.T.astype("f4").tobytes())
        screen_prog["u_color"].value = (0.1, 0.45, 1.0, 0.72)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.disable(moderngl.CULL_FACE)
        screen_vao.render(moderngl.TRIANGLE_STRIP)
        ctx.disable(moderngl.BLEND)

        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
