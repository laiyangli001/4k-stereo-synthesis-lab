#!/usr/bin/env python3
"""Preview a room using Panda3D loading and the existing ModernGL renderer.

This diagnostic keeps the shader, camera, render passes, and draw loop identical
to preview_room_layout.py. Only GLB geometry/material/texture loading is routed
through panda3d-gltf so visual differences isolate the loader boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import glfw
import moderngl
import numpy as np


APP_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
ENVIRONMENTS_DIR = APP_DIR / "xr_viewer" / "environments"
PREVIEW_FINE_MOVE_SPEED_MPS = 1.0

PANDA_DIAGNOSTIC_PRESETS = {
    "01-raw-no-lights-linear": {"runtime_state": False, "simplepbr": False, "lights": False, "ibl": False, "srgb": False},
    "02-raw-lights-linear": {"runtime_state": False, "simplepbr": False, "lights": True, "ibl": False, "srgb": False},
    "03-runtime-state-lights-linear": {"runtime_state": True, "simplepbr": False, "lights": True, "ibl": False, "srgb": False},
    "04-raw-simplepbr-lights-linear": {"runtime_state": False, "simplepbr": True, "lights": True, "ibl": False, "srgb": False},
    "05-raw-simplepbr-ibl-linear": {"runtime_state": False, "simplepbr": True, "lights": True, "ibl": True, "srgb": False},
    "06-raw-simplepbr-ibl-srgb": {"runtime_state": False, "simplepbr": True, "lights": True, "ibl": True, "srgb": True},
    "07-runtime-simplepbr-ibl-srgb": {"runtime_state": True, "simplepbr": True, "lights": True, "ibl": True, "srgb": True},
    "08-runtime-simplepbr-lights-srgb": {"runtime_state": True, "simplepbr": True, "lights": True, "ibl": False, "srgb": True},
    "09-runtime-simplepbr-ibl-srgb-no-lights": {"runtime_state": True, "simplepbr": True, "lights": False, "ibl": True, "srgb": True},
    "10-runtime-simplepbr-srgb-no-lighting": {"runtime_state": True, "simplepbr": True, "lights": False, "ibl": False, "srgb": True},
}


def _resolved_import_path(entry):
    try:
        return Path(entry or os.curdir).resolve()
    except (OSError, RuntimeError):
        return None


# Direct script execution adds src/xr_viewer first, which would shadow the
# installed panda3d-gltf package with this project's xr_viewer/gltf package.
sys.path[:] = [str(APP_DIR)] + [
    entry
    for entry in sys.path
    if _resolved_import_path(entry) not in {APP_DIR, SCRIPT_DIR}
]
os.chdir(APP_DIR)
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

from xr_viewer.gl_state import set_depth_mask  # noqa: E402
from xr_viewer.gltf import (  # noqa: E402
    GltfMaterial,
    OPENGL_VERTEX_FORMAT,
    apply_skybox_profile,
    attach_primitive_contract,
    format_gltf_scene_summary,
    render_pass_from_primitive,
    sort_transparent_primitives,
    summarize_gltf_scene,
    validate_mesh_contract,
)
from xr_viewer.panda_runtime.coordinates import panda_geometry_to_gltf  # noqa: E402


ENV_VERT = """
#version 330
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec2 in_uv1;
out vec3 v_normal;
out vec3 v_position;
out vec2 v_uv;
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform int u_base_texcoord;
void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    v_position = world_pos.xyz;
    v_normal = mat3(transpose(inverse(u_model))) * in_normal;
    v_uv = u_base_texcoord == 1 ? in_uv1 : in_uv;
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
uniform int u_alpha_mode;
uniform float u_alpha_cutoff;
uniform float u_exposure;
uniform float u_gamma;

vec3 gltfSrgbToLinear(vec3 c) {
    c = clamp(c, 0.0, 1.0);
    vec3 lo = c / 12.92;
    vec3 hi = pow((c + vec3(0.055)) / 1.055, vec3(2.4));
    return mix(lo, hi, step(vec3(0.04045), c));
}

vec3 gltfToneMap(vec3 linearColor) {
    linearColor = max(linearColor, vec3(0.0));
    return linearColor / (linearColor + vec3(1.0));
}

vec3 gltfLinearToOutput(vec3 linearColor, float gamma) {
    return pow(clamp(gltfToneMap(linearColor), 0.0, 1.0), vec3(1.0 / max(gamma, 0.001)));
}

void main() {
    vec3 base = u_base_color;
    float alpha = u_alpha;
    if (u_use_texture == 1) {
        vec4 texel = texture(u_tex, v_uv);
        base *= gltfSrgbToLinear(texel.rgb);
        if (u_alpha_mode != 0) {
            alpha *= texel.a;
        }
    }
    if (u_alpha_mode == 1 && alpha < u_alpha_cutoff) {
        discard;
    }
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_camera_pos + vec3(0.0, 0.2, 0.0) - v_position);
    float diff = max(abs(dot(N, L)), 0.12);
    vec3 color = base * (u_ambient_color + u_light_color * diff) * u_exposure;
    fragColor = vec4(gltfLinearToOutput(color, u_gamma), alpha);
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


def _active_view_pose(profile: dict) -> dict:
    view_poses = profile.get("view_poses")
    if isinstance(view_poses, list) and view_poses:
        try:
            idx = int(profile.get("view_pose_index", 0)) % len(view_poses)
        except (TypeError, ValueError):
            idx = 0
        if isinstance(view_poses[idx], dict):
            return view_poses[idx]
    view = profile.get("view_pose", profile.get("camera", {}))
    return view if isinstance(view, dict) else {}


def _pose_position(view: dict, default):
    if isinstance(view, dict):
        if "position" in view:
            return _vec3(view.get("position"), default)
        if all(key in view for key in ("x", "y", "z")):
            return _vec3([view.get("x"), view.get("y"), view.get("z")], default)
    return list(default)


def _pose_rotation_deg(view: dict, default=(0.0, 0.0, 0.0)):
    if isinstance(view, dict):
        if "rotation_deg" in view:
            return _vec3(view.get("rotation_deg"), default)
        if "rotation" in view:
            return [math.degrees(v) for v in _rot_deg(view.get("rotation"), default)]
        if "angle" in view:
            return [float(view.get("angle") or 0.0), 0.0, 0.0]
    return list(default)


def _set_pose_position(view: dict, pos):
    rounded = [round(float(v), 4) for v in pos]
    if any(key in view for key in ("x", "y", "z")):
        view["x"], view["y"], view["z"] = rounded
    else:
        view["position"] = rounded


def _set_pose_rotation_deg(view: dict, rot):
    rounded = [round(float(v), 3) for v in rot]
    view["rotation_deg"] = rounded
    if "angle" in view:
        view["angle"] = rounded[0]


def _resolve_room_dir(room: str) -> Path:
    room_dir = ENVIRONMENTS_DIR / room
    if room_dir.exists():
        return room_dir
    room_key = room.strip().lower()
    if ENVIRONMENTS_DIR.exists():
        for candidate in ENVIRONMENTS_DIR.iterdir():
            if candidate.is_dir() and candidate.name.lower() == room_key:
                return candidate
    return room_dir


def _load_profile(room: str):
    room_dir = _resolve_room_dir(room)
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


def _environment_model_matrix(profile):
    model_pos = _vec3(profile.get("model_position"), [0.0, -1.0, -3.0])
    model_rot = _rot_deg(profile.get("model_rotation_deg", profile.get("model_rotation")), [0.0, 0.0, 0.0])
    model_scale = _vec3(profile.get("model_scale"), [1.0, 1.0, 1.0])
    return _mat_from_trs(model_pos, model_rot, model_scale)


def _profile_projection_planes(profile):
    try:
        near = max(0.01, float(profile.get("xr_projection_near", 0.03)))
    except (TypeError, ValueError):
        near = 0.03
    try:
        far = max(near + 1.0, float(profile.get("xr_projection_far", 200.0)))
    except (TypeError, ValueError):
        far = 200.0
    return near, far


def _projection(aspect, fov_deg=80.0, near=0.03, far=200.0):
    try:
        aspect = float(aspect)
    except (TypeError, ValueError):
        aspect = 1.0
    if not math.isfinite(aspect) or aspect <= 0.0:
        aspect = 1.0
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


def _panda_vertex_column(vdata, name, width, default=0.0):
    """Read one Panda vertex column into a packed float32 NumPy array."""
    from panda3d.core import Geom, GeomVertexReader

    rows = int(vdata.get_num_rows())
    fmt = vdata.get_format()
    if not fmt.has_column(name):
        return np.full((rows, width), float(default), dtype="f4")

    column = fmt.get_column(name)
    array_index = int(fmt.get_array_with(name))
    if array_index >= 0 and column.get_numeric_type() == Geom.NT_float32:
        array_format = fmt.get_array(array_index)
        stride = int(array_format.get_stride())
        data = vdata.get_array(array_index).get_handle().get_data()
        return np.ndarray(
            shape=(rows, width),
            dtype=np.float32,
            buffer=data,
            offset=int(column.get_start()),
            strides=(stride, np.dtype(np.float32).itemsize),
        ).copy()

    reader = GeomVertexReader(vdata, name)
    getter = {2: reader.get_data2f, 3: reader.get_data3f, 4: reader.get_data4f}[width]
    result = np.empty((rows, width), dtype="f4")
    for row in range(rows):
        value = getter()
        result[row] = [float(value[index]) for index in range(width)]
    return result


def _panda_matrix(node_path):
    matrix = node_path.get_net_transform().get_mat()
    return np.array(
        [[matrix.get_cell(row, col) for col in range(4)] for row in range(4)],
        dtype="f4",
    )


def _panda_geometry_vertices(node_path, geom):
    vdata = geom.get_vertex_data()
    positions = _panda_vertex_column(vdata, "vertex", 3)
    normals = _panda_vertex_column(vdata, "normal", 3)
    tangents = _panda_vertex_column(vdata, "tangent", 4)
    if not vdata.get_format().has_column("tangent"):
        tangents[:, 3] = 1.0
    uv0 = _panda_vertex_column(vdata, "texcoord.0", 2)
    uv1 = (
        _panda_vertex_column(vdata, "texcoord.1", 2)
        if vdata.get_format().has_column("texcoord.1")
        else uv0.copy()
    )

    positions, normals, tangents, uv0, uv1 = panda_geometry_to_gltf(
        positions,
        normals,
        tangents,
        uv0,
        uv1,
        _panda_matrix(node_path),
    )
    vertices = np.hstack((positions, normals, uv0, uv1)).astype("f4")
    return vertices, tangents


def _panda_primitive_indices(primitive):
    from panda3d.core import Geom

    primitive = primitive.decompose()
    if not primitive.is_indexed():
        first = int(primitive.get_first_vertex())
        return np.arange(first, first + int(primitive.get_num_vertices()), dtype="u4")
    dtype_by_type = {
        Geom.NT_uint8: np.uint8,
        Geom.NT_uint16: np.uint16,
        Geom.NT_uint32: np.uint32,
    }
    dtype = dtype_by_type.get(primitive.get_index_type())
    if dtype is None:
        return np.array(
            [primitive.get_vertex(index) for index in range(primitive.get_num_vertices())],
            dtype="u4",
        )
    data = primitive.get_vertices().get_handle().get_data()
    return np.frombuffer(data, dtype=dtype, count=primitive.get_num_vertices()).astype("u4")


def _panda_material_state(node_path, geom_node, geom_index):
    from panda3d.core import (
        AlphaTestAttrib,
        MaterialAttrib,
        TextureAttrib,
        TransparencyAttrib,
    )

    state = geom_node.get_geom_state(geom_index).compose(node_path.get_net_state())
    material_attrib = state.get_attrib(MaterialAttrib)
    material = material_attrib.get_material() if material_attrib is not None else None
    base_rgba = (1.0, 1.0, 1.0, 1.0)
    if material is not None:
        if material.has_base_color():
            base_rgba = tuple(float(value) for value in material.get_base_color())
        elif material.has_diffuse():
            base_rgba = tuple(float(value) for value in material.get_diffuse())

    alpha_test = state.get_attrib(AlphaTestAttrib)
    transparency = state.get_attrib(TransparencyAttrib)
    if alpha_test is not None:
        alpha_mode = "MASK"
    elif transparency is not None and transparency.get_mode() != TransparencyAttrib.M_none:
        alpha_mode = "BLEND"
    else:
        alpha_mode = "OPAQUE"
    alpha_cutoff = 0.5
    if alpha_test is not None and hasattr(alpha_test, "get_reference_alpha"):
        alpha_cutoff = float(alpha_test.get_reference_alpha())

    texture = None
    base_texcoord = 0
    texture_attrib = state.get_attrib(TextureAttrib)
    if texture_attrib is not None:
        stages = [
            texture_attrib.get_on_stage(index)
            for index in range(texture_attrib.get_num_on_stages())
        ]
        stage = next((item for item in stages if item.get_name() == "Base Color"), None)
        if stage is None and stages:
            stage = stages[0]
        if stage is not None:
            texture = texture_attrib.get_on_texture(stage)
            texcoord_name = str(stage.get_texcoord_name())
            base_texcoord = 1 if texcoord_name.endswith("1") else 0

    return {
        "base_color": np.array(base_rgba[:3], dtype="f4"),
        "base_alpha": float(base_rgba[3]),
        "alpha_mode": alpha_mode,
        "alpha_cutoff": alpha_cutoff,
        "base_texcoord": base_texcoord,
        "texture": texture,
        "material_name": str(material.get_name()) if material is not None else "",
    }


def _panda_texture_rgba(texture):
    width = int(texture.get_x_size())
    height = int(texture.get_y_size())
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Panda texture has invalid dimensions: {texture.get_name()}")
    data = bytes(texture.get_ram_image_as("RGBA"))
    expected = width * height * 4
    if len(data) != expected:
        raise RuntimeError(
            f"Panda texture has no RGBA RAM image: {texture.get_name()} "
            f"expected={expected} actual={len(data)}"
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)[::-1].copy()


def _load_panda3d_gltf_module():
    import importlib

    try:
        module = importlib.import_module("gltf")
    except ImportError as exc:
        raise RuntimeError(f"panda3d-gltf import failed: {exc}") from exc

    module_path = Path(getattr(module, "__file__", "")).resolve()
    local_contract_dir = APP_DIR / "xr_viewer" / "gltf"
    if module_path.is_relative_to(local_contract_dir):
        raise RuntimeError(
            "panda3d-gltf module name collision: "
            f"resolved project package instead of site-packages: {module_path}"
        )
    if not callable(getattr(module, "load_model", None)):
        raise RuntimeError(f"panda3d-gltf has no load_model(): {module_path}")
    return module


def load_glb_model_with_panda3d(path):
    """Return the native preview loader contract using Panda3D as the source."""
    gltf = _load_panda3d_gltf_module()
    try:
        from panda3d.core import NodePath
    except ImportError as exc:
        raise RuntimeError(f"Panda3D import failed: {exc}") from exc

    root = NodePath(gltf.load_model(str(path)))
    primitives = []
    textures = []
    texture_ids = {}
    geom_paths = root.find_all_matches("**/+GeomNode")
    for geom_path in geom_paths:
        geom_node = geom_path.node()
        for geom_index in range(geom_node.get_num_geoms()):
            geom = geom_node.get_geom(geom_index)
            vertices, tangents = _panda_geometry_vertices(geom_path, geom)
            material = _panda_material_state(geom_path, geom_node, geom_index)
            texture = material.pop("texture")
            tex_id = -1
            if texture is not None:
                texture_key = texture.get_name(), int(texture.get_x_size()), int(texture.get_y_size())
                if texture_key not in texture_ids:
                    texture_ids[texture_key] = len(textures)
                    textures.append(_panda_texture_rgba(texture))
                tex_id = texture_ids[texture_key]

            material_name = material.pop("material_name")
            for primitive_index in range(geom.get_num_primitives()):
                indices = _panda_primitive_indices(geom.get_primitive(primitive_index))
                material_contract = GltfMaterial(
                    base_color=tuple(float(value) for value in material["base_color"]),
                    base_alpha=float(material["base_alpha"]),
                    alpha_mode=material["alpha_mode"],
                    alpha_cutoff=float(material["alpha_cutoff"]),
                )
                record = {
                    "vertices": vertices,
                    "tangent": tangents,
                    "indices": indices,
                    "primitive_mode": 4,
                    "node_name": geom_path.get_name(),
                    "mesh_name": material_name or geom.get_vertex_data().get_name(),
                    "tex_id": tex_id,
                    "material_contract": material_contract,
                    **material,
                }
                attach_primitive_contract(record)
                primitives.append(record)

    print(
        f"[PandaLoad] asset={path} nodes={geom_paths.get_num_paths()} "
        f"primitives={len(primitives)} textures={len(textures)}",
        flush=True,
    )
    return primitives, textures, []


def _make_env_resources(ctx, prog, glb_path: Path, profile):
    prims_data, textures, lights = load_glb_model_with_panda3d(str(glb_path))
    apply_skybox_profile(prims_data, profile)
    summary = summarize_gltf_scene(prims_data, textures, lights)
    print("[Preview] " + format_gltf_scene_summary(summary, label=f"Active environment {glb_path}"))
    skybox = profile.get("skybox", {})
    skybox_mipmaps = bool(skybox.get("mipmaps", False)) if isinstance(skybox, dict) else False
    skybox_tex_ids = {
        int(pd.get("tex_id", -1))
        for pd in prims_data
        if pd.get("render_pass") == "sky"
    }
    local_min = None
    local_max = None
    tex_cache = {}
    for tid, arr in enumerate(textures):
        if arr is None:
            continue
        h, w = arr.shape[:2]
        tex = ctx.texture((w, h), 4, arr.tobytes())
        if tid in skybox_tex_ids and not skybox_mipmaps:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.build_mipmaps()
            tex.anisotropy = 8.0
        tex_cache[tid] = tex

    prims = []
    for pd in prims_data:
        validate_mesh_contract(pd["vertices"], pd["tangent"], pd["indices"])
        vertices = pd["vertices"].astype("f4", copy=False)
        if vertices.size:
            pos = vertices[:, :3]
            mn = pos.min(axis=0)
            mx = pos.max(axis=0)
            local_min = mn if local_min is None else np.minimum(local_min, mn)
            local_max = mx if local_max is None else np.maximum(local_max, mx)
        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(pd["indices"].astype("u4").tobytes())
        vao = ctx.vertex_array(
            prog,
            [(vbo, OPENGL_VERTEX_FORMAT, "in_position", "in_normal", "in_uv", "in_uv1")],
            ibo,
        )
        prims.append({
            "vao": vao,
            "tex_id": int(pd.get("tex_id", -1)),
            "base_color": np.array(pd.get("base_color", [1.0, 1.0, 1.0]), dtype="f4"),
            "base_alpha": float(pd.get("base_alpha", 1.0)),
            "alpha_mode": str(pd.get("alpha_mode", "OPAQUE") or "OPAQUE").upper(),
            "alpha_cutoff": float(pd.get("alpha_cutoff", 0.5)),
            "base_texcoord": int(pd.get("base_texcoord", 0) or 0),
            "render_pass": render_pass_from_primitive(pd),
            "sort_center_local": (
                vertices[:, :3].mean(axis=0).astype("f4")
                if vertices.size
                else np.zeros(3, dtype="f4")
            ),
        })
    return prims, tex_cache, local_min, local_max


def _world_bounds_from_local(local_min, local_max, model):
    if local_min is None or local_max is None:
        return None, None
    corners = np.array([
        [x, y, z, 1.0]
        for x in (float(local_min[0]), float(local_max[0]))
        for y in (float(local_min[1]), float(local_max[1]))
        for z in (float(local_min[2]), float(local_max[2]))
    ], dtype="f4")
    world = (model @ corners.T).T[:, :3]
    return world.min(axis=0), world.max(axis=0)


def _preview_motion_speeds(env_world_min, env_world_max):
    base_move_speed = 0.75
    base_size_speed = 0.8
    if env_world_min is None or env_world_max is None:
        return base_move_speed, base_size_speed

    bounds_size = np.asarray(env_world_max, dtype=np.float64) - np.asarray(env_world_min, dtype=np.float64)
    if bounds_size.size == 0:
        return base_move_speed, base_size_speed

    max_extent = float(np.nanmax(np.abs(bounds_size)))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        return base_move_speed, base_size_speed

    scene_scale = max(1.0, min(80.0, max_extent / 50.0))
    return base_move_speed * scene_scale, base_size_speed * scene_scale


def _diagnostic_size(value):
    try:
        width_text, height_text = str(value).lower().split("x", 1)
        width = int(width_text)
        height = int(height_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("diagnostic size must be WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("diagnostic dimensions must be positive")
    return width, height


def _load_panda_diagnostic_root(glb_path, runtime_state):
    from panda3d.core import NodePath

    gltf = _load_panda3d_gltf_module()
    root = NodePath(gltf.load_model(str(glb_path)))
    if runtime_state:
        from xr_viewer.panda_runtime.scene import _apply_gltf_unlit_extension_hints

        _apply_gltf_unlit_extension_hints(str(glb_path), root)
    return root


def _apply_panda_diagnostic_transform(root, profile):
    from xr_viewer.panda_runtime.coordinates import (
        gltf_position_to_panda,
        gltf_rotation_to_panda_hpr_degrees,
        gltf_scale_to_panda,
    )

    position = _vec3(profile.get("model_position"), [0.0, -1.0, -3.0])
    rotation = _rot_deg(
        profile.get("model_rotation_deg", profile.get("model_rotation")),
        [0.0, 0.0, 0.0],
    )
    scale = _vec3(profile.get("model_scale"), [1.0, 1.0, 1.0])
    root.set_pos(*gltf_position_to_panda(position))
    root.set_hpr(*gltf_rotation_to_panda_hpr_degrees(rotation))
    root.set_scale(*gltf_scale_to_panda(scale))


def _install_panda_diagnostic_lights(base, root, profile, camera_position, enabled):
    from panda3d.core import AmbientLight, LVector3, PointLight
    from xr_viewer.panda_runtime.coordinates import gltf_position_to_panda

    ambient_color = _vec3(profile.get("env_ambient_color"), [0.24, 0.24, 0.26])
    head_color = _vec3(profile.get("env_head_light_color"), [0.70, 0.70, 0.72])
    preview_ambient = [max(0.22, component) for component in ambient_color]
    preview_head = [max(0.85, component) for component in head_color]
    base.render.set_shader_input("d2s_preview_ambient_color", *preview_ambient)
    base.render.set_shader_input("d2s_preview_light_color", *preview_head)
    base.render.set_shader_input("d2s_preview_exposure", 2.2)
    base.render.set_shader_input("camera_world_position", *camera_position)
    if not enabled:
        return []

    nodes = []
    if any(ambient_color):
        ambient = AmbientLight("diagnostic-profile-ambient")
        ambient.set_color((*ambient_color, 1.0))
        node = base.render.attach_new_node(ambient)
        base.render.set_light(node)
        nodes.append(node)
    if any(head_color):
        head = PointLight("diagnostic-profile-head")
        head.set_color((*head_color, 1.0))
        head.set_attenuation(LVector3(1.0, 0.0, 0.0))
        node = base.render.attach_new_node(head)
        node.set_pos(*camera_position)
        base.render.set_light(node)
        nodes.append(node)

    for index, spec in enumerate(profile.get("env_fill_lights", ()) or ()):
        if not isinstance(spec, dict):
            continue
        color = _vec3(spec.get("color"), [0.0, 0.0, 0.0])
        position = gltf_position_to_panda(_vec3(spec.get("position"), [0.0, 0.0, 0.0]))
        try:
            light_range = max(float(spec.get("range", 1.0)), 0.001)
        except (TypeError, ValueError):
            light_range = 1.0
        fill = PointLight(f"diagnostic-profile-fill-{index}")
        fill.set_color((*color, 1.0))
        fill.set_attenuation(LVector3(1.0, 0.0, 1.0 / (light_range * light_range)))
        node = root.attach_new_node(fill)
        node.set_pos(*position)
        base.render.set_light(node)
        nodes.append(node)
    return nodes


def _panda_shader_state_summary(root):
    from panda3d.core import ShaderAttrib

    result = {
        "geom_count": 0,
        "geom_state_count": 0,
        "shader_attrib_count": 0,
        "explicit_shader_count": 0,
        "auto_shader_count": 0,
        "priorities": {},
        "attribs": {},
    }
    priorities = {}
    attribs = {}
    for geom_path in root.find_all_matches("**/+GeomNode"):
        result["geom_count"] += 1
        node = geom_path.node()
        net_state = geom_path.get_net_state()
        for geom_index in range(node.get_num_geoms()):
            result["geom_state_count"] += 1
            state = node.get_geom_state(geom_index).compose(net_state)
            attrib = state.get_attrib(ShaderAttrib)
            if attrib is None:
                continue
            result["shader_attrib_count"] += 1
            if attrib.has_shader():
                result["explicit_shader_count"] += 1
            else:
                result["auto_shader_count"] += 1
            priority = str(int(attrib.get_shader_priority()))
            priorities[priority] = priorities.get(priority, 0) + 1
            text = " ".join(str(attrib).split())
            if len(text) > 240:
                text = text[:237] + "..."
            attribs[text] = attribs.get(text, 0) + 1
    result["priorities"] = priorities
    result["attribs"] = attribs
    return result


def _panda_material_name_summary(root):
    from panda3d.core import MaterialAttrib

    names = {}
    for geom_path in root.find_all_matches("**/+GeomNode"):
        node = geom_path.node()
        for geom_index in range(node.get_num_geoms()):
            state = node.get_geom_state(geom_index).compose(geom_path.get_net_state())
            material_attrib = state.get_attrib(MaterialAttrib)
            material_name = (
                material_attrib.get_material().get_name()
                if material_attrib is not None
                else "<none>"
            )
            names[material_name] = names.get(material_name, 0) + 1
    return dict(sorted(names.items()))


def _diagnostic_neutral_ibl_env_map(simplepbr_module, intensity):
    from panda3d.core import LColor, Texture

    strength = min(1.0, max(0.0, float(intensity)))
    size = 16
    texture = Texture("d2s-diagnostic-neutral-ibl-env")
    texture.setup_cube_map(size, Texture.T_unsigned_byte, Texture.F_rgb)
    pages = []
    for face_index in range(6):
        if face_index == 4:
            color = 178
        elif face_index == 5:
            color = 84
        else:
            color = 128
        value = round(color * strength)
        pages.append(bytes((value, value, value)) * size * size)
    texture.set_ram_image(b"".join(pages))
    texture.set_clear_color(LColor(0.5 * strength, 0.5 * strength, 0.5 * strength, 1.0))
    return simplepbr_module.EnvMap(
        texture,
        prefiltered_size=size,
        prefiltered_samples=16,
        blocking_prepare=True,
    )


def _panda_image_stats(path):
    try:
        from PIL import Image
    except ImportError:
        return {}
    pixels = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    luminance = (
        pixels[..., 0] * 0.2126
        + pixels[..., 1] * 0.7152
        + pixels[..., 2] * 0.0722
    )
    return {
        "rgb_mean": [round(float(value), 6) for value in pixels.mean(axis=(0, 1))],
        "luminance_mean": round(float(luminance.mean()), 6),
        "luminance_p05": round(float(np.percentile(luminance, 5)), 6),
        "luminance_p50": round(float(np.percentile(luminance, 50)), 6),
        "luminance_p95": round(float(np.percentile(luminance, 95)), 6),
    }


def _gltf_material_summary(glb_path):
    from pygltflib import GLTF2

    document = GLTF2().load(str(glb_path))
    materials = tuple(document.materials or ())
    alpha_modes = {"OPAQUE": 0, "MASK": 0, "BLEND": 0}
    for material in materials:
        mode = str(material.alphaMode or "OPAQUE").upper()
        alpha_modes[mode] = alpha_modes.get(mode, 0) + 1
    return {
        "material_count": len(materials),
        "unlit_count": sum(
            "KHR_materials_unlit" in (material.extensions or {})
            for material in materials
        ),
        "base_color_texture_count": sum(
            getattr(material.pbrMetallicRoughness, "baseColorTexture", None)
            is not None
            for material in materials
        ),
        "metallic_roughness_texture_count": sum(
            getattr(material.pbrMetallicRoughness, "metallicRoughnessTexture", None)
            is not None
            for material in materials
        ),
        "normal_texture_count": sum(
            material.normalTexture is not None for material in materials
        ),
        "emissive_texture_count": sum(
            material.emissiveTexture is not None for material in materials
        ),
        "alpha_modes": alpha_modes,
    }


def _render_panda_diagnostic(profile, glb_path, preset_name, output_path, size):
    config = PANDA_DIAGNOSTIC_PRESETS[preset_name]
    width, height = size
    srgb_text = "true" if config["srgb"] else "false"

    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import Filename, PNMImage, load_prc_file_data

    load_prc_file_data(
        f"d2s-panda-diagnostic-{preset_name}",
        "\n".join(
            [
                "window-type offscreen",
                "load-display pandagl",
                f"win-size {width} {height}",
                f"framebuffer-srgb {srgb_text}",
                "audio-library-name null",
                "sync-video false",
                "show-frame-rate-meter false",
                "notify-level-display error",
            ]
        ),
    )
    base = ShowBase(windowType="offscreen")
    pipeline = None
    try:
        if not base.win or not base.win.get_gsg():
            raise RuntimeError("Panda3D diagnostic offscreen window has no GSG")

        view_pose = _active_view_pose(profile)
        view_position = _pose_position(view_pose, [0.0, 1.2, 0.0])
        view_rotation_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
        view_rotation = [math.radians(value) for value in view_rotation_deg]
        from xr_viewer.panda_runtime.coordinates import (
            gltf_position_to_panda,
            gltf_rotation_to_panda_hpr_degrees,
        )

        camera_position = gltf_position_to_panda(view_position)
        base.camera.set_pos(*camera_position)
        base.camera.set_hpr(*gltf_rotation_to_panda_hpr_degrees(view_rotation))
        near, far = _profile_projection_planes(profile)
        vertical_fov = 80.0
        horizontal_fov = math.degrees(
            2.0 * math.atan(math.tan(math.radians(vertical_fov) * 0.5) * width / height)
        )
        base.camLens.set_near_far(near, far)
        base.camLens.set_fov(horizontal_fov, vertical_fov)
        base.set_background_color(1.0, 1.0, 1.0, 1.0)

        if config["simplepbr"]:
            import simplepbr

            env_map = (
                _diagnostic_neutral_ibl_env_map(simplepbr, 1.0)
                if config["ibl"]
                else None
            )
            pipeline = simplepbr.init(
                render_node=base.render,
                window=base.win,
                camera_node=base.cam,
                taskmgr=base.task_mgr,
                msaa_samples=0,
                use_normal_maps=False,
                use_emission_maps=True,
                use_occlusion_maps=False,
                exposure=0.0,
                enable_shadows=False,
                enable_fog=False,
                env_map=env_map,
            )
            base._d2s_simplepbr_enabled = True
        elif config["lights"]:
            base.render.set_shader_auto()

        root = _load_panda_diagnostic_root(glb_path, config["runtime_state"])
        root.reparent_to(base.render)
        _apply_panda_diagnostic_transform(root, profile)
        _install_panda_diagnostic_lights(
            base,
            root,
            profile,
            camera_position,
            config["lights"],
        )
        state_before = _panda_shader_state_summary(root)
        materials_before = _panda_material_name_summary(root)

        for _ in range(4):
            if pipeline is not None:
                base.task_mgr.step()
            else:
                base.graphicsEngine.render_frame()
        state_after = _panda_shader_state_summary(root)
        materials_after = _panda_material_name_summary(root)

        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot = PNMImage()
        if not base.win.get_screenshot(screenshot):
            raise RuntimeError("Panda3D diagnostic screenshot capture failed")
        if not screenshot.write(Filename.from_os_specific(str(output_path))):
            raise RuntimeError(f"Panda3D diagnostic screenshot write failed: {output_path}")

        fb_props = base.win.get_fb_properties()
        actual_srgb = bool(fb_props.get_srgb_color())
        report = {
            "preset": preset_name,
            "config": dict(config),
            "asset": str(glb_path),
            "output": str(output_path),
            "requested_size": [width, height],
            "actual_size": [int(base.win.get_x_size()), int(base.win.get_y_size())],
            "requested_srgb": bool(config["srgb"]),
            "actual_framebuffer_srgb": actual_srgb,
            "materials": _gltf_material_summary(glb_path),
            "resolved_materials_before_render": materials_before,
            "resolved_materials_after_render": materials_after,
            "state_before_render": state_before,
            "state_after_render": state_after,
            "image": _panda_image_stats(output_path),
        }
        report_path = output_path.with_suffix(".json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[PandaDiagnostic] preset={preset_name} output={output_path} "
            f"requested_srgb={config['srgb']} actual_srgb={actual_srgb} "
            f"shader_explicit={state_after['explicit_shader_count']} "
            f"shader_auto={state_after['auto_shader_count']} "
            f"luminance={report['image'].get('luminance_mean', 'n/a')}",
            flush=True,
        )
        return report
    finally:
        base.destroy()


def _run_panda_diagnostic_suite(room, output_dir, size):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for preset_name in PANDA_DIAGNOSTIC_PRESETS:
        output_path = output_dir / f"{preset_name}.png"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            room,
            "--panda-diagnostic",
            preset_name,
            "--diagnostic-output",
            str(output_path),
            "--diagnostic-size",
            f"{size[0]}x{size[1]}",
        ]
        print(f"[PandaDiagnostic] launching {preset_name}", flush=True)
        subprocess.run(command, cwd=str(APP_DIR), check=True)
        reports.append(
            json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
        )
    suite_report = {
        "room": room,
        "output_dir": str(output_dir),
        "presets": reports,
    }
    report_path = output_dir / "suite.json"
    report_path.write_text(
        json.dumps(suite_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[PandaDiagnostic] suite complete: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("room", nargs="?", default="bedroom")
    parser.add_argument("--exposure", type=float, default=None, help="Preview-only brightness multiplier")
    parser.add_argument("--gamma", type=float, default=None, help="Preview-only output gamma")
    parser.add_argument("--center-view", action="store_true", help="Start camera at the transformed model bounds center")
    parser.add_argument("--panda-diagnostic", choices=tuple(PANDA_DIAGNOSTIC_PRESETS))
    parser.add_argument("--diagnostic-output", type=Path)
    parser.add_argument(
        "--diagnostic-suite",
        nargs="?",
        const="logs/panda_render_diagnostics",
        help="Render every Panda diagnostic preset into the optional output directory",
    )
    parser.add_argument("--diagnostic-size", type=_diagnostic_size, default=(1280, 720))
    args = parser.parse_args()

    os.chdir(APP_DIR)
    room_dir, profile_path, profile, glb_path = _load_profile(args.room)
    if args.diagnostic_suite:
        _run_panda_diagnostic_suite(args.room, args.diagnostic_suite, args.diagnostic_size)
        return
    if args.panda_diagnostic:
        output_path = args.diagnostic_output
        if output_path is None:
            output_path = Path("logs/panda_render_diagnostics") / f"{args.panda_diagnostic}.png"
        _render_panda_diagnostic(
            profile,
            glb_path,
            args.panda_diagnostic,
            output_path,
            args.diagnostic_size,
        )
        return
    projection_near, projection_far = _profile_projection_planes(profile)
    view_pose = _active_view_pose(profile)
    if not view_pose:
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

    env_prims, tex_cache, env_local_min, env_local_max = _make_env_resources(ctx, env_prog, glb_path, profile)
    screen_vbo = ctx.buffer(reserve=4 * 5 * 4)
    screen_vao = ctx.vertex_array(screen_prog, [(screen_vbo, "3f 2f", "in_position", "in_uv")])

    env_model = _environment_model_matrix(profile)

    view_pos = _pose_position(view_pose, [0.0, 1.2, 0.0])
    env_world_min, env_world_max = _world_bounds_from_local(env_local_min, env_local_max, env_model)
    if args.center_view and env_world_min is not None and env_world_max is not None:
        view_pos = ((env_world_min + env_world_max) * 0.5).astype(float).tolist()
        _set_pose_position(view_pose, view_pos)
    view_rot_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
    view_rot = [math.radians(v) for v in view_rot_deg]
    preview_exposure = float(args.exposure if args.exposure is not None else profile.get("preview_exposure", 2.2))
    preview_gamma = float(args.gamma if args.gamma is not None else profile.get("preview_gamma", 2.2))
    speed, size_speed = _preview_motion_speeds(env_world_min, env_world_max)
    rot_speed = 45.0
    saved_flash = 0.0
    edit_target = "SCREEN"
    tab_was_down = False
    mouse_look = False
    last_mouse = (0.0, 0.0)

    print(f"Room: {args.room}")
    print(f"Profile: {profile_path}")
    print(f"Preview lighting: exposure={preview_exposure:.2f} gamma={preview_gamma:.2f}")
    print(f"Preview projection: clip={projection_near:.3f}/{projection_far:.1f}")
    print(f"Preview navigation: move_speed={speed:.2f}m/s size_speed={size_speed:.2f}m/s")
    print(f"Preview fine mode: hold Ctrl for {PREVIEW_FINE_MOVE_SPEED_MPS:.2f}m/s movement/size adjustment")
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
        view_rot_deg = _pose_rotation_deg(view_pose, [math.degrees(v) for v in view_rot])
        view_rot_deg[0] -= dx * 0.12
        view_rot_deg[1] = max(-89.0, min(89.0, view_rot_deg[1] - dy * 0.12))
        _set_pose_rotation_deg(view_pose, view_rot_deg)
        view_rot = [math.radians(v) for v in view_rot_deg]

    glfw.set_mouse_button_callback(window, mouse_button_cb)
    glfw.set_cursor_pos_callback(window, cursor_pos_cb)

    def key_down(key):
        return glfw.get_key(window, key) in (glfw.PRESS, glfw.REPEAT)

    def ctrl_down():
        return key_down(glfw.KEY_LEFT_CONTROL) or key_down(glfw.KEY_RIGHT_CONTROL)

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
        view_pos = _pose_position(view_pose, view_pos)
        view_rot_deg = _pose_rotation_deg(view_pose, [math.degrees(v) for v in view_rot])
        changed_screen = False
        changed_view = False

        fine_mode = ctrl_down()
        active_move_speed = PREVIEW_FINE_MOVE_SPEED_MPS if fine_mode else speed
        active_size_speed = PREVIEW_FINE_MOVE_SPEED_MPS if fine_mode else size_speed
        step = active_move_speed * dt
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
                screen["width"] = round(max(0.05, float(screen.get("width", 2.4)) + active_size_speed * dt), 4)
                changed_screen = True
            if key_down(glfw.KEY_MINUS) or key_down(glfw.KEY_KP_SUBTRACT):
                screen["width"] = round(max(0.05, float(screen.get("width", 2.4)) - active_size_speed * dt), 4)
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
            _set_pose_position(view_pose, view_pos)
            _set_pose_rotation_deg(view_pose, view_rot_deg)
            view_rot = [math.radians(v) for v in view_rot_deg]

        if glfw.get_key(window, glfw.KEY_P) == glfw.PRESS:
            _save_profile(profile_path, profile)
            saved_flash = 1.0
        if glfw.get_key(window, glfw.KEY_R) == glfw.PRESS:
            _room_dir, _profile_path, profile, _glb_path = _load_profile(args.room)
            projection_near, projection_far = _profile_projection_planes(profile)
            view_pose = _active_view_pose(profile)
            if not view_pose:
                view_pose = profile.setdefault("view_pose", {})
            screen = profile.setdefault("screen", {})
            view_pos = _pose_position(view_pose, [0.0, 1.2, 0.0])
            view_rot_deg = _pose_rotation_deg(view_pose, [0.0, 0.0, 0.0])
            view_rot = [math.radians(v) for v in view_rot_deg]
            env_model = _environment_model_matrix(profile)
            env_world_min, env_world_max = _world_bounds_from_local(env_local_min, env_local_max, env_model)
            speed, size_speed = _preview_motion_speeds(env_world_min, env_world_max)
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)

        title = (
            f"{args.room} | {edit_target} | {screen.get('name', 'Screen')} | "
            f"view={view_pos} {view_rot_deg} | "
            f"pos={screen.get('position')} rot={screen.get('rotation_deg')} "
            f"w={float(screen.get('width', 2.4)):.3f}m"
        )
        if saved_flash > 0:
            title += " | SAVED"
            saved_flash -= dt
        glfw.set_window_title(window, title)

        ww, wh = glfw.get_window_size(window)
        if ww <= 0 or wh <= 0:
            glfw.poll_events()
            continue
        ctx.viewport = (0, 0, ww, wh)
        aspect = ww / wh
        proj = _projection(aspect, near=projection_near, far=projection_far)
        view = _view_matrix(view_pos, view_rot)
        vp = proj @ view
        cam_pos = np.array(view_pos, dtype="f4")

        ctx.clear(1.0, 1.0, 1.0, 1.0)
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.disable(moderngl.BLEND)

        env_prog["u_mvp"].write(vp.T.astype("f4").tobytes())
        env_prog["u_model"].write(env_model.T.astype("f4").tobytes())
        env_prog["u_camera_pos"].write(cam_pos.tobytes())
        ambient = np.maximum(np.array(_vec3(profile.get("env_ambient_color"), [0.24, 0.24, 0.26]), dtype="f4"), 0.22)
        light = np.maximum(np.array(_vec3(profile.get("env_head_light_color"), [0.70, 0.70, 0.72]), dtype="f4"), 0.85)
        env_prog["u_ambient_color"].value = (float(ambient[0]), float(ambient[1]), float(ambient[2]))
        env_prog["u_light_color"].value = (float(light[0]), float(light[1]), float(light[2]))
        env_prog["u_exposure"].value = max(0.05, preview_exposure)
        env_prog["u_gamma"].value = max(0.1, preview_gamma)
        def draw_env_prim(prim):
            tid = prim["tex_id"]
            if tid in tex_cache:
                tex_cache[tid].use(location=0)
                env_prog["u_use_texture"].value = 1
            else:
                env_prog["u_use_texture"].value = 0
            bc = prim["base_color"]
            alpha_mode = "OPAQUE" if prim.get("render_pass") == "sky" else prim.get("alpha_mode", "OPAQUE")
            alpha_mode_id = 1 if alpha_mode == "MASK" else (2 if alpha_mode == "BLEND" else 0)
            env_prog["u_base_texcoord"].value = 1 if int(prim.get("base_texcoord", 0) or 0) == 1 else 0
            env_prog["u_base_color"].value = (float(bc[0]), float(bc[1]), float(bc[2]))
            env_prog["u_alpha"].value = min(max(float(prim["base_alpha"]), 0.0), 1.0)
            env_prog["u_alpha_mode"].value = alpha_mode_id
            env_prog["u_alpha_cutoff"].value = float(prim.get("alpha_cutoff", 0.5))
            prim["vao"].render(moderngl.TRIANGLES)

        sky_prims = [prim for prim in env_prims if prim.get("render_pass") == "sky"]
        solid_prims = [
            prim for prim in env_prims
            if prim.get("render_pass") in ("opaque", "mask")
        ]
        transparent_prims = [
            prim for prim in env_prims if prim.get("render_pass") == "transparent"
        ]
        if sky_prims:
            ctx.disable(moderngl.CULL_FACE)
            set_depth_mask(False)
            for prim in sky_prims:
                draw_env_prim(prim)
            set_depth_mask(True)
        for prim in solid_prims:
            draw_env_prim(prim)
        if transparent_prims:
            transparent_prims = sort_transparent_primitives(
                transparent_prims,
                cam_pos,
                env_model,
            )
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            set_depth_mask(False)
            for prim in transparent_prims:
                draw_env_prim(prim)
            set_depth_mask(True)
            ctx.disable(moderngl.BLEND)

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
