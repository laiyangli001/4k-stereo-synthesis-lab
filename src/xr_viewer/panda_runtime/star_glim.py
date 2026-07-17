"""Optional, asset-declared StarGlim shader support for Panda environments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


_SIDECAR_FILE_NAME = "star_glim.json"


@dataclass(frozen=True)
class PandaStarGlimSpec:
    """Validated StarGlim inputs that live beside an environment GLB."""

    node_name_patterns: tuple[str, ...]
    stars_texture_path: Path
    mask_texture_path: Path
    intensity: float
    speed: float
    shine_speed: float
    cell_density: float
    cell_offset: float
    cell_soft: float
    cell_value: float
    strength: float


def load_star_glim_spec(asset_path: str | Path) -> PandaStarGlimSpec | None:
    """Load an optional StarGlim sidecar without changing static-sky fallback."""

    asset = Path(asset_path)
    sidecar_path = asset.parent / _SIDECAR_FILE_NAME
    try:
        document = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, dict):
        return None
    if document.get("schema_version") != 1 or document.get("effect") != "star_glim":
        return None

    patterns_value = document.get("node_name_patterns")
    if not isinstance(patterns_value, list):
        return None
    node_name_patterns = tuple(
        value.strip() for value in patterns_value if isinstance(value, str) and value.strip()
    )
    stars_texture_path = _sidecar_texture_path(asset.parent, document.get("stars_texture"))
    mask_texture_path = _sidecar_texture_path(asset.parent, document.get("mask_texture"))
    if not node_name_patterns or stars_texture_path is None or mask_texture_path is None:
        return None

    values = _finite_float_values(
        document,
        "intensity",
        "speed",
        "shine_speed",
        "cell_density",
        "cell_offset",
        "cell_soft",
        "cell_value",
        "strength",
    )
    if values is None:
        return None
    return PandaStarGlimSpec(
        node_name_patterns=node_name_patterns,
        stars_texture_path=stars_texture_path,
        mask_texture_path=mask_texture_path,
        intensity=values[0],
        speed=values[1],
        shine_speed=values[2],
        cell_density=values[3],
        cell_offset=values[4],
        cell_soft=values[5],
        cell_value=values[6],
        strength=values[7],
    )


def apply_star_glim_sidecar(asset_path: str | Path, root: Any, *, base_color_texture: Any) -> tuple[Any, ...]:
    """Bind GPU-only StarGlim rendering to the sidecar-selected sky geometry."""

    spec = load_star_glim_spec(asset_path)
    if spec is None or root is None:
        return ()

    from panda3d.core import TexturePool

    stars_texture = TexturePool.load_texture(str(spec.stars_texture_path))
    mask_texture = TexturePool.load_texture(str(spec.mask_texture_path))
    if stars_texture is None or mask_texture is None:
        return ()

    shader = _star_glim_shader()
    targets: list[Any] = []
    seen: set[str] = set()
    for pattern in spec.node_name_patterns:
        for node_path in root.find_all_matches(f"**/{pattern}"):
            for geom_path in node_path.find_all_matches("**/+GeomNode"):
                key = str(geom_path)
                if key in seen:
                    continue
                background_texture = base_color_texture(geom_path)
                if background_texture is None:
                    continue
                seen.add(key)
                geom_path.set_shader(shader, 30)
                geom_path.set_shader_input("d2s_star_glim_background", background_texture)
                geom_path.set_shader_input("d2s_star_glim_stars", stars_texture)
                geom_path.set_shader_input("d2s_star_glim_mask", mask_texture)
                geom_path.set_shader_input("d2s_star_glim_time", 0.0)
                geom_path.set_shader_input("d2s_star_glim_intensity", spec.intensity)
                geom_path.set_shader_input("d2s_star_glim_speed", spec.speed)
                geom_path.set_shader_input("d2s_star_glim_shine_speed", spec.shine_speed)
                geom_path.set_shader_input("d2s_star_glim_cell_density", spec.cell_density)
                geom_path.set_shader_input("d2s_star_glim_cell_offset", spec.cell_offset)
                geom_path.set_shader_input("d2s_star_glim_cell_soft", spec.cell_soft)
                geom_path.set_shader_input("d2s_star_glim_cell_value", spec.cell_value)
                geom_path.set_shader_input("d2s_star_glim_strength", spec.strength)
                targets.append(geom_path)
    return tuple(targets)


def set_star_glim_time(targets: tuple[Any, ...], time_seconds: float) -> None:
    """Advance the effect with the existing XR-derived animation clock."""

    for target in targets:
        target.set_shader_input("d2s_star_glim_time", float(time_seconds))


def _sidecar_texture_path(directory: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    relative = Path(value)
    if relative.is_absolute():
        return None
    candidate = (directory / relative).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _finite_float_values(document: dict[str, Any], *names: str) -> tuple[float, ...] | None:
    values: list[float] = []
    for name in names:
        value = document.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        number = float(value)
        if not math.isfinite(number):
            return None
        values.append(number)
    return tuple(values)


_STAR_GLIM_SHADER: Any | None = None


def _star_glim_shader() -> Any:
    global _STAR_GLIM_SHADER
    if _STAR_GLIM_SHADER is not None:
        return _STAR_GLIM_SHADER
    from panda3d.core import Shader

    vertex = """
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
attribute vec4 p3d_Vertex;
attribute vec2 p3d_MultiTexCoord0;
varying vec2 v_texcoord;
void main() {
    v_texcoord = p3d_MultiTexCoord0;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""
    fragment = """
#version 120
uniform sampler2D d2s_star_glim_background;
uniform sampler2D d2s_star_glim_stars;
uniform sampler2D d2s_star_glim_mask;
uniform float d2s_star_glim_time;
uniform float d2s_star_glim_intensity;
uniform float d2s_star_glim_speed;
uniform float d2s_star_glim_shine_speed;
uniform float d2s_star_glim_cell_density;
uniform float d2s_star_glim_cell_offset;
uniform float d2s_star_glim_cell_soft;
uniform float d2s_star_glim_cell_value;
uniform float d2s_star_glim_strength;
varying vec2 v_texcoord;

float hash21(vec2 value) {
    return fract(sin(dot(value, vec2(127.1, 311.7))) * 43758.5453123);
}

void main() {
    vec2 drift = vec2(d2s_star_glim_time * d2s_star_glim_speed, 0.0);
    vec3 background = texture2D(d2s_star_glim_background, v_texcoord).rgb;
    vec3 stars = texture2D(d2s_star_glim_stars, v_texcoord + drift).rgb;
    float mask = texture2D(d2s_star_glim_mask, v_texcoord + drift).r;
    float density = max(d2s_star_glim_cell_density, 1.0);
    vec2 cell = floor((v_texcoord + drift) * density + d2s_star_glim_cell_offset);
    float phase = hash21(cell) * 6.2831853;
    float pulse = 0.5 + 0.5 * sin(d2s_star_glim_time * d2s_star_glim_shine_speed + phase);
    float threshold = clamp(d2s_star_glim_cell_value + (1.0 - d2s_star_glim_cell_soft), 0.0, 1.0);
    float twinkle = smoothstep(threshold, 1.0, pulse) * d2s_star_glim_strength;
    gl_FragColor = vec4(background + stars * mask * d2s_star_glim_intensity * twinkle, 1.0);
}
"""
    _STAR_GLIM_SHADER = Shader.make(Shader.SL_GLSL, vertex, fragment)
    return _STAR_GLIM_SHADER
