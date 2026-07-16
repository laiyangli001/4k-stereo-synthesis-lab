"""Bind current viewer assets into the optional Panda3D scene facade."""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Mapping

from .scene import PandaFillLight


@dataclass(frozen=True)
class PandaSceneBindingResult:
    """Result of synchronizing viewer-selected assets into PandaSceneRenderer."""

    loaded: bool
    environment_path: str
    controller_brand: str
    controller_paths: tuple[tuple[str, str], ...]
    ambient_color: tuple[float, float, float]
    head_light_color: tuple[float, float, float]
    fill_light_count: int



def sync_panda_scene_assets_from_viewer(viewer: Any) -> PandaSceneBindingResult | None:
    """Load the active environment and controller GLBs into the Panda facade once.

    This is intentionally gated by the optional renderer request. It does not call
    render_eyes(), does not touch the D3D11 bridge, and does not replace the native
    renderer path.
    """
    renderer = getattr(viewer, "_panda_scene_renderer", None)
    config = getattr(viewer, "_gltf_renderer_config", None)
    if renderer is None or not bool(getattr(config, "panda3d_requested", False)):
        return None

    environment_path = _existing_path(getattr(viewer, "_env_model_path", None))
    controller_brand = str(
        getattr(viewer, "_current_brand", None)
        or getattr(viewer, "_controller_model", "")
        or ""
    )
    controller_paths = _controller_paths(getattr(viewer, "_controllers_root", None), controller_brand)
    environment_transform = _environment_transform(viewer)
    environment_lighting = _environment_lighting(viewer)
    binding_key = (
        environment_path,
        controller_brand,
        controller_paths,
        environment_transform,
        environment_lighting,
    )
    if getattr(viewer, "_panda_scene_binding_key", None) == binding_key:
        return getattr(viewer, "_panda_scene_binding_result", None)

    if getattr(viewer, "_panda_scene_binding_failed_key", None) == binding_key:
        return getattr(viewer, "_panda_scene_binding_result", None)

    try:
        scene = getattr(renderer, "scene", None)
        if scene is not None and hasattr(scene, "load_panda_assets"):
            scene.load_panda_assets = True
        if environment_path:
            renderer.load_environment(environment_path)
            configure_transform = getattr(renderer, "configure_environment_transform", None)
            if callable(configure_transform):
                configure_transform(*environment_transform)
            configure_lighting = getattr(renderer, "configure_environment_lighting", None)
            if callable(configure_lighting):
                configure_lighting(*environment_lighting)

        for hand, path in controller_paths:
            renderer.load_controller(hand, path)
    except Exception as exc:
        viewer._panda_scene_binding_error = f"{type(exc).__name__}: {exc}"
        viewer._panda_scene_binding_failed_key = binding_key
        return None

    result = PandaSceneBindingResult(
        loaded=bool(environment_path or controller_paths),
        environment_path=environment_path,
        controller_brand=controller_brand,
        controller_paths=controller_paths,
        ambient_color=environment_lighting[0],
        head_light_color=environment_lighting[1],
        fill_light_count=len(environment_lighting[2]),
    )
    viewer._panda_scene_binding_key = binding_key
    viewer._panda_scene_binding_failed_key = None
    viewer._panda_scene_binding_result = result
    viewer._panda_scene_binding_error = ""
    return result


def _existing_path(path: Any) -> str:
    if not path:
        return ""
    value = os.fspath(path)
    return value if os.path.isfile(value) else ""


def _controller_paths(root: Any, brand: str) -> tuple[tuple[str, str], ...]:
    if not root or not brand:
        return ()
    base_dir = os.path.join(os.fspath(root), brand)
    paths: list[tuple[str, str]] = []
    for hand in ("left", "right"):
        path = os.path.join(base_dir, f"{hand}.glb")
        if os.path.isfile(path):
            paths.append((hand, path))
    return tuple(paths)


def _environment_transform(viewer: Any) -> tuple[tuple[float, float, float], ...]:
    return (
        _vec3(getattr(viewer, "_env_model_pos", None), (0.0, 0.0, 0.0)),
        _vec3(getattr(viewer, "_env_model_rot", None), (0.0, 0.0, 0.0)),
        _vec3(getattr(viewer, "_env_model_scale", None), (1.0, 1.0, 1.0)),
    )


def _environment_lighting(
    viewer: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[PandaFillLight, ...]]:
    scale = _panda_light_scale(viewer)
    ambient = _scale_vec3(_vec3(getattr(viewer, "_env_ambient_color", None), (0.0, 0.0, 0.0)), scale)
    head = _scale_vec3(_vec3(getattr(viewer, "_env_head_light_color", None), (0.0, 0.0, 0.0)), scale)
    fills = []
    for value in getattr(viewer, "_env_fill_lights", ()) or ():
        if not isinstance(value, Mapping):
            continue
        position = _vec3(value.get("position"), (0.0, 0.0, 0.0))
        color = _scale_vec3(_vec3(value.get("color"), (0.0, 0.0, 0.0)), scale)
        try:
            light_range = max(0.001, float(value.get("range", 1.0)))
        except (TypeError, ValueError):
            light_range = 1.0
        if not math.isfinite(light_range):
            continue
        if any(max(0.0, component) for component in color):
            fills.append(PandaFillLight(position, color, light_range))
    return ambient, head, tuple(fills)


def _panda_light_scale(viewer: Any) -> float:
    profile = getattr(viewer, "_env_profile", None)
    candidates = [getattr(viewer, "_env_panda_light_scale", None)]
    if isinstance(profile, Mapping):
        candidates.extend(
            (
                profile.get("panda_light_scale"),
                profile.get("panda3d_light_scale"),
                profile.get("env_panda_light_scale"),
            )
        )
    for value in candidates:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(scale) and scale > 0.0:
            return scale
    return 1.0


def _scale_vec3(value: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(float(component) * scale for component in value)


def _vec3(value: Any, default: tuple[float, float, float]) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return default
    return tuple(float(component) for component in value[:3])
