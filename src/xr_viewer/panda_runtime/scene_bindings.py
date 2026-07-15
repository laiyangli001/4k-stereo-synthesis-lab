"""Bind current viewer assets into the optional Panda3D scene facade."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any


@dataclass(frozen=True)
class PandaSceneBindingResult:
    """Result of synchronizing viewer-selected assets into PandaSceneRenderer."""

    loaded: bool
    environment_path: str
    controller_brand: str
    controller_paths: tuple[tuple[str, str], ...]
    screen_size: tuple[float, float] | None = None
    screen_target_bound: bool = False


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
    screen_size = _screen_size(viewer)
    binding_key = (environment_path, controller_brand, controller_paths, screen_size)
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
        for hand, path in controller_paths:
            renderer.load_controller(hand, path)
        screen_target_bound = _sync_screen_target(viewer, renderer, screen_size)
    except Exception as exc:
        viewer._panda_scene_binding_error = f"{type(exc).__name__}: {exc}"
        viewer._panda_scene_binding_failed_key = binding_key
        return None

    result = PandaSceneBindingResult(
        loaded=bool(environment_path or controller_paths or screen_target_bound),
        environment_path=environment_path,
        controller_brand=controller_brand,
        controller_paths=controller_paths,
        screen_size=screen_size,
        screen_target_bound=screen_target_bound,
    )
    viewer._panda_scene_binding_key = binding_key
    viewer._panda_scene_binding_failed_key = None
    viewer._panda_scene_binding_result = result
    viewer._panda_scene_binding_error = ""
    return result


def _screen_size(viewer: Any) -> tuple[float, float] | None:
    try:
        width = float(getattr(viewer, "screen_width", 0.0) or 0.0)
        height = float(getattr(viewer, "screen_height", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    if width <= 0.0 or height <= 0.0:
        return None
    return (width, height)


def _sync_screen_target(viewer: Any, renderer: Any, screen_size: tuple[float, float] | None) -> bool:
    if screen_size is None:
        return False
    scene = getattr(renderer, "scene", None)
    attach_root = getattr(scene, "attach_screen_root", None)
    attach_texture = getattr(scene, "attach_screen_texture_target", None)
    if not callable(attach_root) or not callable(attach_texture):
        return False
    if getattr(viewer, "_panda_screen_node_size", None) == screen_size:
        return getattr(viewer, "_panda_screen_node_target", None) is not None

    from .screen_node import create_panda_screen_node_target

    target = create_panda_screen_node_target(*screen_size)
    attach_root(target.root)
    attach_texture(target.texture_target)
    viewer._panda_screen_node_target = target
    viewer._panda_screen_node_size = screen_size
    return True


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
