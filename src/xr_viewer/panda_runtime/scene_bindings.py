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
    binding_key = (environment_path, controller_brand, controller_paths)
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
    except Exception as exc:
        viewer._panda_scene_binding_error = f"{type(exc).__name__}: {exc}"
        viewer._panda_scene_binding_failed_key = binding_key
        return None

    result = PandaSceneBindingResult(
        loaded=bool(environment_path or controller_paths),
        environment_path=environment_path,
        controller_brand=controller_brand,
        controller_paths=controller_paths,
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
