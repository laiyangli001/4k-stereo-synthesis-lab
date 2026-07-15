"""Panda3D glTF renderer adapter layer.

This package is intentionally import-light. The default OpenXR renderer path can
import the contracts and selector without importing Panda3D or creating a GL
context.
"""

from .bridge import PandaBridgeUnavailable, SwapchainResourceKey
from .controller_ray import PandaControllerRayGeometryError, PandaControllerRayGeometryTarget
from .diagnostics import PandaRuntimeSnapshot
from .frame_source import PandaFrameSourceError, PandaFrameSourceInput, build_panda_frame_state
from .nv_dx_bridge import PandaNvDxBridge, ViewerPandaNvDxInteropAdapter
from .runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaAnimationClock,
    PandaAnimationPlaybackState,
    PandaControllerRay,
    PandaEyeView,
    PandaFrameState,
    PandaPose,
    PandaRuntimeConfig,
    PandaRuntimeUnavailable,
    PandaSceneRenderer,
    PandaScreenTextureFrame,
    resolve_gltf_renderer_mode,
)
from .scene_bindings import PandaSceneBindingResult, sync_panda_scene_assets_from_viewer
from .screen_node import PandaScreenNodeTarget, create_panda_screen_node_target
from .screen_texture import PandaScreenTextureUploadError, PandaScreenTextureUploadTarget

__all__ = [
    "GLTF_RENDERER_ENV_VAR",
    "PandaAnimationClock",
    "PandaAnimationPlaybackState",
    "PandaBridgeUnavailable",
    "PandaControllerRay",
    "PandaControllerRayGeometryError",
    "PandaControllerRayGeometryTarget",
    "PandaEyeView",
    "PandaFrameSourceError",
    "PandaFrameSourceInput",
    "PandaFrameState",
    "PandaNvDxBridge",
    "PandaPose",
    "PandaRuntimeConfig",
    "PandaRuntimeSnapshot",
    "PandaSceneBindingResult",
    "PandaRuntimeUnavailable",
    "PandaSceneRenderer",
    "PandaScreenNodeTarget",
    "PandaScreenTextureFrame",
    "PandaScreenTextureUploadError",
    "PandaScreenTextureUploadTarget",
    "SwapchainResourceKey",
    "ViewerPandaNvDxInteropAdapter",
    "build_panda_frame_state",
    "create_panda_screen_node_target",
    "resolve_gltf_renderer_mode",
    "sync_panda_scene_assets_from_viewer",
]
