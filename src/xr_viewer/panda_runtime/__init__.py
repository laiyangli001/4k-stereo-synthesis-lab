"""Panda3D glTF renderer adapter layer.

This package is intentionally import-light. The default OpenXR renderer path can
import the contracts and selector without importing Panda3D or creating a GL
context.
"""

from .bridge import PandaBridgeUnavailable, SwapchainResourceKey
from .controller_ray import PandaControllerRayGeometryError, PandaControllerRayGeometryTarget
from .diagnostics import PandaRuntimeSnapshot
from .runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaAnimationClock,
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
from .screen_texture import PandaScreenTextureUploadError, PandaScreenTextureUploadTarget

__all__ = [
    "GLTF_RENDERER_ENV_VAR",
    "PandaAnimationClock",
    "PandaBridgeUnavailable",
    "PandaControllerRay",
    "PandaControllerRayGeometryError",
    "PandaControllerRayGeometryTarget",
    "PandaEyeView",
    "PandaFrameState",
    "PandaPose",
    "PandaRuntimeConfig",
    "PandaRuntimeSnapshot",
    "PandaRuntimeUnavailable",
    "PandaSceneRenderer",
    "PandaScreenTextureFrame",
    "PandaScreenTextureUploadError",
    "PandaScreenTextureUploadTarget",
    "SwapchainResourceKey",
    "resolve_gltf_renderer_mode",
]
