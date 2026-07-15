"""Panda3D glTF renderer adapter layer.

This package is intentionally import-light. The default OpenXR renderer path can
import the contracts and selector without importing Panda3D or creating a GL
context.
"""

from .bridge import PandaBridgeUnavailable, SwapchainResourceKey
from .controller_ray import PandaControllerRayGeometryError, PandaControllerRayGeometryTarget
from .diagnostics import PandaRuntimeSnapshot
from .frame_source import PandaFrameSourceError, PandaFrameSourceInput, build_panda_frame_state
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
    "PandaFrameSourceError",
    "PandaFrameSourceInput",
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
    "build_panda_frame_state",
    "resolve_gltf_renderer_mode",
]
