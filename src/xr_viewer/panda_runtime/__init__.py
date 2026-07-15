"""Panda3D glTF renderer adapter layer.

This package is intentionally import-light. The default OpenXR renderer path can
import the contracts and selector without importing Panda3D or creating a GL
context.
"""

from .bridge import PandaBridgeUnavailable, SwapchainResourceKey
from .runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaFrameState,
    PandaRuntimeConfig,
    PandaRuntimeUnavailable,
    PandaSceneRenderer,
    resolve_gltf_renderer_mode,
)

__all__ = [
    "GLTF_RENDERER_ENV_VAR",
    "PandaBridgeUnavailable",
    "PandaFrameState",
    "PandaRuntimeConfig",
    "PandaRuntimeUnavailable",
    "PandaSceneRenderer",
    "SwapchainResourceKey",
    "resolve_gltf_renderer_mode",
]
