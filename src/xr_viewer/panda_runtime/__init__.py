"""Panda3D glTF renderer adapter layer.

This package is intentionally import-light. The default OpenXR renderer path can
import the contracts and selector without importing Panda3D or creating a GL
context.
"""

from .bridge import PandaBridgeUnavailable, SwapchainResourceKey
from .diagnostics import PandaRuntimeSnapshot
from .frame_source import PandaFrameSourceError, PandaFrameSourceInput, build_panda_frame_state
from .nv_dx_bridge import PandaNvDxBridge, ViewerPandaNvDxInteropAdapter
from .runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaAnimationClock,
    PandaAnimationPlaybackState,
    PandaEyeView,
    PandaFrameState,
    PandaPose,
    PandaRuntimeConfig,
    PandaRuntimeUnavailable,
    PandaSceneRenderer,
    resolve_gltf_renderer_mode,
)
from .scene_bindings import PandaSceneBindingResult, sync_panda_scene_assets_from_viewer

__all__ = [
    "GLTF_RENDERER_ENV_VAR",
    "PandaAnimationClock",
    "PandaAnimationPlaybackState",
    "PandaBridgeUnavailable",
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
    "SwapchainResourceKey",
    "ViewerPandaNvDxInteropAdapter",
    "build_panda_frame_state",
    "resolve_gltf_renderer_mode",
    "sync_panda_scene_assets_from_viewer",
]
