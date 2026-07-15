"""Renderer facade and selector for the optional Panda3D glTF path."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import Any, Mapping, Protocol

from .bridge import PandaBridge, RenderEyesResult, SwapchainImageRef
from .diagnostics import PandaRuntimeDiagnostics
from .scene import PandaSceneGraph
from .stereo_targets import StereoTargetSpec, StereoTargets


GLTF_RENDERER_ENV_VAR = "D2S_GLTF_RENDERER"
_NATIVE_RENDERER = "native"
_PANDA3D_RENDERER = "panda3d"
_VALID_RENDERERS = frozenset({_NATIVE_RENDERER, _PANDA3D_RENDERER})


class PandaRuntimeUnavailable(RuntimeError):
    """Raised when the Panda3D adapter path cannot be used safely."""


class _Logger(Protocol):
    def __call__(self, message: str) -> None: ...


@dataclass(frozen=True)
class PandaRuntimeConfig:
    """Resolved renderer selection for the optional Panda3D path."""

    renderer_mode: str = _NATIVE_RENDERER
    requested_mode: str = _NATIVE_RENDERER
    fallback_reason: str = ""

    @property
    def panda3d_requested(self) -> bool:
        return self.requested_mode == _PANDA3D_RENDERER

    @property
    def panda3d_enabled(self) -> bool:
        return self.renderer_mode == _PANDA3D_RENDERER


@dataclass(frozen=True)
class PandaFrameState:
    """Frame snapshot passed from the existing OpenXR state machine."""

    predicted_display_time: float = 0.0
    head_pose: Any | None = None
    eye_poses: tuple[Any | None, Any | None] = (None, None)
    controller_poses: Mapping[str, Any] = field(default_factory=dict)
    screen_pose: Any | None = None
    screen_texture: Any | None = None


def resolve_gltf_renderer_mode(
    env: Mapping[str, str] | None = None,
    *,
    panda3d_available: bool = False,
) -> PandaRuntimeConfig:
    """Resolve D2S_GLTF_RENDERER without enabling Panda3D by default."""
    source = os.environ if env is None else env
    raw_mode = str(source.get(GLTF_RENDERER_ENV_VAR, _NATIVE_RENDERER) or _NATIVE_RENDERER)
    requested = raw_mode.strip().lower() or _NATIVE_RENDERER
    if requested not in _VALID_RENDERERS:
        return PandaRuntimeConfig(
            renderer_mode=_NATIVE_RENDERER,
            requested_mode=requested,
            fallback_reason=(
                f"unsupported {GLTF_RENDERER_ENV_VAR}={raw_mode!r}; "
                f"expected 'native' or 'panda3d'"
            ),
        )
    if requested == _PANDA3D_RENDERER and not panda3d_available:
        return PandaRuntimeConfig(
            renderer_mode=_NATIVE_RENDERER,
            requested_mode=requested,
            fallback_reason=(
                "Panda3D renderer adapter is present but not enabled; "
                "Phase 0 OpenXR swapchain gate is still pending"
            ),
        )
    return PandaRuntimeConfig(renderer_mode=requested, requested_mode=requested)


class PandaSceneRenderer:
    """Facade boundary for a future Panda3D scene renderer.

    The class deliberately owns only abstract scene/target/bridge helpers here.
    It does not create Panda3D objects until a concrete implementation is wired
    after Phase 0's real OpenXR swapchain gate passes.
    """

    def __init__(
        self,
        *,
        scene: PandaSceneGraph | None = None,
        targets: StereoTargets | None = None,
        bridge: PandaBridge | None = None,
        diagnostics: PandaRuntimeDiagnostics | None = None,
    ):
        self.scene = scene or PandaSceneGraph()
        self.targets = targets or StereoTargets()
        self.bridge = bridge or PandaBridge()
        self.diagnostics = diagnostics or PandaRuntimeDiagnostics()
        self._released = False
        self._last_frame_state: PandaFrameState | None = None

    @property
    def released(self) -> bool:
        return self._released

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        self.scene.load_environment(asset_path)
        self.diagnostics.record_event("environment_loaded", asset_path)

    def load_controller(self, hand: str, asset_path: str) -> None:
        self._ensure_live()
        self.scene.load_controller(hand, asset_path)
        self.diagnostics.record_event(f"controller_{hand}_loaded", asset_path)

    def update_frame_state(self, frame_state: PandaFrameState) -> None:
        self._ensure_live()
        self._last_frame_state = frame_state
        self.scene.update_frame_state(frame_state)

    def rebuild_targets(self, left: StereoTargetSpec, right: StereoTargetSpec) -> None:
        self._ensure_live()
        self.targets.rebuild(left, right)
        self.diagnostics.record_event("stereo_targets_rebuilt", f"{left.width}x{left.height}/{right.width}x{right.height}")

    def render_eyes(
        self,
        left_image: SwapchainImageRef,
        right_image: SwapchainImageRef,
    ) -> RenderEyesResult:
        self._ensure_live()
        if self._last_frame_state is None:
            raise PandaRuntimeUnavailable("PandaSceneRenderer.update_frame_state must run before render_eyes")
        if not self.targets.ready:
            raise PandaRuntimeUnavailable("PandaSceneRenderer.rebuild_targets must run before render_eyes")
        return self.bridge.render_eyes(
            scene=self.scene,
            targets=self.targets,
            frame_state=self._last_frame_state,
            left_image=left_image,
            right_image=right_image,
        )

    def release(self) -> None:
        if self._released:
            return
        self.bridge.release()
        self.targets.release()
        self.scene.release()
        self._released = True
        self.diagnostics.record_event("released", "PandaSceneRenderer")

    def _ensure_live(self) -> None:
        if self._released:
            raise PandaRuntimeUnavailable("PandaSceneRenderer has been released")


def log_renderer_selection(config: PandaRuntimeConfig, logger: _Logger = print) -> None:
    """Log renderer selection once at the integration boundary."""
    if config.fallback_reason:
        logger(f"[OpenXRViewer] glTF renderer fallback: {config.fallback_reason}")
    elif config.panda3d_enabled:
        logger("[OpenXRViewer] glTF renderer requested: panda3d")
    else:
        logger("[OpenXRViewer] glTF renderer: native")
