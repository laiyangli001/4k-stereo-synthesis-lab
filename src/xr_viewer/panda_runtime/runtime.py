"""Renderer facade and selector for the optional Panda3D glTF path."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import os
import time
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
class PandaPose:
    """Pose value object for a single XR-frame snapshot."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]


@dataclass(frozen=True)
class PandaEyeView:
    """Per-eye view state captured from the same xrLocateViews result."""

    eye_index: int
    pose: PandaPose | None = None
    fov: Mapping[str, float] | None = None
    projection: Any | None = None


@dataclass(frozen=True)
class PandaFrameState:
    """3D model-layer snapshot passed from the existing OpenXR state machine."""

    predicted_display_time: float = 0.0
    animation_time_seconds: float | None = None
    frame_index: int | None = None
    head_pose: Any | None = None
    projection_near: float = 0.01
    projection_far: float = 1000.0
    eye_views: tuple[PandaEyeView | None, PandaEyeView | None] = (None, None)
    eye_poses: tuple[Any | None, Any | None] = (None, None)
    controller_poses: Mapping[str, Any] = field(default_factory=dict)


def validate_frame_state(frame_state: PandaFrameState) -> None:
    """Validate that both eyes share one XR frame snapshot."""
    predicted = float(frame_state.predicted_display_time)
    if not math.isfinite(predicted):
        raise PandaRuntimeUnavailable("PandaFrameState.predicted_display_time must be finite")
    projection_near = float(frame_state.projection_near)
    projection_far = float(frame_state.projection_far)
    if (
        not math.isfinite(projection_near)
        or not math.isfinite(projection_far)
        or projection_near <= 0.0
        or projection_far <= projection_near
    ):
        raise PandaRuntimeUnavailable(
            "PandaFrameState projection clip planes must be finite and satisfy 0 < near < far"
        )
    if len(frame_state.eye_views) != 2:
        raise PandaRuntimeUnavailable("PandaFrameState.eye_views must contain exactly two eyes")
    for expected_eye, eye_view in enumerate(frame_state.eye_views):
        if eye_view is None:
            continue
        if int(eye_view.eye_index) != expected_eye:
            raise PandaRuntimeUnavailable(
                f"PandaFrameState eye {expected_eye} has mismatched eye_index={eye_view.eye_index}"
            )


@dataclass(frozen=True)
class PandaAnimationPlaybackState:
    """Runtime controls for glTF node animation sampling."""

    playback_speed: float = 1.0
    paused: bool = False
    fixed_time_seconds: float | None = None
    loop: bool = True


class PandaAnimationClock:
    """Derive a controlled glTF animation clock from XR predicted display time."""

    def __init__(self) -> None:
        self._origin_predicted_display_time: float | None = None
        self._last_predicted_display_time: float | None = None
        self._last_animation_time_seconds = 0.0
        self._playback_speed = 1.0
        self._paused = False
        self._fixed_time_seconds: float | None = None

    @property
    def origin_predicted_display_time(self) -> float | None:
        return self._origin_predicted_display_time

    @property
    def last_animation_time_seconds(self) -> float:
        return self._last_animation_time_seconds

    @property
    def playback_state(self) -> PandaAnimationPlaybackState:
        return PandaAnimationPlaybackState(
            playback_speed=self._playback_speed,
            paused=self._paused,
            fixed_time_seconds=self._fixed_time_seconds,
        )

    def configure(
        self,
        *,
        playback_speed: float | None = None,
        paused: bool | None = None,
        fixed_time_seconds: float | None | object = ...,
    ) -> PandaAnimationPlaybackState:
        if playback_speed is not None:
            speed = float(playback_speed)
            if not math.isfinite(speed) or speed < 0.0:
                raise PandaRuntimeUnavailable("animation playback_speed must be finite and non-negative")
            self._playback_speed = speed
        if paused is not None:
            self._paused = bool(paused)
        if fixed_time_seconds is not ...:
            if fixed_time_seconds is None:
                self._fixed_time_seconds = None
            else:
                fixed = float(fixed_time_seconds)
                if not math.isfinite(fixed) or fixed < 0.0:
                    raise PandaRuntimeUnavailable("animation fixed_time_seconds must be finite and non-negative")
                self._fixed_time_seconds = fixed
                self._last_animation_time_seconds = fixed
        return self.playback_state

    def sample(self, predicted_display_time: float) -> float:
        predicted = float(predicted_display_time)
        if not math.isfinite(predicted):
            raise PandaRuntimeUnavailable("predicted_display_time must be finite")
        if self._fixed_time_seconds is not None:
            self._last_predicted_display_time = predicted
            return self._fixed_time_seconds
        if self._origin_predicted_display_time is None:
            self._origin_predicted_display_time = predicted
            self._last_predicted_display_time = predicted
            self._last_animation_time_seconds = 0.0
            return 0.0
        if self._last_predicted_display_time is None:
            self._last_predicted_display_time = predicted
        delta_seconds = max(0.0, predicted - self._last_predicted_display_time)
        if predicted > self._last_predicted_display_time:
            self._last_predicted_display_time = predicted
        if not self._paused:
            self._last_animation_time_seconds += delta_seconds * self._playback_speed
        return self._last_animation_time_seconds

    def bind(self, frame_state: PandaFrameState) -> PandaFrameState:
        return replace(
            frame_state,
            animation_time_seconds=self.sample(frame_state.predicted_display_time),
        )

    def reset(self) -> None:
        self._origin_predicted_display_time = None
        self._last_predicted_display_time = None
        self._last_animation_time_seconds = 0.0
        self._fixed_time_seconds = None


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
        self.animation_clock = PandaAnimationClock()
        self._animation_loop = True
        self._released = False
        self._last_frame_state: PandaFrameState | None = None
        self._render_success_count = 0
        self._render_failure_count = 0
        self._last_render_bridge_mode = ""
        self._last_render_left_rendered = False
        self._last_render_right_rendered = False
        self._last_render_error = ""
        self._last_render_cpu_seconds = 0.0

    @property
    def released(self) -> bool:
        return self._released

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        self.scene.load_environment(asset_path)
        self.diagnostics.record_event("environment_loaded", asset_path)

    def configure_environment_transform(self, position: Any, rotation: Any, scale: Any) -> None:
        self._ensure_live()
        self.scene.configure_environment_transform(position, rotation, scale)

    def configure_environment_lighting(
        self,
        ambient_color: Any,
        head_light_color: Any,
        fill_lights: Any,
    ) -> None:
        self._ensure_live()
        self.scene.configure_environment_lighting(ambient_color, head_light_color, fill_lights)


    def load_controller(self, hand: str, asset_path: str) -> None:
        self._ensure_live()
        self.scene.load_controller(hand, asset_path)
        self.diagnostics.record_event(f"controller_{hand}_loaded", asset_path)

    def configure_animation(
        self,
        *,
        playback_speed: float | None = None,
        paused: bool | None = None,
        fixed_time_seconds: float | None | object = ...,
        loop: bool | None = None,
    ) -> PandaAnimationPlaybackState:
        self._ensure_live()
        state = self.animation_clock.configure(
            playback_speed=playback_speed,
            paused=paused,
            fixed_time_seconds=fixed_time_seconds,
        )
        if loop is not None:
            self._animation_loop = bool(loop)
            self.scene.set_animation_looping(self._animation_loop)
        state = replace(state, loop=self._animation_loop)
        self.diagnostics.record_event("animation_configured", _animation_state_detail(state))
        return state

    def update_frame_state(self, frame_state: PandaFrameState) -> None:
        self._ensure_live()
        validate_frame_state(frame_state)
        bound_frame_state = self.animation_clock.bind(frame_state)
        self._last_frame_state = bound_frame_state
        self.scene.update_frame_state(bound_frame_state)

    def rebuild_targets(self, left: StereoTargetSpec, right: StereoTargetSpec) -> None:
        self._ensure_live()
        if self.targets.ready and self.targets.left == left and self.targets.right == right:
            return
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
        render_start = time.perf_counter()
        try:
            result = self.bridge.render_eyes(
                scene=self.scene,
                targets=self.targets,
                frame_state=self._last_frame_state,
                left_image=left_image,
                right_image=right_image,
            )
        except Exception as exc:
            self._last_render_cpu_seconds = time.perf_counter() - render_start
            self._render_failure_count += 1
            self._last_render_error = f"{type(exc).__name__}: {exc}"
            self.diagnostics.record_event("render_failed", self._last_render_error)
            raise
        self._last_render_cpu_seconds = time.perf_counter() - render_start
        self._render_success_count += 1
        self._last_render_error = ""
        self._last_render_bridge_mode = str(result.bridge_mode)
        self._last_render_left_rendered = bool(result.left_rendered)
        self._last_render_right_rendered = bool(result.right_rendered)
        self.diagnostics.record_event(
            "render_eyes",
            _render_result_detail(result, left_image, right_image),
        )
        return result

    def diagnostics_snapshot(self) -> Any:
        return self.diagnostics.snapshot(self)

    def diagnostics_json(self) -> str:
        return self.diagnostics.snapshot_json(self)

    def release(self) -> None:
        if self._released:
            return
        self.bridge.release()
        self.targets.release()
        self.scene.release()
        self.animation_clock.reset()
        self._animation_loop = True
        self._released = True
        self.diagnostics.record_event("released", "PandaSceneRenderer")

    def _ensure_live(self) -> None:
        if self._released:
            raise PandaRuntimeUnavailable("PandaSceneRenderer has been released")


def _render_result_detail(
    result: RenderEyesResult,
    left_image: SwapchainImageRef,
    right_image: SwapchainImageRef,
) -> str:
    return (
        f"mode={result.bridge_mode};"
        f"left={int(result.left_rendered)};"
        f"right={int(result.right_rendered)};"
        f"left_image={left_image.image_index}:{left_image.width}x{left_image.height};"
        f"right_image={right_image.image_index}:{right_image.width}x{right_image.height}"
    )


def _animation_state_detail(state: PandaAnimationPlaybackState) -> str:
    fixed = "none" if state.fixed_time_seconds is None else f"{state.fixed_time_seconds:.6f}"
    return (
        f"speed={state.playback_speed:.6f};"
        f"paused={int(state.paused)};"
        f"fixed={fixed};"
        f"loop={int(state.loop)}"
    )


def log_renderer_selection(config: PandaRuntimeConfig, logger: _Logger = print) -> None:
    """Log renderer selection once at the integration boundary."""
    if config.fallback_reason:
        logger(f"[OpenXRViewer] glTF renderer fallback: {config.fallback_reason}")
    elif config.panda3d_enabled:
        logger("[OpenXRViewer] glTF renderer requested: panda3d")
    else:
        logger("[OpenXRViewer] glTF renderer: native")
