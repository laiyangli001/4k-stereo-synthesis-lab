"""Scene graph ownership contracts for the optional Panda3D renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping

from .coordinates import (
    gltf_position_to_panda,
    gltf_rotation_to_panda_hpr_degrees,
    gltf_scale_to_panda,
)
from .star_glim import apply_star_glim_sidecar, set_star_glim_time


@dataclass(frozen=True)
class PandaAssetRef:
    role: str
    path: str
    loaded_with_panda: bool = False
    node_count: int = 0
    geom_count: int = 0
    animation_channel_count: int = 0
    animation_target_node_count: int = 0
    animation_bound_node_count: int = 0
    animation_duration_seconds: float = 0.0


@dataclass(frozen=True)
class PandaFillLight:
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    range: float


@dataclass(frozen=True)
class PandaEnvironmentLighting:
    ambient_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head_light_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fill_lights: tuple[PandaFillLight, ...] = ()

    @property
    def active(self) -> bool:
        return any(self.ambient_color) or any(self.head_light_color) or bool(self.fill_lights)


@dataclass(frozen=True)
class PandaSceneSnapshot:
    frame_index: int | None = None
    animation_time_seconds: float | None = None
    animation_sample_count: int = 0
    animation_applied_player_count: int = 0
    animation_player_count: int = 0
    animation_channel_count: int = 0
    animation_bound_node_count: int = 0
    controller_hands: tuple[str, ...] = ()
    eye_view_count: int = 0
    applied_controller_hands: tuple[str, ...] = ()


@dataclass
class PandaSceneGraph:
    """Owns glTF asset roots without exposing Panda NodePath to callers."""

    load_panda_assets: bool = False
    environment: PandaAssetRef | None = None
    controllers: dict[str, PandaAssetRef] = field(default_factory=dict)
    frame_state: Any | None = None
    snapshot: PandaSceneSnapshot = field(default_factory=PandaSceneSnapshot)
    released: bool = False
    environment_lighting: PandaEnvironmentLighting = field(default_factory=PandaEnvironmentLighting)
    _environment_root: Any | None = field(default=None, init=False, repr=False)
    _environment_star_glim_nodes: tuple[Any, ...] = field(default_factory=tuple, init=False, repr=False)
    _controller_roots: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _environment_animation_player: Any | None = field(default=None, init=False, repr=False)
    _controller_animation_players: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _animation_loop: bool = field(default=True, init=False, repr=False)
    _animation_sample_count: int = field(default=0, init=False, repr=False)
    _last_animation_time_seconds: float | None = field(default=None, init=False, repr=False)
    _last_render_base: Any | None = field(default=None, init=False, repr=False)
    _light_render_root: Any | None = field(default=None, init=False, repr=False)
    _light_nodes: list[Any] = field(default_factory=list, init=False, repr=False)
    _head_light_node: Any | None = field(default=None, init=False, repr=False)
    _lighting_dirty: bool = field(default=True, init=False, repr=False)

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        asset, root, animation_player = self._make_asset_ref("environment", asset_path)
        self.environment = asset
        self._environment_root = root
        self._environment_star_glim_nodes = (
            apply_star_glim_sidecar(asset_path, root, base_color_texture=_base_color_texture)
            if root is not None
            else ()
        )
        self._environment_animation_player = animation_player
        _set_animation_player_loop(animation_player, self._animation_loop)

    def configure_environment_transform(self, position: Any, rotation: Any, scale: Any) -> None:
        self._ensure_live()
        _apply_environment_transform(self._environment_root, position, rotation, scale)

    def configure_environment_lighting(
        self,
        ambient_color: Any,
        head_light_color: Any,
        fill_lights: Any,
    ) -> None:
        self._ensure_live()
        lighting = PandaEnvironmentLighting(
            ambient_color=_lighting_vec3(ambient_color),
            head_light_color=_lighting_vec3(head_light_color),
            fill_lights=_normalize_fill_lights(fill_lights),
        )
        if lighting == self.environment_lighting:
            return
        self.environment_lighting = lighting
        self._lighting_dirty = True


    def load_controller(self, hand: str, asset_path: str) -> None:
        self._ensure_live()
        key = str(hand).strip().lower()
        if key not in {"left", "right"}:
            raise ValueError("controller hand must be 'left' or 'right'")
        asset, root, animation_player = self._make_asset_ref(f"controller:{key}", asset_path)
        self.controllers[key] = asset
        if root is not None:
            self._controller_roots[key] = root
        else:
            self._controller_roots.pop(key, None)
        if animation_player is not None:
            _set_animation_player_loop(animation_player, self._animation_loop)
            self._controller_animation_players[key] = animation_player
        else:
            self._controller_animation_players.pop(key, None)

    def update_frame_state(self, frame_state: Any) -> None:
        self._ensure_live()
        self.frame_state = frame_state
        applied_controller_hands = self._apply_controller_poses(frame_state)
        animation_time = getattr(frame_state, "animation_time_seconds", None)
        animation_time_seconds = None if animation_time is None else float(animation_time)
        animation_applied_player_count = 0
        if animation_time_seconds is not None:
            animation_applied_player_count = self._apply_animation_time(animation_time_seconds)
            set_star_glim_time(self._environment_star_glim_nodes, animation_time_seconds)
        self.snapshot = _snapshot_from_frame_state(
            frame_state,
            applied_controller_hands,
            animation_time_seconds=animation_time_seconds,
            animation_applied_player_count=animation_applied_player_count,
            animation_player_count=self._animation_player_count(),
            animation_channel_count=self._animation_channel_count(),
            animation_bound_node_count=self._animation_bound_node_count(),
            animation_sample_count=self._animation_sample_count,
        )
        self._update_head_light_pose(frame_state)

    def loaded_assets(self) -> tuple[PandaAssetRef, ...]:
        assets = []
        if self.environment is not None:
            assets.append(self.environment)
        assets.extend(self.controllers[key] for key in sorted(self.controllers))
        return tuple(assets)

    def controller_paths(self) -> Mapping[str, str]:
        return {hand: asset.path for hand, asset in self.controllers.items()}

    def set_animation_looping(self, loop: bool) -> None:
        self._ensure_live()
        self._animation_loop = bool(loop)
        _set_animation_player_loop(self._environment_animation_player, self._animation_loop)
        for player in self._controller_animation_players.values():
            _set_animation_player_loop(player, self._animation_loop)

    def render_to_framebuffers(
        self,
        *,
        targets: Any,
        frame_state: Any,
        left_framebuffer: Any,
        right_framebuffer: Any,
        left_resource: Any | None = None,
        right_resource: Any | None = None,
        make_target_context_current: Any | None = None,
        require_shared_context: bool = False,
    ) -> None:
        self._ensure_live()
        base = getattr(targets, "_panda_base", None)
        if base is None:
            raise RuntimeError("Panda stereo targets have no ShowBase")
        if not getattr(targets, "_panda_textures", None):
            raise RuntimeError("Panda stereo targets have no color textures")
        if len(targets._panda_textures) < 2:
            raise RuntimeError("Panda stereo targets must contain both eye textures")
        make_panda_context_current = getattr(targets, "make_panda_context_current", None)
        if callable(make_panda_context_current):
            make_panda_context_current()
        self._attach_roots_to_base(base)
        _update_eye_cameras(getattr(targets, "_panda_cameras", ()), frame_state)
        _set_framebuffer_srgb(True)
        try:
            _render_panda_base_frame(base)
            _render_panda_base_frame(base)
        finally:
            _set_framebuffer_srgb(False)
        if make_target_context_current is not None:
            make_target_context_current()
        elif require_shared_context:
            raise RuntimeError("Panda OpenGL fallback requires a current target context")
        _drain_target_gl_errors()
        _blit_panda_texture_to_framebuffer(
            _panda_texture_source(targets, 0),
            left_framebuffer,
            _resource_width(left_resource) or _target_width(targets, 0),
            _resource_height(left_resource) or _target_height(targets, 0),
        )
        _blit_panda_texture_to_framebuffer(
            _panda_texture_source(targets, 1),
            right_framebuffer,
            _resource_width(right_resource) or _target_width(targets, 1),
            _resource_height(right_resource) or _target_height(targets, 1),
        )

    def release(self) -> None:
        self._clear_environment_lights()
        self.environment = None
        self.controllers.clear()
        self._environment_root = None
        self._environment_star_glim_nodes = ()
        self._controller_roots.clear()
        self._environment_animation_player = None
        self._controller_animation_players.clear()
        self._animation_loop = True
        self._animation_sample_count = 0
        self._last_animation_time_seconds = None
        self._last_render_base = None
        self.environment_lighting = PandaEnvironmentLighting()
        self._lighting_dirty = True
        self.frame_state = None
        self.snapshot = PandaSceneSnapshot()
        self.released = True

    def _apply_animation_time(self, time_seconds: float) -> int:
        applied = 0
        for player in self._animation_players():
            player.set_time_seconds(time_seconds)
            applied += 1
        self._last_animation_time_seconds = time_seconds
        if applied > 0:
            self._animation_sample_count += 1
        return applied

    def _animation_players(self) -> tuple[Any, ...]:
        players: list[Any] = []
        if self._environment_animation_player is not None:
            players.append(self._environment_animation_player)
        players.extend(self._controller_animation_players[key] for key in sorted(self._controller_animation_players))
        return tuple(players)

    def _animation_player_count(self) -> int:
        return len(self._animation_players())

    def _animation_channel_count(self) -> int:
        return sum(_animation_runtime_int(player, "channel_count") for player in self._animation_players())

    def _animation_bound_node_count(self) -> int:
        return sum(_animation_runtime_int(player, "bound_node_count") for player in self._animation_players())

    def _apply_controller_poses(self, frame_state: Any) -> tuple[str, ...]:
        controller_poses = getattr(frame_state, "controller_poses", {}) or {}
        applied: list[str] = []
        for hand, root in sorted(self._controller_roots.items()):
            pose = controller_poses.get(hand)
            if pose is None:
                continue
            if _apply_pose_to_node_path(root, pose):
                applied.append(hand)
        return tuple(applied)

    def _attach_roots_to_base(self, base: Any) -> None:
        render_root = getattr(base, "render", None)
        if render_root is None:
            raise RuntimeError("Panda ShowBase has no render root")
        if self._last_render_base is not base:
            self._clear_environment_lights()
            for root in _iter_scene_roots(
                self._environment_root,
                self._controller_roots.values(),
            ):
                if hasattr(root, "reparent_to"):
                    root.reparent_to(render_root)
            self._last_render_base = base
            self._lighting_dirty = True
        if self._lighting_dirty:
            self._install_environment_lights(base)
        self._update_head_light_pose(self.frame_state)

    def _install_environment_lights(self, base: Any) -> None:
        self._clear_environment_lights()
        lighting = self.environment_lighting
        self._lighting_dirty = False
        render_root = getattr(base, "render", None)
        if render_root is None:
            raise RuntimeError("Panda ShowBase has no render root")
        _set_preview_diffuse_shader_inputs(render_root, lighting)
        if not lighting.active:
            return
        if not all(hasattr(render_root, name) for name in ("attach_new_node", "set_light")):
            raise RuntimeError("Panda render root cannot own environment lights")
        from panda3d.core import AmbientLight, LVector3, PointLight

        if hasattr(render_root, "set_shader_auto"):
            render_root.set_shader_auto()
        if any(lighting.ambient_color):
            ambient = AmbientLight("d2s-profile-ambient")
            ambient.set_color((*lighting.ambient_color, 1.0))
            ambient_node = render_root.attach_new_node(ambient)
            render_root.set_light(ambient_node)
            self._light_nodes.append(ambient_node)
        if any(lighting.head_light_color):
            head = PointLight("d2s-profile-head")
            head.set_color((*lighting.head_light_color, 1.0))
            head.set_attenuation(LVector3(1.0, 0.0, 0.0))
            self._head_light_node = render_root.attach_new_node(head)
            render_root.set_light(self._head_light_node)
            self._light_nodes.append(self._head_light_node)
        fill_parent = self._environment_root if hasattr(self._environment_root, "attach_new_node") else render_root
        for index, spec in enumerate(lighting.fill_lights):
            fill = PointLight(f"d2s-profile-fill-{index}")
            fill.set_color((*spec.color, 1.0))
            inverse_range_sq = 1.0 / max(spec.range * spec.range, 1e-6)
            fill.set_attenuation(LVector3(1.0, 0.0, inverse_range_sq))
            fill_node = fill_parent.attach_new_node(fill)
            fill_node.set_pos(spec.position[0], -spec.position[2], spec.position[1])
            render_root.set_light(fill_node)
            self._light_nodes.append(fill_node)
        self._light_render_root = render_root
        self._update_head_light_pose(self.frame_state)

    def _clear_environment_lights(self) -> None:
        render_root = self._light_render_root
        for node in self._light_nodes:
            if render_root is not None and hasattr(render_root, "clear_light"):
                try:
                    render_root.clear_light(node)
                except Exception:
                    pass
            if hasattr(node, "remove_node"):
                try:
                    node.remove_node()
                except Exception:
                    pass
        self._light_nodes.clear()
        self._head_light_node = None
        self._light_render_root = None

    def _update_head_light_pose(self, frame_state: Any | None) -> None:
        if self._head_light_node is None or frame_state is None:
            return
        position = _head_position_from_frame_state(frame_state)
        if position is not None and hasattr(self._head_light_node, "set_pos"):
            self._head_light_node.set_pos(*position)

    def _make_asset_ref(self, role: str, asset_path: str) -> tuple[PandaAssetRef, Any | None, Any | None]:
        path = str(Path(asset_path))
        if not self.load_panda_assets:
            return PandaAssetRef(role, path), None, None
        root = _load_panda_root(path)
        node_count, geom_count = _node_counts(root)
        animation_player = _make_node_animation_player(path, root)
        animation_runtime = getattr(animation_player, "runtime", None)
        return (
            PandaAssetRef(
                role,
                path,
                True,
                node_count,
                geom_count,
                int(getattr(animation_runtime, "channel_count", 0) or 0),
                int(getattr(animation_runtime, "target_node_count", 0) or 0),
                int(getattr(animation_runtime, "bound_node_count", 0) or 0),
                float(getattr(animation_runtime, "duration_seconds", 0.0) or 0.0),
            ),
            root,
            animation_player,
        )

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError("PandaSceneGraph has been released")


def _set_preview_diffuse_shader_inputs(render_root: Any, lighting: PandaEnvironmentLighting) -> None:
    if not hasattr(render_root, "set_shader_input"):
        return
    ambient = tuple(max(0.22, float(component)) for component in lighting.ambient_color)
    head = tuple(max(0.85, float(component)) for component in lighting.head_light_color)
    if not lighting.active:
        ambient = (0.24, 0.24, 0.26)
        head = (0.70, 0.70, 0.72)
    try:
        render_root.set_shader_input("d2s_preview_ambient_color", *ambient)
        render_root.set_shader_input("d2s_preview_light_color", *head)
        render_root.set_shader_input("d2s_preview_exposure", 2.2)
        render_root.set_shader_input("camera_world_position", 0.0, 0.0, 0.0)
    except Exception:
        return

def _render_panda_base_frame(base: Any) -> None:
    base.graphicsEngine.render_frame()


def _lighting_vec3(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return (0.0, 0.0, 0.0)
    result = tuple(max(0.0, float(component)) for component in value[:3])
    if not all(math.isfinite(component) for component in result):
        raise ValueError("Panda environment light color must be finite")
    return result


def _normalize_fill_lights(value: Any) -> tuple[PandaFillLight, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    lights: list[PandaFillLight] = []
    for item in value:
        if isinstance(item, PandaFillLight):
            lights.append(item)
            continue
        if not isinstance(item, Mapping):
            continue
        position = _lighting_vec3_signed(item.get("position"))
        color = _lighting_vec3(item.get("color"))
        try:
            light_range = max(0.001, float(item.get("range", 1.0)))
        except (TypeError, ValueError):
            light_range = 1.0
        if not math.isfinite(light_range):
            raise ValueError("Panda environment fill-light range must be finite")
        if any(color):
            lights.append(PandaFillLight(position, color, light_range))
    return tuple(lights)


def _lighting_vec3_signed(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return (0.0, 0.0, 0.0)
    result = tuple(float(component) for component in value[:3])
    if not all(math.isfinite(component) for component in result):
        raise ValueError("Panda environment light position must be finite")
    return result


def _head_position_from_frame_state(frame_state: Any) -> tuple[float, float, float] | None:
    head_pose = getattr(frame_state, "head_pose", None)
    position = getattr(head_pose, "position", None)
    if isinstance(position, (list, tuple)) and len(position) >= 3:
        return tuple(float(component) for component in position[:3])
    positions = []
    for eye_view in getattr(frame_state, "eye_views", ()) or ():
        pose = getattr(eye_view, "pose", None)
        eye_position = getattr(pose, "position", None)
        if isinstance(eye_position, (list, tuple)) and len(eye_position) >= 3:
            positions.append(tuple(float(component) for component in eye_position[:3]))
    if not positions:
        return None
    count = float(len(positions))
    return tuple(sum(position[axis] for position in positions) / count for axis in range(3))

def _snapshot_from_frame_state(
    frame_state: Any,
    applied_controller_hands: tuple[str, ...] = (),
    *,
    animation_time_seconds: float | None = None,
    animation_applied_player_count: int = 0,
    animation_player_count: int = 0,
    animation_channel_count: int = 0,
    animation_bound_node_count: int = 0,
    animation_sample_count: int = 0,
) -> PandaSceneSnapshot:
    eye_views = getattr(frame_state, "eye_views", ()) or ()
    controller_poses = getattr(frame_state, "controller_poses", {}) or {}
    return PandaSceneSnapshot(
        frame_index=getattr(frame_state, "frame_index", None),
        animation_time_seconds=animation_time_seconds,
        animation_sample_count=int(animation_sample_count),
        animation_applied_player_count=int(animation_applied_player_count),
        animation_player_count=int(animation_player_count),
        animation_channel_count=int(animation_channel_count),
        animation_bound_node_count=int(animation_bound_node_count),
        controller_hands=tuple(sorted(str(hand) for hand in controller_poses)),
        eye_view_count=sum(1 for eye_view in eye_views if eye_view is not None),
        applied_controller_hands=applied_controller_hands,
    )


def _set_animation_player_loop(player: Any | None, loop: bool) -> None:
    if player is not None and hasattr(player, "loop"):
        player.loop = bool(loop)


def _animation_runtime_int(player: Any, name: str) -> int:
    runtime = getattr(player, "runtime", None)
    return int(getattr(runtime, name, 0) or 0)


def _iter_scene_roots(
    environment_root: Any | None,
    controller_roots: Any,
) -> tuple[Any, ...]:
    roots: list[Any] = []
    if environment_root is not None:
        roots.append(environment_root)
    roots.extend(root for root in controller_roots if root is not None)
    return tuple(roots)


def _framebuffer_id(framebuffer: Any) -> int:
    value = getattr(framebuffer, "glo", framebuffer)
    value = getattr(value, "value", value)
    framebuffer_id = int(value or 0)
    if framebuffer_id <= 0:
        raise RuntimeError("target framebuffer has no OpenGL id")
    return framebuffer_id


def _panda_texture_native_id(texture: Any) -> int:
    if isinstance(texture, int) and texture > 0:
        return texture
    cached_id = int(getattr(texture, "_d2s_native_id", 0) or 0)
    if cached_id > 0:
        return cached_id
    getter = getattr(texture, "get_native_id", None)
    if callable(getter):
        texture_id = int(getter() or 0)
        if texture_id > 0:
            return texture_id
    context = getattr(texture, "get_texture_context", lambda: None)()
    getter = getattr(context, "get_native_id", None)
    texture_id = int(getter() or 0) if callable(getter) else 0
    if texture_id <= 0:
        raise RuntimeError("Panda color texture has no OpenGL native id")
    return texture_id


def _panda_texture_source(targets: Any, eye_index: int) -> Any:
    ref = getattr(targets, "left_ref" if eye_index == 0 else "right_ref", None)
    native_id = int(getattr(ref, "texture_native_id", 0) or 0)
    if native_id > 0:
        return native_id
    return targets._panda_textures[eye_index]


def _set_framebuffer_srgb(enabled: bool) -> None:
    try:
        from OpenGL.GL import GL_FRAMEBUFFER_SRGB, glDisable, glEnable
    except Exception:
        return
    try:
        if enabled:
            glEnable(GL_FRAMEBUFFER_SRGB)
        else:
            glDisable(GL_FRAMEBUFFER_SRGB)
    except Exception:
        return


def _drain_target_gl_errors() -> None:
    from OpenGL.GL import GL_NO_ERROR, glGetError

    for _ in range(8):
        try:
            err = glGetError()
        except Exception:
            continue
        if err == GL_NO_ERROR:
            return


def _resource_width(resource: Any | None) -> int:
    return int(getattr(getattr(resource, "key", None), "width", 0) or 0)


def _resource_height(resource: Any | None) -> int:
    return int(getattr(getattr(resource, "key", None), "height", 0) or 0)


def _target_width(targets: Any, eye_index: int) -> int:
    spec = getattr(getattr(targets, "left" if eye_index == 0 else "right", None), "width", 0)
    return int(spec or 0)


def _target_height(targets: Any, eye_index: int) -> int:
    spec = getattr(getattr(targets, "left" if eye_index == 0 else "right", None), "height", 0)
    return int(spec or 0)


def _blit_panda_texture_to_framebuffer(texture: Any, framebuffer: Any, width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise RuntimeError("Panda blit dimensions must be positive")
    from OpenGL.GL import (
        GL_COLOR_ATTACHMENT0,
        GL_COLOR_BUFFER_BIT,
        GL_DRAW_FRAMEBUFFER,
        GL_FRAMEBUFFER,
        GL_FRAMEBUFFER_COMPLETE,
        GL_LINEAR,
        GL_READ_FRAMEBUFFER,
        GL_TEXTURE_2D,
        glBindFramebuffer,
        glBlitFramebuffer,
        glCheckFramebufferStatus,
        glDeleteFramebuffers,
        glFramebufferTexture2D,
        glGenFramebuffers,
        glReadBuffer,
    )

    source_texture_id = _panda_texture_native_id(texture)
    target_framebuffer_id = _framebuffer_id(framebuffer)
    read_fbo = int(glGenFramebuffers(1))
    try:
        glBindFramebuffer(GL_READ_FRAMEBUFFER, read_fbo)
        glFramebufferTexture2D(
            GL_READ_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            source_texture_id,
            0,
        )
        if glCheckFramebufferStatus(GL_READ_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError("Panda source framebuffer is incomplete")
        glReadBuffer(GL_COLOR_ATTACHMENT0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target_framebuffer_id)
        glBlitFramebuffer(
            0,
            0,
            width,
            height,
            0,
            0,
            width,
            height,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR,
        )
    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        if read_fbo:
            glDeleteFramebuffers(1, [read_fbo])


def _update_eye_cameras(cameras: Any, frame_state: Any) -> None:
    eye_views = getattr(frame_state, "eye_views", ()) or ()
    projection_near = float(getattr(frame_state, "projection_near", 0.01))
    projection_far = float(getattr(frame_state, "projection_far", 1000.0))
    for eye_index, camera in enumerate(tuple(cameras)[:2]):
        if camera is None or eye_index >= len(eye_views):
            continue
        eye_view = eye_views[eye_index]
        if eye_view is None:
            continue
        pose = getattr(eye_view, "pose", None)
        if pose is not None:
            _apply_pose_to_node_path(camera, pose)
        fov = getattr(eye_view, "fov", None)
        if fov is not None:
            _apply_fov_to_camera(camera, fov)
        _apply_clip_planes_to_camera(camera, projection_near, projection_far)


def _apply_fov_to_camera(camera: Any, fov: Any) -> bool:
    node_getter = getattr(camera, "node", None)
    node = node_getter() if callable(node_getter) else None
    lens_getter = getattr(node, "get_lens", None)
    lens = lens_getter() if callable(lens_getter) else None
    if lens is None or not hasattr(lens, "set_fov"):
        return False
    left = _fov_value(fov, "angle_left", "left")
    right = _fov_value(fov, "angle_right", "right")
    up = _fov_value(fov, "angle_up", "up")
    down = _fov_value(fov, "angle_down", "down")
    if None in (left, right, up, down):
        return False
    import math

    lens.set_fov(math.degrees(abs(left) + abs(right)), math.degrees(abs(up) + abs(down)))
    return True


def _apply_clip_planes_to_camera(camera: Any, near_clip: float, far_clip: float) -> bool:
    node_getter = getattr(camera, "node", None)
    node = node_getter() if callable(node_getter) else None
    lens_getter = getattr(node, "get_lens", None)
    lens = lens_getter() if callable(lens_getter) else None
    if lens is None or not hasattr(lens, "set_near_far"):
        return False
    lens.set_near_far(float(near_clip), float(far_clip))
    return True


def _fov_value(fov: Any, attr_name: str, mapping_name: str) -> float | None:
    if isinstance(fov, Mapping):
        value = fov.get(attr_name, fov.get(mapping_name))
    else:
        value = getattr(fov, attr_name, None)
    return None if value is None else float(value)


def _apply_pose_to_node_path(node_path: Any, pose: Any) -> bool:
    position = getattr(pose, "position", None)
    orientation = getattr(pose, "orientation", None)
    if position is None or orientation is None:
        return False
    if len(position) != 3 or len(orientation) != 4:
        return False
    if not hasattr(node_path, "set_pos_quat"):
        return False
    from panda3d.core import LPoint3, LQuaternion

    x, y, z = (float(value) for value in position)
    qx, qy, qz, qw = (float(value) for value in orientation)
    node_path.set_pos_quat(LPoint3(x, y, z), LQuaternion(qw, qx, qy, qz))
    return True


def _apply_environment_transform(root: Any, position: Any, rotation: Any, scale: Any) -> bool:
    if root is None or not all(hasattr(root, name) for name in ("set_pos", "set_hpr", "set_scale")):
        return False
    root.set_pos(*gltf_position_to_panda(position))
    root.set_hpr(*gltf_rotation_to_panda_hpr_degrees(rotation))
    root.set_scale(*gltf_scale_to_panda(scale))
    return True



def _load_panda_root(asset_path: str) -> Any:
    import gltf
    from panda3d.core import NodePath

    root = NodePath(gltf.load_model(str(asset_path)))
    _apply_gltf_unlit_extension_hints(asset_path, root)
    _apply_gltf_preview_diffuse_materials(asset_path, root)
    return root


def _apply_gltf_unlit_extension_hints(asset_path: str, root: Any) -> None:
    try:
        from pygltflib import GLTF2
    except Exception:
        return
    try:
        document = GLTF2().load(str(asset_path))
    except Exception:
        return
    materials = tuple(document.materials or ())
    unlit_material_ids = {
        index
        for index, material in enumerate(materials)
        if bool(getattr(material, "extensions", None))
        and "KHR_materials_unlit" in getattr(material, "extensions", {})
    }
    if not unlit_material_ids:
        return
    mesh_ids = set()
    for mesh_index, mesh in enumerate(document.meshes or ()):
        for primitive in mesh.primitives or ():
            if primitive.material in unlit_material_ids:
                mesh_ids.add(mesh_index)
                break
    if not mesh_ids:
        return
    for node in document.nodes or ():
        if node.mesh not in mesh_ids or not node.name:
            continue
        node_path = root.find(f"**/{node.name}")
        if node_path.is_empty():
            continue
        if hasattr(node_path, "set_light_off"):
            node_path.set_light_off(10)

def _apply_gltf_preview_diffuse_materials(asset_path: str, root: Any) -> None:
    try:
        from pygltflib import GLTF2
    except Exception:
        return
    try:
        document = GLTF2().load(str(asset_path))
    except Exception:
        return
    material_ids = set()
    for index, material in enumerate(document.materials or ()):
        if "KHR_materials_unlit" in (getattr(material, "extensions", None) or {}):
            continue
        pbr = getattr(material, "pbrMetallicRoughness", None)
        if pbr is None or getattr(pbr, "baseColorTexture", None) is None:
            continue
        if getattr(pbr, "metallicRoughnessTexture", None) is not None:
            continue
        if getattr(material, "normalTexture", None) is not None:
            continue
        if getattr(material, "emissiveTexture", None) is not None:
            continue
        material_ids.add(index)
    if not material_ids:
        return
    mesh_ids = set()
    for mesh_index, mesh in enumerate(document.meshes or ()):
        for primitive in mesh.primitives or ():
            if primitive.material in material_ids:
                mesh_ids.add(mesh_index)
                break
    if not mesh_ids:
        return
    shader = _preview_diffuse_shader()
    for node in document.nodes or ():
        if node.mesh not in mesh_ids or not node.name:
            continue
        node_path = root.find(f"**/{node.name}")
        if node_path.is_empty():
            continue
        for geom_path in node_path.find_all_matches("**/+GeomNode"):
            base_texture = _base_color_texture(geom_path)
            if base_texture is None:
                continue
            geom_path.set_shader(shader, 20)
            geom_path.set_shader_input("d2s_base_color_texture", base_texture)


def _base_color_texture(geom_path: Any) -> Any | None:
    try:
        from panda3d.core import TextureAttrib
    except Exception:
        return None
    try:
        node = geom_path.node()
        if node.get_num_geoms() <= 0:
            return None
        state = node.get_geom_state(0).compose(geom_path.get_state())
        texture_attrib = state.get_attrib(TextureAttrib)
    except Exception:
        return None
    if texture_attrib is None:
        return None
    for index in range(texture_attrib.get_num_on_stages()):
        stage = texture_attrib.get_on_stage(index)
        if stage.get_name() == "Base Color":
            return texture_attrib.get_on_texture(stage)
    return None


_PREVIEW_DIFFUSE_SHADER: Any | None = None


def _preview_diffuse_shader() -> Any:
    global _PREVIEW_DIFFUSE_SHADER
    if _PREVIEW_DIFFUSE_SHADER is not None:
        return _PREVIEW_DIFFUSE_SHADER
    from panda3d.core import Shader

    vertex = """
#version 120
uniform mat4 p3d_ModelViewProjectionMatrix;
uniform mat4 p3d_ModelMatrix;
uniform mat3 p3d_NormalMatrix;
uniform mat4 p3d_TextureMatrix;
attribute vec4 p3d_Vertex;
attribute vec4 p3d_Color;
attribute vec3 p3d_Normal;
attribute vec2 p3d_MultiTexCoord0;
varying vec3 v_normal;
varying vec3 v_world_position;
varying vec4 v_color;
varying vec2 v_texcoord;
void main() {
    v_normal = normalize(p3d_NormalMatrix * p3d_Normal);
    v_world_position = (p3d_ModelMatrix * p3d_Vertex).xyz;
    v_color = p3d_Color;
    v_texcoord = (p3d_TextureMatrix * vec4(p3d_MultiTexCoord0, 0.0, 1.0)).xy;
    gl_Position = p3d_ModelViewProjectionMatrix * p3d_Vertex;
}
"""
    fragment = """
#version 120
uniform struct p3d_MaterialParameters {
    vec4 baseColor;
} p3d_Material;
uniform vec4 p3d_ColorScale;
uniform sampler2D d2s_base_color_texture;
uniform vec3 camera_world_position;
uniform vec3 d2s_preview_ambient_color;
uniform vec3 d2s_preview_light_color;
uniform float d2s_preview_exposure;
varying vec3 v_normal;
varying vec3 v_world_position;
varying vec4 v_color;
varying vec2 v_texcoord;
void main() {
    vec4 texel = texture2D(d2s_base_color_texture, v_texcoord);
    vec4 base_color = p3d_Material.baseColor * v_color * p3d_ColorScale * texel;
    vec3 n = normalize(v_normal);
    vec3 l = normalize(camera_world_position + vec3(0.0, 0.2, 0.0) - v_world_position);
    float diff = max(abs(dot(n, l)), 0.12);
    vec3 linear_color = base_color.rgb * (d2s_preview_ambient_color + d2s_preview_light_color * diff) * d2s_preview_exposure;
    vec3 mapped = linear_color / (linear_color + vec3(1.0));
    gl_FragColor = vec4(mapped, base_color.a);
}
"""
    _PREVIEW_DIFFUSE_SHADER = Shader.make(Shader.SL_GLSL, vertex, fragment)
    return _PREVIEW_DIFFUSE_SHADER

def _make_node_animation_player(asset_path: str, root: Any) -> Any | None:
    from xr_viewer.panda3d_node_animation import (
        GltfNodeAnimationPlayer,
        GltfNodeAnimationRuntime,
    )

    runtime = GltfNodeAnimationRuntime.from_asset(asset_path, root)
    if runtime.channel_count <= 0:
        return None
    return GltfNodeAnimationPlayer(runtime)


def _node_counts(root: Any) -> tuple[int, int]:
    nodes = root.find_all_matches("**")
    geoms = root.find_all_matches("**/+GeomNode")
    return int(nodes.get_num_paths()), int(geoms.get_num_paths())
