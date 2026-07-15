"""Scene graph ownership contracts for the optional Panda3D renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


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
class PandaSceneSnapshot:
    frame_index: int | None = None
    animation_time_seconds: float | None = None
    animation_sample_count: int = 0
    animation_applied_player_count: int = 0
    animation_player_count: int = 0
    animation_channel_count: int = 0
    animation_bound_node_count: int = 0
    controller_hands: tuple[str, ...] = ()
    controller_ray_hands: tuple[str, ...] = ()
    applied_controller_ray_hands: tuple[str, ...] = ()
    screen_pose_present: bool = False
    screen_texture_present: bool = False
    screen_texture_applied: bool = False
    screen_texture_width: int = 0
    screen_texture_height: int = 0
    screen_texture_format: str = ""
    screen_texture_native_id_available: bool = False
    eye_view_count: int = 0
    applied_controller_hands: tuple[str, ...] = ()
    screen_pose_applied: bool = False


@dataclass
class PandaSceneGraph:
    """Owns glTF asset roots without exposing Panda NodePath to callers."""

    load_panda_assets: bool = False
    environment: PandaAssetRef | None = None
    controllers: dict[str, PandaAssetRef] = field(default_factory=dict)
    frame_state: Any | None = None
    snapshot: PandaSceneSnapshot = field(default_factory=PandaSceneSnapshot)
    released: bool = False
    _environment_root: Any | None = field(default=None, init=False, repr=False)
    _controller_roots: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _controller_ray_targets: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _screen_root: Any | None = field(default=None, init=False, repr=False)
    _screen_texture_target: Any | None = field(default=None, init=False, repr=False)
    _environment_animation_player: Any | None = field(default=None, init=False, repr=False)
    _controller_animation_players: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _animation_loop: bool = field(default=True, init=False, repr=False)
    _animation_sample_count: int = field(default=0, init=False, repr=False)
    _last_animation_time_seconds: float | None = field(default=None, init=False, repr=False)
    _last_render_base: Any | None = field(default=None, init=False, repr=False)

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        asset, root, animation_player = self._make_asset_ref("environment", asset_path)
        self.environment = asset
        self._environment_root = root
        self._environment_animation_player = animation_player
        _set_animation_player_loop(animation_player, self._animation_loop)

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
        applied_controller_ray_hands = self._apply_controller_rays(frame_state)
        screen_pose_applied = self._apply_screen_pose(frame_state)
        screen_texture_applied = self._apply_screen_texture(frame_state)
        animation_time = getattr(frame_state, "animation_time_seconds", None)
        animation_time_seconds = None if animation_time is None else float(animation_time)
        animation_applied_player_count = 0
        if animation_time_seconds is not None:
            animation_applied_player_count = self._apply_animation_time(animation_time_seconds)
        self.snapshot = _snapshot_from_frame_state(
            frame_state,
            applied_controller_hands,
            applied_controller_ray_hands,
            screen_pose_applied,
            screen_texture_applied,
            animation_time_seconds=animation_time_seconds,
            animation_applied_player_count=animation_applied_player_count,
            animation_player_count=self._animation_player_count(),
            animation_channel_count=self._animation_channel_count(),
            animation_bound_node_count=self._animation_bound_node_count(),
            animation_sample_count=self._animation_sample_count,
        )

    def loaded_assets(self) -> tuple[PandaAssetRef, ...]:
        assets = []
        if self.environment is not None:
            assets.append(self.environment)
        assets.extend(self.controllers[key] for key in sorted(self.controllers))
        return tuple(assets)

    def controller_paths(self) -> Mapping[str, str]:
        return {hand: asset.path for hand, asset in self.controllers.items()}

    def attach_controller_ray_target(self, hand: str, target: Any) -> None:
        self._ensure_live()
        key = str(hand).strip().lower()
        if key not in {"left", "right"}:
            raise ValueError("controller ray hand must be 'left' or 'right'")
        self._controller_ray_targets[key] = target

    def attach_screen_root(self, root: Any) -> None:
        self._ensure_live()
        self._screen_root = root

    def attach_screen_texture_target(self, target: Any) -> None:
        self._ensure_live()
        self._screen_texture_target = target

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
    ) -> None:
        self._ensure_live()
        base = getattr(targets, "_panda_base", None)
        if base is None:
            raise RuntimeError("Panda stereo targets have no ShowBase")
        if not getattr(targets, "_panda_textures", None):
            raise RuntimeError("Panda stereo targets have no color textures")
        if len(targets._panda_textures) < 2:
            raise RuntimeError("Panda stereo targets must contain both eye textures")
        self._attach_roots_to_base(base)
        _update_eye_cameras(getattr(targets, "_panda_cameras", ()), frame_state)
        base.graphicsEngine.render_frame()
        base.graphicsEngine.render_frame()
        _blit_panda_texture_to_framebuffer(
            targets._panda_textures[0],
            left_framebuffer,
            _resource_width(left_resource) or _target_width(targets, 0),
            _resource_height(left_resource) or _target_height(targets, 0),
        )
        _blit_panda_texture_to_framebuffer(
            targets._panda_textures[1],
            right_framebuffer,
            _resource_width(right_resource) or _target_width(targets, 1),
            _resource_height(right_resource) or _target_height(targets, 1),
        )

    def release(self) -> None:
        self.environment = None
        self.controllers.clear()
        self._environment_root = None
        self._controller_roots.clear()
        self._controller_ray_targets.clear()
        self._screen_root = None
        self._screen_texture_target = None
        self._environment_animation_player = None
        self._controller_animation_players.clear()
        self._animation_loop = True
        self._animation_sample_count = 0
        self._last_animation_time_seconds = None
        self._last_render_base = None
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

    def _apply_controller_rays(self, frame_state: Any) -> tuple[str, ...]:
        controller_rays = getattr(frame_state, "controller_rays", {}) or {}
        applied: list[str] = []
        for hand, target in sorted(self._controller_ray_targets.items()):
            ray = controller_rays.get(hand)
            if ray is None:
                continue
            if _apply_controller_ray_to_target(target, ray):
                applied.append(hand)
        return tuple(applied)

    def _apply_screen_pose(self, frame_state: Any) -> bool:
        if self._screen_root is None:
            return False
        screen_pose = getattr(frame_state, "screen_pose", None)
        if screen_pose is None:
            return False
        return _apply_pose_to_node_path(self._screen_root, screen_pose)

    def _apply_screen_texture(self, frame_state: Any) -> bool:
        if self._screen_texture_target is None:
            return False
        screen_texture = getattr(frame_state, "screen_texture", None)
        if screen_texture is None:
            return False
        if hasattr(self._screen_texture_target, "set_screen_texture"):
            self._screen_texture_target.set_screen_texture(screen_texture)
            return True
        if hasattr(self._screen_texture_target, "set_texture"):
            self._screen_texture_target.set_texture(screen_texture)
            return True
        return False

    def _attach_roots_to_base(self, base: Any) -> None:
        if self._last_render_base is base:
            return
        render_root = getattr(base, "render", None)
        if render_root is None:
            raise RuntimeError("Panda ShowBase has no render root")
        for root in _iter_scene_roots(
            self._environment_root,
            self._controller_roots.values(),
            self._screen_root,
            self._controller_ray_targets.values(),
        ):
            if hasattr(root, "reparent_to"):
                root.reparent_to(render_root)
        self._last_render_base = base

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


def _snapshot_from_frame_state(
    frame_state: Any,
    applied_controller_hands: tuple[str, ...] = (),
    applied_controller_ray_hands: tuple[str, ...] = (),
    screen_pose_applied: bool = False,
    screen_texture_applied: bool = False,
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
    controller_rays = getattr(frame_state, "controller_rays", {}) or {}
    screen_texture = getattr(frame_state, "screen_texture", None)
    return PandaSceneSnapshot(
        frame_index=getattr(frame_state, "frame_index", None),
        animation_time_seconds=animation_time_seconds,
        animation_sample_count=int(animation_sample_count),
        animation_applied_player_count=int(animation_applied_player_count),
        animation_player_count=int(animation_player_count),
        animation_channel_count=int(animation_channel_count),
        animation_bound_node_count=int(animation_bound_node_count),
        controller_hands=tuple(sorted(str(hand) for hand in controller_poses)),
        controller_ray_hands=tuple(sorted(str(hand) for hand in controller_rays)),
        applied_controller_ray_hands=applied_controller_ray_hands,
        screen_pose_present=getattr(frame_state, "screen_pose", None) is not None,
        screen_texture_present=screen_texture is not None,
        screen_texture_applied=screen_texture_applied,
        screen_texture_width=int(getattr(screen_texture, "width", 0) or 0),
        screen_texture_height=int(getattr(screen_texture, "height", 0) or 0),
        screen_texture_format=str(getattr(screen_texture, "format", "") or ""),
        screen_texture_native_id_available=bool(
            getattr(screen_texture, "native_id_available", False)
        ),
        eye_view_count=sum(1 for eye_view in eye_views if eye_view is not None),
        applied_controller_hands=applied_controller_hands,
        screen_pose_applied=screen_pose_applied,
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
    screen_root: Any | None,
    controller_ray_targets: Any,
) -> tuple[Any, ...]:
    roots: list[Any] = []
    if environment_root is not None:
        roots.append(environment_root)
    roots.extend(root for root in controller_roots if root is not None)
    if screen_root is not None:
        roots.append(screen_root)
    for target in controller_ray_targets:
        ray_node = getattr(target, "ray_node", None)
        if ray_node is not None:
            roots.append(ray_node)
    return tuple(roots)


def _framebuffer_id(framebuffer: Any) -> int:
    value = getattr(framebuffer, "glo", framebuffer)
    value = getattr(value, "value", value)
    framebuffer_id = int(value or 0)
    if framebuffer_id <= 0:
        raise RuntimeError("target framebuffer has no OpenGL id")
    return framebuffer_id


def _panda_texture_native_id(texture: Any) -> int:
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


def _fov_value(fov: Any, attr_name: str, mapping_name: str) -> float | None:
    if isinstance(fov, Mapping):
        value = fov.get(attr_name, fov.get(mapping_name))
    else:
        value = getattr(fov, attr_name, None)
    return None if value is None else float(value)


def _apply_controller_ray_to_target(target: Any, ray: Any) -> bool:
    if hasattr(target, "set_controller_ray"):
        target.set_controller_ray(ray)
        return True
    if hasattr(target, "set_ray"):
        target.set_ray(ray)
        return True
    return False


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


def _load_panda_root(asset_path: str) -> Any:
    import gltf
    from panda3d.core import NodePath

    return NodePath(gltf.load_model(str(asset_path)))


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
