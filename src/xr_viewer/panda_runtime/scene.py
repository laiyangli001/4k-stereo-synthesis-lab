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
    controller_hands: tuple[str, ...] = ()
    screen_pose_present: bool = False
    screen_texture_present: bool = False
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
    _environment_root: Any | None = field(default=None, init=False, repr=False)
    _controller_roots: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _environment_animation_player: Any | None = field(default=None, init=False, repr=False)
    _controller_animation_players: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        asset, root, animation_player = self._make_asset_ref("environment", asset_path)
        self.environment = asset
        self._environment_root = root
        self._environment_animation_player = animation_player

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
            self._controller_animation_players[key] = animation_player
        else:
            self._controller_animation_players.pop(key, None)

    def update_frame_state(self, frame_state: Any) -> None:
        self._ensure_live()
        self.frame_state = frame_state
        applied_controller_hands = self._apply_controller_poses(frame_state)
        self.snapshot = _snapshot_from_frame_state(frame_state, applied_controller_hands)
        animation_time = getattr(frame_state, "animation_time_seconds", None)
        if animation_time is not None:
            self._apply_animation_time(float(animation_time))

    def loaded_assets(self) -> tuple[PandaAssetRef, ...]:
        assets = []
        if self.environment is not None:
            assets.append(self.environment)
        assets.extend(self.controllers[key] for key in sorted(self.controllers))
        return tuple(assets)

    def controller_paths(self) -> Mapping[str, str]:
        return {hand: asset.path for hand, asset in self.controllers.items()}

    def release(self) -> None:
        self.environment = None
        self.controllers.clear()
        self._environment_root = None
        self._controller_roots.clear()
        self._environment_animation_player = None
        self._controller_animation_players.clear()
        self.frame_state = None
        self.snapshot = PandaSceneSnapshot()
        self.released = True

    def _apply_animation_time(self, time_seconds: float) -> None:
        if self._environment_animation_player is not None:
            self._environment_animation_player.set_time_seconds(time_seconds)
        for player in self._controller_animation_players.values():
            player.set_time_seconds(time_seconds)

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
) -> PandaSceneSnapshot:
    eye_views = getattr(frame_state, "eye_views", ()) or ()
    controller_poses = getattr(frame_state, "controller_poses", {}) or {}
    return PandaSceneSnapshot(
        frame_index=getattr(frame_state, "frame_index", None),
        controller_hands=tuple(sorted(str(hand) for hand in controller_poses)),
        screen_pose_present=getattr(frame_state, "screen_pose", None) is not None,
        screen_texture_present=getattr(frame_state, "screen_texture", None) is not None,
        eye_view_count=sum(1 for eye_view in eye_views if eye_view is not None),
        applied_controller_hands=applied_controller_hands,
    )


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
