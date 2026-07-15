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


@dataclass
class PandaSceneGraph:
    """Owns glTF asset roots without exposing Panda NodePath to callers."""

    load_panda_assets: bool = False
    environment: PandaAssetRef | None = None
    controllers: dict[str, PandaAssetRef] = field(default_factory=dict)
    frame_state: Any | None = None
    released: bool = False
    _environment_root: Any | None = field(default=None, init=False, repr=False)
    _controller_roots: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        asset, root = self._make_asset_ref("environment", asset_path)
        self.environment = asset
        self._environment_root = root

    def load_controller(self, hand: str, asset_path: str) -> None:
        self._ensure_live()
        key = str(hand).strip().lower()
        if key not in {"left", "right"}:
            raise ValueError("controller hand must be 'left' or 'right'")
        asset, root = self._make_asset_ref(f"controller:{key}", asset_path)
        self.controllers[key] = asset
        if root is not None:
            self._controller_roots[key] = root
        else:
            self._controller_roots.pop(key, None)

    def update_frame_state(self, frame_state: Any) -> None:
        self._ensure_live()
        self.frame_state = frame_state

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
        self.frame_state = None
        self.released = True

    def _make_asset_ref(self, role: str, asset_path: str) -> tuple[PandaAssetRef, Any | None]:
        path = str(Path(asset_path))
        if not self.load_panda_assets:
            return PandaAssetRef(role, path), None
        root = _load_panda_root(path)
        node_count, geom_count = _node_counts(root)
        return PandaAssetRef(role, path, True, node_count, geom_count), root

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError("PandaSceneGraph has been released")


def _load_panda_root(asset_path: str) -> Any:
    import gltf
    from panda3d.core import NodePath

    return NodePath(gltf.load_model(str(asset_path)))


def _node_counts(root: Any) -> tuple[int, int]:
    nodes = root.find_all_matches("**")
    geoms = root.find_all_matches("**/+GeomNode")
    return int(nodes.get_num_paths()), int(geoms.get_num_paths())
