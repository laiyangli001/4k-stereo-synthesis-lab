"""Scene graph ownership contracts for the optional Panda3D renderer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class PandaAssetRef:
    role: str
    path: str


@dataclass
class PandaSceneGraph:
    """Import-light holder for future Panda3D NodePath ownership."""

    environment: PandaAssetRef | None = None
    controllers: dict[str, PandaAssetRef] = field(default_factory=dict)
    frame_state: Any | None = None
    released: bool = False

    def load_environment(self, asset_path: str) -> None:
        self._ensure_live()
        self.environment = PandaAssetRef("environment", str(Path(asset_path)))

    def load_controller(self, hand: str, asset_path: str) -> None:
        self._ensure_live()
        key = str(hand).strip().lower()
        if key not in {"left", "right"}:
            raise ValueError("controller hand must be 'left' or 'right'")
        self.controllers[key] = PandaAssetRef(f"controller:{key}", str(Path(asset_path)))

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
        self.frame_state = None
        self.released = True

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError("PandaSceneGraph has been released")
