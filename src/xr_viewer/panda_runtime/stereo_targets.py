"""Stereo render target lifecycle contracts for the Panda3D adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StereoTargetSpec:
    width: int
    height: int
    format: int | str
    sample_count: int = 1

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("stereo target dimensions must be positive")
        if self.sample_count <= 0:
            raise ValueError("stereo target sample_count must be positive")


@dataclass
class StereoTargets:
    left: StereoTargetSpec | None = None
    right: StereoTargetSpec | None = None
    generation: int = 0
    released: bool = False

    @property
    def ready(self) -> bool:
        return self.left is not None and self.right is not None and not self.released

    def rebuild(self, left: StereoTargetSpec, right: StereoTargetSpec) -> None:
        if self.released:
            raise RuntimeError("StereoTargets has been released")
        self.left = left
        self.right = right
        self.generation += 1

    def release(self) -> None:
        self.left = None
        self.right = None
        self.released = True
