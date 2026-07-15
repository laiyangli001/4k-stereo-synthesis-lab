"""Panda3D screen NodePath and texture target helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .screen_texture import PandaScreenTextureUploadTarget


@dataclass
class PandaScreenNodeTarget:
    """Owns a Panda screen quad and its screen texture upload target."""

    root: Any
    texture_target: PandaScreenTextureUploadTarget
    width: float
    height: float


def create_panda_screen_node_target(width: float, height: float) -> PandaScreenNodeTarget:
    """Create a Panda screen quad centered on its local origin."""
    screen_width = float(width)
    screen_height = float(height)
    if screen_width <= 0.0 or screen_height <= 0.0:
        raise ValueError("screen dimensions must be positive")

    from panda3d.core import CardMaker, NodePath

    maker = CardMaker("d2s-screen-card")
    maker.set_frame(
        -screen_width / 2.0,
        screen_width / 2.0,
        -screen_height / 2.0,
        screen_height / 2.0,
    )
    root = NodePath("d2s-screen-root")
    card = root.attach_new_node(maker.generate())
    card.set_two_sided(True)
    return PandaScreenNodeTarget(
        root=root,
        texture_target=PandaScreenTextureUploadTarget(card),
        width=screen_width,
        height=screen_height,
    )
