"""Bridge contracts from Panda3D OpenGL rendering to OpenXR D3D11 images."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SwapchainImageRef:
    eye_index: int
    image_index: int
    texture: Any
    width: int
    height: int
    format: int | str

    def __post_init__(self) -> None:
        if self.eye_index not in (0, 1):
            raise ValueError("eye_index must be 0 or 1")
        if self.image_index < 0:
            raise ValueError("image_index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("swapchain image dimensions must be positive")


@dataclass(frozen=True)
class RenderEyesResult:
    left_rendered: bool
    right_rendered: bool
    bridge_mode: str

    @property
    def rendered(self) -> bool:
        return self.left_rendered and self.right_rendered


class PandaBridge:
    """Abstract bridge placeholder for NV_DX or CUDA implementations."""

    bridge_mode = "unimplemented"

    def render_eyes(
        self,
        *,
        scene: Any,
        targets: Any,
        frame_state: Any,
        left_image: SwapchainImageRef,
        right_image: SwapchainImageRef,
    ) -> RenderEyesResult:
        raise NotImplementedError("PandaBridge.render_eyes is not implemented yet")

    def release(self) -> None:
        return None
