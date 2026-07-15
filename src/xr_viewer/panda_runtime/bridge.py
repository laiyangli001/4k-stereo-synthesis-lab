"""Bridge contracts from Panda3D OpenGL rendering to OpenXR D3D11 images."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PandaBridgeUnavailable(RuntimeError):
    """Raised when no concrete Panda3D-to-OpenXR bridge is active."""


@dataclass(frozen=True)
class SwapchainImageRef:
    eye_index: int
    image_index: int
    texture: Any
    width: int
    height: int
    format: int | str
    session_generation: int = 0

    def __post_init__(self) -> None:
        if self.eye_index not in (0, 1):
            raise ValueError("eye_index must be 0 or 1")
        if self.image_index < 0:
            raise ValueError("image_index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("swapchain image dimensions must be positive")
        if self.session_generation < 0:
            raise ValueError("session_generation must be non-negative")


@dataclass(frozen=True)
class SwapchainResourceKey:
    session_generation: int
    eye_index: int
    image_index: int
    width: int
    height: int
    format: int | str

    @classmethod
    def from_image(cls, image: SwapchainImageRef) -> "SwapchainResourceKey":
        return cls(
            session_generation=image.session_generation,
            eye_index=image.eye_index,
            image_index=image.image_index,
            width=image.width,
            height=image.height,
            format=image.format,
        )


@dataclass(frozen=True)
class CachedSwapchainResource:
    key: SwapchainResourceKey
    bridge_mode: str
    handle: Any | None = None


@dataclass(frozen=True)
class RenderEyesResult:
    left_rendered: bool
    right_rendered: bool
    bridge_mode: str

    @property
    def rendered(self) -> bool:
        return self.left_rendered and self.right_rendered


@dataclass
class PandaBridge:
    """Bridge facade for future NV_DX or CUDA implementations.

    This base class owns the cache-key policy required by the migration plan.
    Concrete subclasses can populate resource handles after the real OpenXR
    swapchain gate is validated.
    """

    bridge_mode: str = "unimplemented"
    resources: dict[SwapchainResourceKey, CachedSwapchainResource] = field(default_factory=dict)
    released: bool = False

    def resource_key(self, image: SwapchainImageRef) -> SwapchainResourceKey:
        return SwapchainResourceKey.from_image(image)

    def ensure_resource(self, image: SwapchainImageRef) -> CachedSwapchainResource:
        if self.released:
            raise PandaBridgeUnavailable("PandaBridge has been released")
        key = self.resource_key(image)
        resource = self.resources.get(key)
        if resource is None:
            resource = CachedSwapchainResource(key=key, bridge_mode=self.bridge_mode)
            self.resources[key] = resource
        return resource

    def invalidate_session(self, session_generation: int) -> None:
        self.resources = {
            key: resource
            for key, resource in self.resources.items()
            if key.session_generation != session_generation
        }

    def render_eyes(
        self,
        *,
        scene: Any,
        targets: Any,
        frame_state: Any,
        left_image: SwapchainImageRef,
        right_image: SwapchainImageRef,
    ) -> RenderEyesResult:
        self.ensure_resource(left_image)
        self.ensure_resource(right_image)
        raise PandaBridgeUnavailable(
            "PandaBridge has no concrete NV_DX or CUDA implementation enabled yet"
        )

    def release(self) -> None:
        self.resources.clear()
        self.released = True
