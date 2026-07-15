"""NV_DX bridge implementation for Panda3D projection handoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .bridge import PandaBridge, PandaBridgeUnavailable, RenderEyesResult, SwapchainImageRef


class PandaNvDxInteropAdapter(Protocol):
    """Viewer-side NV_DX operations reused by the Panda bridge."""

    def get_or_create_fbo(self, image: SwapchainImageRef) -> Any: ...

    def lock(self, image: SwapchainImageRef) -> None: ...

    def unlock(self, image: SwapchainImageRef) -> None: ...


@dataclass
class ViewerPandaNvDxInteropAdapter:
    """Adapter around the existing OpenXR viewer NV_DX interop cache."""

    viewer: Any

    def get_or_create_fbo(self, image: SwapchainImageRef) -> Any:
        creator = getattr(self.viewer, "_get_or_create_nv_interop_fbo", None)
        if not callable(creator):
            raise PandaBridgeUnavailable("viewer has no NV_DX FBO factory")
        fbo, _raw_fbo = creator(
            image.eye_index,
            image.image_index,
            image.texture,
            image.width,
            image.height,
        )
        return fbo

    def lock(self, image: SwapchainImageRef) -> None:
        self._call_lock_state(image, "_lock_panda_nv_dx_image")

    def unlock(self, image: SwapchainImageRef) -> None:
        self._call_lock_state(image, "_unlock_panda_nv_dx_image")

    def _call_lock_state(self, image: SwapchainImageRef, attr: str) -> None:
        fn = getattr(self.viewer, attr, None)
        if not callable(fn):
            raise PandaBridgeUnavailable(f"viewer has no {attr} hook")
        fn(image.eye_index, image.image_index)


class PandaNvDxBridge(PandaBridge):
    """Concrete bridge shell that locks D3D11 swapchains for Panda rendering."""

    def __init__(self, adapter: PandaNvDxInteropAdapter | None = None):
        super().__init__(bridge_mode="nv_dx")
        self.adapter = adapter

    def render_eyes(
        self,
        *,
        scene: Any,
        targets: Any,
        frame_state: Any,
        left_image: SwapchainImageRef,
        right_image: SwapchainImageRef,
    ) -> RenderEyesResult:
        if self.adapter is None:
            raise PandaBridgeUnavailable("PandaNvDxBridge requires an interop adapter")
        left_resource = self.ensure_resource(left_image)
        right_resource = self.ensure_resource(right_image)
        left_fbo = self.adapter.get_or_create_fbo(left_image)
        right_fbo = self.adapter.get_or_create_fbo(right_image)
        locked: list[SwapchainImageRef] = []
        try:
            for image in (left_image, right_image):
                self.adapter.lock(image)
                locked.append(image)
            renderer = getattr(scene, "render_to_framebuffers", None)
            if not callable(renderer):
                raise PandaBridgeUnavailable("Panda scene has no render_to_framebuffers hook")
            renderer(
                targets=targets,
                frame_state=frame_state,
                left_framebuffer=left_fbo,
                right_framebuffer=right_fbo,
                left_resource=left_resource,
                right_resource=right_resource,
            )
        finally:
            for image in reversed(locked):
                self.adapter.unlock(image)
        return RenderEyesResult(True, True, self.bridge_mode)
