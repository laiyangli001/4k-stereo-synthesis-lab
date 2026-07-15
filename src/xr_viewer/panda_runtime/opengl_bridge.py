"""OpenGL bridge from Panda3D offscreen rendering to OpenXR OpenGL FBOs."""

from __future__ import annotations

from typing import Any

from .bridge import PandaBridge, PandaBridgeUnavailable, RenderEyesResult, SwapchainImageRef


class PandaOpenGLBridge(PandaBridge):
    """Zero-copy OpenGL bridge into already-acquired OpenXR GL framebuffers."""

    def __init__(self, make_target_context_current: Any | None = None) -> None:
        super().__init__(bridge_mode="opengl")
        self.make_target_context_current = make_target_context_current

    def render_eyes(
        self,
        *,
        scene: Any,
        targets: Any,
        frame_state: Any,
        left_image: SwapchainImageRef,
        right_image: SwapchainImageRef,
    ) -> RenderEyesResult:
        left_resource = self.ensure_resource(left_image)
        right_resource = self.ensure_resource(right_image)
        renderer = getattr(scene, "render_to_framebuffers", None)
        if not callable(renderer):
            raise PandaBridgeUnavailable("Panda scene has no render_to_framebuffers hook")
        if not callable(self.make_target_context_current):
            raise PandaBridgeUnavailable(
                "Panda OpenGL fallback requires a target OpenXR GL context callback"
            )
        renderer(
            targets=targets,
            frame_state=frame_state,
            left_framebuffer=left_image.texture,
            right_framebuffer=right_image.texture,
            left_resource=left_resource,
            right_resource=right_resource,
            make_target_context_current=self.make_target_context_current,
            require_shared_context=True,
        )
        return RenderEyesResult(True, True, self.bridge_mode)