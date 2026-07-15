"""Stereo render target lifecycle contracts for the Panda3D adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


@dataclass(frozen=True)
class StereoTargetRef:
    eye_index: int
    spec: StereoTargetSpec
    created_with_panda: bool = False
    texture_native_id: int = 0
    buffer_name: str = ""

    @property
    def texture_native_id_available(self) -> bool:
        return self.texture_native_id > 0


@dataclass
class StereoTargets:
    create_panda_targets: bool = False
    left: StereoTargetSpec | None = None
    right: StereoTargetSpec | None = None
    left_ref: StereoTargetRef | None = None
    right_ref: StereoTargetRef | None = None
    generation: int = 0
    released: bool = False
    _panda_base: Any | None = field(default=None, init=False, repr=False)
    _panda_buffers: list[Any] = field(default_factory=list, init=False, repr=False)
    _panda_textures: list[Any] = field(default_factory=list, init=False, repr=False)
    _panda_cameras: list[Any] = field(default_factory=list, init=False, repr=False)
    _panda_display_regions: list[Any] = field(default_factory=list, init=False, repr=False)

    @property
    def ready(self) -> bool:
        return self.left is not None and self.right is not None and not self.released

    def rebuild(self, left: StereoTargetSpec, right: StereoTargetSpec) -> None:
        if self.released:
            raise RuntimeError("StereoTargets has been released")
        self._release_panda_handles()
        self.left = left
        self.right = right
        if self.create_panda_targets:
            self.left_ref, self.right_ref = self._create_panda_target_pair(left, right)
        else:
            self.left_ref = StereoTargetRef(0, left)
            self.right_ref = StereoTargetRef(1, right)
        self.generation += 1

    def target_refs(self) -> tuple[StereoTargetRef, StereoTargetRef] | tuple[()]:
        if self.left_ref is None or self.right_ref is None:
            return ()
        return self.left_ref, self.right_ref

    def release(self) -> None:
        self._release_panda_handles()
        self.left = None
        self.right = None
        self.left_ref = None
        self.right_ref = None
        self.released = True

    def _create_panda_target_pair(
        self,
        left: StereoTargetSpec,
        right: StereoTargetSpec,
    ) -> tuple[StereoTargetRef, StereoTargetRef]:
        base = _create_panda_base()
        self._panda_base = base
        left_buffer, left_texture, left_native_id, left_camera, left_region = _create_panda_offscreen_target(base, 0, left)
        right_buffer, right_texture, right_native_id, right_camera, right_region = _create_panda_offscreen_target(base, 1, right)
        base.graphicsEngine.render_frame()
        base.graphicsEngine.render_frame()
        self._panda_buffers.extend([left_buffer, right_buffer])
        self._panda_textures.extend([left_texture, right_texture])
        self._panda_cameras.extend([left_camera, right_camera])
        self._panda_display_regions.extend([left_region, right_region])
        return (
            StereoTargetRef(0, left, True, left_native_id, "d2s-panda-eye-0"),
            StereoTargetRef(1, right, True, right_native_id, "d2s-panda-eye-1"),
        )

    def _release_panda_handles(self) -> None:
        self._panda_buffers.clear()
        self._panda_textures.clear()
        self._panda_cameras.clear()
        self._panda_display_regions.clear()
        if self._panda_base is not None:
            self._panda_base.destroy()
            self._panda_base = None


def _create_panda_base() -> Any:
    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import load_prc_file_data

    load_prc_file_data(
        "d2s-panda-stereo-targets",
        "\n".join(
            [
                "window-type offscreen",
                "load-display pandagl",
                "audio-library-name null",
                "sync-video false",
                "show-frame-rate-meter false",
                "notify-level-display error",
            ]
        ),
    )
    base = ShowBase(windowType="offscreen")
    if not base.win:
        base.destroy()
        raise RuntimeError("Panda3D did not create an offscreen window for stereo targets")
    if not base.win.get_gsg():
        base.destroy()
        raise RuntimeError("Panda3D offscreen stereo targets have no GSG")
    return base


def _create_panda_offscreen_target(
    base: Any,
    eye_index: int,
    spec: StereoTargetSpec,
) -> tuple[Any, Any, int, Any, Any]:
    from panda3d.core import Camera, FrameBufferProperties, GraphicsPipe, PerspectiveLens, Texture, WindowProperties

    gsg = base.win.get_gsg()
    texture = Texture(f"d2s-panda-eye-{eye_index}-color")
    framebuffer = FrameBufferProperties()
    framebuffer.set_rgba_bits(8, 8, 8, 8)
    framebuffer.set_depth_bits(24)
    window_props = WindowProperties.size(spec.width, spec.height)
    buffer = base.graphicsEngine.make_output(
        base.pipe,
        f"d2s-panda-eye-{eye_index}",
        -2,
        framebuffer,
        window_props,
        GraphicsPipe.BFRefuseWindow,
        gsg,
        base.win,
    )
    if not buffer:
        raise RuntimeError("Panda3D did not create an offscreen stereo target")
    render_texture_mode = getattr(
        buffer,
        "RTMBindOrCopy",
        getattr(buffer, "RTMCopyTexture", buffer.RTMCopyRam),
    )
    buffer.add_render_texture(texture, render_texture_mode)
    lens = PerspectiveLens()
    lens.set_near_far(0.01, 1000.0)
    lens.set_fov(90.0, 90.0)
    camera = base.render.attach_new_node(Camera(f"d2s-panda-eye-{eye_index}-camera"))
    camera.node().set_lens(lens)
    display_region = buffer.make_display_region()
    display_region.set_camera(camera)
    texture_context = texture.prepare_now(0, gsg.get_prepared_objects(), gsg)
    native_id = int(texture_context.get_native_id()) if texture_context is not None else 0
    try:
        texture._d2s_native_id = native_id
    except Exception:
        pass
    return buffer, texture, native_id, camera, display_region
