"""Phase-0 Panda3D OpenGL offscreen rendering diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from typing import Any


@dataclass(frozen=True)
class Panda3DOffscreenProbeReport:
    window_created: bool
    window_width: int
    window_height: int
    buffer_created: bool
    buffer_width: int
    buffer_height: int
    texture_width: int
    texture_height: int
    texture_format: int
    texture_component_type: int
    texture_has_ram_image: bool
    framebuffer_properties: str
    driver_vendor: str
    driver_renderer: str
    driver_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Panda3DOffscreenProbeError(RuntimeError):
    """Raised when Panda3D offscreen rendering cannot be initialized."""


def panda3d_offscreen_probe_available() -> bool:
    return bool(importlib.util.find_spec("panda3d")) and bool(
        importlib.util.find_spec("direct.showbase.ShowBase")
    )


def inspect_panda3d_offscreen(width: int = 64, height: int = 64) -> Panda3DOffscreenProbeReport:
    if width <= 0 or height <= 0:
        raise Panda3DOffscreenProbeError("Offscreen probe dimensions must be positive")
    if not panda3d_offscreen_probe_available():
        raise Panda3DOffscreenProbeError("Panda3D is unavailable")

    try:
        from direct.showbase.ShowBase import ShowBase
        from panda3d.core import (
            FrameBufferProperties,
            GraphicsPipe,
            Texture,
            WindowProperties,
            load_prc_file_data,
        )
    except ImportError as exc:  # pragma: no cover - guarded above for diagnostics
        raise Panda3DOffscreenProbeError("Panda3D offscreen imports failed") from exc

    load_prc_file_data(
        "d2s-panda-offscreen-probe",
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

    base: Any | None = None
    try:
        base = ShowBase(windowType="offscreen")
        if not base.win:
            raise Panda3DOffscreenProbeError("Panda3D did not create an offscreen window")
        gsg = base.win.get_gsg()
        if not gsg:
            raise Panda3DOffscreenProbeError("Panda3D offscreen window has no GSG")

        texture = Texture("d2s-offscreen-probe-color")
        framebuffer = FrameBufferProperties()
        framebuffer.set_rgba_bits(8, 8, 8, 8)
        framebuffer.set_depth_bits(24)
        window_props = WindowProperties.size(width, height)
        buffer = base.graphicsEngine.make_output(
            base.pipe,
            "d2s-offscreen-probe-buffer",
            -2,
            framebuffer,
            window_props,
            GraphicsPipe.BFRefuseWindow,
            gsg,
            base.win,
        )
        if not buffer:
            raise Panda3DOffscreenProbeError("Panda3D did not create an offscreen buffer")
        buffer.add_render_texture(texture, buffer.RTMCopyRam)
        base.graphicsEngine.render_frame()

        fb_props = buffer.get_fb_properties()
        return Panda3DOffscreenProbeReport(
            window_created=True,
            window_width=base.win.get_x_size(),
            window_height=base.win.get_y_size(),
            buffer_created=True,
            buffer_width=buffer.get_x_size(),
            buffer_height=buffer.get_y_size(),
            texture_width=texture.get_x_size(),
            texture_height=texture.get_y_size(),
            texture_format=int(texture.get_format()),
            texture_component_type=int(texture.get_component_type()),
            texture_has_ram_image=texture.has_ram_image(),
            framebuffer_properties=str(fb_props),
            driver_vendor=str(gsg.get_driver_vendor()),
            driver_renderer=str(gsg.get_driver_renderer()),
            driver_version=str(gsg.get_driver_version()),
        )
    except Panda3DOffscreenProbeError:
        raise
    except Exception as exc:
        raise Panda3DOffscreenProbeError(f"Panda3D offscreen probe failed: {exc}") from exc
    finally:
        if base is not None:
            base.destroy()


def offscreen_report_as_json(report: Panda3DOffscreenProbeReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
