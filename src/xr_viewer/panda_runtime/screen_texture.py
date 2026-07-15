"""Panda3D screen texture upload target for the optional renderer path."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class PandaScreenTextureUploadError(RuntimeError):
    """Raised when a screen texture frame cannot be uploaded to Panda."""


@dataclass
class PandaScreenTextureUploadTarget:
    """Bind latest screen texture frames to a Panda screen NodePath."""

    node_path: Any
    texture_name: str = "d2s-panda-screen"
    sort_priority: int = 1
    texture: Any | None = field(default=None, init=False)
    last_frame_index: int | None = field(default=None, init=False)

    def set_screen_texture(self, screen_texture: Any) -> Any:
        width = int(getattr(screen_texture, "width", 0) or 0)
        height = int(getattr(screen_texture, "height", 0) or 0)
        fmt = str(getattr(screen_texture, "format", "rgba8") or "rgba8").lower()
        if width <= 0 or height <= 0:
            raise PandaScreenTextureUploadError("screen texture dimensions must be positive")
        if fmt not in {"rgba", "rgba8"}:
            raise PandaScreenTextureUploadError(f"unsupported Panda screen texture format: {fmt}")

        texture = self._ensure_texture(width, height)
        payload = getattr(screen_texture, "payload", None)
        if payload is not None:
            data = _payload_to_bytes(payload)
            expected = width * height * 4
            if len(data) != expected:
                raise PandaScreenTextureUploadError(
                    f"screen texture payload byte length {len(data)} does not match expected {expected}"
                )
            texture.set_ram_image(data)

        if not hasattr(self.node_path, "set_texture"):
            raise PandaScreenTextureUploadError("screen texture target node has no set_texture method")
        self.node_path.set_texture(texture, self.sort_priority)
        self.last_frame_index = getattr(screen_texture, "frame_index", None)
        return texture

    def _ensure_texture(self, width: int, height: int) -> Any:
        from panda3d.core import Texture

        if self.texture is None:
            self.texture = Texture(self.texture_name)
        if int(self.texture.get_x_size()) != width or int(self.texture.get_y_size()) != height:
            self.texture.setup_2d_texture(
                width,
                height,
                Texture.T_unsigned_byte,
                Texture.F_rgba8,
            )
        return self.texture


def _payload_to_bytes(payload: Any) -> bytes:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, memoryview):
        return payload.tobytes()
    tobytes = getattr(payload, "tobytes", None)
    if callable(tobytes):
        return tobytes()
    raise PandaScreenTextureUploadError("screen texture payload must be bytes-like")
