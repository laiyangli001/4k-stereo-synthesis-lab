"""Panda3D controller ray geometry target for the optional renderer path."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any


class PandaControllerRayGeometryError(RuntimeError):
    """Raised when a controller ray cannot be represented as Panda geometry."""


@dataclass
class PandaControllerRayGeometryTarget:
    """Render the latest controller ray snapshot as Panda LineSegs geometry."""

    parent_node: Any
    name: str = "d2s-panda-controller-ray"
    color: tuple[float, float, float, float] = (0.2, 0.6, 1.0, 0.85)
    thickness: float = 2.0
    ray_node: Any | None = field(default=None, init=False)
    last_ray: Any | None = field(default=None, init=False)

    def set_controller_ray(self, controller_ray: Any) -> Any | None:
        self._clear_ray_node()
        self.last_ray = controller_ray
        if not bool(getattr(controller_ray, "visible", True)):
            return None
        if not hasattr(self.parent_node, "attach_new_node"):
            raise PandaControllerRayGeometryError("controller ray parent has no attach_new_node method")

        origin = _vector3(getattr(controller_ray, "origin", None), "origin")
        direction = _normalize(_vector3(getattr(controller_ray, "direction", None), "direction"))
        length = float(getattr(controller_ray, "length", 0.0) or 0.0)
        if length <= 0.0:
            raise PandaControllerRayGeometryError("controller ray length must be positive")
        end = tuple(origin[index] + direction[index] * length for index in range(3))

        from panda3d.core import LineSegs

        lines = LineSegs(self.name)
        lines.set_thickness(float(self.thickness))
        lines.set_color(*self.color)
        lines.move_to(*origin)
        lines.draw_to(*end)
        self.ray_node = self.parent_node.attach_new_node(lines.create())
        return self.ray_node

    def _clear_ray_node(self) -> None:
        if self.ray_node is not None:
            self.ray_node.remove_node()
            self.ray_node = None


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if value is None or len(value) != 3:
        raise PandaControllerRayGeometryError(f"controller ray {name} must be a 3D vector")
    return tuple(float(component) for component in value)


def _normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1e-8:
        raise PandaControllerRayGeometryError("controller ray direction must be non-zero")
    return tuple(component / length for component in value)
