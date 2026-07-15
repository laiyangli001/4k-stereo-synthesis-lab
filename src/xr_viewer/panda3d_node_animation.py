"""glTF node transform animation runtime for Panda3D-loaded assets."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import math
import struct
from typing import Any

from pygltflib import GLTF2


_COMPONENT_FORMATS = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_SUPPORTED_TARGET_PATHS = {"translation", "rotation", "scale"}


class GltfNodeAnimationError(RuntimeError):
    """Raised when a glTF node animation cannot be sampled safely."""


@dataclass(frozen=True)
class GltfNodeAnimationChannel:
    target_node: int
    target_name: str
    path: str
    times: tuple[float, ...]
    values: tuple[tuple[float, ...], ...]
    interpolation: str


def _accessor_values(gltf: GLTF2, accessor_index: int) -> tuple[tuple[float, ...], ...]:
    accessors = gltf.accessors or []
    buffer_views = gltf.bufferViews or []
    if accessor_index < 0 or accessor_index >= len(accessors):
        raise GltfNodeAnimationError(f"Accessor index is out of range: {accessor_index}")
    accessor = accessors[accessor_index]
    if accessor.sparse:
        raise GltfNodeAnimationError("Sparse animation accessors are not supported yet")
    if accessor.bufferView is None:
        raise GltfNodeAnimationError("Animation accessor has no bufferView")
    if accessor.bufferView < 0 or accessor.bufferView >= len(buffer_views):
        raise GltfNodeAnimationError(f"BufferView index is out of range: {accessor.bufferView}")

    buffer_view = buffer_views[accessor.bufferView]
    component_format = _COMPONENT_FORMATS.get(accessor.componentType)
    component_count = _TYPE_COMPONENTS.get(accessor.type or "")
    if component_format is None or component_count is None:
        raise GltfNodeAnimationError(
            f"Unsupported accessor format: componentType={accessor.componentType}, type={accessor.type}"
        )

    binary_blob = gltf.binary_blob()
    if binary_blob is None:
        raise GltfNodeAnimationError("GLB binary blob is missing")

    component_size = struct.calcsize("<" + component_format)
    item_size = component_size * component_count
    stride = buffer_view.byteStride or item_size
    if stride < item_size:
        raise GltfNodeAnimationError("Animation accessor byteStride is smaller than item size")

    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = int(accessor.count or 0)
    values: list[tuple[float, ...]] = []
    unpacker = struct.Struct("<" + component_format * component_count)
    for index in range(count):
        offset = start + index * stride
        chunk = binary_blob[offset : offset + item_size]
        if len(chunk) != item_size:
            raise GltfNodeAnimationError("Animation accessor exceeds GLB binary data")
        values.append(tuple(float(value) for value in unpacker.unpack(chunk)))
    return tuple(values)


def _load_animation_channels(gltf: GLTF2) -> tuple[GltfNodeAnimationChannel, ...]:
    channels: list[GltfNodeAnimationChannel] = []
    nodes = gltf.nodes or []
    for animation in gltf.animations or []:
        samplers = animation.samplers or []
        for channel in animation.channels or []:
            target = channel.target
            if target is None or target.node is None or target.path is None:
                continue
            target_node = int(target.node)
            if target_node < 0 or target_node >= len(nodes):
                raise GltfNodeAnimationError(f"Animation target node is out of range: {target_node}")
            if target.path not in _SUPPORTED_TARGET_PATHS:
                raise GltfNodeAnimationError(f"Unsupported animation target path: {target.path}")
            if channel.sampler is None or channel.sampler < 0 or channel.sampler >= len(samplers):
                raise GltfNodeAnimationError(f"Animation sampler is out of range: {channel.sampler}")

            sampler = samplers[channel.sampler]
            times = tuple(value[0] for value in _accessor_values(gltf, sampler.input))
            values = _accessor_values(gltf, sampler.output)
            interpolation = sampler.interpolation or "LINEAR"
            if interpolation == "CUBICSPLINE":
                raise GltfNodeAnimationError("CUBICSPLINE node animation is not supported yet")
            if interpolation not in {"LINEAR", "STEP"}:
                raise GltfNodeAnimationError(f"Unsupported animation interpolation: {interpolation}")
            if len(times) != len(values):
                raise GltfNodeAnimationError("Animation input/output key counts do not match")
            channels.append(
                GltfNodeAnimationChannel(
                    target_node=target_node,
                    target_name=nodes[target_node].name or f"node{target_node}",
                    path=target.path,
                    times=times,
                    values=values,
                    interpolation=interpolation,
                )
            )
    return tuple(channels)


def _slerp_quaternion(
    start: tuple[float, ...],
    end: tuple[float, ...],
    factor: float,
) -> tuple[float, ...]:
    dot = sum(a * b for a, b in zip(start, end))
    if dot < 0.0:
        end = tuple(-value for value in end)
        dot = -dot
    if dot > 0.9995:
        values = tuple(a + (b - a) * factor for a, b in zip(start, end))
    else:
        theta_0 = math.acos(max(-1.0, min(1.0, dot)))
        sin_theta_0 = math.sin(theta_0)
        theta = theta_0 * factor
        scale_start = math.sin(theta_0 - theta) / sin_theta_0
        scale_end = math.sin(theta) / sin_theta_0
        values = tuple(a * scale_start + b * scale_end for a, b in zip(start, end))
    length = math.sqrt(sum(value * value for value in values))
    if length <= 0.0:
        return start
    return tuple(value / length for value in values)


def _sample_values(
    times: tuple[float, ...],
    values: tuple[tuple[float, ...], ...],
    interpolation: str,
    time_seconds: float,
    *,
    rotation: bool = False,
) -> tuple[float, ...]:
    if not times:
        raise GltfNodeAnimationError("Animation channel has no input times")
    if len(times) == 1 or time_seconds <= times[0]:
        return values[0]
    if time_seconds >= times[-1]:
        return values[-1]

    next_index = bisect_right(times, time_seconds)
    last_index = next_index - 1
    if interpolation == "STEP":
        return values[last_index]

    last_time = times[last_index]
    next_time = times[next_index]
    if next_time <= last_time:
        return values[last_index]
    factor = (time_seconds - last_time) / (next_time - last_time)
    if rotation:
        return _slerp_quaternion(values[last_index], values[next_index], factor)
    return tuple(
        last + (next_value - last) * factor
        for last, next_value in zip(values[last_index], values[next_index])
    )


def _node_default_trs(gltf: GLTF2, node_id: int) -> dict[str, tuple[float, ...]]:
    node = (gltf.nodes or [])[node_id]
    if node.matrix is not None:
        raise GltfNodeAnimationError(
            f"Animated node {node.name or node_id} uses matrix defaults; TRS defaults are required"
        )
    return {
        "translation": tuple(float(value) for value in (node.translation or (0.0, 0.0, 0.0))),
        "rotation": tuple(float(value) for value in (node.rotation or (0.0, 0.0, 0.0, 1.0))),
        "scale": tuple(float(value) for value in (node.scale or (1.0, 1.0, 1.0))),
    }


def _select_transform_node(candidates: Any) -> Any | None:
    if candidates.get_num_paths() <= 0:
        return None
    for node_path in candidates:
        if node_path.node().get_type().get_name() != "GeomNode" and node_path.get_num_children() > 0:
            return node_path
    for node_path in candidates:
        if node_path.node().get_type().get_name() != "GeomNode":
            return node_path
    return candidates[0]


class GltfNodeAnimationRuntime:
    """Sample glTF node TRS animations onto Panda3D NodePaths."""

    def __init__(self, gltf: GLTF2, panda_root: Any):
        self._gltf = gltf
        self._channels = _load_animation_channels(gltf)
        self._target_ids = tuple(sorted({channel.target_node for channel in self._channels}))
        self._defaults = {
            target_node: _node_default_trs(gltf, target_node) for target_node in self._target_ids
        }
        self._bound_nodes = self._bind_nodes(panda_root)
        self.duration_seconds = max(
            (channel.times[-1] for channel in self._channels if channel.times),
            default=0.0,
        )

    @classmethod
    def from_asset(cls, asset_path: str | Path, panda_root: Any) -> "GltfNodeAnimationRuntime":
        return cls(GLTF2().load(str(Path(asset_path))), panda_root)

    @property
    def channel_count(self) -> int:
        return len(self._channels)

    @property
    def target_node_count(self) -> int:
        return len(self._target_ids)

    @property
    def target_node_ids(self) -> tuple[int, ...]:
        return self._target_ids

    @property
    def bound_node_count(self) -> int:
        return len(self._bound_nodes)

    def get_bound_node_path(self, target_node: int) -> Any | None:
        return self._bound_nodes.get(target_node)

    def _bind_nodes(self, panda_root: Any) -> dict[int, Any]:
        nodes = self._gltf.nodes or []
        bound: dict[int, Any] = {}
        for target_node in self._target_ids:
            name = nodes[target_node].name or f"node{target_node}"
            candidates = panda_root.find_all_matches("**/" + name)
            node_path = _select_transform_node(candidates)
            if node_path is not None:
                bound[target_node] = node_path
        return bound

    def _sample_node_trs(self, target_node: int, time_seconds: float) -> dict[str, tuple[float, ...]]:
        trs = dict(self._defaults[target_node])
        for channel in self._channels:
            if channel.target_node != target_node:
                continue
            trs[channel.path] = _sample_values(
                channel.times,
                channel.values,
                channel.interpolation,
                time_seconds,
                rotation=channel.path == "rotation",
            )
        return trs

    def apply_sample(self, time_seconds: float, *, loop: bool = True) -> None:
        if loop and self.duration_seconds > 0.0:
            time_seconds = time_seconds % self.duration_seconds

        from panda3d.core import (
            CS_default,
            CS_yup_right,
            LMatrix4,
            LQuaternion,
            TransformState,
        )

        csxform = LMatrix4.convert_mat(CS_yup_right, CS_default)
        csxform_inv = LMatrix4.convert_mat(CS_default, CS_yup_right)
        for target_node, node_path in self._bound_nodes.items():
            trs = self._sample_node_trs(target_node, time_seconds)
            scale = trs["scale"]
            rotation = trs["rotation"]
            translation = trs["translation"]

            gltf_matrix = LMatrix4.scale_mat(scale[0], scale[1], scale[2])
            rotation_matrix = LMatrix4()
            LQuaternion(rotation[3], rotation[0], rotation[1], rotation[2]).extract_to_matrix(
                rotation_matrix
            )
            gltf_matrix *= rotation_matrix
            gltf_matrix *= LMatrix4.translate_mat(
                translation[0],
                translation[1],
                translation[2],
            )
            node_path.set_transform(TransformState.make_mat(csxform_inv * gltf_matrix * csxform))


class GltfNodeAnimationPlayer:
    """Small clock wrapper that advances a glTF node runtime on Panda NodePaths."""

    def __init__(self, runtime: GltfNodeAnimationRuntime, *, loop: bool = True):
        self.runtime = runtime
        self.loop = loop
        self.time_seconds = 0.0

    def set_time_seconds(self, time_seconds: float) -> None:
        self.time_seconds = float(time_seconds)
        self.runtime.apply_sample(self.time_seconds, loop=self.loop)

    def advance(self, delta_seconds: float) -> None:
        self.set_time_seconds(self.time_seconds + float(delta_seconds))
