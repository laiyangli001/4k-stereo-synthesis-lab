"""Phase-0 diagnostics for the optional Panda3D glTF renderer path.

The module deliberately has no Panda3D import at module load time so the
normal OpenXR runtime keeps working when the optional probe dependencies are
not installed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

from pygltflib import GLTF2


@dataclass(frozen=True)
class Panda3DProbeReport:
    asset_path: str
    gltf_animation_count: int
    gltf_animation_channel_count: int
    gltf_animation_target_node_count: int
    gltf_animation_targets_in_active_scene: int
    panda_node_count: int
    panda_geom_count: int
    panda_character_count: int
    panda_animation_bundle_count: int
    panda_animation_names: tuple[str, ...]
    animation_runtime_ready: bool
    animation_runtime_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Panda3DProbeError(RuntimeError):
    """Raised when the optional Panda3D probe cannot inspect an asset."""


def panda3d_probe_available() -> bool:
    """Return whether both optional packages needed by the probe are importable."""
    return bool(importlib.util.find_spec("panda3d")) and bool(
        importlib.util.find_spec("gltf")
    )


def _animation_counts(gltf: GLTF2) -> tuple[int, int]:
    animations = gltf.animations or []
    return len(animations), sum(len(animation.channels or []) for animation in animations)


def _animation_target_node_ids(gltf: GLTF2) -> set[int]:
    targets: set[int] = set()
    for animation in gltf.animations or []:
        for channel in animation.channels or []:
            target = channel.target
            if target is not None and target.node is not None:
                targets.add(int(target.node))
    return targets


def _active_scene_node_ids(gltf: GLTF2) -> set[int]:
    scenes = gltf.scenes or []
    nodes = gltf.nodes or []
    if not scenes:
        return set()
    scene_index = 0 if gltf.scene is None else int(gltf.scene)
    if scene_index < 0 or scene_index >= len(scenes):
        return set()

    reachable: set[int] = set()
    pending = list(scenes[scene_index].nodes or [])
    while pending:
        node_id = int(pending.pop())
        if node_id in reachable or node_id < 0 or node_id >= len(nodes):
            continue
        reachable.add(node_id)
        pending.extend(nodes[node_id].children or [])
    return reachable


def _runtime_status(
    gltf_animation_count: int,
    panda_character_count: int,
    panda_animation_bundle_count: int,
) -> tuple[bool, str]:
    if not gltf_animation_count:
        return True, "asset has no glTF animations"
    if panda_character_count or panda_animation_bundle_count:
        return True, "Panda3D loader exposed animation runtime nodes"
    return False, "glTF animations exist but Panda3D exposed no runtime animation nodes"


def inspect_panda3d_asset(asset_path: str | Path) -> Panda3DProbeReport:
    """Load one GLB through Panda3D and report its animation-runtime boundary.

    This inspection creates no graphics context. It is therefore safe to run
    before the separate offscreen and OpenXR bridge probes.
    """
    path = Path(asset_path).resolve()
    if not path.is_file():
        raise Panda3DProbeError(f"GLB asset does not exist: {path}")
    if not panda3d_probe_available():
        raise Panda3DProbeError(
            "Panda3D probe dependencies are unavailable; install the panda3d-probe extra"
        )

    try:
        import gltf
        from panda3d.core import NodePath
    except ImportError as exc:  # pragma: no cover - guarded above for diagnostics
        raise Panda3DProbeError("Panda3D probe imports failed") from exc

    gltf_document = GLTF2().load(str(path))
    animation_count, channel_count = _animation_counts(gltf_document)
    animation_target_ids = _animation_target_node_ids(gltf_document)
    active_scene_ids = _active_scene_node_ids(gltf_document)
    try:
        root = NodePath(gltf.load_model(str(path)))
    except Exception as exc:
        raise Panda3DProbeError(f"panda3d-gltf failed to load {path.name}: {exc}") from exc

    nodes = root.find_all_matches("**")
    character_nodes = root.find_all_matches("**/+Character")
    bundle_nodes = [
        node_path
        for node_path in nodes
        if node_path.node().get_type().get_name() == "AnimBundleNode"
    ]
    animation_names: list[str] = []
    for character in character_nodes:
        get_names = getattr(character.node(), "get_anim_names", None)
        if callable(get_names):
            animation_names.extend(str(name) for name in get_names())

    ready, reason = _runtime_status(
        animation_count,
        character_nodes.get_num_paths(),
        len(bundle_nodes),
    )
    return Panda3DProbeReport(
        asset_path=str(path),
        gltf_animation_count=animation_count,
        gltf_animation_channel_count=channel_count,
        gltf_animation_target_node_count=len(animation_target_ids),
        gltf_animation_targets_in_active_scene=len(animation_target_ids & active_scene_ids),
        panda_node_count=nodes.get_num_paths(),
        panda_geom_count=root.find_all_matches("**/+GeomNode").get_num_paths(),
        panda_character_count=character_nodes.get_num_paths(),
        panda_animation_bundle_count=len(bundle_nodes),
        panda_animation_names=tuple(sorted(set(animation_names))),
        animation_runtime_ready=ready,
        animation_runtime_reason=reason,
    )


def report_as_json(report: Panda3DProbeReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
