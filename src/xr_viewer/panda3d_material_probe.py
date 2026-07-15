"""Phase-0 glTF material semantics diagnostics for the Panda3D renderer path."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from pygltflib import GLTF2


@dataclass(frozen=True)
class Panda3DMaterialProbeReport:
    asset_path: str
    material_count: int
    image_count: int
    texture_count: int
    alpha_mode_counts: dict[str, int]
    double_sided_count: int
    unlit_material_count: int
    base_color_texture_count: int
    emissive_texture_count: int
    normal_texture_count: int
    occlusion_texture_count: int
    metallic_roughness_texture_count: int
    skybox_material_names: tuple[str, ...]
    transparent_material_names: tuple[str, ...]
    material_semantics_ready: bool
    material_semantics_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Panda3DMaterialProbeError(RuntimeError):
    """Raised when a glTF material semantics report cannot be created."""


def _has_unlit_extension(material: Any) -> bool:
    return bool(material.extensions and "KHR_materials_unlit" in material.extensions)


def _pbr(material: Any) -> Any | None:
    return getattr(material, "pbrMetallicRoughness", None)


def _has_texture(info: Any) -> bool:
    return info is not None and getattr(info, "index", None) is not None


def _material_semantics_status(
    *,
    material_count: int,
    image_count: int,
    texture_count: int,
    skybox_material_names: tuple[str, ...],
) -> tuple[bool, str]:
    if material_count <= 0:
        return False, "asset has no glTF materials"
    if texture_count <= 0 or image_count <= 0:
        return False, "asset has no glTF textures/images"
    if not skybox_material_names:
        return False, "asset has no skybox-like material name"
    return True, "glTF material semantics are inspectable before Panda3D rendering"


def inspect_panda3d_materials(asset_path: str | Path) -> Panda3DMaterialProbeReport:
    path = Path(asset_path).resolve()
    if not path.is_file():
        raise Panda3DMaterialProbeError(f"GLB asset does not exist: {path}")

    try:
        gltf = GLTF2().load(str(path))
    except Exception as exc:
        raise Panda3DMaterialProbeError(f"Failed to load glTF material document: {exc}") from exc

    materials = tuple(gltf.materials or ())
    alpha_modes = Counter((material.alphaMode or "OPAQUE") for material in materials)
    skybox_names = tuple(
        material.name or ""
        for material in materials
        if "skybox" in (material.name or "").lower()
    )
    transparent_names = tuple(
        material.name or ""
        for material in materials
        if (material.alphaMode or "OPAQUE") in {"BLEND", "MASK"}
    )

    base_color_texture_count = 0
    metallic_roughness_texture_count = 0
    for material in materials:
        pbr = _pbr(material)
        if pbr is None:
            continue
        if _has_texture(getattr(pbr, "baseColorTexture", None)):
            base_color_texture_count += 1
        if _has_texture(getattr(pbr, "metallicRoughnessTexture", None)):
            metallic_roughness_texture_count += 1

    ready, reason = _material_semantics_status(
        material_count=len(materials),
        image_count=len(gltf.images or ()),
        texture_count=len(gltf.textures or ()),
        skybox_material_names=skybox_names,
    )
    return Panda3DMaterialProbeReport(
        asset_path=str(path),
        material_count=len(materials),
        image_count=len(gltf.images or ()),
        texture_count=len(gltf.textures or ()),
        alpha_mode_counts=dict(sorted(alpha_modes.items())),
        double_sided_count=sum(1 for material in materials if bool(material.doubleSided)),
        unlit_material_count=sum(1 for material in materials if _has_unlit_extension(material)),
        base_color_texture_count=base_color_texture_count,
        emissive_texture_count=sum(
            1 for material in materials if _has_texture(getattr(material, "emissiveTexture", None))
        ),
        normal_texture_count=sum(
            1 for material in materials if _has_texture(getattr(material, "normalTexture", None))
        ),
        occlusion_texture_count=sum(
            1 for material in materials if _has_texture(getattr(material, "occlusionTexture", None))
        ),
        metallic_roughness_texture_count=metallic_roughness_texture_count,
        skybox_material_names=skybox_names,
        transparent_material_names=transparent_names,
        material_semantics_ready=ready,
        material_semantics_reason=reason,
    )


def material_report_as_json(report: Panda3DMaterialProbeReport) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
