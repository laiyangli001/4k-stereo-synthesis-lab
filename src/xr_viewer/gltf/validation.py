"""glTF extension diagnostics and fail-fast validation helpers."""

SUPPORTED_REQUIRED_EXTENSIONS = {
    "KHR_lights_punctual",
    "KHR_materials_unlit",
    "KHR_texture_transform",
}

SUPPORTED_OPTIONAL_EXTENSIONS = SUPPORTED_REQUIRED_EXTENSIONS | {
    "KHR_materials_emissive_strength",
    "KHR_materials_pbrSpecularGlossiness",
}


def _extension_list(value):
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if isinstance(item, str) and item]


def audit_gltf_extensions(gltf):
    used = set(_extension_list(gltf.get("extensionsUsed")))
    required = set(_extension_list(gltf.get("extensionsRequired")))
    material_extensions = set()
    primitive_extensions = set()

    for material in gltf.get("materials") or []:
        if isinstance(material, dict) and isinstance(material.get("extensions"), dict):
            material_extensions.update(str(key) for key in material["extensions"].keys())
    for mesh in gltf.get("meshes") or []:
        if not isinstance(mesh, dict):
            continue
        for primitive in mesh.get("primitives") or []:
            if isinstance(primitive, dict) and isinstance(primitive.get("extensions"), dict):
                primitive_extensions.update(str(key) for key in primitive["extensions"].keys())

    used.update(material_extensions)
    used.update(primitive_extensions)
    return {
        "extensionsUsed": sorted(used),
        "extensionsRequired": sorted(required),
        "unsupportedRequired": sorted(required - SUPPORTED_REQUIRED_EXTENSIONS),
        "unsupportedOptional": sorted((used - required) - SUPPORTED_OPTIONAL_EXTENSIONS),
        "materialExtensions": sorted(material_extensions),
        "primitiveExtensions": sorted(primitive_extensions),
    }


def raise_unsupported_required_extensions(gltf, path):
    diagnostics = audit_gltf_extensions(gltf)
    unsupported = diagnostics["unsupportedRequired"]
    if unsupported:
        raise ValueError(
            f"Unsupported required glTF extensions for {path}: {', '.join(unsupported)}. "
            "Convert the asset or add decoder/material support before loading it."
        )
    return diagnostics


__all__ = [
    "SUPPORTED_OPTIONAL_EXTENSIONS",
    "SUPPORTED_REQUIRED_EXTENSIONS",
    "audit_gltf_extensions",
    "raise_unsupported_required_extensions",
]
