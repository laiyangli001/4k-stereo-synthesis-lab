"""Stable renderer-facing glTF contract types."""

from ..gltf_contract import (
    ColorSpace,
    D3D11_VERTEX_OFFSETS_BYTES,
    D3D11_VERTEX_STRIDE_BYTES,
    GltfMaterial,
    GltfPrimitive,
    GltfScene,
    OPENGL_VERTEX_FORMAT,
    TANGENT_FLOAT_COUNT,
    TextureBinding,
    TextureTransform,
    VERTEX_FLOAT_COUNT,
    attach_primitive_contract,
    build_primitive_contract,
    validate_mesh_contract,
)

__all__ = [
    "ColorSpace",
    "D3D11_VERTEX_OFFSETS_BYTES",
    "D3D11_VERTEX_STRIDE_BYTES",
    "GltfMaterial",
    "GltfPrimitive",
    "GltfScene",
    "OPENGL_VERTEX_FORMAT",
    "TANGENT_FLOAT_COUNT",
    "TextureBinding",
    "TextureTransform",
    "VERTEX_FLOAT_COUNT",
    "attach_primitive_contract",
    "build_primitive_contract",
    "validate_mesh_contract",
]
