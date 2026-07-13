"""glTF loading entrypoints for the compliance layer."""

from .materials import (
    apply_gltf_sampler_to_texture,
    gltf_texture_cache_key,
    normalize_gltf_sampler,
    parse_gltf_material,
)
from ..gltf_loader import (
    audit_gltf_extensions,
    diagnose_gltf_model,
    format_gltf_scene_summary,
    gltf_primitive_mode_to_moderngl,
    load_glb_model,
    load_gltf_scene,
    summarize_gltf_scene,
)

__all__ = [
    "apply_gltf_sampler_to_texture",
    "audit_gltf_extensions",
    "diagnose_gltf_model",
    "format_gltf_scene_summary",
    "gltf_primitive_mode_to_moderngl",
    "gltf_texture_cache_key",
    "load_glb_model",
    "load_gltf_scene",
    "normalize_gltf_sampler",
    "parse_gltf_material",
    "summarize_gltf_scene",
]
