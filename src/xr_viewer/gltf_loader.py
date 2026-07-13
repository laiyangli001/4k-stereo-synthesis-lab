"""Legacy compatibility facade for glTF loading helpers.

Implementation lives in xr_viewer.gltf.* modules; this module preserves the
old xr_viewer.gltf_loader import surface.
"""

from .gltf.accessors import _coerce_vec_array, _get_accessor
from .gltf.document import (
    _buffer_data,
    _decode_data_uri,
    _gltf_to_dict,
    _load_gltf_document,
    _read_glb_chunks,
)
from .gltf.materials import (
    apply_gltf_sampler_to_texture,
    gltf_texture_cache_key,
    normalize_gltf_sampler,
    parse_gltf_material,
)
from .gltf.primitives import gltf_primitive_mode_to_moderngl, load_glb_model
from .gltf.scene import (
    _apply_normal_transform,
    _apply_transform,
    _build_node_matrices,
    _iter_scene_mesh_nodes,
    _node_local_matrix,
    _orthogonalize_tangent,
    _quat_to_mat4,
    diagnose_gltf_model,
    format_gltf_scene_summary,
    load_gltf_scene,
    summarize_gltf_scene,
)
from .gltf.validation import audit_gltf_extensions
from .gltf.validation import (
    raise_unsupported_required_extensions as _raise_unsupported_required_extensions,
)

__all__ = [
    "_apply_normal_transform",
    "_apply_transform",
    "_buffer_data",
    "_build_node_matrices",
    "_coerce_vec_array",
    "_decode_data_uri",
    "_get_accessor",
    "_gltf_to_dict",
    "_iter_scene_mesh_nodes",
    "_load_gltf_document",
    "_node_local_matrix",
    "_orthogonalize_tangent",
    "_quat_to_mat4",
    "_raise_unsupported_required_extensions",
    "_read_glb_chunks",
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
