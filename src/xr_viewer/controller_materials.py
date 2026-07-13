import json
import os

import numpy as np

from .gltf import GltfMaterial, TextureBinding, TextureTransform, normalize_gltf_sampler
from .material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS


def load_controller_common_config(controllers_root):
    """Load shared glTF/PBR defaults from the environment-owned config.

    The controller-local common.json remains a compatibility fallback for
    installations that have not completed the configuration migration.
    """
    shared_path = os.path.join(os.path.dirname(controllers_root), "environments", "common.json")
    legacy_path = os.path.join(controllers_root, "common.json")
    for path in (shared_path, legacy_path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            continue
    return {}


def _vec(value, size, default):
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        arr = np.asarray(default, dtype=np.float32)
    out = np.asarray(default, dtype=np.float32).copy()
    out[: min(size, arr.size)] = arr[:size]
    return out


def alpha_mode_id(alpha_mode):
    return {"OPAQUE": 0, "MASK": 1, "BLEND": 2}.get(str(alpha_mode or "OPAQUE").upper(), 0)


def _require_gltf_material(material):
    if not isinstance(material, GltfMaterial):
        raise ValueError("controller material must be a GltfMaterial")
    return material


def _texture_binding(material, role):
    binding = material.texture_slots.get(role) if material.texture_slots else None
    if binding is None:
        return None
    if not isinstance(binding, TextureBinding):
        raise ValueError(f"controller material_contract texture slot {role!r} is invalid")
    return binding


def _texture_transform(binding):
    if binding is None:
        return TextureTransform()
    return binding.transform if isinstance(binding.transform, TextureTransform) else TextureTransform()


def _normalize_contract_sampler(sampler):
    try:
        values = tuple(int(item) for item in sampler)
    except (TypeError, ValueError):
        values = ()
    if len(values) == 4:
        return values
    return normalize_gltf_sampler(sampler)


def collect_controller_texture_requests(materials):
    requests = set()
    for material in materials:
        material = _require_gltf_material(material)
        for binding in material.texture_slots.values():
            if not isinstance(binding, TextureBinding):
                raise ValueError("controller material texture slot is invalid")
            if binding.image_id >= 0:
                requests.add((int(binding.image_id), _normalize_contract_sampler(binding.sampler)))
    return requests


def controller_texture_cache_key(prefix, image_id, sampler):
    mag_filter, min_filter, wrap_s, wrap_t = _normalize_contract_sampler(sampler)
    return f"{prefix}:{int(image_id)}:{mag_filter}:{min_filter}:{wrap_s}:{wrap_t}"


def controller_texture_key(prefix, binding):
    if binding is None or binding.image_id < 0:
        return None
    return controller_texture_cache_key(prefix, binding.image_id, binding.sampler)


def prepare_controller_material(material_contract, prefix, config):
    material_contract = _require_gltf_material(material_contract)
    pbr = config.get("pbr", {}) if isinstance(config, dict) else {}
    diagnostics = config.get("diagnostics", {}) if isinstance(config, dict) else {}
    brand = str(prefix).split("/", 1)[0]
    overrides = config.get("brandOverrides", {}) if isinstance(config, dict) else {}
    brand_defaults = overrides.get(brand, {}) if isinstance(overrides, dict) else {}
    if not isinstance(brand_defaults, dict):
        brand_defaults = {}
    texture_slots = {
        binding.role: _texture_binding(material_contract, binding.role)
        for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
    }
    base_binding = texture_slots["base"]
    base_transform = _texture_transform(base_binding)
    alpha_mode = str(material_contract.alpha_mode or "OPAQUE").upper()
    if alpha_mode not in ("OPAQUE", "MASK", "BLEND"):
        alpha_mode = "OPAQUE"
    alpha_mode = str(brand_defaults.get("alphaMode", alpha_mode)).upper()
    if alpha_mode not in ("OPAQUE", "MASK", "BLEND"):
        alpha_mode = "OPAQUE"
    material = {
        "base_color": _vec(material_contract.base_color, 3, (1.0, 1.0, 1.0)),
        "base_alpha": float(material_contract.base_alpha),
        "roughness": float(brand_defaults.get("roughnessFactor", material_contract.roughness)),
        "metallic": float(brand_defaults.get("metallicFactor", material_contract.metallic)),
        "normal_scale": float(material_contract.normal_scale),
        "occlusion_strength": float(material_contract.occlusion_strength),
        "emissive_factor": _vec(material_contract.emissive_factor, 3, (0.0, 0.0, 0.0)),
        "alpha_mode": alpha_mode,
        "alpha_mode_id": alpha_mode_id(alpha_mode),
        "alpha_cutoff": float(material_contract.alpha_cutoff),
        "double_sided": bool(brand_defaults.get("doubleSided", material_contract.double_sided)),
        "unlit": bool(material_contract.unlit),
        "tex_offset": _vec(base_transform.offset, 2, (0.0, 0.0)),
        "tex_scale": _vec(base_transform.scale, 2, (1.0, 1.0)),
        "tex_rotation": float(base_transform.rotation),
        "base_texcoord": int(base_binding.texcoord) if base_binding is not None else 0,
        "normal_texcoord": int(texture_slots["normal"].texcoord) if texture_slots["normal"] is not None else 0,
        "occlusion_texcoord": int(texture_slots["occlusion"].texcoord) if texture_slots["occlusion"] is not None else 0,
        "mr_texcoord": int(texture_slots["mr"].texcoord) if texture_slots["mr"] is not None else 0,
        "emissive_texcoord": int(texture_slots["emissive"].texcoord) if texture_slots["emissive"] is not None else 0,
        "material_mode": str(pbr.get("mode", "environment_pbr") or "environment_pbr"),
        "use_environment_pbr": bool(pbr.get("useEnvironmentPbr", True)),
        "material_diag": str(diagnostics.get("materialMode", "") if isinstance(diagnostics, dict) else "").strip().lower(),
    }
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
        material[binding.material_key] = controller_texture_key(prefix, texture_slots[binding.role])
    return material
