import json
import os

import numpy as np

from .gltf_loader import gltf_texture_cache_key, normalize_gltf_sampler
from .material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS, GLTF_TEXTURE_FIELDS


_TEXTURE_FIELDS = GLTF_TEXTURE_FIELDS


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


def collect_controller_texture_requests(prims_data):
    requests = set()
    for pd in prims_data:
        for _name, tex_field, sampler_field in _TEXTURE_FIELDS:
            tid = int(pd.get(tex_field, -1))
            if tid >= 0:
                requests.add((tid, normalize_gltf_sampler(pd.get(sampler_field))))
    return requests


def controller_texture_key(prefix, pd, tex_field, sampler_field):
    tid = int(pd.get(tex_field, -1))
    if tid < 0:
        return None
    return gltf_texture_cache_key(prefix, tid, pd.get(sampler_field))


def prepare_controller_material(pd, prefix, config):
    defaults = config.get("defaults", {}) if isinstance(config, dict) else {}
    pbr = config.get("pbr", {}) if isinstance(config, dict) else {}
    diagnostics = config.get("diagnostics", {}) if isinstance(config, dict) else {}
    brand = str(prefix).split("/", 1)[0]
    overrides = config.get("brandOverrides", {}) if isinstance(config, dict) else {}
    brand_defaults = overrides.get(brand, {}) if isinstance(overrides, dict) else {}
    if not isinstance(brand_defaults, dict):
        brand_defaults = {}
    base_color = _vec(
        pd.get("base_color", defaults.get("baseColorFactor", (1.0, 1.0, 1.0))),
        3,
        defaults.get("baseColorFactor", (1.0, 1.0, 1.0)),
    )
    emissive = _vec(
        pd.get("emissive_factor", defaults.get("emissiveFactor", (0.0, 0.0, 0.0))),
        3,
        defaults.get("emissiveFactor", (0.0, 0.0, 0.0)),
    )
    alpha_mode = str(pd.get("alpha_mode", defaults.get("alphaMode", "OPAQUE")) or "OPAQUE").upper()
    if alpha_mode not in ("OPAQUE", "MASK", "BLEND"):
        alpha_mode = "OPAQUE"
    alpha_mode = str(brand_defaults.get("alphaMode", alpha_mode)).upper()
    if alpha_mode not in ("OPAQUE", "MASK", "BLEND"):
        alpha_mode = "OPAQUE"
    material = {
        "base_color": base_color,
        "base_alpha": float(pd.get("base_alpha", defaults.get("baseAlpha", 1.0)) or 1.0),
        "roughness": float(brand_defaults.get("roughnessFactor", pd.get("roughness_factor", defaults.get("roughnessFactor", 1.0))) or 1.0),
        "metallic": float(brand_defaults.get("metallicFactor", pd.get("metallic_factor", defaults.get("metallicFactor", 0.0))) or 0.0),
        "normal_scale": float(pd.get("normal_scale", defaults.get("normalScale", 1.0)) or 1.0),
        "occlusion_strength": float(pd.get("occlusion_strength", defaults.get("occlusionStrength", 1.0)) or 1.0),
        "emissive_factor": emissive,
        "alpha_mode": alpha_mode,
        "alpha_mode_id": alpha_mode_id(alpha_mode),
        "alpha_cutoff": float(pd.get("alpha_cutoff", defaults.get("alphaCutoff", 0.5)) or 0.5),
        "double_sided": bool(brand_defaults.get("doubleSided", pd.get("double_sided", defaults.get("doubleSided", False)))),
        "unlit": bool(pd.get("unlit", False)),
        "tex_offset": _vec(pd.get("tex_offset", defaults.get("texOffset", (0.0, 0.0))), 2, defaults.get("texOffset", (0.0, 0.0))),
        "tex_scale": _vec(pd.get("tex_scale", defaults.get("texScale", (1.0, 1.0))), 2, defaults.get("texScale", (1.0, 1.0))),
        "tex_rotation": float(pd.get("tex_rotation", defaults.get("texRotation", 0.0)) or 0.0),
        "base_texcoord": int(pd.get("base_texcoord", defaults.get("baseTexcoord", 0)) or 0),
        "normal_texcoord": int(pd.get("normal_texcoord", defaults.get("normalTexcoord", 0)) or 0),
        "occlusion_texcoord": int(pd.get("occlusion_texcoord", defaults.get("occlusionTexcoord", 0)) or 0),
        "mr_texcoord": int(pd.get("mr_texcoord", defaults.get("metallicRoughnessTexcoord", 0)) or 0),
        "emissive_texcoord": int(pd.get("emissive_texcoord", defaults.get("emissiveTexcoord", 0)) or 0),
        "material_mode": str(pbr.get("mode", "environment_pbr") or "environment_pbr"),
        "use_environment_pbr": bool(pbr.get("useEnvironmentPbr", True)),
        "material_diag": str(diagnostics.get("materialMode", "") if isinstance(diagnostics, dict) else "").strip().lower(),
    }
    for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
        material[binding.material_key] = controller_texture_key(
            prefix,
            pd,
            binding.source_tex_field,
            binding.source_sampler_field,
        )
    return material
