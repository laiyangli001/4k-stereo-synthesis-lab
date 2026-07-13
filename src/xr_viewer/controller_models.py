# Desktop2Stereo OpenXR viewer: controller model loading and brand switching.

import os
import time

import numpy as np

from .controller_materials import (
    collect_controller_texture_requests,
    controller_texture_cache_key,
    load_controller_common_config,
    prepare_controller_material,
)
from .gltf_loader import (
    apply_gltf_sampler_to_texture,
    gltf_primitive_mode_to_moderngl,
    load_glb_model,
    normalize_gltf_sampler,
)


class ControllerModelsMixin:
    """Controller GLB model loading and runtime brand switching."""

    def _load_brand_models(self, brand_name):
        """Load models and configuration for a specific brand, returning {prims_l, prims_r, tex_cache, offset, rot_deg}."""
        base_dir = os.path.join(self._controllers_root, brand_name)
        result = {
            'prims_l': [], 'prims_r': [], 'tex_cache': {},
            'tex_images': {},
            'offset': [0.0, 0.0, 0.0], 'rot_deg': 0.0,
        }
        # Read profile.json
        profile_path = os.path.join(base_dir, 'profile.json')
        if os.path.isfile(profile_path):
            try:
                import json as _json
                with open(profile_path, 'r') as f:
                    prof = _json.load(f)
                overrides = prof.get('overrides', {})
                if overrides.get('model_offset'):
                    result['offset'] = [float(v) for v in overrides['model_offset']]
                if 'model_rotation_deg' in overrides:
                    result['rot_deg'] = float(overrides['model_rotation_deg'])
            except Exception as e:
                print(f"[OpenXRViewer] Failed to read {profile_path}: {e}")

        _dir_key = brand_name
        common_config = load_controller_common_config(self._controllers_root)

        def _create_prims(glb_path, target_list):
            prims_data, textures, _lights = load_glb_model(glb_path)
            _file_stem = os.path.splitext(os.path.basename(glb_path))[0]
            _prefix = f"{_dir_key}/{_file_stem}"
            material_contracts = [pd['material_contract'] for pd in prims_data]
            sampler_requests = collect_controller_texture_requests(material_contracts)
            for tid, sampler in sampler_requests:
                if tid < len(textures) and textures[tid] is not None:
                    cache_key = controller_texture_cache_key(_prefix, tid, sampler)
                    result['tex_images'][cache_key] = textures[tid]
                    if cache_key not in result['tex_cache']:
                        h, w = textures[tid].shape[:2]
                        mtex = self.ctx.texture((w, h), 4, textures[tid].tobytes())
                        apply_gltf_sampler_to_texture(mtex, sampler)
                        mtex.build_mipmaps()
                        result['tex_cache'][cache_key] = mtex
            for pd in prims_data:
                vertices = pd['vertices']
                if getattr(vertices, 'ndim', 0) == 2 and vertices.shape[1] < 10:
                    vertices = np.hstack([vertices[:, :8], vertices[:, 6:8]]).astype(np.float32, copy=False)
                vbo = self.ctx.buffer(vertices.tobytes())
                tangent = pd.get('tangent')
                if not isinstance(tangent, np.ndarray) or tangent.shape[0] != vertices.shape[0]:
                    tangent = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (vertices.shape[0], 1))
                tan_vbo = self.ctx.buffer(tangent.astype(np.float32, copy=False).tobytes())
                ibo = self.ctx.buffer(pd['indices'].tobytes())
                vao = self.ctx.vertex_array(
                    self._controller_prog,
                    [(vbo, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_uv1'),
                     (tan_vbo, '4f', 'in_tangent')],
                    ibo,
                )
                material_contract = pd['material_contract']
                target_list.append({
                    'vao': vao, 'vbo': vbo, 'tan_vbo': tan_vbo, 'ibo': ibo,
                    'vertices': vertices,
                    'indices': pd['indices'],
                    'material_contract': material_contract,
                    'material': prepare_controller_material(material_contract, _prefix, common_config),
                    'render_mode': gltf_primitive_mode_to_moderngl(pd.get('primitive_mode', 4)),
                    'primitive_mode': pd.get('primitive_mode', 4),
                    'tri_count': len(pd['indices']) // 3,
                    'node_name': pd.get('node_name', ''),
                    'mesh_name': pd.get('mesh_name', ''),
                    'press_anim': pd.get('press_anim'),
                    'axis_anim': pd.get('axis_anim'),
                    'anim_key': pd.get('anim_key', ''),
                    'visible_key': pd.get('visible_key', ''),
                })

        try:
            _create_prims(os.path.join(base_dir, 'right.glb'), result['prims_r'])
            _create_prims(os.path.join(base_dir, 'left.glb'),  result['prims_l'])
        except Exception as e:
            print(f"[OpenXRViewer] {brand_name} model load failed: {e}")
            result['prims_l'], result['prims_r'] = [], []
        return result

    def _init_all_controller_models(self):
        """Preload all controller brand models under controllers/."""
        if not os.path.isdir(self._controllers_root):
            return
        brands = sorted(d for d in os.listdir(self._controllers_root)
                    if os.path.isdir(os.path.join(self._controllers_root, d)))
        for bn in brands:
            model = self._load_brand_models(bn)
            self._all_models[bn] = model
            self._available_brands.append(bn)
        # Set default brand
        default = self._controller_model if self._controller_model in self._all_models else (
            self._available_brands[0] if self._available_brands else None)
        if default is None:
            print("[OpenXRViewer] No controller brands available!")
            return
        self._switch_brand(default)
        print(f"[OpenXRViewer] Controller: {self._current_brand}")

    def _switch_brand(self, brand_name):
        """Switch controller brand with zero latency."""
        if brand_name not in self._all_models:
            return
        m = self._all_models[brand_name]
        self._ctrl_prims_l      = m['prims_l']
        self._ctrl_prims_r      = m['prims_r']
        self._ctrl_tex_cache    = m['tex_cache']
        self._ctrl_tex_images   = m.get('tex_images', {})
        self._ctrl_model_offset = m['offset']
        self._ctrl_model_rot_deg = m['rot_deg']
        self._current_brand      = brand_name
        self._brand_switch_osd_t = time.perf_counter()
        print(f"[OpenXRViewer] Switched to: {brand_name} "
            f"offset={self._ctrl_model_offset} rot={self._ctrl_model_rot_deg}")
