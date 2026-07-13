# Desktop2Stereo OpenXR viewer: environment model loading and switching helpers.

import moderngl

from .implementation import *
from .gltf import (
    GltfMaterial,
    OPENGL_VERTEX_FORMAT,
    format_gltf_scene_summary,
    render_pass_from_primitive,
    summarize_gltf_scene,
    validate_mesh_contract,
)
from .material_contract import GLTF_MATERIAL_TEXTURE_BINDINGS


class EnvironmentModelMixin:
    """GLB/procedural environment resource management and runtime switching."""

    def _build_env_model_mat4(self):
        return self._env_model_mat4()


    def _env_model_mat4(self):
        """Return model->world transform for the environment model, cached per frame."""
        fc = getattr(self, '_frame_count', -1)
        transform_key = (
            tuple(float(v) for v in self._env_model_pos),
            tuple(float(v) for v in self._env_model_rot),
            tuple(float(v) for v in self._env_model_scale),
        )
        cached = getattr(self, '_cached_env_model_mat4_frame', -2)
        if fc == cached and transform_key == getattr(self, '_cached_env_model_mat4_key', None):
            return self._cached_env_model_mat4_val
        sx, sy, sz = [float(v) for v in self._env_model_scale]
        yaw, pitch, roll = [float(v) for v in self._env_model_rot]
        cy, sy_ = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        scale = np.eye(4, dtype=np.float32)
        scale[0, 0], scale[1, 1], scale[2, 2] = sx, sy, sz
        ry = np.array([[cy, 0.0, sy_, 0.0],
                       [0.0, 1.0, 0.0, 0.0],
                       [-sy_, 0.0, cy, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        rx = np.array([[1.0, 0.0, 0.0, 0.0],
                       [0.0, cp, -sp, 0.0],
                       [0.0, sp, cp, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        rz = np.array([[cr, -sr, 0.0, 0.0],
                       [sr, cr, 0.0, 0.0],
                       [0.0, 0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        trans = np.eye(4, dtype=np.float32)
        trans[:3, 3] = np.array(self._env_model_pos, dtype=np.float32)
        model_mat = trans @ ry @ rx @ rz @ scale
        self._cached_env_model_mat4_val = model_mat
        self._cached_env_model_mat4_frame = fc
        self._cached_env_model_mat4_key = transform_key
        return model_mat


    @staticmethod
    def _prebake_prim_render_state(prim):
        bc = prim.get('base_color')
        ef = prim.get('emissive_factor')
        to = prim.get('tex_offset')
        ts = prim.get('tex_scale')
        alpha_mode = prim.get('alpha_mode', 'OPAQUE')
        render_pass = render_pass_from_primitive(prim)
        rs = {
            'bc': (float(bc[0]), float(bc[1]), float(bc[2])) if bc is not None else (1.0, 1.0, 1.0),
            'ba': float(prim.get('base_alpha', 1.0)),
            'rf': float(prim.get('roughness_factor', 1.0)),
            'mf': float(prim.get('metallic_factor', 0.0)),
            'ef': (float(ef[0]), float(ef[1]), float(ef[2])) if ef is not None else (0.0, 0.0, 0.0),
            'unlit': 1 if prim.get('unlit', False) else 0,
            'foliage': 1 if prim.get('foliage_mode', False) else 0,
            'am': 0 if alpha_mode == 'OPAQUE' else (1 if alpha_mode == 'MASK' else 2),
            'ac': float(prim.get('alpha_cutoff', 0.5)),
            'render_pass': render_pass,
            'blend': render_pass == 'transparent',
            'double_sided': bool(prim.get('double_sided', False)),
            'to': (float(to[0]), float(to[1])) if to is not None else (0.0, 0.0),
            'ts': (float(ts[0]), float(ts[1])) if ts is not None else (1.0, 1.0),
            'tr': float(prim.get('tex_rotation', 0.0)),
            'base_tc': int(prim.get('base_texcoord', 0)),
            'tk': prim.get('tex_key'),
            'render_mode': prim.get('render_mode', moderngl.TRIANGLES),
            'ns': float(prim.get('normal_scale', 1.0)),
            'os': float(prim.get('occlusion_strength', 1.0)),
            'normal_tc': int(prim.get('normal_texcoord', 0)),
            'occlusion_tc': int(prim.get('occlusion_texcoord', 0)),
            'mr_tc': int(prim.get('mr_texcoord', 0)),
            'emissive_tc': int(prim.get('emissive_texcoord', 0)),
        }
        for uniform, tex_id_key in (
            ('normal', 'normal_tex_id'),
            ('occlusion', 'occlusion_tex_id'),
            ('mr', 'mr_tex_id'),
            ('emissive', 'emissive_tex_id'),
        ):
            tex_id = int(prim.get(tex_id_key, -1))
            sampler = prim.get(f'{uniform}_sampler')
            rs[f'{uniform}_key'] = gltf_texture_cache_key('env', tex_id, sampler) if tex_id >= 0 else None
        prim['_rs'] = rs


    def _transform_env_point(self, point, model_mat):
        p = np.array([float(point[0]), float(point[1]), float(point[2]), 1.0], dtype=np.float32)
        return (model_mat @ p)[:3]


    def _transform_env_direction(self, direction, model_mat):
        d = model_mat[:3, :3] @ np.array(direction, dtype=np.float32)
        return d / (np.linalg.norm(d) + 1e-8)


    def _env_light_range_scale(self):
        try:
            return max(abs(float(v)) for v in self._env_model_scale) or 1.0
        except Exception:
            return 1.0


    def _load_env_model(self, path):
        """Load a glTF environment model from *path*.

        Populates ``self._env_model_prims`` and ``self._env_model_tex_cache``.
        Textures use LINEAR_MIPMAP_LINEAR + mipmaps + 16x anisotropy.
        If the file is corrupt or resources cannot be allocated, this method
        fails silently (prints a warning) and leaves the primitive list empty.
        """
        prims_data = []
        textures = []
        try:
            prims_data, textures, env_lights = load_glb_model(path)
            if env_lights:
                self._scene_lights = env_lights
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to load environment model {path}: {exc}")
            return

        _prefix = "env"
        try:
            # Upload textures. glTF sampler state belongs to textures[], not images[],
            # so cache by image id + sampler tuple.
            sampler_requests = set()
            for pd in prims_data:
                for tex_id_key, sampler_key in (
                    ('tex_id', 'base_sampler'),
                    ('normal_tex_id', 'normal_sampler'),
                    ('occlusion_tex_id', 'occlusion_sampler'),
                    ('mr_tex_id', 'mr_sampler'),
                    ('emissive_tex_id', 'emissive_sampler'),
                ):
                    tid = int(pd.get(tex_id_key, -1))
                    if tid >= 0:
                        sampler_requests.add((tid, normalize_gltf_sampler(pd.get(sampler_key))))
            for tid, sampler in sampler_requests:
                if tid < len(textures) and textures[tid] is not None:
                    cache_key = gltf_texture_cache_key(_prefix, tid, sampler)
                    h, w = textures[tid].shape[:2]
                    mtex = self.ctx.texture((w, h), 4, textures[tid].tobytes())
                    apply_gltf_sampler_to_texture(mtex, sampler)
                    mtex.build_mipmaps()
                    mtex.anisotropy = self._env_texture_anisotropy
                    self._env_model_tex_cache[cache_key] = mtex

            # Create VAOs (bound to _env_prog, no gl_FrontFacing discard)
            baked_lightmap = False
            if isinstance(getattr(self, '_env_profile', None), dict):
                baked_lightmap = bool(self._env_profile.get('baked_lightmap', self._env_profile.get('baked', False)))
            baked_uv1_forced = 0
            for pd in prims_data:
                validate_mesh_contract(pd['vertices'], pd['tangent'], pd['indices'])
                if (
                    baked_lightmap
                    and pd.get('has_uv1', False)
                    and int(pd.get('occlusion_tex_id', -1)) >= 0
                    and int(pd.get('occlusion_texcoord', 0)) != 1
                ):
                    pd['occlusion_texcoord'] = 1
                    baked_uv1_forced += 1
                vbo = self.ctx.buffer(pd['vertices'].tobytes())
                tan_vbo = self.ctx.buffer(pd['tangent'].tobytes())
                ibo = self.ctx.buffer(pd['indices'].tobytes())
                vao = self.ctx.vertex_array(
                    self._env_prog,
                    [(vbo, OPENGL_VERTEX_FORMAT, 'in_position', 'in_normal', 'in_uv', 'in_uv1'),
                     (tan_vbo, '4f', 'in_tangent')],
                    ibo,
                )
                base_color = pd.get('base_color', np.array([1.0, 1.0, 1.0], dtype=np.float32))
                emissive_factor = pd.get('emissive_factor', np.array([0.0, 0.0, 0.0], dtype=np.float32))
                base_alpha = float(pd.get('base_alpha', 1.0))
                alpha_mode = pd.get('alpha_mode', 'OPAQUE')
                vertices = pd.get('vertices')
                if isinstance(vertices, np.ndarray) and len(vertices) > 0:
                    sort_center_local = vertices[:, :3].mean(axis=0).astype(np.float32)
                else:
                    sort_center_local = np.zeros(3, dtype=np.float32)
                tex_key = (
                    gltf_texture_cache_key(_prefix, pd['tex_id'], pd.get('base_sampler'))
                    if pd['tex_id'] >= 0 else None
                )
                normal_tex_id = pd.get('normal_tex_id', -1)
                occlusion_tex_id = pd.get('occlusion_tex_id', -1)
                mr_tex_id = pd.get('mr_tex_id', -1)
                emissive_tex_id = pd.get('emissive_tex_id', -1)
                material_key = (
                    alpha_mode == 'BLEND',
                    tex_key or '',
                    normal_tex_id,
                    occlusion_tex_id,
                    mr_tex_id,
                    emissive_tex_id,
                    tuple(float(x) for x in base_color[:3]),
                    base_alpha,
                    float(pd.get('roughness_factor', 1.0)),
                    float(pd.get('metallic_factor', 0.0)),
                    tuple(float(x) for x in emissive_factor[:3]),
                    bool(pd.get('unlit', False)),
                    alpha_mode,
                    float(pd.get('alpha_cutoff', 0.5)),
                    tuple(float(x) for x in pd.get('tex_offset', np.array([0.0, 0.0], dtype=np.float32))[:2]),
                    tuple(float(x) for x in pd.get('tex_scale', np.array([1.0, 1.0], dtype=np.float32))[:2]),
                    float(pd.get('tex_rotation', 0.0)),
                )
                prim = {
                    'vao': vao, 'vbo': vbo, 'tan_vbo': tan_vbo, 'ibo': ibo,
                    # Keep CPU geometry for the native D3D11 renderer.
                    'vertices': np.ascontiguousarray(pd['vertices'], dtype=np.float32),
                    'indices': np.ascontiguousarray(pd['indices'], dtype=np.uint32),
                    'tex_key': tex_key,
                    'render_mode': gltf_primitive_mode_to_moderngl(pd.get('primitive_mode', 4)),
                    'render_pass': render_pass_from_primitive(pd),
                    'material_contract': pd.get('material_contract'),
                    'tri_count': len(pd['indices']) // 3,
                    'base_color': base_color,
                    'base_alpha': base_alpha,
                    'roughness_factor': pd.get('roughness_factor', 1.0),
                    'metallic_factor': pd.get('metallic_factor', 0.0),
                    'emissive_factor': emissive_factor,
                    'normal_tex_id': normal_tex_id,
                    'normal_sampler': pd.get('normal_sampler'),
                    'normal_texcoord': pd.get('normal_texcoord', 0),
                    'normal_scale': pd.get('normal_scale', 1.0),
                    'occlusion_tex_id': occlusion_tex_id,
                    'occlusion_sampler': pd.get('occlusion_sampler'),
                    'occlusion_texcoord': pd.get('occlusion_texcoord', 0),
                    'occlusion_strength': pd.get('occlusion_strength', 1.0),
                    'unlit': pd.get('unlit', False),
                    'alpha_mode': alpha_mode,
                    'alpha_cutoff': pd.get('alpha_cutoff', 0.5),
                    'mr_tex_id': mr_tex_id,
                    'mr_sampler': pd.get('mr_sampler'),
                    'mr_texcoord': pd.get('mr_texcoord', 0),
                    'emissive_tex_id': emissive_tex_id,
                    'emissive_sampler': pd.get('emissive_sampler'),
                    'emissive_texcoord': pd.get('emissive_texcoord', 0),
                    'double_sided': pd.get('double_sided', False),
                    'foliage_mode': pd.get('foliage_mode', False),
                    'sort_center_local': sort_center_local,
                    'base_texcoord': pd.get('base_texcoord', 0),
                    'tex_offset': pd.get('tex_offset', np.array([0.0, 0.0], dtype=np.float32)),
                    'tex_scale': pd.get('tex_scale', np.array([1.0, 1.0], dtype=np.float32)),
                    'tex_rotation': pd.get('tex_rotation', 0.0),
                    'material_key': material_key,
                }
                material_texture_keys = {}
                for binding in GLTF_MATERIAL_TEXTURE_BINDINGS:
                    tid = int(pd.get(binding.source_tex_field, -1))
                    material_texture_keys[binding.material_key] = (
                        gltf_texture_cache_key(_prefix, tid, pd.get(binding.source_sampler_field))
                        if tid >= 0 else None
                    )
                prim['render_material'] = {
                    **material_texture_keys,
                    'base_color': tuple(float(x) for x in base_color[:3]), 'base_alpha': base_alpha,
                    'roughness': float(pd.get('roughness_factor', 1.0)), 'metallic': float(pd.get('metallic_factor', 0.0)),
                    'alpha_mode_id': 0.0 if alpha_mode == 'OPAQUE' else (1.0 if alpha_mode == 'MASK' else 2.0),
                    'alpha_cutoff': float(pd.get('alpha_cutoff', 0.5)), 'unlit': bool(pd.get('unlit', False)),
                    'double_sided': bool(pd.get('double_sided', False)), 'normal_scale': float(pd.get('normal_scale', 1.0)),
                    'occlusion_strength': float(pd.get('occlusion_strength', 1.0)),
                    'emissive_factor': tuple(float(x) for x in emissive_factor[:3]),
                    'tex_offset': tuple(float(x) for x in pd.get('tex_offset', (0.0, 0.0))[:2]),
                    'tex_scale': tuple(float(x) for x in pd.get('tex_scale', (1.0, 1.0))[:2]),
                    'tex_rotation': float(pd.get('tex_rotation', 0.0)),
                    'base_texcoord': int(pd.get('base_texcoord', 0)), 'normal_texcoord': int(pd.get('normal_texcoord', 0)),
                    'occlusion_texcoord': int(pd.get('occlusion_texcoord', 0)), 'mr_texcoord': int(pd.get('mr_texcoord', 0)),
                    'emissive_texcoord': int(pd.get('emissive_texcoord', 0)),
                }
                prim['d3d_tex_images'] = {
                    key: np.ascontiguousarray(textures[int(tid)], dtype=np.uint8)
                    for key, tid in (
                        (prim['render_material'][binding.material_key], pd.get(binding.source_tex_field, -1))
                        for binding in GLTF_MATERIAL_TEXTURE_BINDINGS
                    )
                    if key and isinstance(tid, (int, np.integer)) and 0 <= int(tid) < len(textures) and textures[int(tid)] is not None
                }
                self._prebake_prim_render_state(prim)
                self._env_model_prims.append(prim)
            if self._env_shading_mode != 'preview':
                self._env_model_prims.sort(key=lambda prim: prim.get('material_key', ()))
            if baked_lightmap:
                occ_count = sum(1 for prim in self._env_model_prims if int(prim.get('occlusion_tex_id', -1)) >= 0)
                occ_uv1 = sum(1 for prim in self._env_model_prims if int(prim.get('occlusion_tex_id', -1)) >= 0 and int(prim.get('occlusion_texcoord', 0)) == 1)
                print(f"[OpenXRViewer] Baked lightmap primitives: occlusion={occ_count} uv1={occ_uv1}")
            if baked_uv1_forced:
                print(f"[OpenXRViewer] Baked lightmap forced occlusion texCoord=1 on {baked_uv1_forced} primitives")
            summary = summarize_gltf_scene(prims_data, textures, env_lights)
            self._env_model_summary = summary
            print("[OpenXRViewer] " + format_gltf_scene_summary(summary, label=f"Active environment {path}"))
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to create environment model resources: {exc}")
            self._release_env_model_resources()


    def _release_env_model_resources(self):
        """Release current room GL resources before reloading or shutdown."""
        for prim in self._env_model_prims:
            for key in ('vao', 'vbo', 'tan_vbo', 'ibo'):
                obj = prim.get(key)
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
        self._env_model_prims = []
        for tex in self._env_model_tex_cache.values():
            try:
                tex.release()
            except Exception:
                pass
        self._env_model_tex_cache = {}
        self._scene_lights = []
        self._env_model_summary = None
        self._env_model_visible = False


    def _generate_default_room(self, target_list=None):
        """Generate a simple room (floor, 4 walls, ceiling) procedurally."""
        if target_list is None:
            target_list = self._env_model_prims
        W, H, D = 4.0, 3.0, 4.0
        import numpy as np
        faces = []
        faces.append((np.array([[-W,0,-D, 0,1,0, 0,0], [W,0,-D, 0,1,0, 1,0], [W,0,D, 0,1,0, 1,1], [-W,0,D, 0,1,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.20, 0.20, 0.22)))
        faces.append((np.array([[-W,0,-D, 0,0,1, 0,0], [W,0,-D, 0,0,1, 1,0], [W,H,-D, 0,0,1, 1,1], [-W,H,-D, 0,0,1, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.30, 0.30, 0.35)))
        faces.append((np.array([[-W,0,-D, 1,0,0, 0,0], [-W,0,D, 1,0,0, 1,0], [-W,H,D, 1,0,0, 1,1], [-W,H,-D, 1,0,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.25, 0.25, 0.30)))
        faces.append((np.array([[W,0,-D, -1,0,0, 0,0], [W,H,-D, -1,0,0, 1,0], [W,H,D, -1,0,0, 1,1], [W,0,D, -1,0,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.28, 0.28, 0.33)))
        faces.append((np.array([[-W,H,-D, 0,-1,0, 0,0], [-W,H,D, 0,-1,0, 1,0], [W,H,D, 0,-1,0, 1,1], [W,H,-D, 0,-1,0, 0,1]], dtype='f4'),
                      np.array([0,1,2, 0,2,3], dtype='u4'), (0.35, 0.35, 0.40)))
        for verts, idx, color in faces:
            material_contract = GltfMaterial(
                base_color=tuple(float(x) for x in color),
                base_alpha=1.0,
                alpha_mode='OPAQUE',
                roughness=1.0,
                metallic=0.0,
                double_sided=False,
            )
            verts = np.hstack([verts, verts[:, 6:8]]).astype('f4')
            vbo = self.ctx.buffer(verts.tobytes())
            # Dummy tangent: (1,0,0,1) -room faces have no normal map anyway
            dummy_tan = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype='f4'), (verts.shape[0], 1))
            tan_vbo = self.ctx.buffer(dummy_tan.tobytes())
            ibo = self.ctx.buffer(idx.tobytes())
            vao = self.ctx.vertex_array(
                self._env_prog,
                [(vbo, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_uv1'),
                 (tan_vbo, '4f', 'in_tangent')],
                ibo,
            )
            prim = {
                'vao': vao, 'vbo': vbo, 'tan_vbo': tan_vbo, 'ibo': ibo,
                'vertices': np.ascontiguousarray(verts, dtype=np.float32),
                'indices': np.ascontiguousarray(idx, dtype=np.uint32),
                'tex_key': None, 'tri_count': 2, 'color': color,
                'material_contract': material_contract,
                'base_color': np.array(color, dtype=np.float32),
                'base_alpha': 1.0,
                'roughness_factor': 1.0,
                'metallic_factor': 0.0,
                'emissive_factor': np.array([0.0, 0.0, 0.0], dtype=np.float32),
                'normal_tex_id': -1,
                'normal_texcoord': 0,
                'normal_scale': 1.0,
                'occlusion_tex_id': -1,
                'occlusion_texcoord': 0,
                'occlusion_strength': 1.0,
                'unlit': False,
                'alpha_mode': 'OPAQUE',
                'alpha_cutoff': 0.5,
                'mr_tex_id': -1,
                'mr_texcoord': 0,
                'emissive_tex_id': -1,
                'emissive_texcoord': 0,
                'double_sided': False,
                'base_texcoord': 0,
                'render_mode': moderngl.TRIANGLES,
                'tex_offset': np.array([0.0, 0.0], dtype=np.float32),
                'tex_scale': np.array([1.0, 1.0], dtype=np.float32),
                'tex_rotation': 0.0,
                'render_material': {
                    'base_key': None, 'normal_key': None, 'occlusion_key': None, 'mr_key': None, 'emissive_key': None,
                    'base_color': tuple(float(x) for x in color), 'base_alpha': 1.0, 'roughness': 1.0, 'metallic': 0.0,
                    'alpha_mode_id': 0.0, 'alpha_cutoff': 0.5, 'unlit': False, 'double_sided': False,
                    'normal_scale': 1.0, 'occlusion_strength': 1.0, 'emissive_factor': (0.0, 0.0, 0.0),
                    'tex_offset': (0.0, 0.0), 'tex_scale': (1.0, 1.0), 'tex_rotation': 0.0,
                    'base_texcoord': 0, 'normal_texcoord': 0, 'occlusion_texcoord': 0,
                    'mr_texcoord': 0, 'emissive_texcoord': 0,
                },
                'd3d_tex_images': {},
            }
            self._prebake_prim_render_state(prim)
            target_list.append(prim)
        if target_list is self._env_model_prims:
            self._env_model_visible = True
            print(f'[OpenXRViewer] Default room generated ({len(faces)} faces)')
        else:
            print(f'[OpenXRViewer] Dark-room geometry built ({len(faces)} faces)')


    def _init_dark_room(self):
        """Build the always-available procedural dark room."""
        self._dark_room_prims = []
        try:
            self._generate_default_room(self._dark_room_prims)
        except Exception as exc:
            print(f"[OpenXRViewer] _init_dark_room failed: {exc}")
            self._dark_room_prims = []


    def _release_dark_room_resources(self):
        """Release the procedural dark-room GL resources."""
        for prim in getattr(self, '_dark_room_prims', []) or []:
            for key in ('vao', 'vbo', 'tan_vbo', 'ibo'):
                obj = prim.get(key)
                if obj is not None:
                    try:
                        obj.release()
                    except Exception:
                        pass
        self._dark_room_prims = []


    def _init_env_model(self):
        """Try loading environment.glb, fall back to built-in room."""
        if not getattr(self, '_environment_enabled', True):
            self._env_model_visible = False
            return
        if getattr(self, '_panorama_background_path', None):
            self._env_model_visible = False
            self._env_model_prims = []
            return
        path = self._env_model_path
        if path is None:
            self._env_model_visible = False
            self._env_model_prims = []
            print(f"[OpenXRViewer] Environment model disabled: {self._environment_model}")
            return
        if os.path.exists(path):
            self._load_env_model(path)
            if self._env_model_prims:
                self._env_model_visible = True
                print(f"[OpenXRViewer] Environment model loaded ({len(self._env_model_prims)} primitives): {self._environment_model}")
                return
        self._generate_default_room()


    def _switch_environment(self, name, *, save_outgoing=True, apply_profile=True):
        if name is None:
            self._release_env_model_resources()
            self._active_environment = None
            self._glow_intensity_multiplier = 0.0
            self._persist_runtime_settings()
            return
        self._switch_environment_model(model_name=name)


    def _switch_environment_model(self, model_name=None):
        """Switch to another room environment during runtime."""
        if not getattr(self, '_environment_enabled', True):
            return False
        models = self._available_environment_models or self._discover_environment_models()
        self._available_environment_models = models
        if not models:
            return False

        current = (self._environment_model or '').strip()
        if model_name is None:
            try:
                idx = models.index(current)
            except ValueError:
                idx = -1
            model_name = models[(idx + 1) % len(models)]
        if model_name == current and self._env_model_prims:
            return False

        print(f"[OpenXRViewer] Switching environment to: {model_name}")
        if getattr(self, '_seat_adjust_active', False):
            self._exit_seat_adjust_mode(save=False)
        if not self._environment_screen_locked():
            self._persist_screen_state()
        self._release_env_model_resources()
        self._environment_model = model_name
        self._kb_cached_position = None
        self._configure_environment_profile()
        self._configure_profile_view_layout()
        self._env_model_init_done = False
        self._ensure_env_model_initialized("Switch")
        self._apply_profile_screen_layout(show_border=True)
        self._xr_profile_space_applied = False
        views = getattr(self, '_last_located_views', None)
        if views:
            self._apply_profile_view_pose_to_xr_space(views)
        has_screen_profile = isinstance(getattr(self, '_screen_profile', None), dict) and bool(self._screen_profile)
        has_view_pose = self._view_pose_has_explicit_layout()
        if not has_view_pose:
            self._reset_xr_space_to_identity()
        if not has_screen_profile:
            if not self._restore_screen_state():
                self._reset_screen_to_default(show_border=True)
        self._persist_runtime_settings()
        return True
