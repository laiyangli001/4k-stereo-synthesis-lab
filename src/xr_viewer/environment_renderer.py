# Desktop2Stereo OpenXR viewer: environment shader and model rendering helpers.

from .implementation import *
from .render import _view_mat_inv


class EnvironmentRendererMixin:
    """Environment shader uniforms and GL primitive rendering."""

    def _screen_light_source_texture(self):
        source_tex = getattr(self, 'color_tex', None)
        source_size = getattr(self, '_texture_size', None)
        if getattr(self, '_runtime_direct_source', False):
            eye_index = int(getattr(self, '_current_eye_index', 0) or 0)
            runtime_textures = getattr(self, '_runtime_eye_textures', []) or []
            if 0 <= eye_index < len(runtime_textures) and runtime_textures[eye_index] is not None:
                source_tex = runtime_textures[eye_index]
                source_size = getattr(self, '_runtime_eye_texture_size', source_size)
            elif runtime_textures and runtime_textures[0] is not None:
                source_tex = runtime_textures[0]
                source_size = getattr(self, '_runtime_eye_texture_size', source_size)
        return source_tex, source_size

    def _apply_cinema_light_uniforms(self, mgl_fbo=None):
        """Push current screen area-light uniforms to the environment shader."""
        if self.screen_height is None or self._screen_light_intensity <= 0.0:
            self._env_prog['u_screen_light_enabled'].value = 0
            self._cl_light_state_key = None
            self._cl_uniform_frame = -5
            return
        fc = getattr(self, '_frame_count', 0)
        pose_key = (
            self.screen_yaw, self.screen_pitch, self.screen_roll,
            self.screen_pan_x, self.screen_pan_y, self.screen_distance,
            self.screen_width, self.screen_height,
        )
        if pose_key != getattr(self, '_cl_pose_key', None):
            sx_pos = float(self.screen_pan_x)
            sy_pos = float(self.screen_pan_y)
            sz_pos = float(-self.screen_distance)
            cy = math.cos(self.screen_yaw)
            sy_ = math.sin(self.screen_yaw)
            cp = math.cos(self.screen_pitch)
            sp = math.sin(self.screen_pitch)
            self._cl_pos = (sx_pos, sy_pos, sz_pos)
            self._cl_normal = (sy_ * cp, -sp, cy * cp)
            self._cl_right = (cy, 0.0, -sy_)
            self._cl_up = (sy_ * sp, cp, cy * sp)
            self._cl_half = (float(self.screen_width) * 0.5, float(self.screen_height) * 0.5)
            self._cl_pose_key = pose_key
        dynamic = bool(getattr(self, '_screen_light_dynamic', False))
        state_key = (
            pose_key,
            getattr(self, '_active_environment', None),
            float(self._screen_light_intensity),
            dynamic,
        )
        last_state_key = getattr(self, '_cl_light_state_key', None)
        last_frame = getattr(self, '_cl_uniform_frame', -999)
        update_interval = 1 if dynamic else 5
        if state_key == last_state_key and (fc - last_frame) < update_interval:
            return
        self._cl_light_state_key = state_key
        self._cl_uniform_frame = fc
        self._advance_glow_color(lerp=float(getattr(self, '_screen_light_lerp', 0.14)))
        sc = getattr(self, '_glow_color', (0.30, 0.55, 1.0))
        source_tex, _source_size = self._screen_light_source_texture()
        intensity = float(self._screen_light_intensity)
        if getattr(self, '_active_environment', None) == 'Dark Room':
            intensity *= 0.9
        self._env_prog['u_screen_light_enabled'].value = 1
        self._env_prog['u_screen_light_pos'].value = self._cl_pos
        self._env_prog['u_screen_light_normal'].value = self._cl_normal
        self._env_prog['u_screen_light_right'].value = self._cl_right
        self._env_prog['u_screen_light_up'].value = self._cl_up
        self._env_prog['u_screen_light_half_size'].value = self._cl_half
        self._env_prog['u_screen_light_color'].value = (float(sc[0]), float(sc[1]), float(sc[2]))
        if source_tex is not None:
            source_tex.use(location=8)
        self._env_prog['u_screen_light_intensity'].value = intensity


    def _render_env_model(self, mgl_fbo, vp_mat, view_mat):
        """Render the glTF environment model in world space."""
        if not self._env_model_visible or not self._env_model_prims:
            return
        perf_t0 = time.perf_counter() if self._env_perf_log else 0.0

        model_mat = self._env_model_mat4()
        view_inv = _view_mat_inv(view_mat)
        cam_pos = view_inv[:3, 3].astype('f4')
        # Use head center so both eyes get identical head-lamp lighting
        head_pos = getattr(self, '_head_pos_w', None)
        if head_pos is not None:
            cam_pos = np.array(head_pos, dtype=np.float32)

        self._env_prog['u_mvp'].write(vp_mat.astype('f4').T.tobytes())
        self._env_prog['u_model'].write(model_mat.T.tobytes())
        self._env_prog['u_camera_pos'].write(cam_pos.tobytes())
        self._env_prog['u_light_color'].value = self._env_head_light_color
        self._env_prog['u_ambient_color'].value = self._env_ambient_color
        self._env_prog['u_env_exposure'].value = self._env_exposure
        self._env_prog['u_env_gamma'].value = self._env_gamma
        self._env_prog['u_emissive_strength'].value = self._env_emissive_strength
        self._env_prog['u_shading_mode'].value = 1 if self._env_shading_mode == 'preview' else 0
        profile = getattr(self, '_env_profile', {}) or {}
        baked_lightmap = bool(profile.get('baked_lightmap', profile.get('baked', False))) if isinstance(profile, dict) else False
        self._env_prog['u_baked_lightmap'].value = 1 if baked_lightmap else 0

        directional = next((light for light in self._scene_lights if light.get('type') == 'directional'), None)
        if directional:
            light_dir = self._transform_env_direction(directional['direction'], model_mat)
            self._env_prog['u_light_dir'].value = (
                float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
            )
            color = directional['color'] * directional['intensity'] * self._env_khr_light_scale
            self._env_prog['u_light_intensity'].value = (
                float(color[0]), float(color[1]), float(color[2])
            )
        else:
            light_dir = self._transform_env_direction(self._env_fallback_dir, model_mat)
            self._env_prog['u_light_dir'].value = (
                float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
            )
            self._env_prog['u_light_intensity'].value = self._env_fallback_dir_color

        fill_specs = []
        range_scale = self._env_light_range_scale()
        for light in self._scene_lights:
            if light.get('type') not in ('point', 'spot') or 'position' not in light:
                continue
            color = light['color'] * light['intensity'] * self._env_khr_light_scale
            light_range = float(light.get('range', 0.0) or 0.0)
            fill_specs.append((
                self._transform_env_point(light['position'], model_mat),
                color,
                (light_range if light_range > 0.0 else 4.0) * range_scale,
            ))
            if len(fill_specs) >= 2:
                break
        for light in self._env_fill_lights:
            if len(fill_specs) >= 2:
                break
            pos = np.array(light.get('position', (0.0, 0.0, 0.0)), dtype=np.float32)
            color = np.array(light.get('color', (0.0, 0.0, 0.0)), dtype=np.float32)
            fill_specs.append((
                self._transform_env_point(pos, model_mat),
                color,
                float(light.get('range', 1.0)) * range_scale,
            ))

        for slot in range(2):
            if slot < len(fill_specs):
                pos, color, light_range = fill_specs[slot]
                self._env_prog[f'u_fill_light_pos{slot}'].value = (
                    float(pos[0]), float(pos[1]), float(pos[2])
                )
                self._env_prog[f'u_fill_light_color{slot}'].value = (
                    float(color[0]), float(color[1]), float(color[2])
                )
                self._env_prog[f'u_fill_light_range{slot}'].value = max(float(light_range), 0.001)
            else:
                self._env_prog[f'u_fill_light_color{slot}'].value = (0.0, 0.0, 0.0)
                self._env_prog[f'u_fill_light_range{slot}'].value = 1.0

        self._apply_cinema_light_uniforms(mgl_fbo)

        glFrontFace(GL_CCW)

        fast_env = self._env_render_quality == 'fast'
        if fast_env:
            self._env_prog['u_use_normal_tex'].value = 0
            self._env_prog['u_use_occlusion_tex'].value = 0
            self._env_prog['u_use_mr_tex'].value = 0
            self._env_prog['u_use_emissive_tex'].value = 0
            self._env_prog['u_normal_scale'].value = 1.0
            self._env_prog['u_occlusion_strength'].value = 1.0
            self._env_prog['u_baked_lightmap'].value = 0

        opaque_prims = []
        blend_prims = []
        for prim in self._env_model_prims:
            rs = prim.get('_rs')
            if rs is None:
                self._prebake_prim_render_state(prim)
                rs = prim.get('_rs', {})
            if rs.get('blend', False):
                blend_prims.append(prim)
            else:
                opaque_prims.append(prim)

        if len(blend_prims) > 1:
            def _blend_sort_key(prim):
                local_center = prim.get('sort_center_local')
                if local_center is None:
                    local_center = np.zeros(3, dtype=np.float32)
                world_center = self._transform_env_point(local_center, model_mat)
                delta = world_center - cam_pos
                return float(np.dot(delta, delta))

            blend_prims.sort(key=_blend_sort_key, reverse=True)

        for prim in opaque_prims + blend_prims:
            rs = prim.get('_rs')
            if rs is None:
                continue
            if rs['double_sided']:
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                self.ctx.enable(moderngl.CULL_FACE)

            self._env_prog['u_base_color_factor'].value = rs['bc']
            self._env_prog['u_base_alpha'].value = rs['ba']
            self._env_prog['u_roughness'].value = rs['rf']
            self._env_prog['u_metallic'].value = rs['mf']
            self._env_prog['u_emissive_factor'].value = rs['ef']
            self._env_prog['u_unlit'].value = rs['unlit']
            self._env_prog['u_foliage_mode'].value = rs['foliage']
            self._env_prog['u_alpha_mode'].value = rs['am']
            self._env_prog['u_alpha_cutoff'].value = rs['ac']

            if rs['blend']:
                self.ctx.enable(moderngl.BLEND)
                self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                self.ctx.depth_mask = False
            else:
                self.ctx.disable(moderngl.BLEND)
                self.ctx.depth_mask = True

            self._env_prog['u_tex_offset'].value = rs['to']
            self._env_prog['u_tex_scale'].value = rs['ts']
            self._env_prog['u_tex_rotation'].value = rs['tr']
            self._env_prog['u_base_texcoord'].value = rs['base_tc']
            tex_key = rs['tk']
            if tex_key and tex_key in self._env_model_tex_cache:
                self._env_model_tex_cache[tex_key].use(location=3)
                self._env_prog['u_use_texture'].value = 1
            else:
                self._env_prog['u_use_texture'].value = 0

            if not fast_env:
                for uniform, location in (
                    ('normal', 4),
                    ('occlusion', 5),
                    ('mr', 6),
                    ('emissive', 7),
                ):
                    cache_key = rs[f'{uniform}_key']
                    use_name = f'u_use_{uniform}_tex'
                    if cache_key and cache_key in self._env_model_tex_cache:
                        self._env_model_tex_cache[cache_key].use(location=location)
                        self._env_prog[use_name].value = 1
                    else:
                        self._env_prog[use_name].value = 0

                self._env_prog['u_normal_scale'].value = rs['ns']
                self._env_prog['u_occlusion_strength'].value = rs['os']
                self._env_prog['u_normal_texcoord'].value = rs['normal_tc']
                self._env_prog['u_occlusion_texcoord'].value = rs['occlusion_tc']
                self._env_prog['u_mr_texcoord'].value = rs['mr_tc']
                self._env_prog['u_emissive_texcoord'].value = rs['emissive_tc']
            prim['vao'].render(rs['render_mode'])

        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self._env_prog['u_use_texture'].value = 1
        self._env_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)
        self._env_prog['u_base_alpha'].value = 1.0

        if self._env_perf_log:
            now = time.perf_counter()
            self._env_perf_accum_ms += (now - perf_t0) * 1000.0
            self._env_perf_samples += 1
            if self._env_perf_last_log <= 0.0:
                self._env_perf_last_log = now
            elif now - self._env_perf_last_log >= 5.0:
                avg_ms = self._env_perf_accum_ms / max(1, self._env_perf_samples)
                print(
                    "[OpenXRViewer] Env perf: "
                    f"fps={self.actual_fps:.1f} "
                    f"prims={len(self._env_model_prims)} "
                    f"avg_env_render={avg_ms:.2f}ms/eye "
                    f"quality={self._env_render_quality} "
                    f"shading={self._env_shading_mode}"
                )
                self._env_perf_last_log = now
                self._env_perf_accum_ms = 0.0
                self._env_perf_samples = 0
