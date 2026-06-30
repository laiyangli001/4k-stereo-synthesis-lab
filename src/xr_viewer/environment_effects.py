# Desktop2Stereo OpenXR viewer: screen glow and background effect helpers.

from .implementation import *


class EnvironmentEffectsMixin:
    """Screen-space room effects such as glow and background lighting."""

    def _screen_effect_model(self, width, height, z_offset=0.0, y_offset=0.0):
        sx = width / 2.0
        sy = height / 2.0
        cy = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        rot_y = np.array([[cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
        rot_z = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
        scale = np.diag([sx, sy, 1.0, 1.0]).astype('f4')
        trans = np.eye(4, dtype='f4')
        trans[0, 3] = self.screen_pan_x
        trans[1, 3] = self.screen_pan_y + y_offset
        trans[2, 3] = -self.screen_distance + z_offset
        return trans @ rot_y @ rot_x @ rot_z @ scale


    def _default_blank_fast_path(self):
        env_name = str(getattr(self, '_environment_model', '') or '').strip().lower()
        if env_name not in ('default', 'none'):
            return False
        if getattr(self, '_active_environment', None) is not None:
            return False
        if getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []):
            return False
        return (
            float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0
            and float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)) <= 0.0
        )


    def _should_render_source_screen_effects(self):
        should_show_source_border = getattr(self, '_should_show_source_border', None)
        if not callable(should_show_source_border):
            return True
        if not hasattr(self, '_runtime_direct_source'):
            return True
        return should_show_source_border()


    def _render_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_glow_intensity', 0.0)) * float(getattr(self, '_glow_intensity_multiplier', 0.0))
        if intensity <= 0.0 or self.screen_height is None:
            return
        if getattr(self, '_glow_prog', None) is None or getattr(self, '_glow_vao', None) is None:
            return

        self._advance_glow_color()
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
        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / max(float(getattr(self, '_glow_ref_screen', 2.4)), 1e-6)
        glow_width = float(getattr(self, '_glow_width_m', 0.50)) * glow_scale
        glow_extent = glow_width * 2.4
        glow_margin = glow_extent
        if (
            str(getattr(self, '_environment_model', '') or '').strip().lower() == 'default'
            and getattr(self, '_active_environment', None) is None
        ):
            surround_margin = max(
                float(getattr(self, '_glow_surround_margin_m', 14.0)),
                max(self.screen_width, self.screen_height) * 1.15,
            )
            glow_margin = max(glow_margin, surround_margin)
            glow_extent = max(glow_extent, glow_margin * 0.92)
        glow_w = self.screen_width + 2.0 * glow_margin
        glow_h = self.screen_height + 2.0 * glow_margin
        inner_w = self.screen_width / glow_w
        inner_h = self.screen_height / glow_h
        uv_scale = max(glow_w, glow_h, 1e-6)
        uv_glow_width = glow_width / uv_scale
        uv_glow_extent = glow_extent / uv_scale

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE
        model = self._screen_effect_model(glow_w, glow_h, z_offset=self._screen_back_offset(0.08))
        mvp = vp_mat @ model
        self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._glow_prog['u_screen_half'].value = (inner_w * 0.5, inner_h * 0.5)
        glow_color = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
        self._glow_prog['u_glow_color'].value = glow_color
        use_source_tex = 1 if source_tex is not None else 0
        self._glow_prog['u_glow_use_tex'].value = use_source_tex
        if source_tex is not None:
            source_tex.use(location=0)
        self._glow_prog['u_glow_width'].value = max(uv_glow_width, 1e-6)
        self._glow_prog['u_glow_extent'].value = max(uv_glow_extent, 1e-6)
        self._glow_prog['u_glow_intensity'].value = intensity
        self._glow_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_frosted_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_frosted_glow_intensity', 0.0))
        intensity *= max(
            float(getattr(self, '_glow_intensity_multiplier', 0.0)),
            float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)),
        )
        if intensity <= 0.0 or self.screen_height is None:
            return
        if getattr(self, '_frosted_glow_prog', None) is None or getattr(self, '_frosted_glow_vao', None) is None:
            return
        source_tex = getattr(self, 'color_tex', None)
        if getattr(self, '_runtime_direct_source', False):
            eye_index = int(getattr(self, '_current_eye_index', 0) or 0)
            runtime_textures = getattr(self, '_runtime_eye_textures', []) or []
            if 0 <= eye_index < len(runtime_textures) and runtime_textures[eye_index] is not None:
                source_tex = runtime_textures[eye_index]
            elif runtime_textures and runtime_textures[0] is not None:
                source_tex = runtime_textures[0]
        if source_tex is None:
            return

        beam_len = max(
            float(getattr(self, '_frosted_glow_margin_m', 3.6)),
            float(getattr(self, 'screen_distance', 0.0)) + max(4.0, max(self.screen_width, self.screen_height) * 0.25),
            max(self.screen_width, self.screen_height) * 0.16,
        )

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        sx = self.screen_width / 2.0
        sy = self.screen_height / 2.0
        cy = math.cos(self.screen_yaw)
        sy_ = math.sin(self.screen_yaw)
        cp = math.cos(self.screen_pitch)
        sp = math.sin(self.screen_pitch)
        cr = math.cos(self.screen_roll)
        sr = math.sin(self.screen_roll)
        rot_y = np.array([[cy, 0, sy_, 0], [0, 1, 0, 0], [-sy_, 0, cy, 0], [0, 0, 0, 1]], dtype='f4')
        rot_x = np.array([[1, 0, 0, 0], [0, cp, -sp, 0], [0, sp, cp, 0], [0, 0, 0, 1]], dtype='f4')
        rot_z = np.array([[cr, -sr, 0, 0], [sr, cr, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype='f4')
        scale = np.diag([sx, sy, beam_len, 1.0]).astype('f4')
        trans = np.eye(4, dtype='f4')
        trans[0, 3] = self.screen_pan_x
        trans[1, 3] = self.screen_pan_y
        trans[2, 3] = -self.screen_distance
        model = trans @ rot_y @ rot_x @ rot_z @ scale
        source_tex.use(location=0)
        self._frosted_glow_prog['u_model'].write(model.T.astype('f4').tobytes())
        self._frosted_glow_prog['u_vp'].write(vp_mat.T.astype('f4').tobytes())
        self._frosted_glow_prog['u_edge_inset'].value = float(getattr(self, '_frosted_glow_inset', 0.045))
        self._frosted_glow_prog['u_lod'].value = float(getattr(self, '_frosted_glow_lod', 5.4))
        self._frosted_glow_prog['u_threshold'].value = float(getattr(self, '_frosted_glow_threshold', 0.46))
        self._frosted_glow_prog['u_intensity'].value = intensity
        self._frosted_glow_prog['u_frost_alpha'].value = float(getattr(self, '_frosted_glow_alpha', 0.42))
        self._frosted_glow_prog['u_noise_scale'].value = 54.0
        self._frosted_glow_prog['u_beam_softness'].value = 0.34
        self._frosted_glow_prog['u_frost_blend'].value = float(getattr(self, '_frosted_glow_blend', 1.35))
        self._frosted_glow_prog['u_beam_thickness'].value = float(getattr(self, '_frosted_glow_thickness', 1.6))
        self._frosted_glow_prog['u_diffuse_scatter'].value = float(getattr(self, '_frosted_glow_diffuse', 0.85))
        self._frosted_glow_prog['u_time'].value = float(time.time())
        self._frosted_glow_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self._frosted_glow_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=4)
        self._frosted_glow_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=8)
        self._frosted_glow_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, first=12)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_frosted_veil(self, mgl_fbo, vp_mat):
        old_values = (
            float(getattr(self, '_frosted_glow_intensity', 0.0)),
            float(getattr(self, '_frosted_glow_alpha', 0.0)),
            float(getattr(self, '_frosted_glow_threshold', 0.0)),
            float(getattr(self, '_frosted_glow_lod', 0.0)),
            float(getattr(self, '_frosted_glow_blend', 0.0)),
            float(getattr(self, '_frosted_glow_thickness', 0.0)),
            float(getattr(self, '_frosted_glow_diffuse', 0.0)),
        )
        try:
            self._frosted_glow_intensity = float(getattr(self, '_frosted_veil_intensity', 1.0))
            self._frosted_glow_alpha = float(getattr(self, '_frosted_veil_alpha', 0.34))
            self._frosted_glow_threshold = float(getattr(self, '_frosted_veil_threshold', 0.28))
            self._frosted_glow_lod = float(getattr(self, '_frosted_veil_lod', 6.6))
            self._frosted_glow_blend = 2.5
            self._frosted_glow_thickness = 3.0
            self._frosted_glow_diffuse = float(getattr(self, '_frosted_glow_diffuse', 1.2))
            self._render_frosted_glow(mgl_fbo, vp_mat)
        finally:
            (
                self._frosted_glow_intensity,
                self._frosted_glow_alpha,
                self._frosted_glow_threshold,
                self._frosted_glow_lod,
                self._frosted_glow_blend,
                self._frosted_glow_thickness,
                self._frosted_glow_diffuse,
            ) = old_values


    def _render_glow_shell(self, mgl_fbo, vp_mat, intensity_multiplier=None):
        if intensity_multiplier is None:
            intensity_multiplier = float(getattr(self, '_glow_shell_intensity_multiplier', 0.0))
        intensity = float(getattr(self, '_glow_intensity', 0.0)) * float(intensity_multiplier)
        if intensity <= 0.0:
            return
        if getattr(self, '_glow_shell_prog', None) is None or getattr(self, '_glow_shell_vao', None) is None:
            return

        self._advance_glow_color()
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
        glow_tex = self._prepare_glow_downsample_texture(source_tex, source_size)
        if mgl_fbo is not None:
            mgl_fbo.use()
        center = getattr(self, '_head_pos_w', None)
        if center is None:
            center = (self.screen_pan_x, self.screen_pan_y, -self.screen_distance * 0.55)
        radius = max(float(getattr(self, '_glow_shell_radius_m', 18.0)), max(self.screen_width, self.screen_height, 1.0) * 0.85)
        height = max(float(getattr(self, '_glow_shell_height_m', 8.5)), self.screen_height * 1.8)

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        self._glow_shell_prog['u_vp'].write(vp_mat.T.astype('f4').tobytes())
        self._glow_shell_prog['u_center'].value = tuple(float(v) for v in center)
        self._glow_shell_prog['u_shell_scale'].value = (radius, height * 0.5, radius)
        self._glow_shell_prog['u_glow_color'].value = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
        use_source_tex = 1 if glow_tex is not None else 0
        self._glow_shell_prog['u_glow_use_tex'].value = use_source_tex
        if glow_tex is not None:
            glow_tex.use(location=0)
        self._glow_shell_prog['u_glow_intensity'].value = intensity
        self._glow_shell_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_screen_background_effects(self, mgl_fbo, vp_mat):
        if not self._should_render_source_screen_effects():
            return
        if self._default_blank_fast_path():
            return
        mode = str(getattr(self, '_glow_mode', 'screen') or 'screen').strip().lower()
        if mode == 'off':
            return
        if (
            float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0
            and float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)) <= 0.0
        ):
            return
        env_active = bool(getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []))
        passthrough_active = getattr(self, '_bg_color_idx', 0) == 1
        if not env_active and not passthrough_active:
            if mode == 'surround':
                self._render_glow_shell(mgl_fbo, vp_mat)
            elif mode == 'screen':
                screen_mult = float(getattr(self, '_glow_intensity_multiplier', 0.0))
                self._render_glow_shell(mgl_fbo, vp_mat, intensity_multiplier=screen_mult * 0.72)

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat):
        if not self._should_render_source_screen_effects():
            return
        mode = str(getattr(self, '_glow_mode', 'screen') or 'screen').strip().lower()
        if mode == 'off':
            return
        if (
            float(getattr(self, '_glow_intensity_multiplier', 0.0)) <= 0.0
            and float(getattr(self, '_glow_shell_intensity_multiplier', 0.0)) <= 0.0
        ):
            return
        if mode == 'veil':
            self._render_frosted_veil(mgl_fbo, vp_mat)
        elif mode == 'frosted':
            self._render_frosted_glow(mgl_fbo, vp_mat)
