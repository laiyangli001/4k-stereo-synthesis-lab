# Desktop2Stereo OpenXR viewer: environment shader and model rendering helpers.

import moderngl

from .implementation import *
from .gl_state import get_depth_mask, set_depth_mask
from .gltf import DEFAULT_GLTF_COLOR_POLICY, sort_transparent_primitives


def _view_mat_inv(view_mat):
    """Fast inverse of a rigid-body view matrix."""
    rot = view_mat[:3, :3]
    trans = view_mat[:3, 3]
    rot_t = rot.T
    inv = np.eye(4, dtype=np.float32)
    inv[:3, :3] = rot_t
    inv[:3, 3] = -(rot_t @ trans)
    return inv


def _read_radiance_hdr(path):
    with open(path, 'rb') as f:
        header = []
        while True:
            line = f.readline()
            if line == b'':
                raise ValueError('HDR header is missing resolution')
            text = line.decode('ascii', errors='replace').strip()
            if not text:
                break
            header.append(text)
        resolution = f.readline().decode('ascii', errors='replace').strip()
        parts = resolution.split()
        if len(parts) != 4 or parts[0] not in ('-Y', '+Y') or parts[2] not in ('+X', '-X'):
            raise ValueError(f'Unsupported HDR resolution: {resolution}')
        height = int(parts[1])
        width = int(parts[3])
        if width <= 0 or height <= 0:
            raise ValueError(f'Invalid HDR size: {width}x{height}')
        rgbe = np.empty((height, width, 4), dtype=np.uint8)
        use_rle = 8 <= width <= 0x7fff
        for y in range(height):
            if not use_rle:
                row = f.read(width * 4)
                if len(row) != width * 4:
                    raise ValueError('Unexpected EOF in HDR data')
                rgbe[y] = np.frombuffer(row, dtype=np.uint8).reshape(width, 4)
                continue
            scanline = f.read(4)
            if len(scanline) != 4:
                raise ValueError('Unexpected EOF in HDR scanline')
            if scanline[0] != 2 or scanline[1] != 2 or (scanline[2] & 0x80):
                rest = f.read(width * 4 - 4)
                if len(rest) != width * 4 - 4:
                    raise ValueError('Unexpected EOF in HDR data')
                rgbe[y] = np.frombuffer(scanline + rest, dtype=np.uint8).reshape(width, 4)
                use_rle = False
                continue
            row_width = (int(scanline[2]) << 8) | int(scanline[3])
            if row_width != width:
                raise ValueError(f'HDR scanline width mismatch: {row_width} != {width}')
            row = np.empty((4, width), dtype=np.uint8)
            for channel in range(4):
                x = 0
                while x < width:
                    count_b = f.read(1)
                    if not count_b:
                        raise ValueError('Unexpected EOF in HDR RLE data')
                    count = count_b[0]
                    if count > 128:
                        repeat = count - 128
                        value = f.read(1)
                        if not value or x + repeat > width:
                            raise ValueError('Invalid HDR RLE repeat')
                        row[channel, x:x + repeat] = value[0]
                        x += repeat
                    else:
                        values = f.read(count)
                        if len(values) != count or x + count > width:
                            raise ValueError('Invalid HDR RLE literal')
                        row[channel, x:x + count] = np.frombuffer(values, dtype=np.uint8)
                        x += count
            rgbe[y] = row.T
    rgb = rgbe[:, :, :3].astype(np.float32)
    exp = rgbe[:, :, 3].astype(np.int32)
    scale = np.zeros_like(exp, dtype=np.float32)
    mask = exp > 0
    scale[mask] = np.ldexp(np.ones(np.count_nonzero(mask), dtype=np.float32), exp[mask] - 136)
    return (rgb * scale[:, :, None]).astype(np.float32), (width, height)


def _hdr_to_ldr_u8(rgb):
    mapped = rgb / (1.0 + rgb)
    output_gamma = float(DEFAULT_GLTF_COLOR_POLICY.default_output_gamma)
    mapped = np.power(np.clip(mapped, 0.0, 1.0), 1.0 / max(output_gamma, 0.001))
    return np.rint(mapped * 255.0).astype(np.uint8)


class EnvironmentRendererMixin:
    """Environment shader uniforms and GL primitive rendering."""

    def _screen_light_source_texture(self):
        frame_id = int(getattr(self, '_frame_count', 0) or 0)
        if getattr(self, '_runtime_direct_source', False):
            scheduler = self._runtime_effect_submit_scheduler()
            source_tex, source_size, source_frame_id = scheduler.latest_safe_light_probe()
            cache_key = (
                frame_id,
                int(source_frame_id or 0),
                int(getattr(source_tex, 'glo', 0) or 0) if source_tex is not None else 0,
                tuple(source_size) if source_size is not None else None,
            )
            if getattr(self, '_screen_light_source_cache_key', None) == cache_key:
                self._breakdown_inc("openxr_screen_light_source_reuse")
                return getattr(self, '_screen_light_source_cache_value', (source_tex, source_size))
            record_age = getattr(self, '_record_screen_effect_safe_age', None)
            if callable(record_age):
                record_age(source_tex, source_frame_id)
            if source_tex is not None and source_size is not None:
                self._breakdown_inc("openxr_screen_light_downsample_source")
                value = (source_tex, source_size)
                self._screen_light_source_cache_key = cache_key
                self._screen_light_source_cache_frame = frame_id
                self._screen_light_source_cache_value = value
                return value
            value = (None, None)
            self._screen_light_source_cache_key = cache_key
            self._screen_light_source_cache_frame = frame_id
            self._screen_light_source_cache_value = value
            return value
        source_tex = getattr(self, 'color_tex', None)
        source_size = getattr(self, '_texture_size', None)
        cache_key = (
            frame_id,
            int(getattr(source_tex, 'glo', 0) or 0) if source_tex is not None else 0,
            tuple(source_size) if source_size is not None else None,
        )
        if getattr(self, '_screen_light_source_cache_key', None) == cache_key:
            self._breakdown_inc("openxr_screen_light_source_reuse")
            return getattr(self, '_screen_light_source_cache_value', (source_tex, source_size))
        cached_light_tex = self._cached_glow_downsample_texture(source_tex, source_size)
        cached_light_size = getattr(self, '_glow_ds_size', None)
        if cached_light_tex is not None and cached_light_size is not None:
            self._breakdown_inc("openxr_screen_light_downsample_source")
            value = (cached_light_tex, cached_light_size)
            self._screen_light_source_cache_key = cache_key
            self._screen_light_source_cache_frame = frame_id
            self._screen_light_source_cache_value = value
            return value
        value = (None, None)
        self._screen_light_source_cache_key = cache_key
        self._screen_light_source_cache_frame = frame_id
        self._screen_light_source_cache_value = value
        return value

    def _bind_screen_light_source_texture(self, location=8):
        source_tex, _source_size = self._screen_light_source_texture()
        if source_tex is None:
            return None
        try:
            source_tex.use(location=location)
        except Exception as exc:
            print(f"[OpenXRViewer] Screen light texture bind failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_screen_light_bind_failed")
            return None
        return source_tex

    def _apply_cinema_light_uniforms(self, mgl_fbo=None):
        """Push current screen area-light uniforms to the environment shader."""
        if self.screen_height is None or self._screen_light_intensity <= 0.0:
            self._env_prog['u_screen_light_enabled'].value = 0
            self._cl_light_state_key = None
            self._cl_uniform_frame = -5
            return
        if self._bind_screen_light_source_texture() is None:
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
        state_key = (
            pose_key,
            getattr(self, '_active_environment', None),
            float(self._screen_light_intensity),
        )
        last_state_key = getattr(self, '_cl_light_state_key', None)
        last_frame = getattr(self, '_cl_uniform_frame', -999)
        if state_key == last_state_key and (fc - last_frame) < 5:
            return
        self._cl_light_state_key = state_key
        self._cl_uniform_frame = fc
        sc = getattr(self, '_glow_color', (0.30, 0.55, 1.0))
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
        self._env_prog['u_screen_light_intensity'].value = intensity


    def _get_panorama_texture(self):
        path = getattr(self, '_panorama_background_path', None)
        if not path:
            return None
        path = os.path.abspath(path)
        if self._panorama_tex is not None and self._panorama_tex_path == path:
            return self._panorama_tex
        if self._panorama_tex is not None:
            try:
                self._panorama_tex.release()
            except Exception:
                pass
            self._panorama_tex = None
            self._panorama_tex_path = None
        try:
            max_tex = int(getattr(self.ctx, 'info', {}).get('GL_MAX_TEXTURE_SIZE', 8192) or 8192)
            if os.path.splitext(path)[1].lower() == '.hdr':
                arr, size = _read_radiance_hdr(path)
                if max(size) > max_tex:
                    raise ValueError(f'HDR texture exceeds GL_MAX_TEXTURE_SIZE: {size[0]}x{size[1]} > {max_tex}')
                try:
                    tex = self.ctx.texture(size, 3, np.asarray(arr, dtype=np.float16).tobytes(), dtype='f2')
                except Exception as exc:
                    print(f"[OpenXRViewer] HDR panorama float texture unavailable, fallback to LDR: {exc}")
                    arr = _hdr_to_ldr_u8(arr)
                    tex = self.ctx.texture(size, 3, arr.tobytes())
            else:
                img = Image.open(path).convert('RGB')
                if max(img.size) > max_tex:
                    scale = float(max_tex) / float(max(img.size))
                    new_size = (
                        max(1, int(round(img.size[0] * scale))),
                        max(1, int(round(img.size[1] * scale))),
                    )
                    resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
                    img = img.resize(new_size, resample)
                arr = np.asarray(img, dtype=np.uint8)
                size = img.size
                tex = self.ctx.texture(size, 3, arr.tobytes())
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            try:
                tex.repeat_x = True
                tex.repeat_y = False
            except Exception:
                pass
            try:
                tex.build_mipmaps()
            except Exception:
                tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._panorama_tex = tex
            self._panorama_tex_path = path
            print(f"[OpenXRViewer] Panorama background loaded: {path} ({size[0]}x{size[1]})")
            return tex
        except Exception as exc:
            print(f"[OpenXRViewer] Panorama background load failed: {exc}")
            return None


    def _panorama_texture_ready(self):
        path = getattr(self, '_panorama_background_path', None)
        if not path or self._panorama_tex is None:
            return None
        if self._panorama_tex_path != os.path.abspath(path):
            return None
        return self._panorama_tex


    def _get_panorama_light_mask_texture(self):
        path = self._panorama_light_mask_path_from_settings()
        if path is None:
            self._record_panorama_light_mask_disabled()
            return None
        if self._panorama_light_mask_tex is not None and self._panorama_light_mask_path == path:
            return self._panorama_light_mask_tex
        if self._panorama_light_mask_tex is not None:
            try:
                self._panorama_light_mask_tex.release()
            except Exception:
                pass
            self._panorama_light_mask_tex = None
            self._panorama_light_mask_path = None
            self._panorama_light_mask_missing_path = None
        if not os.path.isfile(path):
            if getattr(self, '_panorama_light_mask_missing_path', None) != path:
                print(f"[OpenXRViewer] Panorama wall light mask missing: {path}")
                self._panorama_light_mask_missing_path = path
                self._breakdown_inc("openxr_wall_light_mask_missing")
            return None
        try:
            img = Image.open(path).convert('L')
            tex = self.ctx.texture(img.size, 1, np.asarray(img, dtype=np.uint8).tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._panorama_light_mask_tex = tex
            self._panorama_light_mask_path = path
            self._panorama_light_mask_missing_path = None
            print(f"[OpenXRViewer] Panorama wall light mask loaded: {path} ({img.size[0]}x{img.size[1]})")
            self._breakdown_inc("openxr_wall_light_mask_loaded")
            return tex
        except Exception as exc:
            print(f"[OpenXRViewer] Panorama wall light mask load failed: {exc}")
            self._breakdown_inc("openxr_wall_light_mask_failed")
            return None

    def _panorama_light_mask_path_from_settings(self):
        settings = getattr(self, '_panorama_background_settings', {}) or {}
        mask_name = settings.get('wall_light_mask') or settings.get('light_mask')
        if not mask_name:
            return None
        cache_key = (id(settings), str(mask_name), getattr(self, '_panorama_background_path', None))
        if getattr(self, '_panorama_light_mask_path_key', None) == cache_key:
            return getattr(self, '_panorama_light_mask_resolved_path', None)
        base_path = getattr(self, '_panorama_background_path', None)
        base_dir = os.path.dirname(os.path.abspath(base_path)) if base_path else os.getcwd()
        path = str(mask_name)
        path = path if os.path.isabs(path) else os.path.join(base_dir, path)
        path = os.path.abspath(path)
        self._panorama_light_mask_path_key = cache_key
        self._panorama_light_mask_resolved_path = path
        return path


    def _record_panorama_light_mask_disabled(self):
        settings = getattr(self, '_panorama_background_settings', {}) or {}
        key = (id(settings), getattr(self, '_panorama_background_path', None))
        if getattr(self, '_panorama_light_mask_disabled_key', None) == key:
            return
        self._panorama_light_mask_disabled_key = key
        self._breakdown_inc("openxr_wall_light_mask_disabled")


    def _panorama_light_mask_texture_ready(self):
        tex = getattr(self, '_panorama_light_mask_tex', None)
        if tex is None:
            return None
        path = self._panorama_light_mask_path_from_settings()
        if path is None:
            self._record_panorama_light_mask_disabled()
            return None
        if getattr(self, '_panorama_light_mask_path', None) != path:
            return None
        return tex

    def _panorama_render_settings(self):
        settings = getattr(self, '_panorama_background_settings', {}) or {}
        cache_key = (
            id(settings),
            settings.get('yaw_offset_deg'),
            settings.get('exposure'),
            settings.get('flip_y'),
            settings.get('stereo_layout'),
            settings.get('layout'),
            repr(settings.get('screen_light_layout')),
            repr(settings.get('screen_light_uv')),
            repr(settings.get('screen_light_radius')),
        )
        if getattr(self, '_panorama_render_settings_key', None) == cache_key:
            return self._panorama_render_settings_value

        try:
            yaw_offset = math.radians(float(settings.get('yaw_offset_deg', 0.0))) / (2.0 * math.pi)
        except (TypeError, ValueError):
            yaw_offset = 0.0
        try:
            exposure = float(settings.get('exposure', 1.0))
        except (TypeError, ValueError):
            exposure = 1.0
        flip_y = 1 if bool(settings.get('flip_y', False)) else 0
        stereo_layout_raw = str(settings.get('stereo_layout', settings.get('layout', 'mono')) or 'mono').strip().lower()
        stereo_layout = 1 if stereo_layout_raw in ('sbs', 'side_by_side', 'side-by-side', 'stereo_sbs') else 0
        light_layout = settings.get('screen_light_layout')
        if not isinstance(light_layout, dict):
            light_layout = {}
        light_uv = light_layout.get('uv', light_layout.get('center', settings.get('screen_light_uv', (0.5, 0.58))))
        if not isinstance(light_uv, (list, tuple)) or len(light_uv) < 2:
            light_uv = (0.5, 0.58)
        try:
            light_uv = (float(light_uv[0]), float(light_uv[1]))
        except (TypeError, ValueError):
            light_uv = (0.5, 0.58)
        light_radius = light_layout.get('radius', light_layout.get('size', settings.get('screen_light_radius', (0.18, 0.11))))
        if not isinstance(light_radius, (list, tuple)) or len(light_radius) < 2:
            light_radius = (0.18, 0.11)
        try:
            light_radius = (max(0.001, float(light_radius[0])), max(0.001, float(light_radius[1])))
        except (TypeError, ValueError):
            light_radius = (0.18, 0.11)

        value = (yaw_offset, exposure, flip_y, stereo_layout, light_uv, light_radius)
        self._panorama_render_settings_key = cache_key
        self._panorama_render_settings_value = value
        return value


    def _render_panorama_background(self, mgl_fbo, view_mat, proj_mat):
        if self._panorama_prog is None or self._panorama_vao is None:
            return False
        tex = self._panorama_texture_ready()
        if tex is None:
            return False
        yaw_offset, exposure, flip_y, stereo_layout, light_uv, light_radius = self._panorama_render_settings()

        view_rot = np.array(view_mat, dtype=np.float32, copy=True)
        view_rot[:3, 3] = 0.0
        try:
            inv_proj = np.linalg.inv(proj_mat.astype(np.float32))
            inv_view_rot = _view_mat_inv(view_rot)
        except Exception:
            return False

        mgl_fbo.use()
        previous_depth_mask = get_depth_mask()
        try:
            self.ctx.disable(moderngl.DEPTH_TEST)
            set_depth_mask(False)
            self.ctx.disable(moderngl.BLEND)
            self.ctx.disable(moderngl.CULL_FACE)
            glFrontFace(GL_CCW)
            tex.use(location=8)
            screen_tex = self._bind_screen_light_source_texture(location=10)
            mask_tex = self._panorama_light_mask_texture_ready()
            if mask_tex is not None:
                mask_tex.use(location=11)
            self._panorama_prog['u_inv_proj'].write(inv_proj.T.astype('f4').tobytes())
            self._panorama_prog['u_inv_view_rot'].write(inv_view_rot.T.astype('f4').tobytes())
            self._panorama_prog['u_yaw_offset'].value = float(yaw_offset)
            self._panorama_prog['u_exposure'].value = max(0.0, float(exposure))
            self._panorama_prog['u_flip_y'].value = flip_y
            self._panorama_prog['u_stereo_layout'].value = stereo_layout
            self._panorama_prog['u_eye_index'].value = 1 if int(getattr(self, '_current_eye_index', 0) or 0) == 1 else 0
            self._panorama_prog['u_screen_light_enabled'].value = 1 if screen_tex is not None else 0
            self._panorama_prog['u_wall_light_mask_enabled'].value = 1 if mask_tex is not None else 0
            self._panorama_prog['u_screen_light_intensity'].value = max(
                0.0,
                float(getattr(self, '_screen_light_intensity', 0.0) or 0.0) * 0.12,
            )
            self._panorama_prog['u_screen_light_uv'].value = light_uv
            self._panorama_prog['u_screen_light_radius'].value = light_radius
            self._panorama_vao.render(moderngl.TRIANGLE_STRIP)
            return True
        except Exception as exc:
            print(f"[OpenXRViewer] Panorama background render failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_background_panorama_failed")
            return False
        finally:
            self.ctx.disable(moderngl.BLEND)
            self.ctx.disable(moderngl.CULL_FACE)
            set_depth_mask(previous_depth_mask)
            self.ctx.enable(moderngl.DEPTH_TEST)
            glFrontFace(GL_CCW)


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

        previous_depth_mask = get_depth_mask()
        glFrontFace(GL_CCW)

        try:
            fast_env = self._env_render_quality == 'fast'
            if fast_env:
                self._env_prog['u_use_normal_tex'].value = 0
                self._env_prog['u_use_occlusion_tex'].value = 0
                self._env_prog['u_use_mr_tex'].value = 0
                self._env_prog['u_use_emissive_tex'].value = 0
                self._env_prog['u_normal_scale'].value = 1.0
                self._env_prog['u_occlusion_strength'].value = 1.0
                self._env_prog['u_baked_lightmap'].value = 0

            sky_prims = []
            opaque_prims = []
            blend_prims = []
            for prim in self._env_model_prims:
                rs = prim.get('_rs')
                if rs is None:
                    self._prebake_prim_render_state(prim)
                    rs = prim.get('_rs', {})
                if rs.get('render_pass') == 'sky':
                    sky_prims.append(prim)
                elif rs.get('blend', False):
                    blend_prims.append(prim)
                else:
                    opaque_prims.append(prim)

            blend_prims = sort_transparent_primitives(blend_prims, cam_pos, model_mat)

            for prim in sky_prims + opaque_prims + blend_prims:
                rs = prim.get('_rs')
                if rs is None:
                    continue
                is_sky = rs['render_pass'] == 'sky'
                if is_sky or rs['double_sided']:
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
                self._env_prog['u_alpha_mode'].value = 0 if is_sky else rs['am']
                self._env_prog['u_double_sided'].value = 1 if is_sky or rs['double_sided'] else 0
                self._env_prog['u_alpha_cutoff'].value = rs['ac']

                if is_sky:
                    self.ctx.disable(moderngl.BLEND)
                    set_depth_mask(False)
                elif rs['blend']:
                    self.ctx.enable(moderngl.BLEND)
                    self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
                    set_depth_mask(False)
                else:
                    self.ctx.disable(moderngl.BLEND)
                    set_depth_mask(True)

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
        except Exception as exc:
            print(f"[OpenXRViewer] Environment model render failed: {type(exc).__name__}: {exc}")
            self._breakdown_inc("openxr_background_env_model_failed")
            return
        finally:
            self.ctx.disable(moderngl.CULL_FACE)
            self.ctx.disable(moderngl.BLEND)
            set_depth_mask(previous_depth_mask)
            glFrontFace(GL_CCW)
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
