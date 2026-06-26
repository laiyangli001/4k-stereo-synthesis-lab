# Desktop2Stereo OpenXR viewer: environment profile and runtime settings helpers.

from .implementation import *
from .constants import _BG_COLORS


class EnvironmentProfileMixin:
    """Environment profile discovery, persistence, and runtime profile controls."""

    def _discover_environment_models(self):
        """Return room folders that can be switched at runtime."""
        models = []
        root = getattr(self, '_environment_root', None)
        if not root or not os.path.isdir(root):
            return models
        try:
            for name in sorted(os.listdir(root), key=lambda v: v.lower()):
                room_dir = os.path.join(root, name)
                if not os.path.isdir(room_dir):
                    continue
                if os.path.isfile(os.path.join(room_dir, 'profile.json')) or os.path.isfile(os.path.join(room_dir, 'environment.glb')):
                    models.append(name)
        except Exception:
            pass
        selected = (getattr(self, '_environment_model', '') or '').strip()
        if selected and selected.lower() != 'default' and selected not in models:
            models.insert(0, selected)
        return models


    def _reset_environment_profile_defaults(self):
        """Reset runtime room settings before applying another profile."""
        base = getattr(self, '_env_base_settings', None)
        if not isinstance(base, dict):
            return
        self._env_model_pos = list(base['model_pos'])
        self._env_model_rot = list(base['model_rot'])
        self._env_model_scale = list(base['model_scale'])
        self._env_head_light_color = tuple(base['head_light_color'])
        self._env_ambient_color = tuple(base['ambient_color'])
        self._env_fallback_dir = np.array(base['fallback_dir'], dtype=np.float32)
        self._env_fallback_dir = self._env_fallback_dir / (np.linalg.norm(self._env_fallback_dir) + 1e-8)
        self._env_fallback_dir_color = tuple(base['fallback_dir_color'])
        self._env_fill_lights = list(base['fill_lights'])
        self._env_exposure = float(base['exposure'])
        self._env_gamma = float(base['gamma'])
        self._env_emissive_strength = float(base['emissive_strength'])
        self._env_khr_light_scale = float(base['khr_light_scale'])
        self._env_render_quality = str(base['render_quality'])
        self._env_shading_mode = str(base['shading_mode'])
        self._env_texture_anisotropy = float(base['texture_anisotropy'])
        self._env_perf_log = bool(base.get('perf_log', False))
        self._xr_render_scale = float(base['xr_render_scale'])
        self._screen_light_intensity = float(base.get('screen_light_intensity', self._screen_light_intensity))
        self._screen_light_dynamic = bool(base.get('screen_light_dynamic', False))
        self._screen_light_sample_interval = max(1, int(base.get('screen_light_sample_interval', 15)))
        self._screen_light_lerp = max(0.0, min(1.0, float(base.get('screen_light_lerp', 0.14))))


    def _configure_environment_profile(self):
        """Resolve the selected room folder and apply optional profile settings."""
        self._reset_environment_profile_defaults()
        selected = (self._environment_model or 'Default').strip() or 'Default'
        if selected.lower() == 'none':
            selected = 'Default'
        self._environment_enabled = True
        root = self._environment_root
        default_dir = os.path.join(root, 'Default')
        if selected.lower() == 'default' and os.path.isdir(default_dir):
            room_dir = default_dir
        else:
            room_dir = root if selected.lower() == 'default' else os.path.join(root, selected)
        profile_path = os.path.join(room_dir, 'profile.json')
        profile = {}

        if os.path.exists(profile_path):
            try:
                with open(profile_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            except Exception as exc:
                print(f"[OpenXRViewer] Failed to read environment profile {profile_path}: {exc}")

        glb_value = profile.get('glb', 'environment.glb')
        if glb_value in (None, '', False):
            glb_path = None
        else:
            glb_name = str(glb_value)
            glb_path = glb_name if os.path.isabs(glb_name) else os.path.join(room_dir, glb_name)
        if glb_path is not None and not os.path.exists(glb_path) and selected.lower() != 'default':
            fallback = os.path.join(root, 'environment.glb')
            print(f"[OpenXRViewer] Environment '{selected}' missing GLB, fallback to Default")
            selected = 'Default'
            room_dir = root
            glb_path = fallback
            profile = {}

        self._environment_model = selected
        self._active_environment = None if selected.lower() == 'default' else selected
        self._env_profile = profile
        self._env_model_path = glb_path

        self._env_model_pos = self._profile_vec3(profile, ('model_position', 'position'), self._env_model_pos)
        self._env_model_scale = self._profile_vec3(profile, ('model_scale', 'scale'), self._env_model_scale)
        rot_deg = profile.get('model_rotation_deg', profile.get('rotation_deg'))
        if isinstance(rot_deg, (list, tuple)) and len(rot_deg) >= 3:
            try:
                self._env_model_rot = [math.radians(float(rot_deg[0])),
                                       math.radians(float(rot_deg[1])),
                                       math.radians(float(rot_deg[2]))]
            except (TypeError, ValueError):
                pass
        else:
            self._env_model_rot = self._profile_vec3(profile, ('model_rotation', 'rotation'), self._env_model_rot)

        for key, attr in (
            ('env_exposure', '_env_exposure'),
            ('env_gamma', '_env_gamma'),
            ('env_emissive_strength', '_env_emissive_strength'),
            ('env_khr_light_scale', '_env_khr_light_scale'),
            ('khr_light_scale', '_env_khr_light_scale'),
        ):
            if key in profile:
                try:
                    setattr(self, attr, float(profile[key]))
                except (TypeError, ValueError):
                    pass

        quality = profile.get('env_render_quality', profile.get('render_quality'))
        if isinstance(quality, str):
            quality_l = quality.strip().lower()
            if quality_l in ('fast', 'balanced', 'quality'):
                self._env_render_quality = quality_l
        shading_mode = profile.get('env_shading_mode', profile.get('shading_mode'))
        if isinstance(shading_mode, str):
            shading_mode_l = shading_mode.strip().lower()
            if shading_mode_l in ('pbr', 'preview'):
                self._env_shading_mode = shading_mode_l
        if 'env_perf_log' in profile:
            self._env_perf_log = bool(profile.get('env_perf_log'))
        if 'env_texture_anisotropy' in profile:
            try:
                self._env_texture_anisotropy = max(1.0, float(profile['env_texture_anisotropy']))
            except (TypeError, ValueError):
                pass
        if 'xr_render_scale' in profile:
            try:
                self._xr_render_scale = max(0.5, min(2.0, float(profile['xr_render_scale'])))
            except (TypeError, ValueError):
                pass
        quality_filter = profile.get('screen_quality_filter', profile.get('xr_screen_quality_filter'))
        if quality_filter is not None:
            self._screen_quality_filter = bool(quality_filter)
        quality_sharpness = profile.get('screen_quality_sharpness', profile.get('xr_screen_quality_sharpness'))
        if quality_sharpness is not None:
            try:
                self._screen_quality_sharpness = max(0.0, min(1.0, float(quality_sharpness)))
            except (TypeError, ValueError):
                pass
        quality_oversample = profile.get('screen_quality_oversample', profile.get('xr_screen_quality_oversample'))
        if quality_oversample is not None:
            try:
                self._screen_quality_oversample = max(0.75, min(1.5, float(quality_oversample)))
            except (TypeError, ValueError):
                pass
        quad_layer = profile.get('xr_quad_layer_enabled', profile.get('screen_quad_layer'))
        if quad_layer is not None:
            self._xr_quad_layer_enabled = bool(quad_layer)
        quad_debug_offset = profile.get('xr_quad_layer_debug_offset')
        if quad_debug_offset is not None:
            try:
                self._xr_quad_layer_debug_offset = float(quad_debug_offset)
            except (TypeError, ValueError):
                pass
        if 'screen_light_intensity' in profile:
            try:
                self._screen_light_intensity = float(profile['screen_light_intensity'])
            except (TypeError, ValueError):
                pass
        dynamic_light = profile.get('screen_light_dynamic', profile.get('dynamic_screen_light'))
        if dynamic_light is not None:
            self._screen_light_dynamic = bool(dynamic_light)
        if 'screen_light_sample_interval' in profile:
            try:
                self._screen_light_sample_interval = max(1, int(profile['screen_light_sample_interval']))
            except (TypeError, ValueError):
                pass
        if 'screen_light_lerp' in profile:
            try:
                self._screen_light_lerp = max(0.0, min(1.0, float(profile['screen_light_lerp'])))
            except (TypeError, ValueError):
                pass
        if 'dark_room_background' in profile:
            self._dark_room_background = bool(profile.get('dark_room_background'))
        for key, attr in (
            ('glow_intensity', '_glow_intensity'),
            ('glow_width', '_glow_width_m'),
            ('glow_surround_margin', '_glow_surround_margin_m'),
            ('glow_intensity_multiplier', '_glow_intensity_multiplier'),
            ('glow_shell_intensity_multiplier', '_glow_shell_intensity_multiplier'),
            ('glow_shell_radius', '_glow_shell_radius_m'),
            ('glow_shell_height', '_glow_shell_height_m'),
            ('frosted_glow_intensity', '_frosted_glow_intensity'),
            ('frosted_glow_alpha', '_frosted_glow_alpha'),
            ('frosted_glow_threshold', '_frosted_glow_threshold'),
            ('frosted_glow_lod', '_frosted_glow_lod'),
            ('frosted_glow_margin', '_frosted_glow_margin_m'),
            ('frosted_glow_blend', '_frosted_glow_blend'),
            ('frosted_glow_thickness', '_frosted_glow_thickness'),
            ('frosted_glow_inset', '_frosted_glow_inset'),
            ('frosted_glow_diffuse', '_frosted_glow_diffuse'),
            ('frosted_veil_intensity', '_frosted_veil_intensity'),
            ('frosted_veil_alpha', '_frosted_veil_alpha'),
            ('frosted_veil_lod', '_frosted_veil_lod'),
            ('frosted_veil_threshold', '_frosted_veil_threshold'),
            ('frosted_veil_scale', '_frosted_veil_scale'),
            ('frosted_veil_beam_mix', '_frosted_veil_beam_mix'),
        ):
            if key in profile:
                try:
                    setattr(self, attr, float(profile[key]))
                except (TypeError, ValueError):
                    pass
        glow_mode = profile.get('glow_mode')
        if isinstance(glow_mode, str):
            glow_mode = glow_mode.strip().lower()
            if glow_mode in ('veil', 'frosted', 'surround', 'screen', 'off'):
                self._glow_mode = glow_mode

        self._env_head_light_color = tuple(self._profile_vec3(
            profile, ('env_head_light_color', 'head_light_color'), self._env_head_light_color))
        self._env_ambient_color = tuple(self._profile_vec3(
            profile, ('env_ambient_color', 'ambient_color'), self._env_ambient_color))
        self._env_fallback_dir = np.array(self._profile_vec3(
            profile, ('env_directional_dir', 'directional_dir'), self._env_fallback_dir), dtype=np.float32)
        self._env_fallback_dir = self._env_fallback_dir / (np.linalg.norm(self._env_fallback_dir) + 1e-8)
        self._env_fallback_dir_color = tuple(self._profile_vec3(
            profile, ('env_directional_color', 'directional_color'), self._env_fallback_dir_color))

        fill_lights = profile.get('env_fill_lights', profile.get('fallback_lights'))
        if isinstance(fill_lights, list):
            self._env_fill_lights = fill_lights

        presets = profile.get('lighting_presets')
        self._lighting_presets = [p for p in presets if isinstance(p, dict)] if isinstance(presets, list) else []
        try:
            self._lighting_preset_index = int(profile.get('lighting_preset_index', 0))
        except (TypeError, ValueError):
            self._lighting_preset_index = 0
        if self._lighting_presets:
            self._lighting_preset_index %= len(self._lighting_presets)
            self._apply_lighting_preset(self._lighting_presets[self._lighting_preset_index], log=False)

        baked_lightmap = profile.get('baked_lightmap', profile.get('baked', None))
        baked_label = f" baked_lightmap={bool(baked_lightmap)}" if baked_lightmap is not None else ""
        print(
            f"[OpenXRViewer] Environment: {self._environment_model} ({self._env_model_path}) "
            f"quality={self._env_render_quality} shading={self._env_shading_mode} "
            f"xr_scale={self._xr_render_scale:.2f}{baked_label}"
        )


    def _configure_profile_view_layout(self):
        """Cache optional room-specific viewer and screen layout settings."""
        profile = self._env_profile if isinstance(self._env_profile, dict) else {}
        view_poses = profile.get('view_poses')
        if isinstance(view_poses, list):
            self._view_pose_profiles = [p for p in view_poses if isinstance(p, dict)]
        else:
            self._view_pose_profiles = []
        try:
            self._view_pose_index = int(profile.get('view_pose_index', 0))
        except (TypeError, ValueError):
            self._view_pose_index = 0
        if self._view_pose_profiles:
            self._view_pose_index %= len(self._view_pose_profiles)
            view_pose = self._view_pose_profiles[self._view_pose_index]
        else:
            view_pose = profile.get('view_pose', profile.get('camera', {}))
        screen = profile.get('screen', {})
        self._view_pose_profile = view_pose if isinstance(view_pose, dict) else {}
        self._screen_profile = screen if isinstance(screen, dict) else {}
        if self._screen_profile:
            print(f"[OpenXRViewer] Profile screen layout enabled: {self._environment_model}")


    def _screen_profile_value(self, key, default=None):
        screen = getattr(self, '_screen_profile', {}) or {}
        return screen.get(key, default)


    def _apply_lighting_preset(self, preset, log=True):
        """Apply one profile lighting preset at runtime."""
        if not isinstance(preset, dict):
            return False
        for key, attr in (
            ('env_exposure', '_env_exposure'),
            ('env_gamma', '_env_gamma'),
            ('env_emissive_strength', '_env_emissive_strength'),
            ('env_khr_light_scale', '_env_khr_light_scale'),
            ('khr_light_scale', '_env_khr_light_scale'),
            ('screen_light_intensity', '_screen_light_intensity'),
        ):
            if key in preset:
                try:
                    setattr(self, attr, float(preset[key]))
                except (TypeError, ValueError):
                    pass
        for key, attr in (
            ('env_ambient_color', '_env_ambient_color'),
            ('ambient_color', '_env_ambient_color'),
            ('env_head_light_color', '_env_head_light_color'),
            ('head_light_color', '_env_head_light_color'),
            ('env_directional_color', '_env_fallback_dir_color'),
            ('directional_color', '_env_fallback_dir_color'),
        ):
            value = preset.get(key)
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                try:
                    setattr(self, attr, (float(value[0]), float(value[1]), float(value[2])))
                except (TypeError, ValueError):
                    pass
        fill_lights = preset.get('env_fill_lights', preset.get('fallback_lights'))
        if isinstance(fill_lights, list):
            self._env_fill_lights = fill_lights
        for key, attr in (
            ('glow_intensity', '_glow_intensity'),
            ('glow_width', '_glow_width_m'),
            ('glow_surround_margin', '_glow_surround_margin_m'),
            ('glow_intensity_multiplier', '_glow_intensity_multiplier'),
            ('glow_shell_intensity_multiplier', '_glow_shell_intensity_multiplier'),
            ('glow_shell_radius', '_glow_shell_radius_m'),
            ('glow_shell_height', '_glow_shell_height_m'),
            ('frosted_glow_intensity', '_frosted_glow_intensity'),
            ('frosted_glow_alpha', '_frosted_glow_alpha'),
            ('frosted_glow_threshold', '_frosted_glow_threshold'),
            ('frosted_glow_lod', '_frosted_glow_lod'),
            ('frosted_glow_margin', '_frosted_glow_margin_m'),
            ('frosted_glow_blend', '_frosted_glow_blend'),
            ('frosted_glow_thickness', '_frosted_glow_thickness'),
            ('frosted_glow_inset', '_frosted_glow_inset'),
            ('frosted_glow_diffuse', '_frosted_glow_diffuse'),
            ('frosted_veil_intensity', '_frosted_veil_intensity'),
            ('frosted_veil_alpha', '_frosted_veil_alpha'),
            ('frosted_veil_lod', '_frosted_veil_lod'),
            ('frosted_veil_threshold', '_frosted_veil_threshold'),
            ('frosted_veil_scale', '_frosted_veil_scale'),
            ('frosted_veil_beam_mix', '_frosted_veil_beam_mix'),
        ):
            if key in preset:
                try:
                    setattr(self, attr, float(preset[key]))
                except (TypeError, ValueError):
                    pass
        if isinstance(preset.get('glow_mode'), str):
            mode = preset.get('glow_mode', '').strip().lower()
            if mode in ('veil', 'frosted', 'surround', 'screen', 'off'):
                self._glow_mode = mode
        if log:
            name = preset.get('name', f'Preset {getattr(self, "_lighting_preset_index", 0)}')
            print(f"[OpenXRViewer] Lighting preset: {name}")
        return True


    def _settings_path(self):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'settings.yaml')


    def _persist_setting(self, key, value):
        try:
            import yaml
            from utils import read_yaml as _read_yaml
            path = self._settings_path()
            data = _read_yaml(path) if os.path.isfile(path) else {}
            if not isinstance(data, dict):
                data = {}
            data[key] = value
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
        except Exception as exc:
            print(f"[OpenXRViewer] _persist_setting({key!r}) failed: {exc}")


    def _persist_active_environment(self):
        current = (getattr(self, '_environment_model', '') or '').strip()
        if current.lower() in ('default', 'default glow', 'default with glow'):
            val = current
        elif getattr(self, '_active_environment', None):
            val = self._active_environment
        else:
            val = 'Default'
        self._persist_setting('Environment Model', val)


    def _settings_snapshot(self):
        current = (getattr(self, '_environment_model', '') or '').strip()
        if current.lower() in ('default', 'default glow', 'default with glow'):
            env_val = current
        elif getattr(self, '_active_environment', None):
            env_val = self._active_environment
        elif current:
            env_val = current
        else:
            env_val = 'Default'
        ctrl_val = getattr(self, '_current_brand', None) or getattr(self, '_controller_model', 'pico')
        return {
            'Controller Model': ctrl_val,
            'Environment Model': env_val,
        }


    def _persist_runtime_settings(self):
        """Save GUI-facing runtime settings without touching render-only state."""
        try:
            import yaml
            from utils import read_yaml as _read_yaml
            path = self._settings_path()
            data = _read_yaml(path) if os.path.isfile(path) else {}
            if not isinstance(data, dict):
                data = {}
            data.update(self._settings_snapshot())
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
            self._settings_sync_dirty = False
        except Exception as exc:
            print(f"[OpenXRViewer] _persist_runtime_settings failed: {exc}")


    def _mark_runtime_settings_dirty(self):
        self._settings_sync_dirty = True
        self._settings_sync_save_t = time.perf_counter()


    def _flush_runtime_settings_if_idle(self, delay=0.5):
        if not getattr(self, '_settings_sync_dirty', False):
            return
        if time.perf_counter() - getattr(self, '_settings_sync_save_t', 0.0) >= delay:
            self._persist_runtime_settings()


    def _builtin_profile_path(self):
        return os.path.join(self._environment_root, '.builtin_default.json')


    def _persist_screen_state(self):
        """Do not persist Default screen layout; startup should use the default preset."""
        return


    def _restore_screen_state(self):
        """Default starts from the configured screen preset, not stale saved layout."""
        return False


    def _cycle_environment(self):
        """Advance the environment one slot."""
        self._switch_environment_model()


    def _cycle_lighting_preset(self):
        """Cycle through lighting_presets in the current profile."""
        presets = getattr(self, '_lighting_presets', []) or []
        if not presets:
            return False
        self._lighting_preset_index = (int(getattr(self, '_lighting_preset_index', 0)) + 1) % len(presets)
        if isinstance(getattr(self, '_env_profile', None), dict):
            self._env_profile['lighting_preset_index'] = self._lighting_preset_index
        self._apply_lighting_preset(presets[self._lighting_preset_index])
        return True


    def _cycle_light_from_x(self):
        """Toggle lighting preset, or Default glow when no preset exists."""
        if self._cycle_lighting_preset():
            if getattr(self, '_active_environment', None) is None:
                self._save_glow_to_builtin_profile()
            else:
                self._persist_runtime_settings()
            return True
        if self._environment_screen_locked():
            print("[OpenXRViewer] Light toggle unavailable for this environment")
            return False
        current = float(getattr(self, '_glow_intensity_multiplier', 0.0))
        if current > 0.0:
            self._glow_intensity_multiplier = 0.0
            print("[OpenXRViewer] Glow: off")
        else:
            profile = getattr(self, '_env_profile', {}) if isinstance(getattr(self, '_env_profile', {}), dict) else {}
            self._glow_intensity_multiplier = float(profile.get('glow_intensity_multiplier', 1.8))
            print("[OpenXRViewer] Glow: on")
        self._save_glow_to_builtin_profile()
        return True


    def _cycle_glow_mode_from_y(self):
        """Default blank room only: surround -> screen -> off first."""
        if self._environment_screen_locked():
            return False
        env_name = str(getattr(self, '_environment_model', '') or '').strip().lower()
        if env_name not in ('default', 'none') or getattr(self, '_active_environment', None) is not None:
            return False
        modes = ('surround', 'screen', 'off', 'veil', 'frosted')
        current = str(getattr(self, '_glow_mode', 'screen') or 'screen').strip().lower()
        try:
            idx = modes.index(current)
        except ValueError:
            idx = 1
        mode = modes[(idx + 1) % len(modes)]
        profile = getattr(self, '_env_profile', {}) if isinstance(getattr(self, '_env_profile', {}), dict) else {}
        preset = None
        presets = profile.get('lighting_presets') if isinstance(profile, dict) else None
        if isinstance(presets, list):
            for candidate in presets:
                if not isinstance(candidate, dict):
                    continue
                candidate_mode = str(candidate.get('glow_mode', '') or '').strip().lower()
                if candidate_mode == mode:
                    preset = candidate
                    break
        if preset is None:
            preset = profile
        self._apply_lighting_preset(preset, log=False)
        self._glow_mode = mode
        if mode == 'off':
            self._glow_intensity_multiplier = 0.0
            self._glow_shell_intensity_multiplier = 0.0
            label = 'Glow Off'
        elif mode == 'veil':
            self._glow_intensity_multiplier = float(getattr(self, '_glow_intensity_multiplier', 1.85) or 1.85)
            self._glow_shell_intensity_multiplier = 0.0
            label = 'Frosted Veil'
        elif mode == 'frosted':
            self._glow_intensity_multiplier = float(getattr(self, '_glow_intensity_multiplier', 1.85) or 1.85)
            self._glow_shell_intensity_multiplier = 0.0
            label = 'Frosted Glow'
        elif mode == 'screen':
            self._glow_intensity_multiplier = float(getattr(self, '_glow_intensity_multiplier', 1.85) or 1.85)
            self._glow_shell_intensity_multiplier = 0.0
            label = 'Screen Glow'
        else:
            self._glow_intensity_multiplier = 0.0
            self._glow_shell_intensity_multiplier = float(getattr(self, '_glow_shell_intensity_multiplier', 1.85) or 1.85)
            label = 'Surround Glow'
        self._preset_name_overlay = label
        self._preset_osd_show_t = time.perf_counter()
        print(f"[OpenXRViewer] Glow mode: {mode}")
        self._save_glow_to_builtin_profile()
        return True


    def _cycle_view_pose(self):
        """Cycle through multi-seat view_poses in the active environment profile."""
        poses = getattr(self, '_view_pose_profiles', []) or []
        if len(poses) < 2:
            return False
        self._view_pose_index = (int(getattr(self, '_view_pose_index', 0)) + 1) % len(poses)
        self._view_pose_profile = poses[self._view_pose_index]
        if isinstance(getattr(self, '_env_profile', None), dict):
            self._env_profile['view_pose_index'] = self._view_pose_index
        env_name = getattr(self, '_active_environment', None) or getattr(self, '_environment_model', None)
        if env_name and str(env_name).strip().lower() not in ('default', 'default glow', 'default with glow', 'none'):
            profile_path = os.path.join(self._environment_root, str(env_name), 'profile.json')
            try:
                with open(profile_path, 'r', encoding='utf-8-sig') as f:
                    profile = json.load(f)
                if isinstance(profile.get('view_poses'), list):
                    profile['view_pose_index'] = self._view_pose_index
                    with open(profile_path, 'w', encoding='utf-8') as f:
                        json.dump(profile, f, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"[OpenXRViewer] Failed to save view_pose_index: {exc}")
        self._xr_profile_space_applied = False
        view_pose = self._view_pose_profile
        if isinstance(view_pose, dict):
            x = float(view_pose.get('x', 0))
            y = float(view_pose.get('y', 0))
            z = float(view_pose.get('z', 0))
            angle = float(view_pose.get('angle', 0))
            self._apply_seat_adjust_xr_space(x, y, z, angle)
        name = self._view_pose_profile.get('name', f'View {self._view_pose_index + 1}')
        self._preset_name_overlay = name
        self._preset_osd_show_t = time.perf_counter()
        print(f"[OpenXRViewer] View pose: {name} ({self._view_pose_index + 1}/{len(poses)})")
        return True


    def _env_uses_view_pose_cycle(self):
        """Return whether the active profile has multiple usable view poses."""
        return len(getattr(self, '_view_pose_profiles', []) or []) >= 2


    def _save_glow_to_builtin_profile(self):
        """Write glow settings into .builtin_default.json for the Default env."""
        builtin_path = self._builtin_profile_path()
        try:
            profile = {}
            if os.path.isfile(builtin_path):
                with open(builtin_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            profile['glow_intensity'] = float(getattr(self, '_glow_intensity', 0.65))
            profile['glow_width'] = float(getattr(self, '_glow_width_m', 0.16))
            profile['glow_surround_margin'] = float(getattr(self, '_glow_surround_margin_m', 14.0))
            profile['glow_intensity_multiplier'] = float(getattr(self, '_glow_intensity_multiplier', 0.0))
            profile['glow_shell_intensity_multiplier'] = float(getattr(self, '_glow_shell_intensity_multiplier', 0.0))
            profile['glow_shell_radius'] = float(getattr(self, '_glow_shell_radius_m', 18.0))
            profile['glow_shell_height'] = float(getattr(self, '_glow_shell_height_m', 8.5))
            profile['frosted_glow_intensity'] = float(getattr(self, '_frosted_glow_intensity', 2.2))
            profile['frosted_glow_alpha'] = float(getattr(self, '_frosted_glow_alpha', 0.42))
            profile['frosted_glow_threshold'] = float(getattr(self, '_frosted_glow_threshold', 0.46))
            profile['frosted_glow_lod'] = float(getattr(self, '_frosted_glow_lod', 5.4))
            profile['frosted_glow_margin'] = float(getattr(self, '_frosted_glow_margin_m', 3.6))
            profile['frosted_glow_blend'] = float(getattr(self, '_frosted_glow_blend', 1.35))
            profile['frosted_glow_thickness'] = float(getattr(self, '_frosted_glow_thickness', 1.6))
            profile['frosted_glow_inset'] = float(getattr(self, '_frosted_glow_inset', 0.045))
            profile['frosted_glow_diffuse'] = float(getattr(self, '_frosted_glow_diffuse', 0.85))
            profile['frosted_veil_intensity'] = float(getattr(self, '_frosted_veil_intensity', 1.65))
            profile['frosted_veil_alpha'] = float(getattr(self, '_frosted_veil_alpha', 0.58))
            profile['frosted_veil_lod'] = float(getattr(self, '_frosted_veil_lod', 6.2))
            profile['frosted_veil_threshold'] = float(getattr(self, '_frosted_veil_threshold', 0.22))
            profile['frosted_veil_scale'] = float(getattr(self, '_frosted_veil_scale', 1.22))
            profile['frosted_veil_beam_mix'] = float(getattr(self, '_frosted_veil_beam_mix', 0.22))
            profile['glow_mode'] = str(getattr(self, '_glow_mode', 'screen') or 'screen')
            with open(builtin_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[OpenXRViewer] _save_glow_to_builtin_profile failed: {exc}")


    def _passthrough_green_index(self):
        for idx, color in enumerate(_BG_COLORS):
            try:
                if color[1] >= 0.5 and color[0] <= 0.05 and color[2] <= 0.25:
                    return idx
            except (TypeError, IndexError):
                pass
        return 1 if len(_BG_COLORS) > 1 else 0


    def _toggle_passthrough_backdrop(self):
        """Toggle green passthrough backdrop without unloading the room."""
        green_idx = self._passthrough_green_index()
        if self._bg_color_idx == green_idx and self._prev_bg_color_idx is not None:
            self._bg_color_idx = self._prev_bg_color_idx
            self._prev_bg_color_idx = None
            print("[OpenXRViewer] Passthrough backdrop: off")
        else:
            if self._prev_bg_color_idx is None:
                self._prev_bg_color_idx = self._bg_color_idx
            self._bg_color_idx = green_idx
            print("[OpenXRViewer] Passthrough backdrop: on")
