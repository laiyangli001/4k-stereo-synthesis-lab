# xrviewer_env.py
# Desktop2Stereo OpenXR viewer: room/environment profile.
# Shared runtime/rendering code is in xrviewer_core.py; room-specific code lives here.

from .implementation import *
from .constants import _BG_COLORS
from .overlay import OverlayMixin
from .render import _view_mat_inv


class OpenXRViewer(OpenXRViewerCore, OverlayMixin):
    """Room/environment viewer.

    This class keeps the environment-specific behavior separate from the normal
    no-room viewer: room discovery, profile.json layout, GLB room loading,
    environment switching, and environment rendering.
    """

    ENVIRONMENT_MODE = True
    DEFAULT_ENVIRONMENT_MODEL = 'Default'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('environment_model', self.DEFAULT_ENVIRONMENT_MODEL)
        super().__init__(*args, **kwargs)

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


    def _configure_environment_profile(self):
        """Resolve the selected room folder and apply optional profile settings."""
        self._reset_environment_profile_defaults()
        selected = (self._environment_model or 'Default').strip() or 'Default'
        if selected.lower() == 'none':
            self._environment_enabled = False
            self._environment_model = 'None'
            self._env_profile = {}
            self._env_model_path = None
            self._env_model_visible = False
            return
        self._environment_enabled = True
        root = self._environment_root
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

        glb_name = str(profile.get('glb', 'environment.glb') or 'environment.glb')
        glb_path = glb_name if os.path.isabs(glb_name) else os.path.join(room_dir, glb_name)
        if not os.path.exists(glb_path) and selected.lower() != 'default':
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
                self._xr_render_scale = max(0.5, min(1.0, float(profile['xr_render_scale'])))
            except (TypeError, ValueError):
                pass
        if 'screen_light_intensity' in profile:
            try:
                self._screen_light_intensity = float(profile['screen_light_intensity'])
            except (TypeError, ValueError):
                pass

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
            ('glow_intensity_multiplier', '_glow_intensity_multiplier'),
        ):
            if key in preset:
                try:
                    setattr(self, attr, float(preset[key]))
                except (TypeError, ValueError):
                    pass
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
            'Depth Strength': round(float(getattr(self, 'depth_ratio', 1.0)), 4),
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
            self._last_persisted_depth_ratio = float(getattr(self, 'depth_ratio', 1.0))
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
        """Save Default-environment screen layout to .builtin_default.json."""
        if self._environment_screen_locked():
            return
        if getattr(self, '_active_environment', None) is not None:
            return
        state = {
            'width': round(float(self.screen_width), 4),
            'distance': round(float(self.screen_distance), 4),
            'pan_x': round(float(self.screen_pan_x), 4),
            'pan_y': round(float(self.screen_pan_y), 4),
            'yaw': round(float(self.screen_yaw), 6),
            'pitch': round(float(self.screen_pitch), 6),
            'curved': bool(self._screen_curved),
            'preset_index': int(self._preset_index),
        }
        builtin_path = self._builtin_profile_path()
        try:
            profile = {}
            if os.path.isfile(builtin_path):
                with open(builtin_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    profile = loaded
            profile['screen_state'] = state
            with open(builtin_path, 'w', encoding='utf-8') as f:
                json.dump(profile, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[OpenXRViewer] _persist_screen_state failed: {exc}")


    def _restore_screen_state(self):
        """Load Default-environment screen layout from .builtin_default.json."""
        if self._environment_screen_locked():
            return False
        if getattr(self, '_active_environment', None) is not None:
            return False
        state = None
        builtin_path = self._builtin_profile_path()
        try:
            if os.path.isfile(builtin_path):
                with open(builtin_path, 'r', encoding='utf-8-sig') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    state = loaded.get('screen_state')
        except Exception:
            pass
        if not isinstance(state, dict):
            try:
                from utils import read_yaml as _read_yaml
                cfg = _read_yaml(self._settings_path())
                state = cfg.get('Screen State')
            except Exception:
                pass
        if not isinstance(state, dict):
            return False
        self.screen_width = float(state.get('width', self.screen_width))
        self._screen_ref_size = self.screen_width
        self.screen_height = None
        self.screen_distance = float(state.get('distance', self.screen_distance))
        self.screen_pan_x = float(state.get('pan_x', self.screen_pan_x))
        self.screen_pan_y = float(state.get('pan_y', self.screen_pan_y))
        self.screen_yaw = float(state.get('yaw', self.screen_yaw))
        self.screen_pitch = float(state.get('pitch', self.screen_pitch))
        self._screen_curved = bool(state.get('curved', False))
        self._preset_index = int(state.get('preset_index', self._preset_index))
        print(f"[OpenXRViewer] Restored screen state: {state.get('width')}m, dist={state.get('distance')}, curved={state.get('curved')}")
        return True


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
            self._glow_intensity_multiplier = 1.5
            print("[OpenXRViewer] Glow: on")
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
        views = getattr(self, '_last_located_views', None)
        if views:
            self._apply_profile_view_pose_to_xr_space(views)
        self._seat_adjust_osd_dirty = True
        self._seat_adjust_osd_show_t = time.perf_counter()
        name = self._view_pose_profile.get('name', f'View {self._view_pose_index + 1}')
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
            profile['glow_intensity_multiplier'] = float(getattr(self, '_glow_intensity_multiplier', 0.0))
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


    def _environment_screen_locked(self):
        screen = getattr(self, '_screen_profile', {}) or {}
        return isinstance(screen, dict) and bool(screen)


    def _head_model_mat4_from_views(self, views):
        if not views or len(views) < 2 or views[0] is None or views[1] is None:
            return None
        head_mat = xr_pose_to_model_mat4(views[0].pose)
        try:
            p0 = views[0].pose.position
            p1 = views[1].pose.position
            head_mat[:3, 3] = np.array([
                (p0.x + p1.x) / 2.0,
                (p0.y + p1.y) / 2.0,
                (p0.z + p1.z) / 2.0,
            ], dtype=np.float32)
        except Exception:
            return None
        return head_mat


    def _level_head_model_mat4(self, head_mat):
        """Keep head position and yaw, dropping pitch/roll so the room stays level."""
        if head_mat is None:
            return None
        pos = head_mat[:3, 3].copy()
        forward = -head_mat[:3, 2].astype(np.float32)
        forward[1] = 0.0
        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            yaw = 0.0
        else:
            forward = forward / norm
            yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        leveled = euler_to_mat4(yaw, 0.0, 0.0).astype(np.float32)
        leveled[:3, 3] = pos
        return leveled


    def _apply_profile_view_pose_to_xr_space(self, views):
        if self._xr_profile_space_applied:
            return False

        self._xr_profile_space_applied = True
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view:
            return False

        pos_keys = ('position', 'camera_position', 'viewer_position')
        rot_deg_keys = ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg')
        rot_rad_keys = ('rotation', 'camera_rotation', 'viewer_rotation')
        has_pos = any(key in view for key in pos_keys)
        has_rot = any(key in view for key in rot_deg_keys + rot_rad_keys)
        auto_center = bool(view.get('auto_center_on_screen', False))
        if not (has_pos or has_rot or auto_center):
            return False
        if self._xr_session is None or self._xr_space is None or self._xr_ref_space_type is None:
            return False

        raw_head = self._head_model_mat4_from_views(views)
        if raw_head is None:
            self._xr_profile_space_applied = False
            return False

        desired_head = raw_head.copy()
        if auto_center:
            auto_pos = self._auto_view_position_from_screen(view, has_rot, rot_deg_keys, rot_rad_keys)
            if auto_pos is None and has_pos:
                auto_pos = self._profile_vec3(view, pos_keys, raw_head[:3, 3].tolist())
            if auto_pos is not None:
                desired_head[:3, 3] = np.array(auto_pos, dtype=np.float32)
        elif has_pos:
            desired_head[:3, 3] = np.array(
                self._profile_vec3(view, pos_keys, raw_head[:3, 3].tolist()),
                dtype=np.float32,
            )
        if has_rot:
            rot = self._profile_rotation_rad(view, rot_deg_keys, rot_rad_keys, [0.0, 0.0, 0.0])
            desired_head[:3, :3] = euler_to_mat4(*rot)[:3, :3]

        try:
            current_space_in_ref = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
            reference_head = current_space_in_ref @ raw_head
            if auto_center:
                reference_head = self._level_head_model_mat4(reference_head)
            space_in_ref = reference_head @ np.linalg.inv(desired_head)
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=mat4_to_xr_posef(space_in_ref.astype(np.float32)),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to apply profile view_pose to XrSpace: {exc}")
            return False

        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = space_in_ref.astype(np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass

        pos = [round(float(v), 4) for v in desired_head[:3, 3]]
        print(f"[OpenXRViewer] Applied profile view_pose to XrSpace: position={pos}")
        return True


    def _reset_xr_space_to_identity(self):
        """Reset XR reference space offset to identity."""
        if self._xr_session is None or self._xr_ref_space_type is None:
            return
        current = getattr(self, '_xr_space_pose_in_ref', np.eye(4, dtype=np.float32))
        if np.allclose(current, np.eye(4)):
            return
        try:
            new_space = xr.create_reference_space(
                self._xr_session,
                xr.ReferenceSpaceCreateInfo(
                    reference_space_type=self._xr_ref_space_type,
                    pose_in_reference_space=xr.Posef(),
                ),
            )
        except Exception as exc:
            print(f"[OpenXRViewer] Failed to reset XR space: {exc}")
            return
        old_space = self._xr_space
        self._xr_space = new_space
        self._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)
        if old_space is not None:
            try:
                xr.destroy_space(old_space)
            except Exception:
                pass
        print("[OpenXRViewer] XR space reset to identity")


    def _recenter_profile_view_pose(self):
        """Re-apply profile view_pose, used by Home/Menu long press."""
        view = getattr(self, '_view_pose_profile', {}) or {}
        if not isinstance(view, dict) or not view.get('auto_center_on_screen', False):
            return False
        views = getattr(self, '_last_located_views', None)
        if not views or views[0] is None or views[1] is None:
            return False
        self._xr_profile_space_applied = False
        applied = self._apply_profile_view_pose_to_xr_space(views)
        if applied:
            print("[OpenXRViewer] Home recentered view_pose to screen center")
        return applied


    def _auto_view_position_from_screen(self, view, has_view_rot, rot_deg_keys, rot_rad_keys):
        """Compute a viewer position centered on the configured screen."""
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return None

        position = screen.get('position', screen.get('screen_position'))
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            return None
        try:
            screen_pos = np.array(
                [float(position[0]), float(position[1]), float(position[2])],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return None

        width = self._profile_float(screen, ('width', 'screen_width'), self.screen_width)
        ratio = self._profile_float(view, ('distance_width_ratio', 'view_distance_width_ratio'), 0.6)
        distance = self._profile_float(view, ('distance', 'view_distance'), width * ratio)
        distance = max(0.05, float(distance))

        if has_view_rot:
            yaw, pitch, _roll = self._profile_rotation_rad(
                view, rot_deg_keys, rot_rad_keys, [0.0, 0.0, 0.0]
            )
            cp = math.cos(pitch)
            forward = np.array(
                [-math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp],
                dtype=np.float32,
            )
        else:
            screen_rot = self._profile_rotation_rad(
                screen,
                ('rotation_deg', 'screen_rotation_deg'),
                ('rotation', 'screen_rotation'),
                [self.screen_yaw, self.screen_pitch, self.screen_roll],
            )
            screen_normal = euler_to_mat4(*screen_rot)[:3, 2].astype(np.float32)
            forward = -screen_normal

        norm = float(np.linalg.norm(forward))
        if norm < 1e-6:
            return None
        forward = forward / norm

        pos = screen_pos - forward * distance
        right = np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        right_norm = float(np.linalg.norm(right))
        if right_norm > 1e-6:
            right = right / right_norm
            pos += right * self._profile_float(view, ('horizontal_offset', 'x_offset'), 0.0)
        pos[1] += self._profile_float(view, ('vertical_offset', 'y_offset'), 0.0)
        return [float(pos[0]), float(pos[1]), float(pos[2])]


    def _apply_profile_screen_layout(self, show_border=False):
        """Apply fixed room-specific screen layout from profile.json.

        This does not lock the physical headset pose.  OpenXR head tracking stays
        active; the profile only defines where the virtual screen is placed in
        the room.
        """
        screen = getattr(self, '_screen_profile', {}) or {}
        if not isinstance(screen, dict) or not screen:
            return False

        self._reset_orientation_offsets()
        width = self._profile_float(screen, ('width', 'screen_width'), self.screen_width)
        self.screen_width = max(0.05, width)
        self._screen_ref_size = self.screen_width
        self.screen_height = None


        rotation = self._profile_rotation_rad(
            screen,
            ('rotation_deg', 'screen_rotation_deg'),
            ('rotation', 'screen_rotation'),
            [self.screen_yaw, self.screen_pitch, self.screen_roll],
        )

        position = screen.get('position', screen.get('screen_position'))
        if isinstance(position, (list, tuple)) and len(position) >= 3:
            try:
                x, y, z = float(position[0]), float(position[1]), float(position[2])
                self.screen_pan_x = x
                self.screen_pan_y = y
                self.screen_distance = max(0.05, -z)
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            except (TypeError, ValueError):
                return False
        else:
            view = getattr(self, '_view_pose_profile', {}) or {}
            view_pos = self._profile_vec3(
                view,
                ('position', 'camera_position', 'viewer_position'),
                [0.0, float(self._initial_head_y), 0.0],
            )
            view_rot = self._profile_rotation_rad(
                view,
                ('rotation_deg', 'camera_rotation_deg', 'viewer_rotation_deg'),
                ('rotation', 'camera_rotation', 'viewer_rotation'),
                [0.0, 0.0, 0.0],
            )
            yaw, pitch, _roll = view_rot
            distance = self._profile_float(screen, ('distance', 'screen_distance'), self.screen_distance)
            cp = math.cos(pitch)
            fx = -math.sin(yaw) * cp
            fy = math.sin(pitch)
            fz = -math.cos(yaw) * cp
            self.screen_pan_x = float(view_pos[0] + fx * distance)
            self.screen_pan_y = float(view_pos[1] + fy * distance)
            self.screen_distance = max(0.05, -(float(view_pos[2]) + fz * distance))
            if 'rotation_deg' in screen or 'screen_rotation_deg' in screen or 'rotation' in screen or 'screen_rotation' in screen:
                self.screen_yaw, self.screen_pitch, self.screen_roll = rotation
            else:
                self.screen_yaw = math.atan2(-fx, -fz)
                self.screen_pitch = 0.0
                self.screen_roll = 0.0

        self._last_overlay_update = 0.0
        self._border_alpha = 0.0
        if self._keyboard_visible:
            self._anchor_keyboard_below_screen()
        return True


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
            'blend': alpha_mode == 'BLEND',
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
                    [(vbo, '3f 3f 2f 2f', 'in_position', 'in_normal', 'in_uv', 'in_uv1'),
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
                    'tex_key': tex_key,
                    'render_mode': gltf_primitive_mode_to_moderngl(pd.get('primitive_mode', 4)),
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
                'tex_key': None, 'tri_count': 2, 'color': color,
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
        path = self._env_model_path or os.path.join(self._environment_root, 'environment.glb')
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
        self._init_env_model()
        self._apply_profile_screen_layout(show_border=True)
        self._xr_profile_space_applied = False
        views = getattr(self, '_last_located_views', None)
        if views:
            self._apply_profile_view_pose_to_xr_space(views)
        if not self._environment_screen_locked():
            self._reset_xr_space_to_identity()
            if not self._restore_screen_state():
                self._reset_screen_to_default(show_border=True)
        self._persist_runtime_settings()
        return True


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


    def _render_glow(self, mgl_fbo, vp_mat):
        intensity = float(getattr(self, '_glow_intensity', 0.0)) * float(getattr(self, '_glow_intensity_multiplier', 0.0))
        if intensity <= 0.0 or self.screen_height is None:
            return
        if getattr(self, '_screen_curved', False):
            return
        if getattr(self, '_glow_prog', None) is None or getattr(self, '_glow_vao', None) is None:
            return

        self._advance_glow_color()
        screen_long = max(self.screen_width, self.screen_height)
        glow_scale = screen_long / max(float(getattr(self, '_glow_ref_screen', 2.4)), 1e-6)
        glow_width = float(getattr(self, '_glow_width_m', 0.50)) * glow_scale
        glow_range = glow_width * 0.75
        glow_margin = glow_range
        glow_w = self.screen_width + 2.0 * glow_margin
        glow_h = self.screen_height + 2.0 * glow_margin
        inner_w = self.screen_width / glow_w
        inner_h = self.screen_height / glow_h
        uv_glow_range = glow_range / max(glow_w, glow_h, 1e-6)

        self.ctx.depth_mask = False
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.ONE, moderngl.ONE_MINUS_SRC_ALPHA
        model = self._screen_effect_model(glow_w, glow_h, z_offset=-0.002)
        mvp = vp_mat @ model
        self._glow_prog['u_mvp'].write(mvp.T.astype('f4').tobytes())
        self._glow_prog['u_screen_half'].value = (inner_w * 0.5, inner_h * 0.5)
        self._glow_prog['u_glow_color'].value = tuple(getattr(self, '_glow_color', (0.30, 0.55, 1.0)))
        self._glow_prog['u_glow_inv_range'].value = 1.0 / max(uv_glow_range, 1e-6)
        self._glow_prog['u_glow_inv_density_range'].value = 1.0 / max(uv_glow_range * 0.75, 1e-6)
        self._glow_prog['u_glow_intensity'].value = intensity
        self._glow_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_mask = True
        self.ctx.enable(moderngl.DEPTH_TEST)


    def _render_screen_background_effects(self, mgl_fbo, vp_mat):
        env_active = bool(getattr(self, '_env_model_visible', False) and getattr(self, '_env_model_prims', []))
        passthrough_active = getattr(self, '_bg_color_idx', 0) == 1
        dark_room_prims = getattr(self, '_dark_room_prims', []) or []
        if dark_room_prims and not env_active and not passthrough_active and getattr(self, '_bg_color_idx', 0) == 0:
            saved_prims = self._env_model_prims
            saved_visible = self._env_model_visible
            saved_active = getattr(self, '_active_environment', None)
            try:
                self._env_model_prims = dark_room_prims
                self._env_model_visible = True
                self._active_environment = 'Dark Room'
                view_mat = getattr(self, '_current_view_mat', None)
                if view_mat is not None:
                    self._render_env_model(mgl_fbo, vp_mat, view_mat)
                    mgl_fbo.use()
                    glClear(GL_DEPTH_BUFFER_BIT)
            finally:
                self._env_model_prims = saved_prims
                self._env_model_visible = saved_visible
                self._active_environment = saved_active
        elif not env_active and not passthrough_active:
            self._render_glow(mgl_fbo, vp_mat)

    def _render_screen_foreground_effects(self, mgl_fbo, vp_mat):
        return None

    def _apply_cinema_light_uniforms(self):
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
            self._cl_half = (float(self.screen_width) * 0.5, float(self.screen_height) * 0.5)
            self._cl_pose_key = pose_key
        state_key = (pose_key, getattr(self, '_active_environment', None), float(self._screen_light_intensity))
        last_state_key = getattr(self, '_cl_light_state_key', None)
        last_frame = getattr(self, '_cl_uniform_frame', -999)
        if state_key == last_state_key and (fc - last_frame) < 5:
            return
        self._cl_light_state_key = state_key
        self._cl_uniform_frame = fc
        self._advance_glow_color(lerp=0.14)
        sc = getattr(self, '_glow_color', (0.30, 0.55, 1.0))
        intensity = float(self._screen_light_intensity)
        if getattr(self, '_active_environment', None) == 'Dark Room':
            intensity *= 0.9
        self._env_prog['u_screen_light_enabled'].value = 1
        self._env_prog['u_screen_light_pos'].value = self._cl_pos
        self._env_prog['u_screen_light_normal'].value = self._cl_normal
        self._env_prog['u_screen_light_half_size'].value = self._cl_half
        self._env_prog['u_screen_light_color'].value = (float(sc[0]), float(sc[1]), float(sc[2]))
        self._env_prog['u_screen_light_intensity'].value = intensity


    def _render_env_model(self, mgl_fbo, vp_mat, view_mat):
        """Render the glTF environment model in world space."""
        if not self._env_model_visible or not self._env_model_prims:
            return
        perf_t0 = time.perf_counter() if self._env_perf_log else 0.0

        model_mat = self._env_model_mat4()
        view_inv = _view_mat_inv(view_mat)
        cam_pos = view_inv[:3, 3].astype('f4')

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

        self._apply_cinema_light_uniforms()

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


# Standalone smoke test helper shared by the viewer entry modules.
def _smoke_test(viewer_cls):
    if not OPENXR_AVAILABLE:
        print("[TEST] pyopenxr not available - cannot run standalone test")
        sys.exit(1)

    import queue as _q
    W, H = 1280, 720
    white_rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    zero_depth = np.zeros((H, W), dtype=np.float32)

    depth_q = _q.Queue(maxsize=2)
    depth_q.put((white_rgb, zero_depth, time.perf_counter()))

    viewer = viewer_cls(
        frame_size=(W, H),
        fps=60,
        depth_q=depth_q,
        show_fps=True,
    )

    try:
        viewer.run(first_rgb=white_rgb, first_depth=zero_depth)
    except KeyboardInterrupt:
        print("[TEST] Interrupted")
    finally:
        viewer.cleanup()


def _run_standalone_test():
    _smoke_test(OpenXRViewer)


if __name__ == "__main__":
    _run_standalone_test()
