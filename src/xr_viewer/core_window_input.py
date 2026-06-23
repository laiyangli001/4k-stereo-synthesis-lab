# Desktop2Stereo OpenXR viewer: GLFW setup and debug keyboard controls.

import time

import glfw

try:
    import ctypes
except Exception:
    ctypes = None


class CoreWindowInputMixin:
    """Hidden GLFW context setup plus desktop debug key handling."""

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("[OpenXRViewer] GLFW init failed")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)   # hidden -GL context only
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        self.window = glfw.create_window(1, 1, "D2S-XR", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("[OpenXRViewer] GLFW window creation failed")
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        # Keyboard controls -keep a reference so it isn't GC'd
        self._key_callback_ref = self._make_key_callback()
        glfw.set_key_callback(self.window, self._key_callback_ref)
        self._frosted_hotkey_prev = {}

    def _toggle_team_status_panel(self):
        self._team_status_visible = not self._team_status_visible
        if self._team_status_visible:
            self._team_fps_visible = self._team_status_visible

    def _cycle_a_panel(self):
        """Cycle A long-press: hidden ->screen FPS ->screen help ->hidden."""
        self._a_cycle_state = (self._a_cycle_state + 1) % 3
        if self._a_cycle_state == 0:
            self._team_fps_visible = False
            self._team_status_visible = False
            self._team_help_visible = False
        elif self._a_cycle_state == 1:
            self._team_fps_visible = True
            self._team_status_visible = True
            self._team_help_visible = False
        elif self._a_cycle_state == 2:
            self._team_fps_visible = True
            self._team_status_visible = True
            self._team_help_visible = True

    def _cycle_b_panel(self):
        """Cycle B long-press: hidden ->hand FPS ->hand help ->hidden."""
        self._b_cycle_state = (self._b_cycle_state + 1) % 3
        if self._b_cycle_state == 0:
            self._hand_fps_visible = False
            self._fps_overlay_visible = False
        elif self._b_cycle_state == 1:
            self._hand_fps_visible = True
            self._fps_overlay_visible = False
        elif self._b_cycle_state == 2:
            self._hand_fps_visible = True
            self._fps_overlay_visible = True

    def _adjust_frosted_glow_keyboard(self, key, mods=0):
        step = 0.05
        if mods & glfw.MOD_SHIFT:
            step = 0.15
        blend = float(getattr(self, '_frosted_glow_blend', 1.35))
        thickness = float(getattr(self, '_frosted_glow_thickness', 1.6))
        if key == glfw.KEY_LEFT:
            blend = max(0.0, blend - step)
        elif key == glfw.KEY_RIGHT:
            blend = min(2.5, blend + step)
        elif key == glfw.KEY_DOWN:
            thickness = max(0.5, thickness - step)
        elif key == glfw.KEY_UP:
            thickness = min(3.0, thickness + step)
        else:
            return False
        self._frosted_glow_blend = blend
        self._frosted_glow_thickness = thickness
        self._preset_name_overlay = f'Frosted blend {blend:.2f} / thickness {thickness:.2f}'
        self._preset_osd_show_t = time.perf_counter()
        print(
            f"[OpenXRViewer] Frosted glow: "
            f"blend={blend:.2f} thickness={thickness:.2f}"
        )
        return True

    def _adjust_frosted_glow_vk(self, vk):
        vk_to_key = {
            0x25: glfw.KEY_LEFT,
            0x26: glfw.KEY_UP,
            0x27: glfw.KEY_RIGHT,
            0x28: glfw.KEY_DOWN,
        }
        key = vk_to_key.get(int(vk))
        if key is None:
            return False
        return self._adjust_frosted_glow_keyboard(key, 0)

    def _poll_frosted_glow_hotkeys(self):
        if ctypes is None:
            return
        try:
            user32 = ctypes.windll.user32
        except Exception:
            return
        vk_to_key = {
            0x25: glfw.KEY_LEFT,
            0x26: glfw.KEY_UP,
            0x27: glfw.KEY_RIGHT,
            0x28: glfw.KEY_DOWN,
        }
        shift_down = bool(user32.GetAsyncKeyState(0x10) & 0x8000)
        mods = glfw.MOD_SHIFT if shift_down else 0
        prev = getattr(self, '_frosted_hotkey_prev', {})
        for vk, key in vk_to_key.items():
            down = bool(user32.GetAsyncKeyState(vk) & 0x8000)
            if down and not bool(prev.get(vk, False)):
                self._adjust_frosted_glow_keyboard(key, mods)
            prev[vk] = down
        self._frosted_hotkey_prev = prev

    def _make_key_callback(self):
        viewer = self
        def _cb(window, key, scancode, action, mods):
            if action not in (glfw.PRESS, glfw.REPEAT):
                return
            d = 0.1; s = 0.15; p = 0.1; r = 0.05
            screen_locked = viewer._environment_screen_locked()
            if key == glfw.KEY_F:
                viewer._toggle_team_status_panel()
            elif key == glfw.KEY_Z:
                viewer.depth_strength = max(0.0, viewer.depth_strength - 0.01)
            elif key == glfw.KEY_C:
                viewer.depth_strength = min(0.5, viewer.depth_strength + 0.01)
            elif key == glfw.KEY_X:
                viewer.depth_strength = 0.0   # flat mode -no parallax distortion
            elif key == glfw.KEY_V:
                viewer._toggle_quad_layer_compare()
            elif key == glfw.KEY_R:
                viewer._reset_screen_to_default(show_border=True)
            elif key in (glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT):
                viewer._adjust_frosted_glow_keyboard(key, mods)
            elif screen_locked:
                return
            elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
                viewer._screen_ref_size += s; viewer.screen_height = None
            elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
                viewer._screen_ref_size = max(0.8, viewer._screen_ref_size - s)
                viewer.screen_height = None
            elif key == glfw.KEY_Q: viewer.screen_yaw += r
            elif key == glfw.KEY_E: viewer.screen_yaw -= r
            elif key == glfw.KEY_T: viewer.screen_pitch += r
            elif key == glfw.KEY_G: viewer.screen_pitch -= r
        return _cb

    def _toggle_quad_layer_compare(self):
        if not self._xr_quad_layer_enabled:
            self._preset_name_overlay = 'Projection Screen (Quad disabled)'
        elif self._xr_quad_layer_failed or not self._quad_swapchains:
            self._xr_quad_layer_active = False
            self._preset_name_overlay = 'Projection Screen (Quad unavailable)'
        else:
            self._xr_quad_layer_active = not self._xr_quad_layer_active
            self._preset_name_overlay = 'Quad Layer Screen' if self._xr_quad_layer_active else 'Projection Screen'
        self._preset_osd_show_t = time.perf_counter()
        self._publish_runtime_config()
        print(
            "[OpenXRViewer] Screen compare mode: "
            f"{self._preset_name_overlay} "
            f"enabled={self._xr_quad_layer_enabled} "
            f"active={self._xr_quad_layer_active} "
            f"swapchains={len(self._quad_swapchains)} "
            f"array_size={int(self._quad_swapchain_array_size.get(0, 0) or 0)} "
            f"per_eye_layers=True "
            f"stereo_boost={float(getattr(self, '_xr_quad_layer_stereo_boost', 1.0)):.2f} "
            f"failed={self._xr_quad_layer_failed}"
        )
