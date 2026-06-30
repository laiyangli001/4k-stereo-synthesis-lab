from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_keyboard_scales_with_screen_preset(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.implementation import OpenXRViewerCore

    class _Viewer(OpenXRViewerCore):
        def __init__(self):
            pass

    viewer = _Viewer()
    viewer._screen_presets = [("Small", 2.0, 2.0), ("Large", 4.0, 3.0)]
    viewer._preset_index = 0
    viewer.screen_width = 2.0
    viewer.screen_height = 1.125
    viewer.screen_pan_x = 0.0
    viewer.screen_pan_y = 1.0
    viewer.screen_distance = 2.0
    viewer.screen_yaw = 0.0
    viewer.screen_pitch = 0.4
    viewer.screen_roll = 0.7
    viewer._anim_target_pan_x = 1.0
    viewer._anim_target_pan_y = 1.0
    viewer._anim_target_distance = 2.0
    viewer._anim_target_yaw = 0.2
    viewer._anim_target_pitch = 0.4
    viewer._anim_target_roll = 0.7
    viewer._screen_ref_size = 2.0
    viewer._screen_curved = False
    viewer._initial_head_y = 1.0
    viewer._head_pos_w = None
    viewer._head_fwd_w = None
    viewer._keyboard_visible = True
    viewer._keyboard_width = 1.0
    viewer._keyboard_height = 0.2
    viewer._keyboard_tex = object()
    viewer._kb_last_build_width = 1.0
    viewer._screen_footprint_logged = set()
    viewer._border_alpha = 0.0
    viewer._border_idle_t = 0.0
    viewer._last_overlay_update = 0.0
    viewer._preset_name_overlay = None
    viewer._preset_osd_last_key = (0, "Small  2.00 x 1.12 m")
    viewer._reset_orientation_offsets = lambda: None
    viewer._clear_screen_grab_anchors = lambda: None
    viewer._build_keyboard_texture_called = False
    viewer._anchor_keyboard_below_screen_called = False

    def _build_keyboard_texture():
        viewer._build_keyboard_texture_called = True

    def _anchor_keyboard_below_screen():
        viewer._anchor_keyboard_below_screen_called = True

    monkeypatch.setattr(viewer, "_build_keyboard_texture", _build_keyboard_texture)
    monkeypatch.setattr(viewer, "_anchor_keyboard_below_screen", _anchor_keyboard_below_screen)

    assert viewer._apply_preset(1)

    assert viewer.screen_width == 4.0
    assert viewer.screen_pitch == 0.0
    assert viewer.screen_roll == 0.0
    assert viewer._anim_target_pan_x is None
    assert viewer._anim_target_pitch is None
    assert viewer._anim_target_roll is None
    assert "4.00 x 2.25 m" in viewer._preset_name_overlay
    assert "@ 3.00 m" in viewer._preset_name_overlay
    assert viewer._preset_osd_last_key is None
    assert viewer._keyboard_width == 2.0
    assert viewer._keyboard_height > 0.0
    assert viewer._build_keyboard_texture_called
    assert viewer._anchor_keyboard_below_screen_called

    viewer._head_pos_w = np.array([0.0, 1.0, -2.0], dtype=np.float32)
    viewer._head_fwd_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    viewer._initial_head_y = 1.8
    assert viewer._apply_preset(0)
    assert "@ 2.15 m" in viewer._preset_name_overlay


def test_keyboard_gap_is_15_percent_of_screen_height(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.implementation import OpenXRViewerCore

    class _Viewer(OpenXRViewerCore):
        def __init__(self):
            pass

    viewer = _Viewer()
    viewer.screen_width = 3.0
    viewer.screen_height = 2.0
    viewer.screen_pan_x = 0.25
    viewer.screen_pan_y = 1.8
    viewer.screen_distance = 3.0
    viewer.screen_yaw = 0.1
    viewer.frame_size = (3840, 2160)
    viewer._keyboard_height = 0.4
    viewer._kb_yaw_offset = 0.0
    viewer._kb_pitch_offset = 0.0
    viewer._head_pos_w = None

    viewer._anchor_keyboard_below_screen()

    screen_bottom_y = viewer.screen_pan_y - viewer.screen_height / 2.0
    keyboard_top_y = viewer._keyboard_pan_y + viewer._keyboard_height / 2.0
    assert abs(screen_bottom_y - keyboard_top_y - viewer.screen_height * 0.15) < 1e-6
    assert viewer._keyboard_pan_x == viewer.screen_pan_x
    assert viewer._keyboard_distance == viewer.screen_distance - 0.001


def test_cinema_giant_is_default_y_screen_preset():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "self.screen_distance = _float_option(kwargs, 'openxr_screen_distance', 'D2S_OPENXR_SCREEN_DISTANCE', 16.0, 0.25, 40.0)" in impl_text
    assert "self.screen_width    = _float_option(kwargs, 'openxr_screen_width', 'D2S_OPENXR_SCREEN_WIDTH', 16.0, 0.25, 30.0)" in impl_text
    assert "('Cinema Giant', 16.0, 16.0)" in impl_text
    assert "self._default_screen_preset_index = 5" in impl_text
    assert "self._preset_index = self._default_screen_preset_index" in impl_text
    assert "self.screen_width = float(_default_preset[1])" in impl_text
    assert "self.screen_distance = float(_default_preset[2])" in impl_text
    assert "self._initial_screen_preset = ('Headset Recommended', self.screen_width, self.screen_distance)" in impl_text
    assert "self._screen_presets[self._default_screen_preset_index] = self._initial_screen_preset" in impl_text
    assert "self._screen_ref_size = self.screen_width" in impl_text
    assert "self._apply_preset(getattr(self, '_default_screen_preset_index', 5))" in impl_text
    assert "self._apply_preset(3)" not in impl_text


def test_right_grip_orbit_faces_head_while_wrist_roll_is_disabled_by_default():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "os.environ.get('D2S_OPENXR_RIGHT_GRIP_SCREEN_ROTATION', '0')" in impl_text
    assert "self._right_grip_screen_rotation_enabled" in impl_text
    orbit_block = impl_text.split("# -- Right grip: sphere-orbit drag --", 1)[1]
    orbit_block = orbit_block.split("elif both_grips and not grip_now:", 1)[0]
    roll_guard = orbit_block.split("if self._right_grip_screen_rotation_enabled", 1)[1]
    assert "self.screen_yaw" in orbit_block.split("if self._right_grip_screen_rotation_enabled", 1)[0]
    assert "self.screen_pitch" in orbit_block.split("if self._right_grip_screen_rotation_enabled", 1)[0]
    assert "self.screen_roll" in roll_guard


def test_laser_hit_circle_radius_uses_eye_to_hit_distance(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_laser_render import CoreLaserRenderMixin

    class _Viewer(CoreLaserRenderMixin):
        pass

    viewer = _Viewer()
    viewer._current_view_mat = np.eye(4, dtype=np.float32)

    assert viewer._cursor_ring_distance_from_eye(np.array([0.0, 0.0, -4.0]), 1.0) == 4.0
    assert viewer._cursor_ring_distance_from_eye(np.array([0.0, 0.0, -0.5]), 4.0) == 0.5
    assert viewer._cursor_ring_specs(0.5)[0][0] < viewer._cursor_ring_specs(4.0)[0][0]


def test_laser_hit_circle_model_is_shared_by_screen_and_keyboard(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_laser_render import CoreLaserRenderMixin

    class _Viewer(CoreLaserRenderMixin):
        def _screen_basis(self):
            return (
                1.0,
                np.zeros(3, dtype=np.float32),
                np.array([1.0, 0.0, 0.0], dtype=np.float32),
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
                np.array([0.0, 0.0, 1.0], dtype=np.float32),
            )

    viewer = _Viewer()
    viewer._keyboard_pitch = 0.0
    viewer._keyboard_yaw = 0.0
    hit_pos = np.array([0.25, 0.5, -2.0], dtype=np.float32)

    screen_model = viewer._cursor_ring_model('screen', hit_pos, 0.02)
    keyboard_model = viewer._cursor_ring_model('keyboard', hit_pos, 0.02)

    assert np.allclose(screen_model[:3, 3], hit_pos)
    assert np.allclose(keyboard_model[:3, 3], hit_pos)
    assert np.isclose(np.linalg.norm(screen_model[:3, 0]), 0.02)
    assert np.isclose(np.linalg.norm(keyboard_model[:3, 0]), 0.02)

    laser_text = (SRC / "xr_viewer" / "core_laser_render.py").read_text(encoding="utf-8")
    render_block = laser_text.split("def _render_laser_hit_circles", 1)[1]
    render_block = render_block.split("def _controller_anim_delta", 1)[0]
    assert "model = self._cursor_ring_model(hit_target, hit_pos, radius)" in render_block
    assert "if hit_target == 'screen':" not in render_block
    assert "elif hit_target == 'keyboard':" not in render_block


def test_laser_hit_circles_render_without_depth_test_like_keyboard_cursor():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_eye = impl_text.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    hit_circle_block = render_eye.split("# 9. Laser hit circles", 1)[1]
    hit_circle_block = hit_circle_block.split("# 10. FPS/status overlays", 1)[0]

    disable_depth = hit_circle_block.index("self.ctx.disable(moderngl.DEPTH_TEST)")
    draw_hit_circles = hit_circle_block.index("self._render_lasers(mgl_fbo, vp_mat, blend=True)")
    enable_depth = hit_circle_block.index("self.ctx.enable(moderngl.DEPTH_TEST)")
    assert disable_depth < draw_hit_circles < enable_depth
    assert "self.ctx.depth_mask = False" in hit_circle_block
    assert "self.ctx.depth_mask = True" in hit_circle_block


def test_openxr_keyboard_hover_pulses_controller_haptics_on_key_changes():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    actions_text = (SRC / "xr_viewer" / "core_controller_actions.py").read_text(encoding="utf-8")
    input_text = (SRC / "xr_viewer" / "core_openxr_input.py").read_text(encoding="utf-8")
    helpers_text = (SRC / "xr_viewer" / "core_input_helpers.py").read_text(encoding="utf-8")
    cleanup_text = (SRC / "xr_viewer" / "core_cleanup.py").read_text(encoding="utf-8")

    assert "self._haptic_last_l       = 0.0" in impl_text
    assert "self._haptic_last_r       = 0.0" in impl_text
    assert "action_type=xr.ActionType.VIBRATION_OUTPUT" in actions_text
    assert 'action_name="haptic"' in actions_text
    assert actions_text.count('/user/hand/left/output/haptic') >= 6
    assert actions_text.count('/user/hand/right/output/haptic') >= 6
    assert "self._act_haptic = None" in cleanup_text
    assert "def _pulse_haptic(" in input_text
    assert "xr.HapticVibration(" in input_text
    assert "xr.HapticActionInfo(action=action, subaction_path=path)" in input_text
    assert "xr.apply_haptic_feedback(" in input_text
    assert "min_interval_s=0.045" in helpers_text
    assert "idx is not None and idx != prev_hover and not gripping" in helpers_text
    assert "self._pulse_haptic(hand_path, amplitude=0.18, duration_s=0.018, min_interval_s=0.045)" in helpers_text


def test_openxr_controller_button_press_animates_individual_button_nodes(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.gltf_loader import load_glb_model

    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    models_text = (SRC / "xr_viewer" / "controller_models.py").read_text(encoding="utf-8")
    actions_text = (SRC / "xr_viewer" / "core_controller_actions.py").read_text(encoding="utf-8")
    input_text = (SRC / "xr_viewer" / "core_openxr_input.py").read_text(encoding="utf-8")
    laser_text = (SRC / "xr_viewer" / "core_laser_render.py").read_text(encoding="utf-8")
    cleanup_text = (SRC / "xr_viewer" / "core_cleanup.py").read_text(encoding="utf-8")

    assert "self._ctrl_press_l        = {}" in impl_text
    assert "self._ctrl_press_r        = {}" in impl_text
    assert "self._update_controller_press_animation_state(dt, lx, ly, rx, ry)" in impl_text
    assert "'node_name': pd.get('node_name', '')" in models_text
    assert "'press_anim': pd.get('press_anim')" in models_text
    assert "'axis_anim': pd.get('axis_anim')" in models_text
    assert "'anim_key': pd.get('anim_key', '')" in models_text
    assert "'visible_key': pd.get('visible_key', '')" in models_text
    assert "left_stick_touch" in actions_text
    assert "right_stick_touch" in actions_text
    assert "/input/thumbstick/touch" in actions_text
    assert "/input/trackpad/touch" in actions_text
    assert "self._act_left_stick_touch = None" in cleanup_text
    assert "self._act_right_stick_touch = None" in cleanup_text
    assert "def _update_controller_press_animation_state" in input_text
    for anim_key in (
        "trigger",
        "grip",
        "a_button",
        "b_button",
        "x_button",
        "y_button",
        "joystick",
        "joystick_x",
        "joystick_y",
        "joystick_touched",
        "touchpad",
        "touchpad_x",
        "touchpad_y",
        "touchpad_touched",
        "menu_button",
        "left_trigger",
        "right_trigger",
        "left_joystick",
        "right_joystick",
    ):
        assert anim_key in input_text
    assert "left_stick_touched = (" in input_text
    assert '"left_joystick_y": -ly' in input_text
    assert '"right_joystick_y": -ry' in input_text
    assert "visible_key = prim.get('visible_key', '')" in laser_text
    assert "press_map.get(visible_key, 0.0)" in laser_text
    assert "anim_key = prim.get('anim_key', '') or prim.get('node_name', '')" in laser_text
    assert "press_map.get(anim_key, press_map.get(prim.get('node_name', ''), 0.0))" in laser_text
    assert 'press_map.get(f"{anim_key}_x"' in laser_text
    assert 'press_map.get(f"{anim_key}_y"' in laser_text
    assert "_controller_anim_delta" in laser_text
    assert "axis_x_delta" in laser_text
    assert "axis_y_delta" in laser_text
    assert "def _quat_to_mat3" in laser_text
    assert "value_local[:3, :3] = self._quat_to_mat3(self._slerp_quat(q0, q1, t))" in laser_text
    assert "value_world = (anim['value_parent_world'] @ value_local).astype(np.float32)" in laser_text
    assert "r_mat @ t_mat @ press_mat" not in laser_text

    for glb_path in (SRC / "xr_viewer" / "controllers").glob("*/*.glb"):
        prims, _, _ = load_glb_model(glb_path)
        by_key = {prim.get("anim_key"): prim for prim in prims if prim.get("anim_key")}
        assert "trigger" in by_key, glb_path
        assert "joystick" in by_key, glb_path
        assert by_key["trigger"].get("press_anim"), glb_path
        assert by_key["joystick"].get("press_anim"), glb_path
        for anim_key in ("trigger", "joystick"):
            anim = by_key[anim_key]["press_anim"]
            for matrix_key in ("value_parent_world", "value_local", "min_local", "max_local"):
                assert matrix_key in anim, glb_path
        assert by_key["joystick"].get("axis_anim", {}).get("x"), glb_path
        assert by_key["joystick"].get("axis_anim", {}).get("y"), glb_path
        if glb_path.parent.name == "INDEX":
            assert "touchpad" in by_key, glb_path
            assert by_key["touchpad"].get("press_anim"), glb_path


def test_reset_screen_to_default_uses_cinema_giant(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.implementation import OpenXRViewerCore

    class _Viewer(OpenXRViewerCore):
        def __init__(self):
            pass

    viewer = _Viewer()
    viewer._screen_presets = [('100" Projector 1', 2.4, 2.0), ('Cinema Giant', 16.0, 16.0)]
    viewer._default_screen_preset_index = 1
    viewer._preset_index = 0
    viewer.screen_width = 2.4
    viewer.screen_distance = 2.0
    viewer.screen_height = 1.35
    viewer.screen_pan_x = 1.0
    viewer.screen_pan_y = 1.0
    viewer.screen_yaw = 0.5
    viewer.screen_pitch = 0.25
    viewer.screen_roll = 0.0
    viewer._initial_head_y = 1.25
    viewer._head_pos_w = None
    viewer._head_fwd_w = None
    viewer._keyboard_visible = False
    viewer._screen_curved = False
    viewer._border_alpha = 0.0
    viewer._border_idle_t = 0.0
    viewer._apply_profile_screen_layout = lambda show_border=False: False
    viewer._environment_screen_locked = lambda: False
    viewer._reset_orientation_offsets = lambda: None
    viewer._clear_screen_grab_anchors = lambda: None
    viewer._move_env_with_screen_delta = lambda _old: None

    viewer._reset_screen_to_default(show_border=False)

    assert viewer.screen_width == 16.0
    assert viewer._screen_ref_size == 16.0
    assert viewer.screen_distance == 16.0
    assert viewer._preset_index == 1

    viewer._head_pos_w = np.array([0.0, 1.25, -2.0], dtype=np.float32)
    viewer._head_fwd_w = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    viewer._screen_presets[1] = ('Headset Recommended', 23.09, 20.0)
    viewer._reset_screen_to_default(show_border=False)

    assert viewer.screen_width == 23.09
    assert viewer.screen_distance == 22.0
    assert viewer._screen_view_distance() == 20.0


def test_overlay_distance_labels_use_head_to_screen_distance():
    overlay_text = (SRC / "xr_viewer" / "overlay.py").read_text(encoding="utf-8")

    assert "self._cached_screen_dist   = self._screen_view_distance()" in overlay_text
    assert "base_label = preset_label.rsplit(\"  @ \", 1)[0]" in overlay_text
    assert "preset_label = base_label + f\"  @ {self._screen_view_distance():.2f} m\"" in overlay_text
    assert "cur_key = (self._preset_index, base_label)" in overlay_text
    assert "cur_key = (round(self.screen_width, 2), round(self._screen_view_distance(), 2))" in overlay_text
    assert "dist_val = f\"{self._screen_view_distance():.2f} m\"" in overlay_text
    preset_osd_block = overlay_text.split("def _render_preset_osd", 1)[1].split("def _render_screen_osd", 1)[0]
    assert "HOLD  = 5.0" in preset_osd_block
    assert "dist_val = f\"{self.screen_distance:.2f} m\"" not in overlay_text


def test_screen_view_distance_uses_head_to_screen_distance(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.core_screen_state import CoreScreenStateMixin

    class _Viewer(CoreScreenStateMixin):
        pass

    viewer = _Viewer()
    viewer.screen_pan_x = 3.0
    viewer.screen_pan_y = 4.0
    viewer.screen_distance = 12.0
    viewer._head_pos_w = None

    assert viewer._screen_view_distance() == 12.0

    viewer._head_pos_w = np.array([3.0, 4.0, -2.0], dtype=np.float32)
    assert viewer._screen_view_distance() == 10.0


def test_screen_back_offset_scales_for_cinema_giant(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.implementation import OpenXRViewerCore

    class _Viewer(OpenXRViewerCore):
        def __init__(self):
            pass

    viewer = _Viewer()
    viewer.screen_width = 16.0

    assert viewer._screen_back_offset() == 0.2
    assert viewer._screen_back_offset(0.5) == 0.1


def test_screen_border_hides_when_idle_alpha_reaches_zero():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    source_state_text = (SRC / "xr_viewer" / "core_source_state.py").read_text(encoding="utf-8")

    assert "def _should_show_source_border(self, now=None):" in source_state_text
    assert 'if getattr(self, "_hard_idle_active", False):' in source_state_text
    assert 'if not source_active_event.is_set():' in source_state_text
    assert "if not self._has_renderable_source_frame():" in source_state_text
    assert "return self._has_fresh_source_frame(now)" in source_state_text
    assert "should_show_source_border = getattr(self, '_should_show_source_border', None)" in impl_text
    assert "should_show_source_border = getattr(self, '_should_show_source_border', None)" in base_text
    assert "alpha = max(float(getattr(self, '_border_alpha', 0.0)), 0.0)" in impl_text
    assert "if alpha <= 0.0 or self._border_prog is None" in impl_text
    assert "alpha = max(float(getattr(self, '_border_alpha', 0.0)), 0.0)" in base_text
    assert "if alpha <= 0.0:" in base_text
    assert "border_prog = getattr(self, '_metallic_border_prog', None)" in impl_text
    assert "border_prog['u_border_uv'].value" in impl_text


def test_openxr_startup_seed_frame_does_not_mark_fresh_source():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    startup_block = impl_text.split("# Upload the first frame supplied by main.py", 1)[1].split("# Default fallback projection", 1)[0]

    assert "self._update_runtime_frame(first_runtime_result)" in startup_block
    assert "self._update_frame(first_rgb, first_depth)" in startup_block
    assert "self._mark_source_frame_received()" not in startup_block


def test_shader_sources_live_in_glsl_module():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")

    assert "from .glsl import (" in impl_text
    for shader_name in (
        "_WORLD_VERT",
        "_SCREEN_RCAS_FRAG",
        "_ENV_FRAG",
        "_GLOW_FRAG",
    ):
        assert f"{shader_name} =" not in impl_text
        assert f"{shader_name} =" in glsl_text
    assert "uniform float u_glow_width" in glsl_text
    assert "uniform float u_glow_extent" in glsl_text
    assert "u_glow_inv_range" not in glsl_text


def test_screen_size_overlays_display_fixed_16_9_height():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    overlay_text = (SRC / "xr_viewer" / "overlay.py").read_text(encoding="utf-8")

    assert "display_height_m = float(width_m) * 9.0 / 16.0" in impl_text
    assert "self._cached_screen_height = self.screen_width * 9.0 / 16.0" in impl_text
    assert "self._cached_screen_height = self.screen_width * 9.0 / 16.0" in overlay_text
    assert "h = self.screen_width * 9.0 / 16.0" in overlay_text
    assert "h = self.screen_height if self.screen_height is not None else 0.0" not in overlay_text


def test_a_short_press_toggles_curved_screen_only_when_unlocked():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "if not a_now and self._a_last and not self._a_long_fired:" in impl_text
    assert "if not screen_locked:" in impl_text
    assert "self._screen_curved = not self._screen_curved" in impl_text
    assert "if self._screen_curved and getattr(self, '_xr_quad_layer_active', False):" in impl_text
    assert "self._xr_quad_layer_active = False" in impl_text
    assert "self._preset_name_overlay = 'Curved Screen' if self._screen_curved else 'Flat Screen'" in impl_text


def test_curved_screen_uses_same_fragment_path_as_flat_screen():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "elif self._screen_curved and self._curved_prog is not None and not self._runtime_direct_source" not in impl_text
    assert "if self._screen_curved and self._curved_prog is not None:" in impl_text
    assert "screen_tex = self._prepare_screen_quality_texture(" in impl_text
    curved_block = impl_text.split("if self._screen_curved and self._curved_prog is not None:", 1)[1]
    curved_block = curved_block.split("else:", 1)[0]
    assert "self._curved_copy_prog" not in curved_block
    assert "self._curved_copy_vao" not in curved_block
    assert "screen_tex.use(location=0)" in curved_block
    assert "screen_depth_tex.use(location=1)" in curved_block
    assert "self._curved_prog['u_roll'].value = 0.0 if self._runtime_direct_source else self.screen_roll" in curved_block
    assert "self._curved_prog['u_eye_offset'].value = screen_eye_offset" in curved_block
    assert "self._curved_prog['u_depth_strength'].value = screen_depth_strength" in curved_block
    for uniform in ("u_resolution", "u_feather_enabled", "u_feather_width", "u_viewport"):
        assert f"self._curved_prog['{uniform}']" in curved_block


def test_openxr_screen_shader_uniforms_are_initialized_for_flat_and_curved_paths():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_eye = impl_text.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    render_eye = render_eye.split("# Flat border is a foreground guide", 1)[0]

    assert "screen_source_size = (" in render_eye
    assert "runtime_rgb_depth_max_disparity_px = (" in render_eye
    assert "runtime_rgb_depth_render_width = (" in render_eye
    assert "screen_disparity_uv = max(0.0, runtime_rgb_depth_max_disparity_px) / float(runtime_rgb_depth_render_width)" in render_eye
    assert "screen_depth_strength = (" in render_eye
    assert "if self._runtime_direct_source" in render_eye
    assert "_runtime_rgb_depth_depth_strength" in render_eye
    assert "screen_eye_offset = 0.0 if self._runtime_direct_source else eye_sign * screen_disparity_uv / 2.0" in render_eye
    assert "runtime_rgb_depth_stereo_scale" not in render_eye
    assert "runtime_rgb_depth_max_shift_ratio" not in render_eye
    assert "screen_ipd_uv" not in render_eye
    assert "shader_resolution_mode = str(getattr(self, '_openxr_rgb_depth_shader_resolution', 'source') or 'source')" in render_eye
    assert "elif shader_resolution_mode == 'swapchain':" in render_eye
    assert "shader_resolution = None" in render_eye
    assert "f\" max_disparity_px={runtime_rgb_depth_max_disparity_px:.3f}\"" in render_eye
    assert "f\" render_width={runtime_rgb_depth_render_width}\"" in render_eye
    assert "f\" disparity_uv={screen_disparity_uv:.6f}\"" in render_eye
    assert "feather_enabled = bool(runtime_rgb_depth and self._openxr_rgb_depth_feather)" in render_eye
    for program_name in ("self.prog", "self._curved_prog"):
        assert f"{program_name}['u_eye_offset'].value = screen_eye_offset" in render_eye
        assert f"{program_name}['u_depth_strength'].value = screen_depth_strength" in render_eye
        for uniform in ("u_resolution", "u_feather_enabled", "u_feather_width", "u_viewport"):
            assert f"{program_name}['{uniform}']" in render_eye


def test_viewer_shader_uses_beta_subpixel_screen_edge_clip():
    shader_text = (SRC / "viewer" / "viewer.py").read_text(encoding="utf-8")

    assert "smoothstep(-0.001, 0.001, shifted_uv)" in shader_text
    assert "smoothstep(1.001, 0.999, shifted_uv)" in shader_text
    assert "smoothstep(0.0, 0.015, shifted_uv)" not in shader_text


def test_screen_edge_snap_angle_is_6_degrees():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "self._ray_edge_deadzone_rad = math.radians(6.0)" in impl_text


def test_curved_screen_grip_drag_uses_curved_uv_hit_point_not_flat_plane():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    drag_block = impl_text.split("# Per-controller laser requirement", 1)[1]
    drag_block = drag_block.split("# Keyboard grip-to-move", 1)[0]

    assert "if self._screen_curved:" in drag_block
    assert "cursor_uv = self._cursor_uv_l if is_left else self._cursor_uv_r" in drag_block
    assert "hit_world = self._screen_uv_to_world(float(cursor_uv[0]), float(cursor_uv[1]))" in drag_block
    curved_branch = drag_block.split("if self._screen_curved:", 1)[1].split("else:", 1)[0]
    assert "np.dot(screen_normal, ray_dir)" not in curved_branch


def test_curved_border_is_rendered_behind_screen_not_over_image():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_eye = impl_text.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    before_main, after_main = render_eye.split("# Main screen", 1)
    after_main = after_main.split("# Optional screen effects on/around", 1)[0]

    assert "if self._screen_curved:" in before_main
    assert "self._render_border(mgl_fbo, vp_mat)" in before_main
    assert "if not self._screen_curved:" in after_main
    assert "self._render_border(mgl_fbo, vp_mat)" in after_main


def test_quad_layer_is_disabled_and_projection_screen_is_always_drawn():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    quad_gate = impl_text.split("def _quad_layer_can_replace_projection_screen(self):", 1)[1]
    quad_gate = quad_gate.split("def _update_quad_layer_swapchain", 1)[0]
    assert "return False" in quad_gate
    assert "self._xr_quad_layer_active" not in quad_gate

    render_eye = impl_text.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    render_eye = render_eye.split("# Flat border is a foreground guide", 1)[0]
    assert "draw_projection_screen" not in render_eye
    assert "_quad_layer_can_replace_projection_screen" not in render_eye
    assert "screen_depth_tex = self._runtime_depth_texture" in render_eye


def test_curved_screen_geometry_uses_beta_fixed_angle_arc_and_gl_state_reset():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    curved_verts = impl_text.split("def _build_curved_screen_verts", 1)[1]
    curved_verts = curved_verts.split("def _get_or_create_fbo", 1)[0]
    assert "Mirrors the Beta viewer geometry" in curved_verts
    assert "half_ang = min(_CURVED_HALF_ANGLE_RAD, math.pi / 2)" in curved_verts
    assert "radius = half_w / max(half_ang, 1e-6)" in curved_verts
    assert "wz += -(self.screen_distance + dist_offset)" in curved_verts
    assert "radius = self.screen_distance + dist_offset" not in curved_verts

    screen_uv_to_world = impl_text.split("def _screen_uv_to_world", 1)[1]
    screen_uv_to_world = screen_uv_to_world.split("def _screen_basis", 1)[0]
    assert "half_ang = min(_CURVED_HALF_ANGLE_RAD, math.pi / 2)" in screen_uv_to_world
    assert "radius = half_w / max(half_ang, 1e-6)" in screen_uv_to_world
    assert "wz - self.screen_distance" in screen_uv_to_world
    assert "radius = max(self.screen_distance" not in screen_uv_to_world

    laser_hit = impl_text.split("def _laser_screen_hit_uv", 1)[1]
    laser_hit = laser_hit.split("denom = float(np.dot(screen_n, fwd_w))", 1)[0]
    assert "half_ang = min(_CURVED_HALF_ANGLE_RAD, math.pi / 2)" in laser_hit
    assert "radius = half_w / max(half_ang, 1e-6)" in laser_hit
    assert "origin = np.array([self.screen_pan_x, self.screen_pan_y, -self.screen_distance]" in laser_hit
    assert "radius = max(self.screen_distance" not in laser_hit

    render_eye = impl_text.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    render_eye = render_eye.split("if self._screen_curved and self._curved_prog is not None:", 1)[0]
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_eye
    assert "self.ctx.depth_mask = True" in render_eye
    assert "self.ctx.disable(moderngl.BLEND)" in render_eye
    assert "self.ctx.disable(moderngl.CULL_FACE)" in render_eye
    assert "glFrontFace(GL_CCW)" in render_eye


def test_x_long_press_cycles_default_glow_and_y_no_longer_uses_grip_glow():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    y_block = impl_text.split("# Y (left): short press applies default screen preset; hold 1s cycles presets.", 1)[1]
    y_block = y_block.split("# X (left):", 1)[0]
    x_block = impl_text.split("# X (left): short press -> toggle virtual keyboard.", 1)[1]
    x_block = x_block.split("# Thumbstick clicks:", 1)[0]

    assert "_cycle_glow_mode_from_y" not in y_block
    assert "X_GLOW_HOLD = 1.0" in x_block
    assert "blank_default_room = (" in x_block
    assert "env_name in ('default', 'none')" in x_block
    assert "and getattr(self, '_active_environment', None) is None" in x_block
    assert "if blank_default_room:" in x_block
    assert "cycle_glow = getattr(self, '_cycle_glow_mode_from_y', None)" in x_block
    assert "else:\n                        cycle_light = getattr(self, '_cycle_light_from_x', None)" in x_block
    assert "self._keyboard_visible = not self._keyboard_visible" in x_block
