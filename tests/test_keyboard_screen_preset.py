from pathlib import Path


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
    assert viewer._keyboard_width == 2.0
    assert viewer._keyboard_height > 0.0
    assert viewer._build_keyboard_texture_called
    assert viewer._anchor_keyboard_below_screen_called


def test_cinema_giant_is_default_y_screen_preset():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "self.screen_distance = _float_option(kwargs, 'openxr_screen_distance', 'D2S_OPENXR_SCREEN_DISTANCE', 16.0, 0.25, 40.0)" in impl_text
    assert "self.screen_width    = _float_option(kwargs, 'openxr_screen_width', 'D2S_OPENXR_SCREEN_WIDTH', 16.0, 0.25, 20.0)" in impl_text
    assert "('Cinema Giant', 16.0, 16.0)" in impl_text
    assert "self._default_screen_preset_index = 5" in impl_text
    assert "self._preset_index = self._default_screen_preset_index" in impl_text
    assert "self.screen_width = float(_default_preset[1])" in impl_text
    assert "self.screen_distance = float(_default_preset[2])" in impl_text
    assert "self._screen_ref_size = self.screen_width" in impl_text
    assert "self._apply_preset(getattr(self, '_default_screen_preset_index', 5))" in impl_text
    assert "self._apply_preset(3)" not in impl_text


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


def test_screen_border_remains_visible_when_idle():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    assert "alpha = max(float(getattr(self, '_border_alpha', 0.0)), 0.55)" in impl_text
    assert "if alpha <= 0.0 or self._border_prog is None" not in impl_text
    assert "border_prog = getattr(self, '_metallic_border_prog', None)" in impl_text
    assert "border_prog['u_border_uv'].value" in impl_text


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
    assert "self._curved_prog['u_eye_offset'].value = 0.0 if self._runtime_direct_source else eye_sign * screen_ipd_uv / 2.0" in curved_block
    assert "self._curved_prog['u_depth_strength'].value = 0.0 if self._runtime_direct_source else self.depth_strength * self.depth_ratio" in curved_block
    for uniform in ("u_resolution", "u_feather_enabled", "u_feather_width", "u_viewport"):
        assert f"self._curved_prog['{uniform}']" in curved_block


def test_openxr_screen_shader_uniforms_are_initialized_for_flat_and_curved_paths():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_eye = impl_text.split("def _render_eye(self, eye_index, mgl_fbo, view_mat, proj_mat, flip_y=False):", 1)[1]
    render_eye = render_eye.split("# Flat border is a foreground guide", 1)[0]

    assert "screen_source_size = (" in render_eye
    assert "screen_ipd_uv = self.ipd_uv * runtime_rgb_depth_stereo_scale" in render_eye
    for program_name in ("self.prog", "self._curved_prog"):
        for uniform in ("u_resolution", "u_feather_enabled", "u_feather_width", "u_viewport"):
            assert f"{program_name}['{uniform}']" in render_eye


def test_viewer_shader_uses_beta_subpixel_screen_edge_clip():
    shader_text = (SRC / "viewer" / "viewer.py").read_text(encoding="utf-8")

    assert "smoothstep(-0.001, 0.001, shifted_uv)" in shader_text
    assert "smoothstep(1.001, 0.999, shifted_uv)" in shader_text
    assert "smoothstep(0.0, 0.015, shifted_uv)" not in shader_text


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
    render_eye = render_eye.split("def ", 1)[0]
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
