from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _make_default_viewer(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.environment import OpenXRViewer

    class _DefaultViewer(OpenXRViewer):
        def __init__(self):
            pass

    return _DefaultViewer()


def _make_no_room_viewer(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.base import ScreenEffectsMixin

    class _NoRoomViewer(ScreenEffectsMixin):
        def __init__(self):
            pass

    return _NoRoomViewer()


def test_no_room_background_effects_skip_shadow_and_ground(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    viewer._screen_effects_enabled = True
    viewer.screen_height = 9.0
    viewer._render_glow_called = False
    viewer._render_shadow_called = False
    viewer._render_ground_light_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_shadow(*_args):
        viewer._render_shadow_called = True

    def _render_ground_light(*_args):
        viewer._render_ground_light_called = True

    viewer._render_glow = _render_glow
    viewer._render_shadow = _render_shadow
    viewer._render_ground_light = _render_ground_light
    viewer._render_screen_background_effects(None, None)

    assert viewer._render_glow_called
    assert not viewer._render_shadow_called
    assert not viewer._render_ground_light_called


def test_default_screen_state_persistence_is_disabled(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer.screen_width = 16.0
    viewer.screen_distance = 16.0
    viewer.screen_pan_x = 0.0
    viewer.screen_pan_y = 0.0
    viewer._preset_index = 5

    assert not viewer._restore_screen_state()
    viewer.screen_width = 2.4
    viewer.screen_distance = 2.0
    assert viewer._persist_screen_state() is None
    assert viewer.screen_width == 2.4
    assert viewer.screen_distance == 2.0


def test_default_glow_off_uses_blank_fast_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_glow_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)

    assert viewer._default_blank_fast_path()
    assert not viewer._render_glow_called


def test_default_profile_starts_with_glow_off():
    import json

    profile_path = SRC / "xr_viewer" / "environments" / "Default" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["glow_mode"] == "off"
    assert profile["glow_intensity_multiplier"] == 0.0
    assert profile["glow_shell_intensity_multiplier"] == 0.0
    assert profile["lighting_preset_index"] == 0


def test_default_glow_on_keeps_background_effect_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_glow_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert viewer._render_glow_called


def test_default_surround_glow_uses_shell_render_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 1.0
    viewer._render_glow_called = False
    viewer._render_glow_shell_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_glow_shell(*_args):
        viewer._render_glow_shell_called = True

    viewer._render_glow = _render_glow
    viewer._render_glow_shell = _render_glow_shell
    viewer._render_screen_background_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert not viewer._render_glow_called
    assert viewer._render_glow_shell_called


def test_default_frosted_glow_uses_frosted_render_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "frosted"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_frosted_glow_called = False
    viewer._render_glow_called = False

    def _render_frosted_glow(*_args):
        viewer._render_frosted_glow_called = True

    def _render_glow(*_args):
        viewer._render_glow_called = True

    viewer._render_frosted_glow = _render_frosted_glow
    viewer._render_glow = _render_glow
    viewer._render_screen_background_effects(None, None)
    assert not viewer._render_frosted_glow_called
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert viewer._render_frosted_glow_called
    assert not viewer._render_glow_called


def test_default_veil_glow_uses_veil_render_path(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._bg_color_idx = 0
    viewer._glow_mode = "veil"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_frosted_veil_called = False
    viewer._render_frosted_glow_called = False

    def _render_frosted_veil(*_args):
        viewer._render_frosted_veil_called = True

    def _render_frosted_glow(*_args):
        viewer._render_frosted_glow_called = True

    viewer._render_frosted_veil = _render_frosted_veil
    viewer._render_frosted_glow = _render_frosted_glow
    viewer._render_screen_background_effects(None, None)
    assert not viewer._render_frosted_veil_called
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert viewer._render_frosted_veil_called
    assert not viewer._render_frosted_glow_called


def test_default_background_effects_skip_dark_room_board(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = [object()]
    viewer._dark_room_background = True
    viewer._bg_color_idx = 0
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._current_view_mat = object()
    viewer._render_glow_called = False
    viewer._render_env_model_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_env_model(*_args):
        viewer._render_env_model_called = True

    viewer._render_glow = _render_glow
    viewer._render_env_model = _render_env_model
    viewer._render_screen_background_effects(None, None)

    assert viewer._render_glow_called
    assert not viewer._render_env_model_called


def test_default_glow_does_not_skip_curved_screen():
    env_text = (SRC / "xr_viewer" / "environment.py").read_text(encoding="utf-8")

    assert "if getattr(self, '_screen_curved', False):\n            return" not in env_text


def test_default_glow_mode_cycle_from_y(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._glow_mode = "veil"
    viewer._glow_intensity_multiplier = 1.85
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._env_profile = {
        "glow_intensity_multiplier": 0.0,
        "glow_shell_intensity_multiplier": 0.0,
        "lighting_presets": [
            {
                "name": "Glow Off",
                "glow_mode": "off",
                "glow_intensity_multiplier": 0.0,
                "glow_shell_intensity_multiplier": 0.0,
            },
            {
                "name": "Frosted Veil",
                "glow_mode": "veil",
                "glow_intensity_multiplier": 1.85,
                "glow_shell_intensity_multiplier": 0.0,
                "frosted_veil_intensity": 1.35,
            },
            {
                "name": "Frosted Glow",
                "glow_mode": "frosted",
                "glow_intensity_multiplier": 1.85,
                "glow_shell_intensity_multiplier": 0.0,
                "frosted_glow_intensity": 3.0,
            },
            {
                "name": "Surround Glow",
                "glow_mode": "surround",
                "glow_intensity_multiplier": 0.0,
                "glow_shell_intensity_multiplier": 1.85,
            },
            {
                "name": "Screen Glow",
                "glow_mode": "screen",
                "glow_intensity_multiplier": 1.85,
                "glow_shell_intensity_multiplier": 0.0,
            },
        ],
    }
    viewer._save_glow_to_builtin_profile = lambda: None

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "frosted"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "surround"
    assert viewer._glow_intensity_multiplier == 0.0
    assert viewer._glow_shell_intensity_multiplier == 1.85

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "screen"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "off"
    assert viewer._glow_intensity_multiplier == 0.0
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "veil"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0


def test_frosted_glow_shader_uses_screen_texture_bright_blur():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    assert "_FROSTED_GLOW_VERT" in glsl_text
    assert "_FROSTED_GLOW_FRAG" in glsl_text
    assert "in vec3 in_attr" in glsl_text
    assert "uniform mat4 u_model" in glsl_text
    assert "uniform sampler2D u_screen_tex" in glsl_text
    assert "textureLod(u_screen_tex" in glsl_text
    assert "smoothstep(u_threshold, 1.0, luma)" in glsl_text
    assert "u_beam_softness" in glsl_text
    assert "u_frost_blend" in glsl_text
    assert "u_beam_thickness" in glsl_text
    assert "u_edge_inset" in glsl_text
    assert "u_diffuse_scatter" in glsl_text
    assert "edge_blur" in glsl_text
    assert "root_smear" in glsl_text
    assert "sample_edge_area" in glsl_text
    assert "color_keep" in glsl_text


def test_screen_glow_shader_uses_region_color_grid():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")

    assert "_GLOW_DOWNSAMPLE_FRAG" in glsl_text
    assert "uniform sampler2D u_glow_tex" in glsl_text
    assert "uniform sampler2D u_screen_light_tex" in glsl_text
    assert "textureLod(u_glow_tex" in glsl_text
    assert "textureLod(u_screen_light_tex" in glsl_text
    assert "glow_grid_color" in glsl_text


def test_surround_glow_shell_uses_screen_border_color():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")

    assert "_GLOW_SHELL_FRAG" in glsl_text
    assert "uniform sampler2D u_glow_tex" in glsl_text
    assert "uniform int u_glow_use_tex" in glsl_text
    assert "sample_border_color" in glsl_text
    assert "sample_region_reflection" in glsl_text
    assert "vec2 grid = vec2(16.0, 9.0)" in glsl_text
    assert "textureLod(u_glow_tex, sp, 2.2)" in glsl_text
    assert "region_mix" in glsl_text
    assert "side_weight" in glsl_text


def test_screen_glow_sampler_builds_16x9_color_grid(monkeypatch):
    import numpy as np

    viewer = _make_default_viewer(monkeypatch)
    rgb = np.zeros((90, 160, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(160, dtype=np.uint8)[None, :]
    rgb[:, :, 1] = np.arange(90, dtype=np.uint8)[:, None]

    viewer._sample_glow_target_color(rgb, is_tensor=False)

    assert len(viewer._screen_light_target_colors) == 144
    assert viewer._screen_light_target_colors[0][0] < viewer._screen_light_target_colors[15][0]
    assert viewer._screen_light_target_colors[0][1] < viewer._screen_light_target_colors[-1][1]


def test_screen_glow_does_not_trigger_cpu_color_sampling(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._glow_intensity = 1.0
    viewer._glow_intensity_multiplier = 1.0
    viewer._screen_light_dynamic = False
    viewer._bg_color_idx = 0
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = []
    viewer._glow_color_counter = 0
    viewer._sampled = False

    def _sample(*_args):
        viewer._sampled = True

    viewer._sample_glow_target_color = _sample
    viewer._maybe_sample_glow_target_color(None, is_tensor=False)

    assert not viewer._sampled
    assert viewer._glow_color_counter == 0


def test_environment_dynamic_light_does_not_trigger_cpu_color_sampling(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._glow_intensity = 1.0
    viewer._glow_intensity_multiplier = 0.0
    viewer._screen_light_dynamic = True
    viewer._screen_light_intensity = 2.0
    viewer._bg_color_idx = 0
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._dark_room_prims = []
    viewer._glow_color_counter = 0
    viewer._sampled = False

    def _sample(*_args):
        viewer._sampled = True

    viewer._sample_glow_target_color = _sample
    viewer._maybe_sample_glow_target_color(None, is_tensor=False)

    assert not viewer._sampled
    assert viewer._glow_color_counter == 0


def test_default_dark_room_does_not_trigger_cpu_color_sampling(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._glow_intensity = 1.0
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._screen_light_dynamic = True
    viewer._screen_light_intensity = 3.5
    viewer._bg_color_idx = 0
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._dark_room_prims = [object()]
    viewer._glow_color_counter = 0
    viewer._sampled = False

    def _sample(*_args):
        viewer._sampled = True

    viewer._sample_glow_target_color = _sample
    viewer._maybe_sample_glow_target_color(None, is_tensor=False)

    assert not viewer._sampled
    assert viewer._glow_color_counter == 0


def test_frosted_glow_keyboard_adjusts_realtime_params(monkeypatch):
    import glfw

    viewer = _make_default_viewer(monkeypatch)
    viewer._frosted_glow_blend = 1.0
    viewer._frosted_glow_thickness = 1.5
    viewer._preset_name_overlay = ""
    viewer._preset_osd_show_t = 0.0

    assert viewer._adjust_frosted_glow_keyboard(glfw.KEY_RIGHT)
    assert viewer._frosted_glow_blend == 1.05
    assert viewer._frosted_glow_thickness == 1.5

    assert viewer._adjust_frosted_glow_keyboard(glfw.KEY_UP, glfw.MOD_SHIFT)
    assert viewer._frosted_glow_blend == 1.05
    assert viewer._frosted_glow_thickness == 1.65
    assert "Frosted blend 1.05 / thickness 1.65" == viewer._preset_name_overlay

    viewer._frosted_glow_blend = 0.01
    viewer._frosted_glow_thickness = 2.99
    assert viewer._adjust_frosted_glow_keyboard(glfw.KEY_LEFT)
    assert viewer._frosted_glow_blend == 0.0
    assert viewer._adjust_frosted_glow_keyboard(glfw.KEY_UP)
    assert viewer._frosted_glow_thickness == 3.0


def test_frosted_glow_virtual_keyboard_arrows_adjust_params(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._frosted_glow_blend = 1.0
    viewer._frosted_glow_thickness = 1.5
    viewer._preset_name_overlay = ""
    viewer._preset_osd_show_t = 0.0

    assert viewer._adjust_frosted_glow_vk(0x27)
    assert viewer._frosted_glow_blend == 1.05
    assert viewer._frosted_glow_thickness == 1.5

    assert viewer._adjust_frosted_glow_vk(0x28)
    assert viewer._frosted_glow_blend == 1.05
    assert viewer._frosted_glow_thickness == 1.45
