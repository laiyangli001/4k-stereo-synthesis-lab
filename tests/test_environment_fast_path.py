from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_environment_profiles_imports_json_for_profile_io():
    source = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")

    assert "import json" in source
    assert "json.load" in source
    assert "json.dump" in source


def test_environment_primitives_drop_legacy_material_dict_key():
    source = (SRC / "xr_viewer" / "environment_model.py").read_text(encoding="utf-8")
    d3d_source = (SRC / "xr_viewer" / "d3d11_native_renderer.py").read_text(encoding="utf-8")
    generate_default_room = source.split("def _generate_default_room", 1)[1].split("def _init_dark_room", 1)[0]
    load_env_model = source.split("def _load_env_model", 1)[1].split("def _release_env_model_resources", 1)[0]
    draw_environment = d3d_source.split("def _draw_environment_model", 1)[1].split("def _draw_lasers", 1)[0]

    assert "GltfMaterial" in source
    assert "material_contract = GltfMaterial(" in generate_default_room
    assert "'material_contract': material_contract" in generate_default_room
    assert "'render_material':" in generate_default_room
    assert "prim['render_material'] =" in load_env_model
    assert "prim['material']" not in load_env_model
    assert "'material':" not in generate_default_room
    assert 'prim.get("render_material")' in draw_environment
    assert 'prim.get("material")' not in draw_environment
    assert "render_pass_from_primitive(prim)" in source


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


def _set_runtime_effect_safe(viewer, tex, size, frame_id):
    if not hasattr(viewer, "_runtime_effect_scheduler"):
        from xr_viewer.effect_scheduler import EffectScheduler

        scheduler = EffectScheduler()
        viewer._runtime_effect_scheduler = lambda: scheduler
        viewer._runtime_effect_submit_scheduler = lambda: scheduler
    scheduler = viewer._runtime_effect_scheduler()
    pool = scheduler.pool
    slot = pool._idle_slot()
    slot.tex = tex
    slot.size = size
    if size is None:
        slot.frame_id = int(frame_id or 0)
        slot.state = "safe"
        pool.safe_slot = slot
        return
    pool.writing_slot = slot
    scheduler.publish_completed(size[0], size[1], frame_id)
    scheduler.poll_completed()


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


def test_no_room_screen_effects_wait_for_source_ready(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    viewer._screen_effects_enabled = True
    viewer.screen_height = 9.0
    viewer._runtime_direct_source = False
    viewer._should_show_source_border = lambda: False
    viewer._render_glow_called = False
    viewer._render_metallic_border_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_metallic_border(*_args):
        viewer._render_metallic_border_called = True

    viewer._render_glow = _render_glow
    viewer._render_metallic_border = _render_metallic_border
    viewer._render_screen_background_effects(None, None)
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._render_glow_called
    assert not viewer._render_metallic_border_called


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


def test_environment_screen_profile_can_unlock_controller_move(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)

    viewer._screen_profile = {}
    assert not viewer._environment_screen_locked()

    viewer._screen_profile = {"width": 2.4}
    assert viewer._environment_screen_locked()

    viewer._screen_profile = {"width": 2.4, "allow_controller_move": True}
    assert not viewer._environment_screen_locked()

    viewer._screen_profile = {"width": 2.4, "controller_movable": "true"}
    assert not viewer._environment_screen_locked()

    viewer._screen_profile = {"width": 2.4, "unlock_controller_move": "1"}
    assert not viewer._environment_screen_locked()


def test_artemis_allows_controller_screen_move_for_large_scene():
    import json

    profile_path = SRC / "xr_viewer" / "environments" / "Artemis" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert profile["screen"]["allow_controller_move"] is True


def test_artemis_large_scene_uses_active_view_pose_and_long_xr_far_plane():
    import json

    profile_path = SRC / "xr_viewer" / "environments" / "Artemis" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert "view_pose" not in profile
    assert profile["view_pose_index"] == 1
    assert profile["model_position"][:2] == [0.0, 843.0854]
    assert profile["view_poses"][1]["name"] == "Model Center"
    assert all(isinstance(profile["view_poses"][1][axis], (int, float)) for axis in ("x", "y", "z"))
    assert profile["xr_projection_near"] == 0.1
    assert profile["xr_projection_far"] >= 20000.0


def test_openxr_projection_clip_planes_are_profile_configurable():
    impl = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    profiles = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")
    pipeline = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    presenter = (SRC / "xr_viewer" / "projection_layer_presenter.py").read_text(encoding="utf-8")

    assert "self._xr_projection_near = max(0.01" in impl
    assert "self._xr_projection_far = max(self._xr_projection_near + 1.0" in impl
    assert "profile['xr_projection_far']" in profiles
    assert "self._refresh_default_projection()" in pipeline
    assert "_fov_to_proj_mat4(self.default_fov, near=near, far=far)" in pipeline
    assert "_fov_to_proj_mat4_d3d(self.default_fov, near=near, far=far)" in pipeline
    assert "_fov_to_proj_mat4(view.fov, near=near, far=far)" in presenter
    assert "_fov_to_proj_mat4_d3d(view.fov, near=near, far=far)" in presenter


def test_environment_readme_documents_controller_move_unlock():
    source = (SRC / "xr_viewer" / "environments" / "README.md").read_text(encoding="utf-8")

    assert "allow_controller_move" in source
    assert "by default" in source
    assert "controller grip move/resize controls" in source


def test_profile_view_pose_accepts_xyz_position_alias(monkeypatch):
    import math

    viewer = _make_default_viewer(monkeypatch)

    assert viewer._view_pose_has_explicit_layout({"x": 1.0, "y": 2.0, "z": 3.0})
    assert viewer._profile_view_position({"x": 1.0, "y": 2.0, "z": 3.0}, [0.0, 0.0, 0.0]) == [1.0, 2.0, 3.0]
    assert viewer._profile_view_position(
        {"position": [4.0, 5.0, 6.0], "x": 1.0, "y": 2.0, "z": 3.0},
        [0.0, 0.0, 0.0],
    ) == [4.0, 5.0, 6.0]
    assert viewer._profile_view_rotation_rad(
        {"angle": 90.0},
        ("rotation_deg",),
        ("rotation",),
        [0.0, 0.0, 0.0],
    ) == [math.radians(90.0), 0.0, 0.0]


def test_profile_view_pose_xr_space_drops_startup_head_pitch(monkeypatch):
    import math
    import numpy as np
    import xr_viewer.environment_layout as environment_layout
    from xr_viewer.xr_math import euler_to_mat4

    viewer = _make_default_viewer(monkeypatch)
    captured = {}

    class FakeXr:
        class ReferenceSpaceCreateInfo:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        @staticmethod
        def create_reference_space(session, info):
            captured["session"] = session
            captured["pose"] = info.kwargs["pose_in_reference_space"]
            return "new-space"

        @staticmethod
        def destroy_space(_space):
            captured["destroyed"] = True

    def capture_pose(matrix):
        captured["matrix"] = np.array(matrix, dtype=np.float32)
        return "pose"

    monkeypatch.setattr(environment_layout, "xr", FakeXr)
    monkeypatch.setattr(environment_layout, "mat4_to_xr_posef", capture_pose)
    raw_head = euler_to_mat4(0.0, math.radians(18.0), math.radians(7.0)).astype(np.float32)
    raw_head[:3, 3] = np.array([1.0, 1.6, -2.0], dtype=np.float32)
    viewer._head_model_mat4_from_views = lambda _views: raw_head
    viewer._xr_profile_space_applied = False
    viewer._view_pose_profile = {"x": 1.0, "y": 1.6, "z": -2.0}
    viewer._xr_session = "session"
    viewer._xr_space = "old-space"
    viewer._xr_ref_space_type = "local"
    viewer._xr_space_pose_in_ref = np.eye(4, dtype=np.float32)

    assert viewer._apply_profile_view_pose_to_xr_space([object(), object()]) is True

    np.testing.assert_allclose(captured["matrix"][:3, :3], np.eye(3, dtype=np.float32), atol=1e-5)
    np.testing.assert_allclose(captured["matrix"][:3, 3], np.zeros(3, dtype=np.float32), atol=1e-5)

def test_environment_switch_does_not_reset_profile_view_pose_for_movable_screen():
    source = (SRC / "xr_viewer" / "environment_model.py").read_text(encoding="utf-8")
    switch_func = source.split("def _switch_environment_model", 1)[1]

    assert "has_screen_profile =" in switch_func
    assert "has_view_pose = self._view_pose_has_explicit_layout()" in switch_func
    assert "if not has_view_pose:\n            self._reset_xr_space_to_identity()" in switch_func
    assert "if not has_screen_profile:\n            if not self._restore_screen_state():" in switch_func
    assert "if not self._environment_screen_locked():\n            self._reset_xr_space_to_identity()" not in switch_func


def test_left_grip_screen_drag_uses_cursor_uv_and_delta_limit():
    source = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    drag_loop = source.split("# Per-controller laser requirement", 1)[1].split("# Keyboard grip-to-move", 1)[0]

    assert "cursor_uv = self._cursor_uv_l if is_left else self._cursor_uv_r" in drag_loop
    assert "hit_world = self._screen_uv_to_world(float(cursor_uv[0]), float(cursor_uv[1]))" in drag_loop
    assert "delta_len = math.hypot(delta_x, delta_y)" in drag_loop
    assert "max_speed = max(1.0, min(80.0, float(self.screen_width) * 0.75))" in drag_loop
    assert "max_delta = max_speed * max(float(dt), 1.0 / 120.0)" in drag_loop
    assert "self.screen_pan_x += float(np.dot(delta, right_axis))" not in drag_loop
    assert "self.screen_pan_y += float(np.dot(delta, up_axis))" not in drag_loop


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


def test_default_profile_starts_with_surround_glow():
    import json

    profile_path = SRC / "xr_viewer" / "environments" / "Default" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))

    assert "xr_quad_layer_enabled" not in profile
    assert profile["glow_mode"] == "veil"
    assert profile["controller_hdr_lighting"] is False
    assert profile["glow_intensity_multiplier"] == 1.85
    assert profile["glow_shell_intensity_multiplier"] == 0.0
    assert profile["frosted_glow_blend"] == 2.40
    assert profile["frosted_glow_thickness"] == 2.40
    assert profile["lighting_preset_index"] == 0
    assert profile["lighting_presets"][0]["glow_mode"] == "surround"


def test_builtin_environment_profiles_do_not_disable_quad_layer():
    import json

    for name in ("Default", "Cinema"):
        profile_path = SRC / "xr_viewer" / "environments" / name / "profile.json"
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        assert "xr_quad_layer_enabled" not in profile


def test_screen_light_bind_failure_disables_effect_without_raising(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    inc_calls = []

    class Tex:
        def use(self, location=0):
            raise RuntimeError(f"bad bind {location}")

    viewer._screen_light_source_texture = lambda: (Tex(), (16, 9))
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._bind_screen_light_source_texture(location=10) is None
    assert ("openxr_screen_light_bind_failed", 1) in inc_calls


def test_environment_light_bind_failure_disables_uniform(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)

    class Uniform:
        def __init__(self):
            self.value = None

    viewer._env_prog = {"u_screen_light_enabled": Uniform()}
    viewer.screen_height = 9.0
    viewer._screen_light_intensity = 1.0
    viewer._bind_screen_light_source_texture = lambda: None
    viewer._cl_light_state_key = object()
    viewer._cl_uniform_frame = 10

    viewer._apply_cinema_light_uniforms()

    assert viewer._env_prog["u_screen_light_enabled"].value == 0
    assert viewer._cl_light_state_key is None
    assert viewer._cl_uniform_frame == -5


def test_runtime_effect_consumers_only_read_existing_safe_results(monkeypatch):
    safe_tex = object()
    inc_calls = []

    for viewer in (_make_default_viewer(monkeypatch), _make_no_room_viewer(monkeypatch)):
        viewer._frame_count = 7
        viewer._runtime_direct_source = True
        _set_runtime_effect_safe(viewer, safe_tex, (16, 9), 6)
        viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
        viewer._record_screen_effect_safe_age = lambda _source_tex, _frame_id=None: None

        assert viewer._screen_effect_source_texture()[0] is safe_tex

    viewer = _make_default_viewer(monkeypatch)
    viewer._frame_count = 8
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, safe_tex, (16, 9), 7)
    viewer._cached_glow_downsample_texture = lambda _tex, _size: None
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._screen_light_source_texture() == (None, None)
    assert ("openxr_effect_source_promote_failed", 1) not in inc_calls


def test_screen_light_source_lookup_failure_disables_light_without_safe_downsample(monkeypatch):
    safe_tex = object()
    inc_calls = []
    viewer = _make_default_viewer(monkeypatch)
    viewer._frame_count = 8
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, safe_tex, (16, 9), 7)
    viewer._record_screen_effect_safe_age = lambda _source_tex, _frame_id=None: None
    viewer._cached_glow_downsample_texture = lambda *_args: (_ for _ in ()).throw(RuntimeError("cache failed"))
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._screen_light_source_texture() == (None, None)
    assert ("openxr_screen_light_source_failed", 1) not in inc_calls


def test_screen_effect_age_diagnostic_failure_does_not_break_effect_source(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = object()
    inc_calls = []
    viewer._frame_count = 9
    _set_runtime_effect_safe(viewer, source_tex, None, 7)
    viewer._breakdown_add_value = lambda *_args: (_ for _ in ()).throw(RuntimeError("stats failed"))
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    viewer._record_screen_effect_safe_age(source_tex)

    assert ("openxr_effect_ready_age_record_failed", 1) in inc_calls


def test_panorama_profile_config_resolves_image(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    room = tmp_path / "PanoramaRoom"
    room.mkdir()
    image = room / "background.jpg"
    image.write_bytes(b"not-a-real-image")

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "background": {"image": "background.jpg", "yaw_offset_deg": 45.0},
        },
        str(room),
    )

    assert is_panorama
    assert path == str(image)
    assert cfg["image"] == "background.jpg"
    assert cfg["yaw_offset_deg"] == 45.0

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "panorama": {
                "image": "background.jpg",
                "stereo_layout": "sbs",
                "wall_light_mask": "mask.png",
                "screen_light_layout": {"uv": [0.4, 0.6], "radius": [0.2, 0.1]},
            },
        },
        str(room),
    )

    assert is_panorama
    assert path == str(image)
    assert cfg["stereo_layout"] == "sbs"
    assert cfg["wall_light_mask"] == "mask.png"
    assert cfg["screen_light_layout"]["uv"] == [0.4, 0.6]


def test_panorama_profile_auto_wall_mask_bakes_cached_png(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    room = tmp_path / "PanoramaRoom"
    room.mkdir()
    image = room / "background.jpg"
    image.write_bytes(b"not-a-real-image")

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "panorama": {
                "image": "background.jpg",
                "wall_light_mask": "auto",
                "wall_light_mask_resolution": [64, 32],
                "screen_light_layout": {"uv": [0.25, 0.5], "radius": [0.1, 0.2]},
            },
        },
        str(room),
    )

    mask_path = room / cfg["wall_light_mask"]
    assert is_panorama
    assert path == str(image)
    assert mask_path.is_file()
    assert mask_path.parent.name == ".d2s_bake"
    assert cfg["wall_light_mask"].endswith(".png")
    from PIL import Image
    assert Image.open(mask_path).size == (64, 32)


def test_panorama_profile_auto_wall_mask_without_image_does_not_crash(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    room = tmp_path / "PanoramaRoom"
    room.mkdir()

    is_panorama, path, cfg = viewer._panorama_profile_config(
        {
            "environment_type": "panorama",
            "panorama": {
                "wall_light_mask": "auto",
                "wall_light_mask_resolution": [32, 16],
            },
        },
        str(room),
    )

    assert is_panorama
    assert path is None
    assert (room / cfg["wall_light_mask"]).is_file()


def test_environment_discovery_includes_panorama_image_folder(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    root = tmp_path / "environments"
    pano = root / "Pano"
    pano.mkdir(parents=True)
    (pano / "panorama.png").write_bytes(b"not-a-real-image")
    viewer._environment_root = str(root)
    viewer._environment_model = "Default"

    assert "Pano" in viewer._discover_environment_models()


def test_environment_discovery_includes_hdr_panorama_folder(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    root = tmp_path / "environments"
    pano = root / "HDR Pano"
    pano.mkdir(parents=True)
    (pano / "background.hdr").write_bytes(b"not-a-real-hdr")
    viewer._environment_root = str(root)
    viewer._environment_model = "Default"

    assert "HDR Pano" in viewer._discover_environment_models()


def test_radiance_hdr_loader_decodes_flat_rgbe(tmp_path):
    from xr_viewer.environment_renderer import _hdr_to_ldr_u8, _read_radiance_hdr

    hdr = tmp_path / "tiny.hdr"
    hdr.write_bytes(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"
        + bytes([128, 64, 32, 129, 0, 0, 0, 0])
    )

    arr, size = _read_radiance_hdr(str(hdr))

    assert size == (2, 1)
    assert arr.shape == (1, 2, 3)
    assert arr[0, 0, 0] > arr[0, 0, 1] > arr[0, 0, 2]
    assert arr[0, 1].sum() == 0.0
    ldr = _hdr_to_ldr_u8(arr)
    assert ldr.dtype.name == "uint8"
    assert ldr.shape == arr.shape


def test_panorama_hdr_texture_uses_float_upload(monkeypatch, tmp_path):
    from xr_viewer import environment_renderer

    viewer = _make_default_viewer(monkeypatch)
    hdr = tmp_path / "tiny.hdr"
    hdr.write_bytes(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 1 +X 2\n"
        + bytes([128, 64, 32, 129, 0, 0, 0, 0])
    )

    class _Texture:
        filter = None
        repeat_x = False
        repeat_y = False

        def build_mipmaps(self):
            pass

    calls = []

    class _Ctx:
        info = {"GL_MAX_TEXTURE_SIZE": 8192}

        def texture(self, size, components, data, **kwargs):
            calls.append((size, components, len(data), kwargs.get("dtype")))
            return _Texture()

    def _fail_ldr(_arr):
        raise AssertionError("HDR panorama should not be tone-mapped before float upload")

    viewer.ctx = _Ctx()
    viewer._panorama_background_path = str(hdr)
    viewer._panorama_tex = None
    viewer._panorama_tex_path = None
    monkeypatch.setattr(environment_renderer, "_hdr_to_ldr_u8", _fail_ldr)

    assert viewer._get_panorama_texture() is not None
    assert calls == [((2, 1), 3, 2 * 1 * 3 * 2, "f2")]


def test_official_webxr_hdr_environments_are_packaged():
    import pytest

    names = {
        "WebXR Autumn Forest": "autumn_forest_01_2k.hdr",
        "WebXR Cave Wall": "cave_wall_2k.hdr",
        "WebXR Fireplace": "fireplace_2k.hdr",
        "WebXR Georgentor": "georgentor_2k.hdr",
        "WebXR Snowy Park": "snowy_park_01_2k.hdr",
        "WebXR Studio": "studio_small_03_2k.hdr",
    }
    if not (SRC / "xr_viewer" / "environments" / "WebXR Autumn Forest" / "profile.json").is_file():
        pytest.skip("optional WebXR HDR panorama assets are not packaged")
    for env_name, hdr_name in names.items():
        env_dir = SRC / "xr_viewer" / "environments" / env_name
        profile = (env_dir / "profile.json").read_text(encoding="utf-8")
        assert (env_dir / hdr_name).is_file()
        assert '"environment_type": "panorama"' in profile
        assert f'"image": "{hdr_name}"' in profile
        assert '"controller_hdr_lighting": true' in profile


def test_panorama_profile_survives_viewer_initialization(monkeypatch):
    import pytest

    if not (SRC / "xr_viewer" / "environments" / "WebXR Autumn Forest" / "profile.json").is_file():
        pytest.skip("optional WebXR HDR panorama assets are not packaged")
    monkeypatch.chdir(SRC)
    from xr_viewer.environment import OpenXRViewer

    viewer = OpenXRViewer(environment_model="WebXR Autumn Forest", show_preview_window=False)

    assert viewer._env_model_path is None
    assert viewer._panorama_background_path.endswith("autumn_forest_01_2k.hdr")
    assert viewer._panorama_background_settings["image"] == "autumn_forest_01_2k.hdr"
    assert viewer._controller_hdr_lighting is True


def test_async_panorama_background_disables_default_glb_mesh(monkeypatch, tmp_path):
    import json

    root = tmp_path / "environments"
    default = root / "Default"
    default.mkdir(parents=True)
    (default / "profile.json").write_text(
        json.dumps({"xr_quad_layer_enabled": False}),
        encoding="utf-8",
    )
    (default / "environment.glb").write_bytes(b"glb")
    monkeypatch.chdir(SRC)
    from xr_viewer.environment import OpenXRViewer

    viewer = OpenXRViewer(environment_model="None", show_preview_window=False)
    viewer._environment_root = str(root)
    viewer._environment_model = "Default"
    viewer._openxr_panorama_background_enabled = True

    viewer._configure_environment_profile()

    assert viewer._env_model_path is None
    assert viewer._panorama_background_path is None
    assert not hasattr(viewer, "_xr_quad_layer_enabled")

def test_async_panorama_background_keeps_named_glb_room(monkeypatch, tmp_path):
    import json

    root = tmp_path / "environments"
    room = root / "Bedroom"
    room.mkdir(parents=True)
    (room / "profile.json").write_text(
        json.dumps({"glb": "environment.glb"}),
        encoding="utf-8",
    )
    (room / "environment.glb").write_bytes(b"glb")
    viewer = _make_default_viewer(monkeypatch)
    viewer._env_base_settings = {
        "model_pos": [0.0, -1.0, -3.0],
        "model_rot": [0.0, 0.0, 0.0],
        "model_scale": [1.0, 1.0, 1.0],
        "head_light_color": [0.3, 0.3, 0.3],
        "ambient_color": [0.1, 0.1, 0.1],
        "fallback_dir": [0.25, -0.82, -0.52],
        "fallback_dir_color": [1.0, 1.0, 1.0],
        "fill_lights": [],
        "exposure": 1.0,
        "gamma": 2.2,
        "emissive_strength": 1.0,
        "khr_light_scale": 1.0,
        "render_quality": "balanced",
        "shading_mode": "pbr",
        "texture_anisotropy": 1.0,
        "perf_log": False,
        "xr_render_scale": 1.0,
        "screen_light_intensity": 3.5,
        "controller_hdr_lighting": True,
    }
    viewer._environment_root = str(root)
    viewer._environment_model = "Bedroom"
    viewer._openxr_panorama_background_enabled = True
    viewer._screen_light_intensity = 3.5

    viewer._configure_environment_profile()

    assert viewer._env_model_path == str(room / "environment.glb")
    assert viewer._panorama_background_path is None

def test_opengl_projection_renders_env_model_before_screen():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")

    background_idx = impl_text.index("background_presenter.render_projection_background(")
    env_idx = impl_text.index("self._render_env_model(mgl_fbo, vp_mat, view_mat)")
    screen_idx = impl_text.index("self._screen_layer_presenter.render_projection_screen(")

    assert background_idx < env_idx < screen_idx

def test_panorama_environment_skips_glb_initialization(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_enabled = True
    viewer._panorama_background_path = "background.jpg"
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._active_environment = "Pano"

    viewer._init_env_model()

    assert not viewer._env_model_visible
    assert viewer._env_model_prims == []
    assert viewer._active_environment == "Pano"


def test_openxr_loop_uses_fast_env_model_initializer():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    pipeline_text = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    renderer_text = (SRC / "xr_viewer" / "openxr_frame_renderer.py").read_text(encoding="utf-8")

    assert "def _ensure_env_model_initialized" in impl_text
    assert "_ensure_env_model_initialized(\"Preview-only\")" in pipeline_text
    assert "_init_env_model()" not in pipeline_text

    poll_idx = renderer_text.index("self.screen_presenter.poll_screen_frame()")
    quad_idx = renderer_text.index("self.screen_presenter.prepare_frame_layers(screen_frame_uploaded=screen_frame_uploaded)")
    projection_idx = renderer_text.index("self.projection_presenter.render_projection(")
    submit_idx = pipeline_text.index("self.frame_submitter.submit(")
    flush_idx = pipeline_text.index("self.effect_submitter.flush_after_submit(", submit_idx)

    assert poll_idx < quad_idx < projection_idx
    assert submit_idx < flush_idx
    assert "_ensure_env_model_initialized(\"Lazy\")" not in pipeline_text


def test_openxr_no_fresh_but_renderable_source_continues_to_screen_present():
    pipeline_text = (SRC / "xr_viewer" / "openxr_frame_pipeline.py").read_text(encoding="utf-8")
    source_state = (SRC / "xr_viewer" / "core_source_state.py").read_text(encoding="utf-8")
    stale_block = pipeline_text.split("if viewer._session_ready_pending or not viewer._has_fresh_source_frame(now):", 1)[1].split(
        "frame_state, submit_start = self.timing.begin_frame", 1
    )[0]
    gate_text = (SRC / "xr_viewer" / "openxr_frame_gate.py").read_text(encoding="utf-8")

    assert "viewer._poll_source_frame(upload=False)" in stale_block
    assert "_pause_xr_output_for_source_stall" in source_state
    assert "quad_presentable = getattr(viewer, '_quad_layer_screen_presentable', lambda: False)()" not in gate_text
    assert "if not viewer._has_renderable_source_frame():" in gate_text
    assert "self.frame_submitter.submit(" in gate_text

    stale_idx = pipeline_text.index("if viewer._session_ready_pending or not viewer._has_fresh_source_frame(now):")
    poll_upload_idx = pipeline_text.index("self.renderer.render_frame(")
    assert stale_idx < poll_upload_idx


def test_env_model_initializer_skips_panorama_background(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._env_model_init_done = False
    viewer._environment_enabled = True
    viewer._panorama_background_path = "background.jpg"
    viewer._env_model_path = "environment.glb"
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._called = False
    viewer._panorama_called = False

    def _init_env_model():
        viewer._called = True

    def _get_panorama_texture():
        viewer._panorama_called = True
        return object()

    viewer._init_env_model = _init_env_model
    viewer._get_panorama_texture = _get_panorama_texture
    viewer._ensure_env_model_initialized("Test")

    assert viewer._env_model_init_done
    assert not viewer._called
    assert viewer._panorama_called
    assert not viewer._env_model_visible
    assert viewer._env_model_prims == []


def test_openxr_async_plan_uses_runtime_view_pose_and_project_side_complex_bake():
    plan = (ROOT / "docs" / "36-OpenXR_Asynchronous_Decoupled_Rendering_Implementation_Plan.md").read_text(encoding="utf-8")
    report = (ROOT / "docs" / "35-OpenXR_Asynchronous_Decoupled_Rendering_Architecture_Report.md").read_text(encoding="utf-8")

    assert "真实位置与朝向" in plan
    assert "当前目标以 3DoF 原地转头为前提" in plan
    assert "rotation-only" not in plan
    assert "3DoF 前提" in report
    assert "未来若支持 6DoF" in report
    assert "rotation-only" not in report
    assert "使用当前最新 head pose" in report
    assert "复杂房间 mask bake 仍待接入" not in plan
    assert "项目内不实现复杂房间 bake" in plan


def test_panorama_background_is_preloaded_outside_render_path():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    model_text = (SRC / "xr_viewer" / "environment_model.py").read_text(encoding="utf-8")
    init_func = impl_text.split("def _ensure_env_model_initialized", 1)[1].split("    # Main blocking loop", 1)[0]
    settings_func = render_text.split("def _panorama_render_settings", 1)[1].split("def _render_panorama_background", 1)[0]
    render_func = render_text.split("def _render_panorama_background", 1)[1].split("def _render_env_model", 1)[0]
    switch_func = model_text.split("def _switch_environment_model", 1)[1]
    pano_frag = glsl_text.split("_PANORAMA_FRAG", 1)[1].split("_GLOW_DOWNSAMPLE_FRAG", 1)[0]

    assert "get_panorama_texture()" in init_func
    assert "_panorama_texture_ready()" in render_func
    assert "_panorama_light_mask_texture_ready()" in render_func
    assert "_get_panorama_texture()" not in render_func
    assert "_get_panorama_light_mask_texture()" not in render_func
    assert "_panorama_light_mask_path_from_settings()" not in render_func
    assert "_ensure_env_model_initialized(\"Switch\")" in switch_func
    assert "_init_env_model()" not in switch_func
    controller_render = (SRC / "xr_viewer" / "core_laser_render.py").read_text(encoding="utf-8").split(
        "def _render_controllers", 1
    )[1]
    assert "_bind_screen_light_source_texture(location=10)" in controller_render
    assert "_get_panorama_texture()" not in controller_render
    assert "np.linalg.inv(view_mat)" not in controller_render
    assert "uniform sampler2D u_screen_light_tex" in pano_frag
    assert "uniform sampler2D u_wall_light_mask_tex" in pano_frag
    assert "vec3 screen_light_probe_color()" in pano_frag
    assert "textureLod(u_screen_light_tex" in pano_frag
    assert "textureLod(u_wall_light_mask_tex" in pano_frag
    assert "return color * (1.0 / 9.0)" in pano_frag
    assert "u_screen_light_uv" in pano_frag
    assert "u_screen_light_radius" in pano_frag
    assert "uniform int u_stereo_layout" in pano_frag
    assert "uniform int u_eye_index" in pano_frag
    assert "sample_uv.x = pano_uv.x * 0.5 + (u_eye_index == 1 ? 0.5 : 0.0)" in pano_frag
    assert "screen_light_layout" in settings_func
    assert "stereo_layout_raw" in settings_func
    assert "'sbs'" in settings_func
    assert "u_stereo_layout" in render_func
    assert "u_eye_index" in render_func
    assert "light_layout.get('uv'" in settings_func
    assert "light_layout.get('radius'" in settings_func
    assert "return False" in render_func
    assert "return True" in render_func
    assert "except Exception as exc:" in render_func
    assert "openxr_background_panorama_failed" in render_func
    assert "finally:" in render_func
    render_finally = render_func.split("finally:", 1)[1]
    assert "set_depth_mask(previous_depth_mask)" in render_finally
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally
    assert "tex.use(location=8)" in render_func
    assert "_bind_screen_light_source_texture(location=10)" in render_func
    assert "mask_tex.use(location=11)" in render_func
    assert "_view_mat_inv(view_rot)" in render_func
    assert "np.linalg.inv(view_rot)" not in render_func
    assert "_panorama_render_settings()" in render_func
    assert "screen_light_layout" not in render_func
    assert "_panorama_render_settings_key" in settings_func
    assert "_panorama_light_mask_path_from_settings" in render_text
    assert "_panorama_light_mask_path_key" in render_text
    background_presenter = (SRC / "xr_viewer" / "background_presenter.py").read_text(encoding="utf-8")
    background_layer_renderer = (SRC / "xr_viewer" / "background_layer_renderer.py").read_text(encoding="utf-8")
    eye_background = impl_text.split("background_presenter.render_projection_background(", 1)[1].split(
        "if perf_enabled:", 1
    )[0]
    assert "from .background_presenter import BackgroundPresenter" in impl_text
    assert "projection_screen_enabled" not in eye_background
    assert "background_presenter.projection_fallback_needed()" not in eye_background
    assert "def projection_fallback_needed" in background_presenter
    assert "BackgroundLayerRenderer" in background_presenter
    assert "ready = getattr(self.viewer, '_panorama_texture_ready', None)" in background_layer_renderer
    assert "projection_screen_enabled" not in background_presenter
    assert "projection_fallback_needed=getattr(" in eye_background
    assert "if viewer._render_panorama_background(mgl_fbo, view_mat, proj_mat):" in background_presenter
    assert "viewer._breakdown_inc('openxr_background_panorama')" in background_presenter
    assert "if eye_index == 0:" in background_presenter
    assert background_presenter.count("if eye_index == 0:") == 1


def test_projection_screen_path_skips_glb_environment_mesh_hot_path():
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    render_eye = impl_text.split("def _render_eye", 1)[1].split("# 3. Keyboard", 1)[0]
    background_presenter = (SRC / "xr_viewer" / "background_presenter.py").read_text(encoding="utf-8")
    screen_presenter = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")

    assert "_openxr_quad_screen_unavailable_reason" in screen_presenter
    assert "_openxr_projection_screen_unavailable_reason" not in screen_presenter
    assert "draw_projection_screen" not in render_eye
    assert "_openxr_draw_projection_screen" not in render_eye
    assert "projection_screen_enabled=" not in render_eye
    assert "self._screen_layer_presenter.render_projection_screen(" in render_eye
    assert "def render_projection_screen" in screen_presenter
    assert "background_presenter.projection_fallback_needed()" not in render_eye
    assert "and not panorama_configured" not in background_presenter
    assert "viewer._render_env_model(mgl_fbo, vp_mat, view_mat)" not in background_presenter


def test_quad_screen_path_keeps_panorama_projection_fallback(monkeypatch):
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    viewer = _make_default_viewer(monkeypatch)
    viewer._quad_layer_screen_presentable = lambda: True
    viewer._panorama_background_path = "room.hdr"
    viewer._panorama_texture_ready = lambda: object()
    viewer._background_presenter = None
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    viewer._keyboard_visible = False
    viewer._aim_mat_l = None
    viewer._aim_mat_r = None
    viewer._grip_mat_l = None
    viewer._grip_mat_r = None
    viewer._border_alpha = 0.0
    viewer._depth_osd_tex = None
    viewer._screen_osd_tex = None
    viewer._preset_osd_tex = None
    viewer._seat_adjust_osd_tex = None
    viewer._brand_osd_tex = None
    viewer._hand_fps_visible = False
    viewer._overlay_tex = None
    viewer._team_fps_visible = False
    viewer._team_status_tex = None
    viewer._calibration_mode = False
    viewer._fps_overlay_visible = False
    viewer._help_tex = None
    viewer._team_status_visible = False
    viewer._team_help_visible = False
    viewer._team_help_tex = None

    presenter = ScreenLayerPresenter(viewer)
    assert presenter.projection_layer_needed() is True
    assert presenter.projection_layer_reason() == "panorama_projection_fallback"
    assert viewer._background_layer_renderer is not None


def test_screen_layer_presenter_preloads_panorama_background(monkeypatch):
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    viewer = _make_default_viewer(monkeypatch)
    calls = []
    viewer._panorama_background_path = "background.hdr"
    viewer._panorama_tex = None
    viewer._get_panorama_texture = lambda: calls.append("texture")
    viewer._get_panorama_light_mask_texture = lambda: calls.append("mask")

    ScreenLayerPresenter(viewer).ensure_panorama_background_ready()

    assert calls == ["texture", "mask"]


def test_screen_layer_presenter_reuses_background_layer_gate_result(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    viewer = _make_default_viewer(monkeypatch)
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._background_layer_renderer = SimpleNamespace(
        make_background_layers=lambda: ([], False),
        panorama_ready=lambda: (_ for _ in ()).throw(AssertionError("background gate should not run twice")),
        native_background_available=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("background gate should not run twice")),
    )
    viewer._update_quad_layer_swapchains = lambda force=False: [0]
    viewer._make_quad_layer = lambda _eye_index: ctypes.c_int(9)
    viewer._keyboard_visible = False
    viewer._aim_mat_l = None
    viewer._aim_mat_r = None
    viewer._grip_mat_l = None
    viewer._grip_mat_r = None
    viewer._border_alpha = 0.0
    viewer._depth_osd_tex = None
    viewer._screen_osd_tex = None
    viewer._preset_osd_tex = None
    viewer._seat_adjust_osd_tex = None
    viewer._brand_osd_tex = None
    viewer._hand_fps_visible = False
    viewer._overlay_tex = None
    viewer._team_fps_visible = False
    viewer._team_status_tex = None
    viewer._calibration_mode = False
    viewer._fps_overlay_visible = False
    viewer._help_tex = None
    viewer._team_status_visible = False
    viewer._team_help_visible = False
    viewer._team_help_tex = None

    presenter = ScreenLayerPresenter(viewer)
    quad_layers, quad_headers, updated, render_projection, background_headers = presenter.prepare_frame_layers(
        screen_frame_uploaded=True
    )

    assert quad_layers == []
    assert quad_headers == []
    assert updated == []
    assert render_projection is True
    assert background_headers == []
    assert presenter._frame_background_projection_fallback is False


def test_screen_layer_presenter_keeps_quad_when_background_layer_build_fails(monkeypatch):
    import ctypes
    from types import SimpleNamespace
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    viewer = _make_default_viewer(monkeypatch)
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._background_layer_renderer = SimpleNamespace(
        make_background_layers=lambda: (_ for _ in ()).throw(RuntimeError("background failed")),
    )
    viewer._update_quad_layer_swapchains = lambda force=False: [0]
    viewer._make_quad_layer = lambda _eye_index: ctypes.c_int(9)
    presenter = ScreenLayerPresenter(viewer)

    quad_layers, quad_headers, updated, render_projection, background_headers = presenter.prepare_frame_layers(
        screen_frame_uploaded=True
    )

    assert quad_layers == []
    assert quad_headers == []
    assert updated == []
    assert render_projection is True
    assert background_headers == []
    assert presenter._frame_background_projection_fallback is True
    assert ("openxr_background_layer_failed", 1) in inc_calls


def test_background_layer_renderer_reuses_panorama_ready_for_native_gate(monkeypatch):
    monkeypatch.chdir(SRC)
    import xr_viewer.background_layer_renderer as layer_module
    from xr_viewer.background_layer_renderer import BackgroundLayerRenderer

    monkeypatch.setattr(layer_module, "xr", type("XR", (), {"CompositionLayerEquirect2KHR": object})())

    class Viewer:
        pass

    viewer = Viewer()
    calls = []
    viewer._panorama_texture_ready = lambda: calls.append("ready") or object()
    viewer._openxr_equirect_background_supported = False
    viewer._background_equirect_swapchain = None
    viewer._background_equirect_size = None
    viewer._breakdown_inc = lambda *_args, **_kwargs: None

    headers, projection_fallback = BackgroundLayerRenderer(viewer).make_background_layers()

    assert headers == []
    assert projection_fallback is True
    assert calls == ["ready"]


def test_background_layer_renderer_prefers_native_layer_before_projection_and_quad(monkeypatch):
    monkeypatch.chdir(SRC)
    import ctypes
    import xr_viewer.background_layer_renderer as layer_module
    from xr_viewer.background_layer_renderer import BackgroundLayerRenderer
    from xr_viewer.screen_layer_presenter import ScreenLayerPresenter

    class _FakeLayer(ctypes.Structure):
        _fields_ = [("dummy", ctypes.c_int)]

        def __init__(self, **kwargs):
            super().__init__(7)
            self.kwargs = kwargs

    class _FakeXR:
        CompositionLayerBaseHeader = ctypes.c_int
        EyeVisibility = type(
            "EyeVisibility",
            (),
            {"BOTH": "both", "LEFT": "left", "RIGHT": "right"},
        )
        Posef = staticmethod(lambda: "pose")
        Offset2Di = staticmethod(lambda **kwargs: kwargs)
        Extent2Di = staticmethod(lambda **kwargs: kwargs)
        Rect2Di = staticmethod(lambda **kwargs: kwargs)
        SwapchainSubImage = staticmethod(lambda **kwargs: kwargs)
        CompositionLayerEquirect2KHR = _FakeLayer

    monkeypatch.setattr(layer_module, "xr", _FakeXR)

    class Viewer:
        pass

    inc_calls = []
    time_calls = []
    value_calls = []
    viewer = Viewer()
    viewer._panorama_texture_ready = lambda: object()
    viewer._openxr_equirect_background_supported = False
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    viewer._breakdown_add_value = lambda name, value: value_calls.append((name, value))

    headers, projection_fallback = BackgroundLayerRenderer(viewer).make_background_layers()

    assert headers == []
    assert projection_fallback is True
    assert ("openxr_background_projection_fallback", 1) in inc_calls

    tex = object()
    viewer._panorama_texture_ready = lambda: tex
    viewer._openxr_equirect_background_supported = True
    viewer._background_equirect_swapchain = "swapchain"
    viewer._background_equirect_size = (1024, 512)
    viewer._xr_space = "space"
    viewer._panorama_render_settings = lambda: (0.0, 1.0, False, 0, (0.5, 0.5), (0.25, 0.25))
    viewer._background_equirect_uploaded_key = None
    viewer._background_equirect_pending_tex = None
    viewer._frame_count = 10
    renderer = BackgroundLayerRenderer(viewer)
    uploads = []

    def _upload(value):
        uploads.append(value)
        viewer._background_equirect_uploaded_key = renderer._source_key(value)

    monkeypatch.setattr(renderer, "_upload_equirect_texture", _upload)
    headers, projection_fallback = renderer.make_background_layers()

    assert headers == []
    assert projection_fallback is True
    assert viewer._background_equirect_pending_tex is tex
    assert uploads == []
    assert renderer.flush_pending_upload_after_submit() is True
    headers, projection_fallback = renderer.make_background_layers()

    assert len(headers) == 1
    assert projection_fallback is False
    assert renderer._frame_background_layers[0].dummy == 7
    assert renderer._frame_background_layers[0].kwargs["sub_image"]["swapchain"] == "swapchain"
    assert renderer._frame_background_layers[0].kwargs["sub_image"]["image_rect"]["extent"] == {
        "width": 1024,
        "height": 512,
    }
    assert uploads == [tex]
    assert any(name == "openxr_background_upload" for name, _seconds in time_calls)
    assert ("openxr_background_layer", 1) in inc_calls
    assert ("openxr_background_safe_age_frames", 0.0) in value_calls

    viewer._frame_count = 12
    headers, projection_fallback = renderer.make_background_layers()
    assert len(headers) == 1
    assert projection_fallback is False
    assert ("openxr_background_reuse", 1) in inc_calls
    assert ("openxr_background_safe_age_frames", 2.0) in value_calls

    fail_tex = type("Tex", (), {"glo": 41, "size": (1024, 512)})()
    viewer._panorama_texture_ready = lambda: fail_tex
    viewer._background_equirect_pending_tex = fail_tex
    monkeypatch.setattr(renderer, "_upload_equirect_texture", lambda _value: (_ for _ in ()).throw(RuntimeError("upload failed")))
    assert renderer.flush_pending_upload_after_submit() is True
    assert viewer._background_equirect_pending_tex is None
    assert ("openxr_background_layer_upload_failed", 1) in inc_calls
    failed_key = renderer._source_key(fail_tex)
    assert viewer._background_equirect_failed_key == failed_key

    headers, projection_fallback = renderer.make_background_layers()

    assert headers == []
    assert projection_fallback is True
    assert viewer._background_equirect_pending_tex is None
    assert ("openxr_background_layer_upload_suppressed", 1) in inc_calls

    next_tex = type("Tex", (), {"glo": 42, "size": (1024, 512)})()
    viewer._panorama_texture_ready = lambda: next_tex
    headers, projection_fallback = renderer.make_background_layers()

    assert headers == []
    assert projection_fallback is True
    assert viewer._background_equirect_pending_tex is next_tex

    viewer._openxr_background_upload_budget_ms = 0.001
    viewer._openxr_background_upload_budget_skip_armed = False
    slow_tex = type("Tex", (), {"glo": 43, "size": (1024, 512)})()
    viewer._background_equirect_pending_tex = slow_tex
    uploads.clear()

    def _slow_upload(value):
        uploads.append(value)
        import time
        time.sleep(0.001)

    monkeypatch.setattr(renderer, "_upload_equirect_texture", _slow_upload)
    assert renderer.flush_pending_upload_after_submit() is True
    assert viewer._openxr_background_upload_budget_skip_armed is True

    skipped_tex = type("Tex", (), {"glo": 44, "size": (1024, 512)})()
    viewer._background_equirect_pending_tex = skipped_tex
    assert renderer.flush_pending_upload_after_submit() is False
    assert viewer._openxr_background_upload_budget_skip_armed is False
    assert uploads == [slow_tex]
    assert ("openxr_background_upload_budget_skip", 1) in inc_calls

    viewer._panorama_texture_ready = lambda: tex
    viewer._background_equirect_pending_tex = None
    viewer._background_equirect_failed_key = None
    viewer._background_equirect_uploaded_key = renderer._source_key(tex)

    viewer._panorama_render_settings = lambda: (0.0, 1.0, False, 1, (0.5, 0.5), (0.25, 0.25))
    renderer = BackgroundLayerRenderer(viewer)
    uploads = []
    monkeypatch.setattr(renderer, "_upload_equirect_texture", _upload)
    headers, projection_fallback = renderer.make_background_layers()

    assert len(headers) == 2
    assert projection_fallback is False
    left, right = renderer._frame_background_layers
    assert left.kwargs["eye_visibility"] == "left"
    assert right.kwargs["eye_visibility"] == "right"
    assert left.kwargs["sub_image"]["image_rect"] == {
        "offset": {"x": 0, "y": 0},
        "extent": {"width": 512, "height": 512},
    }
    assert right.kwargs["sub_image"]["image_rect"] == {
        "offset": {"x": 512, "y": 0},
        "extent": {"width": 512, "height": 512},
    }
    assert uploads == []

    presenter = ScreenLayerPresenter(viewer)
    composition_layers = []
    presenter.append_frame_layers(
        composition_layers,
        projection_views=[],
        quad_layer_headers=["quad"],
        background_layer_headers=["background"],
    )

    assert composition_layers == ["background", "quad"]


def test_background_presenter_skips_background_without_projection_screen_or_panorama(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.background_presenter import BackgroundPresenter

    class Fbo:
        def use(self):
            raise AssertionError("background FBO should not be touched")

    class Viewer:
        pass

    viewer = Viewer()
    viewer._panorama_background_path = None
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    inc_calls = []
    time_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: time_calls.append((name, seconds))
    viewer._render_panorama_background = lambda *_args: pytest.fail("panorama should be skipped")
    viewer._render_env_model = lambda *_args: pytest.fail("env model should be skipped")

    rendered = BackgroundPresenter(viewer).render_projection_background(
        Fbo(),
        object(),
        object(),
        object(),
        eye_index=0,
    )

    assert rendered is False
    assert ("openxr_background_idle", 1) in inc_calls
    assert any(name == "openxr_background" for name, _seconds in time_calls)


def test_background_presenter_uses_frame_background_fallback_without_gate(monkeypatch):
    monkeypatch.chdir(SRC)
    import xr_viewer.background_presenter as background_module
    from xr_viewer.background_presenter import BackgroundPresenter

    monkeypatch.setattr(background_module, "glClear", lambda *_args: None)

    class Fbo:
        def __init__(self):
            self.used = 0

        def use(self):
            self.used += 1

    class Viewer:
        pass

    viewer = Viewer()
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: None
    viewer._render_panorama_background = lambda *_args: True
    presenter = BackgroundPresenter(viewer)
    presenter.projection_fallback_needed = lambda: pytest.fail("frame fallback result should be reused")
    fbo = Fbo()

    rendered = presenter.render_projection_background(
        fbo,
        object(),
        object(),
        object(),
        eye_index=0,
        projection_fallback_needed=True,
    )

    assert rendered is True
    assert fbo.used == 1
    assert ("openxr_background_panorama", 1) in inc_calls


def test_background_presenter_keeps_panorama_projection_fallback_without_screen(monkeypatch):
    monkeypatch.chdir(SRC)
    import xr_viewer.background_presenter as background_module
    from xr_viewer.background_presenter import BackgroundPresenter

    monkeypatch.setattr(background_module, "glClear", lambda *_args: None)

    class Fbo:
        def __init__(self):
            self.used = 0

        def use(self):
            self.used += 1

    class Viewer:
        pass

    viewer = Viewer()
    viewer._panorama_background_path = "room.hdr"
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: None
    viewer._panorama_texture_ready = lambda: object()
    viewer._render_panorama_background = lambda *_args: True
    viewer._render_env_model = lambda *_args: pytest.fail("env model should be skipped behind panorama")
    fbo = Fbo()

    rendered = BackgroundPresenter(viewer).render_projection_background(
        fbo,
        object(),
        object(),
        object(),
        eye_index=0,
    )

    assert rendered is True
    assert fbo.used == 1
    assert ("openxr_background_panorama", 1) in inc_calls


def test_background_presenter_skips_configured_panorama_until_texture_ready(monkeypatch):
    monkeypatch.chdir(SRC)
    from xr_viewer.background_presenter import BackgroundPresenter

    class Fbo:
        def use(self):
            raise AssertionError("background FBO should not be touched")

    class Viewer:
        pass

    viewer = Viewer()
    viewer._panorama_background_path = "room.hdr"
    viewer._panorama_texture_ready = lambda: None
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._breakdown_add_time = lambda name, seconds: None
    viewer._render_panorama_background = lambda *_args: pytest.fail("panorama should wait until ready")
    viewer._render_env_model = lambda *_args: pytest.fail("configured panorama should not fall back to GLB mesh")

    rendered = BackgroundPresenter(viewer).render_projection_background(
        Fbo(),
        object(),
        object(),
        object(),
        eye_index=0,
    )

    assert rendered is False
    assert ("openxr_background_idle", 1) in inc_calls


def test_quad_layer_build_failure_happens_before_projection_fallback_render():
    frame_renderer = (SRC / "xr_viewer" / "openxr_frame_renderer.py").read_text(encoding="utf-8")

    quad_build_idx = frame_renderer.index("self.screen_presenter.prepare_frame_layers(")
    render_idx = frame_renderer.index("self.projection_presenter.render_projection(", quad_build_idx)
    append_idx = frame_renderer.index("self.screen_presenter.append_frame_layers(", render_idx)
    presenter_text = (SRC / "xr_viewer" / "screen_layer_presenter.py").read_text(encoding="utf-8")

    assert "viewer._xr_quad_layer_failed = True" in presenter_text
    assert quad_build_idx < render_idx < append_idx


def test_env_model_render_failure_restores_gl_state():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    render_func = render_text.split("def _render_env_model", 1)[1]
    render_body = render_func.split("        if self._env_perf_log:", 1)[0]

    assert "previous_depth_mask = get_depth_mask()" in render_body
    assert "try:" in render_body
    assert "except Exception as exc:" in render_body
    assert "openxr_background_env_model_failed" in render_body
    assert "finally:" in render_body
    render_finally = render_body.split("finally:", 1)[1]
    assert "self.ctx.disable(moderngl.CULL_FACE)" in render_finally
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally
    assert "set_depth_mask(previous_depth_mask)" in render_finally
    assert "glFrontFace(GL_CCW)" in render_finally
    assert "self._env_prog['u_use_texture'].value = 1" in render_finally
    assert "self._env_prog['u_base_color_factor'].value = (1.0, 1.0, 1.0)" in render_finally
    assert "self._env_prog['u_base_alpha'].value = 1.0" in render_finally


def test_env_model_initializer_preloads_panorama_light_mask(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._env_model_init_done = False
    viewer._environment_enabled = True
    viewer._panorama_background_path = "background.jpg"
    viewer._env_model_path = None
    viewer._panorama_called = False
    viewer._mask_called = False

    def _get_panorama_texture():
        viewer._panorama_called = True
        return object()

    def _get_panorama_light_mask_texture():
        viewer._mask_called = True
        return object()

    viewer._get_panorama_texture = _get_panorama_texture
    viewer._get_panorama_light_mask_texture = _get_panorama_light_mask_texture
    viewer._ensure_env_model_initialized("Test")

    assert viewer._panorama_called
    assert viewer._mask_called


def test_panorama_wall_light_mask_missing_is_diagnostic(monkeypatch, tmp_path):
    viewer = _make_default_viewer(monkeypatch)
    viewer._panorama_background_path = str(tmp_path / "background.jpg")
    viewer._panorama_background_settings = {"wall_light_mask": "missing-mask.png"}
    viewer._panorama_light_mask_tex = None
    viewer._panorama_light_mask_path = None
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._get_panorama_light_mask_texture() is None
    assert viewer._get_panorama_light_mask_texture() is None

    assert inc_calls == [("openxr_wall_light_mask_missing", 1)]
    assert viewer._panorama_light_mask_missing_path.endswith("missing-mask.png")


def test_panorama_wall_light_mask_disabled_is_diagnostic_once(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._panorama_background_path = "background.jpg"
    viewer._panorama_background_settings = {}
    viewer._panorama_light_mask_tex = object()
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._get_panorama_light_mask_texture() is None
    assert viewer._get_panorama_light_mask_texture() is None
    assert viewer._panorama_light_mask_texture_ready() is None
    assert viewer._panorama_light_mask_texture_ready() is None

    assert inc_calls == [("openxr_wall_light_mask_disabled", 1)]


def test_panorama_wall_light_uses_gpu_light_probe_grid():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    pano_frag = glsl_text.split("_PANORAMA_FRAG", 1)[1].split("_GLOW_DOWNSAMPLE_FRAG", 1)[0]
    probe_func = pano_frag.split("vec3 screen_light_probe_color()", 1)[1].split("void main()", 1)[0]

    assert probe_func.count("textureLod(u_screen_light_tex") == 9
    assert "vec2(0.25, 0.25)" in probe_func
    assert "vec2(0.50, 0.50)" in probe_func
    assert "vec2(0.75, 0.75)" in probe_func
    assert "screen_light_probe_color()" in pano_frag.split("void main()", 1)[1]
    assert ".cpu(" not in pano_frag
    assert ".numpy(" not in pano_frag
    assert "glReadPixels" not in pano_frag


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
    viewer._render_glow_shell_call = None

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_glow_shell(*args, **kwargs):
        viewer._render_glow_shell_call = (args, kwargs)

    viewer._render_glow = _render_glow
    viewer._render_glow_shell = _render_glow_shell
    viewer._render_screen_background_effects(None, None)

    assert not viewer._default_blank_fast_path()
    assert not viewer._render_glow_called
    assert viewer._render_glow_shell_call is not None
    assert viewer._render_glow_shell_call[1]["intensity_multiplier"] == 0.72


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


def test_environment_screen_effects_wait_for_source_ready(monkeypatch):
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
    viewer._runtime_direct_source = False
    viewer._should_show_source_border = lambda: False
    viewer._render_glow_shell_called = False
    viewer._render_frosted_glow_called = False

    def _render_glow_shell(*_args):
        viewer._render_glow_shell_called = True

    def _render_frosted_glow(*_args):
        viewer._render_frosted_glow_called = True

    viewer._render_glow_shell = _render_glow_shell
    viewer._render_frosted_glow = _render_frosted_glow
    viewer._render_screen_background_effects(None, None)
    viewer._glow_mode = "frosted"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0
    viewer._render_screen_foreground_effects(None, None)

    assert not viewer._render_glow_shell_called
    assert not viewer._render_frosted_glow_called


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
    viewer._render_glow_shell_called = False
    viewer._render_env_model_called = False

    def _render_glow(*_args):
        viewer._render_glow_called = True

    def _render_glow_shell(*_args, **_kwargs):
        viewer._render_glow_shell_called = True

    def _render_env_model(*_args):
        viewer._render_env_model_called = True

    viewer._render_glow = _render_glow
    viewer._render_glow_shell = _render_glow_shell
    viewer._render_env_model = _render_env_model
    viewer._render_screen_background_effects(None, None)

    assert not viewer._render_glow_called
    assert viewer._render_glow_shell_called
    assert not viewer._render_env_model_called


def test_default_glow_does_not_skip_curved_screen():
    env_text = (SRC / "xr_viewer" / "environment.py").read_text(encoding="utf-8")

    assert "if getattr(self, '_screen_curved', False):\n            return" not in env_text


def test_default_glow_mode_cycle_from_y(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._environment_model = "Default"
    viewer._active_environment = None
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 0.0
    viewer._glow_shell_intensity_multiplier = 1.85
    viewer._env_profile = {
        "glow_intensity_multiplier": 0.0,
        "glow_shell_intensity_multiplier": 0.0,
        "lighting_presets": [
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
        ],
    }
    viewer._save_glow_to_builtin_profile = lambda: None

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

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "frosted"
    assert viewer._glow_intensity_multiplier == 1.85
    assert viewer._glow_shell_intensity_multiplier == 0.0

    assert viewer._cycle_glow_mode_from_y()
    assert viewer._glow_mode == "surround"
    assert viewer._glow_intensity_multiplier == 0.0
    assert viewer._glow_shell_intensity_multiplier == 1.85


def test_frosted_glow_shader_uses_flat_grid_source_crop():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    assert "_FROSTED_VEIL_VERT" in glsl_text
    assert "_FROSTED_GLOW_FRAG" in glsl_text
    assert "in vec2 v_uv" in glsl_text
    assert "in vec3 v_local" in glsl_text
    assert "uniform mat4 u_model" in glsl_text
    assert "uniform sampler2D u_screen_tex" in glsl_text
    assert "uniform vec4 u_source_crop" in glsl_text
    assert "u_source_crop.xy + v_uv * u_source_crop.zw" in glsl_text
    assert "textureLod(u_screen_tex" in glsl_text
    assert "smoothstep(u_threshold, 1.0, luma)" in glsl_text
    assert "u_beam_softness" in glsl_text
    assert "u_frost_blend" in glsl_text
    assert "u_beam_thickness" in glsl_text
    assert "u_edge_inset" in glsl_text
    assert "u_diffuse_scatter" in glsl_text
    assert "float depth = clamp(v_local.z, 0.0, 1.0)" in glsl_text
    assert "wall_endpoint_fade(v_uv, v_local)" in glsl_text
    assert "u_debug_wall" not in glsl_text
    assert "hash12" in glsl_text


def test_screen_effects_do_not_sample_runtime_eye_texture():
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    source_func = effects_text.split("def _screen_effect_source_texture", 1)[1].split("def _render_glow", 1)[0]
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    no_room_glow = base_text.split("def _render_glow", 1)[1].split("def _render_shadow", 1)[0]

    assert "def _screen_effect_source_texture(self):" in effects_text
    assert "allow_runtime_eye" not in effects_text
    assert "_runtime_effect_submit_scheduler().latest_safe_glow()" in source_func
    assert "_promote_runtime_effect_ready_texture" not in source_func
    assert "_runtime_eye_textures" not in source_func
    assert "_current_eye_index" not in source_func
    assert "def _screen_effect_source_texture(self):" in base_text
    assert "_runtime_effect_submit_scheduler().latest_safe_glow()" in base_text
    assert "_runtime_effect_latest_safe()" not in base_text.split("def _screen_effect_source_texture", 1)[1].split("def _render_screen_background_effects", 1)[0]
    assert "_promote_runtime_effect_ready_texture" not in base_text
    assert "_screen_effect_source_texture()" in no_room_glow
    assert "glow_tex = self._cached_glow_downsample_texture(source_tex, source_size)" in no_room_glow
    assert "_prepare_glow_downsample_texture" not in no_room_glow
    assert "getattr(self, 'color_tex', None)" not in no_room_glow


def test_screen_effect_source_texture_is_cached_per_frame(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 11})()
    viewer._frame_count = 8
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, source_tex, (1280, 720), 4)
    viewer._age_count = 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    def _record_age(_source_tex, _frame_id=None):
        viewer._age_count += 1

    viewer._record_screen_effect_safe_age = _record_age

    assert viewer._screen_effect_source_texture() == (source_tex, (1280, 720))
    assert viewer._screen_effect_source_texture() == (source_tex, (1280, 720))
    assert viewer._age_count == 1
    assert ("openxr_screen_effect_source_reuse", 1) in inc_calls


def test_screen_effect_source_cache_refreshes_when_safe_source_changes(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    first_tex = type("Tex", (), {"glo": 21})()
    next_tex = type("Tex", (), {"glo": 22})()
    viewer._frame_count = 8
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, first_tex, (1280, 720), 4)
    viewer._age_count = 0
    viewer._breakdown_inc = lambda *args, **kwargs: None

    def _record_age(_source_tex, _frame_id=None):
        viewer._age_count += 1

    viewer._record_screen_effect_safe_age = _record_age

    assert viewer._screen_effect_source_texture() == (first_tex, (1280, 720))
    _set_runtime_effect_safe(viewer, next_tex, (1280, 720), 5)
    assert viewer._screen_effect_source_texture() == (next_tex, (1280, 720))

    assert viewer._age_count == 2


def test_no_room_screen_effect_source_texture_uses_safe_runtime_source(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 14})()
    viewer._frame_count = 6
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, source_tex, (640, 360), 2)
    viewer._age_count = 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    def _record_age(_source_tex, _frame_id=None):
        viewer._age_count += 1

    viewer._record_screen_effect_safe_age = _record_age

    assert viewer._screen_effect_source_texture() == (source_tex, (640, 360))
    assert viewer._screen_effect_source_texture() == (source_tex, (640, 360))
    assert viewer._age_count == 1
    assert ("openxr_screen_effect_source_reuse", 1) in inc_calls


def test_no_room_screen_effect_source_texture_caches_color_source(monkeypatch):
    viewer = _make_no_room_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 16})()
    viewer._frame_count = 7
    viewer._runtime_direct_source = False
    viewer.color_tex = source_tex
    viewer._texture_size = (320, 180)
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))

    assert viewer._screen_effect_source_texture() == (source_tex, (320, 180))
    assert viewer._screen_effect_source_texture() == (source_tex, (320, 180))

    assert ("openxr_screen_effect_source_reuse", 1) in inc_calls


def test_screen_effect_safe_age_records_once_per_safe_texture_per_frame(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 12})()
    other_tex = type("Tex", (), {"glo": 13})()
    values = []
    viewer._frame_count = 9
    _set_runtime_effect_safe(viewer, source_tex, None, 5)
    viewer._breakdown_add_value = lambda name, value: values.append((name, value))

    viewer._record_screen_effect_safe_age(source_tex)
    viewer._record_screen_effect_safe_age(source_tex)
    viewer._record_screen_effect_safe_age(other_tex)

    assert values == [
        ("openxr_effect_ready_age_frames", 4.0),
        ("openxr_effect_ready_age_frames", 4.0),
    ]


def test_glow_downsample_cache_is_shared_across_eyes():
    quality_text = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    key_func = quality_text.split("def _glow_downsample_key_and_size", 1)[1].split(
        "def _cached_glow_downsample_texture", 1
    )[0]
    cached_func = quality_text.split("def _cached_glow_downsample_texture", 1)[1].split(
        "def _prepare_glow_downsample_texture", 1
    )[0]
    prepare_func = quality_text.split("def _prepare_glow_downsample_texture", 1)[1].split(
        "def _is_runtime_eye_texture_ready", 1
    )[0]
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    shell_func = effects_text.split("def _render_glow_shell", 1)[1].split(
        "def _render_screen_background_effects", 1
    )[0]

    assert "_glow_ds_cache_key" in cached_func
    assert "_current_eye_index" not in key_func + cached_func + prepare_func
    assert "_runtime_effect_submit_scheduler" in key_func
    assert "scheduler_factory().latest_safe()" in key_func
    assert "source_frame_id if source_frame_id is not None else getattr(self, '_frame_count', 0)" in key_func
    assert "_cached_glow_downsample_texture(source_tex, source_size, target_size=target_size)" in prepare_func
    assert "if getattr(self, '_runtime_direct_source', False):" in shell_func
    assert "latest_safe_downsample(" in shell_func
    runtime_direct_shell = shell_func.split("if getattr(self, '_runtime_direct_source', False):", 1)[1].split(
        "else:", 1
    )[0]
    assert "_cached_glow_downsample_texture" not in runtime_direct_shell
    assert "glow_tex = self._cached_glow_downsample_texture(source_tex, source_size)" in shell_func
    assert "_prepare_glow_downsample_texture" not in shell_func
    assert "except Exception as exc:" in prepare_func
    assert "openxr_glow_downsample_failed" in prepare_func
    assert "finally:" in prepare_func
    render_finally = prepare_func.split("finally:", 1)[1]
    assert "self.ctx.viewport = prev_viewport" in render_finally
    assert "set_depth_mask(prev_depth_mask)" in render_finally
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally


def test_screen_quality_pass_restores_gl_state_on_failure():
    quality_text = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    func = quality_text.split("def _prepare_screen_quality_texture", 1)[1].split(
        "def _ensure_glow_downsample_resources", 1
    )[0]

    assert "try:" in func
    assert "except Exception as exc:" in func
    assert "openxr_screen_quality_failed" in func
    assert "return None" in func
    assert "finally:" in func
    render_finally = func.split("finally:", 1)[1]
    assert "self.ctx.viewport = prev_viewport" in render_finally
    assert "set_depth_mask(prev_depth_mask)" in render_finally
    assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally


def test_screen_light_uses_effect_source_texture_not_runtime_eye_texture():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    source_func = render_text.split("def _screen_light_source_texture", 1)[1].split("def _apply_cinema_light_uniforms", 1)[0]
    runtime_direct_block = source_func.split("if getattr(self, '_runtime_direct_source', False):", 1)[1].split(
        "source_tex = getattr(self, 'color_tex'", 1
    )[0]

    assert "scheduler.latest_safe_light_probe()" in source_func
    assert "_promote_runtime_effect_ready_texture" not in source_func
    assert "_record_screen_effect_safe_age" in source_func
    assert "latest_safe_downsample(" not in runtime_direct_block
    assert "_prepare_glow_downsample_texture" not in source_func
    assert "_cached_glow_downsample_texture" not in runtime_direct_block
    assert "value = (None, None)" in source_func
    assert "_runtime_eye_textures" not in source_func
    assert "_current_eye_index" not in source_func


def test_screen_light_source_texture_reuses_prewarmed_downsample(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    source_tex = type("Tex", (), {"glo": 7})()
    light_tex = object()
    viewer._frame_count = 12
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, source_tex, (1920, 1080), 3)
    viewer._runtime_effect_submit_scheduler().publish_light_probe(light_tex, (96, 54), 3)
    viewer._age_count = 0
    viewer._prepare_count = 0
    inc_calls = []
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._record_screen_effect_safe_age = lambda _source_tex, _frame_id=None: setattr(viewer, "_age_count", viewer._age_count + 1)

    def _prepare(_source_tex, _source_size):
        viewer._prepare_count += 1
        return object()

    viewer._prepare_glow_downsample_texture = _prepare

    assert viewer._screen_light_source_texture() == (light_tex, (96, 54))
    assert viewer._screen_light_source_texture() == (light_tex, (96, 54))
    assert viewer._age_count == 1
    assert viewer._prepare_count == 0
    assert ("openxr_screen_light_downsample_source", 1) in inc_calls
    assert ("openxr_screen_light_source_reuse", 1) in inc_calls


def test_screen_light_source_waits_for_prewarmed_downsample_when_safe_source_changes(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    first_tex = type("Tex", (), {"glo": 31})()
    next_tex = type("Tex", (), {"glo": 32})()
    viewer._frame_count = 12
    viewer._runtime_direct_source = True
    _set_runtime_effect_safe(viewer, first_tex, (1920, 1080), 3)
    viewer._age_count = 0
    viewer._prepare_count = 0
    viewer._breakdown_inc = lambda *args, **kwargs: None
    viewer._record_screen_effect_safe_age = lambda _source_tex, _frame_id=None: setattr(viewer, "_age_count", viewer._age_count + 1)

    def _prepare(_source_tex, _source_size):
        viewer._prepare_count += 1
        return object()

    viewer._prepare_glow_downsample_texture = _prepare

    assert viewer._screen_light_source_texture() == (None, None)
    _set_runtime_effect_safe(viewer, next_tex, (1920, 1080), 4)
    assert viewer._screen_light_source_texture() == (None, None)

    assert viewer._age_count == 2
    assert viewer._prepare_count == 0


def test_runtime_effect_source_texture_is_prepared_for_all_glow_modes(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    cases = [
        ("screen", 1.0, 0.0),
        ("surround", 0.0, 1.0),
        ("veil", 1.0, 0.0),
        ("frosted", 1.0, 0.0),
    ]

    for mode, glow_mult, shell_mult in cases:
        viewer._glow_mode = mode
        viewer._glow_intensity_multiplier = glow_mult
        viewer._glow_shell_intensity_multiplier = shell_mult
        assert viewer._runtime_effects_need_source_texture()

    viewer._glow_mode = "off"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 1.0

    assert not viewer._runtime_effects_need_source_texture()

    viewer._screen_light_intensity = 3.5
    viewer._panorama_background_path = "background.jpg"

    assert viewer._runtime_effects_need_source_texture()

    viewer._panorama_background_path = None
    viewer._env_model_visible = True
    viewer._env_model_prims = [object()]

    assert viewer._runtime_effects_need_source_texture()

    viewer._openxr_async_effects_enabled = False
    viewer._glow_mode = "screen"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 0.0

    assert not viewer._runtime_effects_need_source_texture()


def test_openxr_full_synthesis_preserves_effect_source_before_eye_sampling():
    runtime_text = (SRC / "stereo_runtime" / "runtime.py").read_text(encoding="utf-8")
    pipeline_text = (SRC / "stereo_runtime" / "pipeline.py").read_text(encoding="utf-8")
    core_text = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    scheduler_text = (SRC / "xr_viewer" / "effect_scheduler.py").read_text(encoding="utf-8")
    uploader_text = (SRC / "viewer" / "gl_texture_uploader.py").read_text(encoding="utf-8")

    assert "source_rgb: torch.Tensor | None = None" in runtime_text
    assert "source_rgb=source_rgb" in runtime_text
    assert "openxr_result_from_stereo_result(runtime_result, source_rgb=runtime_rgb)" in pipeline_text
    assert "def _try_update_runtime_effect_source_texture_gpu" in core_text
    assert "CudaGlTextureUploader" in core_text
    assert "ensure_staging(self.ctx, w, h)" in core_text
    assert "slot.tex = ctx.texture(size, 4, dtype='f1')" in scheduler_text
    gpu_upload_func = core_text.split("def _try_update_runtime_effect_source_texture_gpu", 1)[1].split("def _update_runtime_effect_source_texture", 1)[0]
    assert "torch.cuda.current_stream(device_index).synchronize()" not in gpu_upload_func
    assert "ready_event.synchronize()" not in core_text
    assert gpu_upload_func.index("source_rgba =") < gpu_upload_func.index("upload_path = uploader.upload_rgba")
    update_func = core_text.split("def _update_runtime_effect_source_texture", 1)[1].split("def _release_runtime_eye_texture_resources", 1)[0]
    assert "_try_update_runtime_effect_source_texture_gpu" in update_func
    assert "_runtime_eye_to_numpy" not in update_func
    assert ".write(rgba.tobytes())" not in update_func
    assert "self._release_runtime_effect_source_texture()" in update_func
    image_upload = uploader_text.split("def _upload_image", 1)[1].split("def _ensure_pbos", 1)[0]
    assert "glFlush()" not in image_upload
    assert "_screen_effect_source_texture()" in effects_text
    assert "allow_runtime_eye" not in effects_text


def test_screen_glow_shader_uses_region_color_grid():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")

    assert "_GLOW_DOWNSAMPLE_FRAG" in glsl_text
    assert "uniform sampler2D u_glow_tex" in glsl_text
    assert "uniform sampler2D u_screen_light_tex" in glsl_text
    assert "textureLod(u_glow_tex" in glsl_text
    assert "textureLod(u_screen_light_tex" in glsl_text
    assert "glow_grid_color" in glsl_text
    assert "def _screen_effect_source_texture" in effects_text
    assert "_runtime_effect_submit_scheduler().latest_safe()" in effects_text
    assert "_runtime_effect_latest_safe" not in effects_text


def test_no_room_glow_pass_restores_gl_state_on_failure():
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    func = base_text.split("def _render_glow", 1)[1].split("def _render_shadow", 1)[0]

    assert "previous_depth_mask = get_depth_mask()" in func
    assert "try:" in func
    assert "except Exception as exc:" in func
    assert "openxr_screen_glow_failed" in func
    assert "finally:" in func
    render_finally = func.split("finally:", 1)[1]
    assert "self.ctx.disable(moderngl.BLEND)" in render_finally
    assert "set_depth_mask(previous_depth_mask)" in render_finally


def test_screen_effect_entrypoints_keep_failures_soft(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    inc_calls = []
    viewer._should_render_source_screen_effects = lambda: True
    viewer._default_blank_fast_path = lambda: False
    viewer._glow_mode = "surround"
    viewer._glow_intensity_multiplier = 1.0
    viewer._glow_shell_intensity_multiplier = 1.0
    viewer._env_model_visible = False
    viewer._env_model_prims = []
    viewer._bg_color_idx = 0
    viewer._breakdown_inc = lambda name, amount=1: inc_calls.append((name, amount))
    viewer._render_glow_shell = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bg failed"))

    viewer._render_screen_background_effects(None, None)

    viewer._glow_mode = "veil"
    viewer._render_frosted_veil = lambda *_args: (_ for _ in ()).throw(RuntimeError("fg failed"))
    viewer._render_screen_foreground_effects(None, None)

    assert ("openxr_screen_background_effect_failed", 1) in inc_calls
    assert ("openxr_screen_foreground_effect_failed", 1) in inc_calls


def test_screen_effect_passes_restore_gl_state_on_failure():
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    checks = [
        ("def _render_glow", "def _render_frosted_glow", "openxr_screen_glow_failed"),
        ("def _render_frosted_glow", "def _render_frosted_veil", "openxr_frosted_glow_failed"),
        ("def _render_frosted_veil", "def _render_glow_shell", "openxr_frosted_veil_failed"),
        ("def _render_glow_shell", "def _render_screen_background_effects", "openxr_glow_shell_failed"),
    ]

    for start, end, diagnostic in checks:
        func = effects_text.split(start, 1)[1].split(end, 1)[0]
        assert "previous_depth_mask = get_depth_mask()" in func
        assert "try:" in func
        assert "except Exception as exc:" in func
        assert diagnostic in func
        assert "finally:" in func
        render_finally = func.split("finally:", 1)[1]
        assert "self.ctx.disable(moderngl.BLEND)" in render_finally
        assert "set_depth_mask(previous_depth_mask)" in render_finally
        assert "self.ctx.enable(moderngl.DEPTH_TEST)" in render_finally


def test_surround_glow_shell_uses_screen_border_color():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")

    assert "_GLOW_SHELL_FRAG" in glsl_text
    assert "uniform sampler2D u_glow_tex" in glsl_text
    assert "uniform int u_glow_use_tex" in glsl_text
    assert "sample_border_color" in glsl_text
    assert "sample_region_reflection" in glsl_text
    assert "vec2 grid = vec2(4.0, 3.0)" in glsl_text
    shell_frag = glsl_text.split("_GLOW_SHELL_FRAG", 1)[1]
    assert "top_col" in shell_frag
    assert "bottom_col" in shell_frag
    assert "left_col" in shell_frag
    assert "right_col" in shell_frag
    assert "vertical_edges" in shell_frag
    assert "textureLod(u_glow_tex, q, 0.0)" in shell_frag
    assert "region_mix" in shell_frag
    assert "edge_band_depth" not in shell_frag
    assert "for (int" not in shell_frag


def test_realtime_screen_glow_cpu_sampler_is_removed():
    upload_text = (SRC / "xr_viewer" / "core_frame_upload.py").read_text(encoding="utf-8")
    runtime_eye_text = (SRC / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")
    effects_text = (SRC / "xr_viewer" / "environment_effects.py").read_text(encoding="utf-8")
    renderer_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    screen_quality_text = (SRC / "xr_viewer" / "core_screen_quality.py").read_text(encoding="utf-8")
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    base_text = (SRC / "xr_viewer" / "base.py").read_text(encoding="utf-8")
    source_func = effects_text.split("def _screen_effect_source_texture", 1)[1].split("def _render_glow", 1)[0]
    light_func = renderer_text.split("def _screen_light_source_texture", 1)[1].split(
        "def _apply_cinema_light_uniforms", 1
    )[0]
    downsample_func = screen_quality_text.split("def _prepare_glow_downsample_texture", 1)[1].split(
        "def ", 1
    )[0]

    assert not (SRC / "xr_viewer" / "core_glow.py").exists()
    assert "def _sample_glow_target_color" not in upload_text
    assert "_maybe_sample_glow_target_color" not in upload_text
    assert "_maybe_sample_glow_target_color" not in runtime_eye_text
    assert "_mark_upload('sample_glow')" not in upload_text
    assert "_advance_glow_color" not in effects_text
    assert "_glow_target_color" not in effects_text
    assert "_glow_target_color" not in impl_text
    assert "_glow_target_color" not in base_text
    assert "_screen_light_target_colors" not in effects_text
    assert "_screen_light_target_colors" not in impl_text
    assert "_screen_light_colors" not in effects_text
    assert "_screen_light_colors" not in impl_text
    assert "_glow_color_counter" not in impl_text
    for realtime_func in (source_func, light_func, downsample_func):
        assert ".cpu(" not in realtime_func
        assert ".numpy(" not in realtime_func
        assert "glReadPixels" not in realtime_func
        assert ".read(" not in realtime_func
        assert "synchronize(" not in realtime_func


def test_frosted_glow_keyboard_adjustment_is_disabled(monkeypatch):
    import glfw

    viewer = _make_default_viewer(monkeypatch)
    viewer._frosted_glow_blend = 2.40
    viewer._frosted_glow_thickness = 2.40
    viewer._preset_name_overlay = ""
    viewer._preset_osd_show_t = 0.0

    assert not viewer._adjust_frosted_glow_keyboard(glfw.KEY_RIGHT)
    assert not viewer._adjust_frosted_glow_keyboard(glfw.KEY_UP, glfw.MOD_SHIFT)
    assert viewer._frosted_glow_blend == 2.40
    assert viewer._frosted_glow_thickness == 2.40
    assert viewer._preset_name_overlay == ""


def test_frosted_glow_virtual_keyboard_adjustment_is_disabled(monkeypatch):
    viewer = _make_default_viewer(monkeypatch)
    viewer._frosted_glow_blend = 2.40
    viewer._frosted_glow_thickness = 2.40
    viewer._preset_name_overlay = ""
    viewer._preset_osd_show_t = 0.0

    assert not viewer._adjust_frosted_glow_vk(0x27)
    assert not viewer._adjust_frosted_glow_vk(0x28)
    assert viewer._frosted_glow_blend == 2.40
    assert viewer._frosted_glow_thickness == 2.40
    assert viewer._preset_name_overlay == ""


def test_controller_shader_uses_panorama_ibl_reflection():
    glsl_text = (SRC / "xr_viewer" / "glsl.py").read_text(encoding="utf-8")
    render_text = (SRC / "xr_viewer" / "core_laser_render.py").read_text(encoding="utf-8")
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    profile_text = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")
    ctrl_frag = glsl_text.split("_CTRL_FRAG", 1)[1].split("_ENV_VERT", 1)[0]

    assert "uniform sampler2D u_env_tex" in ctrl_frag
    assert "uniform sampler2D u_screen_light_tex" in ctrl_frag
    assert "uniform int u_env_stereo_layout" in ctrl_frag
    assert "uniform int u_env_eye_index" in ctrl_frag
    assert "vec2 env_sample_uv(vec3 dir)" in ctrl_frag
    assert "uv.x = uv.x * 0.5 + (u_env_eye_index == 1 ? 0.5 : 0.0)" in ctrl_frag
    assert "textureLod(u_env_tex, env_sample_uv(R), 3.0)" in ctrl_frag
    assert "textureLod(u_env_tex, env_sample_uv(N), 5.0)" in ctrl_frag
    assert "textureLod(u_screen_light_tex" in ctrl_frag
    assert "u_light_color" not in ctrl_frag
    assert "u_ambient_color" not in ctrl_frag
    assert "_get_panorama_texture" not in render_text
    assert "_bind_screen_light_source_texture(location=10)" in render_text
    assert "_runtime_eye_textures" not in render_text
    assert "u_screen_light_tex" in impl_text
    assert "controller_hdr_lighting" in profile_text
    assert "controller_hdr_reflection" in profile_text
    assert "== '.hdr'" in profile_text


def test_environment_light_binds_latest_safe_texture_before_uniform_cache_skip():
    render_text = (SRC / "xr_viewer" / "environment_renderer.py").read_text(encoding="utf-8")
    impl_text = (SRC / "xr_viewer" / "implementation.py").read_text(encoding="utf-8")
    profiles_text = (SRC / "xr_viewer" / "environment_profiles.py").read_text(encoding="utf-8")
    apply_func = render_text.split("def _apply_cinema_light_uniforms", 1)[1].split("def _get_panorama_texture", 1)[0]
    before_skip = apply_func.split("if state_key == last_state_key", 1)[0]

    assert not (SRC / "xr_viewer" / "render.py").exists()
    assert "from .render import" not in render_text
    assert "def _view_mat_inv" in render_text
    assert "def _bind_screen_light_source_texture" in render_text
    assert "self._bind_screen_light_source_texture()" in before_skip
    assert "source_tex.use(location=8)" not in apply_func
    assert "_screen_light_dynamic" not in render_text
    assert "_screen_light_dynamic" not in impl_text
    assert "_screen_light_dynamic" not in profiles_text
    assert "_screen_light_sample_interval" not in impl_text
    assert "_screen_light_sample_interval" not in profiles_text
    assert "screen_light_lerp" not in render_text
    assert "screen_light_lerp" not in impl_text
    assert "screen_light_lerp" not in profiles_text


def test_controller_touch_bindings_share_profile_suggestion_call():
    source = (SRC / "xr_viewer" / "core_controller_actions.py").read_text(encoding="utf-8")

    assert source.count("xr.suggest_interaction_profile_bindings") == 1
    assert '"/user/hand/left/input/thumbstick/touch"' in source
    assert '"/user/hand/left/input/grip/pose"' in source
