from pathlib import Path

import pytest

from xr_viewer.openxr_panda_frame_state import _eye_fovs, update_panda_frame_state_from_viewer
from xr_viewer.projection_layer_presenter import ProjectionLayerPresenter
from xr_viewer.panda_runtime import scene as panda_scene_module
from xr_viewer.panda_runtime.bridge import (
    PandaBridge,
    PandaBridgeUnavailable,
    RenderEyesResult,
    SwapchainImageRef,
    SwapchainResourceKey,
)
from xr_viewer.panda_runtime.controller_ray import (
    PandaControllerRayGeometryError,
    PandaControllerRayGeometryTarget,
)
from xr_viewer.panda_runtime.frame_source import (
    PandaFrameSourceInput,
    build_panda_frame_state,
    controller_ray_from_vectors,
    mat4_to_panda_pose,
)
from xr_viewer.panda_runtime.nv_dx_bridge import PandaNvDxBridge
from xr_viewer.panda_runtime.opengl_bridge import PandaOpenGLBridge
from xr_viewer.panda_runtime.scene_bindings import sync_panda_scene_assets_from_viewer
from xr_viewer.panda_runtime.screen_node import create_panda_screen_node_target
from xr_viewer.panda_runtime.runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaAnimationClock,
    PandaControllerRay,
    PandaEyeView,
    PandaFrameState,
    PandaPose,
    PandaRuntimeUnavailable,
    PandaSceneRenderer,
    PandaScreenTextureFrame,
    log_renderer_selection,
    resolve_gltf_renderer_mode,
    validate_frame_state,
)
from xr_viewer.panda_runtime.scene import PandaSceneGraph
from xr_viewer.panda_runtime.screen_texture import (
    PandaScreenTextureUploadError,
    PandaScreenTextureUploadTarget,
)
from xr_viewer.panda_runtime.stereo_targets import StereoTargetSpec, StereoTargets


ROOT = Path(__file__).resolve().parents[1]
ARTEMIS = ROOT / "src" / "xr_viewer" / "environments" / "Artemis" / "environment.glb"


class _RecordingBridge(PandaBridge):
    bridge_mode = "test"

    def __init__(self):
        self.calls = []
        self.released = False

    def render_eyes(self, *, scene, targets, frame_state, left_image, right_image):
        self.calls.append((scene, targets, frame_state, left_image, right_image))
        return RenderEyesResult(True, True, self.bridge_mode)

    def release(self):
        self.released = True


class _FakePoseRoot:
    def __init__(self):
        self.pos_quat = None

    def set_pos_quat(self, pos, quat):
        self.pos_quat = (pos, quat)


class _FakeScreenTextureTarget:
    def __init__(self):
        self.screen_texture = None

    def set_screen_texture(self, screen_texture):
        self.screen_texture = screen_texture


class _FakeControllerRayTarget:
    def __init__(self):
        self.controller_ray = None

    def set_controller_ray(self, controller_ray):
        self.controller_ray = controller_ray


class _FakeAnimationRuntime:
    channel_count = 38
    bound_node_count = 19


class _FakeAnimationPlayer:
    def __init__(self):
        self.loop = True
        self.times = []
        self.runtime = _FakeAnimationRuntime()

    def set_time_seconds(self, time_seconds):
        self.times.append(float(time_seconds))


class _FakePandaSceneForBindings:
    def __init__(self):
        self.load_panda_assets = False
        self.screen_root = None
        self.screen_texture_target = None

    def attach_screen_root(self, root):
        self.screen_root = root

    def attach_screen_texture_target(self, target):
        self.screen_texture_target = target


class _FakePandaRendererForBindings:
    def __init__(self):
        self.scene = _FakePandaSceneForBindings()
        self.environments = []
        self.controllers = []
        self.frame_states = []

    def load_environment(self, path):
        self.environments.append(path)

    def load_controller(self, hand, path):
        self.controllers.append((hand, path))

    def update_frame_state(self, frame_state):
        self.frame_states.append(frame_state)


class _FakeNvDxAdapter:
    def __init__(self):
        self.events = []

    def get_or_create_fbo(self, image):
        self.events.append(("fbo", image.eye_index, image.image_index))
        return f"fbo{image.eye_index}"

    def lock(self, image):
        self.events.append(("lock", image.eye_index, image.image_index))

    def unlock(self, image):
        self.events.append(("unlock", image.eye_index, image.image_index))


class _FakePandaRenderableScene:
    def __init__(self):
        self.calls = []

    def render_to_framebuffers(self, **kwargs):
        self.calls.append(kwargs)


class _FakeGraphicsEngine:
    def __init__(self):
        self.render_frame_calls = 0

    def render_frame(self):
        self.render_frame_calls += 1


class _FakePandaBase:
    def __init__(self):
        self.render = object()
        self.graphicsEngine = _FakeGraphicsEngine()


class _FakeReparentableRoot:
    def __init__(self):
        self.parents = []

    def reparent_to(self, parent):
        self.parents.append(parent)


class _FakeModernGlFramebuffer:
    def __init__(self, glo):
        self.glo = glo


class _FakeLens:
    def __init__(self):
        self.fov = None

    def set_fov(self, horizontal, vertical):
        self.fov = (horizontal, vertical)


class _FakeCameraNode:
    def __init__(self, lens):
        self._lens = lens

    def get_lens(self):
        return self._lens


class _FakeEyeCamera(_FakePoseRoot):
    def __init__(self):
        super().__init__()
        self.lens = _FakeLens()
        self._node = _FakeCameraNode(self.lens)

    def node(self):
        return self._node


def test_gltf_renderer_selector_defaults_to_native():
    config = resolve_gltf_renderer_mode({})

    assert config.renderer_mode == "native"
    assert config.requested_mode == "native"
    assert not config.panda3d_requested
    assert not config.panda3d_enabled
    assert config.fallback_reason == ""


def test_gltf_renderer_selector_accepts_panda3d_request_when_available():
    config = resolve_gltf_renderer_mode(
        {GLTF_RENDERER_ENV_VAR: " Panda3D "},
        panda3d_available=True,
    )

    assert config.renderer_mode == "panda3d"
    assert config.requested_mode == "panda3d"
    assert config.panda3d_requested
    assert config.panda3d_enabled


def test_gltf_renderer_selector_falls_back_until_panda3d_is_available():
    config = resolve_gltf_renderer_mode({GLTF_RENDERER_ENV_VAR: "panda3d"})

    assert config.renderer_mode == "native"
    assert config.requested_mode == "panda3d"
    assert config.panda3d_requested
    assert not config.panda3d_enabled
    assert "Phase 0 OpenXR swapchain gate" in config.fallback_reason


def test_gltf_renderer_selector_falls_back_on_invalid_mode():
    messages = []
    config = resolve_gltf_renderer_mode({GLTF_RENDERER_ENV_VAR: "bad"})
    log_renderer_selection(config, messages.append)

    assert config.renderer_mode == "native"
    assert config.requested_mode == "bad"
    assert "unsupported D2S_GLTF_RENDERER" in config.fallback_reason
    assert messages == [f"[OpenXRViewer] glTF renderer fallback: {config.fallback_reason}"]


def test_d3d11_init_resolves_gltf_renderer_selector_without_replacing_native_path():
    source = (ROOT / "src" / "xr_viewer" / "core_openxr_d3d11.py").read_text(encoding="utf-8")

    assert "resolve_gltf_renderer_mode(" in source
    assert "panda3d_available=True" in source
    assert "_gltf_renderer_config" in source
    assert "self._gltf_renderer_config.panda3d_requested" in source
    assert "PandaSceneRenderer()" in source
    assert "D3D11 native renderer active" in source
    assert "self._d3d11_native_renderer = D3D11NativeRenderer" in source


def test_panda_scene_graph_default_mode_stays_import_light():
    scene = PandaSceneGraph()

    scene.load_environment(str(ARTEMIS))

    assert scene.environment is not None
    assert scene.environment.loaded_with_panda is False
    assert scene.environment.node_count == 0
    assert scene.environment.geom_count == 0


def test_panda_scene_graph_can_own_panda_loaded_environment_root():
    pytest.importorskip("gltf")
    pytest.importorskip("panda3d")
    scene = PandaSceneGraph(load_panda_assets=True)

    scene.load_environment(str(ARTEMIS))

    assert scene.environment is not None
    assert scene.environment.loaded_with_panda is True
    assert scene.environment.node_count > 0
    assert scene.environment.geom_count > 0
    assert scene.environment.animation_channel_count == 38
    assert scene.environment.animation_target_node_count == 19
    assert scene.environment.animation_bound_node_count == 19
    assert scene.environment.animation_duration_seconds == pytest.approx(15.0)
    assert scene.loaded_assets() == (scene.environment,)
    assert "_environment_root" not in scene.environment.__dict__

    pose = PandaPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    scene.update_frame_state(
        PandaFrameState(
            animation_time_seconds=7.5,
            frame_index=22,
            eye_views=(PandaEyeView(0, pose=pose), PandaEyeView(1, pose=pose)),
            controller_poses={"right": pose, "left": pose},
            screen_pose=pose,
            screen_texture=object(),
        )
    )
    assert scene.frame_state.animation_time_seconds == pytest.approx(7.5)
    assert scene._environment_animation_player.time_seconds == pytest.approx(7.5)
    assert scene.snapshot.frame_index == 22
    assert scene.snapshot.animation_time_seconds == pytest.approx(7.5)
    assert scene.snapshot.animation_sample_count == 1
    assert scene.snapshot.animation_applied_player_count == 1
    assert scene.snapshot.animation_player_count == 1
    assert scene.snapshot.animation_channel_count == 38
    assert scene.snapshot.animation_bound_node_count == 19
    assert scene.snapshot.controller_hands == ("left", "right")
    assert scene.snapshot.screen_pose_present is True
    assert scene.snapshot.screen_texture_present is True
    assert scene.snapshot.eye_view_count == 2
    assert scene.snapshot.applied_controller_hands == ()
    assert scene.snapshot.screen_pose_applied is False

    scene.release()
    assert scene.released
    assert scene.loaded_assets() == ()


def test_panda_scene_graph_applies_controller_pose_to_loaded_roots():
    pytest.importorskip("panda3d")
    scene = PandaSceneGraph()
    root = _FakePoseRoot()
    scene._controller_roots["left"] = root
    pose = PandaPose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))

    scene.update_frame_state(PandaFrameState(controller_poses={"left": pose}))

    assert scene.snapshot.applied_controller_hands == ("left",)
    assert root.pos_quat is not None
    pos, quat = root.pos_quat
    assert tuple(round(pos[index], 4) for index in range(3)) == (1.0, 2.0, 3.0)
    assert quat.get_r() == pytest.approx(1.0)
    assert quat.get_i() == pytest.approx(0.0)
    assert quat.get_j() == pytest.approx(0.0)
    assert quat.get_k() == pytest.approx(0.0)


def test_panda_scene_graph_applies_controller_ray_to_attached_target():
    scene = PandaSceneGraph()
    target = _FakeControllerRayTarget()
    scene.attach_controller_ray_target("left", target)
    ray = PandaControllerRay(
        origin=(1.0, 2.0, 3.0),
        direction=(0.0, 0.0, -1.0),
        length=12.5,
        hit_target="screen",
    )

    scene.update_frame_state(PandaFrameState(controller_rays={"left": ray}))

    assert scene.snapshot.controller_ray_hands == ("left",)
    assert scene.snapshot.applied_controller_ray_hands == ("left",)
    assert target.controller_ray is ray

    with pytest.raises(ValueError, match="controller ray hand"):
        scene.attach_controller_ray_target("middle", target)


def test_panda_scene_graph_applies_screen_pose_to_attached_root():
    pytest.importorskip("panda3d")
    scene = PandaSceneGraph()
    root = _FakePoseRoot()
    scene.attach_screen_root(root)
    pose = PandaPose((4.0, 5.0, 6.0), (0.0, 0.0, 0.0, 1.0))

    scene.update_frame_state(PandaFrameState(screen_pose=pose))

    assert scene.snapshot.screen_pose_present is True
    assert scene.snapshot.screen_pose_applied is True
    assert root.pos_quat is not None
    pos, quat = root.pos_quat
    assert tuple(round(pos[index], 4) for index in range(3)) == (4.0, 5.0, 6.0)
    assert quat.get_r() == pytest.approx(1.0)


def test_panda_scene_graph_applies_screen_texture_to_attached_target():
    scene = PandaSceneGraph()
    target = _FakeScreenTextureTarget()
    scene.attach_screen_texture_target(target)
    screen_texture = PandaScreenTextureFrame(
        width=1920,
        height=1080,
        format="rgba8",
        native_id=77,
        frame_index=12,
        payload=object(),
    )

    scene.update_frame_state(PandaFrameState(screen_texture=screen_texture))

    assert scene.snapshot.screen_texture_present is True
    assert scene.snapshot.screen_texture_applied is True
    assert scene.snapshot.screen_texture_width == 1920
    assert scene.snapshot.screen_texture_height == 1080
    assert scene.snapshot.screen_texture_format == "rgba8"
    assert scene.snapshot.screen_texture_native_id_available is True
    assert target.screen_texture is screen_texture


def test_stereo_targets_default_mode_stays_import_light():
    targets = StereoTargets()

    targets.rebuild(StereoTargetSpec(64, 64, "rgba8"), StereoTargetSpec(64, 64, "rgba8"))

    left, right = targets.target_refs()
    assert targets.ready
    assert targets.generation == 1
    assert not left.created_with_panda
    assert not right.created_with_panda
    assert not left.texture_native_id_available
    assert not right.texture_native_id_available


def test_stereo_targets_can_create_panda_offscreen_targets():
    pytest.importorskip("panda3d")
    targets = StereoTargets(create_panda_targets=True)

    targets.rebuild(StereoTargetSpec(64, 64, "rgba8"), StereoTargetSpec(64, 64, "rgba8"))

    left, right = targets.target_refs()
    assert targets.ready
    assert left.created_with_panda
    assert right.created_with_panda
    assert left.texture_native_id_available
    assert right.texture_native_id_available
    assert left.buffer_name == "d2s-panda-eye-0"
    assert right.buffer_name == "d2s-panda-eye-1"
    assert len(targets._panda_cameras) == 2
    assert len(targets._panda_display_regions) == 2

    targets.release()
    assert targets.released
    assert targets.target_refs() == ()


def test_panda_bridge_cache_key_includes_session_eye_image_size_and_format():
    bridge = PandaBridge()
    left = SwapchainImageRef(0, 3, object(), 100, 120, "rgba8", session_generation=2)
    right = SwapchainImageRef(1, 4, object(), 100, 120, "rgba8", session_generation=2)

    left_resource = bridge.ensure_resource(left)
    right_resource = bridge.ensure_resource(right)

    assert left_resource.key == SwapchainResourceKey(2, 0, 3, 100, 120, "rgba8")
    assert right_resource.key == SwapchainResourceKey(2, 1, 4, 100, 120, "rgba8")
    assert bridge.ensure_resource(left) is left_resource
    assert len(bridge.resources) == 2

    bridge.invalidate_session(2)
    assert bridge.resources == {}


def test_panda_scene_graph_renders_panda_targets_then_blits_to_framebuffers(monkeypatch):
    blits = []
    monkeypatch.setattr(
        panda_scene_module,
        "_blit_panda_texture_to_framebuffer",
        lambda texture, framebuffer, width, height: blits.append((texture, framebuffer.glo, width, height)),
    )
    monkeypatch.setattr(
        panda_scene_module,
        "_apply_pose_to_node_path",
        lambda node, pose: setattr(node, "pos_quat", (pose.position, pose.orientation)) or True,
    )
    scene = PandaSceneGraph()
    environment_root = _FakeReparentableRoot()
    controller_root = _FakeReparentableRoot()
    screen_root = _FakeReparentableRoot()
    ray_root = _FakeReparentableRoot()
    scene._environment_root = environment_root
    scene._controller_roots["left"] = controller_root
    scene._screen_root = screen_root
    scene._controller_ray_targets["left"] = type("RayTarget", (), {"ray_node": ray_root})()
    targets = StereoTargets()
    targets.left = StereoTargetSpec(100, 120, "rgba8")
    targets.right = StereoTargetSpec(101, 121, "rgba8")
    targets._panda_base = _FakePandaBase()
    targets._panda_textures = ["left-texture", "right-texture"]
    left_camera = _FakeEyeCamera()
    right_camera = _FakeEyeCamera()
    targets._panda_cameras = [left_camera, right_camera]
    left_resource = type("Resource", (), {"key": SwapchainResourceKey(2, 0, 3, 100, 120, "rgba8")})()
    right_resource = type("Resource", (), {"key": SwapchainResourceKey(2, 1, 4, 101, 121, "rgba8")})()

    context_calls = []
    scene.render_to_framebuffers(
        targets=targets,
        frame_state=PandaFrameState(
            eye_views=(
                PandaEyeView(
                    0,
                    pose=PandaPose((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0)),
                    fov={"angle_left": -0.5, "angle_right": 0.5, "angle_up": 0.4, "angle_down": -0.4},
                ),
                PandaEyeView(
                    1,
                    pose=PandaPose((4.0, 5.0, 6.0), (0.0, 0.0, 0.0, 1.0)),
                    fov={"angle_left": -0.6, "angle_right": 0.6, "angle_up": 0.3, "angle_down": -0.3},
                ),
            )
        ),
        left_framebuffer=_FakeModernGlFramebuffer(31),
        right_framebuffer=_FakeModernGlFramebuffer(32),
        left_resource=left_resource,
        right_resource=right_resource,
        make_target_context_current=lambda: context_calls.append("target-context"),
        require_shared_context=True,
    )
    scene.render_to_framebuffers(
        targets=targets,
        frame_state=PandaFrameState(),
        left_framebuffer=_FakeModernGlFramebuffer(33),
        right_framebuffer=_FakeModernGlFramebuffer(34),
        left_resource=left_resource,
        right_resource=right_resource,
        make_target_context_current=lambda: context_calls.append("target-context"),
        require_shared_context=True,
    )

    assert targets._panda_base.graphicsEngine.render_frame_calls == 4
    assert context_calls == ["target-context", "target-context"]
    assert environment_root.parents == [targets._panda_base.render]
    assert controller_root.parents == [targets._panda_base.render]
    assert screen_root.parents == [targets._panda_base.render]
    assert ray_root.parents == [targets._panda_base.render]
    assert left_camera.pos_quat is not None
    assert right_camera.pos_quat is not None
    assert left_camera.lens.fov == pytest.approx((57.2958, 45.8366), rel=1e-4)
    assert right_camera.lens.fov == pytest.approx((68.7549, 34.3775), rel=1e-4)
    assert blits == [
        ("left-texture", 31, 100, 120),
        ("right-texture", 32, 101, 121),
        ("left-texture", 33, 100, 120),
        ("right-texture", 34, 101, 121),
    ]


def test_panda_scene_graph_render_to_framebuffers_requires_panda_targets():
    scene = PandaSceneGraph()

    with pytest.raises(RuntimeError, match="ShowBase"):
        scene.render_to_framebuffers(
            targets=StereoTargets(),
            frame_state=object(),
            left_framebuffer=object(),
            right_framebuffer=object(),
        )


def test_panda_opengl_bridge_renders_to_supplied_framebuffers_and_caches_resources():
    context_calls = []
    bridge = PandaOpenGLBridge(make_target_context_current=lambda: context_calls.append("target-context"))
    scene = _FakePandaRenderableScene()
    left = SwapchainImageRef(0, 3, "left-fbo", 100, 120, "rgba8", session_generation=2)
    right = SwapchainImageRef(1, 4, "right-fbo", 101, 121, "rgba8", session_generation=2)

    result = bridge.render_eyes(
        scene=scene,
        targets="targets",
        frame_state="frame",
        left_image=left,
        right_image=right,
    )

    assert result.rendered is True
    assert result.bridge_mode == "opengl"
    assert len(scene.calls) == 1
    assert scene.calls[0]["left_framebuffer"] == "left-fbo"
    assert scene.calls[0]["right_framebuffer"] == "right-fbo"
    assert callable(scene.calls[0]["make_target_context_current"])
    assert scene.calls[0]["require_shared_context"] is True
    assert context_calls == []
    assert len(bridge.resources) == 2

def test_panda_opengl_bridge_requires_target_context_callback():
    bridge = PandaOpenGLBridge()
    scene = _FakePandaRenderableScene()

    with pytest.raises(PandaBridgeUnavailable, match="target OpenXR GL context"):
        bridge.render_eyes(
            scene=scene,
            targets="targets",
            frame_state="frame",
            left_image=SwapchainImageRef(0, 3, "left-fbo", 100, 120, "rgba8"),
            right_image=SwapchainImageRef(1, 4, "right-fbo", 101, 121, "rgba8"),
        )


def test_panda_nv_dx_bridge_locks_renders_unlocks_and_caches_resources():
    adapter = _FakeNvDxAdapter()
    bridge = PandaNvDxBridge(adapter)
    scene = _FakePandaRenderableScene()
    left = SwapchainImageRef(0, 3, "tex0", 100, 120, 87, session_generation=2)
    right = SwapchainImageRef(1, 4, "tex1", 101, 121, 87, session_generation=2)

    result = bridge.render_eyes(
        scene=scene,
        targets="targets",
        frame_state="frame",
        left_image=left,
        right_image=right,
    )

    assert result.rendered is True
    assert result.bridge_mode == "nv_dx"
    assert adapter.events == [
        ("fbo", 0, 3),
        ("fbo", 1, 4),
        ("lock", 0, 3),
        ("lock", 1, 4),
        ("unlock", 1, 4),
        ("unlock", 0, 3),
    ]
    assert len(scene.calls) == 1
    assert scene.calls[0]["left_framebuffer"] == "fbo0"
    assert scene.calls[0]["right_framebuffer"] == "fbo1"
    assert len(bridge.resources) == 2


def test_panda_nv_dx_bridge_unlocks_when_scene_render_hook_is_missing():
    adapter = _FakeNvDxAdapter()
    bridge = PandaNvDxBridge(adapter)
    left = SwapchainImageRef(0, 0, "tex0", 64, 64, 87)
    right = SwapchainImageRef(1, 0, "tex1", 64, 64, 87)

    with pytest.raises(PandaBridgeUnavailable, match="render_to_framebuffers"):
        bridge.render_eyes(
            scene=object(),
            targets=object(),
            frame_state=object(),
            left_image=left,
            right_image=right,
        )

    assert adapter.events[-2:] == [("unlock", 1, 0), ("unlock", 0, 0)]


def test_unimplemented_panda_bridge_fails_explicitly_after_caching_resources():
    bridge = PandaBridge()

    try:
        bridge.render_eyes(
            scene=object(),
            targets=object(),
            frame_state=object(),
            left_image=SwapchainImageRef(0, 0, object(), 64, 64, "rgba8"),
            right_image=SwapchainImageRef(1, 0, object(), 64, 64, "rgba8"),
        )
    except PandaBridgeUnavailable as exc:
        assert "no concrete NV_DX or CUDA implementation" in str(exc)
    else:
        raise AssertionError("unimplemented PandaBridge must fail explicitly")

    assert len(bridge.resources) == 2


def test_panda_runtime_diagnostics_snapshot_summarizes_assets_targets_and_bridge():
    bridge = PandaBridge()
    bridge.ensure_resource(SwapchainImageRef(0, 2, object(), 64, 72, "rgba8", session_generation=5))
    renderer = PandaSceneRenderer(bridge=bridge)

    renderer.load_environment("Artemis/environment.glb")
    renderer.rebuild_targets(
        StereoTargetSpec(64, 72, "rgba8"),
        StereoTargetSpec(64, 72, "rgba8"),
    )
    pose = PandaPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    renderer.update_frame_state(PandaFrameState(predicted_display_time=100.25, frame_index=10))
    renderer.update_frame_state(
        PandaFrameState(
            predicted_display_time=101.75,
            frame_index=11,
            eye_views=(PandaEyeView(0, pose=pose), PandaEyeView(1, pose=pose)),
            controller_poses={"left": pose, "right": pose},
            controller_rays={"left": PandaControllerRay((0.0, 0.0, 0.0), (0.0, 0.0, -1.0))},
            screen_pose=pose,
        )
    )
    snapshot = renderer.diagnostics_snapshot()
    snapshot_json = renderer.diagnostics_json()

    assert snapshot.released is False
    assert snapshot.frame_predicted_display_time == pytest.approx(101.75)
    assert snapshot.frame_animation_time_seconds == pytest.approx(1.5)
    assert snapshot.frame_index == 11
    assert snapshot.frame_eye_view_count == 2
    assert snapshot.frame_controller_count == 2
    assert snapshot.frame_controller_ray_count == 1
    assert snapshot.frame_screen_pose_present is True
    assert snapshot.scene_animation_time_seconds == pytest.approx(1.5)
    assert snapshot.scene_animation_sample_count == 0
    assert snapshot.scene_animation_applied_player_count == 0
    assert snapshot.scene_animation_player_count == 0
    assert snapshot.scene_animation_channel_count == 0
    assert snapshot.scene_animation_bound_node_count == 0
    assert snapshot.scene_controller_hands == ("left", "right")
    assert snapshot.scene_controller_ray_hands == ("left",)
    assert snapshot.scene_applied_controller_ray_hands == ()
    assert snapshot.scene_screen_pose_present is True
    assert snapshot.scene_screen_texture_present is False
    assert snapshot.scene_screen_texture_applied is False
    assert snapshot.scene_screen_texture_width == 0
    assert snapshot.scene_screen_texture_height == 0
    assert snapshot.scene_screen_texture_format == ""
    assert snapshot.scene_screen_texture_native_id_available is False
    assert snapshot.scene_eye_view_count == 2
    assert snapshot.scene_applied_controller_hands == ()
    assert snapshot.scene_screen_pose_applied is False
    assert snapshot.event_count == 2
    assert snapshot.events == ("environment_loaded", "stereo_targets_rebuilt")
    assert snapshot.scene_assets[0]["role"] == "environment"
    assert snapshot.scene_assets[0]["animation_channel_count"] == 0
    assert snapshot.scene_assets[0]["animation_duration_seconds"] == 0.0
    assert snapshot.target_ready
    assert snapshot.target_generation == 1
    assert snapshot.target_refs[0]["eye_index"] == 0
    assert snapshot.bridge_mode == "unimplemented"
    assert snapshot.bridge_resource_count == 1
    assert snapshot.bridge_resource_keys == ("session=5:eye=0:image=2:size=64x72:format=rgba8",)
    assert snapshot.render_success_count == 0
    assert snapshot.render_failure_count == 0
    assert snapshot.last_render_bridge_mode == ""
    assert snapshot.last_render_left_rendered is False
    assert snapshot.last_render_right_rendered is False
    assert snapshot.last_render_error == ""
    assert snapshot.last_render_cpu_seconds == pytest.approx(0.0)
    assert '"bridge_resource_count": 1' in snapshot_json
    assert '"frame_animation_time_seconds": 1.5' in snapshot_json
    assert '"scene_animation_time_seconds": 1.5' in snapshot_json
    assert '"frame_eye_view_count": 2' in snapshot_json
    assert '"scene_controller_hands": [' in snapshot_json
    assert '"scene_assets"' in snapshot_json


def test_panda_controller_ray_geometry_target_builds_line_segments():
    pytest.importorskip("panda3d")
    from panda3d.core import NodePath

    parent = NodePath("controller-ray-root")
    target = PandaControllerRayGeometryTarget(parent)
    visible_ray = PandaControllerRay((1.0, 2.0, 3.0), (0.0, 0.0, -2.0), length=4.0)

    ray_node = target.set_controller_ray(visible_ray)

    assert ray_node is not None
    assert target.ray_node is ray_node
    assert target.last_ray is visible_ray
    assert parent.get_num_children() == 1

    hidden_ray = PandaControllerRay((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), visible=False)
    assert target.set_controller_ray(hidden_ray) is None
    assert target.ray_node is None
    assert parent.get_num_children() == 0

    with pytest.raises(PandaControllerRayGeometryError, match="non-zero"):
        target.set_controller_ray(PandaControllerRay((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))


def test_panda_screen_texture_upload_target_binds_payload_to_node_path():
    pytest.importorskip("panda3d")
    from panda3d.core import NodePath

    node = NodePath("screen")
    target = PandaScreenTextureUploadTarget(node)
    frame = PandaScreenTextureFrame(
        2,
        2,
        frame_index=33,
        payload=bytes(range(16)),
    )

    texture = target.set_screen_texture(frame)

    assert target.texture is texture
    assert target.last_frame_index == 33
    assert node.get_texture() == texture
    assert texture.get_x_size() == 2
    assert texture.get_y_size() == 2

    with pytest.raises(PandaScreenTextureUploadError, match="byte length"):
        target.set_screen_texture(PandaScreenTextureFrame(2, 2, payload=b"short"))


def test_panda_screen_texture_frame_validates_dimensions():
    screen_texture = PandaScreenTextureFrame(1920, 1080, native_id=42)

    assert screen_texture.native_id_available is True

    with pytest.raises(ValueError, match="dimensions"):
        PandaScreenTextureFrame(0, 1080)


def test_panda_controller_ray_validates_vectors_and_length():
    ray = PandaControllerRay((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), length=5.0)

    assert ray.visible is True
    assert ray.length == pytest.approx(5.0)

    with pytest.raises(ValueError, match="3D vectors"):
        PandaControllerRay((0.0, 0.0), (0.0, 0.0, -1.0))
    with pytest.raises(ValueError, match="positive"):
        PandaControllerRay((0.0, 0.0, 0.0), (0.0, 0.0, -1.0), length=0.0)


def test_panda_frame_source_converts_matrices_and_rays_to_frame_state():
    screen_mat = [
        [1.0, 0.0, 0.0, 4.0],
        [0.0, 1.0, 0.0, 5.0],
        [0.0, 0.0, 1.0, 6.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    eye_mat = [
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 0.0, 0.2],
        [0.0, 0.0, 1.0, 0.3],
        [0.0, 0.0, 0.0, 1.0],
    ]
    screen_texture = PandaScreenTextureFrame(2, 2, payload=bytes(range(16)))
    ray = controller_ray_from_vectors((1, 2, 3), (0, 0, -1), length=7.0, hit_target="screen")

    frame_state = build_panda_frame_state(
        PandaFrameSourceInput(
            predicted_display_time=12.5,
            frame_index=99,
            eye_pose_mats=(eye_mat, None),
            eye_fovs=({"angle_left": -0.5, "angle_right": 0.5, "angle_up": 0.4, "angle_down": -0.4}, None),
            controller_pose_mats={"left": screen_mat},
            controller_rays={"left": ray},
            screen_pose_mat=screen_mat,
            screen_texture=screen_texture,
        )
    )

    assert frame_state.predicted_display_time == pytest.approx(12.5)
    assert frame_state.frame_index == 99
    assert frame_state.eye_views[0].pose.position == pytest.approx((0.1, 0.2, 0.3))
    assert frame_state.eye_views[0].fov["angle_right"] == pytest.approx(0.5)
    assert frame_state.eye_views[1].pose is None
    assert frame_state.controller_poses["left"].position == pytest.approx((4.0, 5.0, 6.0))
    assert frame_state.controller_rays["left"] is ray
    assert frame_state.screen_pose.position == pytest.approx((4.0, 5.0, 6.0))
    assert frame_state.screen_texture is screen_texture

    pose = mat4_to_panda_pose(screen_mat)
    assert pose.orientation == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_openxr_panda_frame_state_carries_eye_fovs():
    left_fov = object()
    right_fov = object()
    views = [type("View", (), {"fov": left_fov})(), type("View", (), {"fov": right_fov})()]

    assert _eye_fovs(views) == (left_fov, right_fov)
    assert _eye_fovs([]) == (None, None)


def test_projection_layer_attempts_panda_opengl_bridge_before_native_fallback(monkeypatch, capsys):
    renderer = PandaSceneRenderer(bridge=_RecordingBridge())
    renderer.update_frame_state(PandaFrameState(predicted_display_time=1.0))
    viewer = type("Viewer", (), {})()
    viewer._use_d3d11 = False
    viewer._gltf_renderer_config = type("Config", (), {"panda3d_enabled": True})()
    viewer._panda_scene_renderer = renderer
    viewer._ensure_projection_swapchains = lambda: True
    viewer._breakdown_inc = lambda _name, amount=1: None
    presenter = ProjectionLayerPresenter(viewer)
    monkeypatch.setattr(presenter, "render_opengl", lambda *args, **kwargs: ["native-opengl"])

    result = presenter.render_projection(
        enabled=True,
        views=[],
        default_fov=None,
        default_proj=None,
        default_proj_d3d=None,
    )
    fallback_result = presenter.render_projection(
        enabled=True,
        views=[],
        default_fov=None,
        default_proj=None,
        default_proj_d3d=None,
    )

    output = capsys.readouterr().out
    assert result == []
    assert fallback_result == ["native-opengl"]
    assert "Panda3D render path status status=bridge-failed" in output
    assert "backend=opengl" in output
    assert "scene-bound-only" not in output

def test_openxr_panda_scene_binding_logs_when_screen_binding_becomes_active(monkeypatch, capsys):
    bindings = iter(
        [
            type(
                "Binding",
                (),
                {
                    "loaded": True,
                    "screen_target_bound": False,
                    "screen_size": None,
                    "environment_path": "Artemis/environment.glb",
                },
            )(),
            type(
                "Binding",
                (),
                {
                    "loaded": True,
                    "screen_target_bound": True,
                    "screen_size": (2.4, 1.35),
                    "environment_path": "Artemis/environment.glb",
                },
            )(),
        ]
    )
    monkeypatch.setattr(
        "xr_viewer.openxr_panda_frame_state.sync_panda_scene_assets_from_viewer",
        lambda viewer: next(bindings),
    )
    renderer = _FakePandaRendererForBindings()
    viewer = type("Viewer", (), {})()
    viewer._gltf_renderer_config = type("Config", (), {"panda3d_requested": True})()
    viewer._panda_scene_renderer = renderer
    viewer._frame_count = 1
    viewer.screen_width = 2.4
    viewer.screen_height = None
    viewer._screen_pose_mat4 = lambda: None
    viewer._ensure_screen_dimensions = lambda: setattr(viewer, "screen_height", 1.35)

    update_panda_frame_state_from_viewer(
        viewer,
        predicted_display_time=10.0,
        views=[],
        screen_frame_uploaded=False,
    )
    update_panda_frame_state_from_viewer(
        viewer,
        predicted_display_time=10.1,
        views=[],
        screen_frame_uploaded=False,
    )

    output = capsys.readouterr().out
    assert "Panda3D scene binding active loaded=True screen_bound=False" in output
    assert "Panda3D scene binding active loaded=True screen_bound=True" in output
    assert len(renderer.frame_states) == 2


def test_panda_frame_state_validates_same_frame_eye_views():
    pose = PandaPose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    validate_frame_state(
        PandaFrameState(
            predicted_display_time=1.0,
            eye_views=(PandaEyeView(0, pose=pose), PandaEyeView(1, pose=pose)),
        )
    )

    with pytest.raises(PandaRuntimeUnavailable, match="mismatched eye_index"):
        validate_frame_state(
            PandaFrameState(
                predicted_display_time=1.0,
                eye_views=(PandaEyeView(1, pose=pose), PandaEyeView(0, pose=pose)),
            )
        )

    with pytest.raises(PandaRuntimeUnavailable, match="predicted_display_time"):
        validate_frame_state(PandaFrameState(predicted_display_time=float("inf")))


def test_panda_animation_clock_uses_xr_predicted_display_time_monotonically():
    clock = PandaAnimationClock()

    assert clock.sample(50.0) == pytest.approx(0.0)
    assert clock.sample(50.25) == pytest.approx(0.25)
    assert clock.sample(50.10) == pytest.approx(0.25)
    assert clock.sample(52.0) == pytest.approx(2.0)
    assert clock.origin_predicted_display_time == pytest.approx(50.0)
    assert clock.last_animation_time_seconds == pytest.approx(2.0)

    clock.reset()
    assert clock.origin_predicted_display_time is None
    assert clock.sample(10.0) == pytest.approx(0.0)


def test_panda_animation_clock_supports_runtime_controls():
    clock = PandaAnimationClock()

    assert clock.sample(100.0) == pytest.approx(0.0)
    assert clock.configure(playback_speed=2.0).playback_speed == pytest.approx(2.0)
    assert clock.sample(100.5) == pytest.approx(1.0)
    assert clock.configure(paused=True).paused is True
    assert clock.sample(101.5) == pytest.approx(1.0)
    assert clock.configure(paused=False, fixed_time_seconds=7.5).fixed_time_seconds == pytest.approx(7.5)
    assert clock.sample(110.0) == pytest.approx(7.5)
    assert clock.configure(fixed_time_seconds=None).fixed_time_seconds is None
    assert clock.sample(111.0) == pytest.approx(9.5)

    with pytest.raises(PandaRuntimeUnavailable, match="playback_speed"):
        clock.configure(playback_speed=-1.0)
    with pytest.raises(PandaRuntimeUnavailable, match="fixed_time_seconds"):
        clock.configure(fixed_time_seconds=float("nan"))


def test_panda_scene_renderer_configures_animation_runtime_controls():
    renderer = PandaSceneRenderer()
    player = _FakeAnimationPlayer()
    renderer.scene._environment_animation_player = player

    state = renderer.configure_animation(playback_speed=0.5, paused=True, fixed_time_seconds=3.0, loop=False)
    renderer.update_frame_state(PandaFrameState(predicted_display_time=10.0))
    renderer.update_frame_state(PandaFrameState(predicted_display_time=12.0))

    assert state.playback_speed == pytest.approx(0.5)
    assert state.paused is True
    assert state.fixed_time_seconds == pytest.approx(3.0)
    assert state.loop is False
    assert player.loop is False
    assert player.times == [pytest.approx(3.0), pytest.approx(3.0)]
    assert renderer.scene.snapshot.animation_sample_count == 2
    assert renderer.scene.snapshot.animation_applied_player_count == 1
    assert renderer.scene.snapshot.animation_player_count == 1
    assert renderer.scene.snapshot.animation_channel_count == 38
    assert renderer.scene.snapshot.animation_bound_node_count == 19
    snapshot = renderer.diagnostics_snapshot()
    assert snapshot.animation_playback_speed == pytest.approx(0.5)
    assert snapshot.animation_paused is True
    assert snapshot.animation_fixed_time_seconds == pytest.approx(3.0)
    assert snapshot.animation_loop is False
    assert snapshot.scene_animation_time_seconds == pytest.approx(3.0)
    assert snapshot.scene_animation_sample_count == 2
    assert snapshot.scene_animation_applied_player_count == 1
    assert snapshot.scene_animation_player_count == 1
    assert snapshot.scene_animation_channel_count == 38
    assert snapshot.scene_animation_bound_node_count == 19
    assert "animation_configured" in snapshot.events


def test_panda_scene_binding_loads_active_environment_and_controllers(tmp_path, monkeypatch):
    env_path = tmp_path / "environment.glb"
    env_path.write_bytes(b"glb")
    controllers = tmp_path / "controllers" / "pico"
    controllers.mkdir(parents=True)
    left = controllers / "left.glb"
    right = controllers / "right.glb"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    screen_target = type(
        "FakeScreenNodeTarget",
        (),
        {"root": "screen-root", "texture_target": "screen-texture-target"},
    )()
    monkeypatch.setattr(
        "xr_viewer.panda_runtime.screen_node.create_panda_screen_node_target",
        lambda width, height: screen_target,
    )
    renderer = _FakePandaRendererForBindings()
    viewer = type("Viewer", (), {})()
    viewer._panda_scene_renderer = renderer
    viewer._gltf_renderer_config = type("Config", (), {"panda3d_requested": True})()
    viewer._env_model_path = str(env_path)
    viewer._controllers_root = str(tmp_path / "controllers")
    viewer._current_brand = "pico"
    viewer._controller_model = "fallback"
    viewer.screen_width = 2.4
    viewer.screen_height = 1.35

    result = sync_panda_scene_assets_from_viewer(viewer)
    second = sync_panda_scene_assets_from_viewer(viewer)

    assert result is second
    assert result.loaded is True
    assert renderer.scene.load_panda_assets is True
    assert renderer.environments == [str(env_path)]
    assert renderer.controllers == [("left", str(left)), ("right", str(right))]
    assert result.screen_size == pytest.approx((2.4, 1.35))
    assert result.screen_target_bound is True
    assert renderer.scene.screen_root is not None
    assert renderer.scene.screen_texture_target is not None
    assert viewer._panda_scene_binding_error == ""


def test_panda_screen_node_target_creates_textured_card():
    pytest.importorskip("panda3d")

    target = create_panda_screen_node_target(2.4, 1.35)

    assert target.width == pytest.approx(2.4)
    assert target.height == pytest.approx(1.35)
    assert target.root.get_name() == "d2s-screen-root"
    assert target.root.get_num_children() == 1
    assert target.texture_target.node_path == target.root.get_child(0)

    with pytest.raises(ValueError, match="screen dimensions"):
        create_panda_screen_node_target(0.0, 1.0)


def test_panda_scene_renderer_facade_contract():
    bridge = _RecordingBridge()
    renderer = PandaSceneRenderer(bridge=bridge)

    renderer.load_environment("Artemis/environment.glb")
    renderer.load_controller("left", "controllers/left.glb")
    left_spec = StereoTargetSpec(100, 120, "rgba8")
    right_spec = StereoTargetSpec(100, 120, "rgba8")
    renderer.rebuild_targets(left_spec, right_spec)
    first_target_generation = renderer.targets.generation
    renderer.rebuild_targets(left_spec, right_spec)
    assert renderer.targets.generation == first_target_generation
    renderer.update_frame_state(PandaFrameState(predicted_display_time=123.0))
    renderer.update_frame_state(PandaFrameState(predicted_display_time=123.5))
    result = renderer.render_eyes(
        SwapchainImageRef(0, 0, object(), 100, 120, "rgba8"),
        SwapchainImageRef(1, 0, object(), 100, 120, "rgba8"),
    )

    assert result.rendered
    assert result.bridge_mode == "test"
    assert len(bridge.calls) == 1
    snapshot = renderer.diagnostics_snapshot()
    assert snapshot.render_success_count == 1
    assert snapshot.render_failure_count == 0
    assert snapshot.last_render_bridge_mode == "test"
    assert snapshot.last_render_left_rendered is True
    assert snapshot.last_render_right_rendered is True
    assert snapshot.last_render_error == ""
    assert snapshot.last_render_cpu_seconds >= 0.0
    assert "render_eyes" in snapshot.events
    frame_state = bridge.calls[0][2]
    assert frame_state.predicted_display_time == pytest.approx(123.5)
    assert frame_state.animation_time_seconds == pytest.approx(0.5)
    assert renderer.scene.frame_state is frame_state
    assert [asset.role for asset in renderer.scene.loaded_assets()] == ["environment", "controller:left"]
    assert renderer.targets.ready
    assert "environment_loaded" in renderer.diagnostics.summary()["events"]

    renderer.release()
    assert renderer.released
    assert bridge.released


def test_panda_scene_renderer_records_render_failure_diagnostics():
    class FailingBridge(_RecordingBridge):
        def render_eyes(self, **_kwargs):
            raise RuntimeError("bridge exploded")

    renderer = PandaSceneRenderer(bridge=FailingBridge())
    renderer.rebuild_targets(
        StereoTargetSpec(100, 120, "rgba8"),
        StereoTargetSpec(100, 120, "rgba8"),
    )
    renderer.update_frame_state(PandaFrameState(predicted_display_time=123.0))

    with pytest.raises(RuntimeError, match="bridge exploded"):
        renderer.render_eyes(
            SwapchainImageRef(0, 0, object(), 100, 120, "rgba8"),
            SwapchainImageRef(1, 0, object(), 100, 120, "rgba8"),
        )

    snapshot = renderer.diagnostics_snapshot()
    assert snapshot.render_success_count == 0
    assert snapshot.render_failure_count == 1
    assert snapshot.last_render_error == "RuntimeError: bridge exploded"
    assert snapshot.last_render_cpu_seconds >= 0.0
    assert "render_failed" in snapshot.events


def test_panda_scene_renderer_requires_frame_state_and_targets():
    renderer = PandaSceneRenderer(bridge=_RecordingBridge())

    try:
        renderer.render_eyes(
            SwapchainImageRef(0, 0, object(), 100, 120, "rgba8"),
            SwapchainImageRef(1, 0, object(), 100, 120, "rgba8"),
        )
    except PandaRuntimeUnavailable as exc:
        assert "update_frame_state" in str(exc)
    else:
        raise AssertionError("render_eyes should require frame state before rendering")
