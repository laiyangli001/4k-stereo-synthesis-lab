from pathlib import Path

import pytest

from xr_viewer.panda_runtime.bridge import (
    PandaBridge,
    PandaBridgeUnavailable,
    RenderEyesResult,
    SwapchainImageRef,
    SwapchainResourceKey,
)
from xr_viewer.panda_runtime.runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaAnimationClock,
    PandaEyeView,
    PandaFrameState,
    PandaPose,
    PandaRuntimeUnavailable,
    PandaSceneRenderer,
    log_renderer_selection,
    resolve_gltf_renderer_mode,
    validate_frame_state,
)
from xr_viewer.panda_runtime.scene import PandaSceneGraph
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

    assert "resolve_gltf_renderer_mode()" in source
    assert "_gltf_renderer_config" in source
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
    assert scene.snapshot.controller_hands == ("left", "right")
    assert scene.snapshot.screen_pose_present is True
    assert scene.snapshot.screen_texture_present is True
    assert scene.snapshot.eye_view_count == 2

    scene.release()
    assert scene.released
    assert scene.loaded_assets() == ()


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
    assert snapshot.frame_screen_pose_present is True
    assert snapshot.scene_controller_hands == ("left", "right")
    assert snapshot.scene_screen_pose_present is True
    assert snapshot.scene_screen_texture_present is False
    assert snapshot.scene_eye_view_count == 2
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
    assert '"bridge_resource_count": 1' in snapshot_json
    assert '"frame_animation_time_seconds": 1.5' in snapshot_json
    assert '"frame_eye_view_count": 2' in snapshot_json
    assert '"scene_controller_hands": [' in snapshot_json
    assert '"scene_assets"' in snapshot_json


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


def test_panda_scene_renderer_facade_contract():
    bridge = _RecordingBridge()
    renderer = PandaSceneRenderer(bridge=bridge)

    renderer.load_environment("Artemis/environment.glb")
    renderer.load_controller("left", "controllers/left.glb")
    renderer.rebuild_targets(
        StereoTargetSpec(100, 120, "rgba8"),
        StereoTargetSpec(100, 120, "rgba8"),
    )
    renderer.update_frame_state(PandaFrameState(predicted_display_time=123.0))
    renderer.update_frame_state(PandaFrameState(predicted_display_time=123.5))
    result = renderer.render_eyes(
        SwapchainImageRef(0, 0, object(), 100, 120, "rgba8"),
        SwapchainImageRef(1, 0, object(), 100, 120, "rgba8"),
    )

    assert result.rendered
    assert result.bridge_mode == "test"
    assert len(bridge.calls) == 1
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
