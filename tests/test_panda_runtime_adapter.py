from pathlib import Path

from xr_viewer.panda_runtime.bridge import PandaBridge, RenderEyesResult, SwapchainImageRef
from xr_viewer.panda_runtime.runtime import (
    GLTF_RENDERER_ENV_VAR,
    PandaFrameState,
    PandaRuntimeUnavailable,
    PandaSceneRenderer,
    log_renderer_selection,
    resolve_gltf_renderer_mode,
)
from xr_viewer.panda_runtime.stereo_targets import StereoTargetSpec


ROOT = Path(__file__).resolve().parents[1]


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
    result = renderer.render_eyes(
        SwapchainImageRef(0, 0, object(), 100, 120, "rgba8"),
        SwapchainImageRef(1, 0, object(), 100, 120, "rgba8"),
    )

    assert result.rendered
    assert result.bridge_mode == "test"
    assert len(bridge.calls) == 1
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
