import torch
from pathlib import Path

from stereo_runtime import (
    OpenXRRenderConfig,
    OpenXRRuntimeResult,
    StereoRuntime,
    StereoRuntimeConfig,
    StereoRuntimeResult,
    openxr_result_from_stereo_result,
)
from stereo_runtime.depth_provider import DepthProfileResult


ROOT = Path(__file__).resolve().parents[1]


class FakeDepthProvider:
    info = {
        "provider": "fake",
        "model_name": "fake",
        "model_id": "fake",
        "depth_resolution": 518,
        "depth_backend": "fake",
    }

    def load(self):
        self.loaded = True

    def predict_profile(self, rgb):
        depth = torch.linspace(
            0.0,
            1.0,
            steps=rgb.shape[-2] * rgb.shape[-1],
            dtype=rgb.dtype,
            device=rgb.device,
        ).reshape(1, 1, rgb.shape[-2], rgb.shape[-1])
        return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)


def test_process_openxr_frame_defaults_to_rgb_depth_runtime_result():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert isinstance(result, OpenXRRuntimeResult)
    assert result.depth.shape == (1, 1, 12, 16)
    assert result.source_rgb is rgb
    assert result.left_eye is rgb
    assert result.right_eye is rgb
    assert result.timing["depth_total_ms"] >= 0.0
    assert result.timing["openxr_render_ms"] == 0.0
    assert result.timing["pack_ms"] == 0.0
    assert result.debug_info["runtime_output_format"] == "openxr_rgb_depth"
    assert result.debug_info["runtime_output_dtype"] == "float32"
    assert result.debug_info["runtime_output_eye_size"] == "16x12"
    assert result.debug_info["runtime_output_display_size"] == "16x12"
    assert result.debug_info["depth_provider_size"] == "518x518"
    assert result.debug_info["depth_render_size"] == "16x12"
    assert result.debug_info["backend"] == "openxr_viewer_shader_dibr"
    assert result.output_format == "openxr_rgb_depth"
    assert result.output_dtype == "float32"
    assert result.output_eye_size == (16, 12)
    assert result.output_display_size == (16, 12)
    assert result.output_pack_backend == "none"
    assert result.active_settings_version == 0
    assert result.hot_reload_class == "no_change"
    assert result.hot_reload_changed_fields == ()


def test_process_openxr_frame_debug_info_carries_preprocess_device_metadata():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)
    rgb._d2s_preprocess_device_transfer = "cpu->cpu"

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert result.debug_info["preprocess_device_transfer"] == "cpu->cpu"
    assert result.debug_info["runtime_output_eye_size"] == "16x12"
    assert result.debug_info["runtime_output_display_size"] == "16x12"
    assert result.output_eye_size == (16, 12)
    assert result.output_display_size == (16, 12)


def test_runtime_debug_info_carries_preprocess_device_metadata():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)
    rgb._d2s_preprocess_backend = "torch_bgr_norm"
    rgb._d2s_preprocess_input_kind = "numpy"
    rgb._d2s_preprocess_device_origin = "cpu"
    rgb._d2s_preprocess_device_output = "cpu"
    rgb._d2s_preprocess_device_transfer = "cpu->cpu"

    result = runtime.process_rgb_frame(rgb)

    assert result.debug_info["preprocess_backend"] == "torch_bgr_norm"
    assert result.debug_info["preprocess_input_kind"] == "numpy"
    assert result.debug_info["preprocess_device_origin"] == "cpu"
    assert result.debug_info["preprocess_device_output"] == "cpu"
    assert result.debug_info["preprocess_device_transfer"] == "cpu->cpu"
    assert result.debug_info["runtime_output_eye_size"] == "16x12"
    assert result.debug_info["runtime_output_display_size"] == "16x12"
    assert result.debug_info["packing_format"] == "half_sbs"
    assert result.output_format == "half_sbs"
    assert result.output_dtype in {"float32", "uint8"}
    assert result.output_eye_size == (16, 12)
    assert result.output_display_size == (16, 12)
    assert result.active_settings_version == result.debug_info["active_settings_version"]
    assert result.hot_reload_class == result.debug_info["hot_reload_class"]
    assert result.hot_reload_changed_fields == tuple(result.debug_info["hot_reload_changed_fields"])


def test_runtime_debug_info_records_eye_and_display_sizes_for_sbs_output():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="full_sbs",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_rgb_frame(rgb)

    assert result.debug_info["runtime_output_eye_size"] == "16x12"
    assert result.debug_info["runtime_output_display_size"] == "32x12"
    assert result.output_eye_size == (16, 12)
    assert result.output_display_size == (32, 12)


def test_openxr_result_from_stereo_result_keeps_full_size_eye_views_for_quality_half_sbs():
    depth = torch.ones(1, 1, 2, 4)
    source_left = torch.zeros(1, 3, 2, 4)
    source_right = torch.ones(1, 3, 2, 4)
    sbs = torch.arange(1 * 3 * 2 * 8, dtype=torch.uint8).reshape(1, 3, 2, 8)
    stereo_result = StereoRuntimeResult(
        depth=depth,
        left_eye=source_left,
        right_eye=source_right,
        sbs=sbs,
        active_settings_version=7,
        hot_reload_class="hot_reload",
        hot_reload_changed_fields=("max_disparity_px",),
        debug_info={"backend": "quality_4k", "runtime_output_format": "half_sbs"},
        timing={"total_ms": 5.0},
        provider_info={"provider": "fake"},
    )

    result = openxr_result_from_stereo_result(stereo_result)

    assert isinstance(result, OpenXRRuntimeResult)
    assert result.depth is depth
    assert result.source_rgb is None
    assert result.left_eye is source_left
    assert result.right_eye is source_right
    assert result.timing == {"total_ms": 5.0}
    assert result.provider_info == {"provider": "fake"}
    assert result.debug_info["backend"] == "quality_4k"
    assert result.debug_info["runtime_output_format"] == "openxr_full_synthesis_eyes"
    assert result.debug_info["runtime_output_dtype"] == "float32"
    assert result.debug_info["runtime_output_eye_size"] == "4x2"
    assert result.debug_info["runtime_output_display_size"] == "4x2"
    assert result.debug_info["runtime_output_pack_backend"] == "none"
    assert result.output_format == "openxr_full_synthesis_eyes"
    assert result.output_dtype == "float32"
    assert result.output_eye_size == (4, 2)
    assert result.output_display_size == (4, 2)
    assert result.output_pack_backend == "none"
    assert result.active_settings_version == 7
    assert result.hot_reload_class == "hot_reload"
    assert result.hot_reload_changed_fields == ("max_disparity_px",)


def test_openxr_result_from_stereo_result_splits_fused_half_sbs_eye_views_and_preserves_display_size():
    depth = torch.ones(1, 1, 2, 4)
    source_left = torch.zeros(1, 3, 2, 4)
    source_right = torch.zeros(1, 3, 2, 4)
    sbs = torch.arange(1 * 3 * 2 * 8, dtype=torch.uint8).reshape(1, 3, 2, 8)
    stereo_result = StereoRuntimeResult(
        depth=depth,
        left_eye=source_left,
        right_eye=source_right,
        sbs=sbs,
        output_format="half_sbs",
        debug_info={
            "backend": "fast_plus",
            "runtime_output_format": "legacy_wrong_format",
            "fast_plus_fused_backend": "triton_half_sbs_uint8",
        },
        timing={"total_ms": 5.0},
        provider_info={"provider": "fake"},
    )

    result = openxr_result_from_stereo_result(stereo_result)

    assert torch.equal(result.left_eye, sbs[:, :, :, :4])
    assert torch.equal(result.right_eye, sbs[:, :, :, 4:8])
    assert result.debug_info["runtime_output_format"] == "openxr_full_synthesis_eyes"
    assert result.debug_info["runtime_output_dtype"] == "uint8"
    assert result.debug_info["runtime_output_eye_size"] == "4x2"
    assert result.debug_info["runtime_output_display_size"] == "8x2"
    assert result.debug_info["runtime_output_pack_backend"] == "split_half_sbs"
    assert result.output_format == "openxr_full_synthesis_eyes"
    assert result.output_dtype == "uint8"
    assert result.output_eye_size == (4, 2)
    assert result.output_display_size == (8, 2)
    assert result.output_pack_backend == "split_half_sbs"


def test_openxr_rgb_depth_debug_info_carries_stereo_scale_and_max_shift():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(
        rgb,
        OpenXRRenderConfig(ipd=0.064, stereo_scale=0.35, depth_strength=2.0, max_shift_ratio=0.0),
    )

    assert result.legacy_shader_uniforms == {
        "ipd": 0.064,
        "depth_strength": 2.0,
        "stereo_scale": 0.35,
        "max_shift_ratio": 0.0,
        "convergence": 0.0,
        "max_disparity_px": None,
        "parallax_preset": "standard",
    }
    assert result.debug_info["openxr_ipd"] == 0.064
    assert result.debug_info["openxr_stereo_scale"] == 0.35
    assert result.debug_info["openxr_max_shift_ratio"] == 0.0
    assert result.debug_info["openxr_legacy_shader_uniforms"] == result.legacy_shader_uniforms


def test_openxr_rgb_depth_debug_info_records_resolved_max_disparity_px():
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(
        rgb,
        OpenXRRenderConfig(
            depth_strength=2.0,
            convergence=0.0,
            ipd_mm=32.0,
            stereo_scale=0.4,
            max_shift_ratio=0.05,
            max_disparity_px=18.0,
            parallax_preset="standard",
        ),
    )

    assert result.debug_info["resolved_max_disparity_px"] == 18.0
    assert result.debug_info["parallax_budget_preset"] == "standard"
    assert result.debug_info["depth_response"] == "linear_clamp_convergence_v1"
    assert result.debug_info["parallax_resolver_version"] == 1
    assert result.legacy_shader_uniforms["max_disparity_px"] == 18.0
    assert result.legacy_shader_uniforms["parallax_preset"] == "standard"
    assert result.debug_info["openxr_max_disparity_px"] == 18.0
    assert result.debug_info["openxr_parallax_preset"] == "standard"


def test_openxr_rgb_depth_viewer_keeps_ipd_separate_from_stereo_scale():
    source = (ROOT / "src" / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")

    assert "output_format = getattr(runtime_result, 'output_format', None) or debug_info.get('runtime_output_format')" in source
    assert "if output_format == 'openxr_rgb_depth':" in source
    assert "legacy_shader_uniforms=getattr(runtime_result, 'legacy_shader_uniforms', None)" in source
    assert 'debug_info.get("openxr_legacy_shader_uniforms")' in source
    assert 'self.ipd_uv = max(0.0, float(uniforms["ipd"]))' in source
    assert 'debug_info.get("openxr_stereo_scale", 1.0)' in source
    assert 'max_shift_ratio = uniforms.get("max_shift_ratio", debug_info.get("openxr_max_shift_ratio", 0.05))' in source
    assert "self._runtime_rgb_depth_max_shift_ratio = max(0.0, float(max_shift_ratio))" in source
    assert 'float(debug_info["openxr_ipd"])) * max(0.0, stereo_scale)' not in source


def test_process_openxr_frame_rgb_depth_applies_temporal_depth_stabilization(monkeypatch):
    monkeypatch.delenv("D2S_OPENXR_PREWARP_EYES", raising=False)
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "0")
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_GAMMA", "1.0")
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.9")

    class VaryingDepthProvider(FakeDepthProvider):
        def __init__(self):
            self.calls = 0
            self.info = FakeDepthProvider.info

        def load(self):
            self.loaded = True

        def predict_profile(self, rgb):
            self.calls += 1
            value = 0.0 if self.calls == 1 else 1.0
            depth = torch.full((1, 1, rgb.shape[-2], rgb.shape[-1]), value, dtype=rgb.dtype, device=rgb.device)
            return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)

    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=VaryingDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    first = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())
    second = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert first.debug_info["runtime_output_format"] == "openxr_rgb_depth"
    assert second.debug_info["runtime_output_format"] == "openxr_rgb_depth"
    assert torch.allclose(first.depth, torch.zeros_like(first.depth))
    assert torch.allclose(second.depth, torch.full_like(second.depth, 0.1))


def test_process_openxr_frame_rgb_depth_applies_gamma_curve(monkeypatch):
    monkeypatch.delenv("D2S_OPENXR_PREWARP_EYES", raising=False)
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "0")
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_GAMMA", "2.0")
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.0")

    class ConstantDepthProvider(FakeDepthProvider):
        def predict_profile(self, rgb):
            depth = torch.full((1, 1, rgb.shape[-2], rgb.shape[-1]), 0.5, dtype=rgb.dtype, device=rgb.device)
            return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)

    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=ConstantDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert result.debug_info["runtime_output_format"] == "openxr_rgb_depth"
    assert result.debug_info["packing_format"] == "none"
    assert torch.allclose(result.depth, torch.full_like(result.depth, 0.25))


def test_process_openxr_frame_rgb_depth_applies_percentile_normalization(monkeypatch):
    monkeypatch.delenv("D2S_OPENXR_PREWARP_EYES", raising=False)
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "20")
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_GAMMA", "1.0")
    monkeypatch.setenv("D2S_OPENXR_RGB_DEPTH_TEMPORAL_ALPHA", "0.0")

    class RampDepthProvider(FakeDepthProvider):
        def predict_profile(self, rgb):
            values = torch.tensor([0.0, 0.1, 0.2, 1.0], dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, 4)
            depth = values.expand(1, 1, rgb.shape[-2], 4).contiguous()
            return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)

    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=RampDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 1, 4)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    row = result.depth[0, 0, 0]
    expected = torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=row.dtype)
    assert torch.allclose(row, expected, atol=1e-5)


def test_openxr_rgb_depth_percentile_default_is_disabled_to_avoid_double_normalize():
    source = (ROOT / "src" / "stereo_runtime" / "runtime.py").read_text(encoding="utf-8")

    assert 'os.environ.get("D2S_OPENXR_RGB_DEPTH_PERCENTILE", "0")' in source


def test_process_openxr_frame_can_pack_prewarped_eye_views_to_uint8(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_PREWARP_EYES", "1")
    monkeypatch.setenv("D2S_RUNTIME_OUTPUT_UINT8", "1")
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert result.left_eye.dtype == torch.uint8
    assert result.right_eye.dtype == torch.uint8
    assert result.debug_info["runtime_output_dtype"] == "uint8"
    assert result.debug_info["runtime_output_pack_backend"] == "torch_float_eye_to_uint8"
    assert "pack_ms" in result.timing


def test_process_openxr_frame_can_keep_prewarped_float_eye_views_when_disabled(monkeypatch):
    monkeypatch.setenv("D2S_OPENXR_PREWARP_EYES", "1")
    monkeypatch.setenv("D2S_OPENXR_RUNTIME_OUTPUT_UINT8", "0")
    config = StereoRuntimeConfig(
        model_id="Distill-Any-Depth-Base",
        cache_dir="models",
        device="cpu",
        depth_backend="pytorch_cuda",
    )
    runtime = StereoRuntime(config, depth_provider=FakeDepthProvider(), collect_memory_stats=False)
    rgb = torch.rand(1, 3, 12, 16)

    result = runtime.process_openxr_frame(rgb, OpenXRRenderConfig())

    assert result.left_eye.dtype == torch.float32
    assert result.right_eye.dtype == torch.float32
    assert result.debug_info["runtime_output_dtype"] == "float32"
    assert result.debug_info["runtime_output_pack_backend"] == "none"
