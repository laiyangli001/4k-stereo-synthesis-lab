import torch
from pathlib import Path

from stereo_runtime import OpenXRRenderConfig, OpenXRRuntimeResult, StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult


ROOT = Path(__file__).resolve().parents[1]


class FakeDepthProvider:
    info = {
        "provider": "fake",
        "model_name": "fake",
        "model_id": "fake",
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
    assert result.debug_info["backend"] == "openxr_viewer_shader_dibr"


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

    assert result.debug_info["openxr_ipd"] == 0.064
    assert result.debug_info["openxr_stereo_scale"] == 0.35
    assert result.debug_info["openxr_max_shift_ratio"] == 0.0


def test_openxr_rgb_depth_viewer_keeps_ipd_separate_from_stereo_scale():
    source = (ROOT / "src" / "xr_viewer" / "core_runtime_eye.py").read_text(encoding="utf-8")

    assert 'self.ipd_uv = max(0.0, float(debug_info["openxr_ipd"]))' in source
    assert "self._runtime_rgb_depth_stereo_scale = max(0.0, float(debug_info.get(\"openxr_stereo_scale\", 1.0)))" in source
    assert "self._runtime_rgb_depth_max_shift_ratio = max(0.0, float(debug_info.get(\"openxr_max_shift_ratio\", 0.05)))" in source
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
