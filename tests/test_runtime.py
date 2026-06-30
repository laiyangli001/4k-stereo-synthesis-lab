import sys
from pathlib import Path

import logging
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime import StereoRuntime, StereoRuntimeConfig
from stereo_runtime.depth_provider import DepthProfileResult, DepthProviderInfo
from stereo_runtime.runtime import RollingRuntimeStats


class FakeDepthProvider:
    def __init__(self) -> None:
        self.load_count = 0
        self.predict_count = 0
        self.close_count = 0
        self.info = DepthProviderInfo(
            provider="fake",
            model_name="fake-depth",
            model_id="fake/model",
            depth_resolution=2,
            cache_dir=".",
            depth_backend="fake",
            runtime="test",
        )

    def load(self) -> None:
        self.load_count += 1

    def predict_profile(self, rgb: torch.Tensor) -> DepthProfileResult:
        self.predict_count += 1
        b, _, h, w = rgb.shape
        depth = torch.linspace(0, 1, w, dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return DepthProfileResult(depth=depth, preprocess_ms=1.0, model_ms=2.0, postprocess_ms=3.0)

    def close(self) -> None:
        self.close_count += 1


def test_runtime_process_rgb_frame_uses_persistent_provider_and_returns_report():
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="half_sbs",
        temporal=False,
        max_disparity_px=18.0,
        parallax_preset="comfort",
    )
    runtime = StereoRuntime(config, depth_provider=provider, stats_window=4, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    first = runtime.process_rgb_frame(rgb)
    second = runtime.process_rgb_frame(rgb)

    assert provider.load_count == 1
    assert provider.predict_count == 2
    assert first.depth.shape == (1, 1, 8, 12)
    assert first.left_eye.shape == rgb.shape
    assert first.right_eye.shape == rgb.shape
    assert first.sbs.shape == rgb.shape
    assert first.timing["depth_preprocess_ms"] == 1.0
    assert first.timing["depth_model_ms"] == 2.0
    assert first.timing["depth_postprocess_ms"] == 3.0
    assert first.timing["synthesis_ms"] >= 0.0
    assert first.debug_info["runtime_depth_backend"] == "pytorch_cuda"
    assert first.debug_info["runtime_output_format"] == "half_sbs"
    assert first.debug_info["runtime_quality_mode"] == "fast"
    assert first.debug_info["stereo_synthesis_mode"] == "packed_synthesis"
    assert first.debug_info["output_format"] == "half_sbs"
    assert first.debug_info["max_disparity_px"] == 18.0
    assert first.debug_info["parallax_preset"] == "comfort"
    assert first.debug_info["depth_provider_size"] == "2x2"
    assert first.debug_info["depth_render_size"] == "12x8"
    assert first.provider_info["provider"] == "fake"
    assert second.timing["total_ms"] >= 0.0

    report = runtime.to_report()
    assert report["depth_backend_resolved"] == "pytorch_cuda"
    assert report["stereo_backend"] == "fast"
    assert report["depth_provider"]["provider"] == "fake"
    assert "last_timing" in report
    assert report["rolling_stats"]["count"] == 2
    assert report["rolling_stats"]["stages"]["depth_model_ms"]["mean"] == 2.0
    assert report["rolling_stats"]["fps"]["latest"] > 0.0

    runtime.close()
    assert provider.close_count == 1


def test_runtime_slow_frame_log_is_debug_only(monkeypatch, caplog, capsys):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    monkeypatch.setenv("D2S_SLOW_RUNTIME_LOG_MS", "0")
    with caplog.at_level(logging.DEBUG, logger="stereo_runtime.runtime"):
        runtime.process_rgb_frame(rgb)

    assert "[StereoRuntime] slow frame:" in caplog.text
    assert "[StereoRuntime] slow frame:" not in capsys.readouterr().out


def test_runtime_frame_refresh_debug_log_is_capped(monkeypatch, caplog):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast",
        output_format="half_sbs",
        temporal=False,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 8, 12)

    monkeypatch.setenv("D2S_SLOW_RUNTIME_LOG_MS", "999999")
    monkeypatch.setenv("D2S_RUNTIME_FRAME_LOG_REFRESH_S", "0.001")
    with caplog.at_level(logging.DEBUG, logger="stereo_runtime.runtime"):
        for _ in range(7):
            runtime._last_runtime_perf_log_ts = -1e9
            runtime.process_rgb_frame(rgb)

    assert caplog.text.count("[StereoRuntime] frame refresh:") == 5

def test_rolling_runtime_stats_reports_percentiles_fps_and_memory():
    stats = RollingRuntimeStats(maxlen=3)
    stats.update({"total_ms": 10.0, "synthesis_ms": 4.0}, {"cuda_peak_memory_allocated_mb": 100.0})
    stats.update({"total_ms": 20.0, "synthesis_ms": 5.0}, {"cuda_peak_memory_allocated_mb": 150.0})
    stats.update({"total_ms": 30.0, "synthesis_ms": 6.0}, {"cuda_peak_memory_allocated_mb": 200.0})
    stats.update({"total_ms": 40.0, "synthesis_ms": 7.0}, {"cuda_peak_memory_allocated_mb": 250.0})

    report = stats.to_report()

    assert report["count"] == 3
    assert report["stages"]["total_ms"]["min"] == 20.0
    assert report["stages"]["total_ms"]["max"] == 40.0
    assert report["stages"]["total_ms"]["mean"] == 30.0
    assert report["stages"]["total_ms"]["median"] == 30.0
    assert report["stages"]["total_ms"]["p90"] == 38.0
    assert report["stages"]["total_ms"]["p99"] == pytest.approx(39.8)
    assert report["fps"]["latest"] == 25.0
    assert report["fps"]["mean_from_mean_ms"] == pytest.approx(1000.0 / 30.0)
    assert report["memory"]["cuda_peak_memory_allocated_mb"]["max"] == 250.0

    stats.reset()
    assert stats.to_report()["count"] == 0


def test_stereo_runtime_exports_new_public_names():
    from stereo_runtime import StereoRuntime, StereoRuntimeConfig
    from stereo_runtime.depth_provider import DepthProviderConfig
    from stereo_runtime.runtime import RollingRuntimeStats as AliasRollingStats

    assert StereoRuntime.__name__ == "StereoRuntime"
    assert StereoRuntimeConfig.__name__ == "StereoRuntimeConfig"
    assert DepthProviderConfig.__name__ == "DepthProviderConfig"
    assert AliasRollingStats is RollingRuntimeStats


def test_fast_plus_fused_uses_resolved_parallax_budget_contract():
    fused_source = (ROOT / "src" / "stereo_runtime" / "fast_plus_fused_triton.py").read_text(encoding="utf-8")
    runtime_source = (ROOT / "src" / "stereo_runtime" / "runtime.py").read_text(encoding="utf-8")

    assert "max_disparity_px: tl.constexpr" in fused_source
    assert "max_disparity_px: float" in fused_source
    assert "depth_strength: tl.constexpr" in fused_source
    assert "depth_strength: float" in fused_source
    assert "max_disparity_px * depth_strength * 0.5" in fused_source
    assert "effective_ipd_m" not in fused_source
    assert "max_shift_ratio" not in fused_source
    assert "width *" not in fused_source
    assert "max_disparity_px=float(budget.max_disparity_px)" in runtime_source
    assert 'depth_strength=max(0.0, float(getattr(stereo_config, "depth_strength", 1.0)))' in runtime_source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for fast_plus_fused Triton")
def test_fast_plus_fused_runtime_emits_uint8_half_sbs(monkeypatch):
    provider = FakeDepthProvider()
    config = StereoRuntimeConfig(
        model_id="lc700x/Distill-Any-Depth-Base-hf",
        model_dir=r"D:\Desktop2Stereo\models\models--lc700x--Distill-Any-Depth-Base-hf",
        depth_backend="pytorch_cuda",
        stereo_quality="fast_plus",
        output_format="half_sbs",
        temporal=True,
    )
    runtime = StereoRuntime(config, depth_provider=provider, collect_memory_stats=False)
    rgb = torch.rand(1, 3, 16, 32, device="cuda", dtype=torch.float32)

    monkeypatch.setenv("D2S_RUNTIME_OUTPUT_UINT8", "1")
    monkeypatch.setenv("D2S_FAST_PLUS_FUSED", "1")
    result = runtime.process_rgb_frame(rgb)
    torch.cuda.synchronize()

    assert result.sbs.shape == rgb.shape
    assert result.sbs.dtype == torch.uint8
    assert result.debug_info["runtime_output_pack_backend"] == "triton_half_sbs_uint8"
    assert result.debug_info["fast_plus_fused_backend"] == "triton_half_sbs_uint8"
    assert result.debug_info["fast_plus_fused_temporal_bypass"] == 1
