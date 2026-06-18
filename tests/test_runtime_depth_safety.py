import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.adapter import StereoRuntimeConfig
from stereo_runtime.runtime import StereoRuntime


def test_image_runtime_applies_depth_safety_by_default(tmp_path):
    runtime = StereoRuntime(
        StereoRuntimeConfig(
            model_id="Distill-Any-Depth-Base",
            model_dir=tmp_path,
            mode="image",
            device="cpu",
            stereo_quality="fast",
            output_format="half_sbs",
            temporal=False,
        ),
        depth_provider=_FakeDepthProvider(_center_bias_depth()),
        collect_memory_stats=False,
    )

    result = runtime.process_rgb_frame(torch.full((1, 3, 32, 32), 0.1))

    assert result.debug_info["depth_safety"]["force_flat_depth"] is True
    assert float(result.depth.var(unbiased=False)) < 1e-6


def test_movie_runtime_does_not_apply_depth_safety_by_default(tmp_path):
    depth = _center_bias_depth()
    runtime = StereoRuntime(
        StereoRuntimeConfig(
            model_id="Distill-Any-Depth-Base",
            model_dir=tmp_path,
            mode="movie",
            device="cpu",
            stereo_quality="fast",
            output_format="half_sbs",
            temporal=False,
        ),
        depth_provider=_FakeDepthProvider(depth),
        collect_memory_stats=False,
    )

    result = runtime.process_rgb_frame(torch.full((1, 3, 32, 32), 0.1))

    assert "depth_safety" not in result.debug_info
    assert torch.allclose(result.depth, depth)


class _FakeDepthProvider:
    def __init__(self, depth: torch.Tensor) -> None:
        self.depth = depth

    def load(self) -> None:
        pass

    def predict(self, rgb_frame: torch.Tensor) -> torch.Tensor:
        return self.depth.to(rgb_frame.device)

    def info(self):
        return {"backend": "fake"}


def _center_bias_depth() -> torch.Tensor:
    y = torch.linspace(-1, 1, 32).view(1, 1, 32, 1)
    x = torch.linspace(-1, 1, 32).view(1, 1, 1, 32)
    radius = (x.square() + y.square()).sqrt().clamp(0, 1)
    return 0.75 - radius * 0.35
