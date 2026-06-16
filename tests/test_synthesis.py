import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab.hole_fill import box_blur
from stereo_lab.synthesis import StereoConfig, synthesize_stereo
from stereo_lab.temporal import TemporalState


def make_inputs(width=64, height=32):
    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rgb = torch.stack([xx, yy, torch.ones_like(xx) * 0.5], dim=0).unsqueeze(0)
    depth = xx.unsqueeze(0).unsqueeze(0)
    return rgb, depth


def test_half_sbs_shape():
    rgb, depth = make_inputs()
    result = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="half_sbs"))
    assert result.left_eye.shape == rgb.shape
    assert result.right_eye.shape == rgb.shape
    assert result.sbs.shape == rgb.shape


def test_full_sbs_shape():
    rgb, depth = make_inputs()
    result = synthesize_stereo(rgb, depth, StereoConfig(backend="fast", output_format="full_sbs"))
    assert result.sbs.shape == (1, 3, 32, 128)


def test_quality_debug_outputs():
    rgb, depth = make_inputs()
    config = StereoConfig(backend="quality_4k", layers=2, output_format="half_sbs", debug_output=True)
    result = synthesize_stereo(rgb, depth, config)
    assert result.sbs.shape == rgb.shape
    assert "occlusion_mask" in result.debug_info
    assert result.debug_info["occlusion_mask"].shape == (1, 1, 32, 64)


def test_hq_uses_at_least_three_layers():
    rgb, depth = make_inputs()
    result = synthesize_stereo(rgb, depth, StereoConfig(backend="hq_4k", layers=2, debug_output=True))
    assert result.debug_info["layers"] == 3


def test_temporal_state_runs_twice():
    rgb, depth = make_inputs()
    state = TemporalState()
    config = StereoConfig(backend="quality_4k", temporal=True)
    first = synthesize_stereo(rgb, depth, config, temporal_state=state)
    second = synthesize_stereo(rgb, depth, config, temporal_state=state)
    assert first.sbs.shape == second.sbs.shape


def test_box_blur_matches_2d_kernel():
    torch.manual_seed(7)
    image = torch.rand(2, 3, 16, 18)
    radius = 3
    k = radius * 2 + 1
    weight = torch.ones(3, 1, k, k, dtype=image.dtype) / float(k * k)
    expected = F.conv2d(image, weight, padding=radius, groups=3)
    actual = box_blur(image, radius=radius)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
