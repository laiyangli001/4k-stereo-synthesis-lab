import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab.hole_fill import box_blur, edge_aware_fill
from stereo_lab.baseline_shift import ShiftParams, compute_shift_px, make_base_grid, warp_horizontal
from stereo_lab.layers import composite_layers, depth_edges, make_depth_layers
from stereo_lab.occlusion import make_occlusion_mask
from stereo_lab.synthesis import StereoConfig, _try_fused_warp_composite2, synthesize_stereo
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


def test_edge_aware_fill_cuda_triton_matches_fallback_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(17)
    image = torch.rand(2, 3, 32, 40, device="cuda")
    mask = (torch.rand(2, 1, 32, 40, device="cuda") > 0.7).float()
    expected = torch.lerp(image, box_blur(image, radius=3), mask)
    actual = edge_aware_fill(image, mask, radius=3, strength=1.0)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_depth_edges_matches_padded_gradient_formula():
    torch.manual_seed(11)
    depth = torch.rand(2, 1, 12, 14)
    threshold = 0.04
    dx = F.pad((depth[..., :, 1:] - depth[..., :, :-1]).abs(), (0, 1, 0, 0))
    dy = F.pad((depth[..., 1:, :] - depth[..., :-1, :]).abs(), (0, 0, 0, 1))
    expected = ((dx + dy) > threshold).float()
    actual = depth_edges(depth, threshold=threshold)
    assert torch.equal(actual, expected)


def test_composite_layers_two_layer_fast_path_matches_formula():
    torch.manual_seed(13)
    warped = [torch.rand(2, 3, 8, 10), torch.rand(2, 3, 8, 10)]
    weights = torch.rand(2, 2, 8, 10)
    expected = warped[0] * weights[:, 0:1] + warped[1] * weights[:, 1:2]
    actual = composite_layers(warped, weights)
    assert torch.equal(actual, expected)


def test_fused_warp_composite_cuda_matches_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    rgb, depth = make_inputs(width=40, height=24)
    rgb = rgb.cuda()
    depth = depth.cuda()
    base_shift = compute_shift_px(depth, rgb.shape[-1], ShiftParams())
    weights = make_depth_layers(depth, layers=2)
    left_layers = []
    right_layers = []
    for idx in range(2):
        layer_shift = base_shift * (0.75 + 0.25 * (idx + 1) / 2)
        left_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=-1.0))
        right_layers.append(warp_horizontal(rgb, layer_shift, eye_sign=1.0))
    expected_left = composite_layers(left_layers, weights)
    expected_right = composite_layers(right_layers, weights)
    actual = _try_fused_warp_composite2(rgb, depth, base_shift, layers=2, symmetric=True)
    assert actual is not None
    actual_left, actual_right = actual
    assert torch.allclose(actual_left, expected_left, atol=5e-4, rtol=1e-4)
    assert torch.allclose(actual_right, expected_right, atol=5e-4, rtol=1e-4)


def test_fused_occlusion_cuda_matches_torch_path_when_available():
    if not torch.cuda.is_available():
        return
    rgb, depth = make_inputs(width=40, height=24)
    depth = depth.cuda()
    base_shift = compute_shift_px(depth, rgb.shape[-1], ShiftParams())
    expected = make_occlusion_mask(depth, base_shift, fused=False)
    actual = make_occlusion_mask(depth, base_shift, fused=True)
    assert torch.equal(actual, expected)


def test_fused_config_false_uses_torch_backends():
    rgb, depth = make_inputs(width=40, height=24)
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, debug_output=True, temporal=False, fused=False),
    )
    assert result.debug_info["warp_composite_backend"] == "torch_grid_sample"
    assert result.debug_info["occlusion_mask_backend"] == "torch_max_pool"
    assert result.debug_info["hole_fill_backend"] == "torch_avg_pool"


def test_disable_triton_env_uses_torch_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("STEREO_LAB_DISABLE_TRITON", "1")
    rgb, depth = make_inputs(width=40, height=24)
    result = synthesize_stereo(
        rgb,
        depth,
        StereoConfig(backend="quality_4k", layers=2, debug_output=True, temporal=False),
    )
    assert result.debug_info["warp_composite_backend"] == "torch_grid_sample"
    assert result.debug_info["occlusion_mask_backend"] == "torch_max_pool"
    assert result.debug_info["hole_fill_backend"] == "torch_avg_pool"


def test_warp_horizontal_matches_cached_grid_formula():
    rgb, depth = make_inputs(width=32, height=16)
    eye_sign = -1.0
    b, _, h, w = rgb.shape
    shift_px = depth * 0.75
    grid = make_base_grid(b, h, w, rgb.device, rgb.dtype).clone()
    shift_norm = (2.0 * shift_px.squeeze(1) / max(w - 1, 1)) * eye_sign
    grid[..., 0] = grid[..., 0] + shift_norm
    expected = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="border", align_corners=True)
    actual = warp_horizontal(rgb, shift_px, eye_sign=eye_sign)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
