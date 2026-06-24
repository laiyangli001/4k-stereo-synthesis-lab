from pathlib import Path

import torch

from stereo_runtime.openxr_visual_regression import (
    OpenXRViewerShaderParams,
    compare_tensors,
    make_depth_proxy_from_rgb,
    make_visual_regression_inputs,
    render_viewer_shader_eye_cpu,
    run_openxr_visual_regression,
)
from stereo_runtime.io import save_rgb


def test_openxr_viewer_shader_regression_matches_beta_when_scale_is_one():
    rgb, depth = make_visual_regression_inputs(width=96, height=54)
    current = OpenXRViewerShaderParams(stereo_scale=1.0, use_stereo_scale=True)
    beta = OpenXRViewerShaderParams(stereo_scale=1.0, use_stereo_scale=False)

    cur_eye = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=current)
    beta_eye = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=beta)

    assert torch.allclose(cur_eye, beta_eye, atol=1e-7, rtol=1e-7)


def test_openxr_viewer_shader_regression_can_compare_resolution_modes():
    rgb, depth = make_visual_regression_inputs(width=96, height=54)
    source = OpenXRViewerShaderParams(stereo_scale=1.0, shader_resolution_mode="source")
    swapchain = OpenXRViewerShaderParams(
        stereo_scale=1.0,
        shader_resolution_mode="swapchain",
        swapchain_width=192,
        swapchain_height=192,
    )

    source_eye = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=source)
    swapchain_eye = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=swapchain)

    assert not torch.equal(source_eye, swapchain_eye)


def test_openxr_viewer_shader_regression_exposes_stereo_scale_delta():
    rgb, depth = make_visual_regression_inputs(width=96, height=54)
    current = OpenXRViewerShaderParams(stereo_scale=0.35, use_stereo_scale=True)
    beta = OpenXRViewerShaderParams(stereo_scale=0.35, use_stereo_scale=False)

    cur_eye = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=current)
    beta_eye = render_viewer_shader_eye_cpu(rgb, depth, eye_sign=1.0, params=beta)
    metrics = compare_tensors(cur_eye, beta_eye)

    assert metrics["mae"] > 0.001
    assert metrics["pct_gt_1_255"] > 0.01


def test_openxr_visual_regression_writes_outputs(tmp_path: Path):
    metrics = run_openxr_visual_regression(output_dir=tmp_path)

    assert (tmp_path / "scaled_source_left.png").exists()
    assert (tmp_path / "scaled_swapchain_left.png").exists()
    assert (tmp_path / "beta_ipd_direct_left.png").exists()
    assert (tmp_path / "beta_direct_swapchain_left.png").exists()
    assert (tmp_path / "diff_scaled_source_left_heatmap.png").exists()
    assert (tmp_path / "diff_scaled_swapchain_left_heatmap.png").exists()
    assert "scaled_source_vs_beta" in metrics
    assert "scaled_swapchain_vs_beta" in metrics
    assert "beta_direct_source_vs_beta" in metrics
    assert "beta_direct_swapchain_vs_beta" in metrics
    assert "ranking_by_mean_mae" in metrics
    assert metrics["ranking_by_mean_mae"][0]["variant"] == "beta_direct_source"
    assert metrics["ranking_by_mean_mae"][0]["mean_mae"] == 0.0


def test_openxr_visual_regression_can_generate_depth_proxy_from_real_rgb(tmp_path: Path):
    rgb, _ = make_visual_regression_inputs(width=64, height=36)
    rgb_path = tmp_path / "frame.png"
    save_rgb(rgb, rgb_path)

    metrics = run_openxr_visual_regression(output_dir=tmp_path / "out", rgb_path=rgb_path)

    assert (tmp_path / "out" / "source_rgb.png").exists()
    assert (tmp_path / "out" / "prepared_depth.png").exists()
    assert metrics["ranking_by_mean_mae"][0]["variant"] == "beta_direct_source"


def test_depth_proxy_from_rgb_matches_frame_shape():
    rgb, _ = make_visual_regression_inputs(width=64, height=36)
    depth = make_depth_proxy_from_rgb(rgb)

    assert depth.shape == (1, 1, 36, 64)
    assert float(depth.min()) >= 0.0
    assert float(depth.max()) <= 1.0
