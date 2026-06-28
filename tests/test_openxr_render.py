import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px, warp_horizontal
from stereo_runtime.openxr_render import (
    OpenXREyeView,
    OpenXRFov,
    OpenXRRenderConfig,
    OpenXRScreenPose,
    build_openxr_eye_mvp,
    build_openxr_screen_model_matrix,
    fov_to_projection_matrix,
    is_pyopenxr_available,
    pyopenxr_module_name,
    pose_to_view_matrix,
    render_openxr_eye,
    render_openxr_stereo,
)


def make_inputs(width=40, height=24):
    y = torch.linspace(0, 1, height)
    x = torch.linspace(0, 1, width)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    rgb = torch.stack([xx, yy, torch.ones_like(xx) * 0.5], dim=0).unsqueeze(0)
    depth = xx.unsqueeze(0).unsqueeze(0)
    return rgb, depth


def test_openxr_roll_zero_matches_horizontal_warp():
    rgb, depth = make_inputs()
    config = OpenXRRenderConfig(screen_roll=0.0, depth_strength=2.0, convergence=0.0)
    expected = warp_horizontal(
        rgb,
        compute_shift_px(depth, rgb.shape[-1], ShiftParams(depth_strength=2.0, convergence=0.0)),
        eye_sign=1.0,
    )
    actual = render_openxr_eye(rgb, depth, eye_sign=1.0, config=config)
    assert torch.equal(actual, expected)


def test_openxr_roll_rotates_parallax_direction():
    rgb, depth = make_inputs(width=16, height=16)
    config = OpenXRRenderConfig(screen_roll=math.pi / 2, depth_strength=2.0, convergence=0.0)

    actual = render_openxr_eye(rgb, depth, eye_sign=1.0, config=config)

    shift_px = compute_shift_px(depth, rgb.shape[-1], ShiftParams(depth_strength=2.0, convergence=0.0))
    y = torch.linspace(-1.0, 1.0, rgb.shape[-2])
    x = torch.linspace(-1.0, 1.0, rgb.shape[-1])
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    shift_y = (2.0 * shift_px.squeeze(1)) / max(rgb.shape[-2] - 1, 1)
    grid = torch.stack((xx.unsqueeze(0), yy.unsqueeze(0) + shift_y), dim=-1)
    expected = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    assert not torch.equal(actual, render_openxr_eye(rgb, depth, eye_sign=1.0, config=OpenXRRenderConfig(screen_roll=0.0)))


def test_openxr_roll_accepts_arbitrary_angle():
    rgb, depth = make_inputs(width=16, height=16)
    roll = math.radians(37.0)
    config = OpenXRRenderConfig(screen_roll=roll, depth_strength=2.0, convergence=0.0)

    actual = render_openxr_eye(rgb, depth, eye_sign=1.0, config=config)

    shift_px = compute_shift_px(depth, rgb.shape[-1], ShiftParams(depth_strength=2.0, convergence=0.0))
    y = torch.linspace(-1.0, 1.0, rgb.shape[-2])
    x = torch.linspace(-1.0, 1.0, rgb.shape[-1])
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    shift_x = (2.0 * shift_px.squeeze(1) * math.cos(roll)) / max(rgb.shape[-1] - 1, 1)
    shift_y = (2.0 * shift_px.squeeze(1) * math.sin(roll)) / max(rgb.shape[-2] - 1, 1)
    grid = torch.stack((xx.unsqueeze(0) + shift_x, yy.unsqueeze(0) + shift_y), dim=-1)
    expected = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="reflection", align_corners=True)

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_openxr_stereo_result_shapes_and_debug():
    rgb, depth = make_inputs(width=33, height=21)
    result = render_openxr_stereo(rgb, depth, OpenXRRenderConfig(screen_roll=0.25))
    assert result.left_eye.shape == rgb.shape
    assert result.right_eye.shape == rgb.shape
    assert result.debug_info["backend"] == "openxr_roll_adaptive_grid_sample"
    assert result.debug_info["screen_roll"] == 0.25
    assert result.debug_info["parallax_budget_preset"] == "standard"
    assert result.debug_info["parallax_resolver_version"] == 1


def test_openxr_screen_model_matrix_uses_screen_pose():
    pose = OpenXRScreenPose(width_m=2.0, height_m=1.0, distance_m=3.0, pan_x_m=0.25, pan_y_m=-0.5)
    model = build_openxr_screen_model_matrix(pose)
    assert torch.allclose(model[:3, 3], torch.tensor([0.25, -0.5, -3.0]))
    assert torch.isclose(model[0, 0], torch.tensor(1.0))
    assert torch.isclose(model[1, 1], torch.tensor(0.5))


def test_openxr_pose_and_projection_matrices():
    view = pose_to_view_matrix((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
    assert torch.equal(view[:3, 3], torch.tensor([-1.0, -2.0, -3.0]))

    projection = fov_to_projection_matrix(-0.5, 0.5, 0.4, -0.4)
    assert projection.shape == (4, 4)
    assert projection[0, 0] > 0
    assert projection[1, 1] > 0
    assert projection[3, 2] == -1.0


def test_pyopenxr_detection_uses_xr_import_name():
    assert pyopenxr_module_name() == "xr"
    assert isinstance(is_pyopenxr_available(), bool)


def test_openxr_eye_mvp_combines_projection_view_and_screen_model():
    eye = OpenXREyeView(
        eye_index=0,
        position_xyz=(0.0, 0.0, 0.0),
        orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        fov=OpenXRFov(angle_left=-0.5, angle_right=0.5, angle_up=0.4, angle_down=-0.4),
    )
    screen = OpenXRScreenPose(width_m=2.0, height_m=1.0, distance_m=3.0)
    mvp = build_openxr_eye_mvp(eye, screen)
    expected = fov_to_projection_matrix(-0.5, 0.5, 0.4, -0.4) @ build_openxr_screen_model_matrix(screen)
    assert torch.allclose(mvp, expected)
