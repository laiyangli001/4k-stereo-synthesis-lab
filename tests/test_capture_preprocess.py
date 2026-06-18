import numpy as np
import pytest

from capture import capture_frame_to_rgb, prepare_rgb_for_depth_runtime


def test_capture_frame_to_rgb_converts_bgr_numpy():
    frame = np.array([[[10, 20, 30], [40, 50, 60]], [[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)

    rgb = capture_frame_to_rgb(frame, target_height=2)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_capture_frame_to_rgb_converts_bgra_numpy():
    frame = np.array(
        [
            [[10, 20, 30, 255], [40, 50, 60, 128]],
            [[10, 20, 30, 255], [40, 50, 60, 128]],
        ],
        dtype=np.uint8,
    )

    rgb = capture_frame_to_rgb(frame, target_height=2)

    assert rgb.shape == (2, 2, 3)
    assert rgb[0, 0].tolist() == [30, 20, 10]


def test_capture_frame_to_rgb_keeps_even_resize_dimensions():
    frame = np.zeros((3, 5, 3), dtype=np.uint8)

    rgb = capture_frame_to_rgb(frame, target_height=5)

    assert rgb.shape[0] % 2 == 0
    assert rgb.shape[1] % 2 == 0


def test_prepare_rgb_for_depth_runtime_accepts_numpy_hwc():
    torch = pytest.importorskip("torch")
    frame = np.full((2, 3, 3), 255, dtype=np.uint8)

    tensor = prepare_rgb_for_depth_runtime(frame, device="cpu")

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 2, 3)
    assert float(tensor.max()) == 1.0


def test_prepare_rgb_for_depth_runtime_accepts_chw_and_bchw():
    torch = pytest.importorskip("torch")
    chw = torch.ones((3, 2, 4), dtype=torch.float32)
    bchw = torch.ones((1, 3, 2, 4), dtype=torch.float32)

    assert prepare_rgb_for_depth_runtime(chw, device="cpu").shape == (3, 2, 4)
    assert prepare_rgb_for_depth_runtime(bchw, device="cpu").shape == (1, 3, 2, 4)


def test_prepare_rgb_for_depth_runtime_does_not_rescale_normalized_input():
    torch = pytest.importorskip("torch")
    frame = torch.full((3, 2, 2), 0.5, dtype=torch.float32)

    tensor = prepare_rgb_for_depth_runtime(frame, device="cpu")

    assert float(tensor.max()) == pytest.approx(0.5)
