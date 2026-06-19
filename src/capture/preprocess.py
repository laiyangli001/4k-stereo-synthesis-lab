from __future__ import annotations

import numpy as np


def capture_frame_to_rgb(frame_raw, target_resolution, *, device=None, use_torch=None, output="auto"):
    """
    Convert a raw capture frame to the RGB frame consumed by stereo_runtime.

    This keeps capture-side responsibilities out of stereo runtime code:
    raw BGRA/BGR frame -> RGB frame, plus the existing processing-resolution
    resize used by the Desktop2Stereo pipeline.
    """
    if use_torch is None:
        use_torch = _is_torch_tensor(frame_raw)
    if use_torch or output == "tensor":
        return _capture_frame_to_rgb_torch(frame_raw, target_resolution, device=device)
    return _capture_frame_to_rgb_numpy(frame_raw, target_resolution)


def prepare_rgb_for_stereo_runtime(frame_rgb, *, device=None):
    """Prepare capture-owned RGB frame for stereo_runtime's strict RGB input contract."""
    import torch

    if device is None:
        from utils import DEVICE
        device = DEVICE

    scale_from_uint8 = False
    if isinstance(frame_rgb, np.ndarray):
        scale_from_uint8 = np.issubdtype(frame_rgb.dtype, np.integer)
        frame_rgb = torch.from_numpy(frame_rgb)
        if frame_rgb.ndim == 3 and frame_rgb.shape[-1] == 3:
            frame_rgb = frame_rgb.permute(2, 0, 1).contiguous()

    if not isinstance(frame_rgb, torch.Tensor):
        raise TypeError("frame_rgb must be a torch.Tensor or numpy.ndarray prepared by capture")

    if frame_rgb.ndim == 3 and frame_rgb.shape[0] == 3:
        tensor = frame_rgb
    elif frame_rgb.ndim == 4 and frame_rgb.shape[1] == 3:
        tensor = frame_rgb
    else:
        raise ValueError(f"frame_rgb must be RGB CHW/BCHW for stereo_runtime, got shape {tuple(frame_rgb.shape)}")

    scale_from_uint8 = scale_from_uint8 or not tensor.is_floating_point()
    tensor = tensor.to(device=device, dtype=torch.float32)
    if scale_from_uint8:
        tensor = tensor.mul_(1.0 / 255.0)
    return tensor.clamp_(0.0, 1.0)


def _is_torch_tensor(value):
    torch_type = getattr(type(value), "__module__", "")
    return torch_type.startswith("torch") and hasattr(value, "permute")


def _capture_frame_to_rgb_torch(frame_raw, target_resolution, *, device=None):
    import torch
    import torch.nn.functional as F

    if device is None:
        from utils import DEVICE
        device = DEVICE

    if isinstance(frame_raw, np.ndarray):
        frame_raw = torch.from_numpy(frame_raw).to(device)

    h0, w0 = frame_raw.shape[:2]
    new_height, new_width = _resolve_target_size(target_resolution, h0, w0)

    try:
        from capture.preprocess_triton import bgr_to_rgb_resize_norm, can_use_triton_preprocess

        if can_use_triton_preprocess(frame_raw):
            out = bgr_to_rgb_resize_norm(frame_raw, new_height, new_width)
            setattr(out, "_d2s_preprocess_backend", "triton_bgr_resize_norm")
            return out
    except Exception:
        pass

    frame_rgb = frame_raw[..., [2, 1, 0]].permute(2, 0, 1).contiguous()
    with torch.no_grad():
        frame_float = frame_rgb.to(device=device, dtype=torch.float32).mul_(1.0 / 255.0)
        if new_height == h0 and new_width == w0:
            setattr(frame_float, "_d2s_preprocess_backend", "torch_bgr_norm")
            return frame_float
        out = F.interpolate(
            frame_float.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
            antialias=new_height < h0 or new_width < w0,
        ).squeeze(0).clamp_(0.0, 1.0)
        setattr(out, "_d2s_preprocess_backend", "torch_bgr_resize_norm")
        return out


def _capture_frame_to_rgb_numpy(frame_raw, target_resolution):
    import cv2

    is_umat = isinstance(frame_raw, cv2.UMat)
    shape = frame_raw.get().shape if is_umat else frame_raw.shape
    h0, w0 = shape[:2]

    new_height, new_width = _resolve_target_size(target_resolution, h0, w0)

    if shape[2] == 4:
        frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGRA2RGB)
    elif shape[2] == 3:
        frame_rgb = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported capture frame channel count: {shape[2]}")

    if new_height == h0 and new_width == w0:
        return frame_rgb

    interpolation = cv2.INTER_AREA if new_height < h0 or new_width < w0 else cv2.INTER_CUBIC
    return cv2.resize(frame_rgb, (new_width, new_height), interpolation=interpolation)


def _resolve_target_size(target_resolution, source_height, source_width):
    if isinstance(target_resolution, (tuple, list)) and len(target_resolution) == 2:
        target_width = _even_size(target_resolution[0])
        target_height = _even_size(target_resolution[1])
        return target_height, target_width

    target_height = _even_size(target_resolution)
    target_width = int(source_width * target_height / source_height)
    target_width = _even_size(target_width)
    return target_height, target_width


def _even_size(value):
    return max(2, (int(value) // 2) * 2)


def prepare_rgb_for_depth_runtime(frame_rgb, *, device=None):
    """Compatibility alias for older host code."""
    return prepare_rgb_for_stereo_runtime(frame_rgb, device=device)
