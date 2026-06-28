from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px

DEVICE_INFO = "unknown"

_FONT_CACHE = {}
_FPS_MASK_CACHE = {"mask": None, "frame": 0, "interval": 8}

_FONT = {
    " ": ["000", "000", "000", "000", "000"],
    ".": ["000", "000", "000", "000", "010"],
    ":": ["000", "010", "000", "010", "000"],
    "0": ["111", "101", "101", "101", "111"],
    "1": ["010", "110", "010", "010", "111"],
    "2": ["111", "001", "111", "100", "111"],
    "3": ["111", "001", "111", "001", "111"],
    "4": ["101", "101", "111", "001", "001"],
    "5": ["111", "100", "111", "001", "111"],
    "6": ["111", "100", "111", "101", "111"],
    "7": ["111", "001", "001", "001", "001"],
    "8": ["111", "101", "111", "101", "111"],
    "9": ["111", "101", "111", "001", "111"],
    "F": ["111", "100", "111", "100", "100"],
    "P": ["111", "101", "111", "100", "100"],
    "S": ["111", "100", "111", "001", "111"],
}


def make_sbs(
    rgb_c,
    depth,
    parallax_budget_uv=0.064,
    depth_strength=2.0,
    convergence=0.0,
    fill_16_9=False,
    display_mode="Half-SBS",
    fps=None,
):
    """Legacy depth-shift SBS output for the old MJPEG stream path."""
    device = _default_device(depth)
    if isinstance(depth, np.ndarray):
        depth_t = torch.from_numpy(depth).to(device=device)
    else:
        depth_t = depth.to(device=device)

    if isinstance(rgb_c, np.ndarray):
        rgb = torch.from_numpy(rgb_c).to(device=depth_t.device, dtype=depth_t.dtype)
        if rgb.ndim == 3 and rgb.shape[2] == 3:
            rgb = rgb.permute(2, 0, 1).contiguous()
    else:
        rgb = rgb_c.to(device=depth_t.device, dtype=depth_t.dtype)

    if rgb.dtype == torch.uint8:
        rgb = rgb.to(dtype=torch.float32)
    if rgb.max() <= 1.5:
        rgb = rgb * 255.0

    if fps is not None:
        rgb = _overlay_fps(rgb, fps)

    if depth_t.ndim == 4:
        depth_t = depth_t[0, 0]
    elif depth_t.ndim == 3:
        depth_t = depth_t[0]

    sbs_tensor = _make_sbs_core(
        rgb=rgb,
        depth=depth_t,
        parallax_budget_uv=parallax_budget_uv,
        depth_strength=depth_strength,
        convergence=convergence,
        fill_16_9=fill_16_9,
        display_mode=display_mode,
    )
    return _chw_tensor_to_numpy(sbs_tensor)


def _make_sbs_core(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    parallax_budget_uv=0.064,
    depth_strength=2.0,
    convergence=0.0,
    fill_16_9=False,
    display_mode="Half-SBS",
) -> torch.Tensor:
    c, h, w = rgb.shape
    img = rgb.unsqueeze(0).clamp(0, 255)
    max_disparity_px = max(0.0, float(parallax_budget_uv)) * float(w)
    shifts = compute_shift_px(
        depth.view(1, 1, h, w),
        w,
        ShiftParams(
            depth_strength=depth_strength,
            convergence=convergence,
            max_disparity_px=max_disparity_px,
        ),
    )[0, 0]

    xs = torch.linspace(-1.0, 1.0, w, device=rgb.device, dtype=rgb.dtype).view(1, 1, w).expand(1, h, w)
    ys = torch.linspace(-1.0, 1.0, h, device=rgb.device, dtype=rgb.dtype).view(1, h, 1).expand(1, h, w)
    shift_norm = shifts.to(dtype=rgb.dtype) * (2.0 / max(w - 1, 1))
    grid_left = torch.stack([xs + shift_norm, ys], dim=-1)
    grid_right = torch.stack([xs - shift_norm, ys], dim=-1)
    left = F.grid_sample(img, grid_left, mode="bilinear", padding_mode="reflection", align_corners=True)[0]
    right = F.grid_sample(img, grid_right, mode="bilinear", padding_mode="reflection", align_corners=True)[0]

    if fill_16_9:
        left = _pad_to_aspect_tensor(left)
        right = _pad_to_aspect_tensor(right)

    if display_mode in ["Half-TAB", "Full-TAB"]:
        out = torch.cat([left, right], dim=1)
    else:
        out = torch.cat([left, right], dim=2)
    if display_mode not in ["Full-SBS", "Full-TAB"]:
        out = F.interpolate(out.unsqueeze(0), size=left.shape[1:], mode="area")[0]
    return out.clamp(0, 255)


def _pad_to_aspect_tensor(tensor: torch.Tensor, target_aspect: float = 16 / 9) -> torch.Tensor:
    _, h, w = tensor.shape
    current_aspect = w / h
    if abs(current_aspect - target_aspect) < 1e-3:
        return tensor
    if current_aspect > target_aspect:
        new_h = int(round(w / target_aspect))
        pad_total = max(0, new_h - h)
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top
        return F.pad(tensor, (0, 0, pad_top, pad_bottom), mode="constant", value=0)
    new_w = int(round(h * target_aspect))
    pad_total = max(0, new_w - w)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return F.pad(tensor, (pad_left, pad_right, 0, 0), mode="constant", value=0)


def _chw_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().clamp(0, 255).to(torch.uint8)
    return tensor.permute(1, 2, 0).cpu().numpy()


def _default_device(depth) -> torch.device:
    if hasattr(depth, "device"):
        return torch.device(depth.device)
    try:
        from utils import DEVICE

        return torch.device(DEVICE)
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_font(device, dtype):
    key = (str(device), dtype)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    chars = sorted(_FONT.keys())
    font_tensor = torch.stack(
        [
            torch.tensor(
                [[1.0 if col == "1" else 0.0 for col in row] for row in _FONT[ch]],
                device=device,
                dtype=dtype,
            )
            for ch in chars
        ]
    )
    _FONT_CACHE[key] = (chars, font_tensor)
    return chars, font_tensor


def _overlay_fps(rgb: torch.Tensor, fps: float):
    device, dtype = rgb.device, rgb.dtype
    h, w = rgb.shape[1:]
    cache = _FPS_MASK_CACHE
    cache["frame"] += 1
    if cache["mask"] is None or cache["frame"] % cache["interval"] == 0 or tuple(cache["mask"].shape) != (h, w):
        chars, font_tensor = _build_font(device, dtype)
        text = f"FPS: {fps:.1f}"
        idxs = torch.tensor([chars.index(ch) if ch in chars else chars.index(" ") for ch in text], device=device)
        scale = max(1, min(8, h // 60))
        char_h, char_w = 5 * scale, 3 * scale
        spacing = scale
        margin_x, margin_y = 2 * scale, 2 * scale
        glyphs = font_tensor[idxs].repeat_interleave(scale, 1).repeat_interleave(scale, 2)
        mask = torch.zeros((h, w), device=device, dtype=dtype)
        for i, glyph in enumerate(glyphs):
            x0 = margin_x + i * (char_w + spacing)
            y0 = margin_y
            x1 = min(w, x0 + char_w)
            y1 = min(h, y0 + char_h)
            if x0 < w and y0 < h:
                mask[y0:y1, x0:x1] = torch.maximum(mask[y0:y1, x0:x1], glyph[: y1 - y0, : x1 - x0])
        cache["mask"] = mask
    alpha = cache["mask"].unsqueeze(0)
    color = torch.tensor([0.0, 255.0, 0.0], device=device, dtype=dtype).view(3, 1, 1)
    return rgb * (1 - alpha) + color * alpha
