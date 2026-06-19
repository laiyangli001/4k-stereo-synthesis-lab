from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _clamp_i32(value, lo: tl.constexpr, hi: tl.constexpr):
    return tl.minimum(tl.maximum(value, lo), hi)


@triton.jit
def _shift_from_depth(
    depth_value,
    width: tl.constexpr,
    depth_strength: tl.constexpr,
    convergence: tl.constexpr,
    effective_ipd_m: tl.constexpr,
    max_shift_ratio: tl.constexpr,
):
    depth_value = tl.minimum(tl.maximum(depth_value, 0.0), 1.0)
    max_px = width * effective_ipd_m * max_shift_ratio
    return -(depth_value - convergence) * depth_strength * max_px


@triton.jit
def _load_depth_at(depth, y, x, width: tl.constexpr, height: tl.constexpr):
    yy = _clamp_i32(y, 0, height - 1)
    xx = _clamp_i32(x, 0, width - 1)
    return tl.load(depth + yy * width + xx)


@triton.jit
def _sample_eye(
    rgb,
    depth,
    channel,
    y,
    target_x,
    eye_sign,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    depth_strength: tl.constexpr,
    convergence: tl.constexpr,
    effective_ipd_m: tl.constexpr,
    max_shift_ratio: tl.constexpr,
):
    tx_i = _clamp_i32(target_x, 0, width - 1)
    depth_value = _load_depth_at(depth, y, tx_i, width, height)
    shift_px = _shift_from_depth(depth_value, width, depth_strength, convergence, effective_ipd_m, max_shift_ratio)
    src_x = target_x.to(tl.float32) + shift_px * eye_sign
    src_x = tl.minimum(tl.maximum(src_x, 0.0), (width - 1) * 1.0)
    x0_f = tl.floor(src_x)
    x0 = x0_f.to(tl.int32)
    x1 = _clamp_i32(x0 + 1, 0, width - 1)
    frac = src_x - x0_f
    base = channel * pixels + y * width
    v0 = tl.load(rgb + base + x0)
    v1 = tl.load(rgb + base + x1)
    return v0 * (1.0 - frac) + v1 * frac


@triton.jit
def _mask_at(
    depth,
    y,
    x,
    width: tl.constexpr,
    height: tl.constexpr,
    depth_strength: tl.constexpr,
    convergence: tl.constexpr,
    effective_ipd_m: tl.constexpr,
    max_shift_ratio: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
):
    found = x < 0
    for dy in tl.static_range(-1, 2):
        yy = y + dy
        valid_y = (yy >= 0) & (yy < height)
        for dx in tl.static_range(-1, 2):
            xx = x + dx
            valid = valid_y & (xx >= 0) & (xx < width)
            center_depth = _load_depth_at(depth, yy, xx, width, height)
            right_depth = _load_depth_at(depth, yy, xx + 1, width, height)
            down_depth = _load_depth_at(depth, yy + 1, xx, width, height)
            center_shift = tl.abs(_shift_from_depth(center_depth, width, depth_strength, convergence, effective_ipd_m, max_shift_ratio))
            right_shift = tl.abs(_shift_from_depth(right_depth, width, depth_strength, convergence, effective_ipd_m, max_shift_ratio))
            down_shift = tl.abs(_shift_from_depth(down_depth, width, depth_strength, convergence, effective_ipd_m, max_shift_ratio))
            depth_edge = (tl.abs(right_depth - center_depth) + tl.abs(down_depth - center_depth)) > edge_threshold
            shift_edge = (tl.abs(right_shift - center_shift) + tl.abs(down_shift - center_shift)) > shift_edge_threshold_px
            found = found | (valid & (depth_edge | shift_edge))
    return found


@triton.jit
def _filled_eye(
    rgb,
    depth,
    channel,
    y,
    target_x,
    eye_sign,
    width: tl.constexpr,
    height: tl.constexpr,
    pixels: tl.constexpr,
    depth_strength: tl.constexpr,
    convergence: tl.constexpr,
    effective_ipd_m: tl.constexpr,
    max_shift_ratio: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
):
    value = _sample_eye(
        rgb,
        depth,
        channel,
        y,
        target_x,
        eye_sign,
        width,
        height,
        pixels,
        depth_strength,
        convergence,
        effective_ipd_m,
        max_shift_ratio,
    )
    mask = _mask_at(
        depth,
        y,
        target_x,
        width,
        height,
        depth_strength,
        convergence,
        effective_ipd_m,
        max_shift_ratio,
        edge_threshold,
        shift_edge_threshold_px,
    )
    tap1 = _sample_eye(rgb, depth, channel, y, target_x - 1, eye_sign, width, height, pixels, depth_strength, convergence, effective_ipd_m, max_shift_ratio)
    tap2 = _sample_eye(rgb, depth, channel, y, target_x + 1, eye_sign, width, height, pixels, depth_strength, convergence, effective_ipd_m, max_shift_ratio)
    tap3 = _sample_eye(rgb, depth, channel, y, target_x - 2, eye_sign, width, height, pixels, depth_strength, convergence, effective_ipd_m, max_shift_ratio)
    tap4 = _sample_eye(rgb, depth, channel, y, target_x + 2, eye_sign, width, height, pixels, depth_strength, convergence, effective_ipd_m, max_shift_ratio)
    filled = (tap1 + tap2 + tap3 + tap4) * 0.25
    return tl.where(mask, value + (filled - value) * 0.60, value)


@triton.jit
def _fast_plus_half_sbs_uint8_kernel(
    rgb,
    depth,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    height: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
    depth_strength: tl.constexpr,
    convergence: tl.constexpr,
    effective_ipd_m: tl.constexpr,
    max_shift_ratio: tl.constexpr,
    edge_threshold: tl.constexpr,
    shift_edge_threshold_px: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = x < half_width
    src_x_out = tl.where(use_left, x, x - half_width)
    target_x0 = src_x_out * 2
    target_x1 = _clamp_i32(target_x0 + 1, 0, width - 1)
    eye_sign = tl.where(use_left, -1.0, 1.0)

    v0 = _filled_eye(
        rgb,
        depth,
        channel,
        y,
        target_x0,
        eye_sign,
        width,
        height,
        pixels,
        depth_strength,
        convergence,
        effective_ipd_m,
        max_shift_ratio,
        edge_threshold,
        shift_edge_threshold_px,
    )
    v1 = _filled_eye(
        rgb,
        depth,
        channel,
        y,
        target_x1,
        eye_sign,
        width,
        height,
        pixels,
        depth_strength,
        convergence,
        effective_ipd_m,
        max_shift_ratio,
        edge_threshold,
        shift_edge_threshold_px,
    )
    value = tl.minimum(tl.maximum((v0 + v1) * 0.5, 0.0), 1.0) * 255.0
    tl.store(out + offsets, value.to(tl.uint8), mask=active)


def can_use_fast_plus_fused_half_sbs_uint8(rgb: torch.Tensor, depth: torch.Tensor) -> bool:
    return (
        rgb.is_cuda
        and depth.is_cuda
        and rgb.dtype == torch.float32
        and depth.dtype == torch.float32
        and rgb.ndim == 4
        and depth.ndim == 4
        and rgb.shape[0] == 1
        and rgb.shape[1] == 3
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and rgb.shape[-2:] == depth.shape[-2:]
        and rgb.shape[-1] % 2 == 0
    )


def make_fast_plus_fused_half_sbs_uint8(
    rgb: torch.Tensor,
    depth: torch.Tensor,
    *,
    depth_strength: float,
    convergence: float,
    effective_ipd_m: float,
    max_shift_ratio: float,
    edge_threshold: float = 0.03,
) -> torch.Tensor:
    rgb = rgb.contiguous()
    depth = depth.contiguous()
    _, channels, height, width = rgb.shape
    out = torch.empty((1, channels, height, width), device=rgb.device, dtype=torch.uint8)
    half_width = width // 2
    pixels = height * width
    total = out.numel()
    max_possible_shift = width * max(0.0, float(effective_ipd_m)) * float(max_shift_ratio) * float(depth_strength)
    shift_edge_threshold_px = max(0.20, max_possible_shift * 0.05)
    block = 256
    grid = (triton.cdiv(total, block),)
    _fast_plus_half_sbs_uint8_kernel[grid](
        rgb,
        depth,
        out,
        total,
        width,
        height,
        half_width,
        pixels,
        float(depth_strength),
        float(convergence),
        float(effective_ipd_m),
        float(max_shift_ratio),
        float(edge_threshold),
        float(shift_edge_threshold_px),
        block,
    )
    return out
