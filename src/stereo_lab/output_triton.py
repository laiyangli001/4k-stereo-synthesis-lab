from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _half_sbs_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    half_width: tl.constexpr,
    pixels: tl.constexpr,
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
    src_x = (src_x_out.to(tl.float32) + 0.5) * (width / half_width) - 0.5
    src_x_clamped = tl.minimum(tl.maximum(src_x, 0.0), width - 1.0)
    x0_float = tl.floor(src_x_clamped)
    x0 = x0_float.to(tl.int64)
    x1 = tl.minimum(x0 + 1, width - 1)
    frac = src_x_clamped - x0_float
    base = channel * pixels + y * width

    left_v0 = tl.load(left + base + x0, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + base + x1, mask=active & use_left, other=0.0)
    right_v0 = tl.load(right + base + x0, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + base + x1, mask=active & ~use_left, other=0.0)
    value = tl.where(use_left, left_v0 + (left_v1 - left_v0) * frac, right_v0 + (right_v1 - right_v0) * frac)
    tl.store(out + offsets, value, mask=active)


def can_use_triton_half_sbs(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.is_cuda
        and right.is_cuda
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
        and left.shape[-1] % 2 == 0
    )


def can_use_triton_full_sbs(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.is_cuda
        and right.is_cuda
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
    )


def can_use_triton_half_tab(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.is_cuda
        and right.is_cuda
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
        and left.shape[-2] % 2 == 0
    )


def can_use_triton_full_tab(left: torch.Tensor, right: torch.Tensor) -> bool:
    return can_use_triton_full_sbs(left, right)


def can_use_triton_depth_map(depth: torch.Tensor, channels: int) -> bool:
    return (
        depth.is_cuda
        and depth.dtype == torch.float32
        and depth.ndim == 4
        and depth.shape[0] == 1
        and depth.shape[1] == 1
        and channels == 3
    )


def _can_use_triton_postprocess(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.is_cuda
        and right.is_cuda
        and left.dtype == torch.float32
        and right.dtype == torch.float32
        and left.ndim == 4
        and right.ndim == 4
        and left.shape == right.shape
        and left.shape[0] == 1
        and left.shape[1] == 3
    )


def can_use_triton_anaglyph(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _can_use_triton_postprocess(left, right)


def can_use_triton_interleaved(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _can_use_triton_postprocess(left, right)


def can_use_triton_leia(left: torch.Tensor, right: torch.Tensor) -> bool:
    return _can_use_triton_postprocess(left, right)


def make_half_sbs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    half_width = width // 2
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_sbs_kernel[grid](left, right, out, total, width, half_width, pixels, block)
    return out


@triton.jit
def _full_sbs_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    out_width: tl.constexpr,
    out_pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    out_pixel = offsets % out_pixels
    y = out_pixel // out_width
    x = out_pixel - y * out_width
    channel = offsets // out_pixels

    use_left = x < width
    src_x = tl.where(use_left, x, x - width)
    src_offset = channel * (out_pixels // 2) + y * width + src_x
    value = tl.where(
        use_left,
        tl.load(left + src_offset, mask=active & use_left, other=0.0),
        tl.load(right + src_offset, mask=active & ~use_left, other=0.0),
    )
    tl.store(out + offsets, value, mask=active)


def make_full_sbs(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    _, channels, height, width = left.shape
    out = torch.empty((1, channels, height, width * 2), device=left.device, dtype=left.dtype)
    out_width = width * 2
    out_pixels = height * out_width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _full_sbs_kernel[grid](left, right, out, total, width, out_width, out_pixels, block)
    return out


@triton.jit
def _half_tab_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    height: tl.constexpr,
    half_height: tl.constexpr,
    width: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = y < half_height
    src_y_out = tl.where(use_left, y, y - half_height)
    src_y = (src_y_out.to(tl.float32) + 0.5) * (height / half_height) - 0.5
    src_y_clamped = tl.minimum(tl.maximum(src_y, 0.0), height - 1.0)
    y0_float = tl.floor(src_y_clamped)
    y0 = y0_float.to(tl.int64)
    y1 = tl.minimum(y0 + 1, height - 1)
    frac = src_y_clamped - y0_float
    channel_base = channel * pixels

    left_v0 = tl.load(left + channel_base + y0 * width + x, mask=active & use_left, other=0.0)
    left_v1 = tl.load(left + channel_base + y1 * width + x, mask=active & use_left, other=0.0)
    right_v0 = tl.load(right + channel_base + y0 * width + x, mask=active & ~use_left, other=0.0)
    right_v1 = tl.load(right + channel_base + y1 * width + x, mask=active & ~use_left, other=0.0)
    value = tl.where(use_left, left_v0 + (left_v1 - left_v0) * frac, right_v0 + (right_v1 - right_v0) * frac)
    tl.store(out + offsets, value, mask=active)


def make_half_tab(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    half_height = height // 2
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _half_tab_kernel[grid](left, right, out, total, height, half_height, width, pixels, block)
    return out


@triton.jit
def _full_tab_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    height: tl.constexpr,
    out_height: tl.constexpr,
    width: tl.constexpr,
    out_pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    out_pixel = offsets % out_pixels
    y = out_pixel // width
    x = out_pixel - y * width
    channel = offsets // out_pixels

    use_left = y < height
    src_y = tl.where(use_left, y, y - height)
    src_offset = channel * (out_pixels // 2) + src_y * width + x
    value = tl.where(
        use_left,
        tl.load(left + src_offset, mask=active & use_left, other=0.0),
        tl.load(right + src_offset, mask=active & ~use_left, other=0.0),
    )
    tl.store(out + offsets, value, mask=active)


def make_full_tab(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    _, channels, height, width = left.shape
    out = torch.empty((1, channels, height * 2, width), device=left.device, dtype=left.dtype)
    out_height = height * 2
    out_pixels = out_height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _full_tab_kernel[grid](left, right, out, total, height, out_height, width, out_pixels, block)
    return out


@triton.jit
def _depth_map_kernel(
    depth,
    out,
    total: tl.constexpr,
    pixels: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    value = tl.load(depth + pixel, mask=active, other=0.0)
    tl.store(out + offsets, value, mask=active)


def make_depth_map(depth: torch.Tensor, channels: int) -> torch.Tensor:
    depth = depth.contiguous()
    _, _, height, width = depth.shape
    out = torch.empty((1, channels, height, width), device=depth.device, dtype=depth.dtype)
    pixels = height * width
    total = out.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _depth_map_kernel[grid](depth, out, total, pixels, block)
    return out


@triton.jit
def _stereo_postprocess_kernel(
    left,
    right,
    out,
    total: tl.constexpr,
    width: tl.constexpr,
    pixels: tl.constexpr,
    mode: tl.constexpr,
    block: tl.constexpr,
):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    active = offsets < total
    pixel = offsets % pixels
    y = pixel // width
    x = pixel - y * width
    channel = offsets // pixels

    use_left = channel == 0
    if mode == 1:
        use_left = (y % 2) == 0
    if mode == 2:
        use_left = (x % 2) == 0

    value = tl.where(
        use_left,
        tl.load(left + offsets, mask=active, other=0.0),
        tl.load(right + offsets, mask=active, other=0.0),
    )
    tl.store(out + offsets, value, mask=active)


def _make_stereo_postprocess(left: torch.Tensor, right: torch.Tensor, mode: int) -> torch.Tensor:
    left = left.contiguous()
    right = right.contiguous()
    out = torch.empty_like(left)
    _, _, height, width = left.shape
    pixels = height * width
    total = left.numel()
    block = 256
    grid = (triton.cdiv(total, block),)
    _stereo_postprocess_kernel[grid](left, right, out, total, width, pixels, mode, block)
    return out


def make_anaglyph(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _make_stereo_postprocess(left, right, mode=0)


def make_interleaved(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _make_stereo_postprocess(left, right, mode=1)


def make_leia(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return _make_stereo_postprocess(left, right, mode=2)
