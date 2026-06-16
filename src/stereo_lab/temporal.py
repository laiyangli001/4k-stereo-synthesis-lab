from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TemporalState:
    left: torch.Tensor | None = None
    right: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    scene_reference: torch.Tensor | None = None
    cooldown_remaining: int = 0
    reset_count: int = 0
    last_scene_delta: float = 0.0

    def reset(self) -> None:
        self.reset_stereo()
        self.scene_reference = None
        self.cooldown_remaining = 0

    def reset_stereo(self) -> None:
        self.left = None
        self.right = None
        self.mask = None


def detect_scene_change(
    rgb: torch.Tensor,
    state: TemporalState | None,
    *,
    threshold: float = 0.22,
    cooldown_frames: int = 3,
    sample_size: int = 64,
) -> bool:
    if state is None:
        return False
    sample = _scene_sample(rgb, sample_size=sample_size)
    if state.scene_reference is None or state.scene_reference.shape != sample.shape:
        state.scene_reference = sample.detach()
        state.last_scene_delta = 0.0
        return False
    delta = float((sample - state.scene_reference.to(sample.device)).abs().mean().item())
    state.last_scene_delta = delta
    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1
        state.scene_reference = sample.detach()
        return False
    changed = delta >= float(threshold)
    state.scene_reference = sample.detach()
    if changed:
        state.reset_count += 1
        state.cooldown_remaining = max(0, int(cooldown_frames))
    return changed


def apply_temporal(
    left: torch.Tensor,
    right: torch.Tensor,
    mask: torch.Tensor | None,
    state: TemporalState | None,
    strength: float = 0.85,
) -> tuple[torch.Tensor, torch.Tensor]:
    if state is None:
        return left, right
    if state.left is None or state.right is None or state.left.shape != left.shape or state.right.shape != right.shape:
        state.left = left.detach()
        state.right = right.detach()
        state.mask = mask.detach() if mask is not None else None
        return left, right

    alpha = float(max(0.0, min(strength, 0.98)))
    if mask is None:
        blend = alpha
    else:
        blend = (mask * alpha).to(device=left.device, dtype=left.dtype)
    left_out = left * (1.0 - blend) + state.left.to(left.device) * blend
    right_out = right * (1.0 - blend) + state.right.to(right.device) * blend
    state.left = left_out.detach()
    state.right = right_out.detach()
    state.mask = mask.detach() if mask is not None else None
    return left_out, right_out


def _scene_sample(rgb: torch.Tensor, *, sample_size: int) -> torch.Tensor:
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if rgb.ndim != 4:
        raise ValueError(f"rgb must be CHW or BCHW, got shape {tuple(rgb.shape)}")
    sample = rgb.detach().float()
    if sample.shape[1] == 3:
        sample = sample[:, 0:1] * 0.299 + sample[:, 1:2] * 0.587 + sample[:, 2:3] * 0.114
    elif sample.shape[1] != 1:
        sample = sample.mean(dim=1, keepdim=True)
    return F.interpolate(sample, size=(sample_size, sample_size), mode="area")
