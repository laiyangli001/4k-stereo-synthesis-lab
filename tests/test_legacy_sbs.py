import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legacy_sbs import make_sbs


def test_legacy_make_sbs_half_sbs_keeps_source_size():
    rgb = torch.full((3, 8, 12), 128.0)
    depth = torch.linspace(0, 1, steps=8 * 12).view(8, 12)

    out = make_sbs(rgb, depth, display_mode="Half-SBS")

    assert isinstance(out, np.ndarray)
    assert out.shape == (8, 12, 3)
    assert out.dtype == np.uint8


def test_legacy_make_sbs_full_sbs_doubles_width():
    rgb = torch.full((3, 8, 12), 128.0)
    depth = torch.zeros(8, 12)

    out = make_sbs(rgb, depth, display_mode="Full-SBS")

    assert out.shape == (8, 24, 3)


def test_legacy_make_sbs_fill_16_9_can_pad_height_for_wide_input():
    rgb = torch.full((3, 8, 32), 128.0)
    depth = torch.zeros(8, 32)

    out = make_sbs(rgb, depth, display_mode="Full-SBS", fill_16_9=True)

    assert out.shape[0] > 8
    assert out.shape[1] == 64


def test_legacy_make_sbs_accepts_numpy_and_fps_overlay():
    rgb = np.full((8, 12, 3), 128, dtype=np.uint8)
    depth = torch.zeros(8, 12)

    out = make_sbs(rgb, depth, display_mode="Half-SBS", fps=60.0)

    assert out.shape == (8, 12, 3)
    assert out.dtype == np.uint8
