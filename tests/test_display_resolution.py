import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


display_module = _load_module("display_module", Path("src") / "utils" / "display.py")
preprocess_module = _load_module("preprocess_module", Path("src") / "capture" / "preprocess.py")
compute_output_resolution = display_module.compute_output_resolution
capture_frame_to_rgb = preprocess_module.capture_frame_to_rgb


def test_full_sbs_wxh_resolution_uses_half_target_width_per_eye():
    assert compute_output_resolution("7680x4320", "Full-SBS", 1, 2) == (3840, 4320)
    assert compute_output_resolution("3840x2160", "Full-SBS", 1, 2) == (1920, 2160)


def test_half_sbs_wxh_resolution_keeps_target_canvas_size():
    assert compute_output_resolution("3840x2160", "Half-SBS", 1, 2) == (3840, 2160)


def test_full_tab_wxh_resolution_uses_half_target_height_per_eye():
    assert compute_output_resolution("3840x2160", "Full-TAB", 1, 2) == (3840, 1080)


def test_numeric_height_resolution_keeps_legacy_meaning():
    assert compute_output_resolution("2160", "Full-SBS", 1, 2) == 2160


def test_wxh_resolution_accepts_common_separators():
    assert compute_output_resolution("7680 * 4320", "full_sbs", 1, 2) == (3840, 4320)
    assert compute_output_resolution("7680x4320", "full_sbs", 1, 2) == (3840, 4320)


def test_capture_preprocess_accepts_exact_width_height_target():
    frame_bgra = np.zeros((4, 8, 4), dtype=np.uint8)
    frame_bgra[..., 0] = 10
    frame_bgra[..., 1] = 20
    frame_bgra[..., 2] = 30
    frame_bgra[..., 3] = 255

    frame_rgb = capture_frame_to_rgb(frame_bgra, (6, 10))

    assert frame_rgb.shape == (10, 6, 3)
    assert tuple(frame_rgb[0, 0]) == (30, 20, 10)