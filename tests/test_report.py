import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stereo_lab.report import depth_comparison_metrics, depth_metrics


def test_depth_metrics_summary_fields():
    depth = torch.linspace(0, 1, 16).view(1, 1, 4, 4)
    metrics = depth_metrics(depth, bins=4)

    assert metrics["min"] == 0.0
    assert metrics["max"] == 1.0
    assert len(metrics["histogram"]) == 4
    assert abs(sum(metrics["histogram"]) - 1.0) < 1e-6
    assert metrics["foreground_background_separation"] > 0


def test_depth_comparison_metrics_identical_depths():
    depth = torch.rand(1, 1, 8, 8)
    metrics = depth_comparison_metrics(depth, depth)

    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["mean_bias"] == 0.0
    assert 0.0 <= metrics["edge_overlap"] <= 1.0
