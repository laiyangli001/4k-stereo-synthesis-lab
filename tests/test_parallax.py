from __future__ import annotations

import pytest
import torch

from stereo_runtime.baseline_shift import ShiftParams, compute_shift_px
from stereo_runtime.parallax import PARALLAX_BUDGET_TABLE, parallax_debug_info, resolve_parallax_budget


def test_resolve_parallax_budget_uses_short_side_table():
    budget = resolve_parallax_budget(
        render_width=1920,
        render_height=1080,
        preset="standard",
        convergence=0.0,
    )

    assert budget.max_disparity_px == 48.0
    assert budget.preset == "standard"
    assert budget.depth_response_name == "linear_clamp_convergence_v1"


def test_parallax_debug_info_records_depth_response_contract():
    budget = resolve_parallax_budget(
        render_width=1920,
        render_height=1080,
        preset="standard",
        convergence=0.0,
    )

    debug = parallax_debug_info(budget)

    assert debug["resolved_max_disparity_px"] == 48.0
    assert debug["parallax_budget_preset"] == "standard"
    assert debug["depth_response"] == "linear_clamp_convergence_v1"
    assert debug["parallax_resolver_version"] == 1


def test_resolve_parallax_budget_defaults_to_standard_preset():
    budget = resolve_parallax_budget(
        render_width=1920,
        render_height=1080,
        preset=None,
        convergence=0.0,
    )

    assert budget.max_disparity_px == 48.0
    assert budget.preset == "standard"


def test_resolve_parallax_budget_interpolates_between_resolution_levels():
    budget = resolve_parallax_budget(
        render_width=2560,
        render_height=1440,
        preset="standard",
        convergence=0.0,
    )

    assert budget.max_disparity_px == PARALLAX_BUDGET_TABLE["standard"][1440]


def test_resolve_parallax_budget_applies_ultrawide_aspect_protection():
    budget = resolve_parallax_budget(
        render_width=3840,
        render_height=1080,
        preset="standard",
        convergence=0.0,
    )

    assert budget.max_disparity_px == pytest.approx(48.0 * 0.70)


def test_compute_shift_px_uses_half_of_total_max_disparity_for_each_eye():
    depth = torch.ones(1, 1, 1, 1)
    shift = compute_shift_px(
        depth,
        1920,
        ShiftParams(convergence=0.0, max_disparity_px=96.0),
    )

    assert shift.item() == pytest.approx(-48.0)


def test_legacy_shift_params_preserve_existing_effective_ipd_formula():
    depth = torch.ones(1, 1, 1, 1)
    legacy = compute_shift_px(
        depth,
        1000,
        ShiftParams(
            depth_strength=1.0,
            convergence=0.0,
            ipd_mm=32.0,
            stereo_scale=0.35,
            max_shift_ratio=0.05,
            parallax_preset="legacy",
        ),
    )
    direct_effective_baseline = compute_shift_px(
        depth,
        1000,
        ShiftParams(
            depth_strength=1.0,
            convergence=0.0,
            ipd=0.0112,
            ipd_mm=None,
            max_shift_ratio=0.05,
            parallax_preset="legacy",
        ),
    )

    assert torch.allclose(legacy, direct_effective_baseline)
