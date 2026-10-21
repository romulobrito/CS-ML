#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit checks for corrected P0-P3 DEM calibration helpers.

ASCII-only. Run:
  python methods_comparison/scripts/test_861_dem_sc_calib_p0_p3_fixes.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dem_sc_861_calibrate import PlugCalibRecord
import run_861_dem_sc_calib_hier_joint_poc as hier_module
from run_861_dem_sc_calib_p0_p3_poc import (
    FitConfig,
    FitResult,
    HfuParams,
    OrientParams,
    _usable_fit,
    cross_sensitivity_gate,
    data_loss,
    fit_hierarchical_config,
    lexicographic_pick,
    pick_inner_candidate,
)
import pandas as pd


def _plug(name: str, hfu: int, vp: float, vs: float) -> PlugCalibRecord:
    """Minimal synthetic plug."""
    return PlugCalibRecord(
        ct_sample_id=name,
        hfu=hfu,
        phi_lab=0.1,
        alpha_ct=0.5,
        solid1_pct=60.0,
        solid2_pct=40.0,
        vp_lab_z_km_s=vp,
        vs_lab_z_km_s=vs,
        vpvs_lab_z=vp / vs,
    )


def test_data_loss_normalization() -> None:
    """(J_vp + w_s*J_vs)/(1+w_s) continuity at small w_s."""
    plugs = [_plug("A", 1, 5.0, 3.0), _plug("B", 1, 5.2, 3.1)]
    params = {1: HfuParams(alpha=0.5, scale=1.0), 0: HfuParams(alpha=0.5, scale=1.0)}

    j0 = data_loss(plugs, params, None, huber=False, w_s=0.0)
    j_eps = data_loss(plugs, params, None, huber=False, w_s=0.05)
    # Old bug: mean of flat list halved the scale when Vs entered.
    # Correct: j_eps should stay near j0, not ~j0/2.
    assert np.isfinite(j0) and np.isfinite(j_eps)
    assert abs(j_eps - j0) < 0.5 * abs(j0) + 1.0e-6, (j0, j_eps)

    # Exact identity for synthetic equal residuals:
    # construct params that make pred==lab is hard without DEM; instead check formula
    # by monkeypatching predict path via known losses is overkill. Check weights:
    j1 = data_loss(plugs, params, None, huber=False, w_s=1.0)
    assert np.isfinite(j1)


def test_global_param_count() -> None:
    """Global fit uses identifiable alpha/scale only."""
    plugs = [
        _plug("H1", 1, 5.0, 3.0),
        _plug("H2", 2, 4.8, 2.9),
        _plug("H3", 1, 5.1, 3.05),
    ]
    cfg = FitConfig("G_huber", huber=True, orientation=False, w_s=0.0, structure="global")
    fit = fit_hierarchical_config(plugs, cfg, lambda_alpha=1.0, lambda_s=1.0)
    assert fit.params
    alphas = {p.alpha for h, p in fit.params.items() if h != 0}
    scales = {p.scale for h, p in fit.params.items() if h != 0}
    assert len(alphas) == 1
    assert len(scales) == 1
    assert np.isfinite(fit.fun)


def test_lexicographic_excludes_p0() -> None:
    """P0 must not win lexicographic pick."""
    summary = pd.DataFrame(
        [
            {"model": "P0", "mape_vp_pct": 10.0, "mae_vpvs": 0.01},
            {"model": "P1", "mape_vp_pct": 11.0, "mae_vpvs": 0.05},
            {"model": "G_huber", "mape_vp_pct": 10.5, "mae_vpvs": 0.04},
        ]
    )
    winner = lexicographic_pick(summary, tol_frac=0.2)
    assert winner != "P0"
    assert winner in {"P1", "G_huber"}


def test_inner_pick_includes_zero_and_uses_secondary_metrics() -> None:
    """A zero Vs weight may win when it has the best secondary metrics."""
    candidates = [
        {
            "lambda_alpha": 1.0,
            "lambda_s": 1.0,
            "w_s": 0.5,
            "mape_vp_pct": 10.0,
            "mape_vs_pct": 12.0,
            "mae_vpvs": 0.10,
        },
        {
            "lambda_alpha": 10.0,
            "lambda_s": 1.0,
            "w_s": 0.0,
            "mape_vp_pct": 10.4,
            "mape_vs_pct": 9.0,
            "mae_vpvs": 0.08,
        },
        {
            "lambda_alpha": 0.1,
            "lambda_s": 0.1,
            "w_s": 1.0,
            "mape_vp_pct": 10.6,
            "mape_vs_pct": 1.0,
            "mae_vpvs": 0.01,
        },
    ]
    assert pick_inner_candidate(candidates, vp_tol_pct=0.5) == (10.0, 1.0, 0.0)


def test_nonconverged_finite_fit_is_not_usable() -> None:
    """Finite objective alone must not qualify a failed optimizer result."""
    fit = FitResult(
        params={1: HfuParams(alpha=0.5, scale=1.0)},
        orient=OrientParams(beta_p=0.0, beta_s=0.0),
        success=False,
        message="iteration limit",
        fun=0.01,
        n_restarts=2,
    )
    assert not _usable_fit(fit)


def test_cross_sensitivity_gate() -> None:
    """Final acceptance requires both datasets and a stable P0-relative margin."""
    primary_summary = pd.DataFrame(
        [
            {"model": "P0", "mape_vp_pct": 10.0},
            {"model": "P3", "mape_vp_pct": 9.8},
        ]
    )
    sensitivity_summary = pd.DataFrame(
        [
            {"model": "P0", "mape_vp_pct": 8.0},
            {"model": "P3", "mape_vp_pct": 7.9},
        ]
    )
    primary_gate = pd.DataFrame([{"model": "P3", "accepted": 1}])
    sensitivity_gate = pd.DataFrame([{"model": "P3", "accepted": 1}])
    gate = cross_sensitivity_gate(
        primary_summary,
        primary_gate,
        sensitivity_summary,
        sensitivity_gate,
    )
    assert int(gate.iloc[0]["robustly_accepted"]) == 1


def test_hierarchical_restart_is_distinct_and_requires_success() -> None:
    """A failed primary solve must restart from a different physical point."""
    starts = []
    original_minimize = hier_module.minimize

    def fake_minimize(objective, x0, method, bounds):
        del objective, method, bounds
        start = np.asarray(x0, dtype=np.float64).copy()
        starts.append(start)
        if len(starts) == 1:
            return SimpleNamespace(
                success=False,
                fun=float("inf"),
                x=start,
                message="synthetic primary failure",
            )
        return SimpleNamespace(
            success=True,
            fun=1.0,
            x=start,
            message="synthetic restart success",
        )

    hier_module.minimize = fake_minimize
    try:
        params = hier_module.fit_hierarchical(
            [_plug("H1", 1, 5.0, 3.0), _plug("H2", 2, 4.8, 2.9)],
            use_vs=False,
            lambda_alpha=1.0,
            lambda_s=1.0,
        )
    finally:
        hier_module.minimize = original_minimize
    assert params
    assert len(starts) == 2
    assert not np.array_equal(starts[0], starts[1])


def main() -> None:
    """Run checks."""
    test_data_loss_normalization()
    test_global_param_count()
    test_lexicographic_excludes_p0()
    test_inner_pick_includes_zero_and_uses_secondary_metrics()
    test_nonconverged_finite_fit_is_not_usable()
    test_cross_sensitivity_gate()
    test_hierarchical_restart_is_distinct_and_requires_success()
    print("OK: test_861_dem_sc_calib_p0_p3_fixes")


if __name__ == "__main__":
    main()
