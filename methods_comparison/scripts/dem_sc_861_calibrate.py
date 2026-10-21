#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse calibration of DEM rock-physics parameters vs ROCKPHYS lab Vp.

Fits per-HFU aspect ratio (alpha) and optional matrix scale factor (K, G)
by minimizing RMSE of Vp_DEM vs Vp_lab (Z-axis, dry, confining pressure).

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from dem_sc_861_core import run_plug_case

ALPHA_BOUNDS: Tuple[float, float] = (0.15, 0.95)
MATRIX_SCALE_BOUNDS: Tuple[float, float] = (0.50, 1.50)

CalibrationScenario = Literal["alpha_only", "alpha_matrix_scale"]


@dataclass(frozen=True)
class PlugCalibRecord:
    """One plug used in HFU-level calibration."""

    ct_sample_id: str
    hfu: int
    phi_lab: float
    alpha_ct: float
    solid1_pct: float
    solid2_pct: float
    vp_lab_z_km_s: float
    vs_lab_z_km_s: float
    vpvs_lab_z: float


@dataclass(frozen=True)
class HfuCalibResult:
    """Calibrated parameters for one HFU."""

    hfu: int
    n_plugs: int
    scenario: CalibrationScenario
    alpha_ct_median: float
    alpha_calibrated: float
    matrix_k_scale: float
    matrix_g_scale: float
    rmse_vp_km_s_before: float
    rmse_vp_km_s_after: float
    mape_vp_pct_before: float
    mape_vp_pct_after: float
    plug_ids: str


def _vp_dem_km_s(
    plug: PlugCalibRecord,
    alpha: float,
    k_scale: float = 1.0,
    g_scale: float = 1.0,
) -> float:
    """Predict DEM Vp for one plug."""
    out = run_plug_case(
        phi_lab=plug.phi_lab,
        alpha=alpha,
        solid1_pct=plug.solid1_pct,
        solid2_pct=plug.solid2_pct,
        matrix_k_scale=k_scale,
        matrix_g_scale=g_scale,
    )
    return float(out["vp_dem_km_s"])


def _rmse_mape(
    plugs: Sequence[PlugCalibRecord],
    alpha: float,
    k_scale: float = 1.0,
    g_scale: float = 1.0,
) -> Tuple[float, float]:
    """RMSE and MAPE (%) for Vp predictions."""
    errs: List[float] = []
    mapes: List[float] = []
    for plug in plugs:
        vp_pred = _vp_dem_km_s(plug, alpha, k_scale, g_scale)
        vp_obs = plug.vp_lab_z_km_s
        errs.append(vp_pred - vp_obs)
        if vp_obs > 0.0:
            mapes.append(abs(vp_pred - vp_obs) / vp_obs * 100.0)
    rmse = float(np.sqrt(np.mean(np.array(errs, dtype=np.float64) ** 2)))
    mape = float(np.mean(mapes)) if mapes else float("nan")
    return rmse, mape


def _alpha_ct_median(plugs: Sequence[PlugCalibRecord]) -> float:
    """CT-derived median alpha for plugs in HFU."""
    return float(np.median([p.alpha_ct for p in plugs]))


def calibrate_hfu_alpha_only(plugs: Sequence[PlugCalibRecord]) -> HfuCalibResult:
    """1D search on alpha; matrix moduli unchanged."""
    if not plugs:
        raise ValueError("empty plug list")
    hfu = plugs[0].hfu
    alpha_ct = _alpha_ct_median(plugs)

    def objective(alpha_arr: np.ndarray) -> float:
        alpha = float(alpha_arr[0])
        rmse, _ = _rmse_mape(plugs, alpha, 1.0, 1.0)
        return rmse

    x0 = np.array([alpha_ct], dtype=np.float64)
    bounds = [ALPHA_BOUNDS]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    alpha_opt = float(np.clip(res.x[0], ALPHA_BOUNDS[0], ALPHA_BOUNDS[1]))

    rmse_b, mape_b = _rmse_mape(plugs, alpha_ct, 1.0, 1.0)
    rmse_a, mape_a = _rmse_mape(plugs, alpha_opt, 1.0, 1.0)

    return HfuCalibResult(
        hfu=hfu,
        n_plugs=len(plugs),
        scenario="alpha_only",
        alpha_ct_median=alpha_ct,
        alpha_calibrated=alpha_opt,
        matrix_k_scale=1.0,
        matrix_g_scale=1.0,
        rmse_vp_km_s_before=rmse_b,
        rmse_vp_km_s_after=rmse_a,
        mape_vp_pct_before=mape_b,
        mape_vp_pct_after=mape_a,
        plug_ids=",".join(p.ct_sample_id for p in plugs),
    )


def calibrate_hfu_alpha_matrix_scale(plugs: Sequence[PlugCalibRecord]) -> HfuCalibResult:
    """Joint fit alpha + uniform matrix scale (K and G)."""
    if not plugs:
        raise ValueError("empty plug list")
    hfu = plugs[0].hfu
    alpha_ct = _alpha_ct_median(plugs)

    def objective(x: np.ndarray) -> float:
        alpha = float(x[0])
        scale = float(x[1])
        rmse, _ = _rmse_mape(plugs, alpha, scale, scale)
        return rmse

    x0 = np.array([alpha_ct, 1.0], dtype=np.float64)
    bounds = [ALPHA_BOUNDS, MATRIX_SCALE_BOUNDS]
    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    alpha_opt = float(np.clip(res.x[0], ALPHA_BOUNDS[0], ALPHA_BOUNDS[1]))
    scale_opt = float(np.clip(res.x[1], MATRIX_SCALE_BOUNDS[0], MATRIX_SCALE_BOUNDS[1]))

    rmse_b, mape_b = _rmse_mape(plugs, alpha_ct, 1.0, 1.0)
    rmse_a, mape_a = _rmse_mape(plugs, alpha_opt, scale_opt, scale_opt)

    return HfuCalibResult(
        hfu=hfu,
        n_plugs=len(plugs),
        scenario="alpha_matrix_scale",
        alpha_ct_median=alpha_ct,
        alpha_calibrated=alpha_opt,
        matrix_k_scale=scale_opt,
        matrix_g_scale=scale_opt,
        rmse_vp_km_s_before=rmse_b,
        rmse_vp_km_s_after=rmse_a,
        mape_vp_pct_before=mape_b,
        mape_vp_pct_after=mape_a,
        plug_ids=",".join(p.ct_sample_id for p in plugs),
    )


def choose_best_scenario(
    alpha_res: HfuCalibResult,
    joint_res: HfuCalibResult,
) -> HfuCalibResult:
    """Pick scenario with lower post-calibration RMSE."""
    if joint_res.rmse_vp_km_s_after <= alpha_res.rmse_vp_km_s_after:
        return joint_res
    return alpha_res


def build_plug_records(
    ct_df: pd.DataFrame,
    lab_val_df: pd.DataFrame,
    exclude_samples: Optional[Sequence[str]] = None,
) -> List[PlugCalibRecord]:
    """Merge CT solids with lab validation table."""
    exclude = set(exclude_samples or ())
    lab_ok = lab_val_df[~lab_val_df["ct_sample_id"].isin(exclude)].copy()
    ct = ct_df.rename(columns={"sample_id": "ct_sample_id"})
    merged = lab_ok.merge(
        ct[
            [
                "ct_sample_id",
                "Phi_lab (pu)",
                "ct_ar_mean",
                "corrected_solid1_pct",
                "corrected_solid2_pct",
            ]
        ],
        on="ct_sample_id",
        how="inner",
    )
    records: List[PlugCalibRecord] = []
    for _, r in merged.iterrows():
        records.append(
            PlugCalibRecord(
                ct_sample_id=str(r["ct_sample_id"]),
                hfu=int(r["HFU"]),
                phi_lab=float(r["Phi_lab (pu)"]),
                alpha_ct=float(r["ct_ar_mean"]),
                solid1_pct=float(r["corrected_solid1_pct"]),
                solid2_pct=float(r["corrected_solid2_pct"]),
                vp_lab_z_km_s=float(r["vp_lab_z_km_s"]),
                vs_lab_z_km_s=float(r["vs_lab_z_km_s"]),
                vpvs_lab_z=float(r["vpvs_lab_z"]),
            )
        )
    return records


def calibrate_all_hfus(
    plugs: Sequence[PlugCalibRecord],
) -> Tuple[pd.DataFrame, Dict[int, HfuCalibResult]]:
    """
    Calibrate each HFU present in plug list.

    Returns table of chosen parameters and dict hfu -> result.
    """
    by_hfu: Dict[int, List[PlugCalibRecord]] = {}
    for plug in plugs:
        by_hfu.setdefault(plug.hfu, []).append(plug)

    chosen: Dict[int, HfuCalibResult] = {}
    rows: List[dict] = []
    for hfu in sorted(by_hfu.keys()):
        group = by_hfu[hfu]
        alpha_res = calibrate_hfu_alpha_only(group)
        joint_res = calibrate_hfu_alpha_matrix_scale(group)
        best = choose_best_scenario(alpha_res, joint_res)
        chosen[hfu] = best
        rows.append(
            {
                "HFU": hfu,
                "n_plugs_calib": best.n_plugs,
                "plug_ids": best.plug_ids,
                "scenario_chosen": best.scenario,
                "alpha_ct_median": best.alpha_ct_median,
                "alpha_calibrated": best.alpha_calibrated,
                "matrix_k_scale": best.matrix_k_scale,
                "matrix_g_scale": best.matrix_g_scale,
                "rmse_vp_before_km_s": best.rmse_vp_km_s_before,
                "rmse_vp_after_km_s": best.rmse_vp_km_s_after,
                "mape_vp_before_pct": best.mape_vp_pct_before,
                "mape_vp_after_pct": best.mape_vp_pct_after,
                "alpha_only_rmse_after": alpha_res.rmse_vp_km_s_after,
                "joint_rmse_after": joint_res.rmse_vp_km_s_after,
            }
        )
    return pd.DataFrame(rows), chosen


def predict_plug_calibrated(
    plug: PlugCalibRecord,
    hfu_params: HfuCalibResult,
) -> dict:
    """DEM prediction using calibrated HFU parameters."""
    out = run_plug_case(
        phi_lab=plug.phi_lab,
        alpha=hfu_params.alpha_calibrated,
        solid1_pct=plug.solid1_pct,
        solid2_pct=plug.solid2_pct,
        matrix_k_scale=hfu_params.matrix_k_scale,
        matrix_g_scale=hfu_params.matrix_g_scale,
    )
    vp_lab = plug.vp_lab_z_km_s
    vp_dem = float(out["vp_dem_km_s"])
    rel_err = 100.0 * (vp_dem - vp_lab) / vp_lab if vp_lab > 0 else float("nan")
    return {
        "ct_sample_id": plug.ct_sample_id,
        "HFU": plug.hfu,
        "phi_lab_pu": plug.phi_lab,
        "alpha_ct": plug.alpha_ct,
        "alpha_calibrated": hfu_params.alpha_calibrated,
        "matrix_k_scale": hfu_params.matrix_k_scale,
        "matrix_g_scale": hfu_params.matrix_g_scale,
        "scenario": hfu_params.scenario,
        "vp_lab_z_km_s": vp_lab,
        "vs_lab_z_km_s": plug.vs_lab_z_km_s,
        "vpvs_lab_z": plug.vpvs_lab_z,
        "vp_dem_uncal_km_s": _vp_dem_km_s(plug, plug.alpha_ct, 1.0, 1.0),
        "vp_dem_calib_km_s": vp_dem,
        "vs_dem_calib_km_s": float(out["vs_dem_km_s"]),
        "vpvs_dem_calib": float(out["vpvs_dem"]),
        "vp_rel_error_uncal_pct": 100.0
        * (_vp_dem_km_s(plug, plug.alpha_ct, 1.0, 1.0) - vp_lab)
        / vp_lab,
        "vp_rel_error_calib_pct": rel_err,
        "vp_abs_rel_error_calib_pct": abs(rel_err),
    }


def global_validation_metrics(val_df: pd.DataFrame, vp_col: str) -> Dict[str, float]:
    """Aggregate metrics for a Vp prediction column vs lab."""
    vp_lab = val_df["vp_lab_z_km_s"].to_numpy(dtype=np.float64)
    vp_pred = val_df[vp_col].to_numpy(dtype=np.float64)
    err = vp_pred - vp_lab
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / vp_lab)) * 100.0)
    bias = float(np.mean(err))
    pearson = float(np.corrcoef(vp_lab, vp_pred)[0, 1]) if len(vp_lab) > 1 else float("nan")
    return {
        "mae_vp_km_s": mae,
        "rmse_vp_km_s": rmse,
        "mape_vp_pct": mape,
        "bias_vp_km_s": bias,
        "pearson_r_vp": pearson,
        "mean_vp_lab_km_s": float(np.mean(vp_lab)),
        "mean_vp_pred_km_s": float(np.mean(vp_pred)),
    }


def loo_predict_one_fold(
    holdout: PlugCalibRecord,
    train_plugs: Sequence[PlugCalibRecord],
) -> dict:
    """
    Leave-one-plug-out: calibrate HFUs on train set, predict held-out plug.

    Each HFU is calibrated only from training plugs belonging to that HFU.
    """
    n_train_hfu = sum(1 for p in train_plugs if p.hfu == holdout.hfu)
    vp_lab = holdout.vp_lab_z_km_s
    vp_uncal = _vp_dem_km_s(holdout, holdout.alpha_ct, 1.0, 1.0)

    if n_train_hfu < 1:
        return {
            "ct_sample_id": holdout.ct_sample_id,
            "HFU": holdout.hfu,
            "held_out": holdout.ct_sample_id,
            "n_train_total": len(train_plugs),
            "n_train_hfu": n_train_hfu,
            "status": "no_train_hfu",
            "vp_lab_z_km_s": vp_lab,
            "vp_dem_uncal_km_s": vp_uncal,
            "vp_dem_loo_km_s": float("nan"),
            "vp_rel_error_uncal_pct": 100.0 * (vp_uncal - vp_lab) / vp_lab,
            "vp_rel_error_loo_pct": float("nan"),
            "vp_abs_rel_error_loo_pct": float("nan"),
        }

    _, chosen = calibrate_all_hfus(train_plugs)
    hfu_res = chosen[holdout.hfu]
    out = run_plug_case(
        phi_lab=holdout.phi_lab,
        alpha=hfu_res.alpha_calibrated,
        solid1_pct=holdout.solid1_pct,
        solid2_pct=holdout.solid2_pct,
        matrix_k_scale=hfu_res.matrix_k_scale,
        matrix_g_scale=hfu_res.matrix_g_scale,
    )
    vp_loo = float(out["vp_dem_km_s"])
    rel_loo = 100.0 * (vp_loo - vp_lab) / vp_lab if vp_lab > 0 else float("nan")

    return {
        "ct_sample_id": holdout.ct_sample_id,
        "HFU": holdout.hfu,
        "held_out": holdout.ct_sample_id,
        "n_train_total": len(train_plugs),
        "n_train_hfu": n_train_hfu,
        "status": "ok",
        "phi_lab_pu": holdout.phi_lab,
        "alpha_ct": holdout.alpha_ct,
        "alpha_loo": hfu_res.alpha_calibrated,
        "matrix_k_scale_loo": hfu_res.matrix_k_scale,
        "matrix_g_scale_loo": hfu_res.matrix_g_scale,
        "scenario_loo": hfu_res.scenario,
        "vp_lab_z_km_s": vp_lab,
        "vs_lab_z_km_s": holdout.vs_lab_z_km_s,
        "vpvs_lab_z": holdout.vpvs_lab_z,
        "vp_dem_uncal_km_s": vp_uncal,
        "vp_dem_loo_km_s": vp_loo,
        "vs_dem_loo_km_s": float(out["vs_dem_km_s"]),
        "vpvs_dem_loo": float(out["vpvs_dem"]),
        "vp_rel_error_uncal_pct": 100.0 * (vp_uncal - vp_lab) / vp_lab,
        "vp_rel_error_loo_pct": rel_loo,
        "vp_abs_rel_error_loo_pct": abs(rel_loo),
        "vpvs_error_loo": float(out["vpvs_dem"]) - holdout.vpvs_lab_z,
    }


def run_loo_validation(
    plugs: Sequence[PlugCalibRecord],
) -> pd.DataFrame:
    """Run leave-one-plug-out cross-validation over all plugs."""
    rows: List[dict] = []
    for holdout in plugs:
        train = [p for p in plugs if p.ct_sample_id != holdout.ct_sample_id]
        rows.append(loo_predict_one_fold(holdout, train))
    return pd.DataFrame(rows)
