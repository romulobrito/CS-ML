#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POC P0-P3: robust hierarchical Vp, orientation, weighted Vs (Well 861).

Sequence (incremental; each kept only if OOF depth-group improves):
  P0 -- M0 baseline (per-HFU Vp)
  P1 -- hierarchical Vp + Huber (robust M2)
  P2 -- P1 + global vertical orientation offsets
  P3 -- P2 + Vs weight w_s in {0, 0.05, 0.10, 0.25, 0.50, 1}

Validation: leave-one-depth/core-out. DSI not used.
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from dem_sc_861_calibrate import (
    ALPHA_BOUNDS,
    MATRIX_SCALE_BOUNDS,
    PlugCalibRecord,
    build_plug_records,
    calibrate_hfu_alpha_matrix_scale,
    calibrate_hfu_alpha_only,
    choose_best_scenario,
)
from dem_sc_861_core import run_plug_case
from ml_861_data import load_ct_samples
from run_861_dem_sc_calib_hier_joint_poc import (
    DEPTH_TOL_M,
    LAB_VAL_CSV,
    assign_depth_groups,
    clear_pred_cache,
    fit_m0,
    HfuParams,
    PlugRow,
    resolve_params_for_hfu,
    _fallback_params,
    _PRED_CACHE,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "calib_p0_p3_poc"
)

HUBER_DELTA = 0.15
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
W_S_GRID = (0.0, 0.05, 0.10, 0.25, 0.50, 1.0)
LAMBDA_BETA = 10.0
MAX_INNER = 3


@dataclass(frozen=True)
class FitConfig:
    """One P-model configuration."""

    name: str
    huber: bool
    orientation: bool
    w_s: float
    structure: str  # "hfu" | "global"
    select_ws: bool = False


@dataclass(frozen=True)
class OrientParams:
    """Global log-velocity offsets for vertical plugs."""

    beta_p: float
    beta_s: float


def is_vertical(sample_id: str) -> bool:
    """True if plug id ends with V."""
    return str(sample_id).upper().endswith("V")


def predict_dem(plug: PlugCalibRecord, alpha: float, scale: float) -> Tuple[float, float]:
    """Cached DEM Vp/Vs."""
    key = (plug.ct_sample_id, float(alpha), float(scale))
    cached = _PRED_CACHE.get(key)
    if cached is not None:
        return cached
    out = run_plug_case(
        phi_lab=plug.phi_lab,
        alpha=alpha,
        solid1_pct=plug.solid1_pct,
        solid2_pct=plug.solid2_pct,
        matrix_k_scale=scale,
        matrix_g_scale=scale,
    )
    pred = (float(out["vp_dem_km_s"]), float(out["vs_dem_km_s"]))
    _PRED_CACHE[key] = pred
    return pred


def apply_orientation(
    vp_dem: float,
    vs_dem: float,
    sample_id: str,
    orient: Optional[OrientParams],
) -> Tuple[float, float]:
    """Apply log-space vertical offsets if enabled."""
    if orient is None or not is_vertical(sample_id):
        return vp_dem, vs_dem
    vp = vp_dem * float(np.exp(orient.beta_p))
    vs = vs_dem * float(np.exp(orient.beta_s))
    return vp, vs


def huber_rel(pred: float, obs: float, delta: float = HUBER_DELTA) -> float:
    """Huber loss on relative residual (pred-obs)/obs."""
    if obs <= 0.0:
        return 1.0e6
    r = (pred - obs) / obs
    ar = abs(r)
    if ar <= delta:
        return 0.5 * r * r
    return float(delta * (ar - 0.5 * delta))


def sq_rel(pred: float, obs: float) -> float:
    """Squared relative residual."""
    if obs <= 0.0:
        return 1.0e6
    r = (pred - obs) / obs
    return float(r * r)


def logistic_alpha(u: float) -> float:
    """Map unconstrained u to alpha in (lo, hi)."""
    lo, hi = ALPHA_BOUNDS
    return float(lo + (hi - lo) * expit(u))


def logistic_scale(v: float) -> float:
    """Map unconstrained v to scale in (lo, hi)."""
    lo, hi = MATRIX_SCALE_BOUNDS
    return float(lo + (hi - lo) * expit(v))


def inv_logistic(x: float, lo: float, hi: float) -> float:
    """Inverse logistic for initialization."""
    z = (x - lo) / (hi - lo)
    z = float(np.clip(z, 1.0e-6, 1.0 - 1.0e-6))
    return float(np.log(z / (1.0 - z)))


def data_loss(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    orient: Optional[OrientParams],
    huber: bool,
    w_s: float,
) -> float:
    """
    Mean Vp loss, optionally combined with weighted Vs.

    Consistent normalization:
      J = J_vp                         if w_s <= 0
      J = (J_vp + w_s * J_vs) / (1+w_s) otherwise
    """
    if not plugs:
        return float("nan")
    vp_vals: List[float] = []
    vs_vals: List[float] = []
    for plug in plugs:
        hp = resolve_params_for_hfu(params, plug.hfu)
        vp_d, vs_d = predict_dem(plug, hp.alpha, hp.scale)
        vp_p, vs_p = apply_orientation(vp_d, vs_d, plug.ct_sample_id, orient)
        if huber:
            vp_vals.append(huber_rel(vp_p, plug.vp_lab_z_km_s))
            if w_s > 0.0:
                vs_vals.append(huber_rel(vs_p, plug.vs_lab_z_km_s))
        else:
            vp_vals.append(sq_rel(vp_p, plug.vp_lab_z_km_s))
            if w_s > 0.0:
                vs_vals.append(sq_rel(vs_p, plug.vs_lab_z_km_s))
    j_vp = float(np.mean(np.asarray(vp_vals, dtype=np.float64)))
    if w_s <= 0.0 or not vs_vals:
        return j_vp
    j_vs = float(np.mean(np.asarray(vs_vals, dtype=np.float64)))
    return (j_vp + float(w_s) * j_vs) / (1.0 + float(w_s))


@dataclass(frozen=True)
class FitResult:
    """Fitted parameters plus optimizer diagnostics."""

    params: Dict[int, HfuParams]
    orient: Optional[OrientParams]
    success: bool
    message: str
    fun: float
    n_restarts: int


def _finite_objective(fun: float) -> bool:
    """True if objective value is usable."""
    return bool(np.isfinite(fun))


def run_minimize_checked(
    objective,
    x0: np.ndarray,
    restarts: Sequence[np.ndarray],
) -> Tuple[np.ndarray, bool, str, float, int]:
    """
    L-BFGS-B with restarts; require finite fun.

    Returns (x_best, success, message, fun, n_restarts_used).
    """
    best_x = np.asarray(x0, dtype=np.float64).copy()
    best_fun = float("inf")
    best_ok = False
    best_msg = "not_run"
    n_used = 0
    starts = [np.asarray(x0, dtype=np.float64)]
    starts.extend(np.asarray(r, dtype=np.float64) for r in restarts)
    for start in starts:
        n_used += 1
        res = minimize(objective, start, method="L-BFGS-B")
        fun = float(res.fun) if res.fun is not None else float("nan")
        ok = bool(res.success) and _finite_objective(fun)
        if ok and fun < best_fun:
            best_fun = fun
            best_x = np.asarray(res.x, dtype=np.float64).copy()
            best_ok = True
            best_msg = str(res.message)
        elif (not best_ok) and _finite_objective(fun) and fun < best_fun:
            # Keep best finite even if optimizer flag failed.
            best_fun = fun
            best_x = np.asarray(res.x, dtype=np.float64).copy()
            best_msg = "finite_but_flag_false:" + str(res.message)
    if not _finite_objective(best_fun):
        return best_x, False, "non_finite_objective", best_fun, n_used
    return best_x, best_ok, best_msg, best_fun, n_used


def fit_p0(plugs: Sequence[PlugCalibRecord]) -> FitResult:
    """Baseline M0 (independent per-HFU Vp calibration)."""
    params = fit_m0(plugs)
    return FitResult(
        params=params,
        orient=None,
        success=True,
        message="m0_closed_form_lbfgs_per_hfu",
        fun=data_loss(plugs, params, None, huber=False, w_s=0.0),
        n_restarts=1,
    )


def fit_hierarchical_config(
    plugs: Sequence[PlugCalibRecord],
    cfg: FitConfig,
    lambda_alpha: float,
    lambda_s: float,
) -> FitResult:
    """
    Fit hierarchical or global model with optional Huber, orientation, Vs weight.

    Global structure uses only (alpha, scale) [-- plus beta if orientation].
    HFU structure uses logistic alpha/scale with hierarchy on (alpha_h - alpha0)
    and (log s_h - log s0).
    """
    if not plugs:
        return FitResult({}, None, False, "empty_plugs", float("nan"), 0)

    alpha_ct = float(np.median([p.alpha_ct for p in plugs]))
    u0 = inv_logistic(alpha_ct, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])
    v0 = inv_logistic(1.0, MATRIX_SCALE_BOUNDS[0], MATRIX_SCALE_BOUNDS[1])

    if cfg.structure == "global":
        # Identifiable params only: u, v [, beta_p, beta_s]
        x0_list = [u0, v0]
        if cfg.orientation:
            x0_list.extend([0.0, 0.0])
        x0 = np.asarray(x0_list, dtype=np.float64)

        def unpack_global(
            x: np.ndarray,
        ) -> Tuple[Dict[int, HfuParams], Optional[OrientParams], float, float]:
            alpha = logistic_alpha(float(x[0]))
            scale = logistic_scale(float(x[1]))
            gp = HfuParams(alpha=alpha, scale=scale)
            params = {p.hfu: gp for p in plugs}
            params[0] = gp
            orient = None
            if cfg.orientation:
                orient = OrientParams(beta_p=float(x[2]), beta_s=float(x[3]))
            return params, orient, alpha, scale

        def objective(x: np.ndarray) -> float:
            params, orient, _, _ = unpack_global(x)
            loss = data_loss(plugs, params, orient, cfg.huber, cfg.w_s)
            if orient is not None:
                loss += float(LAMBDA_BETA) * (orient.beta_p ** 2 + orient.beta_s ** 2)
            return loss

        restarts = [
            np.asarray(x0, dtype=np.float64) + np.array(
                [0.5, -0.5] + ([0.0, 0.0] if cfg.orientation else []),
                dtype=np.float64,
            )[: len(x0)],
            np.asarray(
                [inv_logistic(0.5, *ALPHA_BOUNDS), inv_logistic(0.8, *MATRIX_SCALE_BOUNDS)]
                + ([0.0, 0.0] if cfg.orientation else []),
                dtype=np.float64,
            ),
        ]
        x_best, ok, msg, fun, n_re = run_minimize_checked(objective, x0, restarts)
        params, orient, _, _ = unpack_global(x_best)
        return FitResult(params, orient, ok, msg, fun, n_re)

    # Hierarchical by HFU
    hfus = sorted({p.hfu for p in plugs})
    x0_list = [u0, v0]
    for _ in hfus:
        x0_list.extend([0.0, 0.0])
    if cfg.orientation:
        x0_list.extend([0.0, 0.0])
    x0 = np.asarray(x0_list, dtype=np.float64)

    def unpack_hfu(
        x: np.ndarray,
    ) -> Tuple[Dict[int, HfuParams], Optional[OrientParams], float, float]:
        u_g = float(x[0])
        v_g = float(x[1])
        alpha0 = logistic_alpha(u_g)
        s0 = logistic_scale(v_g)
        params: Dict[int, HfuParams] = {}
        idx = 2
        for hfu in hfus:
            du = float(x[idx])
            dv = float(x[idx + 1])
            idx += 2
            params[hfu] = HfuParams(
                alpha=logistic_alpha(u_g + du),
                scale=logistic_scale(v_g + dv),
            )
        orient = None
        if cfg.orientation:
            orient = OrientParams(beta_p=float(x[idx]), beta_s=float(x[idx + 1]))
        params[0] = HfuParams(alpha=alpha0, scale=s0)
        return params, orient, alpha0, s0

    def objective(x: np.ndarray) -> float:
        params, orient, alpha0, s0 = unpack_hfu(x)
        loss = data_loss(plugs, params, orient, cfg.huber, cfg.w_s)
        hier = 0.0
        for hfu in hfus:
            hp = params[hfu]
            hier += float(lambda_alpha) * (hp.alpha - alpha0) ** 2
            hier += float(lambda_s) * (np.log(hp.scale) - np.log(s0)) ** 2
        if orient is not None:
            hier += float(LAMBDA_BETA) * (orient.beta_p ** 2 + orient.beta_s ** 2)
        return loss + hier

    restarts = [
        np.asarray(x0, dtype=np.float64) * 0.0
        + np.concatenate(
            [
                np.array([u0 + 0.3, v0 - 0.3], dtype=np.float64),
                np.zeros(len(x0) - 2, dtype=np.float64),
            ]
        ),
    ]
    x_best, ok, msg, fun, n_re = run_minimize_checked(objective, x0, restarts)
    params, orient, _, _ = unpack_hfu(x_best)
    return FitResult(params, orient, ok, msg, fun, n_re)


def select_hyperparams(
    train_rows: Sequence[PlugRow],
    cfg: FitConfig,
    nested: bool,
) -> Tuple[float, float, float]:
    """
    Nested pick of (lambda_alpha, lambda_s, w_s).

    Global models skip lambda (unused). P3 with select_ws also searches w_s
    inside the outer-fold training set only.
    Selection score is Vp-only relative squared loss (primary gate).
    """
    if cfg.structure == "global":
        return 0.0, 0.0, float(cfg.w_s)

    w_grid: Sequence[float]
    if cfg.select_ws:
        w_grid = tuple(w for w in W_S_GRID if w > 0.0)
    else:
        w_grid = (float(cfg.w_s),)

    if not nested:
        # Fixed defaults for --fast; still allow explicit cfg.w_s.
        if cfg.select_ws:
            # Cheap default mid weight when not nesting.
            return 1.0, 1.0, 0.5
        return 1.0, 1.0, float(cfg.w_s)

    groups = sorted({r.group_id for r in train_rows})
    if len(groups) < 2:
        return 1.0, 1.0, float(w_grid[0])
    if len(groups) > MAX_INNER:
        idxs = np.linspace(0, len(groups) - 1, MAX_INNER)
        groups = sorted({groups[int(round(i))] for i in idxs})

    best = (1.0, 1.0, float(w_grid[0]))
    best_score = float("inf")
    for la in LAMBDA_GRID:
        for ls in LAMBDA_GRID:
            for ws in w_grid:
                scores: List[float] = []
                trial_cfg = FitConfig(
                    name=cfg.name,
                    huber=cfg.huber,
                    orientation=cfg.orientation,
                    w_s=float(ws),
                    structure=cfg.structure,
                    select_ws=False,
                )
                for gid in groups:
                    tr = [r.record for r in train_rows if r.group_id != gid]
                    te = [r.record for r in train_rows if r.group_id == gid]
                    if not tr or not te:
                        continue
                    fit = fit_hierarchical_config(tr, trial_cfg, la, ls)
                    if not _finite_objective(fit.fun):
                        continue
                    # Vp-only score for hyperparameter selection.
                    scores.append(
                        score_holdout(te, fit.params, fit.orient, huber=False, w_s=0.0)
                    )
                if not scores:
                    continue
                m = float(np.mean(scores))
                if m < best_score:
                    best_score = m
                    best = (float(la), float(ls), float(ws))
    return best


def score_holdout(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    orient: Optional[OrientParams],
    huber: bool,
    w_s: float,
) -> float:
    """Holdout score with consistent data_loss normalization."""
    return data_loss(plugs, params, orient, huber=huber, w_s=w_s)


def predict_table(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    orient: Optional[OrientParams],
    model: str,
    fold_id: int,
    group_id: int,
    lambda_alpha: float,
    lambda_s: float,
    w_s: float,
    opt_success: bool,
    opt_message: str,
) -> List[dict]:
    """Per-plug OOF predictions."""
    rows: List[dict] = []
    for plug in plugs:
        hp = resolve_params_for_hfu(params, plug.hfu)
        vp_d, vs_d = predict_dem(plug, hp.alpha, hp.scale)
        vp_p, vs_p = apply_orientation(vp_d, vs_d, plug.ct_sample_id, orient)
        vp_lab = plug.vp_lab_z_km_s
        vs_lab = plug.vs_lab_z_km_s
        rows.append(
            {
                "model": model,
                "fold_id": fold_id,
                "group_id": group_id,
                "ct_sample_id": plug.ct_sample_id,
                "orientation": "V" if is_vertical(plug.ct_sample_id) else "H",
                "HFU": plug.hfu,
                "alpha": hp.alpha,
                "scale": hp.scale,
                "beta_p": orient.beta_p if orient else 0.0,
                "beta_s": orient.beta_s if orient else 0.0,
                "lambda_alpha": lambda_alpha,
                "lambda_s": lambda_s,
                "w_s": w_s,
                "opt_success": int(opt_success),
                "opt_message": opt_message,
                "vp_lab_km_s": vp_lab,
                "vs_lab_km_s": vs_lab,
                "vpvs_lab": plug.vpvs_lab_z,
                "vp_pred_km_s": vp_p,
                "vs_pred_km_s": vs_p,
                "vpvs_pred": vp_p / vs_p if vs_p > 0 else float("nan"),
                "vp_rel_err": (vp_p - vp_lab) / vp_lab if vp_lab > 0 else float("nan"),
                "vs_rel_err": (vs_p - vs_lab) / vs_lab if vs_lab > 0 else float("nan"),
                "vpvs_err": (vp_p / vs_p) - plug.vpvs_lab_z if vs_p > 0 else float("nan"),
                "alpha_at_bound": int(
                    abs(hp.alpha - ALPHA_BOUNDS[0]) < 1.0e-3
                    or abs(hp.alpha - ALPHA_BOUNDS[1]) < 1.0e-3
                ),
            }
        )
    return rows


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    """OOF metrics per model."""
    rows: List[dict] = []
    for model, sub in pred.groupby("model"):
        vp_err = sub["vp_pred_km_s"] - sub["vp_lab_km_s"]
        vs_err = sub["vs_pred_km_s"] - sub["vs_lab_km_s"]
        joint = 0.5 * (
            sub["vp_rel_err"].to_numpy(dtype=np.float64) ** 2
            + sub["vs_rel_err"].to_numpy(dtype=np.float64) ** 2
        )
        rows.append(
            {
                "model": model,
                "n": int(len(sub)),
                "mape_vp_pct": float(np.mean(np.abs(sub["vp_rel_err"])) * 100.0),
                "rmse_vp_km_s": float(np.sqrt(np.mean(vp_err.to_numpy() ** 2))),
                "bias_vp_km_s": float(np.mean(vp_err)),
                "mape_vs_pct": float(np.mean(np.abs(sub["vs_rel_err"])) * 100.0),
                "rmse_vs_km_s": float(np.sqrt(np.mean(vs_err.to_numpy() ** 2))),
                "mae_vpvs": float(np.mean(np.abs(sub["vpvs_err"]))),
                "joint_rel_sq": float(np.mean(joint)),
                "frac_alpha_at_bound": float(np.mean(sub["alpha_at_bound"])),
                "alpha_std": float(np.std(sub["alpha"])),
                "scale_std": float(np.std(sub["scale"])),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def summarize_by_orientation(pred: pd.DataFrame) -> pd.DataFrame:
    """Split H/V metrics."""
    rows: List[dict] = []
    for (model, orient), sub in pred.groupby(["model", "orientation"]):
        rows.append(
            {
                "model": model,
                "orientation": orient,
                "n": int(len(sub)),
                "mape_vp_pct": float(np.mean(np.abs(sub["vp_rel_err"])) * 100.0),
                "mape_vs_pct": float(np.mean(np.abs(sub["vs_rel_err"])) * 100.0),
                "mae_vpvs": float(np.mean(np.abs(sub["vpvs_err"]))),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "orientation"]).reset_index(drop=True)


def wins_vs_p0(pred: pd.DataFrame) -> pd.DataFrame:
    """Count depth groups beating P0 on MAPE Vp."""
    rows: List[dict] = []
    for gid, gdf in pred.groupby("group_id"):
        scores: Dict[str, float] = {}
        for model, sdf in gdf.groupby("model"):
            scores[str(model)] = float(np.mean(np.abs(sdf["vp_rel_err"])) * 100.0)
        p0 = scores.get("P0", float("nan"))
        for model, score in scores.items():
            rows.append(
                {
                    "group_id": int(gid),
                    "model": model,
                    "mape_vp_pct": score,
                    "beats_p0_vp": int(np.isfinite(p0) and score < p0),
                }
            )
    return pd.DataFrame(rows)


def build_configs(include_global: bool, sweep_ws: bool) -> List[FitConfig]:
    """
    Core sequence: P0, P1, P2, P3 (inner w_s selection).

    Optional --sweep-ws keeps fixed-weight exploratory P3_w* models.
    """
    cfgs = [
        FitConfig("P0", huber=False, orientation=False, w_s=0.0, structure="hfu"),
        FitConfig("P1", huber=True, orientation=False, w_s=0.0, structure="hfu"),
        FitConfig("P2", huber=True, orientation=True, w_s=0.0, structure="hfu"),
        FitConfig(
            "P3",
            huber=True,
            orientation=True,
            w_s=0.5,
            structure="hfu",
            select_ws=True,
        ),
    ]
    if sweep_ws:
        for w in W_S_GRID:
            if w <= 0.0:
                continue
            name = "P3_w{:g}".format(w).replace(".", "p")
            cfgs.append(
                FitConfig(
                    name,
                    huber=True,
                    orientation=True,
                    w_s=float(w),
                    structure="hfu",
                    select_ws=False,
                )
            )
    if include_global:
        cfgs.append(
            FitConfig("G_huber", huber=True, orientation=False, w_s=0.0, structure="global")
        )
        cfgs.append(
            FitConfig(
                "G_huber_orient",
                huber=True,
                orientation=True,
                w_s=0.0,
                structure="global",
            )
        )
    return cfgs


def run_cv(
    rows: Sequence[PlugRow],
    configs: Sequence[FitConfig],
    nested_lambda: bool,
) -> pd.DataFrame:
    """Leave-one-depth CV for all configs."""
    groups = sorted({r.group_id for r in rows})
    pred_rows: List[dict] = []
    for fold_id, gid in enumerate(groups):
        print("outer fold {}/{} group={}".format(fold_id + 1, len(groups), gid), flush=True)
        train_rows = [r for r in rows if r.group_id != gid]
        test_plugs = [r.record for r in rows if r.group_id == gid]
        train_plugs = [r.record for r in train_rows]
        for cfg in configs:
            t0 = time.time()
            if cfg.name == "P0":
                fit = fit_p0(train_plugs)
                la, ls, ws = 0.0, 0.0, 0.0
            else:
                la, ls, ws = select_hyperparams(train_rows, cfg, nested=nested_lambda)
                fit_cfg = FitConfig(
                    name=cfg.name,
                    huber=cfg.huber,
                    orientation=cfg.orientation,
                    w_s=float(ws),
                    structure=cfg.structure,
                    select_ws=False,
                )
                fit = fit_hierarchical_config(train_plugs, fit_cfg, la, ls)
            pred_rows.extend(
                predict_table(
                    test_plugs,
                    fit.params,
                    fit.orient,
                    model=cfg.name,
                    fold_id=fold_id,
                    group_id=gid,
                    lambda_alpha=la,
                    lambda_s=ls,
                    w_s=ws,
                    opt_success=fit.success,
                    opt_message=fit.message,
                )
            )
            print(
                "  {} {:.1f}s lambda=({:.3g},{:.3g}) w_s={:g} ok={}".format(
                    cfg.name, time.time() - t0, la, ls, ws, fit.success
                ),
                flush=True,
            )
    return pd.DataFrame(pred_rows)


def lexicographic_pick(summary: pd.DataFrame, tol_frac: float = 0.05) -> Optional[str]:
    """
    Among non-P0 models with MAPE Vp within tol of best non-P0, pick lowest MAE Vp/Vs.
    """
    sub = summary[summary["model"] != "P0"].copy()
    if sub.empty:
        return None
    best_vp = float(sub["mape_vp_pct"].min())
    near = sub[sub["mape_vp_pct"] <= best_vp * (1.0 + tol_frac)]
    if near.empty:
        return None
    near = near.sort_values(["mae_vpvs", "mape_vp_pct"])
    return str(near.iloc[0]["model"])


def acceptance_gate(
    summary: pd.DataFrame,
    win_sum: pd.DataFrame,
    pred: pd.DataFrame,
    n_groups: int,
) -> pd.DataFrame:
    """
    Automatic acceptance gate vs P0.

    Requires simultaneously:
      - mape_vp <= P0
      - mape_vs < P0 OR mae_vpvs < P0
      - majority of depth groups beat P0 on mape_vp
      - frac_alpha_at_bound <= P0
      - opt_success rate >= 0.8 on OOF rows
    """
    p0 = summary[summary["model"] == "P0"]
    if p0.empty:
        return pd.DataFrame()
    p0r = p0.iloc[0]
    wins_map = {
        str(r["model"]): int(r["n_groups_beats_p0_vp"])
        for _, r in win_sum.iterrows()
    }
    rows: List[dict] = []
    for _, row in summary.iterrows():
        model = str(row["model"])
        if model == "P0":
            continue
        sub = pred[pred["model"] == model]
        opt_rate = float(np.mean(sub["opt_success"])) if len(sub) and "opt_success" in sub else 1.0
        n_win = int(wins_map.get(model, 0))
        vp_ok = float(row["mape_vp_pct"]) <= float(p0r["mape_vp_pct"])
        vs_ok = (float(row["mape_vs_pct"]) < float(p0r["mape_vs_pct"])) or (
            float(row["mae_vpvs"]) < float(p0r["mae_vpvs"])
        )
        maj_ok = n_win > (n_groups / 2.0)
        bound_ok = float(row["frac_alpha_at_bound"]) <= float(p0r["frac_alpha_at_bound"])
        opt_ok = opt_rate >= 0.8
        accepted = bool(vp_ok and vs_ok and maj_ok and bound_ok and opt_ok)
        rows.append(
            {
                "model": model,
                "vp_le_p0": int(vp_ok),
                "vs_or_vpvs_improved": int(vs_ok),
                "majority_groups": int(maj_ok),
                "bounds_le_p0": int(bound_ok),
                "opt_success_rate_ge_0p8": int(opt_ok),
                "n_groups_beats_p0_vp": n_win,
                "opt_success_rate": opt_rate,
                "accepted": int(accepted),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def main() -> None:
    """CLI for P0-P3 robust hierarchical POC."""
    parser = argparse.ArgumentParser(description="POC P0-P3 DEM calib (861).")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fix lambda=1 (skip nested lambda search).",
    )
    parser.add_argument(
        "--exclude-f2911v",
        action="store_true",
        help="Sensitivity run without F2911V.",
    )
    parser.add_argument(
        "--with-global",
        action="store_true",
        help="Also compare global (+ orientation) structures.",
    )
    parser.add_argument(
        "--sweep-ws",
        action="store_true",
        help="Also run fixed-weight exploratory P3_w* models (not nested w_s).",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated model names to run (default: all).",
    )
    args = parser.parse_args()

    clear_pred_cache()
    out_dir = OUT_ROOT / "v2"
    if args.exclude_f2911v:
        out_dir = out_dir / "robust_no_f2911v"
    if not args.fast:
        out_dir = out_dir / "nested"
    else:
        out_dir = out_dir / "fast"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    exclude = ["F2911V"] if args.exclude_f2911v else []
    ct_df = load_ct_samples()
    lab_val = pd.read_csv(LAB_VAL_CSV)
    plugs = build_plug_records(ct_df, lab_val, exclude_samples=exclude)
    depth_by_id = {
        str(r["ct_sample_id"]): float(r["ct_depth_m"]) for _, r in lab_val.iterrows()
    }
    rows = assign_depth_groups(plugs, depth_by_id)
    configs = build_configs(include_global=args.with_global, sweep_ws=args.sweep_ws)
    if args.only.strip():
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            raise ValueError("No configs matched --only={}".format(args.only))

    print("=== P0-P3 POC v2 (corrected) ===")
    print(
        "plugs={} groups={} nested={} configs={}".format(
            len(rows),
            len({r.group_id for r in rows}),
            (not args.fast),
            [c.name for c in configs],
        )
    )

    pred = run_cv(rows, configs, nested_lambda=(not args.fast))
    summary = summarize(pred)
    by_orient = summarize_by_orientation(pred)
    wins = wins_vs_p0(pred)
    win_sum = (
        wins.groupby("model", as_index=False)["beats_p0_vp"]
        .sum()
        .rename(columns={"beats_p0_vp": "n_groups_beats_p0_vp"})
    )
    n_groups = int(len({r.group_id for r in rows}))
    gate = acceptance_gate(summary, win_sum, pred, n_groups=n_groups)
    lex = lexicographic_pick(summary)

    p0 = summary[summary["model"] == "P0"].iloc[0]
    summary = summary.copy()
    summary["d_mape_vp_vs_p0"] = summary["mape_vp_pct"] - float(p0["mape_vp_pct"])
    summary["d_joint_vs_p0"] = summary["joint_rel_sq"] - float(p0["joint_rel_sq"])

    # Median selected w_s for P3 across folds (diagnostic only).
    p3_ws_median = float("nan")
    if "P3" in set(pred["model"]) and "w_s" in pred.columns:
        p3_ws_median = float(np.median(pred.loc[pred["model"] == "P3", "w_s"]))

    pred.to_csv(tables / "oof_predictions.csv", index=False, float_format="%.6f")
    summary.to_csv(tables / "summary_metrics.csv", index=False, float_format="%.6f")
    by_orient.to_csv(tables / "summary_by_orientation.csv", index=False, float_format="%.6f")
    wins.to_csv(tables / "group_mape_vp.csv", index=False, float_format="%.6f")
    win_sum.to_csv(tables / "wins_vs_p0.csv", index=False)
    gate.to_csv(tables / "acceptance_gate.csv", index=False, float_format="%.6f")

    accepted_models = [
        str(r["model"]) for _, r in gate.iterrows() if int(r["accepted"]) == 1
    ]
    meta = {
        "well_id": "861",
        "version": "v2_corrected",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nested_lambda": (not args.fast),
        "exclude_samples": exclude,
        "huber_delta": HUBER_DELTA,
        "lambda_beta": LAMBDA_BETA,
        "w_s_grid": list(W_S_GRID),
        "p3_median_w_s_oof": p3_ws_median,
        "lexicographic_winner": lex,
        "accepted_models": accepted_models,
        "production_reference": "P0",
        "fixes": [
            "data_loss normalized as (J_vp + w_s*J_vs)/(1+w_s)",
            "global model uses identifiable (alpha, scale) only",
            "global skips lambda selection",
            "P3 selects w_s inside outer-fold training",
            "optimizer success/finite checks with restarts",
            "acceptance gate implemented (not docstring-only)",
            "lexicographic_pick excludes P0",
        ],
        "pred_cache_size": len(_PRED_CACHE),
    }
    with open(out_dir / "metrics.json", "w", encoding="ascii") as handle:
        json.dump(meta, handle, indent=2)

    print("\n--- OOF summary ---")
    cols = [
        "model",
        "mape_vp_pct",
        "mape_vs_pct",
        "mae_vpvs",
        "joint_rel_sq",
        "frac_alpha_at_bound",
        "d_mape_vp_vs_p0",
        "n",
    ]
    print(summary[cols].to_string(index=False))
    print("\n--- wins vs P0 (MAPE Vp) ---")
    print(win_sum.to_string(index=False))
    print("\n--- acceptance gate ---")
    print(gate.to_string(index=False))
    print("\n--- by orientation ---")
    print(by_orient.to_string(index=False))
    print("\nlexicographic pick (non-P0):", lex)
    print("accepted models:", accepted_models)
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
