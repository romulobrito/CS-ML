#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POC: hierarchical + joint Vp/Vs DEM calibration (Well 861).

Compares four models under leave-one-depth/core-out (not leave-one-plug):
  M0 -- current: per-HFU, Vp-only (alpha + shared matrix scale)
  M1 -- per-HFU, joint Vp+Vs (relative squared errors)
  M2 -- hierarchical HFU sharing, Vp-only
  M3 -- hierarchical + joint Vp+Vs

Does not modify the production calibrator. ASCII-only.
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

ROOT = Path(__file__).resolve().parents[2]
LAB_VAL_CSV = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "lab_validation"
    / "tables"
    / "dem_vs_lab_validation.csv"
)
OUT_ROOT = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "calib_hier_joint_poc"
)

DEPTH_TOL_M = 0.15
# Nested grid (exclude lambda=0: removes hierarchy and was unstable in dry runs).
LAMBDA_GRID = (0.1, 1.0, 10.0, 100.0)
MODELS = ("M0", "M1", "M2", "M3")

# Cache DEM forward calls with exact float keys (no rounding: FD needs gradients).
_PRED_CACHE: Dict[Tuple[str, float, float], Tuple[float, float]] = {}


def clear_pred_cache() -> None:
    """Reset DEM prediction cache."""
    _PRED_CACHE.clear()


@dataclass(frozen=True)
class PlugRow:
    """Plug with depth for depth/core grouping."""

    record: PlugCalibRecord
    depth_m: float
    group_id: int


@dataclass(frozen=True)
class HfuParams:
    """Per-HFU alpha and common matrix scale s."""

    alpha: float
    scale: float


def _predict_vp_vs(
    plug: PlugCalibRecord,
    alpha: float,
    scale: float,
) -> Tuple[float, float]:
    """DEM dry Vp and Vs (km/s) for one plug (cached, exact keys)."""
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


def _rel_sq_err(pred: float, obs: float) -> float:
    """Squared relative residual; large penalty if obs invalid."""
    if obs <= 0.0:
        return 1.0e6
    return float(((pred - obs) / obs) ** 2)


def joint_data_loss(
    plugs: Sequence[PlugCalibRecord],
    params_by_hfu: Dict[int, HfuParams],
    use_vs: bool,
) -> float:
    """Mean relative squared error over plugs (Vp, optional Vs)."""
    if not plugs:
        return float("nan")
    vals: List[float] = []
    for plug in plugs:
        hp = params_by_hfu.get(plug.hfu)
        if hp is None:
            continue
        vp_p, vs_p = _predict_vp_vs(plug, hp.alpha, hp.scale)
        vals.append(_rel_sq_err(vp_p, plug.vp_lab_z_km_s))
        if use_vs:
            vals.append(_rel_sq_err(vs_p, plug.vs_lab_z_km_s))
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def assign_depth_groups(
    records: Sequence[PlugCalibRecord],
    depth_by_id: Dict[str, float],
    tol_m: float = DEPTH_TOL_M,
) -> List[PlugRow]:
    """
    Cluster plugs whose depths differ by at most tol_m.

    Keeps H/V pairs at nearly the same depth in one fold.
    """
    ordered = sorted(
        records,
        key=lambda p: (depth_by_id[p.ct_sample_id], p.ct_sample_id),
    )
    rows: List[PlugRow] = []
    group_id = -1
    last_depth = float("-inf")
    for plug in ordered:
        depth = float(depth_by_id[plug.ct_sample_id])
        if group_id < 0 or abs(depth - last_depth) > tol_m:
            group_id += 1
            last_depth = depth
        rows.append(PlugRow(record=plug, depth_m=depth, group_id=group_id))
    return rows


def _fallback_params(params: Dict[int, HfuParams]) -> HfuParams:
    """Mean alpha/scale over calibrated HFUs (HFU4-style fallback)."""
    if not params:
        return HfuParams(alpha=0.5, scale=1.0)
    alphas = [p.alpha for p in params.values()]
    scales = [p.scale for p in params.values()]
    return HfuParams(
        alpha=float(np.mean(alphas)),
        scale=float(np.mean(scales)),
    )


def fit_m0(plugs: Sequence[PlugCalibRecord]) -> Dict[int, HfuParams]:
    """Current production-like per-HFU Vp calibration."""
    by_hfu: Dict[int, List[PlugCalibRecord]] = {}
    for plug in plugs:
        by_hfu.setdefault(plug.hfu, []).append(plug)
    out: Dict[int, HfuParams] = {}
    for hfu, group in by_hfu.items():
        alpha_res = calibrate_hfu_alpha_only(group)
        joint_res = calibrate_hfu_alpha_matrix_scale(group)
        best = choose_best_scenario(alpha_res, joint_res)
        out[hfu] = HfuParams(
            alpha=best.alpha_calibrated,
            scale=best.matrix_k_scale,
        )
    return out


def fit_m1(plugs: Sequence[PlugCalibRecord]) -> Dict[int, HfuParams]:
    """Per-HFU joint Vp+Vs relative squared error."""
    by_hfu: Dict[int, List[PlugCalibRecord]] = {}
    for plug in plugs:
        by_hfu.setdefault(plug.hfu, []).append(plug)
    out: Dict[int, HfuParams] = {}
    for hfu, group in by_hfu.items():
        alpha_ct = float(np.median([p.alpha_ct for p in group]))

        def objective(x: np.ndarray) -> float:
            alpha = float(x[0])
            scale = float(x[1])
            return joint_data_loss(
                group,
                {hfu: HfuParams(alpha=alpha, scale=scale)},
                use_vs=True,
            )

        x0 = np.array([alpha_ct, 1.0], dtype=np.float64)
        res = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=[ALPHA_BOUNDS, MATRIX_SCALE_BOUNDS],
        )
        out[hfu] = HfuParams(
            alpha=float(np.clip(res.x[0], ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])),
            scale=float(np.clip(res.x[1], MATRIX_SCALE_BOUNDS[0], MATRIX_SCALE_BOUNDS[1])),
        )
    return out


def _pack_hier_x(
    hfus: Sequence[int],
    alpha0: float,
    s0: float,
    deltas_a: Dict[int, float],
    deltas_s: Dict[int, float],
) -> np.ndarray:
    """Pack hierarchical free parameters."""
    vals = [alpha0, s0]
    for hfu in hfus:
        vals.append(deltas_a[hfu])
        vals.append(deltas_s[hfu])
    return np.asarray(vals, dtype=np.float64)


def _unpack_hier_x(
    x: np.ndarray,
    hfus: Sequence[int],
) -> Tuple[float, float, Dict[int, float], Dict[int, float]]:
    """Unpack hierarchical free parameters."""
    alpha0 = float(x[0])
    s0 = float(x[1])
    deltas_a: Dict[int, float] = {}
    deltas_s: Dict[int, float] = {}
    idx = 2
    for hfu in hfus:
        deltas_a[hfu] = float(x[idx])
        deltas_s[hfu] = float(x[idx + 1])
        idx += 2
    return alpha0, s0, deltas_a, deltas_s


def _params_from_hier(
    alpha0: float,
    s0: float,
    deltas_a: Dict[int, float],
    deltas_s: Dict[int, float],
) -> Dict[int, HfuParams]:
    """Map hierarchical params to clipped per-HFU alpha/scale."""
    out: Dict[int, HfuParams] = {}
    for hfu in deltas_a:
        alpha = float(np.clip(alpha0 + deltas_a[hfu], ALPHA_BOUNDS[0], ALPHA_BOUNDS[1]))
        scale = float(
            np.clip(s0 + deltas_s[hfu], MATRIX_SCALE_BOUNDS[0], MATRIX_SCALE_BOUNDS[1])
        )
        out[hfu] = HfuParams(alpha=alpha, scale=scale)
    return out


def fit_hierarchical(
    plugs: Sequence[PlugCalibRecord],
    use_vs: bool,
    lambda_alpha: float,
    lambda_s: float,
) -> Dict[int, HfuParams]:
    """
    Hierarchical calibration of alpha_h = alpha0 + d_a_h, s_h = s0 + d_s_h.

    HFUs without plugs still receive (alpha0, s0) via global terms.
    """
    hfus = sorted({p.hfu for p in plugs})
    if not hfus:
        return {}

    alpha_ct = float(np.median([p.alpha_ct for p in plugs]))
    deltas_a0 = {h: 0.0 for h in hfus}
    deltas_s0 = {h: 0.0 for h in hfus}
    x0 = _pack_hier_x(hfus, alpha_ct, 1.0, deltas_a0, deltas_s0)

    # Bounds: alpha0, s0 in physical ranges; deltas free but modest.
    bounds: List[Tuple[float, float]] = [
        ALPHA_BOUNDS,
        MATRIX_SCALE_BOUNDS,
    ]
    for _ in hfus:
        bounds.append((-0.80, 0.80))
        bounds.append((-0.80, 0.80))

    def objective(x: np.ndarray) -> float:
        alpha0, s0, da, ds = _unpack_hier_x(x, hfus)
        params = _params_from_hier(alpha0, s0, da, ds)
        data = joint_data_loss(plugs, params, use_vs=use_vs)
        hier = float(lambda_alpha) * float(np.sum(np.asarray(list(da.values())) ** 2))
        hier += float(lambda_s) * float(np.sum(np.asarray(list(ds.values())) ** 2))
        return data + hier

    res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    alpha0, s0, da, ds = _unpack_hier_x(res.x, hfus)
    params = _params_from_hier(alpha0, s0, da, ds)
    # Ensure global defaults exist for missing HFUs (e.g. held-out HFU3).
    params[0] = HfuParams(
        alpha=float(np.clip(alpha0, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])),
        scale=float(np.clip(s0, MATRIX_SCALE_BOUNDS[0], MATRIX_SCALE_BOUNDS[1])),
    )
    return params


def select_lambda_nested(
    train_rows: Sequence[PlugRow],
    use_vs: bool,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
    max_inner_folds: int = 3,
) -> Tuple[float, float]:
    """
    Pick (lambda_alpha, lambda_s) by inner leave-one-depth on train.

    Uses up to max_inner_folds evenly spaced train groups for speed while
    remaining nested (lambdas chosen without the outer holdout).
    """
    groups = sorted({r.group_id for r in train_rows})
    if len(groups) < 2:
        return 1.0, 1.0

    if len(groups) > max_inner_folds:
        idxs = np.linspace(0, len(groups) - 1, max_inner_folds)
        groups = [groups[int(round(i))] for i in idxs]
        # unique preserve order
        seen = set()
        uniq: List[int] = []
        for g in groups:
            if g not in seen:
                seen.add(g)
                uniq.append(g)
        groups = uniq

    best_pair = (1.0, 1.0)
    best_score = float("inf")
    for la in lambda_grid:
        for ls in lambda_grid:
            fold_scores: List[float] = []
            for gid in groups:
                inner_train = [r.record for r in train_rows if r.group_id != gid]
                inner_test = [r.record for r in train_rows if r.group_id == gid]
                if not inner_train or not inner_test:
                    continue
                params = fit_hierarchical(
                    inner_train,
                    use_vs=use_vs,
                    lambda_alpha=la,
                    lambda_s=ls,
                )
                score = score_holdout(inner_test, params, use_vs=True)
                if np.isfinite(score):
                    fold_scores.append(score)
            if not fold_scores:
                continue
            mean_score = float(np.mean(fold_scores))
            if mean_score < best_score:
                best_score = mean_score
                best_pair = (float(la), float(ls))
    return best_pair


def resolve_params_for_hfu(
    params: Dict[int, HfuParams],
    hfu: int,
) -> HfuParams:
    """Return HFU params or hierarchical/global fallback."""
    if hfu in params:
        return params[hfu]
    if 0 in params:
        return params[0]
    return _fallback_params({k: v for k, v in params.items() if k != 0})


def score_holdout(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    use_vs: bool,
) -> float:
    """Joint relative squared loss on holdout plugs."""
    mapped: Dict[int, HfuParams] = {}
    for plug in plugs:
        mapped[plug.hfu] = resolve_params_for_hfu(params, plug.hfu)
    return joint_data_loss(plugs, mapped, use_vs=use_vs)


def predict_rows(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    model: str,
    fold_id: int,
    group_id: int,
    lambda_alpha: float,
    lambda_s: float,
) -> List[dict]:
    """Build per-plug prediction records for one fold."""
    rows: List[dict] = []
    for plug in plugs:
        hp = resolve_params_for_hfu(params, plug.hfu)
        vp_p, vs_p = _predict_vp_vs(plug, hp.alpha, hp.scale)
        vp_lab = plug.vp_lab_z_km_s
        vs_lab = plug.vs_lab_z_km_s
        vpvs_p = vp_p / vs_p if vs_p > 0 else float("nan")
        vpvs_lab = plug.vpvs_lab_z
        rows.append(
            {
                "model": model,
                "fold_id": fold_id,
                "group_id": group_id,
                "ct_sample_id": plug.ct_sample_id,
                "HFU": plug.hfu,
                "alpha": hp.alpha,
                "scale": hp.scale,
                "lambda_alpha": lambda_alpha,
                "lambda_s": lambda_s,
                "vp_lab_km_s": vp_lab,
                "vs_lab_km_s": vs_lab,
                "vpvs_lab": vpvs_lab,
                "vp_pred_km_s": vp_p,
                "vs_pred_km_s": vs_p,
                "vpvs_pred": vpvs_p,
                "vp_rel_err": (vp_p - vp_lab) / vp_lab if vp_lab > 0 else float("nan"),
                "vs_rel_err": (vs_p - vs_lab) / vs_lab if vs_lab > 0 else float("nan"),
                "vpvs_err": vpvs_p - vpvs_lab,
                "alpha_at_bound": int(
                    abs(hp.alpha - ALPHA_BOUNDS[0]) < 1.0e-6
                    or abs(hp.alpha - ALPHA_BOUNDS[1]) < 1.0e-6
                ),
            }
        )
    return rows


def fit_model(
    model: str,
    train_plugs: Sequence[PlugCalibRecord],
    train_rows: Sequence[PlugRow],
    nested_lambda: bool,
) -> Tuple[Dict[int, HfuParams], float, float]:
    """Fit one of M0-M3; returns params and lambdas used."""
    if model == "M0":
        return fit_m0(train_plugs), 0.0, 0.0
    if model == "M1":
        return fit_m1(train_plugs), 0.0, 0.0
    use_vs = model == "M3"
    if nested_lambda:
        la, ls = select_lambda_nested(train_rows, use_vs=use_vs)
    else:
        la, ls = 1.0, 1.0
    params = fit_hierarchical(
        train_plugs,
        use_vs=use_vs,
        lambda_alpha=la,
        lambda_s=ls,
    )
    return params, la, ls


def summarize_oof(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate OOF metrics per model."""
    rows: List[dict] = []
    for model, sub in pred_df.groupby("model"):
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
                "bias_vs_km_s": float(np.mean(vs_err)),
                "mae_vpvs": float(np.mean(np.abs(sub["vpvs_err"]))),
                "joint_rel_sq": float(np.mean(joint)),
                "frac_alpha_at_bound": float(np.mean(sub["alpha_at_bound"])),
                "alpha_std": float(np.std(sub["alpha"])),
                "scale_std": float(np.std(sub["scale"])),
            }
        )
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def run_cv(
    rows: Sequence[PlugRow],
    nested_lambda: bool,
    models: Sequence[str] = MODELS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-depth/core-out for all models."""
    groups = sorted({r.group_id for r in rows})
    pred_rows: List[dict] = []
    for fold_id, gid in enumerate(groups):
        print(
            "outer fold {}/{} group_id={} ...".format(fold_id + 1, len(groups), gid),
            flush=True,
        )
        train_rows = [r for r in rows if r.group_id != gid]
        test_rows = [r for r in rows if r.group_id == gid]
        train_plugs = [r.record for r in train_rows]
        test_plugs = [r.record for r in test_rows]
        for model in models:
            t0 = time.time()
            params, la, ls = fit_model(
                model,
                train_plugs,
                train_rows,
                nested_lambda=nested_lambda and model in ("M2", "M3"),
            )
            pred_rows.extend(
                predict_rows(
                    test_plugs,
                    params,
                    model=model,
                    fold_id=fold_id,
                    group_id=gid,
                    lambda_alpha=la,
                    lambda_s=ls,
                )
            )
            print(
                "  {} done in {:.1f}s (lambda=({:.3g},{:.3g}))".format(
                    model, time.time() - t0, la, ls
                ),
                flush=True,
            )
    pred_df = pd.DataFrame(pred_rows)
    summary = summarize_oof(pred_df)
    return pred_df, summary


def win_count_by_group(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Count depth groups where each model beats M0 on joint_rel_sq."""
    rows: List[dict] = []
    for gid, gdf in pred_df.groupby("group_id"):
        scores: Dict[str, float] = {}
        for model, sdf in gdf.groupby("model"):
            joint = 0.5 * (
                sdf["vp_rel_err"].to_numpy(dtype=np.float64) ** 2
                + sdf["vs_rel_err"].to_numpy(dtype=np.float64) ** 2
            )
            scores[str(model)] = float(np.mean(joint))
        m0 = scores.get("M0", float("nan"))
        for model, score in scores.items():
            rows.append(
                {
                    "group_id": int(gid),
                    "model": model,
                    "joint_rel_sq": score,
                    "beats_m0": int(np.isfinite(m0) and score < m0),
                }
            )
    return pd.DataFrame(rows)


def lambda_choice_summary(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Per-fold lambda choices for hierarchical models."""
    sub = pred_df[pred_df["model"].isin(["M2", "M3"])].copy()
    if sub.empty:
        return pd.DataFrame()
    return (
        sub.groupby(["model", "fold_id", "group_id"], as_index=False)
        .agg(
            lambda_alpha=("lambda_alpha", "first"),
            lambda_s=("lambda_s", "first"),
        )
        .sort_values(["model", "fold_id"])
        .reset_index(drop=True)
    )


def median_lambda_for_model(pred_df: pd.DataFrame, model: str) -> Tuple[float, float]:
    """Median selected lambdas across outer folds for one model."""
    sub = pred_df[pred_df["model"] == model]
    if sub.empty:
        return 1.0, 1.0
    la = float(np.median(sub["lambda_alpha"].to_numpy(dtype=np.float64)))
    ls = float(np.median(sub["lambda_s"].to_numpy(dtype=np.float64)))
    return la, ls


def fit_final_m3(
    plugs: Sequence[PlugCalibRecord],
    lambda_alpha: float,
    lambda_s: float,
) -> Tuple[Dict[int, HfuParams], float, float]:
    """Fit M3 on all plugs with fixed lambdas; returns params and globals."""
    params = fit_hierarchical(
        plugs,
        use_vs=True,
        lambda_alpha=lambda_alpha,
        lambda_s=lambda_s,
    )
    alpha0 = params[0].alpha if 0 in params else float(np.mean([p.alpha for p in params.values()]))
    s0 = params[0].scale if 0 in params else float(np.mean([p.scale for p in params.values()]))
    return params, alpha0, s0


def main() -> None:
    """CLI entry for hierarchical joint calibration POC."""
    parser = argparse.ArgumentParser(
        description="POC hierarchical + joint Vp/Vs DEM calibration (861)."
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip nested lambda search; use lambda=1 for M2/M3.",
    )
    parser.add_argument(
        "--exclude-f2911v",
        action="store_true",
        help="Robust check excluding F2911V.",
    )
    args = parser.parse_args()

    if not LAB_VAL_CSV.is_file():
        raise FileNotFoundError("Missing lab validation CSV: {}".format(LAB_VAL_CSV))

    clear_pred_cache()
    out_dir = OUT_ROOT / "nested" if not args.fast else OUT_ROOT
    if args.exclude_f2911v:
        out_dir = out_dir / "robust_no_f2911v"
    if args.fast and not args.exclude_f2911v:
        out_dir = OUT_ROOT
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    exclude = ["F2911V"] if args.exclude_f2911v else []
    ct_df = load_ct_samples()
    lab_val = pd.read_csv(LAB_VAL_CSV)
    plugs = build_plug_records(ct_df, lab_val, exclude_samples=exclude)
    depth_by_id = {
        str(r["ct_sample_id"]): float(r["ct_depth_m"])
        for _, r in lab_val.iterrows()
    }
    rows = assign_depth_groups(plugs, depth_by_id)
    group_map = (
        pd.DataFrame(
            [
                {
                    "ct_sample_id": r.record.ct_sample_id,
                    "depth_m": r.depth_m,
                    "HFU": r.record.hfu,
                    "group_id": r.group_id,
                }
                for r in rows
            ]
        )
        .sort_values(["group_id", "ct_sample_id"])
        .reset_index(drop=True)
    )

    nested = not args.fast
    print("=== DEM calib hierarchical+joint POC ===")
    print("plugs={}, depth_groups={}, nested_lambda={}".format(
        len(rows),
        group_map["group_id"].nunique(),
        nested,
    ))
    print(group_map.to_string(index=False))

    pred_df, summary = run_cv(rows, nested_lambda=nested)
    wins = win_count_by_group(pred_df)
    win_summary = (
        wins.groupby("model", as_index=False)["beats_m0"]
        .sum()
        .rename(columns={"beats_m0": "n_groups_beats_m0"})
    )
    lambda_df = lambda_choice_summary(pred_df)
    la_m3, ls_m3 = median_lambda_for_model(pred_df, "M3")

    m0 = summary[summary["model"] == "M0"].iloc[0]
    m3 = summary[summary["model"] == "M3"].iloc[0]
    joint_gain_pct = 100.0 * (
        float(m0["joint_rel_sq"]) - float(m3["joint_rel_sq"])
    ) / float(m0["joint_rel_sq"])

    group_map.to_csv(tables / "depth_groups.csv", index=False)
    pred_df.to_csv(tables / "oof_predictions.csv", index=False, float_format="%.6f")
    summary.to_csv(tables / "summary_metrics.csv", index=False, float_format="%.6f")
    wins.to_csv(tables / "group_joint_scores.csv", index=False, float_format="%.6f")
    win_summary.to_csv(tables / "win_vs_m0.csv", index=False)
    if not lambda_df.empty:
        lambda_df.to_csv(tables / "lambda_by_fold.csv", index=False, float_format="%.6f")

    meta = {
        "well_id": "861",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_plugs": len(rows),
        "n_depth_groups": int(group_map["group_id"].nunique()),
        "depth_tol_m": DEPTH_TOL_M,
        "nested_lambda": nested,
        "exclude_samples": exclude,
        "lambda_grid": list(LAMBDA_GRID),
        "models": list(MODELS),
        "m3_median_lambda_alpha": la_m3,
        "m3_median_lambda_s": ls_m3,
        "joint_gain_m3_vs_m0_pct": joint_gain_pct,
        "pred_cache_size": len(_PRED_CACHE),
        "criterion_note": (
            "Success if M3 cuts joint_rel_sq OOF by ~10%+ vs M0, "
            "improves most depth groups, and Vp does not degrade materially."
        ),
    }
    with open(out_dir / "metrics.json", "w", encoding="ascii") as handle:
        json.dump(meta, handle, indent=2)

    print("\n--- OOF summary ---")
    print(summary.to_string(index=False))
    print("\n--- groups beating M0 ---")
    print(win_summary.to_string(index=False))
    print(
        "\nM3 joint gain vs M0: {:.1f}% | median lambda=({:.3g}, {:.3g})".format(
            joint_gain_pct, la_m3, ls_m3
        )
    )
    print("cache entries: {}".format(len(_PRED_CACHE)))
    print("\nOutput: {}".format(out_dir))


if __name__ == "__main__":
    main()
