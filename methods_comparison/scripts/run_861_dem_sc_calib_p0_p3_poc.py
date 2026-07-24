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
    """Mean Vp (+ optional weighted Vs) loss over plugs."""
    if not plugs:
        return float("nan")
    vals: List[float] = []
    for plug in plugs:
        hp = resolve_params_for_hfu(params, plug.hfu)
        vp_d, vs_d = predict_dem(plug, hp.alpha, hp.scale)
        vp_p, vs_p = apply_orientation(vp_d, vs_d, plug.ct_sample_id, orient)
        if huber:
            vals.append(huber_rel(vp_p, plug.vp_lab_z_km_s))
            if w_s > 0.0:
                vals.append(w_s * huber_rel(vs_p, plug.vs_lab_z_km_s))
        else:
            vals.append(sq_rel(vp_p, plug.vp_lab_z_km_s))
            if w_s > 0.0:
                vals.append(w_s * sq_rel(vs_p, plug.vs_lab_z_km_s))
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def fit_p0(plugs: Sequence[PlugCalibRecord]) -> Tuple[Dict[int, HfuParams], Optional[OrientParams]]:
    """Baseline M0."""
    return fit_m0(plugs), None


def fit_hierarchical_config(
    plugs: Sequence[PlugCalibRecord],
    cfg: FitConfig,
    lambda_alpha: float,
    lambda_s: float,
) -> Tuple[Dict[int, HfuParams], Optional[OrientParams]]:
    """
    Fit hierarchical (or global) model with optional Huber, orientation, Vs weight.

    Parameterization: logistic alpha/scale; hierarchy on unconstrained deltas;
    scale penalty uses (log s_h - log s0)^2 via transformed values.
    """
    if cfg.structure == "global":
        hfus = [0]
    else:
        hfus = sorted({p.hfu for p in plugs})
        if not hfus:
            return {}, None

    alpha_ct = float(np.median([p.alpha_ct for p in plugs]))
    u0 = inv_logistic(alpha_ct, ALPHA_BOUNDS[0], ALPHA_BOUNDS[1])
    v0 = inv_logistic(1.0, MATRIX_SCALE_BOUNDS[0], MATRIX_SCALE_BOUNDS[1])

    # x = [u0, v0, du_h..., dv_h..., (beta_p, beta_s)?]
    x0_list = [u0, v0]
    for _ in hfus:
        x0_list.extend([0.0, 0.0])
    if cfg.orientation:
        x0_list.extend([0.0, 0.0])
    x0 = np.asarray(x0_list, dtype=np.float64)

    n_h = len(hfus)

    def unpack(x: np.ndarray) -> Tuple[Dict[int, HfuParams], Optional[OrientParams], float, float]:
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
            if cfg.structure == "global":
                params[0] = HfuParams(alpha=logistic_alpha(u_g + du), scale=logistic_scale(v_g + dv))
            else:
                params[hfu] = HfuParams(
                    alpha=logistic_alpha(u_g + du),
                    scale=logistic_scale(v_g + dv),
                )
        orient = None
        if cfg.orientation:
            orient = OrientParams(beta_p=float(x[idx]), beta_s=float(x[idx + 1]))
        # global fallback for missing HFU
        if cfg.structure != "global":
            params[0] = HfuParams(alpha=alpha0, scale=s0)
        return params, orient, alpha0, s0

    def objective(x: np.ndarray) -> float:
        params, orient, alpha0, s0 = unpack(x)
        # map HFU params for plugs when global: all use params[0]
        use_params = params
        if cfg.structure == "global":
            gp = params[0]
            use_params = {p.hfu: gp for p in plugs}
            use_params[0] = gp
        loss = data_loss(plugs, use_params, orient, cfg.huber, cfg.w_s)
        # hierarchy: deltas in probability space via alpha/s differences
        hier = 0.0
        if cfg.structure == "hfu":
            for hfu in hfus:
                hp = params[hfu]
                hier += float(lambda_alpha) * (hp.alpha - alpha0) ** 2
                hier += float(lambda_s) * (np.log(hp.scale) - np.log(s0)) ** 2
        if orient is not None:
            hier += float(LAMBDA_BETA) * (orient.beta_p ** 2 + orient.beta_s ** 2)
        return loss + hier

    res = minimize(objective, x0, method="L-BFGS-B")
    params, orient, _, _ = unpack(res.x)
    if cfg.structure == "global":
        gp = params[0]
        params = {h: gp for h in sorted({p.hfu for p in plugs})}
        params[0] = gp
    return params, orient


def select_lambda(
    train_rows: Sequence[PlugRow],
    cfg: FitConfig,
) -> Tuple[float, float]:
    """Nested lambda pick with up to MAX_INNER depth groups."""
    groups = sorted({r.group_id for r in train_rows})
    if len(groups) < 2:
        return 1.0, 1.0
    if len(groups) > MAX_INNER:
        idxs = np.linspace(0, len(groups) - 1, MAX_INNER)
        groups = sorted({groups[int(round(i))] for i in idxs})

    best = (1.0, 1.0)
    best_score = float("inf")
    for la in LAMBDA_GRID:
        for ls in LAMBDA_GRID:
            scores: List[float] = []
            for gid in groups:
                tr = [r.record for r in train_rows if r.group_id != gid]
                te = [r.record for r in train_rows if r.group_id == gid]
                if not tr or not te:
                    continue
                params, orient = fit_hierarchical_config(tr, cfg, la, ls)
                # Select lambda by Vp-only OOF (primary criterion).
                scores.append(score_holdout(te, params, orient, huber=False, w_s=0.0))
            if not scores:
                continue
            m = float(np.mean(scores))
            if m < best_score:
                best_score = m
                best = (float(la), float(ls))
    return best


def score_holdout(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    orient: Optional[OrientParams],
    huber: bool,
    w_s: float,
) -> float:
    """Joint relative squared score (always sq for fair comparison)."""
    return data_loss(plugs, params, orient, huber=False, w_s=max(w_s, 1.0e-12) if w_s > 0 else 0.0)


def predict_table(
    plugs: Sequence[PlugCalibRecord],
    params: Dict[int, HfuParams],
    orient: Optional[OrientParams],
    model: str,
    fold_id: int,
    group_id: int,
    lambda_alpha: float,
    lambda_s: float,
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


def build_configs(include_global: bool) -> List[FitConfig]:
    """P0-P3 configurations (P3 expands w_s)."""
    cfgs = [
        FitConfig("P0", huber=False, orientation=False, w_s=0.0, structure="hfu"),
        FitConfig("P1", huber=True, orientation=False, w_s=0.0, structure="hfu"),
        FitConfig("P2", huber=True, orientation=True, w_s=0.0, structure="hfu"),
    ]
    for w in W_S_GRID:
        if w <= 0.0:
            continue
        name = "P3_w{:g}".format(w).replace(".", "p")
        cfgs.append(
            FitConfig(name, huber=True, orientation=True, w_s=float(w), structure="hfu")
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
                params, orient = fit_p0(train_plugs)
                la, ls = 0.0, 0.0
            else:
                if nested_lambda:
                    la, ls = select_lambda(train_rows, cfg)
                else:
                    la, ls = 1.0, 1.0
                params, orient = fit_hierarchical_config(train_plugs, cfg, la, ls)
            pred_rows.extend(
                predict_table(
                    test_plugs,
                    params,
                    orient,
                    model=cfg.name,
                    fold_id=fold_id,
                    group_id=gid,
                    lambda_alpha=la,
                    lambda_s=ls,
                )
            )
            print(
                "  {} {:.1f}s lambda=({:.3g},{:.3g})".format(
                    cfg.name, time.time() - t0, la, ls
                ),
                flush=True,
            )
    return pd.DataFrame(pred_rows)


def lexicographic_pick(summary: pd.DataFrame, tol_frac: float = 0.05) -> Optional[str]:
    """
    Among models with MAPE Vp within tol of best, pick lowest MAE Vp/Vs.

    Excludes P0 from 'candidate improvements' but reports winner name.
    """
    sub = summary.copy()
    best_vp = float(sub["mape_vp_pct"].min())
    near = sub[sub["mape_vp_pct"] <= best_vp * (1.0 + tol_frac)]
    if near.empty:
        return None
    near = near.sort_values(["mae_vpvs", "mape_vp_pct"])
    return str(near.iloc[0]["model"])


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
        "--only",
        type=str,
        default="",
        help="Comma-separated model names to run (default: all).",
    )
    args = parser.parse_args()

    clear_pred_cache()
    out_dir = OUT_ROOT
    if args.exclude_f2911v:
        out_dir = OUT_ROOT / "robust_no_f2911v"
    if not args.fast:
        out_dir = out_dir / "nested"
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
    configs = build_configs(include_global=args.with_global)
    if args.only.strip():
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        configs = [c for c in configs if c.name in wanted]
        if not configs:
            raise ValueError("No configs matched --only={}".format(args.only))

    print("=== P0-P3 POC ===")
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
    lex = lexicographic_pick(summary)

    p0 = summary[summary["model"] == "P0"].iloc[0]
    summary = summary.copy()
    summary["d_mape_vp_vs_p0"] = summary["mape_vp_pct"] - float(p0["mape_vp_pct"])
    summary["d_joint_vs_p0"] = summary["joint_rel_sq"] - float(p0["joint_rel_sq"])

    pred.to_csv(tables / "oof_predictions.csv", index=False, float_format="%.6f")
    summary.to_csv(tables / "summary_metrics.csv", index=False, float_format="%.6f")
    by_orient.to_csv(tables / "summary_by_orientation.csv", index=False, float_format="%.6f")
    wins.to_csv(tables / "group_mape_vp.csv", index=False, float_format="%.6f")
    win_sum.to_csv(tables / "wins_vs_p0.csv", index=False)

    meta = {
        "well_id": "861",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nested_lambda": (not args.fast),
        "exclude_samples": exclude,
        "huber_delta": HUBER_DELTA,
        "lambda_beta": LAMBDA_BETA,
        "w_s_grid": list(W_S_GRID),
        "lexicographic_winner": lex,
        "criterion": (
            "Keep increment if Vp OOF <= P0 and Vs or Vp/Vs improves; "
            "majority of depth groups; less bound hitting."
        ),
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
    print("\n--- by orientation ---")
    print(by_orient.to_string(index=False))
    print("\nlexicographic pick:", lex)
    print("Output:", out_dir)


if __name__ == "__main__":
    main()
