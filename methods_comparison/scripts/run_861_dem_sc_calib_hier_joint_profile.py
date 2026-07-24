#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply final M3 hierarchical+joint DEM calibration to the 87-row profile.

DSI sonic is used only for evaluation (never for fitting).
Compares M3 dry and Gassmann against the production M0 baseline metrics.

Depends on:
  - nested (or fast) POC outputs for median lambdas, optional
  - lab CT HFU stats and sonic log

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dem_sc_861_calibrate import build_plug_records
from dem_sc_861_core import run_from_matrix_moduli_saturated
from ml_861_data import (
    DEM_SC_HFU_CALIB_ROOT,
    DEPTH_COL,
    DLIS_SONIC_CSV,
    load_ct_samples,
    load_logs_enriched,
)
from run_861_dem_sc_calib_hier_joint_poc import (
    LAB_VAL_CSV,
    OUT_ROOT,
    PlugRow,
    assign_depth_groups,
    clear_pred_cache,
    fit_final_m3,
    fit_hierarchical,
    select_lambda_nested,
)
from run_861_dem_sc_gassmann import load_pvt_config, PVT_DEFAULTS_JSON
from run_861_dem_sc_profile_87 import (
    HfuRockParams,
    build_hfu_params_from_lab_calib,
    process_profile,
)
from run_861_dlis_dem_validation import (
    build_validation_table,
    compute_metrics,
    merge_dem_sonic,
)

HFU_CT_STATS = (
    Path(__file__).resolve().parents[2]
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "poc_10plugs"
    / "tables"
    / "hfu_ct_stats.csv"
)
M0_HFU_CSV = DEM_SC_HFU_CALIB_ROOT / "hfu_lab_calibrated.csv"
M0_DRY_METRICS = (
    Path(__file__).resolve().parents[2]
    / "methods_comparison"
    / "data"
    / "processed"
    / "dlis_861"
    / "metrics_validation.json"
)
M0_GASS_METRICS = (
    Path(__file__).resolve().parents[2]
    / "methods_comparison"
    / "data"
    / "processed"
    / "dlis_861_gassmann"
    / "metrics_validation.json"
)
PHI_COL = "Phi_ND (pu)"
HFU_COL = "HFU"


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_nested_lambdas(nested_dir: Path) -> Optional[Tuple[float, float]]:
    """Read median M3 lambdas from nested POC metrics.json."""
    path = nested_dir / "metrics.json"
    if not path.is_file():
        return None
    meta = json.loads(path.read_text(encoding="ascii"))
    if "m3_median_lambda_alpha" not in meta:
        return None
    return (
        float(meta["m3_median_lambda_alpha"]),
        float(meta["m3_median_lambda_s"]),
    )


def build_m3_hfu_table(
    plugs: List,
    depth_rows: List[PlugRow],
    hfu_ct: pd.DataFrame,
    lambda_alpha: float,
    lambda_s: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Fit M3 on all plugs and build profile HFU parameter table.

    HFU4 uses hierarchical globals (alpha0, s0) on mean CT moduli of HFU2/3.
    """
    params, alpha0, s0 = fit_final_m3(plugs, lambda_alpha, lambda_s)
    ct_by = hfu_ct.set_index("HFU")
    rows: List[dict] = []

    for hfu in (1, 2, 3):
        if hfu not in params or hfu not in ct_by.index:
            continue
        hp = params[hfu]
        ct = ct_by.loc[hfu]
        k_ct = float(ct["matrix_k_gpa_median"])
        g_ct = float(ct["matrix_g_gpa_median"])
        rho = float(ct["matrix_rho_gcc_median"])
        n_ct = int(ct["n_plugs"])
        n_calib = sum(1 for p in plugs if p.hfu == hfu)
        rows.append(
            {
                "HFU": hfu,
                "alpha": hp.alpha,
                "alpha_ct_median": float(
                    np.median([p.alpha_ct for p in plugs if p.hfu == hfu])
                ),
                "matrix_k_gpa": k_ct * hp.scale,
                "matrix_g_gpa": g_ct * hp.scale,
                "matrix_k_gpa_ct": k_ct,
                "matrix_g_gpa_ct": g_ct,
                "matrix_k_scale": hp.scale,
                "matrix_g_scale": hp.scale,
                "matrix_rho_gcc": rho,
                "param_source": "m3_hier_joint_vp_vs",
                "n_ct_plugs": n_ct,
                "n_plugs_calib": n_calib,
                "lambda_alpha": lambda_alpha,
                "lambda_s": lambda_s,
            }
        )

    # HFU4: hierarchical prior (alpha0, s0) on avg CT of HFU2/3
    h2 = next((r for r in rows if r["HFU"] == 2), None)
    h3 = next((r for r in rows if r["HFU"] == 3), None)
    if h2 is not None and h3 is not None:
        rows.append(
            {
                "HFU": 4,
                "alpha": alpha0,
                "alpha_ct_median": float("nan"),
                "matrix_k_gpa": 0.5 * (h2["matrix_k_gpa_ct"] + h3["matrix_k_gpa_ct"]) * s0,
                "matrix_g_gpa": 0.5 * (h2["matrix_g_gpa_ct"] + h3["matrix_g_gpa_ct"]) * s0,
                "matrix_k_gpa_ct": 0.5 * (h2["matrix_k_gpa_ct"] + h3["matrix_k_gpa_ct"]),
                "matrix_g_gpa_ct": 0.5 * (h2["matrix_g_gpa_ct"] + h3["matrix_g_gpa_ct"]),
                "matrix_k_scale": s0,
                "matrix_g_scale": s0,
                "matrix_rho_gcc": 0.5 * (h2["matrix_rho_gcc"] + h3["matrix_rho_gcc"]),
                "param_source": "m3_hier_global_hfu4",
                "n_ct_plugs": 0,
                "n_plugs_calib": 0,
                "lambda_alpha": lambda_alpha,
                "lambda_s": lambda_s,
            }
        )

    meta = {
        "alpha0": alpha0,
        "s0": s0,
        "lambda_alpha": lambda_alpha,
        "lambda_s": lambda_s,
        "n_depth_groups_train": int(len({r.group_id for r in depth_rows})),
    }
    return pd.DataFrame(rows).sort_values("HFU").reset_index(drop=True), meta


def process_profile_gassmann(
    logs: pd.DataFrame,
    hfu_params: Dict[int, HfuRockParams],
    pvt: Dict[str, float],
) -> pd.DataFrame:
    """DEM dry + Gassmann on each depth (same interface as production)."""
    work = logs.sort_values(DEPTH_COL).reset_index(drop=True)
    rows: List[dict] = []
    for _, row in work.iterrows():
        depth = float(row[DEPTH_COL])
        hfu = int(row[HFU_COL])
        phi_nd = float(row[PHI_COL])
        if hfu not in hfu_params:
            rows.append({"Depth(m)": depth, "HFU": hfu, "status": "error", "error": "unknown_HFU"})
            continue
        hp = hfu_params[hfu]
        try:
            out = run_from_matrix_moduli_saturated(
                phi=phi_nd,
                alpha=hp.alpha,
                km_gpa=hp.matrix_k_gpa,
                gm_gpa=hp.matrix_g_gpa,
                rho_matrix_gcc=hp.matrix_rho_gcc,
                kf_gpa=pvt["kf_gpa"],
                rho_fluid_gcc=pvt["rho_fluid_gcc"],
                sw=pvt["sw"],
            )
            rows.append(
                {
                    "Depth(m)": depth,
                    "HFU": hfu,
                    "phi_input": phi_nd,
                    "alpha_hfu": hp.alpha,
                    "matrix_k_gpa": hp.matrix_k_gpa,
                    "matrix_g_gpa": hp.matrix_g_gpa,
                    "matrix_rho_gcc": hp.matrix_rho_gcc,
                    "param_source": hp.param_source,
                    **out,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Depth(m)": depth,
                    "HFU": hfu,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def metrics_from_profile(
    profile: pd.DataFrame,
    sonic: pd.DataFrame,
    vp_col: str,
    vs_col: Optional[str] = None,
) -> Dict[str, float]:
    """Merge profile with DSI and compute validation metrics."""
    dem = profile[profile["status"] == "ok"].copy()
    if vp_col != "vp_dem_km_s":
        dem["vp_dem_km_s"] = dem[vp_col]
    if vs_col is not None and vs_col in dem.columns and vs_col != "vs_dem_km_s":
        dem["vs_dem_km_s"] = dem[vs_col]
    if "vs_dem_km_s" not in dem.columns and "vs_dem_dry_km_s" in dem.columns:
        dem["vs_dem_km_s"] = dem["vs_dem_dry_km_s"]
    if "vpvs_dem" not in dem.columns:
        dem["vpvs_dem"] = dem["vp_dem_km_s"] / dem["vs_dem_km_s"]
    if "phi_input" not in dem.columns:
        dem["phi_input"] = np.nan
    if "Phi_Sonic (pu)" not in dem.columns:
        dem["Phi_Sonic (pu)"] = np.nan
    merged = merge_dem_sonic(dem, sonic, merge_tol_m=0.25)
    val = build_validation_table(merged)
    return compute_metrics(val)


def load_json_metrics(path: Path) -> Dict[str, float]:
    """Load existing M0 metrics JSON if present."""
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    """Fit final M3 and evaluate dry + Gassmann profiles vs DSI."""
    parser = argparse.ArgumentParser(
        description="Apply M3 hier+joint DEM params to profile vs DSI."
    )
    parser.add_argument(
        "--nested-dir",
        type=str,
        default=str(OUT_ROOT / "nested"),
        help="Directory with nested POC metrics.json",
    )
    parser.add_argument(
        "--lambda-alpha",
        type=float,
        default=None,
        help="Override lambda_alpha (skip nested median / search).",
    )
    parser.add_argument(
        "--lambda-s",
        type=float,
        default=None,
        help="Override lambda_s.",
    )
    args = parser.parse_args()

    clear_pred_cache()
    out_dir = OUT_ROOT / "profile_m3"
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    nested_dir = Path(args.nested_dir)
    la: Optional[float] = args.lambda_alpha
    ls: Optional[float] = args.lambda_s
    lambda_source = "cli"

    if la is None or ls is None:
        nested_l = load_nested_lambdas(nested_dir)
        if nested_l is not None:
            la, ls = nested_l
            lambda_source = "nested_median"
        else:
            print("Nested metrics not found; selecting lambda on all plugs...")
            ct_df0 = load_ct_samples()
            lab0 = pd.read_csv(LAB_VAL_CSV)
            plugs0 = build_plug_records(ct_df0, lab0)
            depth_by_id0 = {
                str(r["ct_sample_id"]): float(r["ct_depth_m"]) for _, r in lab0.iterrows()
            }
            rows0 = assign_depth_groups(plugs0, depth_by_id0)
            la, ls = select_lambda_nested(rows0, use_vs=True)
            lambda_source = "inner_loo_all_plugs"

    assert la is not None and ls is not None

    print("=== M3 profile evaluation vs DSI ===")
    print("lambda_alpha={:.4g}, lambda_s={:.4g} (source={})".format(la, ls, lambda_source))

    ct_df = load_ct_samples()
    lab_val = pd.read_csv(LAB_VAL_CSV)
    plugs = build_plug_records(ct_df, lab_val)
    depth_by_id = {
        str(r["ct_sample_id"]): float(r["ct_depth_m"]) for _, r in lab_val.iterrows()
    }
    depth_rows = assign_depth_groups(plugs, depth_by_id)
    hfu_ct = pd.read_csv(HFU_CT_STATS)

    hfu_m3, fit_meta = build_m3_hfu_table(plugs, depth_rows, hfu_ct, la, ls)
    hfu_m3.to_csv(tables / "hfu_m3_calibrated.csv", index=False, float_format="%.6f")
    print("\nHFU params (M3):")
    print(hfu_m3[["HFU", "alpha", "matrix_k_scale", "matrix_k_gpa", "param_source"]].to_string(index=False))

    hfu_params = build_hfu_params_from_lab_calib(hfu_m3)
    logs = load_logs_enriched()
    sonic = pd.read_csv(DLIS_SONIC_CSV).sort_values("depth_m").reset_index(drop=True)
    pvt = load_pvt_config(PVT_DEFAULTS_JSON)

    # Dry profile
    dry = process_profile(logs, hfu_params)
    dry.to_csv(tables / "861_dem_sc_profile_m3_dry.csv", index=False, float_format="%.6f")
    # process_profile returns vp_dem_km_s / vs_dem_km_s
    dry_metrics = metrics_from_profile(dry, sonic, vp_col="vp_dem_km_s", vs_col="vs_dem_km_s")

    # Gassmann profile
    gass = process_profile_gassmann(logs, hfu_params, pvt)
    gass.to_csv(
        tables / "861_dem_sc_profile_m3_gassmann.csv",
        index=False,
        float_format="%.6f",
    )
    # saturated velocities use vp_dem_km_s in core helper
    gass_metrics = metrics_from_profile(gass, sonic, vp_col="vp_dem_km_s", vs_col="vs_dem_km_s")

    m0_dry = load_json_metrics(M0_DRY_METRICS)
    m0_gass = load_json_metrics(M0_GASS_METRICS)

    compare_rows = [
        {
            "model": "M0_dry",
            "mape_vp_pct": m0_dry.get("mape_vp_pct"),
            "rmse_vp_km_s": m0_dry.get("rmse_vp_km_s"),
            "bias_vp_km_s": m0_dry.get("bias_vp_km_s"),
            "pearson_r_vp": m0_dry.get("pearson_r_vp"),
            "mae_vpvs": m0_dry.get("mae_vpvs"),
        },
        {
            "model": "M3_dry",
            "mape_vp_pct": dry_metrics.get("mape_vp_pct"),
            "rmse_vp_km_s": dry_metrics.get("rmse_vp_km_s"),
            "bias_vp_km_s": dry_metrics.get("bias_vp_km_s"),
            "pearson_r_vp": dry_metrics.get("pearson_r_vp"),
            "mae_vpvs": dry_metrics.get("mae_vpvs"),
        },
        {
            "model": "M0_gassmann",
            "mape_vp_pct": m0_gass.get("mape_vp_pct"),
            "rmse_vp_km_s": m0_gass.get("rmse_vp_km_s"),
            "bias_vp_km_s": m0_gass.get("bias_vp_km_s"),
            "pearson_r_vp": m0_gass.get("pearson_r_vp"),
            "mae_vpvs": m0_gass.get("mae_vpvs"),
        },
        {
            "model": "M3_gassmann",
            "mape_vp_pct": gass_metrics.get("mape_vp_pct"),
            "rmse_vp_km_s": gass_metrics.get("rmse_vp_km_s"),
            "bias_vp_km_s": gass_metrics.get("bias_vp_km_s"),
            "pearson_r_vp": gass_metrics.get("pearson_r_vp"),
            "mae_vpvs": gass_metrics.get("mae_vpvs"),
        },
    ]
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(tables / "dsi_comparison_m0_vs_m3.csv", index=False, float_format="%.6f")

    meta = {
        "well_id": "861",
        "generated_utc": utc_now_iso(),
        "lambda_source": lambda_source,
        "fit_meta": fit_meta,
        "dry_metrics_m3": dry_metrics,
        "gassmann_metrics_m3": gass_metrics,
        "note": "DSI used only for evaluation; never for calibration.",
    }
    with open(out_dir / "metrics.json", "w", encoding="ascii") as handle:
        json.dump(meta, handle, indent=2)

    print("\n--- DSI comparison ---")
    print(compare_df.to_string(index=False))
    print("\nOutput: {}".format(out_dir))


if __name__ == "__main__":
    main()
