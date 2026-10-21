#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2b: Gassmann saturation on lab-calibrated DEM profile (Well 861).

Reads PVT defaults, applies Gassmann to 87 profile rows, validates vs DSI sonic,
and compares dry vs saturated metrics.

Outputs:
  methods_comparison/data/processed/dem_sc_runs/profile_87_gassmann/
  methods_comparison/data/processed/dlis_861_gassmann/

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dem_sc_861_core import run_from_matrix_moduli_saturated
from ml_861_data import (
    DEM_SC_HFU_CALIB_ROOT,
    DEM_SC_PROFILE_LAB_CALIB_ROOT,
    DEPTH_COL,
    DLIS_PROCESSED_ROOT,
    DLIS_SONIC_CSV,
    load_logs_enriched,
)
from run_861_dem_sc_profile_87 import (
    HfuRockParams,
    build_hfu_params_from_lab_calib,
    build_validation_phi_sonic,
    load_hfu_lab_calibrated,
    pearson_r,
    plot_phi_nd_vs_phi_sonic,
    plot_vp_dem_vs_depth,
    plot_vpvs_vs_depth,
    summary_by_hfu,
)
from run_861_dlis_dem_validation import (
    build_validation_table,
    compute_metrics,
    load_dem_profile,
    load_sonic,
    merge_dem_sonic,
    plot_vp_dem_vs_sonic_depth,
    plot_vp_dem_vs_sonic_scatter,
    plot_vpvs_dem_vs_sonic_depth,
    plot_vp_rel_error_vs_depth,
    summary_by_hfu as dlis_summary_by_hfu,
)

ROOT = Path(__file__).resolve().parents[2]
PVT_DEFAULTS_JSON = ROOT / "methods_comparison" / "data" / "pvt_861_defaults.json"
HFU_LAB_CALIB_CSV = DEM_SC_HFU_CALIB_ROOT / "hfu_lab_calibrated.csv"

GASSMANN_PROFILE_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed" / "dem_sc_runs" / "profile_87_gassmann"
)
GASSMANN_DLIS_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed" / "dlis_861_gassmann"
)

PHI_COL = "Phi_ND (pu)"
PHI_SONIC_COL = "Phi_Sonic (pu)"
PHI_LAB_COL = "Phi_lab (pu)"
HFU_COL = "HFU"
DEFAULT_MERGE_TOL_M = 0.25


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pvt_config(path: Path) -> Dict[str, float]:
    """Load fluid PVT parameters for Gassmann."""
    if not path.is_file():
        raise FileNotFoundError("PVT config missing: {}".format(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "kf_gpa": float(data["kf_gpa"]),
        "rho_fluid_gcc": float(data["rho_fluid_gcc"]),
        "sw": float(data.get("sw", 1.0)),
        "fluid_name": str(data.get("fluid_name", "formation_brine")),
    }


def process_profile_gassmann(
    logs: pd.DataFrame,
    hfu_params: Dict[int, HfuRockParams],
    pvt: Dict[str, float],
) -> pd.DataFrame:
    """Run DEM dry + Gassmann saturation on each depth row."""
    required = (DEPTH_COL, HFU_COL, PHI_COL, PHI_SONIC_COL)
    missing = [c for c in required if c not in logs.columns]
    if missing:
        raise ValueError("Missing log columns: {}".format(missing))

    work = logs.sort_values(DEPTH_COL).reset_index(drop=True)
    rows: List[dict] = []
    for _, row in work.iterrows():
        depth = float(row[DEPTH_COL])
        hfu = int(row[HFU_COL])
        phi_nd = float(row[PHI_COL])
        phi_sonic = float(row[PHI_SONIC_COL])
        phi_lab = (
            float(row[PHI_LAB_COL])
            if PHI_LAB_COL in row and pd.notna(row[PHI_LAB_COL])
            else float("nan")
        )
        if hfu not in hfu_params:
            rows.append({
                "Depth(m)": depth,
                "HFU": hfu,
                "status": "error",
                "error": "unknown_HFU",
            })
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
            rec = {
                "Depth(m)": depth,
                "HFU": hfu,
                "hfu_source": "lab",
                "phi_input": phi_nd,
                "phi_input_source": "Phi_ND",
                "Phi_Sonic (pu)": phi_sonic,
                "Phi_lab (pu)": phi_lab,
                "alpha_hfu": hp.alpha,
                "matrix_k_gpa": hp.matrix_k_gpa,
                "matrix_g_gpa": hp.matrix_g_gpa,
                "matrix_rho_gcc": hp.matrix_rho_gcc,
                "param_source": hp.param_source,
                "n_ct_plugs_calib": hp.n_ct_plugs,
                "fluid_name": pvt["fluid_name"],
                **out,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            rec = {
                "Depth(m)": depth,
                "HFU": hfu,
                "status": "error",
                "error": str(exc),
            }
        rows.append(rec)
    return pd.DataFrame(rows)


def load_dry_profile_for_compare() -> pd.DataFrame:
    """Load lab-calibrated dry profile for before/after metrics."""
    path = DEM_SC_PROFILE_LAB_CALIB_ROOT / "tables" / "861_dem_sc_profile.csv"
    if not path.is_file():
        raise FileNotFoundError("Dry profile missing: {}".format(path))
    df = pd.read_csv(path)
    ok = df[df["status"] == "ok"].copy()
    return ok


def compare_dry_vs_sat_metrics(
    dry_profile: pd.DataFrame,
    sat_profile: pd.DataFrame,
    sonic_csv: Path,
    merge_tol_m: float,
) -> Dict[str, object]:
    """Compute validation metrics for dry and saturated profiles vs sonic."""
    sonic = load_sonic(sonic_csv)
    out: Dict[str, object] = {"merge_tol_m": merge_tol_m}

    for label, prof in (("dry", dry_profile), ("saturated", sat_profile)):
        dem = prof.rename(columns={})
        merged = merge_dem_sonic(dem, sonic, merge_tol_m)
        val = build_validation_table(merged)
        metrics = compute_metrics(val)
        metrics["generated_utc"] = utc_now_iso()
        out[label] = metrics
    return out


def plot_dry_vs_sat_depth(
    dry_val: pd.DataFrame,
    sat_val: pd.DataFrame,
    out_path: Path,
) -> None:
    """Depth track: sonic vs dry DEM vs saturated DEM."""
    dry_ok = dry_val[dry_val["has_sonic_vp"]].sort_values("Depth(m)")
    sat_ok = sat_val[sat_val["has_sonic_vp"]].sort_values("Depth(m)")
    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    ax.plot(
        dry_ok["vp_sonic_km_s"],
        dry_ok["Depth(m)"],
        "o-",
        color="#2ca02c",
        label="Vp sonic (DLIS)",
        markersize=3,
        linewidth=1.0,
    )
    ax.plot(
        dry_ok["vp_dem_km_s"],
        dry_ok["Depth(m)"],
        "s--",
        color="#1f77b4",
        label="Vp DEM dry",
        markersize=3,
        linewidth=1.0,
    )
    ax.plot(
        sat_ok["vp_dem_km_s"],
        sat_ok["Depth(m)"],
        "d-",
        color="#d62728",
        label="Vp DEM Gassmann",
        markersize=3,
        linewidth=1.2,
    )
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: dry vs Gassmann vs DSI sonic")
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description="Well 861 Gassmann phase 2b pipeline.")
    p.add_argument(
        "--pvt-json",
        type=str,
        default=str(PVT_DEFAULTS_JSON),
        help="PVT/fluid JSON for Gassmann.",
    )
    p.add_argument(
        "--merge-tol-m",
        type=float,
        default=DEFAULT_MERGE_TOL_M,
        help="Depth tolerance for sonic merge (m).",
    )
    p.add_argument("--smoke", action="store_true", help="First 15 rows only.")
    return p.parse_args()


def main() -> None:
    """Execute Gassmann profile + DLIS validation."""
    args = parse_args()
    pvt_path = Path(args.pvt_json)
    pvt = load_pvt_config(pvt_path)

    prof_root = GASSMANN_PROFILE_ROOT
    prof_tables = prof_root / "tables"
    prof_figures = prof_root / "figures"
    dlis_root = GASSMANN_DLIS_ROOT
    dlis_tables = dlis_root / "tables"
    dlis_figures = dlis_root / "figures"
    for d in (prof_tables, prof_figures, dlis_tables, dlis_figures):
        d.mkdir(parents=True, exist_ok=True)

    lab_hfu = load_hfu_lab_calibrated(HFU_LAB_CALIB_CSV)
    hfu_params = build_hfu_params_from_lab_calib(lab_hfu)
    lab_hfu.to_csv(prof_tables / "hfu_matrix_moduli.csv", index=False, float_format="%.6f")

    logs = load_logs_enriched()
    if args.smoke:
        logs = logs.head(15).copy()

    profile = process_profile_gassmann(logs, hfu_params, pvt)
    profile.to_csv(prof_tables / "861_dem_sc_profile_gassmann.csv", index=False, float_format="%.6f")
    profile.to_excel(prof_tables / "861_dem_sc_profile_gassmann.xlsx", index=False)

    ok = profile[profile["status"] == "ok"]
    if ok.empty:
        raise RuntimeError("All Gassmann profile rows failed")

    validation_phi = build_validation_phi_sonic(profile)
    validation_phi.to_csv(prof_tables / "validation_phi_sonic.csv", index=False, float_format="%.6f")
    by_hfu = summary_by_hfu(profile)
    by_hfu.to_csv(prof_tables / "summary_by_hfu.csv", index=False, float_format="%.6f")

    plot_vp_dem_vs_depth(profile, prof_figures / "vp_dem_vs_depth.png")
    plot_vpvs_vs_depth(profile, prof_figures / "vpvs_vs_depth.png")
    plot_phi_nd_vs_phi_sonic(profile, prof_figures / "phi_nd_vs_phi_sonic.png")

    val_global = validation_phi[
        (validation_phi["scope"] == "global") & (validation_phi["HFU"] == "all")
    ]
    r_sonic = float("nan")
    if "pearson_r_phi_nd_vs_phi_sonic" in val_global.columns and len(val_global) > 0:
        r_sonic = float(val_global.iloc[0]["pearson_r_phi_nd_vs_phi_sonic"])

    profile_metrics = {
        "n_rows": int(len(logs)),
        "n_ok": int(len(ok)),
        "param_source": "lab_calibrated_gassmann",
        "kf_gpa": pvt["kf_gpa"],
        "rho_fluid_gcc": pvt["rho_fluid_gcc"],
        "sw": pvt["sw"],
        "fluid_name": pvt["fluid_name"],
        "pvt_config": str(pvt_path),
        "pearson_r_phi_nd_vs_phi_sonic": r_sonic,
        "mean_vp_dem_km_s": float(ok["vp_dem_km_s"].mean()),
        "mean_vp_dem_dry_km_s": float(ok["vp_dem_dry_km_s"].mean()),
        "mean_vpvs_dem": float(ok["vpvs_dem"].mean()),
        "mean_delta_vp_km_s_sat_minus_dry": float(
            (ok["vp_dem_km_s"] - ok["vp_dem_dry_km_s"]).mean()
        ),
        "generated_utc": utc_now_iso(),
    }
    (prof_root / "metrics.json").write_text(
        json.dumps(profile_metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    sonic = load_sonic(DLIS_SONIC_CSV)
    merged = merge_dem_sonic(ok, sonic, args.merge_tol_m)
    val_table = build_validation_table(merged)
    val_table.to_csv(dlis_tables / "dem_vs_sonic_validation.csv", index=False, float_format="%.6f")
    sat_metrics = compute_metrics(val_table)
    sat_metrics["generated_utc"] = utc_now_iso()
    sat_metrics["fluid_name"] = pvt["fluid_name"]
    sat_metrics["kf_gpa"] = pvt["kf_gpa"]
    (dlis_root / "metrics_validation.json").write_text(
        json.dumps(sat_metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    hfu_summary = dlis_summary_by_hfu(val_table)
    hfu_summary.to_csv(dlis_tables / "summary_by_hfu.csv", index=False, float_format="%.6f")

    plot_vp_dem_vs_sonic_depth(val_table, dlis_figures / "vp_dem_vs_sonic_depth.png")
    plot_vp_dem_vs_sonic_scatter(val_table, dlis_figures / "vp_dem_vs_sonic_scatter.png")
    plot_vpvs_dem_vs_sonic_depth(val_table, dlis_figures / "vpvs_dem_vs_sonic_depth.png")
    plot_vp_rel_error_vs_depth(val_table, dlis_figures / "vp_rel_error_vs_depth.png")

    dry_profile = load_dry_profile_for_compare()
    compare = compare_dry_vs_sat_metrics(
        dry_profile,
        ok,
        DLIS_SONIC_CSV,
        args.merge_tol_m,
    )
    compare["generated_utc"] = utc_now_iso()
    compare["pvt_config"] = str(pvt_path)
    (dlis_root / "metrics_dry_vs_saturated.json").write_text(
        json.dumps(compare, indent=2) + "\n",
        encoding="utf-8",
    )

    dry_merged = merge_dem_sonic(dry_profile, sonic, args.merge_tol_m)
    dry_val = build_validation_table(dry_merged)
    plot_dry_vs_sat_depth(
        dry_val,
        val_table,
        dlis_figures / "vp_dry_vs_gassmann_vs_sonic_depth.png",
    )

    manifest_lines = [
        "Well 861 -- Etapa 2b Gassmann profile + DLIS validation",
        "Generated: {}".format(utc_now_iso()),
        "PVT: {} (Kf={} GPa, rho_f={} g/cc, Sw={})".format(
            pvt["fluid_name"],
            pvt["kf_gpa"],
            pvt["rho_fluid_gcc"],
            pvt["sw"],
        ),
        "Rows OK: {}/{}".format(len(ok), len(logs)),
        "Dry bias Vp vs sonic (km/s): {:.3f}".format(
            float(compare["dry"]["bias_vp_km_s"])
        ),
        "Saturated bias Vp vs sonic (km/s): {:.3f}".format(
            float(compare["saturated"]["bias_vp_km_s"])
        ),
    ]
    (dlis_root / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    print("Gassmann profile: {}/{} rows OK".format(len(ok), len(logs)))
    print("Mean Vp dry: {:.3f} km/s -> saturated: {:.3f} km/s".format(
        profile_metrics["mean_vp_dem_dry_km_s"],
        profile_metrics["mean_vp_dem_km_s"],
    ))
    print("DLIS validation (saturated): MAPE={:.1f}% bias={:+.3f} km/s".format(
        sat_metrics.get("mape_vp_pct", float("nan")),
        sat_metrics.get("bias_vp_km_s", float("nan")),
    ))
    print("DLIS validation (dry):       MAPE={:.1f}% bias={:+.3f} km/s".format(
        float(compare["dry"]["mape_vp_pct"]),
        float(compare["dry"]["bias_vp_km_s"]),
    ))
    print("Output profile: {}".format(prof_tables / "861_dem_sc_profile_gassmann.csv"))
    print("Output validation: {}".format(dlis_tables / "dem_vs_sonic_validation.csv"))


if __name__ == "__main__":
    main()
