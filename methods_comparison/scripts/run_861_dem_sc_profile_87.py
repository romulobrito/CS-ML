#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2c/2d: DEM/SC profile extrapolation (87 rows) + Phi_Sonic validation.

Uses HFU lab + Phi_ND + CT-calibrated parameters per HFU.
Outputs under methods_comparison/data/processed/dem_sc_runs/profile_87/

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dem_sc_861_core import run_from_matrix_moduli
from ml_861_data import (
    DEM_SC_HFU_CALIB_ROOT,
    DEM_SC_PROFILE_LAB_CALIB_ROOT,
    DEM_SC_PROFILE_ROOT,
    DEPTH_COL,
    load_ct_samples,
    load_logs_enriched,
)

PHI_COL = "Phi_ND (pu)"
PHI_SONIC_COL = "Phi_Sonic (pu)"
PHI_LAB_COL = "Phi_lab (pu)"
HFU_COL = "HFU"

PROFILE_TABLES = DEM_SC_PROFILE_ROOT / "tables"
PROFILE_FIGURES = DEM_SC_PROFILE_ROOT / "figures"
HFU_CALIB_CSV = DEM_SC_HFU_CALIB_ROOT / "hfu_ct_stats.csv"
HFU_LAB_CALIB_CSV = DEM_SC_HFU_CALIB_ROOT / "hfu_lab_calibrated.csv"

REQUIRED_LOG_COLS: Tuple[str, ...] = (
    DEPTH_COL,
    HFU_COL,
    PHI_COL,
    PHI_SONIC_COL,
)


@dataclass(frozen=True)
class HfuRockParams:
    """Rock-physics parameters assigned to one HFU."""

    hfu: int
    alpha: float
    matrix_k_gpa: float
    matrix_g_gpa: float
    matrix_rho_gcc: float
    param_source: str
    n_ct_plugs: int


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create profile output directories."""
    for d in (DEM_SC_PROFILE_ROOT, PROFILE_TABLES, PROFILE_FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def load_hfu_ct_stats(path: Optional[Path] = None) -> pd.DataFrame:
    """Load HFU calibration table from POC."""
    p = path if path is not None else HFU_CALIB_CSV
    if not p.is_file():
        raise FileNotFoundError(
            "HFU calibration missing: {}. Run run_861_dem_sc_poc_plugs.py first.".format(p)
        )
    return pd.read_csv(p)


def global_alpha_median(ct_df: pd.DataFrame) -> float:
    """Global median aspect ratio across CT plugs."""
    return float(ct_df["ct_ar_mean"].median())


def build_hfu_params_from_lab_calib(lab_hfu_df: pd.DataFrame) -> Dict[int, HfuRockParams]:
    """Build per-HFU parameters from lab-calibrated table."""
    params: Dict[int, HfuRockParams] = {}
    for _, row in lab_hfu_df.iterrows():
        hfu = int(row["HFU"])
        params[hfu] = HfuRockParams(
            hfu=hfu,
            alpha=float(row["alpha"]),
            matrix_k_gpa=float(row["matrix_k_gpa"]),
            matrix_g_gpa=float(row["matrix_g_gpa"]),
            matrix_rho_gcc=float(row["matrix_rho_gcc"]),
            param_source=str(row.get("param_source", "lab_calibrated")),
            n_ct_plugs=int(row.get("n_ct_plugs", 0)),
        )
    return params


def load_hfu_lab_calibrated(path: Optional[Path] = None) -> pd.DataFrame:
    """Load lab-calibrated HFU table for profile extrapolation."""
    p = path if path is not None else HFU_LAB_CALIB_CSV
    if not p.is_file():
        raise FileNotFoundError(
            "Lab HFU calibration missing: {}. Run run_861_dem_sc_lab_calibration.py.".format(
                p
            )
        )
    return pd.read_csv(p)


def build_hfu_params(hfu_stats: pd.DataFrame, ct_df: pd.DataFrame) -> Dict[int, HfuRockParams]:
    """
    Build per-HFU parameters from CT stats; HFU4 uses documented fallback.
    """
    params: Dict[int, HfuRockParams] = {}
    alpha_global = global_alpha_median(ct_df)

    for _, row in hfu_stats.iterrows():
        hfu = int(row["HFU"])
        params[hfu] = HfuRockParams(
            hfu=hfu,
            alpha=float(row["alpha_median"]),
            matrix_k_gpa=float(row["matrix_k_gpa_median"]),
            matrix_g_gpa=float(row["matrix_g_gpa_median"]),
            matrix_rho_gcc=float(row["matrix_rho_gcc_median"]),
            param_source="ct_plug_median",
            n_ct_plugs=int(row["n_plugs"]),
        )

    if 4 not in params:
        h2 = params.get(2)
        h3 = params.get(3)
        if h2 is None or h3 is None:
            raise ValueError("HFU4 fallback requires HFU2 and HFU3 calibration")
        params[4] = HfuRockParams(
            hfu=4,
            alpha=alpha_global,
            matrix_k_gpa=0.5 * (h2.matrix_k_gpa + h3.matrix_k_gpa),
            matrix_g_gpa=0.5 * (h2.matrix_g_gpa + h3.matrix_g_gpa),
            matrix_rho_gcc=0.5 * (h2.matrix_rho_gcc + h3.matrix_rho_gcc),
            param_source="fallback_hfu4_avg_hfu2_hfu3",
            n_ct_plugs=0,
        )
    return params


def hfu_params_to_table(params: Dict[int, HfuRockParams]) -> pd.DataFrame:
    """Export HFU parameter table."""
    rows = [
        {
            "HFU": p.hfu,
            "alpha": p.alpha,
            "matrix_k_gpa": p.matrix_k_gpa,
            "matrix_g_gpa": p.matrix_g_gpa,
            "matrix_rho_gcc": p.matrix_rho_gcc,
            "param_source": p.param_source,
            "n_ct_plugs": p.n_ct_plugs,
        }
        for p in sorted(params.values(), key=lambda x: x.hfu)
    ]
    return pd.DataFrame(rows)


def process_profile(
    logs: pd.DataFrame,
    hfu_params: Dict[int, HfuRockParams],
) -> pd.DataFrame:
    """Run DEM/SC on each depth row."""
    missing = [c for c in REQUIRED_LOG_COLS if c not in logs.columns]
    if missing:
        raise ValueError("Missing log columns: {}".format(missing))

    work = logs.sort_values(DEPTH_COL).reset_index(drop=True)
    rows: List[dict] = []
    for _, row in work.iterrows():
        depth = float(row[DEPTH_COL])
        hfu = int(row[HFU_COL])
        phi_nd = float(row[PHI_COL])
        phi_sonic = float(row[PHI_SONIC_COL])
        phi_lab = float(row[PHI_LAB_COL]) if PHI_LAB_COL in row and pd.notna(row[PHI_LAB_COL]) else np.nan

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
            out = run_from_matrix_moduli(
                phi=phi_nd,
                alpha=hp.alpha,
                km_gpa=hp.matrix_k_gpa,
                gm_gpa=hp.matrix_g_gpa,
                rho_matrix_gcc=hp.matrix_rho_gcc,
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


def pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Safe Pearson correlation."""
    if len(x) < 2:
        return float("nan")
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_validation_phi_sonic(profile: pd.DataFrame) -> pd.DataFrame:
    """Validation metrics: phi_input vs Phi_Sonic (global and by HFU)."""
    ok = profile[profile["status"] == "ok"].copy()
    rows: List[dict] = []

    x_all = ok["phi_input"].to_numpy(dtype=np.float64)
    y_all = ok["Phi_Sonic (pu)"].to_numpy(dtype=np.float64)
    rows.append({
        "scope": "global",
        "HFU": "all",
        "n": len(ok),
        "pearson_r_phi_nd_vs_phi_sonic": pearson_r(x_all, y_all),
        "rmse_phi_nd_vs_phi_sonic": float(np.sqrt(np.mean((x_all - y_all) ** 2))),
        "mean_phi_nd": float(np.mean(x_all)),
        "mean_phi_sonic": float(np.mean(y_all)),
    })

    if PHI_LAB_COL in ok.columns:
        lab_mask = ok["Phi_lab (pu)"].notna()
        if lab_mask.sum() >= 2:
            x_lab = ok.loc[lab_mask, "phi_input"].to_numpy(dtype=np.float64)
            y_lab = ok.loc[lab_mask, "Phi_lab (pu)"].to_numpy(dtype=np.float64)
            rows.append({
                "scope": "global",
                "HFU": "all",
                "n": int(lab_mask.sum()),
                "pearson_r_phi_nd_vs_phi_lab": pearson_r(x_lab, y_lab),
                "rmse_phi_nd_vs_phi_lab": float(np.sqrt(np.mean((x_lab - y_lab) ** 2))),
                "mean_phi_nd": float(np.mean(x_lab)),
                "mean_phi_lab": float(np.mean(y_lab)),
            })

    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        xs = sub["phi_input"].to_numpy(dtype=np.float64)
        ys = sub["Phi_Sonic (pu)"].to_numpy(dtype=np.float64)
        rows.append({
            "scope": "by_hfu",
            "HFU": int(hfu),
            "n": len(sub),
            "pearson_r_phi_nd_vs_phi_sonic": pearson_r(xs, ys),
            "rmse_phi_nd_vs_phi_sonic": float(np.sqrt(np.mean((xs - ys) ** 2))),
            "mean_phi_nd": float(np.mean(xs)),
            "mean_phi_sonic": float(np.mean(ys)),
            "mean_vp_dem_km_s": float(sub["vp_dem_km_s"].mean()),
            "mean_vpvs_dem": float(sub["vpvs_dem"].mean()),
        })

    return pd.DataFrame(rows)


def summary_by_hfu(profile: pd.DataFrame) -> pd.DataFrame:
    """Aggregate Vp/Vs by HFU."""
    ok = profile[profile["status"] == "ok"]
    agg = ok.groupby("HFU").agg(
        n=("Depth(m)", "count"),
        depth_min=("Depth(m)", "min"),
        depth_max=("Depth(m)", "max"),
        phi_input_mean=("phi_input", "mean"),
        phi_sonic_mean=("Phi_Sonic (pu)", "mean"),
        vp_dem_km_s_mean=("vp_dem_km_s", "mean"),
        vp_dem_km_s_std=("vp_dem_km_s", "std"),
        vpvs_dem_mean=("vpvs_dem", "mean"),
        vpvs_dem_std=("vpvs_dem", "std"),
        alpha_hfu=("alpha_hfu", "first"),
        param_source=("param_source", "first"),
    )
    return agg.reset_index()


def plot_vp_dem_vs_depth(profile: pd.DataFrame, out_path: Path) -> None:
    """Vp DEM vs depth colored by HFU."""
    ok = profile[profile["status"] == "ok"].sort_values("Depth(m)")
    fig, ax = plt.subplots(figsize=(5.0, 8.0))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        ax.plot(
            sub["vp_dem_km_s"],
            sub["Depth(m)"],
            "o-",
            label="HFU{}".format(hfu),
            color=colors.get(int(hfu), "#333333"),
            markersize=4,
        )
    ax.set_xlabel("Vp DEM (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: theoretical Vp (DEM) vs depth")
    ax.invert_yaxis()
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vpvs_vs_depth(profile: pd.DataFrame, out_path: Path) -> None:
    """Vp/Vs DEM vs depth."""
    ok = profile[profile["status"] == "ok"].sort_values("Depth(m)")
    fig, ax = plt.subplots(figsize=(5.0, 8.0))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        ax.plot(
            sub["vpvs_dem"],
            sub["Depth(m)"],
            "o-",
            label="HFU{}".format(hfu),
            color=colors.get(int(hfu), "#333333"),
            markersize=4,
        )
    ax.set_xlabel("Vp/Vs DEM")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: theoretical Vp/Vs vs depth")
    ax.invert_yaxis()
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_phi_nd_vs_phi_sonic(profile: pd.DataFrame, out_path: Path) -> None:
    """Scatter Phi_ND vs Phi_Sonic."""
    ok = profile[profile["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        ax.scatter(
            sub["Phi_Sonic (pu)"],
            sub["phi_input"],
            label="HFU{}".format(hfu),
            c=colors.get(int(hfu), "#333333"),
            s=50,
            alpha=0.8,
            edgecolors="k",
            linewidths=0.3,
        )
    lim_lo = min(ok["phi_input"].min(), ok["Phi_Sonic (pu)"].min()) * 0.9
    lim_hi = max(ok["phi_input"].max(), ok["Phi_Sonic (pu)"].max()) * 1.1
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, label="1:1")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Phi_Sonic (pu)")
    ax.set_ylabel("Phi_ND input (pu)")
    ax.set_title("Validation: porosity input vs sonic porosity")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vpvs_by_hfu(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart mean Vp/Vs by HFU with std error."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = np.arange(len(summary))
    ax.bar(
        x,
        summary["vpvs_dem_mean"],
        yerr=summary["vpvs_dem_std"],
        capsize=4,
        color="#4c72b0",
        edgecolor="k",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(["HFU{}".format(int(h)) for h in summary["HFU"]])
    ax.set_ylabel("Mean Vp/Vs DEM")
    ax.set_title("Well 861: Vp/Vs by HFU (87-row profile)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_manifest(n_ok: int, n_total: int, metrics: Dict[str, object]) -> None:
    """Write profile MANIFEST."""
    lines = [
        "Well 861 -- Etapa 2c/2d profile (87 rows) + Phi_Sonic validation",
        "Generated: {}".format(utc_now_iso()),
        "",
        "Rows processed: {}/{}".format(n_ok, n_total),
        "Phi input: Phi_ND (pu)",
        "HFU source: laboratory",
        "Pearson r (Phi_ND vs Phi_Sonic): {:.4f}".format(
            metrics.get("pearson_r_phi_nd_vs_phi_sonic", float("nan"))
        ),
        "",
        "tables/",
        "  861_dem_sc_profile.csv / .xlsx",
        "  hfu_matrix_moduli.csv",
        "  validation_phi_sonic.csv",
        "  summary_by_hfu.csv",
        "",
        "figures/",
        "  vp_dem_vs_depth.png",
        "  vpvs_vs_depth.png",
        "  phi_nd_vs_phi_sonic.png",
        "  vpvs_by_hfu.png",
    ]
    (DEM_SC_PROFILE_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_profile_pipeline(
    smoke: bool = False,
    lab_calib_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute 2c + 2d pipeline."""
    out_root = DEM_SC_PROFILE_ROOT
    profile_tables = PROFILE_TABLES
    profile_figures = PROFILE_FIGURES
    param_label = "ct_plug_median"

    use_lab_calib = lab_calib_path is not None and lab_calib_path is not False
    if use_lab_calib:
        use_path = Path(lab_calib_path)
        if not use_path.is_file():
            raise FileNotFoundError("Lab calibration file not found: {}".format(use_path))
        out_root = DEM_SC_PROFILE_LAB_CALIB_ROOT
        profile_tables = out_root / "tables"
        profile_figures = out_root / "figures"
        param_label = "lab_calibrated"

    for d in (out_root, profile_tables, profile_figures):
        d.mkdir(parents=True, exist_ok=True)

    if param_label == "lab_calibrated":
        lab_hfu = load_hfu_lab_calibrated(Path(lab_calib_path))
        hfu_params = build_hfu_params_from_lab_calib(lab_hfu)
        hfu_table = lab_hfu
    else:
        hfu_stats = load_hfu_ct_stats()
        ct_df = load_ct_samples()
        hfu_params = build_hfu_params(hfu_stats, ct_df)
        hfu_table = hfu_params_to_table(hfu_params)

    hfu_table.to_csv(profile_tables / "hfu_matrix_moduli.csv", index=False, float_format="%.6f")

    logs = load_logs_enriched()
    if smoke:
        logs = logs.head(15).copy()

    profile = process_profile(logs, hfu_params)
    csv_path = profile_tables / "861_dem_sc_profile.csv"
    xlsx_path = profile_tables / "861_dem_sc_profile.xlsx"
    profile.to_csv(csv_path, index=False, float_format="%.6f")
    profile.to_excel(xlsx_path, index=False)

    ok = profile[profile["status"] == "ok"]
    n_ok = len(ok)
    if n_ok == 0:
        raise RuntimeError("All profile rows failed")

    validation = build_validation_phi_sonic(profile)
    validation.to_csv(profile_tables / "validation_phi_sonic.csv", index=False, float_format="%.6f")

    by_hfu = summary_by_hfu(profile)
    by_hfu.to_csv(profile_tables / "summary_by_hfu.csv", index=False, float_format="%.6f")

    plot_vp_dem_vs_depth(profile, profile_figures / "vp_dem_vs_depth.png")
    plot_vpvs_vs_depth(profile, profile_figures / "vpvs_vs_depth.png")
    plot_phi_nd_vs_phi_sonic(profile, profile_figures / "phi_nd_vs_phi_sonic.png")
    plot_vpvs_by_hfu(by_hfu, profile_figures / "vpvs_by_hfu.png")

    val_global = validation[
        (validation["scope"] == "global") & (validation["HFU"] == "all")
    ]
    r_sonic = float("nan")
    if "pearson_r_phi_nd_vs_phi_sonic" in val_global.columns and len(val_global) > 0:
        r_sonic = float(val_global.iloc[0]["pearson_r_phi_nd_vs_phi_sonic"])

    metrics = {
        "n_rows": int(len(logs)),
        "n_ok": int(n_ok),
        "param_source": param_label,
        "pearson_r_phi_nd_vs_phi_sonic": r_sonic,
        "mean_vp_dem_km_s": float(ok["vp_dem_km_s"].mean()),
        "mean_vpvs_dem": float(ok["vpvs_dem"].mean()),
        "mean_vp_rel_diff_dem_sc": float(ok["vp_rel_diff_dem_sc"].mean()),
        "smoke": smoke,
        "generated_utc": utc_now_iso(),
    }
    (out_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_lines = [
        "Well 861 -- Etapa 2c profile (87 rows)",
        "Generated: {}".format(utc_now_iso()),
        "HFU parameters: {}".format(param_label),
        "Rows OK: {}/{}".format(n_ok, len(logs)),
    ]
    (out_root / "MANIFEST.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

    return {
        "profile_path": str(xlsx_path),
        "n_ok": n_ok,
        "metrics": metrics,
        "out_root": str(out_root),
    }



def parse_args() -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Well 861 Etapa 2c/2d: DEM/SC profile 87 rows",
    )
    parser.add_argument("--smoke", action="store_true", help="First 15 rows only")
    parser.add_argument(
        "--lab-calib",
        action="store_true",
        help="Use hfu_lab_calibrated.csv (requires run_861_dem_sc_lab_calibration.py)",
    )
    parser.add_argument(
        "--ct-only",
        action="store_true",
        help="Force CT-only HFU params (ignore lab calibration file)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    lab_path: Optional[Path] = None
    if args.ct_only:
        lab_path = False
    elif args.lab_calib:
        lab_path = HFU_LAB_CALIB_CSV
    result = run_profile_pipeline(smoke=args.smoke, lab_calib_path=lab_path)
    m = result["metrics"]
    print("Profile complete: {}/{} rows OK".format(result["n_ok"], m["n_rows"]))
    print("Output: {}".format(result["profile_path"]))
    print("Param source: {}".format(m.get("param_source", "ct_plug_median")))
    print("Pearson r Phi_ND vs Phi_Sonic: {:.4f}".format(m["pearson_r_phi_nd_vs_phi_sonic"]))
    print("Mean Vp DEM: {:.3f} km/s".format(m["mean_vp_dem_km_s"]))


if __name__ == "__main__":
    main()
