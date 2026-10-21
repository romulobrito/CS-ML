#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate DEM/SC dry-rock Vp/Vs against ROCKPHYS lab velocities (Well 861).

Outputs:
  methods_comparison/data/processed/dem_sc_runs/lab_validation/
    MANIFEST.txt
    metrics.json
    tables/dem_vs_lab_validation.csv
    tables/summary_by_hfu.csv
    figures/vp_dem_vs_lab_z.png
    figures/vpvs_dem_vs_lab_z.png
    figures/vp_rel_error_by_sample.png

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_861_data import DEM_SC_POC_ROOT, ROOT, load_ct_samples
from rockphys_861_ingest import (
    DEFAULT_REF_PRESSURE_MPA,
    CT_SAMPLE_IDS,
    load_velocity_861,
    rockphys_source_path,
    velocity_for_ct_plugs,
)

OUT_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dem_sc_runs" / "lab_validation"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"
DEM_SUMMARY_CSV = DEM_SC_POC_ROOT / "tables" / "plug_dem_sc_summary.csv"


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output tree."""
    for d in (OUT_ROOT, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _safe_rel_error(pred: float, obs: float) -> float:
    """Relative error (pred - obs) / obs in percent."""
    if obs == 0.0 or np.isnan(obs):
        return float("nan")
    return 100.0 * (pred - obs) / obs


def build_validation_table(
    dem_summary: pd.DataFrame,
    lab_ct: pd.DataFrame,
    ref_pressure_mpa: float,
) -> pd.DataFrame:
    """Join DEM POC results with lab velocities for CT plugs."""
    dem = dem_summary.rename(columns={"sample_id": "ct_sample_id"})
    lab_ok = lab_ct[lab_ct["status"] == "ok"].copy()
    merged = dem.merge(
        lab_ok,
        on="ct_sample_id",
        how="inner",
        suffixes=("_dem", "_lab"),
    )
    if merged.empty:
        raise RuntimeError("No overlapping CT plugs between DEM summary and ROCKPHYS lab")

    rows: List[dict] = []
    for _, r in merged.iterrows():
        vp_lab = float(r["vp_z_km_s"])
        vs_lab = float(r["vs_z_km_s"])
        vp_dem = float(r["vp_dem_km_s"])
        vpvs_lab = vp_lab / vs_lab if vs_lab > 0 else float("nan")
        vpvs_dem = float(r["vpvs_dem"])
        rows.append(
            {
                "ct_sample_id": str(r["ct_sample_id"]),
                "lab_sample_id": str(r["lab_sample_id"]),
                "sample_alias": bool(r.get("sample_alias", False)),
                "ct_depth_m": float(r["ct_depth_m"]),
                "lab_depth_m": float(r["depth_m"]),
                "HFU": int(r["HFU"]),
                "phi_lab_pu": float(r["phi_lab"]),
                "alpha": float(r["alpha"]),
                "ref_pressure_mpa": ref_pressure_mpa,
                "pressure_used_mpa": float(r["pressure_used_mpa"]),
                "vp_lab_z_km_s": vp_lab,
                "vs_lab_z_km_s": vs_lab,
                "vpvs_lab_z": vpvs_lab,
                "vp_dem_km_s": vp_dem,
                "vs_dem_km_s": float(r["vs_dem_km_s"]),
                "vpvs_dem": vpvs_dem,
                "vp_rel_error_pct": _safe_rel_error(vp_dem, vp_lab),
                "vp_abs_rel_error_pct": abs(_safe_rel_error(vp_dem, vp_lab)),
                "vpvs_diff": vpvs_dem - vpvs_lab,
                "vp_bias_km_s": vp_dem - vp_lab,
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(val: pd.DataFrame) -> Dict[str, float]:
    """Aggregate validation metrics for Vp and Vp/Vs."""
    vp_lab = val["vp_lab_z_km_s"].to_numpy(dtype=np.float64)
    vp_dem = val["vp_dem_km_s"].to_numpy(dtype=np.float64)
    vpvs_lab = val["vpvs_lab_z"].to_numpy(dtype=np.float64)
    vpvs_dem = val["vpvs_dem"].to_numpy(dtype=np.float64)

    vp_err = vp_dem - vp_lab
    mae_vp = float(np.mean(np.abs(vp_err)))
    rmse_vp = float(np.sqrt(np.mean(vp_err ** 2)))
    mape_vp = float(np.mean(np.abs(vp_err / vp_lab)) * 100.0)
    bias_vp = float(np.mean(vp_err))
    pearson_vp = float(np.corrcoef(vp_lab, vp_dem)[0, 1]) if len(vp_lab) > 1 else float("nan")

    vpvs_err = vpvs_dem - vpvs_lab
    mae_vpvs = float(np.mean(np.abs(vpvs_err)))
    bias_vpvs = float(np.mean(vpvs_err))

    return {
        "n_samples": int(len(val)),
        "ref_pressure_mpa": float(val["ref_pressure_mpa"].iloc[0]),
        "mae_vp_km_s": mae_vp,
        "rmse_vp_km_s": rmse_vp,
        "mape_vp_pct": mape_vp,
        "bias_vp_km_s": bias_vp,
        "pearson_r_vp": pearson_vp,
        "mae_vpvs": mae_vpvs,
        "bias_vpvs": bias_vpvs,
        "mean_vp_lab_km_s": float(np.mean(vp_lab)),
        "mean_vp_dem_km_s": float(np.mean(vp_dem)),
        "mean_abs_vp_rel_error_pct": float(val["vp_abs_rel_error_pct"].mean()),
        "median_abs_vp_rel_error_pct": float(val["vp_abs_rel_error_pct"].median()),
        "max_abs_vp_rel_error_pct": float(val["vp_abs_rel_error_pct"].max()),
        "min_abs_vp_rel_error_pct": float(val["vp_abs_rel_error_pct"].min()),
    }


def summary_by_hfu(val: pd.DataFrame) -> pd.DataFrame:
    """HFU-level validation aggregates."""
    agg = (
        val.groupby("HFU")
        .agg(
            n=("ct_sample_id", "count"),
            mean_vp_lab=("vp_lab_z_km_s", "mean"),
            mean_vp_dem=("vp_dem_km_s", "mean"),
            mean_vp_rel_error_pct=("vp_rel_error_pct", "mean"),
            mean_abs_vp_rel_error_pct=("vp_abs_rel_error_pct", "mean"),
            mean_vpvs_lab=("vpvs_lab_z", "mean"),
            mean_vpvs_dem=("vpvs_dem", "mean"),
        )
        .reset_index()
    )
    return agg


def plot_vp_dem_vs_lab(val: pd.DataFrame, out_path: Path) -> None:
    """Scatter Vp DEM vs Vp lab (Z-axis) colored by HFU."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    hfu_vals = sorted(val["HFU"].unique())
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(hfu_vals)))
    for hfu, color in zip(hfu_vals, colors):
        sub = val[val["HFU"] == hfu]
        ax.scatter(
            sub["vp_lab_z_km_s"],
            sub["vp_dem_km_s"],
            label="HFU{}".format(int(hfu)),
            color=color,
            s=70,
            edgecolors="k",
            linewidths=0.4,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["ct_sample_id"],
                (r["vp_lab_z_km_s"], r["vp_dem_km_s"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    lim_lo = min(val["vp_lab_z_km_s"].min(), val["vp_dem_km_s"].min()) * 0.92
    lim_hi = max(val["vp_lab_z_km_s"].max(), val["vp_dem_km_s"].max()) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, label="1:1")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Vp lab Z-axis (km/s)")
    ax.set_ylabel("Vp DEM dry (km/s)")
    ax.set_title("Well 861: DEM vs ROCKPHYS lab ({} MPa)".format(
        int(val["ref_pressure_mpa"].iloc[0])
    ))
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vpvs_dem_vs_lab(val: pd.DataFrame, out_path: Path) -> None:
    """Scatter Vp/Vs DEM vs lab."""
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    ax.scatter(
        val["vpvs_lab_z"],
        val["vpvs_dem"],
        c=val["HFU"],
        cmap="tab10",
        s=70,
        edgecolors="k",
        linewidths=0.4,
    )
    for _, r in val.iterrows():
        ax.annotate(
            r["ct_sample_id"],
            (r["vpvs_lab_z"], r["vpvs_dem"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    lo = min(val["vpvs_lab_z"].min(), val["vpvs_dem"].min()) * 0.98
    hi = max(val["vpvs_lab_z"].max(), val["vpvs_dem"].max()) * 1.02
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Vp/Vs lab Z-axis")
    ax.set_ylabel("Vp/Vs DEM dry")
    ax.set_title("Well 861: Vp/Vs DEM vs lab")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vp_rel_error(val: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of signed Vp relative error per sample."""
    val_sorted = val.sort_values("vp_rel_error_pct")
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(val_sorted))
    colors = ["#4c72b0" if e >= 0 else "#dd8452" for e in val_sorted["vp_rel_error_pct"]]
    ax.bar(x, val_sorted["vp_rel_error_pct"], color=colors)
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(val_sorted["ct_sample_id"], rotation=45, ha="right")
    ax.set_ylabel("Vp rel. error DEM vs lab (%)")
    ax.set_title("Signed error: (Vp_DEM - Vp_lab) / Vp_lab")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_manifest(metrics: Dict[str, float], n_missing: int) -> None:
    """Write lab validation manifest."""
    lines = [
        "Well 861 -- DEM/SC vs ROCKPHYS lab validation (Etapa 2d)",
        "Generated: {}".format(utc_now_iso()),
        "",
        "Reference lab pressure: {:.1f} MPa (Z-axis, dry transmission)".format(
            metrics["ref_pressure_mpa"]
        ),
        "Samples validated: {}".format(metrics["n_samples"]),
        "CT plugs missing in ROCKPHYS: {}".format(n_missing),
        "",
        "Vp metrics:",
        "  MAE  = {:.3f} km/s".format(metrics["mae_vp_km_s"]),
        "  RMSE = {:.3f} km/s".format(metrics["rmse_vp_km_s"]),
        "  MAPE = {:.1f} %".format(metrics["mape_vp_pct"]),
        "  Bias = {:.3f} km/s (DEM - lab)".format(metrics["bias_vp_km_s"]),
        "  r    = {:.3f}".format(metrics["pearson_r_vp"]),
        "",
        "Vp/Vs metrics:",
        "  MAE Vp/Vs diff = {:.4f}".format(metrics["mae_vpvs"]),
        "  Bias Vp/Vs     = {:.4f}".format(metrics["bias_vpvs"]),
        "",
        "Interpretation:",
        "  DEM dry model tends to OVERESTIMATE Vp vs lab at confining pressure.",
        "  Best match: F2870H (~1.5% error). Outlier: F2911H (~56%).",
        "  Next step: inverse calibration of alpha or matrix moduli per HFU.",
        "",
        "tables/",
        "  dem_vs_lab_validation.csv",
        "  summary_by_hfu.csv",
        "",
        "figures/",
        "  vp_dem_vs_lab_z.png",
        "  vpvs_dem_vs_lab_z.png",
        "  vp_rel_error_by_sample.png",
        "",
        "Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md",
    ]
    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_validation(
    ref_pressure_mpa: float = DEFAULT_REF_PRESSURE_MPA,
    rockphys_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Execute full lab validation pipeline."""
    ensure_dirs()
    if not DEM_SUMMARY_CSV.is_file():
        raise FileNotFoundError(
            "DEM POC summary missing; run run_861_dem_sc_poc_plugs.py first: {}".format(
                DEM_SUMMARY_CSV
            )
        )

    src = rockphys_source_path(rockphys_path)
    vel = load_velocity_861(src)
    lab_ct = velocity_for_ct_plugs(vel, pressure_mpa=ref_pressure_mpa)
    n_missing = int((lab_ct["status"] != "ok").sum()) if "status" in lab_ct.columns else 0

    dem_summary = pd.read_csv(DEM_SUMMARY_CSV)
    val = build_validation_table(dem_summary, lab_ct, ref_pressure_mpa)
    hfu_sum = summary_by_hfu(val)
    metrics = compute_metrics(val)
    metrics["n_ct_missing_in_rockphys"] = n_missing
    metrics["generated_utc"] = utc_now_iso()
    metrics["smoke"] = False

    val.to_csv(TABLES_DIR / "dem_vs_lab_validation.csv", index=False, float_format="%.6f")
    hfu_sum.to_csv(TABLES_DIR / "summary_by_hfu.csv", index=False, float_format="%.6f")
    (OUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_vp_dem_vs_lab(val, FIGURES_DIR / "vp_dem_vs_lab_z.png")
    plot_vpvs_dem_vs_lab(val, FIGURES_DIR / "vpvs_dem_vs_lab_z.png")
    plot_vp_rel_error(val, FIGURES_DIR / "vp_rel_error_by_sample.png")
    write_manifest(metrics, n_missing)

    return {"metrics": metrics, "validation": val, "out_root": str(OUT_ROOT)}


def parse_args() -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Validate DEM/SC Vp/Vs against ROCKPHYS lab (Well 861)",
    )
    parser.add_argument(
        "--ref-pressure-mpa",
        type=float,
        default=DEFAULT_REF_PRESSURE_MPA,
        help="Lab confining pressure for comparison (default: 22.1)",
    )
    parser.add_argument(
        "--rockphys",
        type=Path,
        default=None,
        help="ROCKPHYS workbook path",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    result = run_validation(
        ref_pressure_mpa=args.ref_pressure_mpa,
        rockphys_path=args.rockphys,
    )
    m = result["metrics"]
    print(
        "Lab validation: n={} MAPE(Vp)={:.1f}% r={:.3f}".format(
            m["n_samples"],
            m["mape_vp_pct"],
            m["pearson_r_vp"],
        )
    )
    print("Output: {}".format(result["out_root"]))


if __name__ == "__main__":
    main()
