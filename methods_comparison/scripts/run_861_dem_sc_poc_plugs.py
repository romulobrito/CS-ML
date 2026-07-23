#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2a POC: DEM/SC rock physics on 10 CT plugs (Well 861).

Outputs under methods_comparison/data/processed/dem_sc_runs/poc_10plugs/
with tables/, figures/, and MANIFEST.txt.

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dem_sc_861_core import run_plug_case
from ml_861_data import ROOT, load_ct_samples

DEM_SC_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed" / "dem_sc_runs"
)
POC_ROOT = DEM_SC_ROOT / "poc_10plugs"
TABLES_DIR = POC_ROOT / "tables"
FIGURES_DIR = POC_ROOT / "figures"
HFU_CALIB_DIR = DEM_SC_ROOT / "hfu_calibration"

REQUIRED_COLS: Tuple[str, ...] = (
    "sample_id",
    "HFU",
    "Phi_lab (pu)",
    "ct_ar_mean",
    "corrected_solid1_pct",
    "corrected_solid2_pct",
    "ct_depth_m",
)


def utc_now_iso() -> str:
    """UTC timestamp for manifest."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directory tree."""
    for d in (POC_ROOT, TABLES_DIR, FIGURES_DIR, HFU_CALIB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def validate_ct_table(df: pd.DataFrame) -> None:
    """Ensure required columns exist."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing CT columns: {}".format(missing))


def process_plugs(df: pd.DataFrame) -> pd.DataFrame:
    """Run DEM/SC for each plug row."""
    rows: List[dict] = []
    for _, row in df.iterrows():
        sid = str(row["sample_id"])
        try:
            out = run_plug_case(
                phi_lab=float(row["Phi_lab (pu)"]),
                alpha=float(row["ct_ar_mean"]),
                solid1_pct=float(row["corrected_solid1_pct"]),
                solid2_pct=float(row["corrected_solid2_pct"]),
            )
            rec = {
                "sample_id": sid,
                "ct_depth_m": float(row["ct_depth_m"]),
                "HFU": int(row["HFU"]),
                "solid1_pct": float(row["corrected_solid1_pct"]),
                "solid2_pct": float(row["corrected_solid2_pct"]),
                **out,
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            rec = {
                "sample_id": sid,
                "ct_depth_m": float(row["ct_depth_m"]),
                "HFU": int(row["HFU"]),
                "status": "error",
                "error": str(exc),
            }
        rows.append(rec)
    return pd.DataFrame(rows)


def build_hfu_calibration(summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CT-derived parameters by HFU for later profile extrapolation."""
    ok = summary[summary["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful plug runs for HFU calibration")

    agg_spec = {
        "sample_id": "count",
        "phi_lab": ["mean", "median", "std", "min", "max"],
        "alpha": ["mean", "median", "std", "min", "max"],
        "matrix_k_gpa": ["mean", "median"],
        "matrix_g_gpa": ["mean", "median"],
        "matrix_rho_gcc": ["mean", "median"],
        "vp_dem_km_s": ["mean", "median"],
        "vpvs_dem": ["mean", "median"],
    }
    grouped = ok.groupby("HFU").agg(agg_spec)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns.values]
    grouped = grouped.reset_index()
    grouped.rename(columns={"sample_id_count": "n_plugs"}, inplace=True)
    return grouped


def build_dem_sc_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Per-plug DEM vs SC comparison metrics."""
    ok = summary[summary["status"] == "ok"].copy()
    cols = [
        "sample_id",
        "HFU",
        "phi_lab",
        "alpha",
        "vp_dem_km_s",
        "vp_sc_km_s",
        "vpvs_dem",
        "vpvs_sc",
        "vp_rel_diff_dem_sc",
        "vpvs_rel_diff_dem_sc",
        "dem_k_gpa",
        "sc_k_gpa",
        "dem_g_gpa",
        "sc_g_gpa",
    ]
    comp = ok[cols].copy()
    comp["k_rel_diff_dem_sc"] = (comp["dem_k_gpa"] - comp["sc_k_gpa"]).abs() / comp["dem_k_gpa"]
    comp["g_rel_diff_dem_sc"] = (comp["dem_g_gpa"] - comp["sc_g_gpa"]).abs() / comp["dem_g_gpa"]
    return comp


def plot_dem_vs_sc_vp(comp: pd.DataFrame, out_path: Path) -> None:
    """Scatter DEM vs SC P-wave velocity."""
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for hfu in sorted(comp["HFU"].unique()):
        sub = comp[comp["HFU"] == hfu]
        ax.scatter(
            sub["vp_dem_km_s"],
            sub["vp_sc_km_s"],
            label="HFU{} (n={})".format(hfu, len(sub)),
            c=colors.get(int(hfu), "#333333"),
            s=80,
            edgecolors="k",
            linewidths=0.5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["sample_id"],
                (r["vp_dem_km_s"], r["vp_sc_km_s"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    lims = [
        min(comp["vp_dem_km_s"].min(), comp["vp_sc_km_s"].min()) * 0.95,
        max(comp["vp_dem_km_s"].max(), comp["vp_sc_km_s"].max()) * 1.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=1.0, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Vp DEM (km/s)")
    ax.set_ylabel("Vp SC (km/s)")
    ax.set_title("Well 861 POC: DEM vs SC (10 plugs, dry)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dem_vs_sc_vpvs(comp: pd.DataFrame, out_path: Path) -> None:
    """Bar comparison of Vp/Vs DEM vs SC per plug."""
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(comp))
    w = 0.35
    ax.bar(x - w / 2, comp["vpvs_dem"], w, label="DEM", color="#4c72b0")
    ax.bar(x + w / 2, comp["vpvs_sc"], w, label="SC", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels(comp["sample_id"], rotation=45, ha="right")
    ax.set_ylabel("Vp/Vs")
    ax.set_title("Well 861 POC: Vp/Vs DEM vs SC by plug")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vp_vs_phi_by_hfu(summary: pd.DataFrame, out_path: Path) -> None:
    """Vp (DEM) vs porosity colored by HFU."""
    ok = summary[summary["status"] == "ok"]
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        ax.scatter(
            sub["phi_lab"],
            sub["vp_dem_km_s"],
            label="HFU{}".format(hfu),
            c=colors.get(int(hfu), "#333333"),
            s=90,
            edgecolors="k",
            linewidths=0.5,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                r["sample_id"],
                (r["phi_lab"], r["vp_dem_km_s"]),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xlabel("Phi_lab (v/v)")
    ax.set_ylabel("Vp DEM (km/s)")
    ax.set_title("Well 861 POC: Vp vs porosity by HFU")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_alpha(
    phi_lab: float,
    solid1_pct: float,
    solid2_pct: float,
    alpha_ref: float,
    sample_id: str,
    out_path: Path,
    alpha_grid: Optional[Sequence[float]] = None,
) -> None:
    """Vp DEM sensitivity to aspect ratio for one reference plug."""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.35, 0.85, 26)
    vp_list: List[float] = []
    for a in alpha_grid:
        res = run_plug_case(phi_lab, float(a), solid1_pct, solid2_pct)
        vp_list.append(res["vp_dem_km_s"])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(alpha_grid, vp_list, "b-", linewidth=2.0)
    ax.axvline(alpha_ref, color="r", linestyle="--", label="ct_ar_mean={:.3f}".format(alpha_ref))
    ax.set_xlabel("Aspect ratio alpha")
    ax.set_ylabel("Vp DEM (km/s)")
    ax.set_title("Sensitivity: {} (phi={:.3f})".format(sample_id, phi_lab))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_manifest(
    n_plugs: int,
    n_ok: int,
    metrics: Dict[str, float],
) -> None:
    """Write human-readable MANIFEST."""
    lines = [
        "Well 861 -- Etapa 2a POC (DEM/SC on 10 CT plugs)",
        "Generated: {}".format(utc_now_iso()),
        "",
        "Plugs processed: {}/{}".format(n_ok, n_plugs),
        "Mean Vp DEM-SC rel diff: {:.4f}".format(metrics.get("mean_vp_rel_diff", float("nan"))),
        "Mean Vp/Vs DEM-SC rel diff: {:.4f}".format(metrics.get("mean_vpvs_rel_diff", float("nan"))),
        "",
        "tables/",
        "  plug_dem_sc_summary.csv      -- full per-plug results",
        "  dem_sc_comparison.csv        -- DEM vs SC metrics",
        "  hfu_ct_stats.csv             -- copy in hfu_calibration/",
        "",
        "figures/",
        "  dem_vs_sc_vp.png",
        "  dem_vs_sc_vpvs.png",
        "  vp_vs_phi_by_hfu.png",
        "  sensitivity_alpha_median_plug.png",
        "",
        "Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md",
    ]
    (POC_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_poc(smoke: bool = False) -> Dict[str, object]:
    """Execute full POC pipeline."""
    ensure_dirs()
    df = load_ct_samples()
    validate_ct_table(df)
    if smoke:
        df = df.head(2).copy()

    summary = process_plugs(df)
    summary_path = TABLES_DIR / "plug_dem_sc_summary.csv"
    summary.to_csv(summary_path, index=False, float_format="%.6f")

    ok = summary[summary["status"] == "ok"]
    n_ok = len(ok)
    if n_ok == 0:
        raise RuntimeError("All plug runs failed; see plug_dem_sc_summary.csv")

    comp = build_dem_sc_comparison(summary)
    comp_path = TABLES_DIR / "dem_sc_comparison.csv"
    comp.to_csv(comp_path, index=False, float_format="%.6f")

    hfu_cal = build_hfu_calibration(summary)
    hfu_path = HFU_CALIB_DIR / "hfu_ct_stats.csv"
    hfu_cal.to_csv(hfu_path, index=False, float_format="%.6f")
    hfu_copy = TABLES_DIR / "hfu_ct_stats.csv"
    hfu_cal.to_csv(hfu_copy, index=False, float_format="%.6f")

    plot_dem_vs_sc_vp(comp, FIGURES_DIR / "dem_vs_sc_vp.png")
    plot_dem_vs_sc_vpvs(comp, FIGURES_DIR / "dem_vs_sc_vpvs.png")
    plot_vp_vs_phi_by_hfu(summary, FIGURES_DIR / "vp_vs_phi_by_hfu.png")

    median_row = ok.iloc[len(ok) // 2]
    plot_sensitivity_alpha(
        phi_lab=float(median_row["phi_lab"]),
        solid1_pct=float(median_row["solid1_pct"]),
        solid2_pct=float(median_row["solid2_pct"]),
        alpha_ref=float(median_row["alpha"]),
        sample_id=str(median_row["sample_id"]),
        out_path=FIGURES_DIR / "sensitivity_alpha_median_plug.png",
    )

    metrics = {
        "n_plugs": int(len(df)),
        "n_ok": int(n_ok),
        "mean_vp_rel_diff_dem_sc": float(ok["vp_rel_diff_dem_sc"].mean()),
        "max_vp_rel_diff_dem_sc": float(ok["vp_rel_diff_dem_sc"].max()),
        "mean_vpvs_rel_diff_dem_sc": float(ok["vpvs_rel_diff_dem_sc"].mean()),
        "mean_vp_dem_km_s": float(ok["vp_dem_km_s"].mean()),
        "mean_vpvs_dem": float(ok["vpvs_dem"].mean()),
        "smoke": smoke,
        "generated_utc": utc_now_iso(),
    }
    metrics_path = POC_ROOT / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    write_manifest(len(df), n_ok, {
        "mean_vp_rel_diff": metrics["mean_vp_rel_diff_dem_sc"],
        "mean_vpvs_rel_diff": metrics["mean_vpvs_rel_diff_dem_sc"],
    })

    return {
        "summary_path": str(summary_path),
        "metrics": metrics,
        "n_ok": n_ok,
    }


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Well 861 Etapa 2a: DEM/SC POC on CT plugs",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Process only first 2 plugs (quick test)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    result = run_poc(smoke=args.smoke)
    print("POC complete: {}/{} plugs OK".format(result["n_ok"], result["metrics"]["n_plugs"]))
    print("Summary: {}".format(result["summary_path"]))
    print("Mean Vp DEM-SC rel diff: {:.4f}".format(result["metrics"]["mean_vp_rel_diff_dem_sc"]))


if __name__ == "__main__":
    main()
