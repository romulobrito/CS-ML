#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate DEM/SC calibrated profile Vp/Vs against DSI sonic log (Well 861).

Outputs:
  methods_comparison/data/processed/dlis_861/
    tables/dem_vs_sonic_validation.csv
    tables/summary_by_hfu.csv
    metrics_validation.json
    figures/vp_dem_vs_sonic_depth.png
    figures/vp_dem_vs_sonic_scatter.png
    figures/vpvs_dem_vs_sonic_depth.png
    figures/vp_rel_error_vs_depth.png

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_861_data import (
    DEPTH_COL,
    DEM_SC_PROFILE_LAB_CALIB_ROOT,
    ROOT,
)

OUT_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dlis_861"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"
SONIC_CSV = TABLES_DIR / "sonic_log.csv"
DEM_PROFILE_CSV = DEM_SC_PROFILE_LAB_CALIB_ROOT / "tables" / "861_dem_sc_profile.csv"
VAL_CSV = TABLES_DIR / "dem_vs_sonic_validation.csv"
SUMMARY_CSV = TABLES_DIR / "summary_by_hfu.csv"
METRICS_JSON = OUT_ROOT / "metrics_validation.json"

DEFAULT_MERGE_TOL_M = 0.25


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (OUT_ROOT, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(
        description="Validate Well 861 DEM profile vs DSI sonic log."
    )
    p.add_argument(
        "--merge-tol-m",
        type=float,
        default=DEFAULT_MERGE_TOL_M,
        help="Depth tolerance for merge_asof (m).",
    )
    p.add_argument(
        "--sonic-csv",
        type=str,
        default=str(SONIC_CSV),
        help="Path to sonic_log.csv.",
    )
    p.add_argument(
        "--dem-csv",
        type=str,
        default=str(DEM_PROFILE_CSV),
        help="Path to calibrated DEM profile CSV.",
    )
    return p.parse_args()


def _safe_rel_error(pred: float, obs: float) -> float:
    """Relative error (pred - obs) / obs in percent."""
    if obs == 0.0 or np.isnan(obs):
        return float("nan")
    return 100.0 * (pred - obs) / obs


def load_sonic(path: Path) -> pd.DataFrame:
    """Load cropped sonic log table."""
    if not path.is_file():
        raise FileNotFoundError(
            "Sonic log missing: {}. Run run_861_dlis_sonic_extract.py first.".format(path)
        )
    df = pd.read_csv(path)
    if "depth_m" not in df.columns:
        raise ValueError("sonic_log.csv missing depth_m column")
    return df.sort_values("depth_m").reset_index(drop=True)


def load_dem_profile(path: Path) -> pd.DataFrame:
    """Load calibrated DEM profile (87 rows)."""
    if not path.is_file():
        raise FileNotFoundError(
            "DEM profile missing: {}. Run run_861_dem_sc_profile_87.py --lab-calib.".format(
                path
            )
        )
    df = pd.read_csv(path)
    if DEPTH_COL not in df.columns:
        raise ValueError("DEM profile missing {}".format(DEPTH_COL))
    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise RuntimeError("No OK rows in DEM profile")
    return ok.sort_values(DEPTH_COL).reset_index(drop=True)


def merge_dem_sonic(
    dem: pd.DataFrame,
    sonic: pd.DataFrame,
    merge_tol_m: float,
) -> pd.DataFrame:
    """Nearest-neighbor merge of DEM profile depths to sonic samples."""
    left = dem.rename(columns={DEPTH_COL: "depth_m"}).copy()
    right = sonic[
        [
            "depth_m",
            "dtco_usft",
            "dtsm_usft",
            "vp_sonic_km_s",
            "vs_sonic_km_s",
            "vpvs_sonic_calc",
            "vpvs_sonic_dlis",
            "sphi",
        ]
    ].rename(columns={"depth_m": "sonic_depth_m"})

    merged = pd.merge_asof(
        left.sort_values("depth_m"),
        right.sort_values("sonic_depth_m"),
        left_on="depth_m",
        right_on="sonic_depth_m",
        direction="nearest",
        tolerance=merge_tol_m,
    )
    merged["depth_delta_m"] = merged["sonic_depth_m"] - merged["depth_m"]
    merged["has_sonic_vp"] = merged["vp_sonic_km_s"].notna()
    merged["has_sonic_vs"] = merged["vs_sonic_km_s"].notna()
    return merged


def build_validation_table(merged: pd.DataFrame) -> pd.DataFrame:
    """Add error columns for rows with sonic Vp."""
    rows: List[dict] = []
    for _, r in merged.iterrows():
        vp_dem = float(r["vp_dem_km_s"])
        vp_sonic = float(r["vp_sonic_km_s"]) if pd.notna(r["vp_sonic_km_s"]) else float("nan")
        vs_dem = float(r["vs_dem_km_s"])
        vs_sonic = float(r["vs_sonic_km_s"]) if pd.notna(r["vs_sonic_km_s"]) else float("nan")
        vpvs_dem = float(r["vpvs_dem"])
        vpvs_sonic = (
            float(r["vpvs_sonic_calc"])
            if pd.notna(r["vpvs_sonic_calc"])
            else float("nan")
        )
        rows.append(
            {
                "Depth(m)": float(r["depth_m"]),
                "sonic_depth_m": float(r["sonic_depth_m"]) if pd.notna(r["sonic_depth_m"]) else float("nan"),
                "depth_delta_m": float(r["depth_delta_m"]) if pd.notna(r["depth_delta_m"]) else float("nan"),
                "HFU": int(r["HFU"]),
                "phi_input": float(r["phi_input"]),
                "Phi_Sonic (pu)": float(r["Phi_Sonic (pu)"]),
                "vp_dem_km_s": vp_dem,
                "vp_sonic_km_s": vp_sonic,
                "vs_dem_km_s": vs_dem,
                "vs_sonic_km_s": vs_sonic,
                "vpvs_dem": vpvs_dem,
                "vpvs_sonic": vpvs_sonic,
                "dtco_usft": float(r["dtco_usft"]) if pd.notna(r["dtco_usft"]) else float("nan"),
                "dtsm_usft": float(r["dtsm_usft"]) if pd.notna(r["dtsm_usft"]) else float("nan"),
                "vp_rel_error_pct": _safe_rel_error(vp_dem, vp_sonic),
                "vp_abs_rel_error_pct": abs(_safe_rel_error(vp_dem, vp_sonic))
                if pd.notna(vp_sonic)
                else float("nan"),
                "vp_bias_km_s": vp_dem - vp_sonic if pd.notna(vp_sonic) else float("nan"),
                "vpvs_diff": vpvs_dem - vpvs_sonic if pd.notna(vpvs_sonic) else float("nan"),
                "has_sonic_vp": bool(r["has_sonic_vp"]),
            }
        )
    return pd.DataFrame(rows)


def compute_metrics(val: pd.DataFrame) -> Dict[str, float]:
    """Aggregate validation metrics for matched rows with sonic Vp."""
    ok = val[val["has_sonic_vp"]].copy()
    n_total = int(len(val))
    n_matched = int(len(ok))
    out: Dict[str, float] = {
        "n_profile_rows": n_total,
        "n_matched_vp": n_matched,
        "match_fraction": float(n_matched / n_total) if n_total > 0 else float("nan"),
    }
    if n_matched == 0:
        return out

    vp_lab = ok["vp_sonic_km_s"].to_numpy(dtype=np.float64)
    vp_dem = ok["vp_dem_km_s"].to_numpy(dtype=np.float64)
    err = vp_dem - vp_lab
    out["mae_vp_km_s"] = float(np.mean(np.abs(err)))
    out["rmse_vp_km_s"] = float(np.sqrt(np.mean(err ** 2)))
    out["mape_vp_pct"] = float(np.mean(np.abs(err / vp_lab)) * 100.0)
    out["bias_vp_km_s"] = float(np.mean(err))
    out["pearson_r_vp"] = (
        float(np.corrcoef(vp_lab, vp_dem)[0, 1]) if n_matched > 1 else float("nan")
    )
    out["mean_vp_sonic_km_s"] = float(np.mean(vp_lab))
    out["mean_vp_dem_km_s"] = float(np.mean(vp_dem))
    out["median_abs_vp_rel_error_pct"] = float(ok["vp_abs_rel_error_pct"].median())

    ok_vs = ok[ok["vs_sonic_km_s"].notna()]
    if len(ok_vs) > 0:
        vpvs_sonic = ok_vs["vpvs_sonic"].to_numpy(dtype=np.float64)
        vpvs_dem = ok_vs["vpvs_dem"].to_numpy(dtype=np.float64)
        vpvs_err = vpvs_dem - vpvs_sonic
        out["n_matched_vpvs"] = int(len(ok_vs))
        out["mae_vpvs"] = float(np.mean(np.abs(vpvs_err)))
        out["bias_vpvs"] = float(np.mean(vpvs_err))
        out["mean_vpvs_sonic"] = float(np.mean(vpvs_sonic))
        out["mean_vpvs_dem"] = float(np.mean(vpvs_dem))
    return out


def summary_by_hfu(val: pd.DataFrame) -> pd.DataFrame:
    """HFU-level validation aggregates."""
    ok = val[val["has_sonic_vp"]].copy()
    if ok.empty:
        return pd.DataFrame()
    agg = (
        ok.groupby("HFU")
        .agg(
            n=("Depth(m)", "count"),
            mean_vp_sonic=("vp_sonic_km_s", "mean"),
            mean_vp_dem=("vp_dem_km_s", "mean"),
            mean_vp_bias=("vp_bias_km_s", "mean"),
            mean_abs_vp_rel_error_pct=("vp_abs_rel_error_pct", "mean"),
            mean_vpvs_sonic=("vpvs_sonic", "mean"),
            mean_vpvs_dem=("vpvs_dem", "mean"),
        )
        .reset_index()
    )
    return agg


def plot_vp_dem_vs_sonic_depth(val: pd.DataFrame, out_path: Path) -> None:
    """Depth track: Vp DEM vs Vp sonic."""
    ok = val[val["has_sonic_vp"]].sort_values("Depth(m)")
    fig, ax = plt.subplots(figsize=(5.5, 8.0))
    ax.plot(
        ok["vp_sonic_km_s"],
        ok["Depth(m)"],
        "o-",
        color="#2ca02c",
        label="Vp sonic (DLIS)",
        markersize=4,
        linewidth=1.2,
    )
    ax.plot(
        ok["vp_dem_km_s"],
        ok["Depth(m)"],
        "s-",
        color="#1f77b4",
        label="Vp DEM (lab calib)",
        markersize=4,
        linewidth=1.2,
    )
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: DEM vs DSI sonic Vp")
    ax.invert_yaxis()
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vp_dem_vs_sonic_scatter(val: pd.DataFrame, out_path: Path) -> None:
    """Scatter Vp DEM vs Vp sonic colored by HFU."""
    ok = val[val["has_sonic_vp"]].copy()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        ax.scatter(
            sub["vp_sonic_km_s"],
            sub["vp_dem_km_s"],
            label="HFU{}".format(int(hfu)),
            c=colors.get(int(hfu), "#333333"),
            s=55,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.85,
        )
    lim_lo = min(ok["vp_sonic_km_s"].min(), ok["vp_dem_km_s"].min()) * 0.92
    lim_hi = max(ok["vp_sonic_km_s"].max(), ok["vp_dem_km_s"].max()) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.0, label="1:1")
    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Vp sonic (km/s)")
    ax.set_ylabel("Vp DEM calib (km/s)")
    ax.set_title("Validation: DEM vs DSI sonic Vp")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vpvs_dem_vs_sonic_depth(val: pd.DataFrame, out_path: Path) -> None:
    """Depth track: Vp/Vs DEM vs sonic."""
    ok = val[val["vpvs_sonic"].notna()].sort_values("Depth(m)")
    if ok.empty:
        return
    fig, ax = plt.subplots(figsize=(5.5, 8.0))
    ax.plot(
        ok["vpvs_sonic"],
        ok["Depth(m)"],
        "o-",
        color="#2ca02c",
        label="Vp/Vs sonic",
        markersize=4,
    )
    ax.plot(
        ok["vpvs_dem"],
        ok["Depth(m)"],
        "s-",
        color="#1f77b4",
        label="Vp/Vs DEM",
        markersize=4,
    )
    ax.set_xlabel("Vp/Vs")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: DEM vs DSI sonic Vp/Vs")
    ax.invert_yaxis()
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_vp_rel_error_vs_depth(val: pd.DataFrame, out_path: Path) -> None:
    """Relative Vp error vs depth."""
    ok = val[val["has_sonic_vp"]].sort_values("Depth(m)")
    fig, ax = plt.subplots(figsize=(5.5, 8.0))
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}
    for hfu in sorted(ok["HFU"].unique()):
        sub = ok[ok["HFU"] == hfu]
        ax.scatter(
            sub["vp_rel_error_pct"],
            sub["Depth(m)"],
            label="HFU{}".format(int(hfu)),
            c=colors.get(int(hfu), "#333333"),
            s=45,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Vp rel. error DEM vs sonic (%)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: Vp error vs depth")
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def update_manifest(metrics: Dict[str, float]) -> None:
    """Append validation summary to MANIFEST if present."""
    manifest = OUT_ROOT / "MANIFEST.txt"
    lines = [
        "",
        "--- Sonic vs DEM validation ---",
        "generated_utc: {}".format(metrics.get("generated_utc", "")),
        "n_matched_vp: {}".format(metrics.get("n_matched_vp", 0)),
        "mape_vp_pct: {:.2f}".format(metrics.get("mape_vp_pct", float("nan"))),
        "rmse_vp_km_s: {:.3f}".format(metrics.get("rmse_vp_km_s", float("nan"))),
        "pearson_r_vp: {:.4f}".format(metrics.get("pearson_r_vp", float("nan"))),
    ]
    if manifest.is_file():
        text = manifest.read_text(encoding="utf-8").rstrip()
        manifest.write_text(text + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    else:
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(
    merge_tol_m: float = DEFAULT_MERGE_TOL_M,
    sonic_csv: Optional[Path] = None,
    dem_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """Run validation pipeline."""
    ensure_dirs()
    sonic_path = sonic_csv if sonic_csv is not None else SONIC_CSV
    dem_path = dem_csv if dem_csv is not None else DEM_PROFILE_CSV

    sonic = load_sonic(sonic_path)
    dem = load_dem_profile(dem_path)
    merged = merge_dem_sonic(dem, sonic, merge_tol_m)
    val = build_validation_table(merged)
    val.to_csv(VAL_CSV, index=False, float_format="%.6f")

    summary = summary_by_hfu(val)
    if not summary.empty:
        summary.to_csv(SUMMARY_CSV, index=False, float_format="%.6f")

    metrics = compute_metrics(val)
    metrics["generated_utc"] = utc_now_iso()
    metrics["merge_tol_m"] = merge_tol_m
    METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    plot_vp_dem_vs_sonic_depth(val, FIGURES_DIR / "vp_dem_vs_sonic_depth.png")
    plot_vp_dem_vs_sonic_scatter(val, FIGURES_DIR / "vp_dem_vs_sonic_scatter.png")
    plot_vpvs_dem_vs_sonic_depth(val, FIGURES_DIR / "vpvs_dem_vs_sonic_depth.png")
    plot_vp_rel_error_vs_depth(val, FIGURES_DIR / "vp_rel_error_vs_depth.png")
    update_manifest(metrics)

    print("Sonic vs DEM validation complete: {}".format(OUT_ROOT))
    print("  matched: {}/{}".format(
        metrics.get("n_matched_vp", 0),
        metrics.get("n_profile_rows", 0),
    ))
    if metrics.get("n_matched_vp", 0) > 0:
        print("  MAPE Vp: {:.2f}%".format(metrics["mape_vp_pct"]))
        print("  RMSE Vp: {:.3f} km/s".format(metrics["rmse_vp_km_s"]))
        print("  Pearson r: {:.4f}".format(metrics["pearson_r_vp"]))
    return val


if __name__ == "__main__":
    args = parse_args()
    main(
        merge_tol_m=args.merge_tol_m,
        sonic_csv=Path(args.sonic_csv),
        dem_csv=Path(args.dem_csv),
    )
