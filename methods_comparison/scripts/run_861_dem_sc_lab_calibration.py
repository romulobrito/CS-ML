#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inverse calibration of DEM parameters per HFU vs ROCKPHYS lab Vp.

Outputs:
  methods_comparison/data/processed/dem_sc_runs/lab_calibration/
    MANIFEST.txt
    metrics.json
    tables/hfu_calibrated_params.csv
    tables/plug_validation_calibrated.csv
    tables/hfu_calibration_comparison.csv
  dem_sc_runs/hfu_calibration/hfu_lab_calibrated.csv  (for profile extrapolation)

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

from dem_sc_861_calibrate import (
    HfuCalibResult,
    PlugCalibRecord,
    build_plug_records,
    calibrate_all_hfus,
    global_validation_metrics,
    predict_plug_calibrated,
    run_loo_validation,
)
from ml_861_data import (
    DEM_SC_HFU_CALIB_ROOT,
    DEM_SC_LAB_VALIDATION_ROOT,
    DEM_SC_ROOT,
    ROOT,
    load_ct_samples,
)

OUT_ROOT = DEM_SC_ROOT / "lab_calibration"
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"
HFU_CALIB_OUT = DEM_SC_HFU_CALIB_ROOT / "hfu_lab_calibrated.csv"
HFU_CALIB_ROBUST_OUT = DEM_SC_HFU_CALIB_ROOT / "hfu_lab_calibrated_robust.csv"
LAB_VAL_CSV = DEM_SC_LAB_VALIDATION_ROOT / "tables" / "dem_vs_lab_validation.csv"
HFU_CT_STATS = DEM_SC_HFU_CALIB_ROOT / "hfu_ct_stats.csv"

# Optional robust run excluding orientation outlier with very low lab Vp.
ROBUST_EXCLUDE: Tuple[str, ...] = ("F2911V",)


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (OUT_ROOT, TABLES_DIR, FIGURES_DIR, DEM_SC_HFU_CALIB_ROOT):
        d.mkdir(parents=True, exist_ok=True)


def build_profile_hfu_table(
    hfu_calib_df: pd.DataFrame,
    hfu_ct_stats: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge lab-calibrated alpha/scale with CT HFU matrix moduli for profile use.

    HFU4 (no lab plugs) keeps CT fallback medians with scale=1.
    """
    ct_by_hfu = hfu_ct_stats.set_index("HFU")
    rows: List[dict] = []
    calibrated_hfus = set(hfu_calib_df["HFU"].astype(int).tolist())

    for _, cal in hfu_calib_df.iterrows():
        hfu = int(cal["HFU"])
        if hfu not in ct_by_hfu.index:
            continue
        ct = ct_by_hfu.loc[hfu]
        k_scale = float(cal["matrix_k_scale"])
        g_scale = float(cal["matrix_g_scale"])
        rows.append(
            {
                "HFU": hfu,
                "alpha": float(cal["alpha_calibrated"]),
                "alpha_ct_median": float(cal["alpha_ct_median"]),
                "matrix_k_gpa": float(ct["matrix_k_gpa_median"]) * k_scale,
                "matrix_g_gpa": float(ct["matrix_g_gpa_median"]) * g_scale,
                "matrix_k_gpa_ct": float(ct["matrix_k_gpa_median"]),
                "matrix_g_gpa_ct": float(ct["matrix_g_gpa_median"]),
                "matrix_k_scale": k_scale,
                "matrix_g_scale": g_scale,
                "matrix_rho_gcc": float(ct["matrix_rho_gcc_median"]),
                "param_source": "lab_calibrated_{}".format(cal["scenario_chosen"]),
                "n_ct_plugs": int(ct["n_plugs"]),
                "n_plugs_calib": int(cal["n_plugs_calib"]),
                "rmse_vp_after_km_s": float(cal["rmse_vp_after_km_s"]),
            }
        )

    # HFU4 fallback from CT stats (not in lab calib)
    if 4 not in calibrated_hfus and 4 not in [r["HFU"] for r in rows]:
        if 4 in ct_by_hfu.index:
            pass
    # Rebuild HFU4 from profile logic if missing
    if 4 not in [r["HFU"] for r in rows]:
        h2_row = next((r for r in rows if r["HFU"] == 2), None)
        h3_row = next((r for r in rows if r["HFU"] == 3), None)
        if h2_row and h3_row:
            rows.append(
                {
                    "HFU": 4,
                    "alpha": 0.5 * (h2_row["alpha"] + h3_row["alpha"]),
                    "alpha_ct_median": float("nan"),
                    "matrix_k_gpa": 0.5
                    * (h2_row["matrix_k_gpa_ct"] + h3_row["matrix_k_gpa_ct"]),
                    "matrix_g_gpa": 0.5
                    * (h2_row["matrix_g_gpa_ct"] + h3_row["matrix_g_gpa_ct"]),
                    "matrix_k_gpa_ct": 0.5
                    * (h2_row["matrix_k_gpa_ct"] + h3_row["matrix_k_gpa_ct"]),
                    "matrix_g_gpa_ct": 0.5
                    * (h2_row["matrix_g_gpa_ct"] + h3_row["matrix_g_gpa_ct"]),
                    "matrix_k_scale": 1.0,
                    "matrix_g_scale": 1.0,
                    "matrix_rho_gcc": 0.5
                    * (h2_row["matrix_rho_gcc"] + h3_row["matrix_rho_gcc"]),
                    "param_source": "fallback_hfu4_avg_hfu2_hfu3_lab",
                    "n_ct_plugs": 0,
                    "n_plugs_calib": 0,
                    "rmse_vp_after_km_s": float("nan"),
                }
            )

    return pd.DataFrame(rows).sort_values("HFU").reset_index(drop=True)


def plot_vp_before_after(plug_df: pd.DataFrame, out_path: Path) -> None:
    """Scatter lab Vp vs uncalibrated and calibrated DEM."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        plug_df["vp_lab_z_km_s"],
        plug_df["vp_dem_uncal_km_s"],
        label="Uncalibrated (CT alpha)",
        color="#4c72b0",
        s=60,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        plug_df["vp_lab_z_km_s"],
        plug_df["vp_dem_calib_km_s"],
        label="Lab-calibrated HFU",
        color="#55a868",
        s=60,
        edgecolors="k",
        linewidths=0.3,
    )
    lo = plug_df["vp_lab_z_km_s"].min() * 0.9
    hi = plug_df["vp_lab_z_km_s"].max() * 1.08
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Vp lab Z (km/s)")
    ax.set_ylabel("Vp DEM (km/s)")
    ax.set_title("Well 861: before vs after HFU calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_by_hfu(hfu_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart CT vs calibrated alpha per HFU."""
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    x = np.arange(len(hfu_df))
    w = 0.35
    ax.bar(x - w / 2, hfu_df["alpha_ct_median"], w, label="CT median", color="#4c72b0")
    ax.bar(x + w / 2, hfu_df["alpha_calibrated"], w, label="Lab calibrated", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(["HFU{}".format(int(h)) for h in hfu_df["HFU"]])
    ax.set_ylabel("Aspect ratio alpha")
    ax.set_title("HFU calibration: alpha CT vs lab-fit")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_error_before_after(plug_df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar of abs rel error before/after."""
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = np.arange(len(plug_df))
    w = 0.35
    ax.bar(
        x - w / 2,
        plug_df["vp_rel_error_uncal_pct"].abs(),
        w,
        label="|err| uncalibrated",
        color="#4c72b0",
    )
    ax.bar(
        x + w / 2,
        plug_df["vp_abs_rel_error_calib_pct"],
        w,
        label="|err| calibrated",
        color="#55a868",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plug_df["ct_sample_id"], rotation=45, ha="right")
    ax.set_ylabel("|Vp rel. error| (%)")
    ax.set_title("Per-plug |Vp error| before vs after calibration")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loo_error(loo_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of LOO |Vp rel error| per plug."""
    ok = loo_df[loo_df["status"] == "ok"].sort_values("vp_abs_rel_error_loo_pct")
    fig, ax = plt.subplots(figsize=(9.0, 4.5))
    x = np.arange(len(ok))
    w = 0.35
    ax.bar(
        x - w / 2,
        ok["vp_rel_error_uncal_pct"].abs(),
        w,
        label="Uncalibrated",
        color="#4c72b0",
    )
    ax.bar(
        x + w / 2,
        ok["vp_abs_rel_error_loo_pct"],
        w,
        label="LOO calibrated",
        color="#c44e52",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(ok["ct_sample_id"], rotation=45, ha="right")
    ax.set_ylabel("|Vp rel. error| (%)")
    ax.set_title("Leave-one-plug-out vs uncalibrated")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mape_comparison(
    m_insample: Dict[str, float],
    m_loo: Dict[str, float],
    m_uncal: Dict[str, float],
    out_path: Path,
) -> None:
    """Bar chart comparing MAPE: uncalibrated, in-sample, LOO."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    labels = ["Uncalibrated", "In-sample calib", "LOO calib"]
    values = [
        m_uncal["mape_vp_pct"],
        m_insample["mape_vp_pct"],
        m_loo["mape_vp_pct"],
    ]
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("MAPE Vp (%)")
    ax.set_title("Honest error: in-sample vs LOO")
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.5, "{:.1f}%".format(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_loo_manifest(
    m_uncal: Dict[str, float],
    m_insample: Dict[str, float],
    m_loo: Dict[str, float],
) -> None:
    """Append LOO summary to manifest."""
    path = OUT_ROOT / "MANIFEST.txt"
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    lines = [
        "",
        "--- Leave-one-plug-out validation ---",
        "Uncalibrated MAPE: {:.1f}%  RMSE: {:.3f} km/s".format(
            m_uncal["mape_vp_pct"], m_uncal["rmse_vp_km_s"]
        ),
        "In-sample MAPE:    {:.1f}%  RMSE: {:.3f} km/s".format(
            m_insample["mape_vp_pct"], m_insample["rmse_vp_km_s"]
        ),
        "LOO MAPE:          {:.1f}%  RMSE: {:.3f} km/s  r: {:.3f}".format(
            m_loo["mape_vp_pct"], m_loo["rmse_vp_km_s"], m_loo["pearson_r_vp"]
        ),
        "",
        "LOO is the honest generalization estimate (each plug predicted",
        "with HFU params fit on the remaining plugs only).",
        "",
        "tables/loo_validation.csv",
        "figures/loo_error_by_sample.png",
        "figures/mape_insample_vs_loo.png",
    ]
    path.write_text(existing + "\n".join(lines) + "\n", encoding="utf-8")


def run_loo_pipeline(
    plugs: Sequence[PlugCalibRecord],
    insample_metrics: Dict[str, float],
) -> Dict[str, object]:
    """Execute LOO CV and write artifacts."""
    loo_df = run_loo_validation(plugs)
    loo_df.to_csv(TABLES_DIR / "loo_validation.csv", index=False, float_format="%.6f")

    ok = loo_df[loo_df["status"] == "ok"]
    m_uncal = global_validation_metrics(ok, "vp_dem_uncal_km_s")
    m_loo = global_validation_metrics(ok, "vp_dem_loo_km_s")

    loo_metrics = {
        "n_folds": int(len(loo_df)),
        "n_ok": int(len(ok)),
        "uncalibrated": m_uncal,
        "insample_calibrated": insample_metrics,
        "loo_calibrated": m_loo,
        "mape_loo_minus_insample_pct_points": m_loo["mape_vp_pct"]
        - insample_metrics["mape_vp_pct"],
        "generated_utc": utc_now_iso(),
    }
    (OUT_ROOT / "metrics_loo.json").write_text(
        json.dumps(loo_metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_loo_error(loo_df, FIGURES_DIR / "loo_error_by_sample.png")
    plot_mape_comparison(
        insample_metrics,
        m_loo,
        m_uncal,
        FIGURES_DIR / "mape_insample_vs_loo.png",
    )
    write_loo_manifest(m_uncal, insample_metrics, m_loo)

    return {"loo_df": loo_df, "metrics_loo": loo_metrics}


def write_manifest(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float],
    hfu_df: pd.DataFrame,
    robust: bool,
    hfu_out_name: str,
) -> None:
    """Write calibration manifest."""
    lines = [
        "Well 861 -- DEM lab inverse calibration (Etapa 2e)",
        "Generated: {}".format(utc_now_iso()),
        "",
        "Method: per-HFU fit minimizing RMSE(Vp_DEM, Vp_lab).",
        "Scenarios tested: alpha_only vs alpha+matrix_scale; best RMSE chosen.",
        "Robust mode (exclude F2911V): {}".format(robust),
        "",
        "Global Vp metrics (all CT plugs with lab):",
        "  BEFORE  MAPE={:.1f}%  RMSE={:.3f} km/s  r={:.3f}".format(
            metrics_before["mape_vp_pct"],
            metrics_before["rmse_vp_km_s"],
            metrics_before["pearson_r_vp"],
        ),
        "  AFTER   MAPE={:.1f}%  RMSE={:.3f} km/s  r={:.3f}".format(
            metrics_after["mape_vp_pct"],
            metrics_after["rmse_vp_km_s"],
            metrics_after["pearson_r_vp"],
        ),
        "",
        "HFU calibrated parameters:",
    ]
    for _, r in hfu_df.iterrows():
        lines.append(
            "  HFU{}: alpha {:.3f} -> {:.3f}, scale K/G={:.3f}, scenario={}, RMSE {:.3f}->{:.3f} km/s".format(
                int(r["HFU"]),
                r["alpha_ct_median"],
                r["alpha_calibrated"],
                r["matrix_k_scale"],
                r["scenario_chosen"],
                r["rmse_vp_before_km_s"],
                r["rmse_vp_after_km_s"],
            )
        )
    lines.extend(
        [
            "",
            "Profile extrapolation table: ../hfu_calibration/{}".format(hfu_out_name),
            "",
            "Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md",
        ]
    )
    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_calibration(
    exclude_samples: Optional[Sequence[str]] = None,
    robust: bool = False,
) -> Dict[str, object]:
    """Execute HFU inverse calibration pipeline."""
    ensure_dirs()
    if not LAB_VAL_CSV.is_file():
        raise FileNotFoundError(
            "Run run_861_dem_sc_lab_validation.py first: {}".format(LAB_VAL_CSV)
        )
    if not HFU_CT_STATS.is_file():
        raise FileNotFoundError("Missing HFU CT stats: {}".format(HFU_CT_STATS))

    excl = list(exclude_samples or ())
    if robust:
        excl = list(set(excl) | set(ROBUST_EXCLUDE))

    ct_df = load_ct_samples()
    lab_val = pd.read_csv(LAB_VAL_CSV)
    hfu_ct = pd.read_csv(HFU_CT_STATS)

    plugs = build_plug_records(ct_df, lab_val, exclude_samples=excl)
    hfu_calib_df, chosen = calibrate_all_hfus(plugs)

    plug_rows: List[dict] = []
    for plug in plugs:
        hfu_res = chosen[plug.hfu]
        plug_rows.append(predict_plug_calibrated(plug, hfu_res))
    plug_df = pd.DataFrame(plug_rows)

    metrics_before = global_validation_metrics(plug_df, "vp_dem_uncal_km_s")
    metrics_after = global_validation_metrics(plug_df, "vp_dem_calib_km_s")

    hfu_calib_df.to_csv(
        TABLES_DIR / "hfu_calibrated_params.csv",
        index=False,
        float_format="%.6f",
    )
    plug_df.to_csv(
        TABLES_DIR / "plug_validation_calibrated.csv",
        index=False,
        float_format="%.6f",
    )

    # Side-by-side HFU comparison (alpha-only vs joint)
    hfu_calib_df.to_csv(
        TABLES_DIR / "hfu_calibration_comparison.csv",
        index=False,
        float_format="%.6f",
    )

    profile_hfu = build_profile_hfu_table(hfu_calib_df, hfu_ct)
    hfu_out = HFU_CALIB_ROBUST_OUT if robust else HFU_CALIB_OUT
    profile_hfu.to_csv(hfu_out, index=False, float_format="%.6f")

    plot_vp_before_after(plug_df, FIGURES_DIR / "vp_before_after_calib.png")
    plot_alpha_by_hfu(hfu_calib_df, FIGURES_DIR / "alpha_calibrated_by_hfu.png")
    plot_error_before_after(plug_df, FIGURES_DIR / "vp_error_before_after.png")

    metrics = {
        "n_plugs_calibrated": int(len(plugs)),
        "n_hfu_calibrated": int(len(hfu_calib_df)),
        "robust_exclude_f2911v": robust,
        "excluded_samples": excl,
        "before": metrics_before,
        "after": metrics_after,
        "mape_improvement_pct_points": metrics_before["mape_vp_pct"]
        - metrics_after["mape_vp_pct"],
        "rmse_improvement_km_s": metrics_before["rmse_vp_km_s"]
        - metrics_after["rmse_vp_km_s"],
        "generated_utc": utc_now_iso(),
        "smoke": False,
    }
    metrics_name = "metrics_robust.json" if robust else "metrics_standard.json"
    (OUT_ROOT / metrics_name).write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (OUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(metrics_before, metrics_after, hfu_calib_df, robust, hfu_out.name)

    loo_result = run_loo_pipeline(plugs, metrics_after)

    return {
        "metrics": metrics,
        "hfu_calib": hfu_calib_df,
        "plug_df": plug_df,
        "hfu_out": str(hfu_out),
        "loo": loo_result["metrics_loo"],
    }


def parse_args() -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Inverse calibrate DEM HFU params vs ROCKPHYS lab Vp",
    )
    parser.add_argument(
        "--robust",
        action="store_true",
        help="Exclude F2911V (F2911H lab outlier) from HFU2 calibration",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    result = run_calibration(robust=args.robust)
    m = result["metrics"]
    loo = result["loo"]
    print(
        "Calibration: MAPE {:.1f}% -> {:.1f}%  RMSE {:.3f} -> {:.3f} km/s".format(
            m["before"]["mape_vp_pct"],
            m["after"]["mape_vp_pct"],
            m["before"]["rmse_vp_km_s"],
            m["after"]["rmse_vp_km_s"],
        )
    )
    print(
        "LOO (honest): MAPE {:.1f}%  RMSE {:.3f} km/s  r={:.3f}".format(
            loo["loo_calibrated"]["mape_vp_pct"],
            loo["loo_calibrated"]["rmse_vp_km_s"],
            loo["loo_calibrated"]["pearson_r_vp"],
        )
    )
    print("HFU table: {}".format(result["hfu_out"]))


if __name__ == "__main__":
    main()
