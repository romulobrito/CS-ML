#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-check Berryman DEM: dem_sc_861_core vs rockphypy.EM.Berryman_DEM.

Outputs:
  methods_comparison/data/processed/dem_sc_runs/crosscheck_rockphypy/
    MANIFEST.txt
    metrics.json
    tables/dem_crosscheck_plugs.csv
    tables/dem_crosscheck_sweep.csv
    tables/dem_crosscheck_step_sensitivity.csv
    figures/k_rel_diff_plugs.png

Requires: pip install rockphypy (optional dependency; script exits with message if missing).

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dem_sc_861_core import (
    berryman_dem,
    dry_density,
    matrix_from_solids,
    velocities_from_moduli,
)
from ml_861_data import ROOT

OUT_ROOT = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "crosscheck_rockphypy"
)
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"

PLUG_POC_CSV = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "poc_10plugs"
    / "tables"
    / "plug_dem_sc_summary.csv"
)
PLUG_CALIB_CSV = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "dem_sc_runs"
    / "lab_calibration"
    / "tables"
    / "plug_validation_calibrated.csv"
)

ALPHA_SWEEP: Tuple[float, ...] = (0.05, 0.19, 0.55, 0.95)
PHI_SWEEP: Tuple[float, ...] = (0.05, 0.11, 0.18)
STEP_GRID: Tuple[float, ...] = (0.001, 0.005, 0.01, 0.02)


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (OUT_ROOT, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def rel_pct_diff(reference: float, other: float) -> float:
    """Relative percent difference |other - reference| / |reference|."""
    denom = max(abs(reference), 1.0e-12)
    return 100.0 * abs(other - reference) / denom


def import_rockphypy():
    """Import rockphypy.EM or raise with install hint."""
    try:
        from rockphypy import EM  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "rockphypy is not installed. Run: .venv/bin/pip install rockphypy"
        ) from exc
    return EM


def rockphypy_dem_moduli(
    em_module,
    km_gpa: float,
    gm_gpa: float,
    ki_gpa: float,
    gi_gpa: float,
    alpha: float,
    phi: float,
) -> Tuple[float, float]:
    """Final K_dry, G_dry from rockphypy Berryman_DEM (last ODE step)."""
    k_series, g_series, _t = em_module.Berryman_DEM(
        km_gpa, gm_gpa, ki_gpa, gi_gpa, alpha, phi
    )
    k_final = float(np.asarray(k_series).ravel()[-1])
    g_final = float(np.asarray(g_series).ravel()[-1])
    return k_final, g_final


def local_dem_moduli(
    km_gpa: float,
    gm_gpa: float,
    ki_gpa: float,
    gi_gpa: float,
    alpha: float,
    phi: float,
    t_inc: float = 0.005,
) -> Tuple[float, float]:
    """Final K_dry, G_dry from dem_sc_861_core (optional step size)."""
    if abs(t_inc - 0.005) < 1.0e-12:
        result = berryman_dem(km_gpa, gm_gpa, ki_gpa, gi_gpa, alpha, phi)
        return result.k_gpa, result.g_gpa
    return _berryman_dem_step(km_gpa, gm_gpa, ki_gpa, gi_gpa, alpha, phi, t_inc)


def _berryman_dem_step(
    km_gpa: float,
    gm_gpa: float,
    ki_gpa: float,
    gi_gpa: float,
    alpha: float,
    phi: float,
    t_inc: float,
) -> Tuple[float, float]:
    """Berryman DEM with custom integration step (local ODE only)."""
    from scipy.integrate import odeint

    from dem_sc_861_core import _dem_ode, clip_porosity

    phi_use = clip_porosity(phi)
    if phi_use <= 0.0:
        return km_gpa, gm_gpa
    t_grid = np.arange(0.0, phi_use + t_inc, t_inc, dtype=np.float64)
    params = (float(gi_gpa), float(ki_gpa), float(alpha))
    y0 = [float(km_gpa), float(gm_gpa)]
    sol = odeint(_dem_ode, y0, t_grid, args=(params,))
    k_final = float(sol[-1, 0])
    g_final = float(sol[-1, 1])
    if k_final <= 0.0 or g_final <= 0.0:
        raise ValueError("DEM produced non-positive moduli")
    return k_final, g_final


def build_plug_cases() -> pd.DataFrame:
    """Merge POC matrix inputs with calibrated alpha/scale per plug."""
    poc = pd.read_csv(PLUG_POC_CSV)
    calib = pd.read_csv(PLUG_CALIB_CSV)
    merged = poc.merge(
        calib[
            [
                "ct_sample_id",
                "alpha_calibrated",
                "matrix_k_scale",
                "matrix_g_scale",
            ]
        ],
        left_on="sample_id",
        right_on="ct_sample_id",
        how="inner",
    )
    if merged.empty:
        raise RuntimeError("No overlapping plugs between POC and calibration tables")
    return merged


def run_plug_crosscheck(em_module) -> pd.DataFrame:
    """Compare local vs rockphypy DEM on 10 CT plugs (calibrated alpha/scale)."""
    plugs = build_plug_cases()
    rows: List[Dict[str, object]] = []
    for _, row in plugs.iterrows():
        phi = float(row["phi_lab"])
        alpha = float(row["alpha_calibrated"])
        k_scale = float(row["matrix_k_scale"])
        g_scale = float(row["matrix_g_scale"])
        s1 = float(row["solid1_pct"])
        s2 = float(row["solid2_pct"])
        matrix = matrix_from_solids(s1, s2)
        km = matrix.k_gpa * k_scale
        gm = matrix.g_gpa * g_scale
        ki, gi = 0.0, 0.0

        k_loc, g_loc = local_dem_moduli(km, gm, ki, gi, alpha, phi)
        k_rp, g_rp = rockphypy_dem_moduli(em_module, km, gm, ki, gi, alpha, phi)
        rho = dry_density(phi, matrix.rho_gcc)
        vp_loc = velocities_from_moduli(k_loc, g_loc, rho).vp_km_s
        vp_rp = velocities_from_moduli(k_rp, g_rp, rho).vp_km_s

        rows.append(
            {
                "sample_id": row["sample_id"],
                "HFU": int(row["HFU"]),
                "phi_lab": phi,
                "alpha_calibrated": alpha,
                "matrix_k_gpa": km,
                "matrix_g_gpa": gm,
                "k_dry_local_gpa": k_loc,
                "g_dry_local_gpa": g_loc,
                "k_dry_rockphypy_gpa": k_rp,
                "g_dry_rockphypy_gpa": g_rp,
                "k_rel_diff_pct": rel_pct_diff(k_loc, k_rp),
                "g_rel_diff_pct": rel_pct_diff(g_loc, g_rp),
                "vp_dry_local_km_s": vp_loc,
                "vp_dry_rockphypy_km_s": vp_rp,
                "vp_rel_diff_pct": rel_pct_diff(vp_loc, vp_rp),
            }
        )
    return pd.DataFrame(rows)


def run_synthetic_sweep(em_module, km: float, gm: float) -> pd.DataFrame:
    """Grid over alpha and phi with fixed matrix moduli."""
    rows: List[Dict[str, object]] = []
    for alpha in ALPHA_SWEEP:
        for phi in PHI_SWEEP:
            k_loc, g_loc = local_dem_moduli(km, gm, 0.0, 0.0, alpha, phi)
            k_rp, g_rp = rockphypy_dem_moduli(em_module, km, gm, 0.0, 0.0, alpha, phi)
            rows.append(
                {
                    "alpha": alpha,
                    "phi": phi,
                    "matrix_k_gpa": km,
                    "matrix_g_gpa": gm,
                    "k_rel_diff_pct": rel_pct_diff(k_loc, k_rp),
                    "g_rel_diff_pct": rel_pct_diff(g_loc, g_rp),
                }
            )
    return pd.DataFrame(rows)


def run_step_sensitivity(
    km: float,
    gm: float,
    alpha: float,
    phi: float,
) -> pd.DataFrame:
    """Local DEM stability vs integration step (no rockphypy)."""
    rows: List[Dict[str, object]] = []
    ref_k, ref_g = local_dem_moduli(km, gm, 0.0, 0.0, alpha, phi, t_inc=0.001)
    for t_inc in STEP_GRID:
        k_val, g_val = local_dem_moduli(km, gm, 0.0, 0.0, alpha, phi, t_inc=t_inc)
        rows.append(
            {
                "t_inc": t_inc,
                "alpha": alpha,
                "phi": phi,
                "k_dry_gpa": k_val,
                "g_dry_gpa": g_val,
                "k_rel_diff_vs_1e-3_pct": rel_pct_diff(ref_k, k_val),
                "g_rel_diff_vs_1e-3_pct": rel_pct_diff(ref_g, g_val),
            }
        )
    return pd.DataFrame(rows)


def plot_plug_k_diff(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of K_dry relative difference per plug."""
    work = df.sort_values("sample_id")
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    x = np.arange(len(work))
    ax.bar(x, work["k_rel_diff_pct"], color="#1f77b4", label="K_dry")
    ax.bar(
        x,
        work["g_rel_diff_pct"],
        bottom=0.0,
        alpha=0.55,
        color="#ff7f0e",
        label="G_dry",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(work["sample_id"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Rel. diff local vs rockphypy (%)")
    ax.set_title("Berryman DEM cross-check: 10 CT plugs (calibrated params)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_manifest(metrics: Dict[str, object]) -> None:
    """Write MANIFEST.txt."""
    lines = [
        "DEM cross-check: dem_sc_861_core vs rockphypy.EM.Berryman_DEM",
        "Generated: {}".format(metrics.get("generated_utc", "")),
        "",
        "Tables:",
        "  tables/dem_crosscheck_plugs.csv",
        "  tables/dem_crosscheck_sweep.csv",
        "  tables/dem_crosscheck_step_sensitivity.csv",
        "Figures:",
        "  figures/k_rel_diff_plugs.png",
        "",
        "Plug K_dry mean rel diff: {:.4f}%".format(
            metrics.get("plugs_k_rel_diff_mean_pct", float("nan"))
        ),
        "Plug Vp mean rel diff: {:.4f}%".format(
            metrics.get("plugs_vp_rel_diff_mean_pct", float("nan"))
        ),
        "Sweep max K rel diff: {:.4f}%".format(
            metrics.get("sweep_k_rel_diff_max_pct", float("nan"))
        ),
        "Note: rockphypy DEM() calls PQ with K and G swapped; see metrics.json.",
    ]
    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_crosscheck() -> Dict[str, object]:
    """Execute full cross-check and write artifacts."""
    ensure_dirs()
    em_module = import_rockphypy()

    plugs_df = run_plug_crosscheck(em_module)
    plugs_df.to_csv(
        TABLES_DIR / "dem_crosscheck_plugs.csv",
        index=False,
        float_format="%.6f",
    )

    # Representative matrix moduli (HFU2 median scale on 50/50 solids)
    mat = matrix_from_solids(50.0, 50.0)
    km_ref = mat.k_gpa * 0.705585
    gm_ref = mat.g_gpa * 0.705585
    sweep_df = run_synthetic_sweep(em_module, km_ref, gm_ref)
    sweep_df.to_csv(
        TABLES_DIR / "dem_crosscheck_sweep.csv",
        index=False,
        float_format="%.6f",
    )

    step_df = run_step_sensitivity(km_ref, gm_ref, alpha=0.190827, phi=0.075)
    step_df.to_csv(
        TABLES_DIR / "dem_crosscheck_step_sensitivity.csv",
        index=False,
        float_format="%.6f",
    )

    plot_plug_k_diff(plugs_df, FIGURES_DIR / "k_rel_diff_plugs.png")

    metrics: Dict[str, object] = {
        "well_id": "861",
        "rockphypy_available": True,
        "n_plugs": int(len(plugs_df)),
        "plugs_k_rel_diff_mean_pct": float(plugs_df["k_rel_diff_pct"].mean()),
        "plugs_k_rel_diff_max_pct": float(plugs_df["k_rel_diff_pct"].max()),
        "plugs_g_rel_diff_mean_pct": float(plugs_df["g_rel_diff_pct"].mean()),
        "plugs_g_rel_diff_max_pct": float(plugs_df["g_rel_diff_pct"].max()),
        "plugs_vp_rel_diff_mean_pct": float(plugs_df["vp_rel_diff_pct"].mean()),
        "plugs_vp_rel_diff_max_pct": float(plugs_df["vp_rel_diff_pct"].max()),
        "sweep_k_rel_diff_max_pct": float(sweep_df["k_rel_diff_pct"].max()),
        "sweep_g_rel_diff_max_pct": float(sweep_df["g_rel_diff_pct"].max()),
        "step_sensitivity_max_k_pct": float(step_df["k_rel_diff_vs_1e-3_pct"].max()),
        "rockphypy_pq_order_note": (
            "rockphypy EM.DEM calls PQ(G_eff, K_eff, ...) instead of PQ(K_eff, G_eff, ...); "
            "local pq_factors matches EM.PQ(Km, Gm, Ki, Gi, alpha) at t=0."
        ),
        "generated_utc": utc_now_iso(),
    }
    (OUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    write_manifest(metrics)
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry."""
    parser = argparse.ArgumentParser(
        description="Cross-check Berryman DEM: dem_sc_861_core vs rockphypy"
    )
    parser.parse_args(argv)
    try:
        metrics = run_crosscheck()
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    print(
        "OK crosscheck_rockphypy plugs={} K_mean_diff={:.2f}% Vp_mean_diff={:.2f}%".format(
            metrics["n_plugs"],
            metrics["plugs_k_rel_diff_mean_pct"],
            metrics["plugs_vp_rel_diff_mean_pct"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
