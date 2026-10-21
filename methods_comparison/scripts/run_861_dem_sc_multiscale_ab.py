#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A/B experiment: monoscale calibrated (M1) vs sequential multiscale (M2a, M2b).

Additive to the existing DEM/SC pipeline; does not change profile extrapolation.

Planning: methods_comparison/planning/etapa2f_dem_multiscale_ab_poco861.md
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

from dem_sc_861_calibrate import global_validation_metrics
from dem_sc_861_multiscale import (
    GAIN_THRESHOLD_MAPE_PP,
    build_multiscale_plug_records,
    compute_hfu_ar_micro_oracle_median,
    compute_hfu_multiscale_medians,
    decide_recommendation,
    load_hfu_matrix_scales,
    multiscale_plug_row,
    predict_multiscale_m2a,
    predict_multiscale_m2b,
    predict_multiscale_m2b_forward,
)
from ml_861_data import (
    DEM_SC_HFU_CALIB_ROOT,
    DEM_SC_LAB_CALIBRATION_ROOT,
    DEM_SC_LAB_VALIDATION_ROOT,
    DEM_SC_MULTISCALE_AB_ROOT,
    load_ct_samples,
)

ROBUST_EXCLUDE: Tuple[str, ...] = ("F2911V",)

OUT_ROOT = DEM_SC_MULTISCALE_AB_ROOT
TABLES_DIR = OUT_ROOT / "tables"
FIGURES_DIR = OUT_ROOT / "figures"
HFU_CALIB_CSV = DEM_SC_HFU_CALIB_ROOT / "hfu_lab_calibrated.csv"
M1_PLUG_CSV = DEM_SC_LAB_CALIBRATION_ROOT / "tables" / "plug_validation_calibrated.csv"
M1_LOO_CSV = DEM_SC_LAB_CALIBRATION_ROOT / "tables" / "loo_validation.csv"
LAB_VAL_CSV = DEM_SC_LAB_VALIDATION_ROOT / "tables" / "dem_vs_lab_validation.csv"


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (OUT_ROOT, TABLES_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_m1_plug_table() -> pd.DataFrame:
    """Load monoscale calibrated plug validation (M1)."""
    if not M1_PLUG_CSV.is_file():
        raise FileNotFoundError(
            "M1 plug table missing: {}. Run run_861_dem_sc_lab_calibration.py.".format(
                M1_PLUG_CSV
            )
        )
    return pd.read_csv(M1_PLUG_CSV)


def load_m1_loo_table() -> pd.DataFrame:
    """Load monoscale LOO validation (M1)."""
    if not M1_LOO_CSV.is_file():
        raise FileNotFoundError(
            "M1 LOO table missing: {}. Run run_861_dem_sc_lab_calibration.py.".format(
                M1_LOO_CSV
            )
        )
    return pd.read_csv(M1_LOO_CSV)


def load_lab_validation() -> pd.DataFrame:
    """Load lab validation merge table."""
    if not LAB_VAL_CSV.is_file():
        raise FileNotFoundError("Lab validation missing: {}".format(LAB_VAL_CSV))
    return pd.read_csv(LAB_VAL_CSV)


def load_hfu_calib() -> pd.DataFrame:
    """Load HFU calibration parameters from M1."""
    if not HFU_CALIB_CSV.is_file():
        raise FileNotFoundError("HFU calib missing: {}".format(HFU_CALIB_CSV))
    return pd.read_csv(HFU_CALIB_CSV)


def run_m2a_comparison(
    records: Sequence,
    m1_df: pd.DataFrame,
    scales: Dict[int, Tuple[float, float]],
) -> pd.DataFrame:
    """Run M2a per plug and build comparison rows."""
    m1_by_id = m1_df.set_index("ct_sample_id")
    rows: List[dict] = []
    for rec in records:
        k_scale, g_scale = scales[rec.hfu]
        fwd = predict_multiscale_m2a(rec, k_scale, g_scale)
        vp_m1 = float(m1_by_id.loc[rec.ct_sample_id, "vp_dem_calib_km_s"])
        rows.append(
            multiscale_plug_row(
                rec,
                "M2a_oracle_per_plug",
                fwd,
                rec.f_meso,
                rec.f_micro,
                rec.ar_meso,
                vp_m1,
                k_scale,
                g_scale,
            )
        )
    return pd.DataFrame(rows)


def run_m2b_comparison(
    records: Sequence,
    m1_df: pd.DataFrame,
    scales: Dict[int, Tuple[float, float]],
    hfu_medians: Optional[Dict] = None,
) -> pd.DataFrame:
    """Run M2b with HFU medians (in-sample or LOO-recomputed)."""
    if hfu_medians is None:
        hfu_medians = compute_hfu_multiscale_medians(records)
    m1_by_id = m1_df.set_index("ct_sample_id")
    rows: List[dict] = []
    for rec in records:
        med = hfu_medians[rec.hfu]
        k_scale, g_scale = scales[rec.hfu]
        fwd = predict_multiscale_m2b(rec, med, k_scale, g_scale)
        vp_m1 = float(m1_by_id.loc[rec.ct_sample_id, "vp_dem_calib_km_s"])
        rows.append(
            multiscale_plug_row(
                rec,
                "M2b_oracle_hfu_median",
                fwd,
                med.f_meso_median,
                med.f_micro_median,
                med.ar_meso_median,
                vp_m1,
                k_scale,
                g_scale,
            )
        )
    return pd.DataFrame(rows)


def run_m2b_forward_comparison(
    records: Sequence,
    m1_df: pd.DataFrame,
    scales: Dict[int, Tuple[float, float]],
) -> pd.DataFrame:
    """M2b forward in-sample: HFU medians + HFU median oracle AR_micro, no plug fit."""
    hfu_medians = compute_hfu_multiscale_medians(records)
    ar_micro_by_hfu = compute_hfu_ar_micro_oracle_median(records, scales, hfu_medians)
    m1_by_id = m1_df.set_index("ct_sample_id")
    rows: List[dict] = []
    for rec in records:
        med = hfu_medians[rec.hfu]
        k_scale, g_scale = scales[rec.hfu]
        ar_micro_hfu = ar_micro_by_hfu[rec.hfu]
        fwd = predict_multiscale_m2b_forward(rec, med, ar_micro_hfu, k_scale, g_scale)
        vp_m1 = float(m1_by_id.loc[rec.ct_sample_id, "vp_dem_calib_km_s"])
        rows.append(
            {
                "ct_sample_id": rec.ct_sample_id,
                "HFU": rec.hfu,
                "model": "M2b_forward",
                "phi_lab_pu": rec.phi_lab,
                "f_meso_used": med.f_meso_median,
                "f_micro_used": med.f_micro_median,
                "ar_meso_used": med.ar_meso_median,
                "ar_micro_hfu_median": ar_micro_hfu,
                "vp_lab_z_km_s": rec.vp_lab_z_km_s,
                "vp_m1_calib_km_s": vp_m1,
                "vp_m2_km_s": fwd.vp_dem_km_s,
                "vp_abs_rel_error_m1_pct": abs(
                    100.0 * (vp_m1 - rec.vp_lab_z_km_s) / rec.vp_lab_z_km_s
                ),
                "vp_abs_rel_error_m2_pct": fwd.rel_error_pct,
            }
        )
    return pd.DataFrame(rows)


def run_m2b_loo_forward(
    records: Sequence,
    m1_loo_df: pd.DataFrame,
    scales: Dict[int, Tuple[float, float]],
) -> pd.DataFrame:
    """LOO M2b forward: HFU stats from train only; hold-out without AR_micro inversion."""
    m1_loo_by_id = m1_loo_df.set_index("ct_sample_id")
    rows: List[dict] = []
    for holdout in records:
        train = [r for r in records if r.ct_sample_id != holdout.ct_sample_id]
        train_hfu = [r for r in train if r.hfu == holdout.hfu]
        k_scale, g_scale = scales[holdout.hfu]
        vp_m1_loo = float(m1_loo_by_id.loc[holdout.ct_sample_id, "vp_dem_loo_km_s"])
        vp_lab = holdout.vp_lab_z_km_s

        if len(train_hfu) < 1:
            rows.append(
                {
                    "ct_sample_id": holdout.ct_sample_id,
                    "HFU": holdout.hfu,
                    "held_out": holdout.ct_sample_id,
                    "status": "no_train_hfu",
                    "n_train_hfu": 0,
                    "vp_lab_z_km_s": vp_lab,
                    "vp_m1_loo_km_s": vp_m1_loo,
                    "vp_m2b_forward_loo_km_s": float("nan"),
                    "vp_abs_rel_error_m1_loo_pct": abs(
                        100.0 * (vp_m1_loo - vp_lab) / vp_lab
                    ),
                    "vp_abs_rel_error_m2b_forward_loo_pct": float("nan"),
                }
            )
            continue

        hfu_medians = compute_hfu_multiscale_medians(train)
        med = hfu_medians[holdout.hfu]
        ar_micro_by_hfu = compute_hfu_ar_micro_oracle_median(train_hfu, scales, hfu_medians)
        ar_micro_hfu = ar_micro_by_hfu[holdout.hfu]
        fwd = predict_multiscale_m2b_forward(holdout, med, ar_micro_hfu, k_scale, g_scale)
        err_m2 = 100.0 * (fwd.vp_dem_km_s - vp_lab) / vp_lab
        err_m1 = 100.0 * (vp_m1_loo - vp_lab) / vp_lab
        rows.append(
            {
                "ct_sample_id": holdout.ct_sample_id,
                "HFU": holdout.hfu,
                "held_out": holdout.ct_sample_id,
                "status": "ok",
                "n_train_hfu": len(train_hfu),
                "f_meso_median_loo": med.f_meso_median,
                "f_micro_median_loo": med.f_micro_median,
                "ar_meso_median_loo": med.ar_meso_median,
                "ar_micro_hfu_median_loo": ar_micro_hfu,
                "vp_lab_z_km_s": vp_lab,
                "vp_m1_loo_km_s": vp_m1_loo,
                "vp_m2b_forward_loo_km_s": fwd.vp_dem_km_s,
                "vp_rel_error_m1_loo_pct": err_m1,
                "vp_rel_error_m2b_forward_loo_pct": err_m2,
                "vp_abs_rel_error_m1_loo_pct": abs(err_m1),
                "vp_abs_rel_error_m2b_forward_loo_pct": abs(err_m2),
            }
        )
    return pd.DataFrame(rows)


def run_m2b_loo_oracle(
    records: Sequence,
    m1_loo_df: pd.DataFrame,
    scales: Dict[int, Tuple[float, float]],
) -> pd.DataFrame:
    """LOO M2b oracle (diagnostic): still inverts AR_micro on held-out plug."""
    m1_loo_by_id = m1_loo_df.set_index("ct_sample_id")
    rows: List[dict] = []
    for holdout in records:
        train = [r for r in records if r.ct_sample_id != holdout.ct_sample_id]
        train_hfu = [r for r in train if r.hfu == holdout.hfu]
        k_scale, g_scale = scales[holdout.hfu]
        vp_m1_loo = float(m1_loo_by_id.loc[holdout.ct_sample_id, "vp_dem_loo_km_s"])
        vp_lab = holdout.vp_lab_z_km_s

        if len(train_hfu) < 1:
            continue

        hfu_medians = compute_hfu_multiscale_medians(train)
        med = hfu_medians[holdout.hfu]
        fwd = predict_multiscale_m2b(holdout, med, k_scale, g_scale)
        err_m2 = 100.0 * (fwd.vp_dem_km_s - vp_lab) / vp_lab
        rows.append(
            {
                "ct_sample_id": holdout.ct_sample_id,
                "HFU": holdout.hfu,
                "vp_m2b_oracle_loo_km_s": fwd.vp_dem_km_s,
                "vp_abs_rel_error_m2b_oracle_loo_pct": abs(err_m2),
            }
        )
    return pd.DataFrame(rows)


def build_summary_metrics(
    m1_df: pd.DataFrame,
    m2a_oracle_df: pd.DataFrame,
    m2b_oracle_df: pd.DataFrame,
    m2b_forward_df: pd.DataFrame,
    m1_loo_df: pd.DataFrame,
    m2b_forward_loo_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate MAPE/RMSE for all models."""
    m1_metrics = global_validation_metrics(m1_df, "vp_dem_calib_km_s")
    m2a_metrics = global_validation_metrics(
        m2a_oracle_df.assign(vp_pred=m2a_oracle_df["vp_m2_km_s"]), "vp_pred"
    )
    m2b_oracle_metrics = global_validation_metrics(
        m2b_oracle_df.assign(vp_pred=m2b_oracle_df["vp_m2_km_s"]), "vp_pred"
    )
    m2b_forward_metrics = global_validation_metrics(
        m2b_forward_df.assign(vp_pred=m2b_forward_df["vp_m2_km_s"]), "vp_pred"
    )
    m1_loo_ok = m1_loo_df[m1_loo_df["status"] == "ok"].copy()
    m1_loo_metrics = global_validation_metrics(m1_loo_ok, "vp_dem_loo_km_s")
    m2b_loo_ok = m2b_forward_loo_df[m2b_forward_loo_df["status"] == "ok"].copy()
    m2b_forward_loo_metrics = global_validation_metrics(
        m2b_loo_ok, "vp_m2b_forward_loo_km_s"
    )

    rows = [
        {"model": "M1_monoscale_calibrated", "scope": "in_sample", **m1_metrics},
        {"model": "M2a_oracle_per_plug", "scope": "in_sample", **m2a_metrics},
        {"model": "M2b_oracle_hfu_median", "scope": "in_sample", **m2b_oracle_metrics},
        {"model": "M2b_forward_hfu_median", "scope": "in_sample", **m2b_forward_metrics},
        {"model": "M1_monoscale_calibrated", "scope": "loo", **m1_loo_metrics},
        {
            "model": "M2b_forward_hfu_median",
            "scope": "loo",
            **m2b_forward_loo_metrics,
        },
    ]
    return pd.DataFrame(rows)


def plot_vp_crossplot(
    m1_df: pd.DataFrame,
    m2b_forward_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Lab Vp vs M1 and M2b forward predictions."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(
        m1_df["vp_lab_z_km_s"],
        m1_df["vp_dem_calib_km_s"],
        label="M1 mono calib",
        color="#4c72b0",
        s=60,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.scatter(
        m2b_forward_df["vp_lab_z_km_s"],
        m2b_forward_df["vp_m2_km_s"],
        label="M2b forward HFU",
        color="#55a868",
        s=60,
        edgecolors="k",
        linewidths=0.3,
    )
    lo = m1_df["vp_lab_z_km_s"].min() * 0.9
    hi = m1_df["vp_lab_z_km_s"].max() * 1.08
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Vp lab Z (km/s)")
    ax.set_ylabel("Vp predicted (km/s)")
    ax.set_title("Well 861: M1 vs M2b forward (in-sample)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_abs_error_by_plug_three_models(
    m2a_oracle_df: pd.DataFrame,
    m2b_forward_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Grouped bar: |rel error| M1 vs M2a oracle vs M2b forward per plug."""
    merged = m2a_oracle_df.merge(
        m2b_forward_df[["ct_sample_id", "vp_abs_rel_error_m2_pct"]].rename(
            columns={"vp_abs_rel_error_m2_pct": "vp_abs_err_m2b_forward_pct"}
        ),
        on="ct_sample_id",
    )
    fig, ax = plt.subplots(figsize=(10.0, 4.5))
    x = np.arange(len(merged))
    w = 0.25
    ax.bar(x - w, merged["vp_abs_rel_error_m1_pct"], w, label="M1", color="#4c72b0")
    ax.bar(x, merged["vp_abs_rel_error_m2_pct"], w, label="M2a oracle", color="#dd8452")
    ax.bar(x + w, merged["vp_abs_err_m2b_forward_pct"], w, label="M2b forward", color="#55a868")
    ax.set_xticks(x)
    ax.set_xticklabels(merged["ct_sample_id"], rotation=45, ha="right")
    ax.set_ylabel("|Vp rel error| (%)")
    ax.set_title("Absolute Vp error: M1 vs M2a oracle vs M2b forward")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_combined_plug_table(
    m2a_oracle_df: pd.DataFrame,
    m2b_oracle_df: pd.DataFrame,
    m2b_forward_df: pd.DataFrame,
) -> pd.DataFrame:
    """Single table with M1, oracle and forward multiscale columns per plug."""
    base = m2a_oracle_df[
        [
            "ct_sample_id",
            "HFU",
            "phi_lab_pu",
            "f_meso_used",
            "f_micro_used",
            "ar_meso_used",
            "vp_lab_z_km_s",
            "vp_m1_calib_km_s",
            "vp_abs_rel_error_m1_pct",
        ]
    ].rename(
        columns={
            "f_meso_used": "f_meso_m2a",
            "f_micro_used": "f_micro_m2a",
            "ar_meso_used": "ar_meso_m2a",
        }
    )
    m2a_cols = m2a_oracle_df[
        ["ct_sample_id", "ar_micro_fit", "vp_m2_km_s", "vp_abs_rel_error_m2_pct"]
    ].rename(
        columns={
            "ar_micro_fit": "ar_micro_m2a_oracle",
            "vp_m2_km_s": "vp_m2a_oracle_km_s",
            "vp_abs_rel_error_m2_pct": "vp_abs_err_m2a_oracle_pct",
        }
    )
    m2b_oracle_cols = m2b_oracle_df[
        ["ct_sample_id", "ar_micro_fit", "vp_m2_km_s", "vp_abs_rel_error_m2_pct"]
    ].rename(
        columns={
            "ar_micro_fit": "ar_micro_m2b_oracle",
            "vp_m2_km_s": "vp_m2b_oracle_km_s",
            "vp_abs_rel_error_m2_pct": "vp_abs_err_m2b_oracle_pct",
        }
    )
    m2b_fwd_cols = m2b_forward_df[
        [
            "ct_sample_id",
            "ar_micro_hfu_median",
            "vp_m2_km_s",
            "vp_abs_rel_error_m2_pct",
        ]
    ].rename(
        columns={
            "ar_micro_hfu_median": "ar_micro_m2b_forward",
            "vp_m2_km_s": "vp_m2b_forward_km_s",
            "vp_abs_rel_error_m2_pct": "vp_abs_err_m2b_forward_pct",
        }
    )
    return (
        base.merge(m2a_cols, on="ct_sample_id")
        .merge(m2b_oracle_cols, on="ct_sample_id")
        .merge(m2b_fwd_cols, on="ct_sample_id")
    )


def run_experiment(robust_exclude: bool = False) -> dict:
    """Full A/B pipeline."""
    ensure_dirs()
    exclude = list(ROBUST_EXCLUDE) if robust_exclude else []

    ct_df = load_ct_samples()
    lab_val = load_lab_validation()
    m1_df = load_m1_plug_table()
    m1_loo_df = load_m1_loo_table()
    hfu_calib = load_hfu_calib()
    scales = load_hfu_matrix_scales(hfu_calib)

    if exclude:
        m1_df = m1_df[~m1_df["ct_sample_id"].isin(exclude)].copy()
        m1_loo_df = m1_loo_df[~m1_loo_df["ct_sample_id"].isin(exclude)].copy()

    records = build_multiscale_plug_records(ct_df, lab_val, exclude_samples=exclude)
    if not records:
        raise RuntimeError("No multiscale plug records built")

    m2a_oracle_df = run_m2a_comparison(records, m1_df, scales)
    m2b_oracle_df = run_m2b_comparison(records, m1_df, scales)
    m2b_forward_df = run_m2b_forward_comparison(records, m1_df, scales)
    m2b_forward_loo_df = run_m2b_loo_forward(records, m1_loo_df, scales)
    m2b_oracle_loo_df = run_m2b_loo_oracle(records, m1_loo_df, scales)

    summary_df = build_summary_metrics(
        m1_df,
        m2a_oracle_df,
        m2b_oracle_df,
        m2b_forward_df,
        m1_loo_df,
        m2b_forward_loo_df,
    )
    combined_df = build_combined_plug_table(m2a_oracle_df, m2b_oracle_df, m2b_forward_df)

    combined_df.to_csv(TABLES_DIR / "plug_comparison.csv", index=False, float_format="%.6f")
    summary_df.to_csv(TABLES_DIR / "summary_metrics.csv", index=False, float_format="%.6f")
    m2b_forward_loo_df.to_csv(
        TABLES_DIR / "loo_m2b_forward_comparison.csv", index=False, float_format="%.6f"
    )
    if len(m2b_oracle_loo_df) > 0:
        m2b_oracle_loo_df.to_csv(
            TABLES_DIR / "loo_m2b_oracle_diagnostic.csv",
            index=False,
            float_format="%.6f",
        )

    plot_vp_crossplot(
        m1_df, m2b_forward_df, FIGURES_DIR / "vp_crossplot_m1_vs_m2b_forward.png"
    )
    plot_abs_error_by_plug_three_models(
        m2a_oracle_df,
        m2b_forward_df,
        FIGURES_DIR / "abs_error_m1_m2a_oracle_m2b_forward.png",
    )

    def _mape(model: str, scope: str) -> float:
        row = summary_df[(summary_df["model"] == model) & (summary_df["scope"] == scope)]
        if len(row) == 0:
            return float("nan")
        return float(row.iloc[0]["mape_vp_pct"])

    mape_m1 = _mape("M1_monoscale_calibrated", "in_sample")
    mape_m2a_oracle = _mape("M2a_oracle_per_plug", "in_sample")
    mape_m2b_forward = _mape("M2b_forward_hfu_median", "in_sample")
    mape_m1_loo = _mape("M1_monoscale_calibrated", "loo")
    mape_m2b_forward_loo = _mape("M2b_forward_hfu_median", "loo")

    recommendation, recommendation_rationale = decide_recommendation(
        mape_m1_loo,
        mape_m2b_forward_loo,
        mape_m2b_forward,
        mape_m1,
    )

    metrics = {
        "n_plugs": len(records),
        "robust_exclude": robust_exclude,
        "excluded_samples": exclude,
        "gain_threshold_mape_pp": GAIN_THRESHOLD_MAPE_PP,
        "m1_in_sample": global_validation_metrics(m1_df, "vp_dem_calib_km_s"),
        "m2a_oracle_in_sample": global_validation_metrics(
            m2a_oracle_df.assign(vp_pred=m2a_oracle_df["vp_m2_km_s"]), "vp_pred"
        ),
        "m2b_forward_in_sample": global_validation_metrics(
            m2b_forward_df.assign(vp_pred=m2b_forward_df["vp_m2_km_s"]), "vp_pred"
        ),
        "m1_loo": global_validation_metrics(
            m1_loo_df[m1_loo_df["status"] == "ok"], "vp_dem_loo_km_s"
        ),
        "m2b_forward_loo": global_validation_metrics(
            m2b_forward_loo_df[m2b_forward_loo_df["status"] == "ok"],
            "vp_m2b_forward_loo_km_s",
        ),
        "delta_mape_in_sample_m2b_forward_vs_m1_pp": mape_m1 - mape_m2b_forward,
        "delta_mape_loo_m2b_forward_vs_m1_pp": mape_m1_loo - mape_m2b_forward_loo,
        "m2a_oracle_note": (
            "M2a/M2b oracle invert AR_micro to lab Vp; diagnostic ceiling only"
        ),
        "primary_metric": "m2b_forward in-sample and LOO vs M1",
        "recommendation": recommendation,
        "recommendation_rationale": recommendation_rationale,
        "generated_utc": utc_now_iso(),
    }
    (OUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = [
        "Well 861 -- Etapa 2f multiscale A/B experiment",
        "Generated: {}".format(metrics["generated_utc"]),
        "Planning: methods_comparison/planning/etapa2f_dem_multiscale_ab_poco861.md",
        "Plugs: {}".format(len(records)),
        "Primary metric: M2b forward LOO vs M1 LOO",
        "Recommendation: {} ({})".format(
            recommendation, recommendation_rationale
        ),
        "MAPE M1 in-sample: {:.2f}%".format(mape_m1),
        "MAPE M2b forward in-sample: {:.2f}%".format(mape_m2b_forward),
        "MAPE M1 LOO: {:.2f}%".format(mape_m1_loo),
        "MAPE M2b forward LOO: {:.2f}%".format(mape_m2b_forward_loo),
        "MAPE M2a oracle in-sample (diagnostic): {:.2f}%".format(mape_m2a_oracle),
        "",
        "Tables:",
        "  tables/plug_comparison.csv",
        "  tables/summary_metrics.csv",
        "  tables/loo_m2b_forward_comparison.csv",
        "Figures:",
        "  figures/vp_crossplot_m1_vs_m2b_forward.png",
        "  figures/abs_error_m1_m2a_oracle_m2b_forward.png",
    ]
    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Well 861 DEM multiscale A/B vs monoscale calibrated (Etapa 2f)."
    )
    parser.add_argument(
        "--robust-exclude",
        action="store_true",
        help="Exclude orientation outlier F2911V from the experiment.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    metrics = run_experiment(robust_exclude=args.robust_exclude)
    print("Multiscale A/B complete.")
    print(
        "Recommendation: {} -- {}".format(
            metrics["recommendation"],
            metrics["recommendation_rationale"],
        )
    )
    print(
        "MAPE M1={:.2f}%  M2b forward={:.2f}%  M2b forward LOO={:.2f}%".format(
            metrics["m1_in_sample"]["mape_vp_pct"],
            metrics["m2b_forward_in_sample"]["mape_vp_pct"],
            metrics["m2b_forward_loo"]["mape_vp_pct"],
        )
    )
    print("Output: {}".format(OUT_ROOT))


if __name__ == "__main__":
    main()
