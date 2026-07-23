#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential multiscale DEM for Well 861 plug validation (M2a / M2b).

Forward-only Berryman DEM chain with meso-macro + micro inclusions.
AR_micro inverted by 1D grid search (no scipy regression).

Planning: methods_comparison/planning/etapa2f_dem_multiscale_ab_poco861.md
ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from dem_sc_861_core import (
    berryman_dem,
    dry_density,
    matrix_from_solids,
    velocities_from_moduli,
)

AR_MICRO_MIN: float = 0.01
AR_MICRO_MAX: float = 1.0
AR_MICRO_STEP: float = 0.001
FRACTION_SUM_TOL: float = 1.0e-3
GAIN_THRESHOLD_MAPE_PP: float = 2.0


@dataclass(frozen=True)
class MultiscalePlugRecord:
    """One CT plug with pore-scale fractions for multiscale DEM."""

    ct_sample_id: str
    hfu: int
    phi_lab: float
    f_meso: float
    f_micro: float
    ar_meso: float
    solid1_pct: float
    solid2_pct: float
    vp_lab_z_km_s: float
    vs_lab_z_km_s: float
    vpvs_lab_z: float
    alpha_ct: float


@dataclass(frozen=True)
class MultiscaleForwardResult:
    """Dry-rock multiscale DEM prediction."""

    vp_dem_km_s: float
    vs_dem_km_s: float
    vpvs_dem: float
    ar_micro: float
    dem_k_gpa: float
    dem_g_gpa: float
    matrix_k_gpa: float
    matrix_g_gpa: float
    rel_error_pct: float


@dataclass(frozen=True)
class HfuMultiscaleMedians:
    """HFU-level CT medians for M2b."""

    hfu: int
    n_plugs: int
    f_meso_median: float
    f_micro_median: float
    ar_meso_median: float


def validate_pore_fractions(f_meso: float, f_micro: float, sample_id: str) -> None:
    """Ensure meso + micro fractions sum to unity within tolerance."""
    total = float(f_meso) + float(f_micro)
    if abs(total - 1.0) > FRACTION_SUM_TOL:
        raise ValueError(
            "Pore fractions must sum to 1 for {}: f_meso={} f_micro={} sum={}".format(
                sample_id, f_meso, f_micro, total
            )
        )


def build_multiscale_plug_records(
    ct_df: pd.DataFrame,
    lab_val_df: pd.DataFrame,
    exclude_samples: Optional[Sequence[str]] = None,
) -> List[MultiscalePlugRecord]:
    """Merge CT pore fractions with lab validation table."""
    exclude = set(exclude_samples or ())
    lab_ok = lab_val_df[~lab_val_df["ct_sample_id"].isin(exclude)].copy()
    ct = ct_df.rename(columns={"sample_id": "ct_sample_id"})
    merged = lab_ok.merge(
        ct[
            [
                "ct_sample_id",
                "Phi_lab (pu)",
                "ct_ar_mean",
                "phi_meso_macropores_vv",
                "phi_micropores_vv",
                "ar_meso_macropores",
                "corrected_solid1_pct",
                "corrected_solid2_pct",
            ]
        ],
        on="ct_sample_id",
        how="inner",
    )
    records: List[MultiscalePlugRecord] = []
    for _, row in merged.iterrows():
        sid = str(row["ct_sample_id"])
        f_meso = float(row["phi_meso_macropores_vv"])
        f_micro = float(row["phi_micropores_vv"])
        validate_pore_fractions(f_meso, f_micro, sid)
        records.append(
            MultiscalePlugRecord(
                ct_sample_id=sid,
                hfu=int(row["HFU"]),
                phi_lab=float(row["Phi_lab (pu)"]),
                f_meso=f_meso,
                f_micro=f_micro,
                ar_meso=float(row["ar_meso_macropores"]),
                solid1_pct=float(row["corrected_solid1_pct"]),
                solid2_pct=float(row["corrected_solid2_pct"]),
                vp_lab_z_km_s=float(row["vp_lab_z_km_s"]),
                vs_lab_z_km_s=float(row["vs_lab_z_km_s"]),
                vpvs_lab_z=float(row["vpvs_lab_z"]),
                alpha_ct=float(row["ct_ar_mean"]),
            )
        )
    return records


def compute_hfu_multiscale_medians(
    records: Sequence[MultiscalePlugRecord],
) -> Dict[int, HfuMultiscaleMedians]:
    """Median CT pore fractions and meso AR per HFU."""
    by_hfu: Dict[int, List[MultiscalePlugRecord]] = {}
    for rec in records:
        by_hfu.setdefault(rec.hfu, []).append(rec)
    out: Dict[int, HfuMultiscaleMedians] = {}
    for hfu in sorted(by_hfu.keys()):
        group = by_hfu[hfu]
        out[hfu] = HfuMultiscaleMedians(
            hfu=hfu,
            n_plugs=len(group),
            f_meso_median=float(np.median([r.f_meso for r in group])),
            f_micro_median=float(np.median([r.f_micro for r in group])),
            ar_meso_median=float(np.median([r.ar_meso for r in group])),
        )
    return out


def berryman_dem_sequential(
    km_gpa: float,
    gm_gpa: float,
    phi_total: float,
    inclusions: Sequence[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Sequential Berryman DEM for dry pores.

    Each inclusion is (f_within_pore, aspect_ratio). Porosity increment is
    f_within_pore * phi_total. Inclusions are applied in decreasing phi order.
    """
    phi_use = float(phi_total)
    if phi_use <= 0.0:
        return float(km_gpa), float(gm_gpa)

    steps: List[Tuple[float, float]] = []
    for f_pore, alpha in inclusions:
        phi_inc = float(f_pore) * phi_use
        if phi_inc > 0.0:
            steps.append((phi_inc, float(alpha)))
    if not steps:
        return float(km_gpa), float(gm_gpa)

    steps.sort(key=lambda item: item[0], reverse=True)
    k_cur = float(km_gpa)
    g_cur = float(gm_gpa)
    for phi_inc, alpha in steps:
        dem = berryman_dem(k_cur, g_cur, 0.0, 0.0, alpha, phi_inc)
        k_cur = dem.k_gpa
        g_cur = dem.g_gpa
    return k_cur, g_cur


def vp_multiscale_forward(
    phi_lab: float,
    f_meso: float,
    f_micro: float,
    ar_meso: float,
    ar_micro: float,
    solid1_pct: float,
    solid2_pct: float,
    matrix_k_scale: float = 1.0,
    matrix_g_scale: float = 1.0,
) -> MultiscaleForwardResult:
    """Forward multiscale DEM to dry Vp for one plug configuration."""
    validate_pore_fractions(f_meso, f_micro, "forward")
    matrix = matrix_from_solids(solid1_pct, solid2_pct)
    km = matrix.k_gpa * float(matrix_k_scale)
    gm = matrix.g_gpa * float(matrix_g_scale)
    k_dem, g_dem = berryman_dem_sequential(
        km,
        gm,
        phi_lab,
        ((f_meso, ar_meso), (f_micro, ar_micro)),
    )
    rho = dry_density(phi_lab, matrix.rho_gcc)
    vel = velocities_from_moduli(k_dem, g_dem, rho)
    return MultiscaleForwardResult(
        vp_dem_km_s=vel.vp_km_s,
        vs_dem_km_s=vel.vs_km_s,
        vpvs_dem=vel.vp_vs,
        ar_micro=float(ar_micro),
        dem_k_gpa=k_dem,
        dem_g_gpa=g_dem,
        matrix_k_gpa=km,
        matrix_g_gpa=gm,
        rel_error_pct=float("nan"),
    )


def invert_ar_micro_grid(
    vp_target_km_s: float,
    phi_lab: float,
    f_meso: float,
    f_micro: float,
    ar_meso: float,
    solid1_pct: float,
    solid2_pct: float,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> MultiscaleForwardResult:
    """
    1D grid search for AR_micro minimizing |Vp_pred - Vp_target| / Vp_target.

    Matches the reference Equinor scan over aspect ratio (no regression).
    """
    if vp_target_km_s <= 0.0:
        raise ValueError("vp_target_km_s must be positive")

    best_ar = AR_MICRO_MIN
    best_err = float("inf")
    best_fwd: Optional[MultiscaleForwardResult] = None

    ar_vals = np.arange(AR_MICRO_MIN, AR_MICRO_MAX + AR_MICRO_STEP * 0.5, AR_MICRO_STEP)
    for ar_micro in ar_vals:
        fwd = vp_multiscale_forward(
            phi_lab=phi_lab,
            f_meso=f_meso,
            f_micro=f_micro,
            ar_meso=ar_meso,
            ar_micro=float(ar_micro),
            solid1_pct=solid1_pct,
            solid2_pct=solid2_pct,
            matrix_k_scale=matrix_k_scale,
            matrix_g_scale=matrix_g_scale,
        )
        err = abs(fwd.vp_dem_km_s - vp_target_km_s) / vp_target_km_s
        if err < best_err:
            best_err = err
            best_ar = float(ar_micro)
            best_fwd = fwd

    if best_fwd is None:
        raise RuntimeError("AR_micro grid search failed")

    return MultiscaleForwardResult(
        vp_dem_km_s=best_fwd.vp_dem_km_s,
        vs_dem_km_s=best_fwd.vs_dem_km_s,
        vpvs_dem=best_fwd.vpvs_dem,
        ar_micro=best_ar,
        dem_k_gpa=best_fwd.dem_k_gpa,
        dem_g_gpa=best_fwd.dem_g_gpa,
        matrix_k_gpa=best_fwd.matrix_k_gpa,
        matrix_g_gpa=best_fwd.matrix_g_gpa,
        rel_error_pct=100.0 * best_err,
    )


def fit_ar_micro_oracle(
    record: MultiscalePlugRecord,
    f_meso: float,
    f_micro: float,
    ar_meso: float,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> float:
    """Oracle AR_micro fit to lab Vp (diagnostic only, not profile prediction)."""
    result = invert_ar_micro_grid(
        vp_target_km_s=record.vp_lab_z_km_s,
        phi_lab=record.phi_lab,
        f_meso=f_meso,
        f_micro=f_micro,
        ar_meso=ar_meso,
        solid1_pct=record.solid1_pct,
        solid2_pct=record.solid2_pct,
        matrix_k_scale=matrix_k_scale,
        matrix_g_scale=matrix_g_scale,
    )
    return result.ar_micro


def predict_multiscale_forward(
    record: MultiscalePlugRecord,
    f_meso: float,
    f_micro: float,
    ar_meso: float,
    ar_micro: float,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> MultiscaleForwardResult:
    """Forward multiscale DEM with fixed AR_micro (no inversion to lab Vp)."""
    fwd = vp_multiscale_forward(
        phi_lab=record.phi_lab,
        f_meso=f_meso,
        f_micro=f_micro,
        ar_meso=ar_meso,
        ar_micro=ar_micro,
        solid1_pct=record.solid1_pct,
        solid2_pct=record.solid2_pct,
        matrix_k_scale=matrix_k_scale,
        matrix_g_scale=matrix_g_scale,
    )
    vp_lab = record.vp_lab_z_km_s
    rel = 100.0 * abs(fwd.vp_dem_km_s - vp_lab) / vp_lab if vp_lab > 0 else float("nan")
    return MultiscaleForwardResult(
        vp_dem_km_s=fwd.vp_dem_km_s,
        vs_dem_km_s=fwd.vs_dem_km_s,
        vpvs_dem=fwd.vpvs_dem,
        ar_micro=float(ar_micro),
        dem_k_gpa=fwd.dem_k_gpa,
        dem_g_gpa=fwd.dem_g_gpa,
        matrix_k_gpa=fwd.matrix_k_gpa,
        matrix_g_gpa=fwd.matrix_g_gpa,
        rel_error_pct=rel,
    )


def compute_hfu_ar_micro_oracle_median(
    records: Sequence[MultiscalePlugRecord],
    scales: Dict[int, Tuple[float, float]],
    hfu_medians: Dict[int, HfuMultiscaleMedians],
) -> Dict[int, float]:
    """Per-HFU median of oracle AR_micro fits on the given plug subset."""
    by_hfu: Dict[int, List[float]] = {}
    for rec in records:
        med = hfu_medians[rec.hfu]
        k_scale, g_scale = scales[rec.hfu]
        ar_micro = fit_ar_micro_oracle(
            rec,
            med.f_meso_median,
            med.f_micro_median,
            med.ar_meso_median,
            k_scale,
            g_scale,
        )
        by_hfu.setdefault(rec.hfu, []).append(ar_micro)
    return {hfu: float(np.median(vals)) for hfu, vals in by_hfu.items()}


def predict_multiscale_m2b_forward(
    record: MultiscalePlugRecord,
    hfu_medians: HfuMultiscaleMedians,
    ar_micro_hfu: float,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> MultiscaleForwardResult:
    """M2b forward: HFU median inputs + fixed AR_micro (no plug-level inversion)."""
    return predict_multiscale_forward(
        record,
        hfu_medians.f_meso_median,
        hfu_medians.f_micro_median,
        hfu_medians.ar_meso_median,
        ar_micro_hfu,
        matrix_k_scale,
        matrix_g_scale,
    )


def predict_multiscale_m2a(
    record: MultiscalePlugRecord,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> MultiscaleForwardResult:
    """M2a: plug-local CT fractions + inverted AR_micro."""
    return invert_ar_micro_grid(
        vp_target_km_s=record.vp_lab_z_km_s,
        phi_lab=record.phi_lab,
        f_meso=record.f_meso,
        f_micro=record.f_micro,
        ar_meso=record.ar_meso,
        solid1_pct=record.solid1_pct,
        solid2_pct=record.solid2_pct,
        matrix_k_scale=matrix_k_scale,
        matrix_g_scale=matrix_g_scale,
    )


def predict_multiscale_m2b(
    record: MultiscalePlugRecord,
    hfu_medians: HfuMultiscaleMedians,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> MultiscaleForwardResult:
    """M2b: HFU median fractions + inverted AR_micro."""
    return invert_ar_micro_grid(
        vp_target_km_s=record.vp_lab_z_km_s,
        phi_lab=record.phi_lab,
        f_meso=hfu_medians.f_meso_median,
        f_micro=hfu_medians.f_micro_median,
        ar_meso=hfu_medians.ar_meso_median,
        solid1_pct=record.solid1_pct,
        solid2_pct=record.solid2_pct,
        matrix_k_scale=matrix_k_scale,
        matrix_g_scale=matrix_g_scale,
    )


def load_hfu_matrix_scales(hfu_calib_csv: pd.DataFrame) -> Dict[int, Tuple[float, float]]:
    """Map HFU -> (matrix_k_scale, matrix_g_scale) from M1 calibration."""
    scales: Dict[int, Tuple[float, float]] = {}
    for _, row in hfu_calib_csv.iterrows():
        hfu = int(row["HFU"])
        scales[hfu] = (float(row["matrix_k_scale"]), float(row["matrix_g_scale"]))
    return scales


def multiscale_plug_row(
    record: MultiscalePlugRecord,
    model: str,
    fwd: MultiscaleForwardResult,
    f_meso_used: float,
    f_micro_used: float,
    ar_meso_used: float,
    vp_m1_km_s: float,
    matrix_k_scale: float,
    matrix_g_scale: float,
) -> dict:
    """One row for plug comparison table."""
    vp_lab = record.vp_lab_z_km_s
    err_m2 = 100.0 * (fwd.vp_dem_km_s - vp_lab) / vp_lab
    err_m1 = 100.0 * (vp_m1_km_s - vp_lab) / vp_lab
    return {
        "ct_sample_id": record.ct_sample_id,
        "HFU": record.hfu,
        "model": model,
        "phi_lab_pu": record.phi_lab,
        "f_meso_used": f_meso_used,
        "f_micro_used": f_micro_used,
        "ar_meso_used": ar_meso_used,
        "ar_micro_fit": fwd.ar_micro,
        "matrix_k_scale": matrix_k_scale,
        "matrix_g_scale": matrix_g_scale,
        "vp_lab_z_km_s": vp_lab,
        "vp_m1_calib_km_s": vp_m1_km_s,
        "vp_m2_km_s": fwd.vp_dem_km_s,
        "vp_rel_error_m1_pct": err_m1,
        "vp_rel_error_m2_pct": err_m2,
        "vp_abs_rel_error_m1_pct": abs(err_m1),
        "vp_abs_rel_error_m2_pct": abs(err_m2),
        "delta_abs_error_pp": abs(err_m2) - abs(err_m1),
    }


def decide_recommendation(
    mape_m1_loo: float,
    mape_m2b_forward_loo: float,
    mape_m2b_forward_in_sample: float,
    mape_m1_in_sample: float,
    threshold_pp: float = GAIN_THRESHOLD_MAPE_PP,
) -> Tuple[str, str]:
    """
    Apply pre-registered decision rule.

    Primary for profile adoption: in-sample M2b forward (HFU constants, no plug fit).
    LOO must corroborate with gain above threshold on both scopes to investigate.
    """
    delta_loo = mape_m1_loo - mape_m2b_forward_loo
    delta_in = mape_m1_in_sample - mape_m2b_forward_in_sample
    if delta_in >= threshold_pp and delta_loo >= threshold_pp:
        return (
            "investigate_multiscale",
            "Both in-sample and LOO MAPE gain >= {:.1f} p.p.".format(threshold_pp),
        )
    if delta_loo >= threshold_pp and delta_in < threshold_pp:
        return (
            "keep_monoscale",
            "LOO gain {:.2f} p.p. without in-sample gain; not enough for profile.".format(
                delta_loo
            ),
        )
    return (
        "keep_monoscale",
        "M2b forward does not beat M1 by >= {:.1f} p.p. on primary metrics.".format(
            threshold_pp
        ),
    )
