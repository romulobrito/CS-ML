#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2g -- Phases B-E: DEM multiscale with NMR fractions (Well 861).

B) Merge CMR log with 87-row Gassmann profile
C) Validate NMR fractions vs CT fractions (10 plugs)
D) Forward DEM multiscale + Gassmann on profile rows with CMR
E) Compare M1 (monoscale Gassmann) vs M3 (multiscale NMR Gassmann)

No regression: all parameters inherited from Etapa 2e lab calibration.
AR_micro per HFU from Etapa 2f oracle medians.

Planning: methods_comparison/planning/etapa2g_dem_multiscale_nmr_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dem_sc_861_core import (
    berryman_dem,
    gassmann_bulk_saturation,
    matrix_from_solids,
    saturated_density,
    velocities_from_moduli,
)
from dem_sc_861_multiscale import (
    GAIN_THRESHOLD_MAPE_PP,
    berryman_dem_sequential,
)
from ml_861_data import (
    DEFAULT_CT,
    DEFAULT_ENRICHED,
    DEPTH_COL,
    DLIS_GASSMANN_ROOT,
    DLIS_PROCESSED_ROOT,
    DEM_SC_MULTISCALE_AB_ROOT,
    load_ct_samples,
    load_logs_enriched,
)

ROOT = Path(__file__).resolve().parents[2]

CMR_CSV = DLIS_PROCESSED_ROOT / "tables" / "cmr_log_861.csv"
GASSMANN_VALIDATION_CSV = DLIS_GASSMANN_ROOT / "tables" / "dem_vs_sonic_validation.csv"
HFU_LAB_CALIB_CSV = (
    ROOT / "methods_comparison" / "data" / "processed"
    / "dem_sc_runs" / "hfu_calibration" / "hfu_lab_calibrated.csv"
)
MULTISCALE_AB_PLUG_CSV = DEM_SC_MULTISCALE_AB_ROOT / "tables" / "plug_comparison.csv"

PVT_JSON = ROOT / "methods_comparison" / "data" / "pvt_861_defaults.json"

OUT_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed"
    / "dem_sc_runs" / "multiscale_nmr"
)
OUT_TABLES = OUT_ROOT / "tables"
OUT_FIGURES = OUT_ROOT / "figures"

MERGE_TOL_M: float = 0.20
FRACTION_SUM_TOL: float = 0.30


def utc_now_iso() -> str:
    """UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (OUT_ROOT, OUT_TABLES, OUT_FIGURES):
        d.mkdir(parents=True, exist_ok=True)


def load_pvt() -> Dict[str, float]:
    """Load PVT fluid properties for Gassmann."""
    data = json.loads(PVT_JSON.read_text(encoding="utf-8"))
    return {
        "kf_gpa": float(data["kf_gpa"]),
        "rho_fluid_gcc": float(data["rho_fluid_gcc"]),
        "sw": float(data.get("sw", 1.0)),
    }


def load_hfu_params() -> pd.DataFrame:
    """Load lab-calibrated HFU params (alpha, matrix_k/g_gpa, rho)."""
    return pd.read_csv(HFU_LAB_CALIB_CSV)


def load_ar_micro_per_hfu() -> Dict[int, float]:
    """
    Load AR_micro per HFU from the Etapa 2f oracle medians.

    Uses the M2b forward AR_micro column from plug_comparison.csv,
    computing median per HFU.
    """
    df = pd.read_csv(MULTISCALE_AB_PLUG_CSV)
    result: Dict[int, float] = {}
    for hfu, group in df.groupby("HFU"):
        result[int(hfu)] = float(group["ar_micro_m2b_forward"].median())
    return result


def load_ar_meso_per_hfu() -> Dict[int, float]:
    """Load AR_meso per HFU from plug_comparison (CT median)."""
    df = pd.read_csv(MULTISCALE_AB_PLUG_CSV)
    result: Dict[int, float] = {}
    for hfu, group in df.groupby("HFU"):
        result[int(hfu)] = float(group["ar_meso_m2a"].median())
    return result


# ---- Phase B: Merge CMR with profile ----

def phase_b_merge(
    gassmann_df: pd.DataFrame,
    cmr_df: pd.DataFrame,
    merge_tol_m: float = MERGE_TOL_M,
) -> pd.DataFrame:
    """
    Merge CMR curves onto the Gassmann 87-row profile.

    Computes NMR pore fractions f_macro_nmr and f_micro_nmr.
    """
    gass = gassmann_df.copy().sort_values("Depth(m)")
    cmr = cmr_df[["depth_m", "cmrp_3ms", "cmff", "bfv"]].copy().sort_values("depth_m")

    merged = pd.merge_asof(
        gass,
        cmr,
        left_on="Depth(m)",
        right_on="depth_m",
        direction="nearest",
        tolerance=merge_tol_m,
    )

    has_cmr = merged["cmrp_3ms"].notna() & (merged["cmrp_3ms"] > 0.005)
    merged["has_cmr"] = has_cmr

    merged["f_macro_nmr"] = np.nan
    merged["f_micro_nmr"] = np.nan

    if has_cmr.any():
        total = merged.loc[has_cmr, "cmff"] + merged.loc[has_cmr, "bfv"]
        merged.loc[has_cmr, "f_macro_nmr"] = merged.loc[has_cmr, "cmff"] / total
        merged.loc[has_cmr, "f_micro_nmr"] = merged.loc[has_cmr, "bfv"] / total

    n_matched = int(has_cmr.sum())
    n_total = len(merged)
    print("  Phase B: {}/{} rows matched CMR".format(n_matched, n_total))

    frac_sum = merged.loc[has_cmr, "f_macro_nmr"] + merged.loc[has_cmr, "f_micro_nmr"]
    bad_sum = (frac_sum - 1.0).abs() > FRACTION_SUM_TOL
    if bad_sum.any():
        print(
            "  WARNING: {} rows with |f_macro + f_micro - 1| > {}".format(
                int(bad_sum.sum()), FRACTION_SUM_TOL
            )
        )
    return merged


# ---- Phase C: Validate NMR vs CT fractions (10 plugs) ----

def phase_c_validate_ct(
    profile_merged: pd.DataFrame,
    ct_df: pd.DataFrame,
    merge_tol_m: float = 0.5,
) -> pd.DataFrame:
    """
    Compare NMR fractions with CT fractions at the 10 plug depths.

    Returns a validation table with both CT and NMR fractions per plug.
    """
    ct = ct_df.copy()
    if "ct_depth_m" not in ct.columns:
        raise ValueError("ct_depth_m column missing in CT table")

    prof = profile_merged[
        ["Depth(m)", "f_macro_nmr", "f_micro_nmr", "cmrp_3ms", "cmff", "bfv", "has_cmr"]
    ].copy().sort_values("Depth(m)")

    rows: List[dict] = []
    for _, plug in ct.iterrows():
        sid = str(plug["sample_id"])
        ct_depth = float(plug["ct_depth_m"])
        f_meso_ct = float(plug["phi_meso_macropores_vv"])
        f_micro_ct = float(plug["phi_micropores_vv"])

        idx = (prof["Depth(m)"] - ct_depth).abs().idxmin()
        row_prof = prof.loc[idx]
        depth_delta = float(row_prof["Depth(m)"]) - ct_depth
        has_cmr = bool(row_prof["has_cmr"])

        rec = {
            "ct_sample_id": sid,
            "ct_depth_m": ct_depth,
            "profile_depth_m": float(row_prof["Depth(m)"]),
            "depth_delta_m": depth_delta,
            "has_cmr": has_cmr,
            "f_macro_ct": f_meso_ct,
            "f_micro_ct": f_micro_ct,
            "f_macro_nmr": float(row_prof["f_macro_nmr"]) if has_cmr else float("nan"),
            "f_micro_nmr": float(row_prof["f_micro_nmr"]) if has_cmr else float("nan"),
            "cmrp_3ms": float(row_prof["cmrp_3ms"]) if has_cmr else float("nan"),
        }
        if has_cmr:
            rec["delta_f_macro"] = rec["f_macro_nmr"] - rec["f_macro_ct"]
            rec["delta_f_micro"] = rec["f_micro_nmr"] - rec["f_micro_ct"]
        else:
            rec["delta_f_macro"] = float("nan")
            rec["delta_f_micro"] = float("nan")
        rows.append(rec)

    val_df = pd.DataFrame(rows)
    n_with_cmr = int(val_df["has_cmr"].sum())
    print("  Phase C: {}/10 plugs matched with CMR".format(n_with_cmr))

    matched = val_df[val_df["has_cmr"]].copy()
    if len(matched) > 2:
        r_macro = float(matched["f_macro_ct"].corr(matched["f_macro_nmr"]))
        mae_macro = float((matched["f_macro_ct"] - matched["f_macro_nmr"]).abs().mean())
        r_micro = float(matched["f_micro_ct"].corr(matched["f_micro_nmr"]))
        mae_micro = float((matched["f_micro_ct"] - matched["f_micro_nmr"]).abs().mean())
        print("  Phase C: f_macro -- Pearson r={:.3f} MAE={:.4f}".format(r_macro, mae_macro))
        print("  Phase C: f_micro -- Pearson r={:.3f} MAE={:.4f}".format(r_micro, mae_micro))

    return val_df


# ---- Phase D: DEM multiscale + Gassmann forward ----

def dem_multiscale_gassmann_forward(
    phi_total: float,
    f_macro: float,
    f_micro: float,
    ar_meso: float,
    ar_micro: float,
    km_gpa: float,
    gm_gpa: float,
    rho_matrix_gcc: float,
    kf_gpa: float,
    rho_fluid_gcc: float,
    sw: float = 1.0,
) -> Dict[str, float]:
    """
    Sequential DEM multiscale (dry) + Gassmann saturation.

    Inclusions: (f_macro, ar_meso) and (f_micro, ar_micro), applied
    in decreasing phi_inc order.
    """
    k_dry, g_dry = berryman_dem_sequential(
        km_gpa, gm_gpa, phi_total,
        ((f_macro, ar_meso), (f_micro, ar_micro)),
    )

    k_sat, g_sat = gassmann_bulk_saturation(k_dry, g_dry, km_gpa, kf_gpa, phi_total)
    rho_sat = saturated_density(phi_total, rho_matrix_gcc, rho_fluid_gcc, sw)
    vel = velocities_from_moduli(k_sat, g_sat, rho_sat)

    rho_dry = (1.0 - phi_total) * rho_matrix_gcc
    vel_dry = velocities_from_moduli(k_dry, g_dry, rho_dry)

    return {
        "vp_m3_dry_km_s": vel_dry.vp_km_s,
        "vs_m3_dry_km_s": vel_dry.vs_km_s,
        "vp_m3_km_s": vel.vp_km_s,
        "vs_m3_km_s": vel.vs_km_s,
        "vpvs_m3": vel.vp_vs,
        "k_dry_gpa": k_dry,
        "g_dry_gpa": g_dry,
        "k_sat_gpa": k_sat,
        "g_sat_gpa": g_sat,
        "rho_sat_gcc": rho_sat,
    }


def phase_d_forward(
    profile_merged: pd.DataFrame,
    hfu_params: pd.DataFrame,
    ar_micro_per_hfu: Dict[int, float],
    ar_meso_per_hfu: Dict[int, float],
    pvt: Dict[str, float],
) -> pd.DataFrame:
    """Run DEM multiscale + Gassmann on each profile row with CMR."""
    hfu_lookup: Dict[int, dict] = {}
    for _, row in hfu_params.iterrows():
        hfu = int(row["HFU"])
        hfu_lookup[hfu] = {
            "km_gpa": float(row["matrix_k_gpa"]),
            "gm_gpa": float(row["matrix_g_gpa"]),
            "rho_gcc": float(row["matrix_rho_gcc"]),
            "alpha": float(row["alpha"]),
        }

    rows: List[dict] = []
    n_ok = 0
    n_skip = 0

    for _, prof_row in profile_merged.iterrows():
        depth = float(prof_row["Depth(m)"])
        hfu = int(prof_row["HFU"])
        has_cmr = bool(prof_row["has_cmr"])

        rec: dict = {
            "Depth(m)": depth,
            "HFU": hfu,
            "has_cmr": has_cmr,
        }

        if not has_cmr or hfu not in hfu_lookup:
            rec["status"] = "skip_no_cmr" if not has_cmr else "skip_unknown_hfu"
            n_skip += 1
            rows.append(rec)
            continue

        hp = hfu_lookup[hfu]
        phi_total = float(prof_row.get("phi_input", float("nan")))
        if not np.isfinite(phi_total) or phi_total <= 0.0:
            rec["status"] = "skip_no_phi"
            n_skip += 1
            rows.append(rec)
            continue

        f_macro = float(prof_row["f_macro_nmr"])
        f_micro = float(prof_row["f_micro_nmr"])

        ar_meso = ar_meso_per_hfu.get(hfu, 0.55)
        ar_micro = ar_micro_per_hfu.get(hfu, 0.05)

        try:
            out = dem_multiscale_gassmann_forward(
                phi_total=phi_total,
                f_macro=f_macro,
                f_micro=f_micro,
                ar_meso=ar_meso,
                ar_micro=ar_micro,
                km_gpa=hp["km_gpa"],
                gm_gpa=hp["gm_gpa"],
                rho_matrix_gcc=hp["rho_gcc"],
                kf_gpa=pvt["kf_gpa"],
                rho_fluid_gcc=pvt["rho_fluid_gcc"],
                sw=pvt["sw"],
            )

            vp_m1 = float(prof_row.get("vp_dem_km_s", float("nan")))
            vp_sonic = float(prof_row.get("vp_sonic_km_s", float("nan")))

            rec.update(out)
            rec["phi_input"] = phi_total
            rec["f_macro_nmr"] = f_macro
            rec["f_micro_nmr"] = f_micro
            rec["ar_meso_hfu"] = ar_meso
            rec["ar_micro_hfu"] = ar_micro
            rec["vp_m1_km_s"] = vp_m1
            rec["vp_sonic_km_s"] = vp_sonic

            if np.isfinite(vp_sonic) and vp_sonic > 0:
                rec["vp_m1_rel_error_pct"] = 100.0 * (vp_m1 - vp_sonic) / vp_sonic
                rec["vp_m3_rel_error_pct"] = 100.0 * (out["vp_m3_km_s"] - vp_sonic) / vp_sonic
                rec["vp_m1_abs_rel_error_pct"] = abs(rec["vp_m1_rel_error_pct"])
                rec["vp_m3_abs_rel_error_pct"] = abs(rec["vp_m3_rel_error_pct"])
            rec["status"] = "ok"
            n_ok += 1
        except Exception as exc:
            rec["status"] = "error"
            rec["error"] = str(exc)
            n_skip += 1

        rows.append(rec)

    print("  Phase D: {} ok, {} skipped".format(n_ok, n_skip))
    return pd.DataFrame(rows)


# ---- Phase E: Metrics and plots ----

def phase_e_metrics(
    results: pd.DataFrame,
) -> Dict[str, object]:
    """Compute comparison metrics M1 vs M3 on rows with sonic reference."""
    ok = results[(results["status"] == "ok") & results["vp_sonic_km_s"].notna()].copy()
    ok = ok[ok["vp_sonic_km_s"] > 0].copy()

    n = len(ok)
    if n == 0:
        return {"n_valid": 0, "recommendation": "no_data"}

    def _metrics(pred_col: str, ref_col: str, prefix: str) -> dict:
        pred = ok[pred_col].to_numpy()
        ref = ok[ref_col].to_numpy()
        err = pred - ref
        abs_rel = np.abs(err / ref) * 100.0
        return {
            "{}_mae_km_s".format(prefix): float(np.mean(np.abs(err))),
            "{}_rmse_km_s".format(prefix): float(np.sqrt(np.mean(err ** 2))),
            "{}_mape_pct".format(prefix): float(np.mean(abs_rel)),
            "{}_bias_km_s".format(prefix): float(np.mean(err)),
            "{}_pearson_r".format(prefix): float(np.corrcoef(pred, ref)[0, 1]),
        }

    m1 = _metrics("vp_m1_km_s", "vp_sonic_km_s", "m1")
    m3 = _metrics("vp_m3_km_s", "vp_sonic_km_s", "m3")

    delta_mape = m1["m1_mape_pct"] - m3["m3_mape_pct"]
    if delta_mape >= GAIN_THRESHOLD_MAPE_PP:
        recommendation = "investigate_multiscale_nmr"
        rationale = "M3 MAPE gain {:.2f} p.p. >= threshold {:.1f}".format(
            delta_mape, GAIN_THRESHOLD_MAPE_PP
        )
    else:
        recommendation = "keep_monoscale"
        rationale = "M3 MAPE gain {:.2f} p.p. < threshold {:.1f}".format(
            delta_mape, GAIN_THRESHOLD_MAPE_PP
        )

    return {
        "n_valid": n,
        **m1,
        **m3,
        "delta_mape_m3_vs_m1_pp": delta_mape,
        "gain_threshold_pp": GAIN_THRESHOLD_MAPE_PP,
        "recommendation": recommendation,
        "recommendation_rationale": rationale,
    }


def plot_crossplot_m1_m3(results: pd.DataFrame, out_path: Path) -> None:
    """Crossplot Vp sonic vs Vp predicted (M1 and M3)."""
    ok = results[(results["status"] == "ok") & results["vp_sonic_km_s"].notna()].copy()
    if len(ok) < 3:
        return

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    vmin = 3.5
    vmax = 7.0
    ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, label="1:1")
    ax.scatter(ok["vp_sonic_km_s"], ok["vp_m1_km_s"], s=20, alpha=0.7,
              edgecolors="blue", facecolors="none", label="M1 monoscale")
    ax.scatter(ok["vp_sonic_km_s"], ok["vp_m3_km_s"], s=20, alpha=0.7,
              edgecolors="red", facecolors="none", label="M3 NMR multiscale")
    ax.set_xlabel("Vp sonic (km/s)")
    ax.set_ylabel("Vp predicted (km/s)")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.set_title("M1 vs M3 vs Sonic (Well 861)")
    ax.legend(loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print("  Wrote {}".format(out_path))


def plot_profile_depth(results: pd.DataFrame, out_path: Path) -> None:
    """Depth profile of Vp: sonic, M1, M3."""
    ok = results[(results["status"] == "ok") & results["vp_sonic_km_s"].notna()].copy()
    if len(ok) < 3:
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    ax.plot(ok["vp_sonic_km_s"], ok["Depth(m)"], "k-", linewidth=1.0, label="Sonic")
    ax.plot(ok["vp_m1_km_s"], ok["Depth(m)"], "b--", linewidth=0.8, label="M1 monoscale")
    ax.plot(ok["vp_m3_km_s"], ok["Depth(m)"], "r-.", linewidth=0.8, label="M3 NMR")
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.legend(loc="lower left")
    ax.set_title("Vp profile: Sonic vs M1 vs M3")
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print("  Wrote {}".format(out_path))


def plot_cmr_fractions(results: pd.DataFrame, out_path: Path) -> None:
    """NMR pore fractions vs depth."""
    ok = results[results["status"] == "ok"].copy()
    if len(ok) < 3:
        return

    fig, ax = plt.subplots(1, 1, figsize=(5, 10))
    ax.barh(ok["Depth(m)"], ok["f_macro_nmr"], height=0.15,
            color="steelblue", alpha=0.7, label="f_macro (CMFF)")
    ax.barh(ok["Depth(m)"], ok["f_micro_nmr"], height=0.15,
            left=ok["f_macro_nmr"], color="coral", alpha=0.7, label="f_micro (BFV)")
    ax.set_xlabel("Pore fraction")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.legend(loc="lower left")
    ax.set_title("NMR pore fractions (Well 861)")
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print("  Wrote {}".format(out_path))


def plot_ct_vs_nmr_fractions(val_df: pd.DataFrame, out_path: Path) -> None:
    """Crossplot of CT vs NMR pore fractions (10 plugs)."""
    matched = val_df[val_df["has_cmr"]].copy()
    if len(matched) < 2:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    ax.scatter(matched["f_macro_ct"], matched["f_macro_nmr"], s=40, edgecolors="blue",
              facecolors="none")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("f_macro CT")
    ax.set_ylabel("f_macro NMR")
    ax.set_title("Macro fraction: CT vs NMR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    ax = axes[1]
    ax.scatter(matched["f_micro_ct"], matched["f_micro_nmr"], s=40, edgecolors="red",
              facecolors="none")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("f_micro CT")
    ax.set_ylabel("f_micro NMR")
    ax.set_title("Micro fraction: CT vs NMR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print("  Wrote {}".format(out_path))


def main() -> None:
    """CLI entry point for Etapa 2g multiscale NMR pipeline."""
    parser = argparse.ArgumentParser(
        description="DEM multiscale + Gassmann with NMR fractions (Well 861)."
    )
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test.")
    args = parser.parse_args()

    ensure_dirs()

    print("=== Etapa 2g: DEM multiscale NMR (Well 861) ===")

    # Load inputs
    cmr_df = pd.read_csv(CMR_CSV)
    print("CMR log: {} rows".format(len(cmr_df)))

    gassmann_df = pd.read_csv(GASSMANN_VALIDATION_CSV)
    print("Gassmann validation: {} rows".format(len(gassmann_df)))

    ct_df = load_ct_samples()
    print("CT samples: {} rows".format(len(ct_df)))

    hfu_params = load_hfu_params()
    ar_micro_per_hfu = load_ar_micro_per_hfu()
    ar_meso_per_hfu = load_ar_meso_per_hfu()
    pvt = load_pvt()

    print("AR_micro per HFU: {}".format(ar_micro_per_hfu))
    print("AR_meso per HFU: {}".format(ar_meso_per_hfu))
    print("PVT: Kf={} GPa, rho_f={} g/cc, Sw={}".format(
        pvt["kf_gpa"], pvt["rho_fluid_gcc"], pvt["sw"]
    ))

    if args.smoke:
        gassmann_df = gassmann_df.head(15).copy()

    # Phase B: Merge CMR with profile
    print("\n--- Phase B: Merge CMR with Gassmann profile ---")
    profile_merged = phase_b_merge(gassmann_df, cmr_df)

    # Phase C: Validate NMR vs CT
    print("\n--- Phase C: Validate NMR vs CT fractions ---")
    ct_val = phase_c_validate_ct(profile_merged, ct_df)
    ct_val.to_csv(
        OUT_TABLES / "cmr_vs_ct_validation.csv", index=False, float_format="%.6f"
    )
    plot_ct_vs_nmr_fractions(ct_val, OUT_FIGURES / "crossplot_ct_vs_nmr_fractions.png")

    # Phase D: Forward DEM multiscale + Gassmann
    print("\n--- Phase D: DEM multiscale + Gassmann forward ---")
    results = phase_d_forward(
        profile_merged, hfu_params, ar_micro_per_hfu, ar_meso_per_hfu, pvt
    )
    results.to_csv(
        OUT_TABLES / "profile_m1_vs_m3.csv", index=False, float_format="%.6f"
    )

    # Phase E: Metrics and plots
    print("\n--- Phase E: Metrics M1 vs M3 ---")
    metrics = phase_e_metrics(results)
    metrics["generated_utc"] = utc_now_iso()

    for key in sorted(metrics.keys()):
        val = metrics[key]
        if isinstance(val, float):
            print("  {}: {:.4f}".format(key, val))
        else:
            print("  {}: {}".format(key, val))

    metrics_path = OUT_ROOT / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote {}".format(metrics_path))

    # Summary table
    summary_rows = []
    for prefix, label in [("m1", "M1_monoscale_gassmann"), ("m3", "M3_multiscale_nmr_gassmann")]:
        summary_rows.append({
            "model": label,
            "n_valid": metrics["n_valid"],
            "mae_vp_km_s": metrics["{}_mae_km_s".format(prefix)],
            "rmse_vp_km_s": metrics["{}_rmse_km_s".format(prefix)],
            "mape_vp_pct": metrics["{}_mape_pct".format(prefix)],
            "bias_vp_km_s": metrics["{}_bias_km_s".format(prefix)],
            "pearson_r_vp": metrics["{}_pearson_r".format(prefix)],
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        OUT_TABLES / "summary_metrics.csv", index=False, float_format="%.6f"
    )

    # Plots
    plot_crossplot_m1_m3(results, OUT_FIGURES / "crossplot_vp_m1_vs_m3.png")
    plot_profile_depth(results, OUT_FIGURES / "profile_vp_depth.png")
    plot_cmr_fractions(results, OUT_FIGURES / "cmr_fractions_depth.png")

    # Manifest
    manifest_lines = [
        "Well 861 -- Etapa 2g: DEM multiscale NMR",
        "Generated: {}".format(utc_now_iso()),
        "",
        "Source CMR: {}".format(CMR_CSV),
        "Source Gassmann: {}".format(GASSMANN_VALIDATION_CSV),
        "HFU params: {}".format(HFU_LAB_CALIB_CSV),
        "AR_micro per HFU: {}".format(ar_micro_per_hfu),
        "AR_meso per HFU: {}".format(ar_meso_per_hfu),
        "PVT: Kf={} GPa, rho_f={} g/cc, Sw={}".format(
            pvt["kf_gpa"], pvt["rho_fluid_gcc"], pvt["sw"]
        ),
        "",
        "N valid rows: {}".format(metrics["n_valid"]),
        "M1 MAPE: {:.2f}%".format(metrics["m1_mape_pct"]),
        "M3 MAPE: {:.2f}%".format(metrics["m3_mape_pct"]),
        "Delta MAPE (M1-M3): {:.2f} p.p.".format(metrics["delta_mape_m3_vs_m1_pp"]),
        "Recommendation: {}".format(metrics["recommendation"]),
        "Rationale: {}".format(metrics["recommendation_rationale"]),
    ]
    manifest_path = OUT_ROOT / "MANIFEST.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print("Wrote {}".format(manifest_path))
    print("\nDone.")


if __name__ == "__main__":
    main()
