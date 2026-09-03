#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well 861: fluid Kf sensitivity for Gassmann + hybrid residual (Entrega 2).

Scenarios (same DEM dry frame and Phi_ND; only fluid in Gassmann changes):
  1) kf_adopted_2p2     -- PVT default Kf=2.2 GPa, rho=1.03, Sw=1
  2) kf_well_median     -- median KFluid/RhoFluid from rock861 in study interval
  3) nmr_wood_z         -- Wood mix with SWIRR(z), KBrine, KOil (depth-varying)
  4) nmr_wood_median    -- constant median of Wood Kf/rho from scenario 3

Also regenerates depth-track figures used in Entrega 2 slides:
  fig3_kf_sw_study_interval.png
  fig3_kf_sw_nmr_study_interval.png

Outputs:
  methods_comparison/data/processed/kf_fluid_sensitivity_861/
  methods_comparison/latex/figures/fig3_kf_sw_*.png  (when --write-latex-figures)

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (  # noqa: E402
    DEPTH_COL,
    DLIS_SONIC_CSV,
    RESIDUAL_VP_TARGET,
    build_residual_feature_columns,
    build_xy_from_columns,
    depth_block_splits,
    load_logs_enriched,
)
from run_861_dem_sc_gassmann import (  # noqa: E402
    HFU_LAB_CALIB_CSV,
    PHI_COL,
    PHI_LAB_COL,
    PHI_SONIC_COL,
    PVT_DEFAULTS_JSON,
    load_pvt_config,
)

HFU_COL = "HFU"
from run_861_dem_sc_profile_87 import (  # noqa: E402
    build_hfu_params_from_lab_calib,
    load_hfu_lab_calibrated,
)
from run_861_dlis_dem_validation import (  # noqa: E402
    build_validation_table,
    load_sonic,
    merge_dem_sonic,
)
from run_861_ml_residual import (  # noqa: E402
    evaluate_depth_blocks_oof,
    vp_metrics_vs_sonic,
)
from dem_sc_861_core import run_from_matrix_moduli_saturated  # noqa: E402

ROCK_CSV = ROOT / "data" / "rock861.csv"
LOG_CSV = ROOT / "data" / "log861.csv"
OUT_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed" / "kf_fluid_sensitivity_861"
)
RESULTS_ROOT = (
    ROOT / "methods_comparison" / "results" / "kf_fluid_sensitivity_861"
)
LATEX_FIG_DIR = ROOT / "methods_comparison" / "latex" / "figures"
DEFAULT_MERGE_TOL_M = 0.25
N_BLOCKS = 3
N_ESTIMATORS = 200
RANDOM_STATE = 42

# Rounded values published in poco861_etapa3_entrega2.tex (slides 14/16).
REFERENCE_SLIDE: Dict[str, Dict[str, Dict[str, float]]] = {
    "kf_adopted_2p2": {
        "gassmann": {"mape_pct": 7.28, "bias_km_s": 0.271},
        "hybrid_rf": {"mape_pct": 2.78, "bias_km_s": 0.014},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "kf_well_median": {
        "gassmann": {"mape_pct": 7.26, "bias_km_s": 0.293},
        "hybrid_rf": {"mape_pct": 2.82, "bias_km_s": 0.018},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "nmr_wood_z": {
        "gassmann": {"mape_pct": 7.65, "bias_km_s": 0.26},
        "hybrid_rf": {"mape_pct": 2.97, "bias_km_s": 0.03},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.00},
    },
    "nmr_wood_median": {
        "gassmann": {"mape_pct": 7.65, "bias_km_s": 0.26},
        "hybrid_rf": {"mape_pct": 2.95, "bias_km_s": 0.03},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.00},
    },
}


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clip01(x: np.ndarray) -> np.ndarray:
    """Clip saturations to [0, 1]."""
    return np.clip(x.astype(np.float64), 0.0, 1.0)


def wood_kf(
    sw: np.ndarray,
    k_brine: np.ndarray,
    k_oil: np.ndarray,
) -> np.ndarray:
    """Wood / Reuss fluid bulk modulus mix (GPa)."""
    sw_u = _clip01(sw)
    so = 1.0 - sw_u
    inv = sw_u / k_brine + so / k_oil
    return 1.0 / inv


def wood_rho(
    sw: np.ndarray,
    rho_brine: np.ndarray,
    rho_oil: np.ndarray,
) -> np.ndarray:
    """Linear density mix (g/cc)."""
    sw_u = _clip01(sw)
    so = 1.0 - sw_u
    return sw_u * rho_brine + so * rho_oil


def study_interval_from_logs(logs: pd.DataFrame) -> Tuple[float, float]:
    """Topo/base from the 87-row model table."""
    z = logs[DEPTH_COL].to_numpy(dtype=np.float64)
    return float(np.min(z)), float(np.max(z))


def load_fluid_tables() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load rock861 and log861 CSVs."""
    if not ROCK_CSV.is_file():
        raise FileNotFoundError("Missing {}".format(ROCK_CSV))
    if not LOG_CSV.is_file():
        raise FileNotFoundError("Missing {}".format(LOG_CSV))
    rock = pd.read_csv(ROCK_CSV)
    log = pd.read_csv(LOG_CSV)
    for col in ("MD", "SWIRR", "KFluid", "KBrine", "KOil", "RhoFluid", "RhoBrine", "RhoOil"):
        if col not in rock.columns:
            raise ValueError("rock861 missing column: {}".format(col))
        rock[col] = pd.to_numeric(rock[col], errors="coerce")
    for col in ("MD", "SW"):
        if col not in log.columns:
            raise ValueError("log861 missing column: {}".format(col))
        log[col] = pd.to_numeric(log[col], errors="coerce")
    return rock, log


def attach_fluid_to_profile(
    logs: pd.DataFrame,
    rock: pd.DataFrame,
    log: pd.DataFrame,
) -> pd.DataFrame:
    """Nearest-depth attach of LAS Sw and rock fluid columns onto 87 rows."""
    base = logs[[DEPTH_COL, HFU_COL, PHI_COL, PHI_SONIC_COL]].copy()
    if PHI_LAB_COL in logs.columns:
        base[PHI_LAB_COL] = logs[PHI_LAB_COL]

    rock_use = rock[
        ["MD", "SWIRR", "KFluid", "RhoFluid", "KBrine", "KOil", "RhoBrine", "RhoOil"]
    ].dropna(subset=["MD"]).sort_values("MD")
    log_use = log[["MD", "SW"]].dropna(subset=["MD"]).sort_values("MD")

    left = base.rename(columns={DEPTH_COL: "depth_m"}).sort_values("depth_m")
    merged = pd.merge_asof(
        left,
        rock_use.rename(columns={"MD": "rock_md"}),
        left_on="depth_m",
        right_on="rock_md",
        direction="nearest",
    )
    merged = pd.merge_asof(
        merged.sort_values("depth_m"),
        log_use.rename(columns={"MD": "log_md", "SW": "sw_las"}),
        left_on="depth_m",
        right_on="log_md",
        direction="nearest",
    )
    merged = merged.rename(columns={"depth_m": DEPTH_COL})
    merged["sw_nmr"] = _clip01(merged["SWIRR"].to_numpy(dtype=np.float64))
    merged["so_nmr"] = 1.0 - merged["sw_nmr"]
    merged["kf_wood_gpa"] = wood_kf(
        merged["sw_nmr"].to_numpy(dtype=np.float64),
        merged["KBrine"].to_numpy(dtype=np.float64),
        merged["KOil"].to_numpy(dtype=np.float64),
    )
    merged["rho_wood_gcc"] = wood_rho(
        merged["sw_nmr"].to_numpy(dtype=np.float64),
        merged["RhoBrine"].to_numpy(dtype=np.float64),
        merged["RhoOil"].to_numpy(dtype=np.float64),
    )
    return merged.reset_index(drop=True)


def process_profile_with_fluid(
    profile: pd.DataFrame,
    hfu_params: Dict[int, Any],
    kf: np.ndarray,
    rho_f: np.ndarray,
    sw: np.ndarray,
    fluid_name: str,
) -> pd.DataFrame:
    """DEM dry + Gassmann with per-row fluid properties."""
    rows: List[dict] = []
    for i, row in profile.iterrows():
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
            rows.append(
                {
                    DEPTH_COL: depth,
                    HFU_COL: hfu,
                    "status": "error",
                    "error": "unknown_HFU",
                }
            )
            continue
        hp = hfu_params[hfu]
        try:
            out = run_from_matrix_moduli_saturated(
                phi=phi_nd,
                alpha=hp.alpha,
                km_gpa=hp.matrix_k_gpa,
                gm_gpa=hp.matrix_g_gpa,
                rho_matrix_gcc=hp.matrix_rho_gcc,
                kf_gpa=float(kf[i]),
                rho_fluid_gcc=float(rho_f[i]),
                sw=float(sw[i]),
            )
            rows.append(
                {
                    DEPTH_COL: depth,
                    HFU_COL: hfu,
                    "phi_input": phi_nd,
                    "Phi_Sonic (pu)": phi_sonic,
                    PHI_LAB_COL: phi_lab,
                    "fluid_name": fluid_name,
                    "kf_used_gpa": float(kf[i]),
                    "rho_fluid_used_gcc": float(rho_f[i]),
                    "sw_used": float(sw[i]),
                    **out,
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    DEPTH_COL: depth,
                    HFU_COL: hfu,
                    "status": "error",
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows)


def validate_vs_sonic(
    sat_profile: pd.DataFrame,
    sonic: pd.DataFrame,
    merge_tol_m: float,
) -> pd.DataFrame:
    """Build Gassmann vs sonic validation table."""
    ok = sat_profile[sat_profile["status"] == "ok"].copy()
    merged = merge_dem_sonic(ok, sonic, merge_tol_m)
    return build_validation_table(merged)


def build_residual_from_validation(
    validation: pd.DataFrame,
    logs: pd.DataFrame,
) -> pd.DataFrame:
    """Merge enriched logs with a scenario validation table."""
    ok = validation[validation["has_sonic_vp"] == True].copy()  # noqa: E712
    if ok.empty:
        raise RuntimeError("No sonic-matched rows in validation")
    pick = ok[
        [DEPTH_COL, "vp_dem_km_s", "vp_sonic_km_s", "vp_bias_km_s", "phi_input"]
    ].rename(columns={"vp_dem_km_s": "vp_gassmann_km_s"})
    merged = logs.merge(pick, on=DEPTH_COL, how="inner")
    merged["vp_residual_km_s"] = (
        merged["vp_sonic_km_s"] - merged["vp_gassmann_km_s"]
    )
    merged = merged.sort_values(DEPTH_COL).reset_index(drop=True)
    if len(merged) != 87:
        raise RuntimeError("Expected 87 merged rows, got {}".format(len(merged)))
    return merged


def oof_hybrid_metrics(
    dataset: pd.DataFrame,
    n_blocks: int = N_BLOCKS,
    random_state: int = RANDOM_STATE,
) -> Dict[str, Dict[str, float]]:
    """RF and linear residual OOF hybrid metrics vs sonic."""
    feature_cols = build_residual_feature_columns(dataset)
    bundle = build_xy_from_columns(
        dataset,
        target=RESIDUAL_VP_TARGET,
        feature_columns=feature_cols,
    )
    # Sanity: depth_block_splits expects a DataFrame (not ndarray).
    _ = depth_block_splits(bundle.df, n_blocks=n_blocks)

    def rf_factory() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=random_state,
        )

    def lr_factory() -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        )

    out: Dict[str, Dict[str, float]] = {}
    for name, factory in (("hybrid_rf", rf_factory), ("hybrid_lr", lr_factory)):
        _, oof_resid = evaluate_depth_blocks_oof(factory, bundle, n_blocks=n_blocks)
        vp_hyb = (
            bundle.df["vp_gassmann_km_s"].to_numpy(dtype=np.float64) + oof_resid
        )
        m = vp_metrics_vs_sonic(
            vp_hyb,
            bundle.df["vp_sonic_km_s"].to_numpy(dtype=np.float64),
        )
        out[name] = {
            "mape_pct": float(m.mape_pct),
            "bias_km_s": float(m.bias_km_s),
            "rmse_km_s": float(m.rmse_km_s),
            "n": float(m.n),
        }
    return out


def metrics_dict_from_validation(validation: pd.DataFrame) -> Dict[str, float]:
    """Gassmann MAPE/bias vs sonic."""
    ok = validation[validation["has_sonic_vp"] == True]  # noqa: E712
    m = vp_metrics_vs_sonic(
        ok["vp_dem_km_s"].to_numpy(dtype=np.float64),
        ok["vp_sonic_km_s"].to_numpy(dtype=np.float64),
    )
    return {
        "mape_pct": float(m.mape_pct),
        "bias_km_s": float(m.bias_km_s),
        "rmse_km_s": float(m.rmse_km_s),
        "n": float(m.n),
    }


def run_scenario(
    scenario_id: str,
    profile: pd.DataFrame,
    hfu_params: Dict[int, Any],
    logs: pd.DataFrame,
    sonic: pd.DataFrame,
    kf: np.ndarray,
    rho_f: np.ndarray,
    sw: np.ndarray,
    fluid_name: str,
    merge_tol_m: float,
) -> Dict[str, Any]:
    """One fluid scenario: Gassmann + hybrid RF/LR."""
    sat = process_profile_with_fluid(
        profile, hfu_params, kf=kf, rho_f=rho_f, sw=sw, fluid_name=fluid_name
    )
    validation = validate_vs_sonic(sat, sonic, merge_tol_m)
    gass = metrics_dict_from_validation(validation)
    residual_df = build_residual_from_validation(validation, logs)
    hybrid = oof_hybrid_metrics(residual_df)
    return {
        "scenario_id": scenario_id,
        "fluid_name": fluid_name,
        "kf_median_gpa": float(np.nanmedian(kf)),
        "rho_median_gcc": float(np.nanmedian(rho_f)),
        "sw_median": float(np.nanmedian(sw)),
        "gassmann": gass,
        "hybrid_rf": hybrid["hybrid_rf"],
        "hybrid_lr": hybrid["hybrid_lr"],
        "sat_profile": sat,
        "validation": validation,
    }


def plot_las_interval(
    rock: pd.DataFrame,
    log: pd.DataFrame,
    z0: float,
    z1: float,
    kf_adopted: float,
    kf_well_med: float,
    out_path: Path,
) -> None:
    """Depth tracks: LAS Sw, KFluid, Cf in study interval."""
    m = (rock["MD"] >= z0) & (rock["MD"] <= z1)
    r = rock.loc[m].copy()
    ml = (log["MD"] >= z0) & (log["MD"] <= z1)
    lsub = log.loc[ml].copy()

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 8.0), sharey=True)
    ax0, ax1, ax2 = axes

    ax0.plot(lsub["SW"], lsub["MD"], color="#1f77b4", linewidth=1.2, label="Sw LAS")
    ax0.set_xlabel("Saturation")
    ax0.set_xlim(0.0, 1.05)
    ax0.set_title("Sw (LAS)")
    ax0.invert_yaxis()
    ax0.axhline(z0, color="0.5", linestyle="--", linewidth=0.8)
    ax0.axhline(z1, color="0.5", linestyle="--", linewidth=0.8)
    ax0.grid(True, alpha=0.25)

    ax1.plot(r["KFluid"], r["MD"], color="#d62728", linewidth=1.2, label="Kf poço")
    ax1.axvline(kf_adopted, color="#2ca02c", linestyle="--", linewidth=1.2, label="Kf=2.2")
    ax1.axvline(kf_well_med, color="#ff7f0e", linestyle=":", linewidth=1.4, label="mediana")
    ax1.set_xlabel("Kf (GPa)")
    ax1.set_title("Modulo do fluido")
    ax1.axhline(z0, color="0.5", linestyle="--", linewidth=0.8)
    ax1.axhline(z1, color="0.5", linestyle="--", linewidth=0.8)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, loc="best")

    cf = 1.0 / r["KFluid"].to_numpy(dtype=np.float64)
    ax2.plot(cf, r["MD"], color="#9467bd", linewidth=1.2, label="Cf")
    ax2.axvline(1.0 / kf_adopted, color="#2ca02c", linestyle="--", linewidth=1.2, label="1/2.2")
    ax2.set_xlabel("Cf = 1/Kf (1/GPa)")
    ax2.set_title("Compressibilidade")
    ax2.axhline(z0, color="0.5", linestyle="--", linewidth=0.8)
    ax2.axhline(z1, color="0.5", linestyle="--", linewidth=0.8)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7, loc="best")

    ax0.set_ylabel("Depth (m)")
    fig.suptitle(
        "Poco 861: Sw LAS e Kf no intervalo [{:.1f}, {:.1f}] m".format(z0, z1),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nmr_interval(
    rock: pd.DataFrame,
    z0: float,
    z1: float,
    kf_adopted: float,
    kf_nmr_med: float,
    out_path: Path,
) -> None:
    """Depth tracks: SWIRR/So, Wood Kf, Cf in study interval."""
    m = (rock["MD"] >= z0) & (rock["MD"] <= z1)
    r = rock.loc[m].copy()
    sw = _clip01(r["SWIRR"].to_numpy(dtype=np.float64))
    so = 1.0 - sw
    kf = wood_kf(
        sw,
        r["KBrine"].to_numpy(dtype=np.float64),
        r["KOil"].to_numpy(dtype=np.float64),
    )
    cf = 1.0 / kf
    md = r["MD"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 8.0), sharey=True)
    ax0, ax1, ax2 = axes

    ax0.fill_betweenx(md, 0.0, sw, color="#1f77b4", alpha=0.85, label="Sw NMR")
    ax0.fill_betweenx(md, sw, 1.0, color="#2ca02c", alpha=0.75, label="So=1-Sw")
    ax0.set_xlim(0.0, 1.0)
    ax0.set_xlabel("Saturation")
    ax0.set_title("NMR (SWIRR)")
    ax0.invert_yaxis()
    ax0.axhline(z0, color="0.5", linestyle="--", linewidth=0.8)
    ax0.axhline(z1, color="0.5", linestyle="--", linewidth=0.8)
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=7, loc="best")

    ax1.plot(kf, md, color="#d62728", linewidth=1.2, label="Kf Wood+NMR")
    ax1.axvline(kf_adopted, color="#2ca02c", linestyle="--", linewidth=1.2, label="Kf=2.2")
    ax1.axvline(kf_nmr_med, color="#ff7f0e", linestyle=":", linewidth=1.4, label="mediana NMR")
    ax1.set_xlabel("Kf (GPa)")
    ax1.set_title("Modulo do fluido")
    ax1.axhline(z0, color="0.5", linestyle="--", linewidth=0.8)
    ax1.axhline(z1, color="0.5", linestyle="--", linewidth=0.8)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, loc="best")

    ax2.plot(cf, md, color="#9467bd", linewidth=1.2, label="Cf")
    ax2.axvline(1.0 / kf_adopted, color="#2ca02c", linestyle="--", linewidth=1.2, label="1/2.2")
    ax2.set_xlabel("Cf = 1/Kf (1/GPa)")
    ax2.set_title("Compressibilidade")
    ax2.axhline(z0, color="0.5", linestyle="--", linewidth=0.8)
    ax2.axhline(z1, color="0.5", linestyle="--", linewidth=0.8)
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7, loc="best")

    ax0.set_ylabel("Depth (m)")
    fig.suptitle(
        "Poco 861: Wood+NMR (SWIRR) no intervalo [{:.1f}, {:.1f}] m".format(z0, z1),
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compare_to_reference(
    results: Dict[str, Dict[str, Any]],
    mape_tol: float = 0.05,
    bias_tol: float = 0.02,
) -> Tuple[bool, List[str]]:
    """Check computed metrics against slide reference values."""
    lines: List[str] = []
    ok_all = True
    for sid, ref_stages in REFERENCE_SLIDE.items():
        if sid not in results:
            ok_all = False
            lines.append("MISSING scenario {}".format(sid))
            continue
        for stage, ref in ref_stages.items():
            got = results[sid][stage]
            dm = abs(got["mape_pct"] - ref["mape_pct"])
            db = abs(got["bias_km_s"] - ref["bias_km_s"])
            stage_ok = (dm <= mape_tol) and (db <= bias_tol)
            ok_all = ok_all and stage_ok
            lines.append(
                "{:16s} {:10s} MAPE got={:6.3f} ref={:6.2f} d={:.3f} | "
                "bias got={:+.4f} ref={:+.3f} d={:.4f} {}".format(
                    sid,
                    stage,
                    got["mape_pct"],
                    ref["mape_pct"],
                    dm,
                    got["bias_km_s"],
                    ref["bias_km_s"],
                    db,
                    "OK" if stage_ok else "FAIL",
                )
            )
    return ok_all, lines


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(
        description="Well 861 Kf fluid sensitivity (Gassmann + hybrid OOF)"
    )
    p.add_argument("--out-root", type=Path, default=OUT_ROOT)
    p.add_argument(
        "--write-latex-figures",
        action="store_true",
        help="Copy regenerated figures into methods_comparison/latex/figures/",
    )
    p.add_argument(
        "--check-slides",
        action="store_true",
        help="Exit non-zero if metrics disagree with Entrega 2 slide table.",
    )
    p.add_argument("--merge-tol-m", type=float, default=DEFAULT_MERGE_TOL_M)
    p.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    out_root = args.out_root.resolve()
    tables = out_root / "tables"
    figures = out_root / "figures"
    for d in (tables, figures):
        d.mkdir(parents=True, exist_ok=True)

    pvt = load_pvt_config(PVT_DEFAULTS_JSON)
    logs = load_logs_enriched()
    z0, z1 = study_interval_from_logs(logs)
    rock, log = load_fluid_tables()
    profile = attach_fluid_to_profile(logs, rock, log)
    sonic = load_sonic(DLIS_SONIC_CSV)

    hfu_table = load_hfu_lab_calibrated(HFU_LAB_CALIB_CSV)
    hfu_params = build_hfu_params_from_lab_calib(hfu_table)

    # Interval medians from rock (same window as slides).
    rock_iv = rock[(rock["MD"] >= z0) & (rock["MD"] <= z1)]
    kf_well = float(rock_iv["KFluid"].median())
    rho_well = float(rock_iv["RhoFluid"].median())

    n = len(profile)
    ones = np.ones(n, dtype=np.float64)

    scenarios: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = [
        (
            "kf_adopted_2p2",
            ones * float(pvt["kf_gpa"]),
            ones * float(pvt["rho_fluid_gcc"]),
            ones * float(pvt["sw"]),
            "pvt_default_2p2",
        ),
        (
            "kf_well_median",
            ones * kf_well,
            ones * rho_well,
            ones * 1.0,
            "rock_KFluid_median",
        ),
        (
            "nmr_wood_z",
            profile["kf_wood_gpa"].to_numpy(dtype=np.float64),
            profile["rho_wood_gcc"].to_numpy(dtype=np.float64),
            ones * 1.0,
            "wood_SWIRR_depth",
        ),
    ]
    kf_nmr_med = float(np.nanmedian(profile["kf_wood_gpa"].to_numpy(dtype=np.float64)))
    rho_nmr_med = float(np.nanmedian(profile["rho_wood_gcc"].to_numpy(dtype=np.float64)))
    scenarios.append(
        (
            "nmr_wood_median",
            ones * kf_nmr_med,
            ones * rho_nmr_med,
            ones * 1.0,
            "wood_SWIRR_median",
        )
    )

    results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[dict] = []

    for sid, kf, rho_f, sw, fname in scenarios:
        print("Running scenario {} ...".format(sid))
        res = run_scenario(
            scenario_id=sid,
            profile=profile,
            hfu_params=hfu_params,
            logs=logs,
            sonic=sonic,
            kf=kf,
            rho_f=rho_f,
            sw=sw,
            fluid_name=fname,
            merge_tol_m=float(args.merge_tol_m),
        )
        results[sid] = res
        res["validation"].to_csv(
            tables / "{}_validation.csv".format(sid),
            index=False,
            float_format="%.6f",
        )
        res["sat_profile"].to_csv(
            tables / "{}_sat_profile.csv".format(sid),
            index=False,
            float_format="%.6f",
        )
        for stage in ("gassmann", "hybrid_rf", "hybrid_lr"):
            summary_rows.append(
                {
                    "scenario_id": sid,
                    "stage": stage,
                    "mape_pct": res[stage]["mape_pct"],
                    "bias_km_s": res[stage]["bias_km_s"],
                    "rmse_km_s": res[stage]["rmse_km_s"],
                    "kf_median_gpa": res["kf_median_gpa"],
                    "rho_median_gcc": res["rho_median_gcc"],
                    "sw_median": res["sw_median"],
                    "fluid_name": fname,
                }
            )
        print(
            "  Gassmann MAPE={:.3f}% bias={:+.4f} | RF={:.3f}% | LR={:.3f}%".format(
                res["gassmann"]["mape_pct"],
                res["gassmann"]["bias_km_s"],
                res["hybrid_rf"]["mape_pct"],
                res["hybrid_lr"]["mape_pct"],
            )
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(tables / "summary_metrics.csv", index=False, float_format="%.6f")

    profile.to_csv(tables / "profile_with_fluids.csv", index=False, float_format="%.6f")

    fig_las = figures / "fig3_kf_sw_study_interval.png"
    fig_nmr = figures / "fig3_kf_sw_nmr_study_interval.png"
    plot_las_interval(
        rock=rock,
        log=log,
        z0=z0,
        z1=z1,
        kf_adopted=float(pvt["kf_gpa"]),
        kf_well_med=kf_well,
        out_path=fig_las,
    )
    plot_nmr_interval(
        rock=rock,
        z0=z0,
        z1=z1,
        kf_adopted=float(pvt["kf_gpa"]),
        kf_nmr_med=kf_nmr_med,
        out_path=fig_nmr,
    )

    if args.write_latex_figures:
        LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for src in (fig_las, fig_nmr):
            dst = LATEX_FIG_DIR / src.name
            shutil.copy2(src, dst)
            print("Wrote latex figure {}".format(dst))

    manifest = {
        "generated_utc": utc_now_iso(),
        "well_id": "861",
        "study_interval_m": [z0, z1],
        "pvt_default": pvt,
        "kf_well_median_gpa": kf_well,
        "rho_well_median_gcc": rho_well,
        "kf_nmr_wood_median_gpa": kf_nmr_med,
        "rho_nmr_wood_median_gcc": rho_nmr_med,
        "n_blocks_oof": N_BLOCKS,
        "n_estimators_rf": N_ESTIMATORS,
        "random_state": int(args.random_state),
        "scenarios": {
            sid: {
                "fluid_name": results[sid]["fluid_name"],
                "kf_median_gpa": results[sid]["kf_median_gpa"],
                "rho_median_gcc": results[sid]["rho_median_gcc"],
                "gassmann": results[sid]["gassmann"],
                "hybrid_rf": results[sid]["hybrid_rf"],
                "hybrid_lr": results[sid]["hybrid_lr"],
            }
            for sid in results
        },
        "reference_slide": REFERENCE_SLIDE,
    }
    (out_root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    ok, lines = compare_to_reference(results)
    check_path = tables / "slide_check.txt"
    check_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Versionable summary copies (processed/ is gitignored).
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tables / "summary_metrics.csv", RESULTS_ROOT / "summary_metrics.csv")
    shutil.copy2(check_path, RESULTS_ROOT / "slide_check.txt")
    (RESULTS_ROOT / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("--- slide check ---")
    for line in lines:
        print(line)
    print("OUT {}".format(out_root))
    print("RESULTS {}".format(RESULTS_ROOT))
    if args.check_slides and not ok:
        print("FAIL: metrics disagree with Entrega 2 slide table")
        return 1
    if ok:
        print("OK: metrics match Entrega 2 slide table within tolerance")
    else:
        print("WARN: metrics differ from slide table (see slide_check.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
