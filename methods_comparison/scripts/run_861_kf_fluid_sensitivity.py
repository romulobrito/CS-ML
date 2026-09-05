#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well 861: fluid Kf sensitivity for Gassmann + hybrid residual (Entrega 2).

Scenarios (same DEM dry frame and Phi_ND; only fluid in Gassmann changes):
  1) kf_adopted_2p2     -- PVT default Kf=2.2 GPa, rho=1.03, Sw=1
  2) kf_well_median     -- median KFluid/RhoFluid from rock861 in study interval
  3) nmr_wood_z         -- Wood/Reuss mix with SWIRR(z) (iso-stress reference)
  4) nmr_wood_median    -- constant median of Wood Kf/rho from scenario 3
  5) nmr_vrh_z          -- VRH mix with SWIRR(z) (alternative mixing scenario)
  6) nmr_vrh_median     -- constant median of VRH Kf/rho from scenario 5

Mixing laws (fluids):
  Wood/Reuss is the iso-stress mix (pressure equilibration during the wave).
  VRH = (Voigt + Reuss)/2 is an alternative mixing scenario, not a physical
  correction of Wood. Patchy/heterogeneous saturation is a different hypothesis.

Gassmann Sw after a mix:
  Kf and rho already represent the pore-filling mixture, so Gassmann is called
  with sw_gassmann=1. That is not a second saturation factor. sw_mix (from
  clipped SWIRR) is stored separately.

SWIRR clip:
  Values outside [0, 1] are clipped. Negative SWIRR mapped to 0 is a numerical
  bound, not evidence of pure oil. SWIRR is an irreducible-water proxy, not
  necessarily in-situ Sw.

Also regenerates depth-track figures used in Entrega 2 slides:
  fig3_kf_sw_study_interval.png
  fig3_kf_sw_nmr_study_interval.png  (VRH mix only; Wood is computed, not plotted)

Outputs:
  methods_comparison/data/processed/kf_fluid_sensitivity_861/
  methods_comparison/latex/figures/fig3_kf_sw_*.png  (when --write-latex-figures)
  methods_comparison/results/kf_fluid_sensitivity_861/  (versionable summary)

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
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

# Rounded values for --check-slides. Adopted / well / VRH match Entrega 2
# slides 14 and 16 (MAPE 2 dp, bias 3 dp). Wood is computed internally
# (not shown on the slides); those entries follow the same formatting.
REFERENCE_SLIDE: Dict[str, Dict[str, Dict[str, float]]] = {
    "kf_adopted_2p2": {
        "gassmann": {"mape_pct": 7.28, "bias_km_s": 0.271},
        "hybrid_rf": {"mape_pct": 2.78, "bias_km_s": 0.014},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "kf_well_median": {
        "gassmann": {"mape_pct": 7.28, "bias_km_s": 0.295},
        "hybrid_rf": {"mape_pct": 2.81, "bias_km_s": 0.017},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "nmr_wood_z": {
        "gassmann": {"mape_pct": 7.65, "bias_km_s": 0.261},
        "hybrid_rf": {"mape_pct": 2.97, "bias_km_s": 0.026},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "nmr_wood_median": {
        "gassmann": {"mape_pct": 7.65, "bias_km_s": 0.261},
        "hybrid_rf": {"mape_pct": 2.95, "bias_km_s": 0.026},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "nmr_vrh_z": {
        "gassmann": {"mape_pct": 7.63, "bias_km_s": 0.266},
        "hybrid_rf": {"mape_pct": 2.91, "bias_km_s": 0.016},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
    "nmr_vrh_median": {
        "gassmann": {"mape_pct": 7.62, "bias_km_s": 0.267},
        "hybrid_rf": {"mape_pct": 2.92, "bias_km_s": 0.020},
        "hybrid_lr": {"mape_pct": 2.10, "bias_km_s": 0.002},
    },
}

MAPE_PUBLISHED_DECIMALS = 2
BIAS_PUBLISHED_DECIMALS = 3


@dataclass(frozen=True)
class FluidScenario:
    """One Gassmann fluid case. Mix Sw and Gassmann Sw are distinct fields."""

    scenario_id: str
    kf: np.ndarray
    rho_f: np.ndarray
    sw_gassmann: np.ndarray
    fluid_name: str
    mix_law: str
    sw_mix: Optional[np.ndarray] = None


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def format_mape_pct(value: float) -> str:
    """MAPE as published on the slides (two decimal places)."""
    return "{:.{n}f}".format(float(value), n=MAPE_PUBLISHED_DECIMALS)


def format_bias_km_s(value: float) -> str:
    """Bias as published on the slides (three decimal places, km/s)."""
    return "{:.{n}f}".format(float(value), n=BIAS_PUBLISHED_DECIMALS)


def published_mape_matches(got: float, ref: float) -> bool:
    """True when MAPE matches the published two-decimal string."""
    return format_mape_pct(got) == format_mape_pct(ref)


def published_bias_matches(got: float, ref: float) -> bool:
    """True when bias matches the published three-decimal string."""
    return format_bias_km_s(got) == format_bias_km_s(ref)


def jsonable(value: Any) -> Any:
    """Replace non-finite floats with None so strict JSON stays valid."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def write_json(path: Path, payload: Any) -> None:
    """Write JSON with allow_nan=False after mapping NaN to null."""
    path.write_text(
        json.dumps(jsonable(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _clip01(x: np.ndarray) -> np.ndarray:
    """Clip saturations to [0, 1]."""
    return np.clip(x.astype(np.float64), 0.0, 1.0)


def reuss_kf(
    sw: np.ndarray,
    k_brine: np.ndarray,
    k_oil: np.ndarray,
) -> np.ndarray:
    """Reuss / Wood fluid bulk modulus (iso-stress, GPa)."""
    sw_u = _clip01(sw)
    so = 1.0 - sw_u
    inv = sw_u / k_brine + so / k_oil
    return 1.0 / inv


def voigt_kf(
    sw: np.ndarray,
    k_brine: np.ndarray,
    k_oil: np.ndarray,
) -> np.ndarray:
    """Voigt fluid bulk modulus (iso-strain, GPa)."""
    sw_u = _clip01(sw)
    so = 1.0 - sw_u
    return sw_u * k_brine + so * k_oil


def vrh_kf(
    sw: np.ndarray,
    k_brine: np.ndarray,
    k_oil: np.ndarray,
) -> np.ndarray:
    """Voigt-Reuss-Hill fluid bulk modulus (GPa).

    Alternative mixing scenario: K_VRH = (K_V + K_R) / 2.
    K_R is Wood (harmonic). This average is not a patchy-saturation model.
    """
    return 0.5 * (
        voigt_kf(sw, k_brine, k_oil) + reuss_kf(sw, k_brine, k_oil)
    )


def mix_rho(
    sw: np.ndarray,
    rho_brine: np.ndarray,
    rho_oil: np.ndarray,
) -> np.ndarray:
    """Volume-weighted fluid density mix (g/cc)."""
    sw_u = _clip01(sw)
    so = 1.0 - sw_u
    return sw_u * rho_brine + so * rho_oil


# Backward-compatible aliases used by older call sites / notebooks.
wood_kf = reuss_kf
wood_rho = mix_rho


def self_check_mix_laws() -> List[str]:
    """Algebra checks for Wood/VRH mix identities."""
    fails: List[str] = []
    sw = np.array([0.0, 0.31, 1.0], dtype=np.float64)
    kb = np.array([2.8, 2.8, 2.8], dtype=np.float64)
    ko = np.array([0.8, 0.8, 0.8], dtype=np.float64)
    kv = voigt_kf(sw, kb, ko)
    kr = reuss_kf(sw, kb, ko)
    kh = vrh_kf(sw, kb, ko)
    if abs(float(kr[0]) - 0.8) > 1e-12:
        fails.append("Wood Sw=0 must equal K_oil")
    if abs(float(kr[2]) - 2.8) > 1e-12:
        fails.append("Wood Sw=1 must equal K_brine")
    if abs(float(kv[0]) - 0.8) > 1e-12 or abs(float(kv[2]) - 2.8) > 1e-12:
        fails.append("Voigt extremes must match end-members")
    if np.any(kr - 1e-12 > kv):
        fails.append("Reuss must be <= Voigt")
    if np.any(np.abs(kh - 0.5 * (kv + kr)) > 1e-12):
        fails.append("VRH must be the arithmetic mean of Voigt and Reuss")
    if np.any((kh + 1e-12 < kr) | (kh - 1e-12 > kv)):
        fails.append("K_R <= K_VRH <= K_V must hold")
    rho_b = np.array([1.03, 1.03, 1.03], dtype=np.float64)
    rho_o = np.array([0.70, 0.70, 0.70], dtype=np.float64)
    rho = mix_rho(sw, rho_b, rho_o)
    expect = sw * rho_b + (1.0 - sw) * rho_o
    if np.any(np.abs(rho - expect) > 1e-12):
        fails.append("Density mix must be volume-weighted")
    return fails


def self_check_slide_rounding() -> List[str]:
    """Formatted slide check must reject stale one-tick rounding errors."""
    fails: List[str] = []
    if format_mape_pct(2.8109659637213316) != "2.81":
        fails.append("well-median RF MAPE must round to 2.81")
    if format_bias_km_s(0.016727797089864327) != "0.017":
        fails.append("well-median RF bias must round to 0.017")
    if published_mape_matches(2.82, 2.81):
        fails.append("stale MAPE 2.82 must not match published 2.81")
    if published_bias_matches(0.018, 0.017):
        fails.append("stale bias 0.018 must not match published 0.017")
    if not published_mape_matches(2.8109659637213316, 2.81):
        fails.append("2.81096 must match published MAPE 2.81")
    if not published_bias_matches(0.016727797089864327, 0.017):
        fails.append("0.01673 must match published bias 0.017")
    return fails


def self_check_json_nan() -> List[str]:
    """NaN must serialize as JSON null, not the token NaN."""
    payload = {"sw_mix_median": float("nan"), "ok": 1.0}
    try:
        text = json.dumps(jsonable(payload), allow_nan=False)
    except ValueError as exc:
        return ["jsonable failed to drop NaN: {}".format(exc)]
    if "NaN" in text:
        return ["strict JSON still contains NaN"]
    loaded = json.loads(text)
    if loaded["sw_mix_median"] is not None:
        return ["missing mix median must be JSON null"]
    if loaded["ok"] != 1.0:
        return ["finite floats must be preserved"]
    return []


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


def swirr_clip_table(md: np.ndarray, sw_raw: np.ndarray) -> pd.DataFrame:
    """Rows where SWIRR was outside [0, 1] before clipping."""
    raw = sw_raw.astype(np.float64)
    outside = (raw < 0.0) | (raw > 1.0)
    return pd.DataFrame(
        {
            "MD": md[outside],
            "SWIRR_raw": raw[outside],
            "SWIRR_clipped": _clip01(raw[outside]),
            "clip_reason": np.where(raw[outside] < 0.0, "lt_0", "gt_1"),
        }
    )


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
    sw_raw = merged["SWIRR"].to_numpy(dtype=np.float64)
    merged["swirr_raw"] = sw_raw
    merged["swirr_clipped_flag"] = (sw_raw < 0.0) | (sw_raw > 1.0)
    merged["sw_nmr"] = _clip01(sw_raw)
    merged["so_nmr"] = 1.0 - merged["sw_nmr"]
    kb = merged["KBrine"].to_numpy(dtype=np.float64)
    ko = merged["KOil"].to_numpy(dtype=np.float64)
    sw = merged["sw_nmr"].to_numpy(dtype=np.float64)
    merged["kf_wood_gpa"] = reuss_kf(sw, kb, ko)
    merged["kf_vrh_gpa"] = vrh_kf(sw, kb, ko)
    merged["rho_mix_gcc"] = mix_rho(
        sw,
        merged["RhoBrine"].to_numpy(dtype=np.float64),
        merged["RhoOil"].to_numpy(dtype=np.float64),
    )
    return merged.reset_index(drop=True)


def process_profile_with_fluid(
    profile: pd.DataFrame,
    hfu_params: Dict[int, Any],
    kf: np.ndarray,
    rho_f: np.ndarray,
    sw_gassmann: np.ndarray,
    fluid_name: str,
    mix_law: str,
    sw_mix: Optional[np.ndarray],
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
                sw=float(sw_gassmann[i]),
            )
            mix_val = float("nan") if sw_mix is None else float(sw_mix[i])
            rows.append(
                {
                    DEPTH_COL: depth,
                    HFU_COL: hfu,
                    "phi_input": phi_nd,
                    "Phi_Sonic (pu)": phi_sonic,
                    PHI_LAB_COL: phi_lab,
                    "fluid_name": fluid_name,
                    "mix_law": mix_law,
                    "kf_used_gpa": float(kf[i]),
                    "rho_fluid_used_gcc": float(rho_f[i]),
                    "sw_gassmann": float(sw_gassmann[i]),
                    "sw_mix": mix_val,
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
    scenario: FluidScenario,
    profile: pd.DataFrame,
    hfu_params: Dict[int, Any],
    logs: pd.DataFrame,
    sonic: pd.DataFrame,
    merge_tol_m: float,
) -> Dict[str, Any]:
    """One fluid scenario: Gassmann + hybrid RF/LR."""
    sat = process_profile_with_fluid(
        profile,
        hfu_params,
        kf=scenario.kf,
        rho_f=scenario.rho_f,
        sw_gassmann=scenario.sw_gassmann,
        fluid_name=scenario.fluid_name,
        mix_law=scenario.mix_law,
        sw_mix=scenario.sw_mix,
    )
    validation = validate_vs_sonic(sat, sonic, merge_tol_m)
    gass = metrics_dict_from_validation(validation)
    residual_df = build_residual_from_validation(validation, logs)
    hybrid = oof_hybrid_metrics(residual_df)
    sw_mix_median = (
        None
        if scenario.sw_mix is None
        else float(np.nanmedian(scenario.sw_mix))
    )
    return {
        "scenario_id": scenario.scenario_id,
        "fluid_name": scenario.fluid_name,
        "mix_law": scenario.mix_law,
        "kf_median_gpa": float(np.nanmedian(scenario.kf)),
        "rho_median_gcc": float(np.nanmedian(scenario.rho_f)),
        "sw_gassmann_median": float(np.nanmedian(scenario.sw_gassmann)),
        "sw_mix_median": sw_mix_median,
        "gassmann": gass,
        "hybrid_rf": hybrid["hybrid_rf"],
        "hybrid_lr": hybrid["hybrid_lr"],
        "sat_profile": sat,
        "validation": validation,
    }


def _style_depth_track(
    ax: Any,
    xlabel: str,
    title: str,
    z0: float,
    z1: float,
    *,
    legend_fs: float = 10.5,
) -> None:
    """Shared typography for portrait depth tracks on Beamer slides."""
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=15, pad=8)
    ax.tick_params(axis="both", labelsize=12, width=1.1, length=5)
    ax.axhline(z0, color="0.45", linestyle="--", linewidth=1.0)
    ax.axhline(z1, color="0.45", linestyle="--", linewidth=1.0)
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=legend_fs, loc="best", framealpha=0.92, borderpad=0.3)


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

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 7.6), sharey=True)
    ax0, ax1, ax2 = axes

    ax0.fill_betweenx(
        lsub["MD"].to_numpy(dtype=np.float64),
        0.0,
        lsub["SW"].to_numpy(dtype=np.float64),
        color="#1f77b4",
        alpha=0.85,
        label="Sw LAS",
    )
    ax0.set_xlim(0.0, 1.05)
    ax0.invert_yaxis()
    _style_depth_track(ax0, "Sw", "Sw (LAS)", z0, z1)

    ax1.plot(r["KFluid"], r["MD"], color="#d62728", linewidth=1.8, label="KFluid")
    ax1.axvline(
        kf_adopted,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="Kf=2.2 adopted",
    )
    ax1.axvline(
        kf_well_med,
        color="#ff7f0e",
        linestyle=":",
        linewidth=2.0,
        label="Kf median 2.83",
    )
    _style_depth_track(ax1, "Kf (GPa)", "Fluid bulk modulus", z0, z1)

    cf = 1.0 / r["KFluid"].to_numpy(dtype=np.float64)
    ax2.plot(cf, r["MD"], color="#9467bd", linewidth=1.8, label="Cf")
    ax2.axvline(
        1.0 / kf_adopted,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="1/2.2",
    )
    _style_depth_track(ax2, "Cf (1/GPa)", "Compressibility", z0, z1)

    ax0.set_ylabel("Depth (m)", fontsize=14)
    fig.suptitle(
        "Well 861: LAS Sw and Kf in [{:.1f}, {:.1f}] m".format(z0, z1),
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_nmr_interval(
    rock: pd.DataFrame,
    z0: float,
    z1: float,
    kf_adopted: float,
    kf_vrh_med: float,
    out_path: Path,
) -> None:
    """Depth tracks: SWIRR/So and VRH Kf/Cf in study interval."""
    m = (rock["MD"] >= z0) & (rock["MD"] <= z1)
    r = rock.loc[m].copy()
    sw_raw = r["SWIRR"].to_numpy(dtype=np.float64)
    sw = _clip01(sw_raw)
    so = 1.0 - sw
    kb = r["KBrine"].to_numpy(dtype=np.float64)
    ko = r["KOil"].to_numpy(dtype=np.float64)
    kf_v = vrh_kf(sw, kb, ko)
    md = r["MD"].to_numpy(dtype=np.float64)
    clipped = (sw_raw < 0.0) | (sw_raw > 1.0)

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 7.6), sharey=True)
    ax0, ax1, ax2 = axes

    ax0.fill_betweenx(md, 0.0, sw, color="#1f77b4", alpha=0.85, label="Sw NMR")
    ax0.fill_betweenx(md, sw, 1.0, color="#2ca02c", alpha=0.75, label="So = 1-Sw")
    if np.any(clipped):
        ax0.scatter(
            sw[clipped],
            md[clipped],
            s=36,
            c="red",
            zorder=3,
            label="SWIRR clip to 0",
        )
    ax0.set_xlim(0.0, 1.0)
    ax0.invert_yaxis()
    _style_depth_track(ax0, "Saturation", "NMR (SWIRR)", z0, z1)

    ax1.plot(kf_v, md, color="#d62728", linewidth=1.8, label="VRH")
    ax1.axvline(
        kf_adopted,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="Kf=2.2 adopted",
    )
    ax1.axvline(
        kf_vrh_med,
        color="#ff7f0e",
        linestyle=":",
        linewidth=2.0,
        label="VRH med. 1.40",
    )
    _style_depth_track(ax1, "Kf (GPa)", "Fluid bulk modulus", z0, z1, legend_fs=10.0)

    ax2.plot(1.0 / kf_v, md, color="#9467bd", linewidth=1.8, label="Cf VRH")
    ax2.axvline(
        1.0 / kf_adopted,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        label="1/2.2",
    )
    _style_depth_track(ax2, "Cf (1/GPa)", "Compressibility", z0, z1)

    ax0.set_ylabel("Depth (m)", fontsize=14)
    fig.suptitle(
        "Well 861: NMR VRH mix in [{:.1f}, {:.1f}] m".format(z0, z1),
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compare_to_reference(
    results: Dict[str, Dict[str, Any]],
) -> Tuple[bool, List[str]]:
    """Check computed metrics against published slide rounding.

    Pass/fail uses formatted MAPE (2 dp) and bias (3 dp). Raw deltas are
    logged only for diagnosis; they are not a tolerance window.
    """
    lines: List[str] = []
    ok_all = True
    for sid, ref_stages in REFERENCE_SLIDE.items():
        if sid not in results:
            ok_all = False
            lines.append("MISSING scenario {}".format(sid))
            continue
        for stage, ref in ref_stages.items():
            got = results[sid][stage]
            mape_got = format_mape_pct(got["mape_pct"])
            mape_ref = format_mape_pct(ref["mape_pct"])
            bias_got = format_bias_km_s(got["bias_km_s"])
            bias_ref = format_bias_km_s(ref["bias_km_s"])
            stage_ok = (mape_got == mape_ref) and (bias_got == bias_ref)
            ok_all = ok_all and stage_ok
            lines.append(
                "{:16s} {:10s} MAPE got={} ref={} | bias got={:+.3f} ref={:+.3f} {}".format(
                    sid,
                    stage,
                    mape_got,
                    mape_ref,
                    got["bias_km_s"],
                    ref["bias_km_s"],
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
    mix_fails = self_check_mix_laws()
    mix_fails.extend(self_check_slide_rounding())
    mix_fails.extend(self_check_json_nan())
    if mix_fails:
        raise RuntimeError("self-check failed: {}".format("; ".join(mix_fails)))

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

    rock_iv = rock[(rock["MD"] >= z0) & (rock["MD"] <= z1)]
    kf_well = float(rock_iv["KFluid"].median())
    rho_well = float(rock_iv["RhoFluid"].median())

    clip_rock = swirr_clip_table(
        rock_iv["MD"].to_numpy(dtype=np.float64),
        rock_iv["SWIRR"].to_numpy(dtype=np.float64),
    )
    clip_profile = swirr_clip_table(
        profile[DEPTH_COL].to_numpy(dtype=np.float64),
        profile["swirr_raw"].to_numpy(dtype=np.float64),
    )
    clip_rock.to_csv(tables / "swirr_clip_rock_interval.csv", index=False, float_format="%.6f")
    clip_profile.to_csv(
        tables / "swirr_clip_profile87.csv", index=False, float_format="%.6f"
    )

    n = len(profile)
    ones = np.ones(n, dtype=np.float64)
    sw_mix = profile["sw_nmr"].to_numpy(dtype=np.float64)
    rho_mix = profile["rho_mix_gcc"].to_numpy(dtype=np.float64)
    kf_wood = profile["kf_wood_gpa"].to_numpy(dtype=np.float64)
    kf_vrh = profile["kf_vrh_gpa"].to_numpy(dtype=np.float64)
    kf_wood_med = float(np.nanmedian(kf_wood))
    kf_vrh_med = float(np.nanmedian(kf_vrh))
    rho_nmr_med = float(np.nanmedian(rho_mix))

    scenarios: List[FluidScenario] = [
        FluidScenario(
            scenario_id="kf_adopted_2p2",
            kf=ones * float(pvt["kf_gpa"]),
            rho_f=ones * float(pvt["rho_fluid_gcc"]),
            sw_gassmann=ones * float(pvt["sw"]),
            fluid_name="pvt_default_2p2",
            mix_law="none",
        ),
        FluidScenario(
            scenario_id="kf_well_median",
            kf=ones * kf_well,
            rho_f=ones * rho_well,
            sw_gassmann=ones * 1.0,
            fluid_name="rock_KFluid_median",
            mix_law="none",
        ),
        FluidScenario(
            scenario_id="nmr_wood_z",
            kf=kf_wood,
            rho_f=rho_mix,
            sw_gassmann=ones * 1.0,
            fluid_name="wood_SWIRR_depth",
            mix_law="wood",
            sw_mix=sw_mix,
        ),
        FluidScenario(
            scenario_id="nmr_wood_median",
            kf=ones * kf_wood_med,
            rho_f=ones * rho_nmr_med,
            sw_gassmann=ones * 1.0,
            fluid_name="wood_SWIRR_median",
            mix_law="wood",
            sw_mix=sw_mix,
        ),
        FluidScenario(
            scenario_id="nmr_vrh_z",
            kf=kf_vrh,
            rho_f=rho_mix,
            sw_gassmann=ones * 1.0,
            fluid_name="vrh_SWIRR_depth",
            mix_law="vrh",
            sw_mix=sw_mix,
        ),
        FluidScenario(
            scenario_id="nmr_vrh_median",
            kf=ones * kf_vrh_med,
            rho_f=ones * rho_nmr_med,
            sw_gassmann=ones * 1.0,
            fluid_name="vrh_SWIRR_median",
            mix_law="vrh",
            sw_mix=sw_mix,
        ),
    ]

    results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[dict] = []

    for scn in scenarios:
        print("Running scenario {} ...".format(scn.scenario_id))
        res = run_scenario(
            scenario=scn,
            profile=profile,
            hfu_params=hfu_params,
            logs=logs,
            sonic=sonic,
            merge_tol_m=float(args.merge_tol_m),
        )
        results[scn.scenario_id] = res
        res["validation"].to_csv(
            tables / "{}_validation.csv".format(scn.scenario_id),
            index=False,
            float_format="%.6f",
        )
        res["sat_profile"].to_csv(
            tables / "{}_sat_profile.csv".format(scn.scenario_id),
            index=False,
            float_format="%.6f",
        )
        for stage in ("gassmann", "hybrid_rf", "hybrid_lr"):
            summary_rows.append(
                {
                    "scenario_id": scn.scenario_id,
                    "stage": stage,
                    "mape_pct": res[stage]["mape_pct"],
                    "bias_km_s": res[stage]["bias_km_s"],
                    "rmse_km_s": res[stage]["rmse_km_s"],
                    "kf_median_gpa": res["kf_median_gpa"],
                    "rho_median_gcc": res["rho_median_gcc"],
                    "sw_gassmann_median": res["sw_gassmann_median"],
                    "sw_mix_median": res["sw_mix_median"],
                    "mix_law": res["mix_law"],
                    "fluid_name": scn.fluid_name,
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
        kf_vrh_med=kf_vrh_med,
        out_path=fig_nmr,
    )

    if args.write_latex_figures:
        LATEX_FIG_DIR.mkdir(parents=True, exist_ok=True)
        for src in (fig_las, fig_nmr):
            dst = LATEX_FIG_DIR / src.name
            shutil.copy2(src, dst)
            print("Wrote latex figure {}".format(dst))

    clip_qc = {
        "n_rock_interval": int(len(rock_iv)),
        "n_swirr_clipped_rock": int(len(clip_rock)),
        "n_profile87": int(len(profile)),
        "n_swirr_clipped_profile87": int(int(profile["swirr_clipped_flag"].sum())),
        "note": (
            "Clipping SWIRR to [0, 1] is a numerical bound. "
            "Negative values mapped to 0 are not evidence of pure oil. "
            "SWIRR is an irreducible-water proxy, not necessarily in-situ Sw."
        ),
    }

    manifest = {
        "generated_utc": utc_now_iso(),
        "well_id": "861",
        "study_interval_m": [z0, z1],
        "pvt_default": pvt,
        "kf_well_median_gpa": kf_well,
        "rho_well_median_gcc": rho_well,
        "kf_nmr_wood_median_gpa": kf_wood_med,
        "kf_nmr_vrh_median_gpa": kf_vrh_med,
        "rho_nmr_mix_median_gcc": rho_nmr_med,
        "sw_mix_median": float(np.nanmedian(sw_mix)),
        "sw_gassmann_after_mix": 1.0,
        "mix_note": (
            "Wood/Reuss is the iso-stress fluid mix. VRH is an alternative "
            "mixing scenario, not a physical correction of Wood. After mixing, "
            "Gassmann uses sw_gassmann=1 because Kf and rho already represent "
            "the pore-filling fluid."
        ),
        "swirr_clip": clip_qc,
        "n_blocks_oof": N_BLOCKS,
        "n_estimators_rf": N_ESTIMATORS,
        "random_state": int(args.random_state),
        "published_rounding": {
            "mape_decimals": MAPE_PUBLISHED_DECIMALS,
            "bias_decimals": BIAS_PUBLISHED_DECIMALS,
            "note": (
                "Slide check compares formatted MAPE (2 dp) and bias (3 dp) "
                "to REFERENCE_SLIDE. It does not parse the LaTeX table."
            ),
        },
        "scenarios": {
            sid: {
                "fluid_name": results[sid]["fluid_name"],
                "mix_law": results[sid]["mix_law"],
                "kf_median_gpa": results[sid]["kf_median_gpa"],
                "rho_median_gcc": results[sid]["rho_median_gcc"],
                "sw_gassmann_median": results[sid]["sw_gassmann_median"],
                "sw_mix_median": results[sid]["sw_mix_median"],
                "gassmann": results[sid]["gassmann"],
                "hybrid_rf": results[sid]["hybrid_rf"],
                "hybrid_lr": results[sid]["hybrid_lr"],
            }
            for sid in results
        },
        "reference_slide": REFERENCE_SLIDE,
    }
    write_json(out_root / "MANIFEST.json", manifest)

    ok, lines = compare_to_reference(results)
    check_path = tables / "slide_check.txt"
    check_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(tables / "summary_metrics.csv", RESULTS_ROOT / "summary_metrics.csv")
    shutil.copy2(check_path, RESULTS_ROOT / "slide_check.txt")
    shutil.copy2(
        tables / "swirr_clip_rock_interval.csv",
        RESULTS_ROOT / "swirr_clip_rock_interval.csv",
    )
    write_json(RESULTS_ROOT / "MANIFEST.json", manifest)

    print("--- slide check ---")
    for line in lines:
        print(line)
    print("OUT {}".format(out_root))
    print("RESULTS {}".format(RESULTS_ROOT))
    print(
        "SWIRR clip rock interval: {} / {} rows".format(
            clip_qc["n_swirr_clipped_rock"], clip_qc["n_rock_interval"]
        )
    )
    if args.check_slides and not ok:
        print("FAIL: metrics disagree with Entrega 2 slide table")
        return 1
    if ok:
        print("OK: metrics match Entrega 2 published rounding")
    else:
        print("WARN: metrics differ from slide table (see slide_check.txt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
