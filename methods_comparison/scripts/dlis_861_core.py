#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DLIS sonic extraction helpers for Well 861 (3-BRSA-861-SPS).

Extracts DTCO/DTSM/VPVS from ait_pex_dsi.dlis, calibrates TDEP to meters
against the Auddys enriched depth column, and converts slowness to velocity.

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dlisio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SENTINELS: Tuple[float, ...] = (-999.25, -999.0, -9999.0)

DEFAULT_RAW_DIR = ROOT / "methods_comparison" / "data"
DEFAULT_DSI_GLOB = "*ait_pex_dsi*.dlis"

# Frame 60B in logical_file[0] carries the DSI processed curves.
DSI_LOGICAL_FILE = 0
DSI_FRAME_INDEX = 0
DSI_FRAME_NAME = "60B"

SONIC_MNEMONICS: Tuple[str, ...] = (
    "DTCO",
    "DTSM",
    "VPVS",
    "SPHI",
    "ITT",
)

DTCO_VALID_USFT: Tuple[float, float] = (40.0, 200.0)
DTSM_VALID_USFT: Tuple[float, float] = (80.0, 300.0)
VPVS_VALID: Tuple[float, float] = (1.2, 2.5)

# Feet per meter constant for us/ft -> m/s conversion.
USFT_TO_M_PER_S: float = 304800.0

# TDEP storage units -> units per meter. The DLIS channel declares its own
# unit, which is authoritative; the empirical search is kept only as a
# cross-check and as a fallback for files with no declared unit.
TDEP_UNITS_PER_METER: Dict[str, float] = {
    "m": 1.0,
    "cm": 100.0,
    "mm": 1000.0,
    "in": 39.37007874015748,
    "0.1 in": 393.7007874015748,
    "ft": 3.280839895013123,
    "0.1 ft": 32.80839895013123,
}

# Relative disagreement above which the fitted scale is considered suspect.
TDEP_SCALE_REL_TOL: float = 0.005


def normalize_unit_label(units: str) -> str:
    """Normalize a DLIS unit string for table lookup."""
    return " ".join(str(units).strip().lower().replace("inch", "in").split())


def declared_tdep_scale(units: str) -> Optional[float]:
    """
    Physical TDEP-per-meter factor from the unit declared in the DLIS.

    Returns None when the unit is absent or unrecognized.
    """
    key = normalize_unit_label(units)
    if not key:
        return None
    return TDEP_UNITS_PER_METER.get(key)


@dataclass(frozen=True)
class DepthCalibration:
    """TDEP to meters mapping used for sonic extraction."""

    method: str
    tdep_scale: float
    depth_formula: str
    n_samples_in_window: int
    depth_min_m: float
    depth_max_m: float
    auddys_depth_min_m: float
    auddys_depth_max_m: float
    merge_matches: int
    declared_units: str = ""
    declared_scale: Optional[float] = None
    fitted_scale: Optional[float] = None
    fitted_merge_matches: Optional[int] = None
    scale_source: str = "fitted"
    scale_rel_diff: Optional[float] = None
    scale_warning: str = ""


@dataclass(frozen=True)
class SonicExtractResult:
    """Sonic log table plus calibration metadata."""

    sonic: pd.DataFrame
    calibration: DepthCalibration
    source_file: str
    logical_file: int
    frame_name: str


def clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace DLIS sentinels and non-finite values with NaN."""
    out = np.asarray(arr, dtype=np.float64).copy()
    for sentinel in SENTINELS:
        out[np.isclose(out, sentinel, atol=1.0e-3)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def us_per_ft_to_velocity_m_s(slowness_usft: np.ndarray) -> np.ndarray:
    """
    Convert compressional/shear slowness (us/ft) to velocity (m/s).

    V (m/s) = 304800 / DTCO (us/ft).
    """
    s = np.asarray(slowness_usft, dtype=np.float64)
    out = np.full_like(s, np.nan)
    valid = np.isfinite(s) & (s > 0.0)
    out[valid] = USFT_TO_M_PER_S / s[valid]
    return out


def find_dsi_dlis_path(raw_dir: Path, glob_pattern: str = DEFAULT_DSI_GLOB) -> Path:
    """Resolve the DSI DLIS file path."""
    hits = sorted(raw_dir.glob(glob_pattern))
    if not hits:
        raise FileNotFoundError(
            "No DSI DLIS matching {!r} in {}".format(glob_pattern, raw_dir)
        )
    if len(hits) > 1:
        hits = sorted(hits, key=lambda p: len(p.name))
    return hits[0]


def _mask_slowness(
    values: np.ndarray,
    valid_range: Tuple[float, float],
) -> np.ndarray:
    """Keep slowness values inside a plausible range."""
    lo, hi = valid_range
    out = clean_array(values)
    bad = (out < lo) | (out > hi)
    out[bad] = np.nan
    return out


def extract_dsi_frame(
    dlis_path: Path,
    logical_file: int = DSI_LOGICAL_FILE,
    frame_index: int = DSI_FRAME_INDEX,
) -> Tuple[pd.DataFrame, str, str, str]:
    """
    Read TDEP and sonic mnemonics from the DSI 60B frame.

    Returns (frame_df, frame_name, source_name, tdep_units). The DataFrame has
    columns: tdep, depth_m (NaN until calibrated), dtco_usft, dtsm_usft, vpvs,
    sphi, itt.
    """
    files = dlisio.dlis.load(str(dlis_path))
    with files:
        if logical_file >= len(files):
            raise IndexError(
                "logical_file {} out of range for {} ({} files)".format(
                    logical_file, dlis_path.name, len(files)
                )
            )
        lf = files[logical_file]
        frames = list(lf.frames)
        if frame_index >= len(frames):
            raise IndexError(
                "frame_index {} out of range for {} ({} frames)".format(
                    frame_index, dlis_path.name, len(frames)
                )
            )
        frame = frames[frame_index]
        curves = frame.curves()
        names = list(curves.dtype.names)
        if "TDEP" not in names:
            raise KeyError("TDEP missing in {} frame {}".format(dlis_path.name, frame.name))

        tdep_units = ""
        for channel in frame.channels:
            if channel.name == "TDEP":
                tdep_units = str(channel.units or "")
                break

        tdep = clean_array(np.asarray(curves["TDEP"], dtype=np.float64))
        rows: Dict[str, np.ndarray] = {"tdep": tdep, "depth_m": np.full_like(tdep, np.nan)}

        if "DTCO" in names:
            rows["dtco_usft"] = _mask_slowness(
                np.asarray(curves["DTCO"], dtype=np.float64),
                DTCO_VALID_USFT,
            )
        else:
            rows["dtco_usft"] = np.full_like(tdep, np.nan)

        if "DTSM" in names:
            rows["dtsm_usft"] = _mask_slowness(
                np.asarray(curves["DTSM"], dtype=np.float64),
                DTSM_VALID_USFT,
            )
        else:
            rows["dtsm_usft"] = np.full_like(tdep, np.nan)

        for mnem, col in (("VPVS", "vpvs"), ("SPHI", "sphi"), ("ITT", "itt")):
            if mnem in names:
                arr = clean_array(np.asarray(curves[mnem], dtype=np.float64))
                if col == "vpvs":
                    bad = (arr < VPVS_VALID[0]) | (arr > VPVS_VALID[1])
                    arr[bad] = np.nan
                rows[col] = arr
            else:
                rows[col] = np.full_like(tdep, np.nan)

        df = pd.DataFrame(rows)
        df = df[np.isfinite(df["tdep"])].copy()
        df = df.sort_values("tdep").reset_index(drop=True)
        return df, frame.name, dlis_path.name, tdep_units


def _count_merge_matches(
    sonic_depths: np.ndarray,
    auddys_depths: np.ndarray,
    tolerance_m: float,
) -> int:
    """Count how many Auddys depths get a nearest sonic sample within tolerance."""
    if len(sonic_depths) == 0 or len(auddys_depths) == 0:
        return 0
    left = pd.DataFrame({"depth_m": np.sort(sonic_depths)})
    right = pd.DataFrame({"aud_depth_m": np.sort(auddys_depths)})
    merged = pd.merge_asof(
        right,
        left,
        left_on="aud_depth_m",
        right_on="depth_m",
        direction="nearest",
        tolerance=tolerance_m,
    )
    return int(merged["depth_m"].notna().sum())


def calibrate_tdep_scale(
    tdep: np.ndarray,
    auddys_depths: Sequence[float],
    depth_min_m: float,
    depth_max_m: float,
    merge_tolerance_m: float = 0.25,
    scale_min: float = 370.0,
    scale_max: float = 410.0,
    scale_step: float = 0.1,
    tdep_units: str = "",
    rel_tol: float = TDEP_SCALE_REL_TOL,
) -> DepthCalibration:
    """
    Map TDEP to meters as depth_m = TDEP / scale.

    The empirical search maximizes Auddys merge coverage, but a scale fitted to
    force overlap can silently invent a depth registration. When the DLIS
    declares a TDEP unit, that unit is authoritative: the fitted scale is only
    a cross-check, and a disagreement beyond rel_tol raises a warning.
    """
    tdep_arr = np.asarray(tdep, dtype=np.float64)
    aud = np.asarray(list(auddys_depths), dtype=np.float64)
    aud_min = float(np.min(aud))
    aud_max = float(np.max(aud))

    def window_stats(scale: float) -> Tuple[int, int, float, float]:
        depth = tdep_arr / scale
        in_window = (depth >= depth_min_m) & (depth <= depth_max_m)
        n_window = int(in_window.sum())
        matches = _count_merge_matches(depth[in_window], aud, merge_tolerance_m)
        depth_win = depth[in_window]
        if len(depth_win) == 0:
            return n_window, matches, float("nan"), float("nan")
        return n_window, matches, float(np.min(depth_win)), float(np.max(depth_win))

    best_scale = scale_min
    best_matches = -1
    best_n_window = -1
    scale = scale_min
    while scale <= scale_max + 1.0e-9:
        n_window, matches, _, _ = window_stats(scale)
        if matches > best_matches or (matches == best_matches and n_window > best_n_window):
            best_matches = matches
            best_n_window = n_window
            best_scale = scale
        scale += scale_step

    declared = declared_tdep_scale(tdep_units)
    scale_used = float(best_scale)
    source = "fitted"
    rel_diff: Optional[float] = None
    warning_msg = ""

    if declared is not None and declared > 0.0:
        rel_diff = float(abs(best_scale - declared) / declared)
        scale_used = float(declared)
        source = "declared_units"
        if rel_diff > rel_tol:
            warning_msg = (
                "TDEP scale fitted to Auddys depths ({:.4f}) disagrees with the "
                "unit declared in the DLIS ({!r} -> {:.4f}) by {:.2%}. Using the "
                "declared unit; the fitted value would shift depths by about "
                "{:.1f} m at {:.0f} m."
            ).format(
                best_scale,
                tdep_units,
                declared,
                rel_diff,
                abs(aud_min * (declared / best_scale - 1.0)),
                aud_min,
            )
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
    else:
        warning_msg = (
            "DLIS declares no usable TDEP unit ({!r}); falling back to the "
            "scale fitted against Auddys depths ({:.4f}). Depth registration "
            "is unverified."
        ).format(tdep_units, best_scale)
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)

    n_window_used, matches_used, dmin, dmax = window_stats(scale_used)

    return DepthCalibration(
        method="tdep_divide_scale",
        tdep_scale=scale_used,
        depth_formula="depth_m = TDEP / tdep_scale",
        n_samples_in_window=n_window_used,
        depth_min_m=dmin,
        depth_max_m=dmax,
        auddys_depth_min_m=aud_min,
        auddys_depth_max_m=aud_max,
        merge_matches=int(matches_used),
        declared_units=str(tdep_units),
        declared_scale=declared,
        fitted_scale=float(best_scale),
        fitted_merge_matches=int(best_matches),
        scale_source=source,
        scale_rel_diff=rel_diff,
        scale_warning=warning_msg,
    )


def apply_depth_calibration(
    sonic_raw: pd.DataFrame,
    calibration: DepthCalibration,
) -> pd.DataFrame:
    """Attach depth_m and velocity columns to the raw sonic table."""
    out = sonic_raw.copy()
    out["depth_m"] = out["tdep"] / calibration.tdep_scale
    out["vp_sonic_m_s"] = us_per_ft_to_velocity_m_s(out["dtco_usft"].to_numpy())
    out["vs_sonic_m_s"] = us_per_ft_to_velocity_m_s(out["dtsm_usft"].to_numpy())
    out["vp_sonic_km_s"] = out["vp_sonic_m_s"] / 1000.0
    out["vs_sonic_km_s"] = out["vs_sonic_m_s"] / 1000.0
    vpvs_calc = out["vp_sonic_m_s"] / out["vs_sonic_m_s"]
    out["vpvs_sonic_calc"] = vpvs_calc
    out["vpvs_sonic_dlis"] = out["vpvs"]
    return out


def crop_depth_window(
    sonic: pd.DataFrame,
    depth_min_m: float,
    depth_max_m: float,
) -> pd.DataFrame:
    """Keep rows inside the MOGNO depth window with at least one sonic curve."""
    work = sonic[
        (sonic["depth_m"] >= depth_min_m) & (sonic["depth_m"] <= depth_max_m)
    ].copy()
    has_sonic = (
        work["dtco_usft"].notna()
        | work["dtsm_usft"].notna()
        | work["vpvs"].notna()
    )
    work = work[has_sonic].copy()
    return work.sort_values("depth_m").reset_index(drop=True)


def extract_sonic_log(
    raw_dir: Path,
    auddys_depths: Sequence[float],
    depth_min_m: float,
    depth_max_m: float,
    merge_tolerance_m: float = 0.25,
    dlis_glob: str = DEFAULT_DSI_GLOB,
) -> SonicExtractResult:
    """Full sonic extraction pipeline for Well 861."""
    dlis_path = find_dsi_dlis_path(raw_dir, dlis_glob)
    raw_df, frame_name, source_name, tdep_units = extract_dsi_frame(dlis_path)
    calibration = calibrate_tdep_scale(
        raw_df["tdep"].to_numpy(),
        auddys_depths=auddys_depths,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        merge_tolerance_m=merge_tolerance_m,
        tdep_units=tdep_units,
    )
    sonic = apply_depth_calibration(raw_df, calibration)
    sonic = crop_depth_window(sonic, depth_min_m, depth_max_m)
    return SonicExtractResult(
        sonic=sonic,
        calibration=calibration,
        source_file=source_name,
        logical_file=DSI_LOGICAL_FILE,
        frame_name=frame_name,
    )


def calibration_to_dict(cal: DepthCalibration) -> Dict[str, float]:
    """Serialize calibration dataclass to a JSON-safe dict."""
    return {
        "method": cal.method,
        "tdep_scale": cal.tdep_scale,
        "depth_formula": cal.depth_formula,
        "n_samples_in_window": cal.n_samples_in_window,
        "depth_min_m": cal.depth_min_m,
        "depth_max_m": cal.depth_max_m,
        "auddys_depth_min_m": cal.auddys_depth_min_m,
        "auddys_depth_max_m": cal.auddys_depth_max_m,
        "merge_matches": cal.merge_matches,
        "declared_units": cal.declared_units,
        "declared_scale": cal.declared_scale,
        "fitted_scale": cal.fitted_scale,
        "fitted_merge_matches": cal.fitted_merge_matches,
        "scale_source": cal.scale_source,
        "scale_rel_diff": cal.scale_rel_diff,
        "scale_warning": cal.scale_warning,
    }


def write_calibration_json(cal: DepthCalibration, path: Path) -> None:
    """Write depth calibration metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(calibration_to_dict(cal), indent=2) + "\n",
        encoding="utf-8",
    )
