#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 2g -- Fase A: CMR (NMR) extraction for Well 861.

Extracts CMRP_3MS, CMFF, BFV from the CMR DLIS, calibrates depth
against the sonic TDEP scale, and writes cmr_log_861.csv.

Planning: methods_comparison/planning/etapa2g_dem_multiscale_nmr_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import dlisio
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = ROOT / "methods_comparison" / "data"
DEFAULT_CMR_GLOB = "*861*_8_cmr_ecs.dlis"
DLIS_PROCESSED_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dlis_861"

SENTINELS: Tuple[float, ...] = (-999.25, -999.0, -9999.0)

DEPTH_MIN_M: float = 5205.0
DEPTH_MAX_M: float = 5234.0

# CMR ECS: logical_file[0], frame[3] = 75B (82 channels with BFV, CMRP_3MS, CMFF)
CMR_LOGICAL_FILE: int = 0
CMR_FRAME_INDEX: int = 3
CMR_FRAME_NAME: str = "75B"

CMR_MNEMONICS: Tuple[str, ...] = ("CMRP_3MS", "CMFF", "BFV")

# 60B frame for GR cross-check depth calibration
GR_FRAME_INDEX: int = 0
GR_FRAME_NAME: str = "60B"
GR_MNEMONIC: str = "GR"

# Depth calibration: TDEP scale found by maximizing GR correlation
# with Auddys enriched logs (GR corr=0.987, GR mean match 19.8 vs 19.5 API).
CALIBRATED_TDEP_SCALE: float = 393.7


@dataclass(frozen=True)
class CmrExtractResult:
    """CMR extraction output with metadata."""

    cmr_log: pd.DataFrame
    tdep_scale: float
    source_file: str
    n_raw: int
    n_cropped: int
    depth_min_m: float
    depth_max_m: float
    gr_corr_with_sonic: float


def utc_now_iso() -> str:
    """UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def clean_array(arr: np.ndarray) -> np.ndarray:
    """Replace DLIS sentinels and non-finite values with NaN."""
    out = np.asarray(arr, dtype=np.float64).copy()
    for sentinel in SENTINELS:
        out[np.isclose(out, sentinel, atol=1.0e-3)] = np.nan
    out[~np.isfinite(out)] = np.nan
    return out


def _resolve_glob(raw_dir: Path, pattern: str) -> Path:
    """Find a single DLIS file matching a glob pattern."""
    hits = sorted(raw_dir.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            "No file matching {!r} in {}".format(pattern, raw_dir)
        )
    if len(hits) > 1:
        hits = sorted(hits, key=lambda p: len(p.name))
    return hits[0]


def extract_cmr_curves(
    dlis_path: Path,
    logical_file: int = CMR_LOGICAL_FILE,
    frame_index: int = CMR_FRAME_INDEX,
    mnemonics: Tuple[str, ...] = CMR_MNEMONICS,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], str]:
    """
    Extract TDEP and NMR curves from the CMR DLIS 75B frame.

    Returns (tdep_raw, {mnemonic: array}, frame_name).
    """
    files = dlisio.dlis.load(str(dlis_path))
    with files:
        if logical_file >= len(files):
            raise IndexError(
                "logical_file {} out of range ({} files)".format(
                    logical_file, len(files)
                )
            )
        lf = files[logical_file]
        frames = list(lf.frames)
        if frame_index >= len(frames):
            raise IndexError(
                "frame_index {} out of range ({} frames)".format(
                    frame_index, len(frames)
                )
            )
        frame = frames[frame_index]
        curves = frame.curves()
        names = list(curves.dtype.names)

        if "TDEP" not in names:
            raise KeyError("TDEP not found in {}".format(frame.name))

        tdep = clean_array(np.asarray(curves["TDEP"], dtype=np.float64))
        extracted: Dict[str, np.ndarray] = {}
        for mnem in mnemonics:
            if mnem not in names:
                raise KeyError(
                    "{!r} not found in frame {} (available: {})".format(
                        mnem, frame.name, names[:20]
                    )
                )
            extracted[mnem] = clean_array(
                np.asarray(curves[mnem], dtype=np.float64)
            )
        return tdep, extracted, frame.name


def extract_cmr_gr(
    dlis_path: Path,
    logical_file: int = CMR_LOGICAL_FILE,
    frame_index: int = GR_FRAME_INDEX,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract TDEP and GR from the CMR 60B frame for depth cross-check."""
    files = dlisio.dlis.load(str(dlis_path))
    with files:
        lf = files[logical_file]
        frames = list(lf.frames)
        frame = frames[frame_index]
        curves = frame.curves()
        names = list(curves.dtype.names)
        if GR_MNEMONIC not in names:
            raise KeyError(
                "GR not found in frame {} of {}".format(frame.name, dlis_path.name)
            )
        tdep = clean_array(np.asarray(curves["TDEP"], dtype=np.float64))
        gr = clean_array(np.asarray(curves[GR_MNEMONIC], dtype=np.float64))
        return tdep, gr


def calibrate_cmr_depth_by_gr(
    dlis_path: Path,
    enriched_xlsx_path: Path,
    scale_min: float = 385.0,
    scale_max: float = 420.0,
    scale_step: float = 0.1,
    merge_tol_m: float = 0.15,
) -> Tuple[float, float]:
    """
    Find TDEP scale by maximizing GR correlation between CMR and Auddys logs.

    GR is the same measurement in both files, so high correlation + matching
    mean validates depth alignment. Returns (best_scale, best_gr_corr).
    """
    logs = pd.read_excel(enriched_xlsx_path)
    aud_df = (
        logs[["Depth(m)", "GR (API)"]]
        .dropna()
        .rename(columns={"Depth(m)": "depth_m", "GR (API)": "gr_aud"})
        .sort_values("depth_m")
    )

    files = dlisio.dlis.load(str(dlis_path))
    with files:
        lf = files[CMR_LOGICAL_FILE]
        frame = list(lf.frames)[GR_FRAME_INDEX]
        curves = frame.curves()
        tdep_60 = clean_array(np.asarray(curves["TDEP"], dtype=np.float64))
        gr_raw = clean_array(np.asarray(curves[GR_MNEMONIC], dtype=np.float64))

    best_scale = CALIBRATED_TDEP_SCALE
    best_corr = -2.0

    scale = scale_min
    while scale <= scale_max + 1.0e-9:
        depth_60 = tdep_60 / scale
        gr_df = pd.DataFrame({"depth_m": depth_60, "gr_cmr": gr_raw}).dropna()
        gr_df = gr_df[
            (gr_df["depth_m"] >= DEPTH_MIN_M) & (gr_df["depth_m"] <= DEPTH_MAX_M)
        ].sort_values("depth_m")

        if len(gr_df) < 10:
            scale += scale_step
            continue

        merged = pd.merge_asof(
            aud_df, gr_df, on="depth_m", direction="nearest", tolerance=merge_tol_m
        )
        valid = merged["gr_cmr"].notna() & merged["gr_aud"].notna()
        n_valid = int(valid.sum())
        if n_valid < 20:
            scale += scale_step
            continue

        corr = float(
            merged.loc[valid, "gr_cmr"].corr(merged.loc[valid, "gr_aud"])
        )
        if corr > best_corr:
            best_corr = corr
            best_scale = scale

        scale += scale_step

    return float(best_scale), float(best_corr)


def extract_cmr_log(
    raw_dir: Path,
    enriched_xlsx_path: Path,
    cmr_glob: str = DEFAULT_CMR_GLOB,
) -> CmrExtractResult:
    """Full CMR extraction pipeline for Well 861."""
    dlis_path = _resolve_glob(raw_dir, cmr_glob)
    print("  Source: {}".format(dlis_path.name))

    # Calibrate depth using GR correlation with Auddys enriched logs
    tdep_scale, gr_corr = calibrate_cmr_depth_by_gr(
        dlis_path, enriched_xlsx_path
    )
    print("  TDEP scale: {:.1f}  (GR corr: {:.4f})".format(tdep_scale, gr_corr))

    tdep_raw, curves, frame_name = extract_cmr_curves(dlis_path)
    print(
        "  CMR 75B frame: {} rows, mnemonics={}".format(
            len(tdep_raw), list(curves.keys())
        )
    )

    depth_m = tdep_raw / tdep_scale

    cmr_df = pd.DataFrame({"depth_m": depth_m, "tdep": tdep_raw})
    for mnem in CMR_MNEMONICS:
        col_name = mnem.lower()
        cmr_df[col_name] = curves[mnem]

    cmr_df = cmr_df[np.isfinite(cmr_df["depth_m"])].copy()
    n_raw = len(cmr_df)

    cmr_df = cmr_df[
        (cmr_df["depth_m"] >= DEPTH_MIN_M)
        & (cmr_df["depth_m"] <= DEPTH_MAX_M)
    ].copy()
    cmr_df = cmr_df.sort_values("depth_m").reset_index(drop=True)
    n_cropped = len(cmr_df)

    # Remove rows with near-zero/invalid porosity
    valid_porosity = cmr_df["cmrp_3ms"] > 0.005
    n_before = n_cropped
    cmr_df = cmr_df[valid_porosity].reset_index(drop=True)
    n_cropped = len(cmr_df)
    print("  Porosity filter: {} -> {} rows (removed {} with CMRP < 0.005)".format(
        n_before, n_cropped, n_before - n_cropped
    ))

    for col in ("cmrp_3ms", "cmff", "bfv"):
        n_valid = int(cmr_df[col].notna().sum())
        vals = cmr_df[col].dropna()
        if len(vals) > 0:
            print("  {}: {} valid, mean={:.4f} min={:.4f} max={:.4f}".format(
                col, n_valid, float(vals.mean()), float(vals.min()), float(vals.max())
            ))

    qc_sum = cmr_df["cmff"] + cmr_df["bfv"]
    qc_ratio = qc_sum / cmr_df["cmrp_3ms"]
    valid_ratio = qc_ratio.dropna()
    if len(valid_ratio) > 0:
        print(
            "  (CMFF+BFV)/CMRP_3MS: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
                float(valid_ratio.mean()),
                float(valid_ratio.std()),
                float(valid_ratio.min()),
                float(valid_ratio.max()),
            )
        )

    dmin = float(cmr_df["depth_m"].min()) if n_cropped > 0 else float("nan")
    dmax = float(cmr_df["depth_m"].max()) if n_cropped > 0 else float("nan")

    return CmrExtractResult(
        cmr_log=cmr_df,
        tdep_scale=tdep_scale,
        source_file=dlis_path.name,
        n_raw=n_raw,
        n_cropped=n_cropped,
        depth_min_m=dmin,
        depth_max_m=dmax,
        gr_corr_with_sonic=gr_corr,
    )


def main() -> None:
    """CLI entry point for CMR extraction."""
    parser = argparse.ArgumentParser(
        description="Extract CMR (NMR) curves from Well 861 DLIS."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing DLIS files.",
    )
    parser.add_argument(
        "--cmr-glob",
        type=str,
        default=DEFAULT_CMR_GLOB,
        help="Glob pattern for the CMR DLIS file.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_dir():
        raise FileNotFoundError("Raw directory not found: {}".format(raw_dir))

    enriched_xlsx = (
        ROOT / "methods_comparison" / "data" / "processed"
        / "861_integrated_logs_enriched.xlsx"
    )
    if not enriched_xlsx.is_file():
        raise FileNotFoundError(
            "Enriched logs not found: {}".format(enriched_xlsx)
        )

    out_dir = DLIS_PROCESSED_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Etapa 2g Fase A: CMR extraction (Well 861) ===")
    print("Source dir: {}".format(raw_dir))

    result = extract_cmr_log(raw_dir, enriched_xlsx, cmr_glob=args.cmr_glob)

    out_csv = out_dir / "cmr_log_861.csv"
    result.cmr_log.to_csv(out_csv, index=False, float_format="%.6f")
    print("Wrote {} ({} rows)".format(out_csv, result.n_cropped))

    meta = {
        "source_file": result.source_file,
        "tdep_scale": result.tdep_scale,
        "n_raw": result.n_raw,
        "n_cropped": result.n_cropped,
        "depth_min_m": result.depth_min_m,
        "depth_max_m": result.depth_max_m,
        "gr_corr_with_sonic": result.gr_corr_with_sonic,
        "generated_utc": utc_now_iso(),
    }
    meta_path = DLIS_PROCESSED_ROOT / "cmr_extraction_meta.json"
    meta_path.write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print("Wrote {}".format(meta_path))
    print("Done.")


if __name__ == "__main__":
    main()
