#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract DSI sonic log (DTCO/DTSM/VPVS) for Well 861 from DLIS.

Outputs:
  methods_comparison/data/processed/dlis_861/
    MANIFEST.txt
    depth_calibration.json
    tables/sonic_log.csv
    tables/sonic_log_full.csv

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from dlis_861_core import (
    DEFAULT_RAW_DIR,
    SonicExtractResult,
    calibration_to_dict,
    extract_sonic_log,
    write_calibration_json,
)
from ml_861_data import DEPTH_COL, ROOT, load_logs_enriched

OUT_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "dlis_861"
TABLES_DIR = OUT_ROOT / "tables"
CALIB_JSON = OUT_ROOT / "depth_calibration.json"
SONIC_CSV = TABLES_DIR / "sonic_log.csv"
SONIC_FULL_CSV = TABLES_DIR / "sonic_log_full.csv"
METRICS_JSON = OUT_ROOT / "metrics.json"


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output directories."""
    for d in (OUT_ROOT, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    p = argparse.ArgumentParser(description="Extract Well 861 DSI sonic log from DLIS.")
    p.add_argument(
        "--raw-dir",
        type=str,
        default=str(DEFAULT_RAW_DIR),
        help="Directory containing ait_pex_dsi.dlis.",
    )
    p.add_argument(
        "--depth-min",
        type=float,
        default=0.0,
        help="Depth window minimum (m). Default: Auddys min.",
    )
    p.add_argument(
        "--depth-max",
        type=float,
        default=0.0,
        help="Depth window maximum (m). Default: Auddys max.",
    )
    p.add_argument(
        "--merge-tol-m",
        type=float,
        default=0.25,
        help="Tolerance for TDEP scale calibration vs Auddys depths.",
    )
    return p.parse_args()


def sonic_output_columns() -> list:
    """Ordered export columns for the cropped sonic table."""
    return [
        "depth_m",
        "tdep",
        "dtco_usft",
        "dtsm_usft",
        "vp_sonic_km_s",
        "vs_sonic_km_s",
        "vpvs_sonic_calc",
        "vpvs_sonic_dlis",
        "sphi",
        "itt",
    ]


def build_metrics(result: SonicExtractResult) -> dict:
    """Summary metrics for the extracted sonic log."""
    sonic = result.sonic
    n = len(sonic)
    n_dtco = int(sonic["dtco_usft"].notna().sum())
    n_dtsm = int(sonic["dtsm_usft"].notna().sum())
    n_vp = int(sonic["vp_sonic_km_s"].notna().sum())
    cal = result.calibration
    metrics = {
        "generated_utc": utc_now_iso(),
        "source_file": result.source_file,
        "logical_file": result.logical_file,
        "frame_name": result.frame_name,
        "n_rows_cropped": n,
        "n_dtco_valid": n_dtco,
        "n_dtsm_valid": n_dtsm,
        "n_vp_valid": n_vp,
        "depth_min_m": cal.depth_min_m,
        "depth_max_m": cal.depth_max_m,
        "tdep_scale": cal.tdep_scale,
        "auddys_merge_matches": cal.merge_matches,
        "tdep_declared_units": cal.declared_units,
        "tdep_declared_scale": cal.declared_scale,
        "tdep_fitted_scale": cal.fitted_scale,
        "tdep_scale_source": cal.scale_source,
        "tdep_scale_rel_diff": cal.scale_rel_diff,
        "tdep_scale_warning": cal.scale_warning,
    }
    if n_vp > 0:
        vp = sonic["vp_sonic_km_s"].dropna()
        metrics["vp_sonic_mean_km_s"] = float(vp.mean())
        metrics["vp_sonic_median_km_s"] = float(vp.median())
        metrics["vp_sonic_min_km_s"] = float(vp.min())
        metrics["vp_sonic_max_km_s"] = float(vp.max())
    return metrics


def write_manifest(result: SonicExtractResult, metrics: dict) -> None:
    """Write MANIFEST.txt."""
    cal = result.calibration
    lines = [
        "Well 861 -- DLIS sonic extraction",
        "generated_utc: {}".format(metrics["generated_utc"]),
        "",
        "Source: {}".format(result.source_file),
        "Frame: {} (logical_file={})".format(result.frame_name, result.logical_file),
        "",
        "Depth calibration:",
        "  method: {}".format(cal.method),
        "  formula: {}".format(cal.depth_formula),
        "  tdep_scale: {:.4f} (source: {})".format(cal.tdep_scale, cal.scale_source),
        "  declared TDEP units: {!r} -> {}".format(
            cal.declared_units,
            "{:.4f}".format(cal.declared_scale) if cal.declared_scale else "unknown",
        ),
        "  fitted scale (cross-check): {}".format(
            "{:.4f}".format(cal.fitted_scale) if cal.fitted_scale else "n/a"
        ),
        "  scale disagreement: {}".format(
            "{:.2%}".format(cal.scale_rel_diff) if cal.scale_rel_diff is not None else "n/a"
        ),
        "  warning: {}".format(cal.scale_warning or "none"),
        "  Auddys depth range: {:.2f} - {:.2f} m".format(
            cal.auddys_depth_min_m, cal.auddys_depth_max_m
        ),
        "  merge matches (tol): {}".format(cal.merge_matches),
        "",
        "Cropped sonic log:",
        "  rows: {}".format(metrics["n_rows_cropped"]),
        "  DTCO valid: {}".format(metrics["n_dtco_valid"]),
        "  DTSM valid: {}".format(metrics["n_dtsm_valid"]),
        "  Vp valid: {}".format(metrics["n_vp_valid"]),
        "",
        "Outputs:",
        "  tables/sonic_log.csv",
        "  tables/sonic_log_full.csv",
        "  depth_calibration.json",
        "  metrics.json",
        "",
        "Next step:",
        "  python methods_comparison/scripts/run_861_dlis_dem_validation.py",
    ]
    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(raw_dir: Optional[Path] = None, depth_min: float = 0.0, depth_max: float = 0.0,
         merge_tol_m: float = 0.25) -> SonicExtractResult:
    """Run extraction and write artifacts."""
    ensure_dirs()
    raw = Path(raw_dir) if raw_dir is not None else DEFAULT_RAW_DIR

    logs = load_logs_enriched()
    aud_depths = logs[DEPTH_COL].to_numpy(dtype=np.float64)
    dmin = float(aud_depths.min()) if depth_min <= 0.0 else depth_min
    dmax = float(aud_depths.max()) if depth_max <= 0.0 else depth_max

    result = extract_sonic_log(
        raw_dir=raw,
        auddys_depths=aud_depths,
        depth_min_m=dmin,
        depth_max_m=dmax,
        merge_tolerance_m=merge_tol_m,
    )

    cols = sonic_output_columns()
    result.sonic[cols].to_csv(SONIC_CSV, index=False, float_format="%.6f")

    full_cols = cols + [
        "vp_sonic_m_s",
        "vs_sonic_m_s",
        "vpvs",
    ]
    present = [c for c in full_cols if c in result.sonic.columns]
    result.sonic[present].to_csv(SONIC_FULL_CSV, index=False, float_format="%.6f")

    write_calibration_json(result.calibration, CALIB_JSON)
    metrics = build_metrics(result)
    METRICS_JSON.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    write_manifest(result, metrics)

    print("Sonic extraction complete: {}".format(OUT_ROOT))
    print("  rows: {}".format(metrics["n_rows_cropped"]))
    print("  tdep_scale: {:.4f}".format(metrics["tdep_scale"]))
    print("  Vp median: {:.3f} km/s".format(metrics.get("vp_sonic_median_km_s", float("nan"))))
    return result


if __name__ == "__main__":
    args = parse_args()
    main(
        raw_dir=Path(args.raw_dir),
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        merge_tol_m=args.merge_tol_m,
    )
