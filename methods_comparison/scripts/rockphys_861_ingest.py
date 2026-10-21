#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load and normalize ROCKPHYS_Database lab tables for Well 861 (MOGNO).

Source: methods_comparison/data/ROCKPHYS_Database_04_12_2024 (7).xlsx
Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml_861_data import ROOT

DEFAULT_ROCKPHYS_XLSX = (
    ROOT
    / "methods_comparison"
    / "data"
    / "ROCKPHYS_Database_04_12_2024 (7).xlsx"
)

WELL_861_TOKEN = "861"
DEFAULT_REF_PRESSURE_MPA = 22.1

# CT pipeline sample IDs; F2911V in CT maps to F2911H in ROCKPHYS lab book.
CT_SAMPLE_IDS: Tuple[str, ...] = (
    "F2829V",
    "F2830H",
    "F2852H",
    "F2854H",
    "F2859H",
    "F2870H",
    "F2880H",
    "F2910H",
    "F2911V",
    "F2935H",
)

LAB_SAMPLE_ALIASES: Dict[str, str] = {
    "F2911V": "F2911H",
}

VELOCITY_COLUMN_NAMES: Tuple[str, ...] = (
    "sample_id",
    "depth_m",
    "well",
    "p_freq_mhz",
    "s_freq_mhz",
    "test_type",
    "load_pressure_mpa",
    "vp_z_km_s",
    "vs_z_km_s",
    "vp_x_km_s",
    "vs_x_km_s",
    "vp_y_km_s",
    "vs_y_km_s",
    "bulk_density_gcc",
    "e_gpa",
    "poisson",
    "g_gpa",
    "k_gpa",
    "cb_1_gpa",
    "load_pressure_mpa_sat_section",
    "vp_z_km_s_sat",
    "vs_z_km_s_sat",
    "vp_x_km_s_sat",
    "vs_x_km_s_sat",
    "vp_y_km_s_sat",
    "vs_y_km_s_sat",
    "bulk_density_gcc_sat",
    "e_gpa_sat",
    "poisson_sat",
    "g_gpa_sat",
    "k_gpa_sat",
    "cb_1_gpa_sat",
    "notes",
)


def rockphys_source_path(path: Optional[Path] = None) -> Path:
    """Resolve ROCKPHYS workbook path."""
    p = Path(path) if path is not None else DEFAULT_ROCKPHYS_XLSX
    if not p.is_file():
        raise FileNotFoundError(str(p))
    return p


def _is_well_861(series: pd.Series) -> pd.Series:
    """True for rows tagged as Well 861."""
    return series.astype(str).str.contains(WELL_861_TOKEN, na=False)


def _read_velocity_raw(path: Path) -> pd.DataFrame:
    """Parse Velocity sheet with merged dry/sat column layout."""
    raw = pd.read_excel(path, sheet_name="Velocity", header=2)
    ncol = len(raw.columns)
    names = list(VELOCITY_COLUMN_NAMES[:ncol])
    raw.columns = names
    return raw


def load_velocity_861(path: Optional[Path] = None) -> pd.DataFrame:
    """
    All dry-velocity lab rows for Well 861.

    Returns long table with one row per (sample, confining pressure).
    """
    p = rockphys_source_path(path)
    df = _read_velocity_raw(p)
    df = df[_is_well_861(df["well"])].copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["depth_m"] = pd.to_numeric(df["depth_m"], errors="coerce")
    df["load_pressure_mpa"] = pd.to_numeric(df["load_pressure_mpa"], errors="coerce")
    for col in (
        "vp_z_km_s",
        "vs_z_km_s",
        "vp_x_km_s",
        "vs_x_km_s",
        "vp_y_km_s",
        "vs_y_km_s",
        "bulk_density_gcc",
        "k_gpa",
        "g_gpa",
        "poisson",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["vp_mean_axes_km_s"] = df[["vp_z_km_s", "vp_x_km_s", "vp_y_km_s"]].mean(axis=1)
    df["vs_mean_axes_km_s"] = df[["vs_z_km_s", "vs_x_km_s", "vs_y_km_s"]].mean(axis=1)
    valid = df["vp_z_km_s"].notna() & df["vs_z_km_s"].notna()
    df["vpvs_z"] = np.where(valid, df["vp_z_km_s"] / df["vs_z_km_s"], np.nan)
    df = df.sort_values(["sample_id", "load_pressure_mpa"]).reset_index(drop=True)
    return df


def load_porosity_861(path: Optional[Path] = None) -> pd.DataFrame:
    """Unconfined porosity and dry bulk density for Well 861."""
    p = rockphys_source_path(path)
    df = pd.read_excel(p, sheet_name="Porosity", header=1)
    df = df[_is_well_861(df["Well / Location"])].copy()
    df = df.rename(
        columns={
            "Sample": "sample_id",
            "Depth (m)": "depth_m",
            "Well / Location": "well",
            "Unconfined Porosity (%)": "porosity_pct",
            "Bulk Density (dry) (g/cm³)": "bulk_density_dry_gcc",
            "Grain Density (g/cm³)": "grain_density_gcc",
            "Observation": "observation",
        }
    )
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["porosity_pu"] = pd.to_numeric(df["porosity_pct"], errors="coerce") / 100.0
    return df.sort_values("depth_m").reset_index(drop=True)


def load_mineralogy_861(path: Optional[Path] = None) -> pd.DataFrame:
    """Mineral fractions and reference moduli for Well 861."""
    p = rockphys_source_path(path)
    df = pd.read_excel(p, sheet_name="Mineralogy", header=1)
    df = df[_is_well_861(df["Well / Location"])].copy()
    df = df.rename(
        columns={
            "Sample": "sample_id",
            "Depth (m)": "depth_m",
            "Well / Location": "well",
        }
    )
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df.sort_values("depth_m").reset_index(drop=True)


def load_rock_info_861(path: Optional[Path] = None) -> pd.DataFrame:
    """Rock metadata for Well 861."""
    p = rockphys_source_path(path)
    df = pd.read_excel(p, sheet_name="Rock info", header=1)
    well_col = "Well" if "Well" in df.columns else "Well / Location"
    df = df[_is_well_861(df[well_col])].copy()
    df = df.rename(columns={"Sample": "sample_id", "Depth (m)": "depth_m", well_col: "well"})
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    return df.sort_values("depth_m").reset_index(drop=True)


def resolve_lab_sample_id(ct_sample_id: str) -> str:
    """Map CT sample id to ROCKPHYS lab sample id when orientations differ."""
    return LAB_SAMPLE_ALIASES.get(ct_sample_id, ct_sample_id)


def velocity_at_pressure(
    velocity_df: pd.DataFrame,
    pressure_mpa: float = DEFAULT_REF_PRESSURE_MPA,
) -> pd.DataFrame:
    """
    One row per sample at the requested confining pressure.

    If exact pressure missing, use the closest available step.
    """
    rows: List[dict] = []
    for sample_id, grp in velocity_df.groupby("sample_id"):
        grp = grp.sort_values("load_pressure_mpa")
        pressures = grp["load_pressure_mpa"].to_numpy(dtype=np.float64)
        if pressures.size == 0:
            continue
        idx = int(np.argmin(np.abs(pressures - pressure_mpa)))
        row = grp.iloc[idx].to_dict()
        row["pressure_target_mpa"] = pressure_mpa
        row["pressure_used_mpa"] = float(pressures[idx])
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).reset_index(drop=True)


def velocity_for_ct_plugs(
    velocity_df: pd.DataFrame,
    ct_sample_ids: Sequence[str] = CT_SAMPLE_IDS,
    pressure_mpa: float = DEFAULT_REF_PRESSURE_MPA,
) -> pd.DataFrame:
    """
    Lab velocities aligned to CT plug list (with orientation aliases).

    Adds ct_sample_id and lab_sample_id columns.
    """
    ref = velocity_at_pressure(velocity_df, pressure_mpa=pressure_mpa)
    ref_by_sample = ref.set_index("sample_id", drop=False)
    rows: List[dict] = []
    for ct_id in ct_sample_ids:
        lab_id = resolve_lab_sample_id(ct_id)
        if lab_id not in ref_by_sample.index:
            rows.append(
                {
                    "ct_sample_id": ct_id,
                    "lab_sample_id": lab_id,
                    "status": "missing_in_rockphys",
                }
            )
            continue
        rec = ref_by_sample.loc[lab_id].to_dict()
        rec["ct_sample_id"] = ct_id
        rec["lab_sample_id"] = lab_id
        rec["sample_alias"] = ct_id != lab_id
        rec["status"] = "ok"
        rows.append(rec)
    return pd.DataFrame(rows)


def ingest_summary(velocity_df: pd.DataFrame) -> Dict[str, object]:
    """High-level counts for manifest/metrics."""
    samples = sorted(velocity_df["sample_id"].dropna().unique().tolist())
    pressures = sorted(
        velocity_df["load_pressure_mpa"].dropna().unique().tolist()
    )
    return {
        "n_velocity_rows": int(len(velocity_df)),
        "n_samples": int(len(samples)),
        "samples": samples,
        "load_pressures_mpa": pressures,
        "depth_min_m": float(velocity_df["depth_m"].min()),
        "depth_max_m": float(velocity_df["depth_m"].max()),
    }
