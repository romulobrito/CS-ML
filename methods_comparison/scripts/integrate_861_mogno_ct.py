#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integrate Well 861 wireline/lab table with MOGNO microCT pore statistics.

Reads legacy logs Excel (data/Auddys_table.xlsx) and General MOGNO PoreInfo workbook;
writes ML-ready tables under methods_comparison/data/processed/ with 861_ prefix.

ASCII-only. See methods_comparison/planning/etapa1_dataset_ml_poco861.md.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

WELL_ID = "861"
LOGS_SHEET = "Logs"
DEPTH_COL = "Depth(m)"
DEPTH_TOLERANCE_M = 0.5

CANONICAL_LOG_COLUMNS: Tuple[str, ...] = (
    "Depth(m)",
    "GR (API)",
    "Density (g/cc)",
    "Res_Deep",
    "Res_Shallow",
    "Phi_Neutron (pu)",
    "Phi_Sonic (pu)",
    "Phi_ND (pu)",
    "Phi_lab (pu)",
    "k_lab (mD)",
    "RQI",
    "FZI_lab",
    "Lithotype",
    "HFU",
)

HFU_LABELS: Dict[int, str] = {
    1: "Poor",
    2: "Medium",
    3: "Good",
    4: "Excellent",
}

SAMPLE_SHEETS: Tuple[str, ...] = (
    "F2829V",
    "F2830H",
    "F2852H",
    "F2854H",
    "F2859H",
    "F2880H",
    "F2910H",
    "F2911V",
    "F2935H",
    "F2870H",
)

PORE_KV_MAP: Dict[str, str] = {
    "Mean Aspect Ratio": "ct_ar_mean",
    "Median Aspect Ratio": "ct_ar_median",
    "Mean Gamma": "ct_mean_gamma",
    "Porosity (%)": "ct_porosity_pct",
    "Microcroporosity (%) (Anselmetti et al., 1998)": "ct_phi_micro_pct",
    "Macro-mesoporosity (%) (Anselmetti et al., 1998)": "ct_phi_macro_meso_pct",
    "Permeability (Kozeny) (mD)": "ct_k_kozeny_md",
    "Permeability (Kozeny-Carman) (mD)": "ct_k_kozeny_carman_md",
    "Mean pore diameter (microns)": "ct_mean_pore_diameter_um",
    "Specific pore surface (mean) (microns^-1)": "ct_specific_pore_surface",
    "Pixel resolution (microns)": "ct_pixel_resolution_um",
}


@dataclass
class IntegrateConfig:
    """Runtime paths and tolerances for the 861 integration pipeline."""

    logs_path: Path
    general_xlsx: Path
    out_dir: Path
    depth_tolerance_m: float = DEPTH_TOLERANCE_M
    expected_log_rows: int = 87
    expected_ct_samples: int = 10


@dataclass
class QcReport:
    """Quality-control summary written to JSON."""

    well_id: str
    generated_utc: str
    n_log_rows: int
    n_ct_samples: int
    depth_tolerance_m: float
    per_sample: List[Dict[str, Any]] = field(default_factory=list)
    unit_warnings: List[str] = field(default_factory=list)
    collision_warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "well_id": self.well_id,
            "generated_utc": self.generated_utc,
            "n_log_rows": self.n_log_rows,
            "n_ct_samples": self.n_ct_samples,
            "depth_tolerance_m": self.depth_tolerance_m,
            "per_sample": self.per_sample,
            "unit_warnings": self.unit_warnings,
            "collision_warnings": self.collision_warnings,
            "errors": self.errors,
        }


def _normalize_sample_id(raw: str) -> str:
    text = str(raw).strip()
    text = text.replace("(6mm)", "").replace("(6 mm)", "").strip()
    return text


def _sample_orientation(sample_id: str) -> str:
    suffix = sample_id[-1].upper()
    if suffix in ("V", "H"):
        return suffix
    return ""


def _sample_diameter_mm(sample_id: str) -> float:
    if sample_id.upper().startswith("F2870"):
        return 6.0
    return 2.5


def _to_float(value: Any) -> Optional[float]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_quality(delta_m: float, tolerance_m: float) -> str:
    if delta_m > tolerance_m:
        return "error"
    if delta_m > 0.25:
        return "warn"
    return "ok"


def load_861_logs_table(logs_path: Path) -> pd.DataFrame:
    """Load and normalize the Well 861 logs+lab table."""
    if not logs_path.is_file():
        raise FileNotFoundError(str(logs_path))

    df = pd.read_excel(logs_path, sheet_name=LOGS_SHEET)
    missing = [c for c in CANONICAL_LOG_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Missing columns in {}: {}".format(LOGS_SHEET, missing))

    out = df[list(CANONICAL_LOG_COLUMNS)].copy()
    out = out.sort_values(DEPTH_COL).drop_duplicates(subset=[DEPTH_COL], keep="first")
    out = out.reset_index(drop=True)

    out["well_id"] = WELL_ID
    out["depth_index"] = np.arange(len(out), dtype=np.int64)
    out["hfu_label"] = out["HFU"].map(HFU_LABELS)

    lab_cols = ["Phi_lab (pu)", "k_lab (mD)", "FZI_lab", "HFU"]
    if out[lab_cols].isnull().any().any():
        raise ValueError("Null values found in required lab columns")

    depths = out[DEPTH_COL].to_numpy(dtype=np.float64)
    if np.any(np.diff(depths) < 0):
        raise ValueError("Depth(m) is not non-decreasing after sort")

    return out


def _parse_plots_index(general_xlsx: Path) -> pd.DataFrame:
    """Extract sample depth, aspect ratio, and lab/CT porosity from Plots sheet."""
    plots = pd.read_excel(general_xlsx, sheet_name="Plots", header=None)
    rows: List[Dict[str, Any]] = []

    for row_idx in range(3, 13):
        depth = _to_float(plots.iloc[row_idx, 13])
        sample_raw = plots.iloc[row_idx, 14]
        ar_plots = _to_float(plots.iloc[row_idx, 15])
        if sample_raw is None or (isinstance(sample_raw, float) and np.isnan(sample_raw)):
            continue
        if depth is None:
            continue

        sample_id = _normalize_sample_id(str(sample_raw))
        phi_ultra = None
        phi_ct = None
        phi_macro = None
        phi_micro = None
        if row_idx + 1 < len(plots):
            next_row = plots.iloc[row_idx + 1]
            next_sample = next_row.iloc[29]
            if isinstance(next_sample, str) and _normalize_sample_id(next_sample) == sample_id:
                phi_ultra = _to_float(next_row.iloc[30])
                phi_ct = _to_float(next_row.iloc[31])
                phi_macro = _to_float(next_row.iloc[32])
                phi_micro = _to_float(next_row.iloc[33])

        rows.append(
            {
                "sample_id": sample_id,
                "ct_depth_m": depth,
                "ar_mean_plots": ar_plots,
                "phi_ultrapore_ambient_pct": phi_ultra,
                "phi_ct_image_pct": phi_ct,
                "phi_macro_meso_frac": phi_macro,
                "phi_micro_frac": phi_micro,
                "orientation": _sample_orientation(sample_id),
                "diameter_mm": _sample_diameter_mm(sample_id),
            }
        )

    if not rows:
        raise ValueError("No CT samples parsed from Plots sheet")

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["sample_id"], keep="first")
    return out


def _parse_plan1_summary(general_xlsx: Path) -> pd.DataFrame:
    """Wide Plan1 metrics -> long table keyed by sample_id."""
    plan1 = pd.read_excel(general_xlsx, sheet_name="Plan1", header=None)
    header_row = 3
    sample_cols: List[Tuple[int, str]] = []
    for col_idx in range(1, plan1.shape[1]):
        raw = plan1.iloc[header_row, col_idx]
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            continue
        sample_cols.append((col_idx, _normalize_sample_id(str(raw))))

    metric_rows = {
        "phi_meso_macropores_vv": 4,
        "ar_meso_macropores": 5,
        "phi_micropores_vv": 6,
    }

    records: List[Dict[str, Any]] = []
    for col_idx, sample_id in sample_cols:
        rec: Dict[str, Any] = {"sample_id": sample_id}
        for metric, row_idx in metric_rows.items():
            rec[metric] = _to_float(plan1.iloc[row_idx, col_idx])
        records.append(rec)

    return pd.DataFrame(records)


def _parse_mineral_phases(general_xlsx: Path) -> pd.DataFrame:
    """Mineral dual thresh original and volume-corrected phase fractions."""
    mineral = pd.read_excel(general_xlsx, sheet_name="Mineral dual thresh", header=None)
    header_row = 2

    def _block(col_start: int, col_end: int, prefix: str) -> pd.DataFrame:
        sample_cols: List[Tuple[int, str]] = []
        for col_idx in range(col_start, col_end + 1):
            raw = mineral.iloc[header_row, col_idx]
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                continue
            sample_cols.append((col_idx, _normalize_sample_id(str(raw))))

        phase_rows = {
            prefix + "phi_ct_pct": 3,
            prefix + "solid1_pct": 4,
            prefix + "solid2_pct": 5,
        }
        records: List[Dict[str, Any]] = []
        for col_idx, sample_id in sample_cols:
            rec: Dict[str, Any] = {"sample_id": sample_id}
            for field_name, row_idx in phase_rows.items():
                rec[field_name] = _to_float(mineral.iloc[row_idx, col_idx])
            records.append(rec)
        return pd.DataFrame(records)

    original = _block(3, 12, "original_")
    corrected = _block(16, 25, "corrected_")

    if original.empty and corrected.empty:
        return pd.DataFrame(columns=["sample_id"])

    merged = pd.merge(original, corrected, on="sample_id", how="outer")
    return merged


def _parse_pore_sheet_kv(general_xlsx: Path, sheet_name: str) -> Dict[str, Any]:
    """Parse key-value pore parameters from a per-sample sheet."""
    df = pd.read_excel(general_xlsx, sheet_name=sheet_name, header=None)
    kv: Dict[str, Any] = {"sample_id": sheet_name}
    for _, row in df.iterrows():
        key = row.iloc[0]
        if not isinstance(key, str):
            continue
        key = key.strip()
        if key not in PORE_KV_MAP:
            continue
        kv[PORE_KV_MAP[key]] = _to_float(row.iloc[1])
    return kv


def build_ct_samples_table(general_xlsx: Path) -> pd.DataFrame:
    """Assemble one row per microCT sample."""
    plots_df = _parse_plots_index(general_xlsx)
    plan1_df = _parse_plan1_summary(general_xlsx)
    mineral_df = _parse_mineral_phases(general_xlsx)

    pore_rows: List[Dict[str, Any]] = []
    xl = pd.ExcelFile(general_xlsx)
    for sheet in SAMPLE_SHEETS:
        if sheet not in xl.sheet_names:
            raise ValueError("Missing expected sheet in General workbook: {}".format(sheet))
        pore_rows.append(_parse_pore_sheet_kv(general_xlsx, sheet))
    pore_df = pd.DataFrame(pore_rows)

    out = plots_df.merge(plan1_df, on="sample_id", how="left")
    out = out.merge(mineral_df, on="sample_id", how="left")
    out = out.merge(pore_df, on="sample_id", how="left", suffixes=("", "_dup"))

    # Drop accidental duplicate columns from merge
    dup_cols = [c for c in out.columns if c.endswith("_dup")]
    if dup_cols:
        out = out.drop(columns=dup_cols)

    out["phi_ultrapore_pu"] = out["phi_ultrapore_ambient_pct"] / 100.0
    out = out.sort_values("ct_depth_m").reset_index(drop=True)
    return out


def join_ct_to_logs(
    ct_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    tolerance_m: float,
) -> pd.DataFrame:
    """Nearest-depth join CT samples to log rows."""
    left = ct_df.sort_values("ct_depth_m").reset_index(drop=True)
    right = logs_df.sort_values(DEPTH_COL).reset_index(drop=True)

    merged = pd.merge_asof(
        left,
        right,
        left_on="ct_depth_m",
        right_on=DEPTH_COL,
        direction="nearest",
        tolerance=tolerance_m,
        suffixes=("", "_logdup"),
    )

    merged["log_depth_m"] = merged[DEPTH_COL]
    merged["depth_delta_m"] = (merged["ct_depth_m"] - merged["log_depth_m"]).abs()
    merged["match_quality"] = merged["depth_delta_m"].apply(
        lambda d: _match_quality(float(d), tolerance_m)
    )
    return merged


def enrich_logs_with_ct(
    logs_df: pd.DataFrame,
    integrated_ct: pd.DataFrame,
    tolerance_m: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach sparse CT columns to full log table.

    When two CT plugs share the same nearest log depth, keep the match with
    smaller depth_delta_m and record a QC collision warning for the other.
    """
    out = logs_df.copy()
    out["has_ct_sample"] = False
    out["sample_id"] = pd.Series([pd.NA] * len(out), dtype="object")
    out["ct_depth_m"] = np.nan
    out["depth_delta_m"] = np.nan
    out["match_quality"] = pd.Series([pd.NA] * len(out), dtype="object")

    ct_cols = [c for c in integrated_ct.columns if c not in (
        "log_depth_m",
        "depth_delta_m",
        "match_quality",
        DEPTH_COL,
    )]

    string_ct_cols = {"sample_id", "orientation", "match_quality"}
    for col in ct_cols:
        if col not in out.columns:
            if col in string_ct_cols:
                out[col] = pd.Series([pd.NA] * len(out), dtype="object")
            else:
                out[col] = np.nan

    collisions: List[str] = []
    ct_sorted = integrated_ct.sort_values("depth_delta_m").reset_index(drop=True)

    for _, ct_row in ct_sorted.iterrows():
        log_depth = ct_row.get("log_depth_m")
        if log_depth is None or (isinstance(log_depth, float) and np.isnan(log_depth)):
            continue
        idx = (out[DEPTH_COL] - float(log_depth)).abs().idxmin()
        delta = abs(float(out.loc[idx, DEPTH_COL]) - float(log_depth))
        if delta > tolerance_m:
            continue
        if bool(out.loc[idx, "has_ct_sample"]):
            existing = out.loc[idx, "sample_id"]
            collisions.append(
                "log depth {:.2f} m: kept {}, skipped {} (smaller delta wins)".format(
                    float(log_depth),
                    existing,
                    ct_row["sample_id"],
                )
            )
            continue
        out.loc[idx, "has_ct_sample"] = True
        out.loc[idx, "sample_id"] = ct_row["sample_id"]
        out.loc[idx, "ct_depth_m"] = ct_row["ct_depth_m"]
        out.loc[idx, "depth_delta_m"] = ct_row["depth_delta_m"]
        out.loc[idx, "match_quality"] = ct_row["match_quality"]
        for col in ct_cols:
            if col in integrated_ct.columns:
                out.loc[idx, col] = ct_row[col]

    return out, collisions


def build_qc_report(
    integrated_ct: pd.DataFrame,
    n_log_rows: int,
    tolerance_m: float,
) -> QcReport:
    """Build per-sample QC records and global warnings."""
    report = QcReport(
        well_id=WELL_ID,
        generated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_log_rows=n_log_rows,
        n_ct_samples=len(integrated_ct),
        depth_tolerance_m=tolerance_m,
    )

    report.unit_warnings.append(
        "phi_ct_image_pct is local microCT subsample; not directly comparable to Phi_lab (pu)"
    )
    report.unit_warnings.append(
        "ct_k_kozeny_md is model-derived; large deviation from k_lab is expected in some samples"
    )

    for _, row in integrated_ct.iterrows():
        phi_lab = _to_float(row.get("Phi_lab (pu)"))
        phi_ultra = _to_float(row.get("phi_ultrapore_ambient_pct"))
        phi_ct = _to_float(row.get("phi_ct_image_pct"))
        k_lab = _to_float(row.get("k_lab (mD)"))
        k_koz = _to_float(row.get("ct_k_kozeny_md"))
        delta = _to_float(row.get("depth_delta_m")) or 0.0
        quality = str(row.get("match_quality", "unknown"))

        entry: Dict[str, Any] = {
            "sample_id": row["sample_id"],
            "ct_depth_m": _to_float(row.get("ct_depth_m")),
            "log_depth_m": _to_float(row.get("log_depth_m")),
            "depth_delta_m": delta,
            "phi_lab_pu": phi_lab,
            "phi_ultrapore_pct": phi_ultra,
            "phi_ct_image_pct": phi_ct,
            "k_lab_md": k_lab,
            "k_kozeny_md": k_koz,
            "hfu": _to_float(row.get("HFU")),
            "fzi_lab": _to_float(row.get("FZI_lab")),
            "match_quality": quality,
        }
        report.per_sample.append(entry)

        if quality == "error":
            report.errors.append(
                "sample {} depth_delta_m={:.3f} exceeds tolerance {:.3f}".format(
                    row["sample_id"], delta, tolerance_m
                )
            )

    return report


def write_manifest(path: Path, cfg: IntegrateConfig, qc: QcReport) -> None:
    """Human-readable integration manifest."""
    lines = [
        "861_INTEGRATION_MANIFEST",
        "well_id: {}".format(WELL_ID),
        "logs_source: {}".format(cfg.logs_path),
        "general_xlsx: {}".format(cfg.general_xlsx),
        "out_dir: {}".format(cfg.out_dir),
        "generated_utc: {}".format(qc.generated_utc),
        "n_log_rows: {}".format(qc.n_log_rows),
        "n_ct_samples: {}".format(qc.n_ct_samples),
        "depth_tolerance_m: {}".format(cfg.depth_tolerance_m),
        "",
        "Outputs:",
        "  861_logs_table.xlsx",
        "  861_ct_samples.csv",
        "  861_integrated_ct_samples.xlsx",
        "  861_integrated_logs_enriched.xlsx",
        "  861_integration_qc.json",
        "  861_ML_DATASET_README.md",
        "",
        "QC errors: {}".format(len(qc.errors)),
        "QC collision warnings: {}".format(len(qc.collision_warnings)),
    ]
    for warn in qc.collision_warnings:
        lines.append("  COLLISION: {}".format(warn))
    for err in qc.errors:
        lines.append("  ERROR: {}".format(err))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ml_readme(path: Path) -> None:
    """Dataset dictionary for downstream ML."""
    text = """# 861 ML Dataset README

Auto-generated by integrate_861_mogno_ct.py.

## Tables

| File | Rows | Use |
|------|------|-----|
| 861_integrated_logs_enriched.xlsx | 87 | Full well profile; sparse CT on ~10 depths |
| 861_integrated_ct_samples.xlsx | 10 | Multiscale rows with full CT + nearest log |

## Features (X) -- wireline

- GR (API), Density (g/cc), Res_Deep, Res_Shallow
- Phi_Neutron (pu), Phi_Sonic (pu), Phi_ND (pu)
- Lithotype, HFU (use carefully in cross-validation)

## Features (X) -- microCT (ct_ prefix)

- ct_ar_mean, ct_mean_gamma, ct_porosity_pct
- phi_meso_macropores_vv, ar_meso_macropores
- corrected_solid1_pct, corrected_solid2_pct

## Targets (y) candidates

- FZI_lab (primary in existing RF scripts)
- Phi_lab (pu), k_lab (mD), HFU

## Leakage warning

Do not use RQI, FZI_lab, Phi_lab, k_lab as inputs when predicting FZI_lab.

## Split guidance

Prefer depth blocks or leave-one-plug-out (10 CT samples), not i.i.d. random split.

## Units

- _pu: fraction v/v (0-1)
- _pct: percent (0-100)
- _vv: volume fraction (0-1)
- _md: millidarcy
"""
    path.write_text(text, encoding="utf-8")


def run_integration(cfg: IntegrateConfig) -> QcReport:
    """Execute full Etapa 1 pipeline."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    logs_df = load_861_logs_table(cfg.logs_path)
    ct_df = build_ct_samples_table(cfg.general_xlsx)
    integrated_ct = join_ct_to_logs(ct_df, logs_df, cfg.depth_tolerance_m)
    enriched, collisions = enrich_logs_with_ct(
        logs_df, integrated_ct, cfg.depth_tolerance_m
    )

    if len(logs_df) != cfg.expected_log_rows:
        raise ValueError(
            "Expected {} log rows, got {}".format(cfg.expected_log_rows, len(logs_df))
        )
    if len(ct_df) != cfg.expected_ct_samples:
        raise ValueError(
            "Expected {} CT samples, got {}".format(cfg.expected_ct_samples, len(ct_df))
        )

    logs_df.to_excel(cfg.out_dir / "861_logs_table.xlsx", sheet_name=LOGS_SHEET, index=False)
    ct_df.to_csv(cfg.out_dir / "861_ct_samples.csv", index=False)
    integrated_ct.to_excel(cfg.out_dir / "861_integrated_ct_samples.xlsx", index=False)
    enriched.to_excel(cfg.out_dir / "861_integrated_logs_enriched.xlsx", index=False)

    qc = build_qc_report(integrated_ct, len(logs_df), cfg.depth_tolerance_m)
    qc.collision_warnings.extend(collisions)
    if qc.errors:
        raise ValueError("QC errors: {}".format("; ".join(qc.errors)))

    qc_path = cfg.out_dir / "861_integration_qc.json"
    qc_path.write_text(json.dumps(qc.to_dict(), indent=2), encoding="utf-8")
    write_manifest(cfg.out_dir / "861_INTEGRATION_MANIFEST.txt", cfg, qc)
    write_ml_readme(cfg.out_dir / "861_ML_DATASET_README.md")

    return qc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI argument parser."""
    default_logs = ROOT / "data" / "Auddys_table.xlsx"
    default_general = (
        ROOT
        / "methods_comparison"
        / "data"
        / "General_Output_Results_PoreInfo_samples_w861_MOGNOTomo.xlsx"
    )
    default_out = ROOT / "methods_comparison" / "data" / "processed"

    parser = argparse.ArgumentParser(
        description="Integrate Well 861 logs/lab with MOGNO microCT for ML datasets."
    )
    parser.add_argument("--logs-path", type=Path, default=default_logs)
    parser.add_argument("--general-xlsx", type=Path, default=default_general)
    parser.add_argument("--out-dir", type=Path, default=default_out)
    parser.add_argument(
        "--depth-tolerance-m",
        type=float,
        default=DEPTH_TOLERANCE_M,
        help="Max |ct_depth - log_depth| for merge_asof (m)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    cfg = IntegrateConfig(
        logs_path=args.logs_path.resolve(),
        general_xlsx=args.general_xlsx.resolve(),
        out_dir=args.out_dir.resolve(),
        depth_tolerance_m=float(args.depth_tolerance_m),
    )
    qc = run_integration(cfg)
    print(
        "OK well={} log_rows={} ct_samples={} out_dir={}".format(
            qc.well_id, qc.n_log_rows, qc.n_ct_samples, cfg.out_dir
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
