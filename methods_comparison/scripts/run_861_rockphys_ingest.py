#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ingest ROCKPHYS_Database lab tables for Well 861 into processed/.

Outputs:
  methods_comparison/data/processed/rockphys_861/
    MANIFEST.txt
    metrics.json
    tables/861_rockphys_velocity.csv
    tables/861_rockphys_porosity.csv
    tables/861_rockphys_mineralogy.csv
    tables/861_rockphys_rock_info.csv
    tables/861_rockphys_velocity_ct_plugs.csv

Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import pandas as pd

from ml_861_data import ROOT
from rockphys_861_ingest import (
    DEFAULT_REF_PRESSURE_MPA,
    ingest_summary,
    load_mineralogy_861,
    load_porosity_861,
    load_rock_info_861,
    load_velocity_861,
    rockphys_source_path,
    velocity_for_ct_plugs,
)

OUT_ROOT = ROOT / "methods_comparison" / "data" / "processed" / "rockphys_861"
TABLES_DIR = OUT_ROOT / "tables"


def utc_now_iso() -> str:
    """UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ensure_dirs() -> None:
    """Create output folders."""
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def write_manifest(summary: Dict[str, object], source: Path) -> None:
    """Human-readable ingest manifest."""
    lines = [
        "Well 861 -- ROCKPHYS lab ingest (Etapa 2)",
        "Generated: {}".format(utc_now_iso()),
        "",
        "Source workbook: {}".format(source.name),
        "Velocity rows (861): {}".format(summary["n_velocity_rows"]),
        "Unique samples: {}".format(summary["n_samples"]),
        "Depth range (m): {:.2f} -- {:.2f}".format(
            summary["depth_min_m"],
            summary["depth_max_m"],
        ),
        "Confining pressures (MPa): {}".format(
            ", ".join("{:.1f}".format(p) for p in summary["load_pressures_mpa"])
        ),
        "",
        "tables/",
        "  861_rockphys_velocity.csv         -- all pressure steps",
        "  861_rockphys_velocity_ct_plugs.csv -- CT plugs at {:.1f} MPa".format(
            DEFAULT_REF_PRESSURE_MPA
        ),
        "  861_rockphys_porosity.csv",
        "  861_rockphys_mineralogy.csv",
        "  861_rockphys_rock_info.csv",
        "",
        "Note: CT sample F2911V maps to lab id F2911H (orientation label).",
        "",
        "Planning: methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md",
    ]
    (OUT_ROOT / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ingest(
    source: Path | None = None,
    ref_pressure_mpa: float = DEFAULT_REF_PRESSURE_MPA,
) -> Dict[str, object]:
    """Load ROCKPHYS sheets and write processed CSV tables."""
    ensure_dirs()
    src = rockphys_source_path(source)

    vel = load_velocity_861(src)
    por = load_porosity_861(src)
    minr = load_mineralogy_861(src)
    info = load_rock_info_861(src)
    ct_vel = velocity_for_ct_plugs(vel, pressure_mpa=ref_pressure_mpa)

    vel.to_csv(TABLES_DIR / "861_rockphys_velocity.csv", index=False, float_format="%.6f")
    por.to_csv(TABLES_DIR / "861_rockphys_porosity.csv", index=False, float_format="%.6f")
    minr.to_csv(TABLES_DIR / "861_rockphys_mineralogy.csv", index=False, float_format="%.6f")
    info.to_csv(TABLES_DIR / "861_rockphys_rock_info.csv", index=False, float_format="%.6f")
    ct_vel.to_csv(
        TABLES_DIR / "861_rockphys_velocity_ct_plugs.csv",
        index=False,
        float_format="%.6f",
    )

    summary = ingest_summary(vel)
    n_ct_ok = int((ct_vel["status"] == "ok").sum()) if "status" in ct_vel.columns else 0
    metrics = {
        **summary,
        "ref_pressure_mpa": ref_pressure_mpa,
        "n_ct_plugs_matched": n_ct_ok,
        "n_ct_plugs_total": int(len(ct_vel)),
        "n_porosity_rows": int(len(por)),
        "n_mineralogy_rows": int(len(minr)),
        "n_rock_info_rows": int(len(info)),
        "source_file": src.name,
        "generated_utc": utc_now_iso(),
    }
    (OUT_ROOT / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    write_manifest(summary, src)

    return {
        "metrics": metrics,
        "out_root": str(OUT_ROOT),
    }


def parse_args() -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Ingest ROCKPHYS_Database lab data for Well 861",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Path to ROCKPHYS_Database xlsx (default: data/ROCKPHYS_...xlsx)",
    )
    parser.add_argument(
        "--ref-pressure-mpa",
        type=float,
        default=DEFAULT_REF_PRESSURE_MPA,
        help="Reference confining pressure for CT plug extract (default: 22.1)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    result = run_ingest(source=args.source, ref_pressure_mpa=args.ref_pressure_mpa)
    m = result["metrics"]
    print(
        "ROCKPHYS ingest: {} samples, {}/{} CT plugs at {:.1f} MPa".format(
            m["n_samples"],
            m["n_ct_plugs_matched"],
            m["n_ct_plugs_total"],
            m["ref_pressure_mpa"],
        )
    )
    print("Output: {}".format(result["out_root"]))


if __name__ == "__main__":
    main()
