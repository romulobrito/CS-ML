#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLP-CSGM protocol helpers for Well 861 MOGNO (Phi_lab profile).

Planning: methods_comparison/planning/etapa1f_clp_csgm_phi_lab_poco861.md
ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ml_861_data import (
    CLP_861_DEPTH_MAX_M,
    CLP_861_DEPTH_MIN_M,
    CLP_861_ML_ROOT,
    CLP_861_PRIMARY_TARGET,
    CLP_861_SCENARIO_PLUG_SPARSE,
    CLP_861_SCENARIO_RHO_SUBSAMPLE,
    CLP_861_SCENARIO_WIRELINE_PLUS_CT,
    CT_FEATURE_COLUMNS,
    DEFAULT_CT,
    DEFAULT_ENRICHED,
    DEPTH_COL,
    LOG_FEATURE_COLUMNS,
    clp_861_scenario_dir,
    load_ct_samples,
    load_logs_enriched,
)

DEPTH_TOLERANCE_M = 0.5

# Canonical rename for auddys_smoke_direct_ub.py (Logs sheet style).
CLP_U_CHANNEL_MAP: Dict[str, str] = {
    "Density (g/cc)": "density",
    "GR (API)": "gr",
    "Res_Deep": "res_deep",
    "Res_Shallow": "res_shallow",
    "Phi_Neutron (pu)": "phi_neutron",
    "Phi_Sonic (pu)": "phi_sonic",
    "Phi_ND (pu)": "phi_nd",
    "Lithotype": "lithotype",
}

CLP_TARGET_CANONICAL = "phi_lab"


@dataclass(frozen=True)
class PlugMeasurementRow:
    """One plug mapped to an enriched table row index."""

    sample_id: str
    ct_depth_m: float
    log_depth_m: float
    row_index: int
    depth_delta_m: float
    phi_lab_pu: float


@dataclass(frozen=True)
class Clp861RunPaths:
    """Standard artifact paths for one CLP-861 run."""

    run_root: Path
    tables: Path
    figures: Path
    logs: Path

    @staticmethod
    def from_scenario_run(scenario: str, run_id: str) -> "Clp861RunPaths":
        """Build paths under clp_861/phi_lab/<scenario>/runs/<run_id>/."""
        root = clp_861_scenario_dir(scenario) / "runs" / run_id
        return Clp861RunPaths(
            run_root=root,
            tables=root / "tables",
            figures=root / "figures",
            logs=root / "logs",
        )

    def ensure_dirs(self) -> None:
        """Create output directories."""
        self.tables.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)


def u_channels_csv() -> str:
    """Comma-separated u channel list for auddys_smoke_direct_ub."""
    return ",".join(CLP_U_CHANNEL_MAP[c] for c in LOG_FEATURE_COLUMNS)


def nearest_row_index(depths: np.ndarray, target_m: float) -> Tuple[int, float]:
    """Return (row_index, abs depth delta) for nearest depth."""
    idx = int(np.argmin(np.abs(depths - target_m)))
    delta = float(abs(depths[idx] - target_m))
    return idx, delta


def load_plug_measurement_rows(
    enriched_path: Optional[Path] = None,
    ct_path: Optional[Path] = None,
    tolerance_m: float = DEPTH_TOLERANCE_M,
) -> List[PlugMeasurementRow]:
    """
    Map each CT plug to the nearest enriched row (same rule as integration QC).

    Returns one entry per plug (10 rows), even when two plugs share one log row.
    """
    enriched = load_logs_enriched(enriched_path)
    ct = load_ct_samples(ct_path)
    depths = enriched[DEPTH_COL].to_numpy(dtype=np.float64)

    rows: List[PlugMeasurementRow] = []
    for _, plug in ct.sort_values("ct_depth_m").iterrows():
        log_depth = plug.get("log_depth_m")
        if log_depth is None or (isinstance(log_depth, float) and np.isnan(log_depth)):
            ct_depth = float(plug["ct_depth_m"])
            idx, delta = nearest_row_index(depths, ct_depth)
            log_depth = float(depths[idx])
        else:
            log_depth = float(log_depth)
            idx, delta = nearest_row_index(depths, log_depth)

        if delta > tolerance_m:
            raise ValueError(
                "Plug {} delta {:.3f} m exceeds tolerance {:.3f} m".format(
                    plug["sample_id"], delta, tolerance_m
                )
            )

        phi = plug.get("Phi_lab (pu)")
        if phi is None or (isinstance(phi, float) and np.isnan(phi)):
            phi = enriched.loc[idx, CLP_861_PRIMARY_TARGET]
        rows.append(
            PlugMeasurementRow(
                sample_id=str(plug["sample_id"]),
                ct_depth_m=float(plug["ct_depth_m"]),
                log_depth_m=log_depth,
                row_index=idx,
                depth_delta_m=delta,
                phi_lab_pu=float(phi),
            )
        )
    return rows


def plug_row_indices_unique(
    plugs: Sequence[PlugMeasurementRow],
) -> List[int]:
    """Sorted unique enriched row indices with at least one plug."""
    return sorted({p.row_index for p in plugs})


def export_plug_indices_csv(plugs: Sequence[PlugMeasurementRow], out_path: Path) -> None:
    """Write plug-to-row mapping for PROTOCOL and b mask construction."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        [
            {
                "sample_id": p.sample_id,
                "ct_depth_m": p.ct_depth_m,
                "log_depth_m": p.log_depth_m,
                "row_index": p.row_index,
                "depth_delta_m": p.depth_delta_m,
                "phi_lab_pu": p.phi_lab_pu,
            }
            for p in plugs
        ]
    )
    df.to_csv(out_path, index=False)


def compare_rf_baseline_dir() -> Path:
    """Directory for CLP vs RF comparison tables."""
    return CLP_861_ML_ROOT / "compare_rf_baseline"


def default_enriched_path() -> Path:
    """Default 87-row enriched table."""
    return DEFAULT_ENRICHED


def default_ct_path() -> Path:
    """Default 10-row CT table."""
    return DEFAULT_CT


def mogno_depth_bounds() -> Tuple[float, float]:
    """Inclusive MOGNO interval for CLP-861."""
    return CLP_861_DEPTH_MIN_M, CLP_861_DEPTH_MAX_M


def scenario_choices() -> Tuple[str, ...]:
    """Valid --scenario values for run_861_clp_csgm_phi_lab.py."""
    return (
        CLP_861_SCENARIO_PLUG_SPARSE,
        CLP_861_SCENARIO_RHO_SUBSAMPLE,
        CLP_861_SCENARIO_WIRELINE_PLUS_CT,
    )


def ct_u_column_names() -> Tuple[str, ...]:
    """ct_* columns for wireline_plus_ct_u scenario."""
    return CT_FEATURE_COLUMNS
