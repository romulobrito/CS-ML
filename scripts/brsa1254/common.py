#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared constants for the well 9-BRSA-1254-RJS plug study.

This study does not edit the article runners. ASCII only.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "1254"
OUT_DIR = REPO_ROOT / "outputs" / "reviewer_revision" / "brsa1254_plug_holdout"

PLUG_XLS = DATA_DIR / "9-brsa-1254-rjs_petrofisica_plugues.xls"
PASS_FILES: Dict[str, Path] = {
    "12a2_quad": DATA_DIR / "9-brsa-1254-rjs_12a2_quad.dlis",
    "12f_quad": DATA_DIR / "9-brsa-1254-rjs_12f_quad.dlis",
}
RESISTIVITY_BY_PASS: Dict[str, str] = {
    "12a2_quad": "RT10",
    "12f_quad": "RT90",
}
SCALAR_CURVES: Tuple[str, ...] = ("GR", "RHOB", "NPHI")

WELL_ANP = "9-BRSA-1254-RJS"
DIRECTION = "VH1"
SAMPLE_TYPE = "Plug"
DEPTH_MATCH_MAX_M = 0.25
EXPECTED_CANDIDATES = 161
EXPECTED_DEPTH_MIN_M = 4935.05
EXPECTED_DEPTH_MAX_M = 5005.45
DEPTH_ENDPOINT_TOL_M = 0.05

N_BLOCKS = 3
VAL_PLUGS = 16
WINDOW_LEN = 8
WINDOW_STEP = 1
RHOS: Tuple[float, ...] = (0.25, 0.50)
SEEDS: Tuple[int, ...] = (7, 23, 41)
U_COLUMNS: Tuple[str, ...] = ("gr", "rhob", "nphi", "rt")

# Same finite grid as scripts/run_r3m03_simple_baselines.py.
SMOOTH_ALPHAS: Tuple[float, ...] = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    1.0e7,
    1.0e8,
    1.0e9,
    1.0e10,
    1.0e11,
    1.0e12,
)
SMOOTH_LINEAR_LIMIT = float("inf")

LAMBDA_GRID: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
LATENT_DIM = 4
HIDDEN_DIM = 32
AE_EPOCHS = 80
CSGM_ITERS = 200
CSGM_RESTARTS = 2


def parse_float(value: object) -> float:
    """Parse a spreadsheet number that may use a decimal comma."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("Missing numeric value.")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", ".")
    if text == "":
        raise ValueError("Empty numeric value.")
    return float(text)


def load_candidate_plugs(path: Path) -> pd.DataFrame:
    """Vertical plugs with positive longitudinal permeability, sorted by depth."""
    frame = pd.read_excel(path, sheet_name="Sheet0")
    # Workbook headers contain non-ASCII characters. Match stable tokens.
    columns = [str(name) for name in frame.columns]
    well_col = _column_containing(columns, "Anp")
    type_col = _column_containing(columns, "Tipo")
    dir_col = _column_containing(columns, "Dir")
    depth_col = _column_containing(columns, "Profundidade")
    perm_col = _column_containing(columns, "Permeabilidade")
    missing = [
        label
        for label, found in (
            ("well", well_col),
            ("type", type_col),
            ("direction", dir_col),
            ("depth", depth_col),
            ("permeability", perm_col),
        )
        if found == ""
    ]
    if missing:
        raise ValueError("Plug table is missing columns: {0}".format(", ".join(missing)))

    rows: List[Dict[str, float]] = []
    for excel_row, record in frame.iterrows():
        well = str(record[well_col]).strip()
        sample_type = str(record[type_col]).strip()
        direction = str(record[dir_col]).strip()
        if well != WELL_ANP or sample_type != SAMPLE_TYPE or direction != DIRECTION:
            continue
        try:
            permeability = parse_float(record[perm_col])
            depth_m = parse_float(record[depth_col])
        except ValueError:
            continue
        if not math.isfinite(permeability) or permeability <= 0.0:
            continue
        if not math.isfinite(depth_m):
            continue
        rows.append(
            {
                "excel_row": float(excel_row),
                "depth_m": depth_m,
                "k_md": permeability,
                "log10_k": float(math.log10(permeability)),
            }
        )
    if not rows:
        raise ValueError("No positive VH1 plugs were found.")
    plugs = pd.DataFrame(rows).sort_values("depth_m").reset_index(drop=True)
    plugs.insert(0, "plug_index", np.arange(len(plugs), dtype=np.int64))
    if plugs["depth_m"].duplicated().any():
        raise ValueError("Candidate plugs repeat a depth.")
    return plugs


def _column_containing(columns: Sequence[str], token: str) -> str:
    matches = [name for name in columns if token in name]
    if len(matches) != 1:
        return ""
    return matches[0]


def assert_candidate_inventory(plugs: pd.DataFrame) -> None:
    """Stop if the spreadsheet is not the 161-plug inventory already counted."""
    count = int(plugs.shape[0])
    depth_min = float(plugs["depth_m"].min())
    depth_max = float(plugs["depth_m"].max())
    if count != EXPECTED_CANDIDATES:
        raise ValueError(
            "Expected {0} positive VH1 plugs, found {1}.".format(EXPECTED_CANDIDATES, count)
        )
    if abs(depth_min - EXPECTED_DEPTH_MIN_M) > DEPTH_ENDPOINT_TOL_M:
        raise ValueError("Depth minimum {0} is outside the expected interval.".format(depth_min))
    if abs(depth_max - EXPECTED_DEPTH_MAX_M) > DEPTH_ENDPOINT_TOL_M:
        raise ValueError("Depth maximum {0} is outside the expected interval.".format(depth_max))


def observation_count(n_plugs: int, rho: float) -> int:
    """How many plugs in a block are measurements at this frozen fraction."""
    count = int(round(float(rho) * int(n_plugs)))
    return max(1, min(count, int(n_plugs) - 1))


def mask_seed(seed: int, fold_id: int, rho: float) -> int:
    """Independent stream for one fold, fraction, and model seed."""
    return int(seed) + 1009 * int(fold_id) + 17 * int(round(float(rho) * 100.0))
