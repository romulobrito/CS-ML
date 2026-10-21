#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Join the 161 laboratory plugs to extra well-log channels.

Reads the local plug workbook and the quad DLIS already stored under
data/1254. Writes one CSV next to those files. Does not print measurements.
ASCII only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dlisio
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DATA_DIR,
    DEPTH_MATCH_MAX_M,
    PASS_FILES,
    PLUG_XLS,
    WELL_ANP,
    assert_candidate_inventory,
    load_candidate_plugs,
    parse_float,
)

CHANNELS: Tuple[str, ...] = (
    "GR",
    "RHOB",
    "NPHI",
    "RT10",
    "RT90",
    "DT",
    "PE",
    "CALI",
    "DRHO",
)
OUT_CSV = DATA_DIR / "poco_1254_plugs_channels.csv"


def _column_containing(columns: Sequence[str], token: str) -> str:
    """Return the single header that contains token, or an empty string."""
    matches = [name for name in columns if token in name]
    if len(matches) != 1:
        return ""
    return matches[0]


def load_plug_identity(path: Path) -> pd.DataFrame:
    """Add sample id and laboratory porosity onto the 161-plug inventory."""
    plugs = load_candidate_plugs(path)
    assert_candidate_inventory(plugs)
    frame = pd.read_excel(path, sheet_name="Sheet0")
    columns = [str(name) for name in frame.columns]
    sample_col = _column_containing(columns, "Nr.")
    porosity_col = _column_containing(columns, "Porosidade")
    if sample_col == "" or porosity_col == "":
        raise ValueError("Plug workbook is missing sample id or porosity.")
    sample_ids: List[str] = []
    porosity: List[float] = []
    for excel_row in plugs["excel_row"].tolist():
        record = frame.iloc[int(excel_row)]
        sample_ids.append(str(record[sample_col]).strip())
        porosity.append(parse_float(record[porosity_col]))
    plugs = plugs.copy()
    plugs.insert(1, "sample_id", sample_ids)
    plugs.insert(2, "well", WELL_ANP)
    plugs["porosity_pct"] = porosity
    return plugs


def _scalar_curve(frame: dlisio.dlis.Frame, name: str) -> Optional[np.ndarray]:
    """Return one 1-D curve, or None when the channel is absent."""
    for channel in frame.channels:
        if str(channel.name) != name:
            continue
        values = np.asarray(channel.curves(), dtype=np.float64)
        if values.ndim != 1:
            return None
        return values
    return None


def _best_frame(path: Path) -> Dict[str, np.ndarray]:
    """Pick the frame that contains depth plus every requested channel."""
    logical_files = dlisio.dlis.load(str(path))
    chosen: Optional[Dict[str, np.ndarray]] = None
    try:
        for logical in logical_files:
            for frame in logical.frames:
                depth = _scalar_curve(frame, "TDEP")
                if depth is None:
                    continue
                curves: Dict[str, np.ndarray] = {"TDEP": depth}
                complete = True
                for name in CHANNELS:
                    values = _scalar_curve(frame, name)
                    if values is None or int(values.shape[0]) != int(depth.shape[0]):
                        complete = False
                        break
                    curves[name] = values
                if not complete:
                    continue
                finite = int(np.sum(np.isfinite(depth)))
                if chosen is None or finite > int(np.sum(np.isfinite(chosen["TDEP"]))):
                    chosen = curves
    finally:
        logical_files.close()
    if chosen is None:
        raise ValueError("No frame with all channels in {0}.".format(path.name))
    return chosen


def _nearest_rows(depth: np.ndarray, plug_depth: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Index of the nearest finite depth sample and the absolute gap in meters."""
    finite = np.isfinite(depth)
    order = np.argsort(depth[finite])
    depth_sorted = depth[finite][order]
    index_sorted = np.flatnonzero(finite)[order]
    nearest = np.searchsorted(depth_sorted, plug_depth, side="left")
    rows = np.full(plug_depth.shape[0], -1, dtype=np.int64)
    deltas = np.full(plug_depth.shape[0], np.inf, dtype=np.float64)
    for plug_id, target in enumerate(plug_depth.tolist()):
        left = int(nearest[plug_id])
        candidates = []
        if left > 0:
            candidates.append(left - 1)
        if left < int(depth_sorted.shape[0]):
            candidates.append(left)
        if not candidates:
            continue
        local = min(candidates, key=lambda pos: abs(float(depth_sorted[pos]) - target))
        rows[plug_id] = int(index_sorted[local])
        deltas[plug_id] = abs(float(depth_sorted[local]) - target)
    return rows, deltas


def align_pass(plugs: pd.DataFrame, pass_name: str, path: Path) -> pd.DataFrame:
    """Sample every requested channel at the nearest log depth."""
    curves = _best_frame(path)
    rows, deltas = _nearest_rows(
        curves["TDEP"],
        plugs["depth_m"].to_numpy(dtype=np.float64),
    )
    table = plugs.copy()
    table.insert(0, "pass_name", pass_name)
    table["log_depth_m"] = np.nan
    table["depth_delta_m"] = deltas
    for name in CHANNELS:
        table[name] = np.nan
    for plug_id, row in enumerate(rows.tolist()):
        if row < 0 or float(deltas[plug_id]) > DEPTH_MATCH_MAX_M:
            continue
        table.at[plug_id, "log_depth_m"] = float(curves["TDEP"][row])
        for name in CHANNELS:
            table.at[plug_id, name] = float(curves[name][row])
    return table


def main() -> None:
    """Write the joined plug table and print counts only."""
    if not PLUG_XLS.is_file():
        raise FileNotFoundError(str(PLUG_XLS))
    plugs = load_plug_identity(PLUG_XLS)
    tables = []
    for pass_name, path in PASS_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(str(path))
        tables.append(align_pass(plugs, pass_name, path))
    # Prefer the pass that leaves fewer empty channel cells.
    def _empty_cells(frame: pd.DataFrame) -> int:
        return int(frame.loc[:, list(CHANNELS)].isna().sum().sum())

    tables.sort(key=_empty_cells)
    chosen = tables[0]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(OUT_CSV, index=False)
    matched = int(np.isfinite(chosen["log_depth_m"].to_numpy(dtype=np.float64)).sum())
    print("plugs", int(chosen.shape[0]))
    print("pass", str(chosen["pass_name"].iloc[0]))
    print("matched_within_m", DEPTH_MATCH_MAX_M, matched)
    for name in CHANNELS:
        finite = int(np.isfinite(pd.to_numeric(chosen[name], errors="coerce")).sum())
        print("finite", name, finite)
    print("wrote", OUT_CSV)


if __name__ == "__main__":
    main()
