#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align positive VH1 plugs to one quad DLIS pass.

The pass is chosen from depth coverage and finite log curves only.
Permeability is not used in that choice. ASCII only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dlisio
import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    DATA_DIR,
    DEPTH_MATCH_MAX_M,
    OUT_DIR,
    PASS_FILES,
    PLUG_XLS,
    RESISTIVITY_BY_PASS,
    SCALAR_CURVES,
    assert_candidate_inventory,
    load_candidate_plugs,
)


def _scalar_curve(frame: dlisio.dlis.Frame, name: str) -> Optional[np.ndarray]:
    """Return one 1-D curve, or None when the channel is absent or not scalar."""
    for channel in frame.channels:
        if str(channel.name) != name:
            continue
        values = np.asarray(channel.curves(), dtype=np.float64)
        if values.ndim != 1:
            return None
        return values
    return None


def _score_frame(
    frame: dlisio.dlis.Frame,
    plug_depth: np.ndarray,
    resistivity_name: str,
) -> Optional[Dict[str, object]]:
    """Count plugs with a nearby finite GR, RHOB, NPHI, and resistivity sample."""
    depth = _scalar_curve(frame, "TDEP")
    if depth is None:
        return None
    curves: Dict[str, np.ndarray] = {"TDEP": depth}
    for name in SCALAR_CURVES:
        values = _scalar_curve(frame, name)
        if values is None or int(values.shape[0]) != int(depth.shape[0]):
            return None
        curves[name] = values
    resistivity = _scalar_curve(frame, resistivity_name)
    if resistivity is None or int(resistivity.shape[0]) != int(depth.shape[0]):
        return None
    curves[resistivity_name] = resistivity

    finite_depth = np.isfinite(depth)
    if int(np.sum(finite_depth)) < 3:
        return None
    order = np.argsort(depth[finite_depth])
    depth_sorted = depth[finite_depth][order]
    index_sorted = np.flatnonzero(finite_depth)[order]
    nearest = np.searchsorted(depth_sorted, plug_depth, side="left")
    chosen: List[int] = []
    deltas: List[float] = []
    for plug_id, target in enumerate(plug_depth.tolist()):
        left = int(nearest[plug_id])
        candidates = []
        if left > 0:
            candidates.append(left - 1)
        if left < int(depth_sorted.shape[0]):
            candidates.append(left)
        if not candidates:
            chosen.append(-1)
            deltas.append(float("inf"))
            continue
        local = min(candidates, key=lambda pos: abs(float(depth_sorted[pos]) - target))
        delta = abs(float(depth_sorted[local]) - target)
        chosen.append(int(index_sorted[local]))
        deltas.append(delta)

    matched = 0
    for plug_id, row in enumerate(chosen):
        if row < 0 or deltas[plug_id] > DEPTH_MATCH_MAX_M:
            continue
        finite_logs = all(np.isfinite(curves[name][row]) for name in SCALAR_CURVES)
        finite_rt = bool(np.isfinite(resistivity[row]))
        if finite_logs and finite_rt:
            matched += 1
    matched_deltas = [
        deltas[plug_id]
        for plug_id, row in enumerate(chosen)
        if row >= 0 and deltas[plug_id] <= DEPTH_MATCH_MAX_M
    ]
    median_delta = float(np.median(np.asarray(matched_deltas, dtype=np.float64))) if matched_deltas else float("inf")
    return {
        "frame_name": str(frame.name),
        "n_depth": int(np.sum(finite_depth)),
        "depth_min_m": float(np.min(depth[finite_depth])),
        "depth_max_m": float(np.max(depth[finite_depth])),
        "n_matched": int(matched),
        "median_abs_delta_m": median_delta,
        "curves": curves,
        "row_for_plug": chosen,
        "delta_for_plug": deltas,
        "resistivity_name": resistivity_name,
    }


def _best_frame(path: Path, plug_depth: np.ndarray, resistivity_name: str) -> Dict[str, object]:
    """Pick the frame in one DLIS file with the most matched plugs."""
    logical_files = dlisio.dlis.load(str(path))
    scored: List[Dict[str, object]] = []
    try:
        for logical in logical_files:
            for frame in logical.frames:
                result = _score_frame(frame, plug_depth, resistivity_name)
                if result is not None:
                    scored.append(result)
    finally:
        logical_files.close()
    if not scored:
        raise ValueError("No usable frame in {0}.".format(path.name))
    scored.sort(
        key=lambda item: (
            -int(item["n_matched"]),
            float(item["median_abs_delta_m"]),
            str(item["frame_name"]),
        )
    )
    return scored[0]


def choose_pass(plugs: pd.DataFrame) -> Tuple[str, Dict[str, object], pd.DataFrame]:
    """Choose the pass with more plugs inside the depth tolerance and finite curves."""
    plug_depth = plugs["depth_m"].to_numpy(dtype=np.float64)
    ranking: List[Dict[str, object]] = []
    best_name = ""
    best: Optional[Dict[str, object]] = None
    for pass_name, path in PASS_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(str(path))
        frame = _best_frame(path, plug_depth, RESISTIVITY_BY_PASS[pass_name])
        ranking.append(
            {
                "pass_name": pass_name,
                "frame_name": frame["frame_name"],
                "n_matched": frame["n_matched"],
                "median_abs_delta_m": frame["median_abs_delta_m"],
                "depth_min_m": frame["depth_min_m"],
                "depth_max_m": frame["depth_max_m"],
                "n_depth": frame["n_depth"],
                "resistivity": frame["resistivity_name"],
            }
        )
        if best is None:
            best_name = pass_name
            best = frame
            continue
        better_count = int(frame["n_matched"]) > int(best["n_matched"])
        same_count = int(frame["n_matched"]) == int(best["n_matched"])
        closer = float(frame["median_abs_delta_m"]) < float(best["median_abs_delta_m"])
        if better_count or (same_count and closer):
            best_name = pass_name
            best = frame
    if best is None:
        raise RuntimeError("Pass selection failed.")
    return best_name, best, pd.DataFrame(ranking)


def build_aligned_table(
    plugs: pd.DataFrame,
    pass_name: str,
    frame: Dict[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Keep plugs whose nearest sample is inside the tolerance with finite curves."""
    curves = frame["curves"]
    if not isinstance(curves, dict):
        raise TypeError("Frame curves are not a mapping.")
    resistivity_name = str(frame["resistivity_name"])
    rows_for_plug = frame["row_for_plug"]
    deltas = frame["delta_for_plug"]
    if not isinstance(rows_for_plug, list) or not isinstance(deltas, list):
        raise TypeError("Frame match lists are missing.")
    kept: List[Dict[str, float]] = []
    dropped: List[Dict[str, float]] = []
    for plug_id, plug in plugs.iterrows():
        row = int(rows_for_plug[int(plug_id)])
        delta = float(deltas[int(plug_id)])
        record = {
            "plug_index": int(plug["plug_index"]),
            "excel_row": float(plug["excel_row"]),
            "depth_m": float(plug["depth_m"]),
            "k_md": float(plug["k_md"]),
            "log10_k": float(plug["log10_k"]),
            "depth_delta_m": delta,
        }
        if row < 0 or delta > DEPTH_MATCH_MAX_M:
            dropped.append(record)
            continue
        log_values = {
            "gr": float(curves["GR"][row]),
            "rhob": float(curves["RHOB"][row]),
            "nphi": float(curves["NPHI"][row]),
            "rt": float(curves[resistivity_name][row]),
            "log_depth_m": float(curves["TDEP"][row]),
        }
        if not all(np.isfinite(value) for value in log_values.values()):
            dropped.append(record)
            continue
        record.update(log_values)
        kept.append(record)
    if not kept:
        raise ValueError("Every plug fell outside the log match.")
    aligned = pd.DataFrame(kept).sort_values("depth_m").reset_index(drop=True)
    aligned["plug_index"] = np.arange(len(aligned), dtype=np.int64)
    aligned.insert(0, "pass_name", pass_name)
    return aligned, pd.DataFrame(dropped)


def main() -> None:
    """Write the aligned plug-log table and the pass-coverage ranking."""
    if not PLUG_XLS.is_file():
        raise FileNotFoundError(str(PLUG_XLS))
    plugs = load_candidate_plugs(PLUG_XLS)
    assert_candidate_inventory(plugs)
    pass_name, frame, ranking = choose_pass(plugs)
    aligned, dropped = build_aligned_table(plugs, pass_name, frame)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    aligned_path = table_dir / "plug_log_aligned.csv"
    dropped_path = table_dir / "dropped_plugs.csv"
    ranking_path = table_dir / "pass_coverage.csv"
    aligned.to_csv(aligned_path, index=False)
    dropped.to_csv(dropped_path, index=False)
    ranking.to_csv(ranking_path, index=False)
    selection = {
        "pass_name": pass_name,
        "frame_name": str(frame["frame_name"]),
        "resistivity": str(frame["resistivity_name"]),
        "depth_match_max_m": DEPTH_MATCH_MAX_M,
        "n_candidates": int(plugs.shape[0]),
        "n_kept": int(aligned.shape[0]),
        "n_dropped": int(dropped.shape[0]),
        "candidate_depth_min_m": float(plugs["depth_m"].min()),
        "candidate_depth_max_m": float(plugs["depth_m"].max()),
        "kept_depth_min_m": float(aligned["depth_m"].min()),
        "kept_depth_max_m": float(aligned["depth_m"].max()),
        "tie_break": "more matched plugs, then smaller median absolute depth gap",
        "permeability_used_in_pass_choice": False,
        "data_dir": str(DATA_DIR),
    }
    (OUT_DIR / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "pass",
        pass_name,
        "kept",
        int(aligned.shape[0]),
        "dropped",
        int(dropped.shape[0]),
    )
    print(ranking.to_string(index=False))
    print("wrote", aligned_path)


if __name__ == "__main__":
    main()
