#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot direct_ub parity scatter from tables/parity_pooled.npz (no model re-run).

Requires a run produced after sir_cs_benchmark_direct_ub.py started saving
parity_pooled.npz alongside 09_parity_ground_truth_vs_prediction.png.

Example:
  python scripts/plot_direct_ub_parity_from_npz.py \\
    --run-root outputs/direct_ub_robustness/dct_subsample/measurement_noise_std/v_0p04/runs/noise_0p04_20260423_112415 \\
    --in-place

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import numpy as np

from sir_cs_pipeline_optimized import Config, plot_parity_ground_truth_vs_predictions


def _load_cfg_stub(run_root: str) -> Config:
    path = os.path.join(run_root, "config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing {path}")
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)
    cfg = Config()
    profile = data.get("config_profile", "")
    if not profile:
        raise ValueError("config.json has no config_profile")
    cfg.config_profile = profile  # type: ignore[assignment]
    cfg.run_lfista = bool(data.get("run_lfista", False))
    cfg.dual_cs_solver = bool(data.get("dual_cs_solver", False))
    cfg.save_dir = os.path.abspath(run_root)
    cfg.plots_subdir = "figures"
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description="Replot direct_ub parity from parity_pooled.npz.")
    p.add_argument("--run-root", type=str, required=True, help="Path to .../runs/<run_id>/")
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Write to run-root/figures/09_parity_ground_truth_vs_prediction.png (default).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Custom output PNG path (ignored if --in-place).",
    )
    args = p.parse_args()
    run_root = os.path.abspath(args.run_root.strip())
    npz_path = os.path.join(run_root, "tables", "parity_pooled.npz")
    if not os.path.isfile(npz_path):
        print(f"Missing {npz_path} (re-run benchmark without --no-parity to create it).", file=sys.stderr)
        sys.exit(2)
    raw = np.load(npz_path)
    merged: Dict[str, np.ndarray] = {str(k): np.asarray(raw[k]) for k in raw.files}
    cfg = _load_cfg_stub(run_root)
    if args.in_place or not args.out.strip():
        out_png = os.path.join(run_root, "figures", "09_parity_ground_truth_vs_prediction.png")
    else:
        out_png = os.path.abspath(args.out.strip())
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plot_parity_ground_truth_vs_predictions(cfg, merged, out_png)
    print(f"Wrote {out_png}", flush=True)


if __name__ == "__main__":
    main()
