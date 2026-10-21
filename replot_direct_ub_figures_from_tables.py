#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-generate direct_ub benchmark comparison PNGs from existing tables/ CSVs.

Uses current METHOD_COLORS and method_order_for_cfg in sir_cs_pipeline_optimized
(no re-training). Typical use: fix legend colors after updating METHOD_COLORS.

Examples:
  python replot_direct_ub_figures_from_tables.py \\
    --run-root outputs/direct_ub_benchmark/runs/20260422_102236

  python replot_direct_ub_figures_from_tables.py \\
    --run-root outputs/direct_ub_benchmark/runs/20260422_102236 --in-place
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import pandas as pd

from sir_cs_pipeline_optimized import Config, save_all_comparison_plots


def _load_cfg_stub(run_root: str) -> Config:
    """Minimal Config for plotting: profile + flags from run config.json."""
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
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replot direct_ub figures from summary.csv (same layout, current colors)."
    )
    p.add_argument(
        "--run-root",
        type=str,
        required=True,
        help="Absolute or relative path to .../runs/<run_id>/",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Write PNGs to run-root/figures/ (replaces same filenames).",
    )
    p.add_argument(
        "--out-subdir",
        type=str,
        default="",
        help=(
            "Subdirectory under run-root for PNGs (e.g. figures_colors_fixed). "
            "Ignored if --in-place. Default when omitted: figures_colors_fixed."
        ),
    )
    args = p.parse_args()

    run_root = os.path.abspath(args.run_root.strip())
    tables = os.path.join(run_root, "tables")
    summary_path = os.path.join(tables, "summary.csv")
    per_seed_path = os.path.join(tables, "summary_by_seed.csv")
    if not os.path.isdir(run_root):
        print(f"Not a directory: {run_root}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(summary_path) or not os.path.isfile(per_seed_path):
        print(f"Need {summary_path} and {per_seed_path}", file=sys.stderr)
        sys.exit(2)

    cfg = _load_cfg_stub(run_root)
    if args.in_place:
        cfg.plots_subdir = "figures"
    else:
        sub = args.out_subdir.strip().strip("/").strip("\\")
        cfg.plots_subdir = sub if sub else "figures_colors_fixed"

    summary = pd.read_csv(summary_path)
    per_seed = pd.read_csv(per_seed_path)

    paths = save_all_comparison_plots(cfg, summary, per_seed)
    out_dir = os.path.join(cfg.save_dir, cfg.plots_subdir)
    print(f"Wrote {len(paths)} files under {out_dir}", flush=True)
    for q in sorted(paths):
        print(q, flush=True)


if __name__ == "__main__":
    main()
