#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate Sec. 4 EDA figures and main depth-profile figures for the CLP-CSGM paper.

Writes PNGs directly to paper_clp_csgm/figures/ (same filenames used in LaTeX).

EDA figures (fast, data-only):
  eda_target_summary.png
  eda_crosswell_vc_distribution.png
  eda_f03_gr_porosity_depth.png
  eda_channel_target_correlations.png

Depth-profile figures (re-run single benchmark cells):
  10_depth_profile_porosity.png
  10_depth_profile_crosswell_vc.png

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PAPER_FIG_DIR = os.path.join(REPO_ROOT, "paper_clp_csgm", "figures")


def _run(cmd: List[str]) -> None:
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def regenerate_eda_figures() -> None:
    """Regenerate EDA PNGs via clp_csgm_paper_assets (tables + EDA plots)."""
    script = os.path.join(REPO_ROOT, "scripts", "clp_csgm_paper_assets.py")
    _run([sys.executable, script])


def _copy_depth_profile(src_run_fig: str, paper_name: str) -> str:
    if not os.path.isfile(src_run_fig):
        raise FileNotFoundError("Missing depth-profile source: " + src_run_fig)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    dst = os.path.join(PAPER_FIG_DIR, paper_name)
    shutil.copy2(src_run_fig, dst)
    print("Copied " + dst, flush=True)
    return dst


def regenerate_f03_depth_profile(seed: int, rho: float) -> str:
    """Run F03 GR-only benchmark cell and copy depth-profile figure to the paper."""
    run_root = os.path.join(REPO_ROOT, "outputs", "paper_regen", "f03_depth_profile")
    _run(
        [
            sys.executable,
            "sir_cs_benchmark_real_well_direct_ub.py",
            "--data-path",
            "data/F03-4_AC+GR+Porosity.txt",
            "--u-channels",
            "gr",
            "--seeds",
            str(int(seed)),
            "--rhos",
            "{:.2f}".format(float(rho)),
            "--no-lfista",
            "--run-csgm-m2",
            "--base-dir",
            run_root,
            "--run-id",
            "seed{}_rho{}".format(int(seed), str(rho).replace(".", "p")),
        ]
    )
    run_dirs = sorted(
        [
            os.path.join(run_root, "runs", d)
            for d in os.listdir(os.path.join(run_root, "runs"))
        ],
        key=os.path.getmtime,
    )
    if not run_dirs:
        raise RuntimeError("F03 depth-profile run produced no output directory.")
    src = os.path.join(run_dirs[-1], "figures", "10_depth_profile_porosity.png")
    return _copy_depth_profile(src, "10_depth_profile_porosity.png")


def regenerate_crosswell_depth_profile(seed: int, rho: float, step: int) -> str:
    """Run cross-well benchmark cell and copy depth-profile figure to the paper."""
    run_root = os.path.join(REPO_ROOT, "outputs", "paper_regen", "crosswell_depth_profile")
    _run(
        [
            sys.executable,
            "sir_cs_benchmark_multi_well_vc.py",
            "--train-path",
            "data/F02-1,F03-2,F06-1_6logs_30dB.txt",
            "--test-path",
            "data/F03-4_6logs_30dB.txt",
            "--channels",
            "sonic,rhob,ai,vp",
            "--target",
            "vc",
            "--step",
            str(int(step)),
            "--seeds",
            str(int(seed)),
            "--rhos",
            "{:.2f}".format(float(rho)),
            "--no-lfista",
            "--run-csgm-m2",
            "--base-dir",
            run_root,
            "--run-id",
            "step{}_seed{}_rho{}".format(int(step), int(seed), str(rho).replace(".", "p")),
        ]
    )
    run_dirs = sorted(
        [
            os.path.join(run_root, "runs", d)
            for d in os.listdir(os.path.join(run_root, "runs"))
        ],
        key=os.path.getmtime,
    )
    if not run_dirs:
        raise RuntimeError("Cross-well depth-profile run produced no output directory.")
    src = os.path.join(run_dirs[-1], "figures", "10_depth_profile_target.png")
    return _copy_depth_profile(src, "10_depth_profile_crosswell_vc.png")


def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate paper EDA and depth-profile figures.")
    p.add_argument(
        "--eda-only",
        action="store_true",
        help="Regenerate only the four Sec. 4 EDA figures (skip depth profiles).",
    )
    p.add_argument(
        "--depth-only",
        action="store_true",
        help="Regenerate only the two depth-profile figures (skip EDA).",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--rho", type=float, default=0.20)
    p.add_argument("--cross-step", type=int, default=16)
    args = p.parse_args()

    do_eda = not bool(args.depth_only)
    do_depth = not bool(args.eda_only)

    if do_eda:
        regenerate_eda_figures()
    if do_depth:
        regenerate_f03_depth_profile(seed=int(args.seed), rho=float(args.rho))
        regenerate_crosswell_depth_profile(
            seed=int(args.seed),
            rho=float(args.rho),
            step=int(args.cross_step),
        )
    print("Done. Figures are under " + PAPER_FIG_DIR, flush=True)


if __name__ == "__main__":
    main()
