#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RMSE (mean across seeds) vs measurement_ratio for hybrid_lfista_joint
from a merged long CSV (merge_direct_ub_noise_robustness.py output).

One panel, one line per measurement_noise_std value; optional error bars from
rmse_ci95_half if present else rmse_std_across_seeds.

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Plot joint RMSE vs rho across noise arms.")
    p.add_argument("--merged", type=str, required=True, help="Long-format merged CSV path.")
    p.add_argument("--out", type=str, required=True, help="Output PNG path (parent dirs created).")
    p.add_argument(
        "--method",
        type=str,
        default="hybrid_lfista_joint",
        help="Method column value to plot (default: hybrid_lfista_joint).",
    )
    args = p.parse_args()
    path = args.merged.strip()
    out_path = args.out.strip()
    method = args.method.strip()
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Missing file: {path}", file=sys.stderr)
        sys.exit(2)
    need = {"measurement_ratio", "method", "measurement_noise_std", "rmse_mean"}
    if not need.issubset(df.columns):
        print(f"Expected columns {sorted(need)} in {path}", file=sys.stderr)
        sys.exit(2)
    sub = df[df["method"] == method].copy()
    if sub.empty:
        print(f"No rows for method={method!r}", file=sys.stderr)
        sys.exit(2)
    noises = sorted(sub["measurement_noise_std"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for n in noises:
        g = sub[sub["measurement_noise_std"] == n].sort_values("measurement_ratio")
        x = g["measurement_ratio"].to_numpy()
        y = g["rmse_mean"].to_numpy()
        if "rmse_ci95_half" in g.columns:
            err = g["rmse_ci95_half"].to_numpy()
        elif "rmse_std_across_seeds" in g.columns:
            err = g["rmse_std_across_seeds"].to_numpy()
        else:
            err = None
        label = f"noise_std={n:g}"
        if err is not None:
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3.0, label=label, linewidth=1.2)
        else:
            ax.plot(x, y, marker="o", label=label, linewidth=1.2)
    ax.set_xlabel("measurement_ratio")
    ax.set_ylabel("test RMSE (mean across seeds)")
    ax.set_title(f"{method}: DCT + subsample M, measurement noise sweep")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
