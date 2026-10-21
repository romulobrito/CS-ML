#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot RMSE from merge_direct_ub_residual_k_robustness.py output.

Default: one method, RMSE vs measurement_ratio, one line per residual_k.

Optional: at a fixed rho, RMSE vs residual_k for several methods (comparison).

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _pick_yerr(g: pd.DataFrame) -> Optional[np.ndarray]:
    if "rmse_ci95_half" in g.columns:
        return g["rmse_ci95_half"].to_numpy()
    if "rmse_std_across_seeds" in g.columns:
        return g["rmse_std_across_seeds"].to_numpy()
    return None


def plot_joint_vs_rho(df: pd.DataFrame, method: str, out_path: str) -> None:
    need = {"measurement_ratio", "method", "residual_k", "rmse_mean"}
    if not need.issubset(df.columns):
        print(f"Expected columns {sorted(need)} in merged CSV.", file=sys.stderr)
        sys.exit(2)
    sub = df[df["method"] == method].copy()
    if sub.empty:
        print(f"No rows for method={method!r}", file=sys.stderr)
        sys.exit(2)
    ks = sorted(int(x) for x in sub["residual_k"].unique())
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for k in ks:
        g = sub[sub["residual_k"] == k].sort_values("measurement_ratio")
        x = g["measurement_ratio"].to_numpy()
        y = g["rmse_mean"].to_numpy()
        err = _pick_yerr(g)
        label = f"k={k}"
        if err is not None:
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3.0, label=label, linewidth=1.2)
        else:
            ax.plot(x, y, marker="o", label=label, linewidth=1.2)
    ax.set_xlabel("measurement_ratio")
    ax.set_ylabel("test RMSE (mean across seeds)")
    ax.set_title(f"{method}: DCT + subsample M, residual_k sweep")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def plot_methods_vs_k_at_rho(
    df: pd.DataFrame,
    rho: float,
    methods: list[str],
    out_path: str,
) -> None:
    need = {"measurement_ratio", "method", "residual_k", "rmse_mean"}
    if not need.issubset(df.columns):
        print(f"Expected columns {sorted(need)} in merged CSV.", file=sys.stderr)
        sys.exit(2)
    sub = df[np.isclose(df["measurement_ratio"].to_numpy(), float(rho), rtol=0.0, atol=1e-9)].copy()
    if sub.empty:
        print(f"No rows at measurement_ratio={rho}.", file=sys.stderr)
        sys.exit(2)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for m in methods:
        g = sub[sub["method"] == m].sort_values("residual_k")
        if g.empty:
            print(f"Warning: no rows for method={m!r} at rho={rho}.", file=sys.stderr)
            continue
        x = g["residual_k"].to_numpy(dtype=int)
        y = g["rmse_mean"].to_numpy()
        err = _pick_yerr(g)
        if err is not None:
            ax.errorbar(x, y, yerr=err, marker="o", capsize=3.0, label=m, linewidth=1.2)
        else:
            ax.plot(x, y, marker="o", label=m, linewidth=1.2)
    ax.set_xlabel("residual_k")
    ax.set_ylabel("test RMSE (mean across seeds)")
    ax.set_title(f"RMSE vs k at measurement_ratio={rho:g}")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot direct_ub residual_k robustness from merged CSV.")
    p.add_argument("--merged", type=str, required=True, help="Long-format merged CSV path.")
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Output PNG for RMSE vs measurement_ratio (one line per k).",
    )
    p.add_argument(
        "--method",
        type=str,
        default="hybrid_lfista_joint",
        help="Method for the vs-rho panel (default: hybrid_lfista_joint).",
    )
    p.add_argument(
        "--at-rho",
        type=float,
        default=None,
        help="If set, also write RMSE vs residual_k at this measurement_ratio.",
    )
    p.add_argument(
        "--out-at-rho",
        type=str,
        default="",
        help="Output PNG for vs-k panel (required if --at-rho is set).",
    )
    p.add_argument(
        "--methods-at-rho",
        type=str,
        default="hybrid_lfista_joint,mlp_concat_ub,pca_regression_ub",
        help="Comma-separated methods for the vs-k panel.",
    )
    args = p.parse_args()
    path = args.merged.strip()
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Missing file: {path}", file=sys.stderr)
        sys.exit(2)
    out_vs_rho = args.out.strip()
    if out_vs_rho:
        plot_joint_vs_rho(df, args.method.strip(), out_vs_rho)
    if args.at_rho is not None:
        out_k = args.out_at_rho.strip()
        if not out_k:
            print("--out-at-rho is required when --at-rho is set.", file=sys.stderr)
            sys.exit(2)
        methods = [x.strip() for x in str(args.methods_at_rho).split(",") if x.strip()]
        plot_methods_vs_k_at_rho(df, float(args.at_rho), methods, out_k)
    if not out_vs_rho and args.at_rho is None:
        print("Provide --out and/or --at-rho with --out-at-rho.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
