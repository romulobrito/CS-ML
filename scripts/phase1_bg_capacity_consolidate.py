#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 1 consolidation script for cross-well Vc benchmark.

Reads the per-bg_type production CSVs (DCT basis), produces:
  - bg_capacity_summary.csv: long-format summary with bg_type tag
  - bg_capacity_focus.csv:   focus methods only (ml_only, ae_regression_ub,
    hybrid_lfista_joint), pivoted on bg_type and rho
  - bg_capacity_focus.tex:   LaTeX-friendly version (booktabs, ascii only)
  - figures/p1_rmse_vs_rho_per_bg.png: comparative line plot
  - figures/p1_rmse_bar_per_rho.png:   bar grid

Run from repo root:
  python scripts/phase1_bg_capacity_consolidate.py

ASCII-only.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

RUNS_DIR = os.path.join(
    _REPO_ROOT, "outputs", "cross_well_vc", "direct_ub", "runs"
)
RUN_IDS: Tuple[Tuple[str, str], ...] = (
    ("linear", "prod_bg_linear_dct"),
    ("shallow", "prod_bg_shallow_dct"),
    ("mlp2", "prod_bg_mlp2_dct"),
)
OUT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "cross_well_vc", "direct_ub", "runs", "_aggregate"
)
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

FOCUS_METHODS: Tuple[str, ...] = (
    "ml_only",
    "ae_regression_ub",
    "hybrid_lfista_joint",
)


def _read_summary(run_id: str) -> pd.DataFrame:
    path = os.path.join(RUNS_DIR, run_id, "tables", "summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rows: List[pd.DataFrame] = []
    for bg, run_id in RUN_IDS:
        df = _read_summary(run_id)
        df = df.copy()
        df["bg_type"] = bg
        df["run_id"] = run_id
        rows.append(df)
    long_df = pd.concat(rows, ignore_index=True)
    long_path = os.path.join(TABLES_DIR, "bg_capacity_summary.csv")
    long_df.to_csv(long_path, index=False)
    print("Saved:", long_path)

    focus_df = long_df[long_df["method"].isin(FOCUS_METHODS)].copy()
    focus_path = os.path.join(TABLES_DIR, "bg_capacity_focus.csv")
    focus_df.to_csv(focus_path, index=False)
    print("Saved:", focus_path)

    pivot = focus_df.pivot_table(
        index=["measurement_ratio", "method"],
        columns="bg_type",
        values="rmse_mean",
        aggfunc="mean",
    ).round(4)
    pivot_path = os.path.join(TABLES_DIR, "bg_capacity_pivot_rmse.csv")
    pivot.to_csv(pivot_path)
    print("Saved:", pivot_path)

    tex_path = os.path.join(TABLES_DIR, "bg_capacity_focus.tex")
    with open(tex_path, "w", encoding="ascii") as f:
        f.write("% phase1 bg capacity, RMSE mean (across seeds), DCT basis.\n")
        f.write(pivot.to_latex(float_format="%.4f"))
    print("Saved:", tex_path)

    rhos = sorted(long_df["measurement_ratio"].unique())
    bg_order = [b for b, _ in RUN_IDS]
    method_colors = {
        "ml_only": "#1f77b4",
        "ae_regression_ub": "#2ca02c",
        "hybrid_lfista_joint": "#d62728",
    }
    method_label = {
        "ml_only": "ml_only",
        "ae_regression_ub": "ae_regression_ub",
        "hybrid_lfista_joint": "SIR-CS (hybrid_lfista_joint)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, bg in zip(axes, bg_order):
        sub = long_df[(long_df["bg_type"] == bg) & long_df["method"].isin(FOCUS_METHODS)]
        for m in FOCUS_METHODS:
            r = sub[sub["method"] == m].sort_values("measurement_ratio")
            ax.errorbar(
                r["measurement_ratio"].values,
                r["rmse_mean"].values,
                yerr=r["rmse_std_across_seeds"].fillna(0.0).values,
                marker="o",
                linewidth=1.6,
                capsize=3,
                color=method_colors[m],
                label=method_label[m],
            )
        ax.set_title("LFISTA bg = {}".format(bg))
        ax.set_xlabel("measurement_ratio rho")
        ax.set_xticks(rhos)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE on test well (mean across seeds)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Phase 1 cross-well Vc | bg capacity vs SIR-CS performance (DCT, F03-4 held out)",
        y=1.02,
    )
    p1 = os.path.join(FIGURES_DIR, "p1_rmse_vs_rho_per_bg.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p1)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    width = 0.22
    x = np.arange(len(rhos), dtype=np.float64)
    for k, bg in enumerate(bg_order):
        sub = long_df[(long_df["bg_type"] == bg) & (long_df["method"] == "hybrid_lfista_joint")]
        sub = sub.sort_values("measurement_ratio")
        ax.bar(
            x + (k - 1) * width,
            sub["rmse_mean"].values,
            width,
            yerr=sub["rmse_std_across_seeds"].fillna(0.0).values,
            capsize=3,
            label="bg={}".format(bg),
        )
    base_sub = long_df[(long_df["bg_type"] == "mlp2") & (long_df["method"] == "ml_only")]
    base_sub = base_sub.sort_values("measurement_ratio")
    ax.plot(
        x,
        base_sub["rmse_mean"].values,
        "k--",
        linewidth=1.5,
        label="ml_only baseline",
    )
    ae_sub = long_df[(long_df["bg_type"] == "mlp2") & (long_df["method"] == "ae_regression_ub")]
    ae_sub = ae_sub.sort_values("measurement_ratio")
    ax.plot(
        x,
        ae_sub["rmse_mean"].values,
        "g-.",
        linewidth=1.5,
        label="ae_regression_ub baseline",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rhos])
    ax.set_xlabel("measurement_ratio rho")
    ax.set_ylabel("RMSE on test well")
    ax.set_title(
        "SIR-CS hybrid_lfista_joint RMSE per bg_type vs rho (DCT, F03-4)"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    p2 = os.path.join(FIGURES_DIR, "p1_lfista_rmse_bar_per_rho.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p2)

    print("\n=== Phase 1 ranking by mean RMSE across rhos (lower better) ===")
    rank_rows = []
    for bg in bg_order:
        for m in FOCUS_METHODS:
            sub = long_df[(long_df["bg_type"] == bg) & (long_df["method"] == m)]
            if sub.empty:
                continue
            mean_rmse = float(sub["rmse_mean"].mean())
            mean_std = float(sub["rmse_std_across_seeds"].fillna(0.0).mean())
            rank_rows.append({
                "bg_type": bg,
                "method": m,
                "rmse_mean_over_rhos": mean_rmse,
                "rmse_std_avg": mean_std,
            })
    rank_df = pd.DataFrame(rank_rows).sort_values(
        ["method", "rmse_mean_over_rhos"]
    )
    print(rank_df.to_string(index=False))
    rank_path = os.path.join(TABLES_DIR, "bg_capacity_rank.csv")
    rank_df.to_csv(rank_path, index=False)
    print("Saved:", rank_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
