#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export depth-profile figure: RF vs CLP prior RF vs CLP residual RF (Well 861).

Reads OOF pointwise CSVs from two plug-fixed runs and writes a figure under
methods_comparison/latex/figures/ for the Etapa 1 report.
ASCII-only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ML_ROOT = REPO_ROOT / "methods_comparison" / "data" / "processed" / "ml_runs"
FIG_DIR = REPO_ROOT / "methods_comparison" / "latex" / "figures"

OBS_COLOR = "#212121"
RF_COLOR = "#6A1B9A"
PRIOR_COLOR = "#E65100"
RESIDUAL_COLOR = "#00897B"
PLUG_EDGE = "#2E7D32"

DEFAULT_PRIOR_CSV = (
    ML_ROOT
    / "clp_861/phi_lab/plug_sparse_b/runs/plug_fixed_rf_prior_20260617/tables"
    / "oof_profile_predictions.csv"
)
DEFAULT_RESIDUAL_CSV = (
    ML_ROOT
    / "clp_861/phi_lab/plug_sparse_b/runs/plug_fixed_rf_residual_20260617/tables"
    / "oof_profile_predictions.csv"
)


def load_merged_oof(prior_csv: Path, residual_csv: Path) -> pd.DataFrame:
    """Merge RF, CLP prior and CLP residual OOF columns on row_index."""
    prior = pd.read_csv(prior_csv)
    resid = pd.read_csv(residual_csv)
    cols_prior = [c for c in prior.columns if c.startswith("clp_oof_")]
    cols_resid = [c for c in resid.columns if c.startswith("clp_oof_")]
    if len(cols_prior) != 1 or len(cols_resid) != 1:
        raise ValueError("Expected one clp_oof_* column per run CSV.")
    merged = prior[
        ["row_index", "depth_m", "phi_lab", "fold_id_oof", "rf_oof"]
    ].merge(
        prior[["row_index", cols_prior[0]]].rename(
            columns={cols_prior[0]: "clp_prior_rf"}
        ),
        on="row_index",
        how="inner",
    )
    merged = merged.merge(
        resid[["row_index", cols_resid[0]]].rename(
            columns={cols_resid[0]: "clp_residual_rf"}
        ),
        on="row_index",
        how="inner",
    )
    return merged.sort_values("depth_m").reset_index(drop=True)


def plot_three_method_panels(df: pd.DataFrame, save_path: Path, seed: int) -> None:
    """Three side-by-side depth panels: RF, CLP prior RF, CLP residual RF."""
    depth = df["depth_m"].to_numpy(dtype=np.float64)
    obs = df["phi_lab"].to_numpy(dtype=np.float64)
    rf = df["rf_oof"].to_numpy(dtype=np.float64)
    clp_a = df["clp_prior_rf"].to_numpy(dtype=np.float64)
    clp_b = df["clp_residual_rf"].to_numpy(dtype=np.float64)

    fold_bounds = []
    for fold_id in sorted(df["fold_id_oof"].unique()):
        sub = df.loc[df["fold_id_oof"] == fold_id]
        fold_bounds.append(float(sub["depth_m"].max()))

    panels = [
        ("RF pontual", rf, RF_COLOR),
        ("CLP prior RF (variante A)", clp_a, PRIOR_COLOR),
        ("CLP residual RF (variante B)", clp_b, RESIDUAL_COLOR),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 7.0), sharey=True)
    for ax, (title, pred, color) in zip(axes, panels):
        m_obs = np.isfinite(obs)
        m_pred = np.isfinite(pred)
        ax.plot(
            obs[m_obs],
            depth[m_obs],
            color="#9E9E9E",
            linewidth=1.2,
            linestyle="--",
            label="observado",
            zorder=2,
        )
        ax.plot(
            pred[m_pred],
            depth[m_pred],
            color=color,
            linewidth=2.0,
            label=title,
            zorder=3,
        )
        for zb in fold_bounds[:-1]:
            ax.axhline(zb, color="#BDBDBD", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_xlabel(r"$\phi_{\mathrm{lab}}$ (pu)")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        ax.invert_yaxis()

    axes[0].set_ylabel("profundidade (m)")
    fig.suptitle(
        "861 MOGNO: perfis OOF por bloco (seed={})".format(int(seed)),
        fontsize=12,
        y=1.01,
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_three_method_overlay(df: pd.DataFrame, save_path: Path, seed: int) -> None:
    """Single depth panel with RF, both CLP variants and observed."""
    depth = df["depth_m"].to_numpy(dtype=np.float64)
    obs = df["phi_lab"].to_numpy(dtype=np.float64)
    rf = df["rf_oof"].to_numpy(dtype=np.float64)
    clp_a = df["clp_prior_rf"].to_numpy(dtype=np.float64)
    clp_b = df["clp_residual_rf"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    m_obs = np.isfinite(obs)
    ax.plot(obs[m_obs], depth[m_obs], color=OBS_COLOR, linewidth=1.4, label="observado")
    ax.plot(rf[np.isfinite(rf)], depth[np.isfinite(rf)], color=RF_COLOR, linewidth=2.0, label="RF pontual")
    ax.plot(
        clp_a[np.isfinite(clp_a)],
        depth[np.isfinite(clp_a)],
        color=PRIOR_COLOR,
        linewidth=2.0,
        label="CLP prior RF (A)",
    )
    ax.plot(
        clp_b[np.isfinite(clp_b)],
        depth[np.isfinite(clp_b)],
        color=RESIDUAL_COLOR,
        linewidth=2.0,
        label="CLP residual RF (B)",
    )
    ax.set_xlabel(r"$\phi_{\mathrm{lab}}$ (pu)")
    ax.set_ylabel("profundidade (m)")
    ax.set_title("Comparacao unificada OOF (seed={})".format(int(seed)), fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.invert_yaxis()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="CLP three-method depth figure for LaTeX.")
    p.add_argument("--prior-csv", type=Path, default=DEFAULT_PRIOR_CSV)
    p.add_argument("--residual-csv", type=Path, default=DEFAULT_RESIDUAL_CSV)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--out-panels",
        type=Path,
        default=FIG_DIR / "fig_clp_three_methods_depth_panels.png",
    )
    p.add_argument(
        "--out-overlay",
        type=Path,
        default=FIG_DIR / "fig_clp_three_methods_depth_overlay.png",
    )
    return p.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = parse_args()
    df = load_merged_oof(args.prior_csv, args.residual_csv)
    plot_three_method_panels(df, args.out_panels, int(args.seed))
    plot_three_method_overlay(df, args.out_overlay, int(args.seed))
    print("OK", args.out_panels)
    print("OK", args.out_overlay)


if __name__ == "__main__":
    main()
