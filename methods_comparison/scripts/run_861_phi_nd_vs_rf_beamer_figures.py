#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beamer figures: Phi_ND direct vs RF OOF for Phi_lab (Well 861).

Outputs:
  diagnostics_861/figures/phi_nd_vs_rf_depth_profile.png
  diagnostics_861/figures/phi_nd_vs_rf_bottom_panels.png
  diagnostics_861/figures/phi_nd_vs_rf_beamer_composite.png
  diagnostics_861/figures/phi_nd_vs_rf_metrics_bars.png
  diagnostics_861/figures/phi_nd_vs_rf_scatter_panels.png

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (  # noqa: E402
    DEPTH_COL,
    DEFAULT_ENRICHED,
    WELL_PROFILE_PRIMARY_TARGET,
    build_xy,
    depth_block_splits,
    load_logs_enriched,
)
from ml_861_metrics import collect_depth_block_oof  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402

FIG_ROOT = (
    REPO_ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "ml_runs"
    / "diagnostics_861"
    / "figures"
)
PHI_ND_COL = "Phi_ND (pu)"
TARGET = WELL_PROFILE_PRIMARY_TARGET
COLOR_LAB = "#212121"
COLOR_ND = "#1565C0"
COLOR_RF = "#2E7D32"
COLOR_FOLD = "#B71C1C"


def _fold_boundaries_m(df: pd.DataFrame, n_blocks: int) -> List[float]:
    """Mid-depth boundaries between contiguous depth blocks."""
    folds = depth_block_splits(df, n_blocks=n_blocks)
    bounds: List[float] = []
    for i in range(len(folds) - 1):
        d_hi = float(folds[i].depth_max_m)
        d_lo = float(folds[i + 1].depth_min_m)
        bounds.append(0.5 * (d_hi + d_lo))
    return bounds


def _plot_depth_profile(
    work: pd.DataFrame,
    n_blocks: int,
    out_path: Path,
) -> None:
    """Depth track: Phi_lab, Phi_ND direct and RF OOF."""
    plot_df = work.sort_values(DEPTH_COL).copy()
    depth = plot_df[DEPTH_COL].to_numpy(dtype=np.float64)
    y_lab = plot_df[TARGET].to_numpy(dtype=np.float64)
    y_nd = plot_df["phi_nd_direct"].to_numpy(dtype=np.float64)
    y_rf = plot_df["rf_oof"].to_numpy(dtype=np.float64)

    fig, ax = plt.subplots(figsize=(5.0, 5.4))
    ax.plot(y_nd, depth, color=COLOR_ND, linewidth=1.7, label="Direct Phi_ND")
    ax.plot(y_rf, depth, color=COLOR_RF, linewidth=1.5, alpha=0.92, label="RF OOF")
    ax.scatter(
        y_lab,
        depth,
        s=26,
        color=COLOR_LAB,
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
        label="Phi_lab",
    )
    for bound in _fold_boundaries_m(plot_df, n_blocks=n_blocks):
        ax.axhline(bound, color=COLOR_FOLD, linestyle="--", linewidth=0.9, alpha=0.75)

    ax.set_xlabel("Porosity (v/v)", fontsize=9)
    ax.set_ylabel("Depth (m)", fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    ax.set_title(
        "Depth profile (red dashes = OOF blocks)",
        fontsize=10,
        pad=4,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_bottom_panels(
    work: pd.DataFrame,
    summary: Dict[str, Dict[str, float]],
    out_path: Path,
) -> None:
    """Wide bottom row for Beamer: metrics + two scatter panels."""
    y = work[TARGET].to_numpy(dtype=np.float64)
    nd = work["phi_nd_direct"].to_numpy(dtype=np.float64)
    rf = work["rf_oof"].to_numpy(dtype=np.float64)
    lo = float(min(np.min(y), np.min(nd), np.min(rf)) - 0.01)
    hi = float(max(np.max(y), np.max(nd), np.max(rf)) + 0.01)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))

    ax_bar = axes[0]
    labels = ["Phi_ND", "RF OOF"]
    keys = ["phi_nd_direct", "rf_oof"]
    rmse_vals = [summary[k]["rmse"] for k in keys]
    r2_vals = [summary[k]["r2"] for k in keys]
    x_pos = np.arange(2)
    bar_w = 0.34
    ax_bar.bar(
        x_pos - bar_w / 2.0,
        rmse_vals,
        bar_w,
        color=COLOR_ND,
        label="RMSE",
    )
    ax_bar.bar(
        x_pos + bar_w / 2.0,
        r2_vals,
        bar_w,
        color=COLOR_RF,
        label="R2",
    )
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=9)
    ax_bar.set_ylabel("Value", fontsize=9)
    ax_bar.set_title("Global metrics", fontsize=10)
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(fontsize=8, loc="upper right")
    for i, val in enumerate(rmse_vals):
        ax_bar.text(i - bar_w / 2.0, val + 0.012, "{:.3f}".format(val), ha="center", fontsize=8)
    for i, val in enumerate(r2_vals):
        ax_bar.text(i + bar_w / 2.0, val + 0.012, "{:.3f}".format(val), ha="center", fontsize=8)

    ax_nd = axes[1]
    ax_nd.scatter(
        y,
        nd,
        s=34,
        alpha=0.82,
        color=COLOR_ND,
        edgecolors="white",
        linewidths=0.35,
    )
    ax_nd.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=1.0)
    ax_nd.set_xlim(lo, hi)
    ax_nd.set_ylim(lo, hi)
    ax_nd.set_aspect("equal", adjustable="box")
    ax_nd.set_xlabel("Observed Phi_lab (v/v)", fontsize=9)
    ax_nd.set_ylabel("Predicted porosity (v/v)", fontsize=9)
    ax_nd.set_title("Direct Phi_ND", fontsize=10)
    ax_nd.grid(True, alpha=0.25)

    ax_rf = axes[2]
    ax_rf.scatter(
        y,
        rf,
        s=34,
        alpha=0.82,
        color=COLOR_RF,
        edgecolors="white",
        linewidths=0.35,
    )
    ax_rf.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=1.0)
    ax_rf.set_xlim(lo, hi)
    ax_rf.set_ylim(lo, hi)
    ax_rf.set_aspect("equal", adjustable="box")
    ax_rf.set_xlabel("Observed Phi_lab (v/v)", fontsize=9)
    ax_rf.set_ylabel("Predicted porosity (v/v)", fontsize=9)
    ax_rf.set_title("RF OOF", fontsize=10)
    ax_rf.grid(True, alpha=0.25)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_beamer_composite(
    work: pd.DataFrame,
    summary: Dict[str, Dict[str, float]],
    n_blocks: int,
    out_path: Path,
) -> None:
    """Single Beamer panel: depth track + metrics + scatter."""
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(9.6, 6.8))
    gs = gridspec.GridSpec(
        2,
        3,
        figure=fig,
        height_ratios=[2.6, 0.72],
        hspace=0.62,
        wspace=0.30,
        top=0.93,
        bottom=0.07,
        left=0.07,
        right=0.98,
    )

    ax_depth = fig.add_subplot(gs[0, :])
    plot_df = work.sort_values(DEPTH_COL).copy()
    depth = plot_df[DEPTH_COL].to_numpy(dtype=np.float64)
    y_lab = plot_df[TARGET].to_numpy(dtype=np.float64)
    y_nd = plot_df["phi_nd_direct"].to_numpy(dtype=np.float64)
    y_rf = plot_df["rf_oof"].to_numpy(dtype=np.float64)

    ax_depth.plot(y_nd, depth, color=COLOR_ND, linewidth=1.7, label="Direct Phi_ND")
    ax_depth.plot(y_rf, depth, color=COLOR_RF, linewidth=1.5, alpha=0.92, label="RF OOF")
    ax_depth.scatter(
        y_lab,
        depth,
        s=24,
        color=COLOR_LAB,
        edgecolors="white",
        linewidths=0.35,
        zorder=4,
        label="Phi_lab",
    )
    for bound in _fold_boundaries_m(plot_df, n_blocks=n_blocks):
        ax_depth.axhline(
            bound,
            color=COLOR_FOLD,
            linestyle="--",
            linewidth=0.9,
            alpha=0.75,
        )
    ax_depth.set_xlabel("Porosity (v/v)")
    ax_depth.set_ylabel("Depth (m)")
    ax_depth.invert_yaxis()
    ax_depth.grid(True, alpha=0.25)
    ax_depth.legend(loc="lower right", fontsize=8, ncol=3, framealpha=0.92)
    ax_depth.set_title(
        "Depth profile (red dashes = OOF blocks)",
        fontsize=10,
        pad=6,
    )

    ax_bar = fig.add_subplot(gs[1, 0])
    labels = ["Phi_ND", "RF"]
    keys = ["phi_nd_direct", "rf_oof"]
    rmse_vals = [summary[k]["rmse"] for k in keys]
    r2_vals = [summary[k]["r2"] for k in keys]
    x_pos = np.arange(2)
    width = 0.36
    ax_bar.bar(x_pos - width / 2.0, rmse_vals, width, color=COLOR_ND, label="RMSE")
    ax_bar.bar(x_pos + width / 2.0, r2_vals, width, color=COLOR_RF, label="R2")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels, fontsize=8)
    ax_bar.set_title("Global metrics", fontsize=8, pad=3)
    ax_bar.grid(True, axis="y", alpha=0.25)
    ax_bar.legend(fontsize=7, loc="upper right")

    y = work[TARGET].to_numpy(dtype=np.float64)
    nd = work["phi_nd_direct"].to_numpy(dtype=np.float64)
    rf = work["rf_oof"].to_numpy(dtype=np.float64)
    lo = float(min(np.min(y), np.min(nd), np.min(rf)) - 0.01)
    hi = float(max(np.max(y), np.max(nd), np.max(rf)) + 0.01)

    ax_nd = fig.add_subplot(gs[1, 1])
    ax_nd.scatter(y, nd, s=22, alpha=0.8, color=COLOR_ND, edgecolors="white", linewidths=0.3)
    ax_nd.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=0.9)
    ax_nd.set_xlim(lo, hi)
    ax_nd.set_ylim(lo, hi)
    ax_nd.set_aspect("equal", adjustable="box")
    ax_nd.set_xlabel("Phi_lab (v/v)", fontsize=8)
    ax_nd.set_ylabel("Predicted (v/v)", fontsize=8)
    ax_nd.set_title("Direct Phi_ND", fontsize=8, pad=3)
    ax_nd.grid(True, alpha=0.25)

    ax_rf = fig.add_subplot(gs[1, 2])
    ax_rf.scatter(y, rf, s=22, alpha=0.8, color=COLOR_RF, edgecolors="white", linewidths=0.3)
    ax_rf.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=0.9)
    ax_rf.set_xlim(lo, hi)
    ax_rf.set_ylim(lo, hi)
    ax_rf.set_aspect("equal", adjustable="box")
    ax_rf.set_xlabel("Phi_lab (v/v)", fontsize=8)
    ax_rf.set_title("RF OOF", fontsize=8, pad=3)
    ax_rf.grid(True, alpha=0.25)

    fig.suptitle("Well 861: direct Phi_ND vs RF OOF (Phi_lab)", fontsize=11, y=0.985)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask]
    yp = y_pred[mask]
    corr = float(np.corrcoef(yt, yp)[0, 1]) if len(yt) > 1 else float("nan")
    return {
        "rmse": _rmse(yt, yp),
        "mae": float(mean_absolute_error(yt, yp)),
        "r2": float(r2_score(yt, yp)),
        "corr": corr,
        "n": int(len(yt)),
    }


def _load_comparison_df(
    data_path: Path,
    n_estimators: int,
    n_blocks: int,
    random_state: int,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Build depth table with observed, Phi_ND direct and RF OOF."""
    df = load_logs_enriched(data_path)
    bundle = build_xy(df, target=TARGET, feature_mode="log_only")

    rf_factory = lambda: RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    rf_oof, _ = collect_depth_block_oof(rf_factory, bundle, n_blocks=n_blocks)

    work = bundle.df.copy()
    work["phi_nd_direct"] = work[PHI_ND_COL].to_numpy(dtype=np.float64)
    work["rf_oof"] = rf_oof
    work = work.dropna(subset=[TARGET, PHI_ND_COL])
    work = work[np.isfinite(work["rf_oof"].to_numpy(dtype=np.float64))]

    y = work[TARGET].to_numpy(dtype=np.float64)
    nd = work["phi_nd_direct"].to_numpy(dtype=np.float64)
    rf = work["rf_oof"].to_numpy(dtype=np.float64)

    summary = {
        "phi_nd_direct": _metrics(y, nd),
        "rf_oof": _metrics(y, rf),
    }
    return work, summary


def _plot_metrics_bars(summary: Dict[str, Dict[str, float]], out_path: Path) -> None:
    """Grouped bar chart for RMSE and R2."""
    labels = ["Direct Phi_ND", "RF OOF"]
    keys = ["phi_nd_direct", "rf_oof"]
    rmse_vals = [summary[k]["rmse"] for k in keys]
    r2_vals = [summary[k]["r2"] for k in keys]
    colors = ["#1565C0", "#2E7D32"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    ax0 = axes[0]
    bars0 = ax0.bar(labels, rmse_vals, color=colors, width=0.55)
    ax0.set_ylabel("RMSE (pu)")
    ax0.set_title("Absolute error")
    ax0.set_ylim(0.0, max(rmse_vals) * 1.25)
    ax0.grid(True, axis="y", alpha=0.25)
    for bar, val in zip(bars0, rmse_vals):
        ax0.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.001,
            "{:.3f}".format(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax1 = axes[1]
    bars1 = ax1.bar(labels, r2_vals, color=colors, width=0.55)
    ax1.set_ylabel("Global R2")
    ax1.set_title("Explained variance")
    ax1.set_ylim(0.0, max(r2_vals) * 1.18)
    ax1.grid(True, axis="y", alpha=0.25)
    for bar, val in zip(bars1, r2_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.015,
            "{:.3f}".format(val),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle("Well 861: direct Phi_ND vs RF OOF (Phi_lab)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_scatter_panels(work: pd.DataFrame, out_path: Path) -> None:
    """Observed vs predicted for Phi_ND direct and RF OOF."""
    y = work[TARGET].to_numpy(dtype=np.float64)
    nd = work["phi_nd_direct"].to_numpy(dtype=np.float64)
    rf = work["rf_oof"].to_numpy(dtype=np.float64)

    lo = float(min(np.min(y), np.min(nd), np.min(rf)) - 0.01)
    hi = float(max(np.max(y), np.max(nd), np.max(rf)) + 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), sharex=True, sharey=True)

    panels = [
        ("Direct Phi_ND", nd, "#1565C0"),
        ("RF OOF (8 logs)", rf, "#2E7D32"),
    ]
    for ax, (title, pred, color) in zip(axes, panels):
        ax.scatter(y, pred, s=28, alpha=0.75, color=color, edgecolors="white", linewidths=0.4)
        ax.plot([lo, hi], [lo, hi], "--", color="#616161", linewidth=1.0, label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed Phi_lab (v/v)")
        ax.set_ylabel("Predicted porosity (v/v)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Well 861: observed vs predicted (n=87)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run(
    data_path: Path,
    out_dir: Path,
    n_estimators: int = 200,
    n_blocks: int = 3,
    random_state: int = 42,
) -> Dict[str, object]:
    """Generate Beamer comparison figures and summary JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    work, summary = _load_comparison_df(
        data_path=data_path,
        n_estimators=n_estimators,
        n_blocks=n_blocks,
        random_state=random_state,
    )

    bar_path = out_dir / "phi_nd_vs_rf_metrics_bars.png"
    scatter_path = out_dir / "phi_nd_vs_rf_scatter_panels.png"
    depth_path = out_dir / "phi_nd_vs_rf_depth_profile.png"
    bottom_path = out_dir / "phi_nd_vs_rf_bottom_panels.png"
    composite_path = out_dir / "phi_nd_vs_rf_beamer_composite.png"
    _plot_metrics_bars(summary, bar_path)
    _plot_scatter_panels(work, scatter_path)
    _plot_depth_profile(work, n_blocks=n_blocks, out_path=depth_path)
    _plot_bottom_panels(work, summary, out_path=bottom_path)
    _plot_beamer_composite(work, summary, n_blocks=n_blocks, out_path=composite_path)

    meta = {
        "well_id": "861",
        "target": TARGET,
        "n_points": int(len(work)),
        "phi_nd_direct": summary["phi_nd_direct"],
        "rf_oof": summary["rf_oof"],
        "pearson_phi_nd_vs_phi_lab": summary["phi_nd_direct"]["corr"],
        "protocol": {
            "phi_nd_direct": "wireline column used as prediction (no ML, global metrics)",
            "rf_oof": "RandomForest depth-block OOF ({} blocks)".format(n_blocks),
        },
        "figures": {
            "metrics_bars": str(bar_path),
            "scatter_panels": str(scatter_path),
            "depth_profile": str(depth_path),
            "bottom_panels": str(bottom_path),
            "beamer_composite": str(composite_path),
        },
    }
    meta_path = out_dir / "phi_nd_vs_rf_beamer_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phi_ND vs RF Beamer figures (Well 861).")
    p.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    p.add_argument("--out-dir", type=Path, default=FIG_ROOT)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    meta = run(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        n_estimators=int(args.n_estimators),
        n_blocks=int(args.n_blocks),
        random_state=int(args.random_state),
    )
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
