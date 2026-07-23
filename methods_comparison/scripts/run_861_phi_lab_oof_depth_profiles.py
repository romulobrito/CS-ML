#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OOF depth profiles for Phi_lab well-profile regressors (Well 861).

Generates depth vs porosity figures (observed + OOF prediction) for RF, GB,
XGBoost, MLP and Linear Regression under depth-block CV (3 folds).

Protocol: same as run_861_ml_baseline.py (enriched, log_only, depth_block_3).
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    DEPTH_COL,
    DEFAULT_ENRICHED,
    WELL_PROFILE_PRIMARY_TARGET,
    build_xy,
    depth_block_splits,
    load_logs_enriched,
    well_profile_phi_compare_dir,
)
from ml_861_metrics import FoldMetrics, _rmse, collect_depth_block_oof
from run_861_ml_baseline import REGRESSOR_CHOICES, model_factory

OBSERVED_COLOR = "#212121"
FOLD_BOUNDARY_COLOR = "#B71C1C"

REGRESSOR_COLORS: Dict[str, str] = {
    "rf": "#2E7D32",
    "gb": "#6A1B9A",
    "xgb": "#E65100",
    "mlp": "#C62828",
    "lr": "#0277BD",
}

REGRESSOR_LABELS: Dict[str, str] = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "lr": "Linear Regression",
}


def _regressor_label(name: str) -> str:
    return REGRESSOR_LABELS.get(name, name)


def _fold_boundaries_m(df: pd.DataFrame, n_blocks: int) -> List[float]:
    """Mid-depth boundaries between contiguous depth blocks."""
    folds = depth_block_splits(df, n_blocks=n_blocks)
    bounds: List[float] = []
    for i in range(len(folds) - 1):
        d_hi = float(folds[i].depth_max_m)
        d_lo = float(folds[i + 1].depth_min_m)
        bounds.append(0.5 * (d_hi + d_lo))
    return bounds


def _draw_fold_boundaries(ax: plt.Axes, boundaries: Sequence[float]) -> None:
    for b in boundaries:
        if np.isfinite(b):
            ax.axhline(float(b), color=FOLD_BOUNDARY_COLOR, linestyle="--", linewidth=0.9, alpha=0.75)


def collect_all_oof(
    regressors: Sequence[str],
    smoke: bool,
    random_state: int,
    n_blocks: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, List[FoldMetrics]]]:
    """Build bundle and OOF predictions for each regressor."""
    df = load_logs_enriched()
    bundle = build_xy(df, target=WELL_PROFILE_PRIMARY_TARGET, feature_mode="log_only")
    depth = bundle.df[DEPTH_COL].to_numpy(dtype=np.float64)
    observed = bundle.y.astype(np.float64)

    oof_by_reg: Dict[str, np.ndarray] = {}
    folds_by_reg: Dict[str, List[FoldMetrics]] = {}

    for reg in regressors:
        try:
            factory = model_factory(reg, smoke=smoke, random_state=random_state, small_sample=False)
        except ImportError as exc:
            print("SKIP {}: {}".format(reg, exc))
            continue
        oof, fold_metrics = collect_depth_block_oof(factory, bundle, n_blocks=n_blocks)
        oof_by_reg[reg] = oof
        folds_by_reg[reg] = fold_metrics

    if not oof_by_reg:
        raise RuntimeError("No regressors produced OOF predictions.")

    return bundle.df, depth, observed, oof_by_reg, folds_by_reg


def build_oof_table(
    depth: np.ndarray,
    observed: np.ndarray,
    oof_by_reg: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Tabular OOF predictions for all regressors."""
    data: Dict[str, np.ndarray] = {
        DEPTH_COL: depth,
        WELL_PROFILE_PRIMARY_TARGET: observed,
    }
    for reg, oof in oof_by_reg.items():
        data["oof_{}".format(reg)] = oof
    return pd.DataFrame(data)


def plot_depth_overlay(
    depth: np.ndarray,
    observed: np.ndarray,
    oof_by_reg: Dict[str, np.ndarray],
    fold_boundaries: Sequence[float],
    out_path: Path,
    title: str,
) -> None:
    """Single panel: observed + all OOF model profiles."""
    fig, ax = plt.subplots(figsize=(6.5, 7.5))
    m_obs = np.isfinite(observed)
    ax.plot(
        observed[m_obs],
        depth[m_obs],
        color=OBSERVED_COLOR,
        linewidth=1.8,
        label="observed (phi_lab)",
        zorder=2,
    )
    for reg in sorted(oof_by_reg.keys()):
        oof = oof_by_reg[reg]
        m = np.isfinite(oof)
        ax.plot(
            oof[m],
            depth[m],
            color=REGRESSOR_COLORS.get(reg, "#7f7f7f"),
            linewidth=2.0,
            label=_regressor_label(reg),
            zorder=3,
        )
    _draw_fold_boundaries(ax, fold_boundaries)
    ax.set_xlabel("porosity (pu)")
    ax.set_ylabel("depth (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.invert_yaxis()
    fig.suptitle(title, fontsize=12, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_depth_panels(
    depth: np.ndarray,
    observed: np.ndarray,
    oof_by_reg: Dict[str, np.ndarray],
    fold_boundaries: Sequence[float],
    out_path: Path,
    title: str,
) -> None:
    """One panel per regressor: dashed observed + colored OOF."""
    regs = sorted(oof_by_reg.keys())
    n_cols = len(regs)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 7.5), sharey=True)
    if n_cols == 1:
        axes = np.array([axes])
    m_obs = np.isfinite(observed)
    for ax, reg in zip(axes, regs):
        oof = oof_by_reg[reg]
        m = np.isfinite(oof)
        ax.plot(
            observed[m_obs],
            depth[m_obs],
            color="#9E9E9E",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            label="observed",
        )
        ax.plot(
            oof[m],
            depth[m],
            color=REGRESSOR_COLORS.get(reg, "#7f7f7f"),
            linewidth=2.2,
            label="OOF pred.",
        )
        _draw_fold_boundaries(ax, fold_boundaries)
        rmse = _rmse(observed[m], oof[m])
        ax.set_title("{} (RMSE={:.3f})".format(_regressor_label(reg), rmse), fontsize=9)
        ax.set_xlabel("porosity (pu)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=7)
    axes[0].set_ylabel("depth (m)")
    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle(title, fontsize=12, y=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_rmse_by_fold(
    folds_by_reg: Dict[str, List[FoldMetrics]],
    out_path: Path,
) -> None:
    """Grouped bar chart of fold RMSE per regressor."""
    regs = sorted(folds_by_reg.keys())
    fold_ids = sorted({f.fold_id for fl in folds_by_reg.values() for f in fl})
    x = np.arange(len(fold_ids), dtype=np.float64)
    width = 0.8 / max(len(regs), 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, reg in enumerate(regs):
        by_fold = {f.fold_id: f.rmse for f in folds_by_reg[reg]}
        vals = [by_fold.get(int(fid), float("nan")) for fid in fold_ids]
        offset = (float(i) - 0.5 * (len(regs) - 1)) * width
        ax.bar(
            x + offset,
            vals,
            width=width,
            label=_regressor_label(reg),
            color=REGRESSOR_COLORS.get(reg, "#7f7f7f"),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["fold {}".format(int(f)) for f in fold_ids])
    ax.set_ylabel("RMSE (pu)")
    ax.set_xlabel("depth-block fold")
    ax.set_title("Phi_lab OOF RMSE by depth block (Well 861)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_phi_lab_oof_depth_profiles(
    out_dir: Path,
    regressors: Sequence[str],
    smoke: bool = False,
    random_state: int = 42,
    n_blocks: int = 3,
) -> Dict[str, Path]:
    """Generate OOF depth figures and tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    tables_dir = out_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    n_blocks_run = 2 if smoke else n_blocks
    df, depth, observed, oof_by_reg, folds_by_reg = collect_all_oof(
        regressors, smoke, random_state, n_blocks_run
    )
    fold_bounds = _fold_boundaries_m(df, n_blocks_run)

    oof_table = build_oof_table(depth, observed, oof_by_reg)
    oof_csv = tables_dir / "oof_predictions_phi_lab_depth.csv"
    oof_table.to_csv(oof_csv, index=False)

    summary_rows: List[Dict[str, object]] = []
    for reg, oof in oof_by_reg.items():
        m = np.isfinite(oof)
        summary_rows.append(
            {
                "regressor": reg,
                "label": _regressor_label(reg),
                "oof_rmse": _rmse(observed[m], oof[m]),
                "n_oof": int(np.sum(m)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = tables_dir / "oof_summary_phi_lab.csv"
    summary_df.to_csv(summary_csv, index=False)

    title = "861 MOGNO phi_lab OOF depth-block CV (n={})".format(len(observed))
    paths: Dict[str, Path] = {
        "overlay": figures_dir / "phi_lab_oof_depth_overlay.png",
        "panels": figures_dir / "phi_lab_oof_depth_panels.png",
        "rmse_by_fold": figures_dir / "phi_lab_oof_rmse_by_depth_fold.png",
        "oof_csv": oof_csv,
        "summary_csv": summary_csv,
    }

    plot_depth_overlay(depth, observed, oof_by_reg, fold_bounds, paths["overlay"], title)
    plot_depth_panels(depth, observed, oof_by_reg, fold_bounds, paths["panels"], title)
    plot_rmse_by_fold(folds_by_reg, paths["rmse_by_fold"])

    meta = {
        "target": WELL_PROFILE_PRIMARY_TARGET,
        "protocol": "depth_block_{}".format(n_blocks_run),
        "regressors": list(oof_by_reg.keys()),
        "fold_boundaries_m": fold_bounds,
        "figures": {k: str(v) for k, v in paths.items() if v.suffix == ".png"},
    }
    (out_dir / "oof_depth_profiles_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return paths


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phi_lab OOF depth profiles for Well 861 five-regressor baseline"
    )
    parser.add_argument("--out-dir", type=Path, default=well_profile_phi_compare_dir())
    parser.add_argument(
        "--regressor",
        choices=REGRESSOR_CHOICES,
        default="all",
        help="Regressor subset (default: all five)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-blocks", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.regressor == "all":
        regs = ["rf", "gb", "xgb", "mlp", "lr"]
    else:
        regs = [str(args.regressor)]

    paths = run_phi_lab_oof_depth_profiles(
        out_dir=args.out_dir.resolve(),
        regressors=regs,
        smoke=bool(args.smoke),
        random_state=int(args.random_state),
        n_blocks=int(args.n_blocks),
    )
    print("PHI_LAB_OOF_DEPTH_PROFILES")
    print("OUT_DIR", args.out_dir.resolve())
    for key, path in paths.items():
        print("{} {}".format(key.upper(), path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
