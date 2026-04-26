#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate tables and figures for the CLP-CSGM paper.

The script is intentionally read-only with respect to experiment outputs: it
loads existing raw data and benchmark CSV files, then writes paper-ready assets
under paper_clp_csgm/.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import multi_well_vc as mwv
import real_well_f03 as rwf


PAPER_DIR = os.path.join(REPO_ROOT, "paper_clp_csgm")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
TAB_DIR = os.path.join(PAPER_DIR, "tables")

CROSS_TRAIN = os.path.join(REPO_ROOT, "data", "F02-1,F03-2,F06-1_6logs_30dB.txt")
CROSS_TEST = os.path.join(REPO_ROOT, "data", "F03-4_6logs_30dB.txt")
F03_REAL = os.path.join(REPO_ROOT, "data", "F03-4_AC+GR+Porosity.txt")

CROSS_SUMMARY = os.path.join(
    REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid", "tables", "summary_focus_clp_csgm_vs_ub.csv"
)
CROSS_WINNERS = os.path.join(
    REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid", "tables", "csgm_vs_ae_winners.csv"
)
CROSS_RANK = os.path.join(
    REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid", "tables", "overall_rank.csv"
)
F03_SUMMARY = os.path.join(
    REPO_ROOT,
    "outputs",
    "real_well_f03",
    "direct_ub",
    "runs",
    "f03_full_gr_only_clp_csgm_ridge",
    "tables",
    "summary_focus_clp_csgm_vs_ub.csv",
)


@dataclass(frozen=True)
class TableSpec:
    """Container for a DataFrame and its output stem."""

    frame: pd.DataFrame
    stem: str
    float_format: str = "%.4f"


def ensure_dirs() -> None:
    """Create output folders for paper assets."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def _latex_escape(value: object) -> str:
    """Escape a small subset of LaTeX-special characters for table cells."""
    text = str(value)
    repl = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in repl.items():
        text = text.replace(old, new)
    return text


def write_table(spec: TableSpec) -> str:
    """Write CSV and LaTeX versions of a paper table."""
    csv_path = os.path.join(TAB_DIR, "{}.csv".format(spec.stem))
    tex_path = os.path.join(TAB_DIR, "{}.tex".format(spec.stem))
    spec.frame.to_csv(csv_path, index=False)
    with open(tex_path, "w", encoding="ascii") as f:
        f.write(spec.frame.to_latex(index=False, escape=True, float_format=lambda x: spec.float_format % x))
    return tex_path


def _series_stats(values: np.ndarray) -> Dict[str, float]:
    """Return robust univariate statistics for a numeric vector."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "n": 0.0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "max": np.nan,
        }
    return {
        "n": float(v.size),
        "mean": float(np.mean(v)),
        "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
        "min": float(np.min(v)),
        "q25": float(np.percentile(v, 25)),
        "median": float(np.median(v)),
        "q75": float(np.percentile(v, 75)),
        "max": float(np.max(v)),
    }


def _load_cross_segments() -> Tuple[List[mwv.WellSegment], List[mwv.WellSegment]]:
    """Load train and held-out cross-well Vc segments."""
    channels = ("sonic", "rhob", "gr", "ai", "vp")
    train = mwv.load_6log_file(CROSS_TRAIN, target_name="vc", channels=channels)
    test = mwv.load_6log_file(CROSS_TEST, target_name="vc", channels=channels)
    return train, test


def build_dataset_summary() -> pd.DataFrame:
    """Summarize rows, depth spans, targets, and channels used by the article."""
    train, test = _load_cross_segments()
    rows: List[Dict[str, object]] = []
    for split, segs in (("cross_train", train), ("cross_test", test)):
        for seg in segs:
            stats = _series_stats(seg.target)
            rows.append(
                {
                    "dataset": split,
                    "well": seg.name,
                    "target": seg.target_name,
                    "n_rows": int(seg.n_rows),
                    "depth_min": float(np.min(seg.depth)),
                    "depth_max": float(np.max(seg.depth)),
                    "target_mean": stats["mean"],
                    "target_std": stats["std"],
                    "target_min": stats["min"],
                    "target_max": stats["max"],
                }
            )

    f03 = rwf.load_f03_table(F03_REAL)
    stats = _series_stats(f03.porosity)
    rows.append(
        {
            "dataset": "real_f03_gr",
            "well": "F03-4",
            "target": "porosity",
            "n_rows": int(f03.n_rows),
            "depth_min": float(np.min(f03.depth)),
            "depth_max": float(np.max(f03.depth)),
            "target_mean": stats["mean"],
            "target_std": stats["std"],
            "target_min": stats["min"],
            "target_max": stats["max"],
        }
    )
    return pd.DataFrame(rows)


def build_correlation_table() -> pd.DataFrame:
    """Compute channel-target correlations for the cross-well and F03 data."""
    train, test = _load_cross_segments()
    rows: List[Dict[str, object]] = []
    for split, segs in (("cross_train", train), ("cross_test", test)):
        for channel in ("sonic", "rhob", "gr", "ai", "vp"):
            xs: List[np.ndarray] = []
            ys: List[np.ndarray] = []
            for seg in segs:
                xs.append(np.asarray(seg.channels[channel], dtype=np.float64))
                ys.append(np.asarray(seg.target, dtype=np.float64))
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            corr = float(np.corrcoef(x, y)[0, 1])
            rows.append({"dataset": split, "channel": channel, "target": "vc", "pearson_r": corr})

    f03 = rwf.load_f03_table(F03_REAL)
    for channel, values in (("ac", f03.ac), ("gr", f03.gr)):
        corr = float(np.corrcoef(np.asarray(values, dtype=np.float64), f03.porosity)[0, 1])
        rows.append({"dataset": "real_f03", "channel": channel, "target": "porosity", "pearson_r": corr})
    return pd.DataFrame(rows)


def build_crosswell_winners_table() -> pd.DataFrame:
    """Format CLP-CSGM vs AE winners for the paper."""
    df = pd.read_csv(CROSS_WINNERS)
    out = df[
        [
            "step",
            "n_train_approx",
            "measurement_ratio",
            "csgm_rmse",
            "ae_rmse",
            "csgm_gap_vs_ae_pct",
        ]
    ].copy()
    out = out.rename(
        columns={
            "measurement_ratio": "rho",
            "csgm_rmse": "clp_csgm_rmse",
            "ae_rmse": "ae_ub_rmse",
            "csgm_gap_vs_ae_pct": "gap_vs_ae_pct",
        }
    )
    return out


def build_crosswell_rank_table() -> pd.DataFrame:
    """Format the overall cross-well ranking."""
    df = pd.read_csv(CROSS_RANK)
    out = df[["method_label", "rmse_mean_all_cells", "relative_l2_mean_all_cells", "cells"]].copy()
    out = out.rename(
        columns={
            "method_label": "method",
            "rmse_mean_all_cells": "mean_rmse",
            "relative_l2_mean_all_cells": "mean_relative_l2",
        }
    )
    return out


def build_f03_table() -> pd.DataFrame:
    """Format the real-well F03 GR-only benchmark summary."""
    df = pd.read_csv(F03_SUMMARY)
    out = df[["measurement_ratio", "method_label", "rmse_mean", "mae_mean", "relative_l2_mean"]].copy()
    out = out.rename(
        columns={
            "measurement_ratio": "rho",
            "method_label": "method",
            "rmse_mean": "rmse",
            "mae_mean": "mae",
            "relative_l2_mean": "relative_l2",
        }
    )
    return out.sort_values(["rho", "rmse"]).reset_index(drop=True)


def plot_crosswell_target_hist() -> str:
    """Plot target distributions for train and held-out cross-well Vc data."""
    train, test = _load_cross_segments()
    y_train = np.concatenate([seg.target for seg in train])
    y_test = np.concatenate([seg.target for seg in test])
    out = os.path.join(FIG_DIR, "eda_crosswell_vc_distribution.png")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(y_train, bins=50, alpha=0.65, density=True, label="train wells")
    ax.hist(y_test, bins=50, alpha=0.65, density=True, label="held-out F03-4")
    ax.set_xlabel("Vc")
    ax.set_ylabel("density")
    ax.set_title("Cross-well Vc target distribution")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_f03_depth_profile() -> str:
    """Plot F03 GR and porosity versus depth for EDA."""
    f03 = rwf.load_f03_table(F03_REAL)
    out = os.path.join(FIG_DIR, "eda_f03_gr_porosity_depth.png")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 6.0), sharey=True)
    axes[0].plot(f03.gr, f03.depth, color="#1f77b4", linewidth=0.9)
    axes[0].set_xlabel("GR")
    axes[0].set_ylabel("depth (m)")
    axes[0].set_title("Gamma ray")
    axes[1].plot(f03.porosity, f03.depth, color="#d62728", linewidth=0.9)
    axes[1].set_xlabel("porosity")
    axes[1].set_title("Porosity")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
    fig.suptitle("F03-4 real-well profile used in the GR-only benchmark", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_correlations(corr: pd.DataFrame) -> str:
    """Plot absolute channel-target correlations from EDA."""
    out = os.path.join(FIG_DIR, "eda_channel_target_correlations.png")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    labels = [
        "{}:{}".format(str(row.dataset), str(row.channel))
        for row in corr.itertuples(index=False)
    ]
    ax.bar(np.arange(len(labels)), corr["pearson_r"].values, color="#4c78a8")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Pearson r")
    ax.set_title("Channel-target linear correlation")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_crosswell_results() -> str:
    """Plot RMSE versus measurement ratio for the cross-well low-data grid."""
    df = pd.read_csv(CROSS_SUMMARY)
    out = os.path.join(FIG_DIR, "results_crosswell_rmse_vs_rho.png")
    labels = {
        "ridge_prior_csgm": "CLP-CSGM Ridge",
        "ae_regression_ub": "AE [u,b]",
        "ml_only": "ML only",
        "mlp_concat_ub": "MLP [u,b]",
        "pca_regression_ub": "PCA [u,b]",
    }
    colors = {
        "ridge_prior_csgm": "#d62728",
        "ae_regression_ub": "#17becf",
        "ml_only": "#1f77b4",
        "mlp_concat_ub": "#2ca02c",
        "pca_regression_ub": "#9467bd",
    }
    steps = sorted(df["step"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(4.2 * len(steps), 4.0), sharey=True)
    if len(steps) == 1:
        axes = np.array([axes])
    for ax, step in zip(axes, steps):
        sub_step = df[df["step"] == step]
        for method in labels:
            sub = sub_step[sub_step["method"] == method].sort_values("measurement_ratio")
            if sub.empty:
                continue
            ax.errorbar(
                sub["measurement_ratio"],
                sub["rmse_mean"],
                yerr=sub["rmse_ci95_half"].fillna(0.0),
                marker="o",
                capsize=3,
                linewidth=2.2 if method == "ridge_prior_csgm" else 1.4,
                color=colors[method],
                label=labels[method],
            )
        ax.set_title("step = {}".format(step))
        ax.set_xlabel("measurement ratio rho")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE")
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("Cross-well Vc low-data benchmark", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_f03_results() -> str:
    """Plot RMSE versus measurement ratio for the real-well F03 benchmark."""
    df = pd.read_csv(F03_SUMMARY)
    out = os.path.join(FIG_DIR, "results_f03_gr_only_rmse_vs_rho.png")
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    colors = {
        "CLP-CSGM Ridge": "#d62728",
        "AE [u,b]": "#17becf",
        "ML only": "#1f77b4",
        "MLP [u,b]": "#2ca02c",
        "PCA [u,b]": "#9467bd",
    }
    for label in ["CLP-CSGM Ridge", "AE [u,b]", "ML only", "MLP [u,b]", "PCA [u,b]"]:
        sub = df[df["method_label"] == label].sort_values("measurement_ratio")
        if sub.empty:
            continue
        ax.errorbar(
            sub["measurement_ratio"],
            sub["rmse_mean"],
            yerr=sub["rmse_ci95_half"].fillna(0.0),
            marker="o",
            capsize=3,
            linewidth=2.2 if label == "CLP-CSGM Ridge" else 1.4,
            color=colors[label],
            label=label,
        )
    ax.set_xlabel("measurement ratio rho")
    ax.set_ylabel("RMSE")
    ax.set_title("Real-well F03 GR-only porosity benchmark")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_asset_manifest(paths: Iterable[str]) -> None:
    """Write a compact manifest for generated paper assets."""
    manifest = os.path.join(PAPER_DIR, "ASSET_MANIFEST.txt")
    rels = [os.path.relpath(p, PAPER_DIR) for p in paths]
    with open(manifest, "w", encoding="ascii") as f:
        f.write("CLP-CSGM paper assets generated by scripts/clp_csgm_paper_assets.py\n")
        f.write("\n")
        for rel in sorted(rels):
            f.write("{}\n".format(rel))


def main() -> None:
    """Generate all paper assets."""
    ensure_dirs()
    generated: List[str] = []

    dataset_summary = build_dataset_summary()
    corr = build_correlation_table()
    tables = [
        TableSpec(dataset_summary, "eda_dataset_summary", "%.4f"),
        TableSpec(corr, "eda_channel_target_correlations", "%.4f"),
        TableSpec(build_crosswell_winners_table(), "results_crosswell_clp_vs_ae", "%.5f"),
        TableSpec(build_crosswell_rank_table(), "results_crosswell_overall_rank", "%.5f"),
        TableSpec(build_f03_table(), "results_f03_gr_only", "%.5f"),
    ]
    for spec in tables:
        generated.append(write_table(spec))
        generated.append(os.path.join(TAB_DIR, "{}.csv".format(spec.stem)))

    generated.extend(
        [
            plot_crosswell_target_hist(),
            plot_f03_depth_profile(),
            plot_correlations(corr),
            plot_crosswell_results(),
            plot_f03_results(),
        ]
    )
    write_asset_manifest(generated)
    print("Generated {} CLP-CSGM paper assets under {}".format(len(generated), PAPER_DIR))


if __name__ == "__main__":
    main()
