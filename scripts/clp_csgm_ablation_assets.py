#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build CLP-CSGM ablation tables and figures for the paper.

ASCII-only source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_clp_csgm"
FIG_DIR = PAPER / "figures"
TAB_DIR = PAPER / "tables"

CROSS_RUNS = {
    8: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step08_clp_csgm_ablation_ridge",
    16: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step16_clp_csgm_ablation_ridge",
    32: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step32_clp_csgm_ablation_ridge",
}
F03_RUN = (
    ROOT
    / "outputs"
    / "real_well_f03"
    / "direct_ub"
    / "runs"
    / "f03_gr_only_clp_csgm_ablation_ridge"
)
F03_MLP_RUN = (
    ROOT
    / "outputs"
    / "real_well_f03"
    / "direct_ub"
    / "runs"
    / "f03_gr_only_clp_csgm_mlp_prior_full"
)
MLP_CROSS_RUNS = {
    8: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step08_clp_csgm_mlp_prior_full",
    16: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step16_clp_csgm_mlp_prior_full",
    32: ROOT / "outputs" / "cross_well_vc" / "direct_ub" / "runs" / "crosswell_step32_clp_csgm_mlp_prior_full",
}

METHOD_LABELS: Dict[str, str] = {
    "ridge_prior_csgm": "CLP-CSGM Ridge",
    "ridge_prior_only_decoder": "Prior-only Ridge",
    "measurement_only_csgm": "Measurement-only CSGM",
    "ae_regression_ub": "AE [u,b]",
}


def _ensure_dirs() -> None:
    """Create paper asset folders."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TAB_DIR.mkdir(parents=True, exist_ok=True)


def _read_summary(path: Path) -> pd.DataFrame:
    """Read one summary table and validate expected columns."""
    summary_path = path / "tables" / "summary.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(str(summary_path))
    df = pd.read_csv(summary_path)
    required = {"measurement_ratio", "method", "rmse_mean", "mae_mean", "relative_l2_mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing columns in {}: {}".format(summary_path, sorted(missing)))
    return df


def _write_latex(df: pd.DataFrame, path: Path) -> None:
    """Write a compact LaTeX table fragment."""
    text = df.to_latex(index=False, escape=False, float_format="{:.4f}".format)
    path.write_text(text, encoding="utf-8")


def _wide_ablation(df: pd.DataFrame, index_cols: List[str]) -> pd.DataFrame:
    """Build a wide RMSE table and percent gains for ablation interpretation."""
    methods = list(METHOD_LABELS)
    wide = (
        df[df["method"].isin(methods)]
        .pivot_table(index=index_cols, columns="method", values="rmse_mean")
        .reset_index()
    )
    wide["gain_vs_prior_only_pct"] = (
        100.0
        * (wide["ridge_prior_only_decoder"] - wide["ridge_prior_csgm"])
        / wide["ridge_prior_only_decoder"]
    )
    wide["gain_vs_measurement_only_pct"] = (
        100.0
        * (wide["measurement_only_csgm"] - wide["ridge_prior_csgm"])
        / wide["measurement_only_csgm"]
    )
    wide["gain_vs_ae_pct"] = (
        100.0 * (wide["ae_regression_ub"] - wide["ridge_prior_csgm"]) / wide["ae_regression_ub"]
    )
    return wide


def build_crosswell_assets() -> pd.DataFrame:
    """Create cross-well ablation tables and figure."""
    rows = []
    for step, run_dir in CROSS_RUNS.items():
        df = _read_summary(run_dir)
        df.insert(0, "step", int(step))
        rows.append(df)
    long_df = pd.concat(rows, ignore_index=True)
    long_out = TAB_DIR / "ablation_crosswell_summary.csv"
    long_df.to_csv(long_out, index=False)

    wide = _wide_ablation(long_df, ["step", "measurement_ratio"])
    wide_out = TAB_DIR / "ablation_crosswell_wide.csv"
    wide.to_csv(wide_out, index=False)

    table = wide[
        [
            "step",
            "measurement_ratio",
            "ridge_prior_csgm",
            "ridge_prior_only_decoder",
            "measurement_only_csgm",
            "ae_regression_ub",
            "gain_vs_prior_only_pct",
            "gain_vs_measurement_only_pct",
        ]
    ].copy()
    table.columns = [
        "Step",
        "$\\rho$",
        "CLP-CSGM",
        "Prior-only",
        "Meas.-only",
        "AE $[u,b]$",
        "Gain vs prior (\\%)",
        "Gain vs meas. (\\%)",
    ]
    _write_latex(table, TAB_DIR / "ablation_crosswell_wide.tex")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.7), sharey=True)
    colors = {
        "ridge_prior_csgm": "#d62728",
        "ridge_prior_only_decoder": "#ff9896",
        "measurement_only_csgm": "#7f7f7f",
        "ae_regression_ub": "#17becf",
    }
    for ax, step in zip(axes, [8, 16, 32]):
        sub = long_df[long_df["step"] == step]
        for method, label in METHOD_LABELS.items():
            mdf = sub[sub["method"] == method].sort_values("measurement_ratio")
            ax.plot(
                mdf["measurement_ratio"],
                mdf["rmse_mean"],
                marker="o",
                label=label,
                color=colors[method],
            )
        ax.set_title("step = {}".format(step))
        ax.set_xlabel("Measurement ratio")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("RMSE")
    axes[-1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_crosswell_rmse.png", dpi=200)
    plt.close(fig)
    return wide


def build_f03_assets() -> pd.DataFrame:
    """Create F03 GR-only ablation tables and figure."""
    long_df = _read_summary(F03_RUN)
    long_df.to_csv(TAB_DIR / "ablation_f03_summary.csv", index=False)
    wide = _wide_ablation(long_df, ["measurement_ratio"])
    wide.to_csv(TAB_DIR / "ablation_f03_wide.csv", index=False)

    table = wide[
        [
            "measurement_ratio",
            "ridge_prior_csgm",
            "ridge_prior_only_decoder",
            "measurement_only_csgm",
            "ae_regression_ub",
            "gain_vs_prior_only_pct",
            "gain_vs_measurement_only_pct",
        ]
    ].copy()
    table.columns = [
        "$\\rho$",
        "CLP-CSGM",
        "Prior-only",
        "Meas.-only",
        "AE $[u,b]$",
        "Gain vs prior (\\%)",
        "Gain vs meas. (\\%)",
    ]
    _write_latex(table, TAB_DIR / "ablation_f03_wide.tex")

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = {
        "ridge_prior_csgm": "#d62728",
        "ridge_prior_only_decoder": "#ff9896",
        "measurement_only_csgm": "#7f7f7f",
        "ae_regression_ub": "#17becf",
    }
    for method, label in METHOD_LABELS.items():
        mdf = long_df[long_df["method"] == method].sort_values("measurement_ratio")
        ax.plot(
            mdf["measurement_ratio"],
            mdf["rmse_mean"],
            marker="o",
            label=label,
            color=colors[method],
        )
    ax.set_xlabel("Measurement ratio")
    ax.set_ylabel("RMSE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_f03_rmse.png", dpi=200)
    plt.close(fig)
    return wide


def build_prior_class_assets() -> None:
    """Create Ridge-vs-MLP prior sensitivity assets."""
    cross_rows = []
    for step, run_dir in CROSS_RUNS.items():
        ridge = _read_summary(run_dir)
        ridge = ridge[ridge["method"] == "ridge_prior_csgm"].copy()
        ridge.insert(0, "step", int(step))
        cross_rows.append(ridge)
    for step, run_dir in MLP_CROSS_RUNS.items():
        mlp = _read_summary(run_dir)
        mlp = mlp[mlp["method"] == "mlp_prior_csgm"].copy()
        mlp.insert(0, "step", int(step))
        cross_rows.append(mlp)
    cross_long = pd.concat(cross_rows, ignore_index=True)
    cross_long.to_csv(TAB_DIR / "ablation_prior_class_crosswell_summary.csv", index=False)
    cross_wide = (
        cross_long.pivot_table(
            index=["step", "measurement_ratio"], columns="method", values="rmse_mean"
        )
        .reset_index()
        .sort_values(["step", "measurement_ratio"])
    )
    cross_wide["ridge_gain_vs_mlp_pct"] = (
        100.0 * (cross_wide["mlp_prior_csgm"] - cross_wide["ridge_prior_csgm"])
        / cross_wide["mlp_prior_csgm"]
    )
    cross_wide.to_csv(TAB_DIR / "ablation_prior_class_crosswell_wide.csv", index=False)
    cross_table = cross_wide.copy()
    cross_table.columns = ["Step", "$\\rho$", "CLP-CSGM MLP", "CLP-CSGM Ridge", "Ridge gain (\\%)"]
    _write_latex(cross_table, TAB_DIR / "ablation_prior_class_crosswell_wide.tex")

    f03_ridge = _read_summary(F03_RUN)
    f03_ridge = f03_ridge[f03_ridge["method"] == "ridge_prior_csgm"].copy()
    f03_mlp = _read_summary(F03_MLP_RUN)
    f03_mlp = f03_mlp[f03_mlp["method"] == "mlp_prior_csgm"].copy()
    f03_long = pd.concat([f03_ridge, f03_mlp], ignore_index=True)
    f03_long.to_csv(TAB_DIR / "ablation_prior_class_f03_summary.csv", index=False)
    f03_wide = (
        f03_long.pivot_table(index=["measurement_ratio"], columns="method", values="rmse_mean")
        .reset_index()
        .sort_values("measurement_ratio")
    )
    f03_wide["ridge_gain_vs_mlp_pct"] = (
        100.0 * (f03_wide["mlp_prior_csgm"] - f03_wide["ridge_prior_csgm"])
        / f03_wide["mlp_prior_csgm"]
    )
    f03_wide.to_csv(TAB_DIR / "ablation_prior_class_f03_wide.csv", index=False)
    f03_table = f03_wide.copy()
    f03_table.columns = ["$\\rho$", "CLP-CSGM MLP", "CLP-CSGM Ridge", "Ridge gain (\\%)"]
    _write_latex(f03_table, TAB_DIR / "ablation_prior_class_f03_wide.tex")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))
    for method, label, color in [
        ("ridge_prior_csgm", "CLP-CSGM Ridge", "#d62728"),
        ("mlp_prior_csgm", "CLP-CSGM MLP", "#8c564b"),
    ]:
        for step in [8, 16, 32]:
            sub = cross_long[
                (cross_long["step"] == step) & (cross_long["method"] == method)
            ].sort_values("measurement_ratio")
            axes[0].plot(
                sub["measurement_ratio"],
                sub["rmse_mean"],
                marker="o",
                linestyle="-" if method == "ridge_prior_csgm" else "--",
                color=color,
                alpha=0.35 + 0.2 * ([8, 16, 32].index(step)),
                label="{} step {}".format(label, step),
            )
    axes[0].set_title("Cross-well Vc")
    axes[0].set_xlabel("Measurement ratio")
    axes[0].set_ylabel("RMSE")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)

    for method, label, color in [
        ("ridge_prior_csgm", "CLP-CSGM Ridge", "#d62728"),
        ("mlp_prior_csgm", "CLP-CSGM MLP", "#8c564b"),
    ]:
        sub = f03_long[f03_long["method"] == method].sort_values("measurement_ratio")
        axes[1].plot(
            sub["measurement_ratio"],
            sub["rmse_mean"],
            marker="o",
            linestyle="-" if method == "ridge_prior_csgm" else "--",
            color=color,
            label=label,
        )
    axes[1].set_title("F03 GR-only")
    axes[1].set_xlabel("Measurement ratio")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ablation_prior_class_ridge_vs_mlp.png", dpi=200)
    plt.close(fig)


def write_manifest(paths: Iterable[Path]) -> None:
    """Write an ablation asset manifest."""
    lines = ["CLP-CSGM ablation assets", ""]
    for path in paths:
        lines.append(str(path.relative_to(PAPER)))
    (PAPER / "ABLATION_ASSET_MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate all ablation paper assets."""
    _ensure_dirs()
    cross = build_crosswell_assets()
    f03 = build_f03_assets()
    build_prior_class_assets()
    write_manifest(
        [
            TAB_DIR / "ablation_crosswell_summary.csv",
            TAB_DIR / "ablation_crosswell_wide.csv",
            TAB_DIR / "ablation_crosswell_wide.tex",
            TAB_DIR / "ablation_f03_summary.csv",
            TAB_DIR / "ablation_f03_wide.csv",
            TAB_DIR / "ablation_f03_wide.tex",
            FIG_DIR / "ablation_crosswell_rmse.png",
            FIG_DIR / "ablation_f03_rmse.png",
            FIG_DIR / "ablation_prior_class_ridge_vs_mlp.png",
        ]
    )
    print("Cross-well cells: {}".format(len(cross)))
    print("F03 ratios: {}".format(len(f03)))


if __name__ == "__main__":
    main()
