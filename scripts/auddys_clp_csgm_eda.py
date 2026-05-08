#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLP-CSGM-oriented EDA for Auddys_table.xlsx.

This script creates a reproducible artifact bundle for the first-stage analysis
of the Auddys dataset, aligned with the existing direct-UB benchmark workflow.

Outputs are saved under:
  <base_dir>/runs/<run_id>/
    tables/
    figures/
    logs/
    PROTOCOL.txt
    DATASET_MANIFEST.txt
    recommendations_for_runner.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EDAConfig:
    """User-facing knobs for the EDA run."""

    excel_path: str
    base_dir: str
    run_id: str
    target: str
    train_frac: float
    val_frac: float
    window_len: int
    step: int


def _parse_args() -> EDAConfig:
    parser = argparse.ArgumentParser(
        description="CLP-CSGM-oriented exploratory analysis for Auddys Excel dataset."
    )
    parser.add_argument(
        "--excel-path",
        type=str,
        default="data/Auddys_table.xlsx",
        help="Path to the source Excel file.",
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="outputs/auddys_clp_csgm_eda",
        help="Base output directory for run artifacts.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run folder name; default is timestamp.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="k_lab",
        choices=("k_lab", "phi_lab"),
        help="Primary target for CLP-focused recommendations.",
    )
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--window-len", type=int, default=64)
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()

    rid = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    return EDAConfig(
        excel_path=args.excel_path,
        base_dir=args.base_dir,
        run_id=rid,
        target=args.target,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        window_len=int(args.window_len),
        step=int(args.step),
    )


def _prepare_dirs(cfg: EDAConfig) -> Dict[str, str]:
    run_root = os.path.join(cfg.base_dir, "runs", cfg.run_id)
    out = {
        "run_root": run_root,
        "tables": os.path.join(run_root, "tables"),
        "figures": os.path.join(run_root, "figures"),
        "logs": os.path.join(run_root, "logs"),
    }
    for p in out.values():
        os.makedirs(p, exist_ok=True)
    return out


def _load_data(excel_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xls = pd.ExcelFile(excel_path)
    if "Logs" not in xls.sheet_names:
        raise ValueError("Missing required sheet 'Logs' in Excel file.")
    df_logs = pd.read_excel(excel_path, sheet_name="Logs")
    df_legend = pd.read_excel(excel_path, sheet_name="Legend") if "Legend" in xls.sheet_names else pd.DataFrame()

    col_map = {
        "Depth(m)": "depth_m",
        "Density (g/cc)": "density",
        "GR (API)": "gr",
        "Res_Deep": "res_deep",
        "Res_Shallow": "res_shallow",
        "Phi_Neutron (pu)": "phi_neutron",
        "Phi_Sonic (pu)": "phi_sonic",
        "Phi_ND (pu)": "phi_nd",
        "Phi_lab (pu)": "phi_lab",
        "k_lab (mD)": "k_lab",
        "RQI": "rqi",
        "FZI_lab": "fzi_lab",
        "Lithotype": "lithotype",
        "HFU": "hfu",
    }
    missing = [c for c in col_map if c not in df_logs.columns]
    if missing:
        raise ValueError("Missing expected columns in Logs sheet: {}".format(missing))
    df_logs = df_logs.rename(columns=col_map).copy()
    return df_logs, df_legend


def _write_tables(df: pd.DataFrame, out_tables: str) -> Dict[str, str]:
    num_cols = [
        "depth_m",
        "density",
        "gr",
        "res_deep",
        "res_shallow",
        "phi_neutron",
        "phi_sonic",
        "phi_nd",
        "phi_lab",
        "k_lab",
        "rqi",
        "fzi_lab",
    ]
    cat_cols = ["lithotype", "hfu"]

    written: Dict[str, str] = {}

    schema = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "null_pct": [float(df[c].isna().mean() * 100.0) for c in df.columns],
        }
    )
    path = os.path.join(out_tables, "eda_schema.csv")
    schema.to_csv(path, index=False)
    written["schema"] = path

    summary = df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    path = os.path.join(out_tables, "eda_numeric_summary.csv")
    summary.to_csv(path)
    written["numeric_summary"] = path

    pearson = df[num_cols].corr(method="pearson")
    path = os.path.join(out_tables, "eda_corr_pearson.csv")
    pearson.to_csv(path)
    written["corr_pearson"] = path

    spearman = df[num_cols].corr(method="spearman")
    path = os.path.join(out_tables, "eda_corr_spearman.csv")
    spearman.to_csv(path)
    written["corr_spearman"] = path

    class_counts = []
    for col in cat_cols:
        vc = df[col].value_counts(dropna=False).sort_index()
        for cls, n in vc.items():
            class_counts.append({"column": col, "class": str(cls), "count": int(n)})
    class_df = pd.DataFrame(class_counts)
    path = os.path.join(out_tables, "eda_class_counts.csv")
    class_df.to_csv(path, index=False)
    written["class_counts"] = path

    g_hfu = (
        df.groupby("hfu")[["k_lab", "phi_lab", "gr", "res_deep", "res_shallow", "phi_sonic", "phi_nd"]]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    g_hfu.columns = ["_".join([str(p) for p in c if p != ""]).strip("_") for c in g_hfu.columns.to_flat_index()]
    path = os.path.join(out_tables, "eda_group_by_hfu.csv")
    g_hfu.to_csv(path, index=False)
    written["group_by_hfu"] = path

    g_litho = (
        df.groupby("lithotype")[["k_lab", "phi_lab", "gr", "res_deep", "res_shallow", "phi_sonic", "phi_nd"]]
        .agg(["count", "mean", "std", "median"])
        .reset_index()
    )
    g_litho.columns = ["_".join([str(p) for p in c if p != ""]).strip("_") for c in g_litho.columns.to_flat_index()]
    path = os.path.join(out_tables, "eda_group_by_lithotype.csv")
    g_litho.to_csv(path, index=False)
    written["group_by_lithotype"] = path

    quality_rows = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_depth_rows": int(df["depth_m"].duplicated().sum()),
        "depth_monotonic_increasing": bool(np.all(np.diff(df["depth_m"].to_numpy(dtype=float)) > 0.0)),
    }
    quality = pd.DataFrame([quality_rows])
    path = os.path.join(out_tables, "eda_quality_checks.csv")
    quality.to_csv(path, index=False)
    written["quality"] = path

    return written


def _plot_depth_tracks(df: pd.DataFrame, out_figures: str) -> List[str]:
    saved: List[str] = []

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(df["depth_m"], df["gr"], color="tab:green", linewidth=1.0)
    axes[0].set_ylabel("GR")
    axes[0].grid(alpha=0.25)
    axes[1].plot(df["depth_m"], df["res_deep"], color="tab:blue", linewidth=1.0, label="Res_Deep")
    axes[1].plot(df["depth_m"], df["res_shallow"], color="tab:orange", linewidth=1.0, label="Res_Shallow")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].set_ylabel("Res")
    axes[1].grid(alpha=0.25)
    axes[2].plot(df["depth_m"], df["phi_lab"], color="tab:red", linewidth=1.1, label="Phi_lab")
    axes[2].plot(df["depth_m"], df["phi_sonic"], color="tab:purple", linewidth=0.9, label="Phi_sonic")
    axes[2].plot(df["depth_m"], df["phi_nd"], color="tab:brown", linewidth=0.9, label="Phi_ND")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].set_ylabel("Phi")
    axes[2].grid(alpha=0.25)
    axes[3].plot(df["depth_m"], df["k_lab"], color="black", linewidth=1.0)
    axes[3].set_ylabel("k_lab")
    axes[3].set_xlabel("Depth (m)")
    axes[3].set_yscale("log")
    axes[3].grid(alpha=0.25)
    fig.suptitle("Depth-oriented tracks (CLP-CSGM context)", fontsize=12)
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    path = os.path.join(out_figures, "01_depth_tracks.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(df["phi_lab"], bins=16, color="tab:red", alpha=0.85, edgecolor="black")
    axes[0].set_title("Phi_lab distribution")
    axes[0].set_xlabel("Phi_lab")
    axes[0].grid(alpha=0.2)
    axes[1].hist(df["k_lab"], bins=16, color="tab:gray", alpha=0.85, edgecolor="black")
    axes[1].set_title("k_lab distribution (log x)")
    axes[1].set_xlabel("k_lab")
    axes[1].set_xscale("log")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    path = os.path.join(out_figures, "02_target_histograms.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    return saved


def _plot_correlations(df: pd.DataFrame, out_figures: str) -> List[str]:
    saved: List[str] = []
    cols = [
        "density",
        "gr",
        "res_deep",
        "res_shallow",
        "phi_neutron",
        "phi_sonic",
        "phi_nd",
        "phi_lab",
        "k_lab",
    ]
    corr = df[cols].corr(method="pearson").to_numpy(dtype=float)
    labels = cols
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, "{:.2f}".format(corr[i, j]), ha="center", va="center", fontsize=7)
    ax.set_title("Pearson correlation matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path = os.path.join(out_figures, "03_corr_heatmap_pearson.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(path)
    return saved


def _plot_class_views(df: pd.DataFrame, out_figures: str) -> List[str]:
    saved: List[str] = []
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.boxplot(column="k_lab", by="hfu", ax=axes[0], grid=False)
    axes[0].set_yscale("log")
    axes[0].set_title("k_lab by HFU")
    axes[0].set_xlabel("HFU")
    axes[0].set_ylabel("k_lab")
    df.boxplot(column="phi_lab", by="hfu", ax=axes[1], grid=False)
    axes[1].set_title("phi_lab by HFU")
    axes[1].set_xlabel("HFU")
    axes[1].set_ylabel("phi_lab")
    fig.suptitle("")
    fig.tight_layout()
    path = os.path.join(out_figures, "04_boxplot_by_hfu.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df.boxplot(column="k_lab", by="lithotype", ax=axes[0], grid=False)
    axes[0].set_yscale("log")
    axes[0].set_title("k_lab by Lithotype")
    axes[0].set_xlabel("Lithotype")
    axes[0].set_ylabel("k_lab")
    df.boxplot(column="phi_lab", by="lithotype", ax=axes[1], grid=False)
    axes[1].set_title("phi_lab by Lithotype")
    axes[1].set_xlabel("Lithotype")
    axes[1].set_ylabel("phi_lab")
    fig.suptitle("")
    fig.tight_layout()
    path = os.path.join(out_figures, "05_boxplot_by_lithotype.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    saved.append(path)
    return saved


def _plot_scatter_screen(df: pd.DataFrame, out_figures: str) -> List[str]:
    saved: List[str] = []
    features = ["density", "gr", "res_deep", "res_shallow", "phi_neutron", "phi_sonic", "phi_nd"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    axes_flat = axes.ravel()
    for i, feat in enumerate(features):
        ax = axes_flat[i]
        ax.scatter(df[feat], df["k_lab"], s=14, alpha=0.75)
        ax.set_yscale("log")
        ax.set_xlabel(feat)
        ax.set_ylabel("k_lab")
        ax.grid(alpha=0.2)
    axes_flat[7].scatter(df["depth_m"], df["k_lab"], s=14, alpha=0.75, color="tab:purple")
    axes_flat[7].set_yscale("log")
    axes_flat[7].set_xlabel("depth_m")
    axes_flat[7].set_ylabel("k_lab")
    axes_flat[7].grid(alpha=0.2)
    axes_flat[8].scatter(df["depth_m"], df["phi_lab"], s=14, alpha=0.75, color="tab:red")
    axes_flat[8].set_xlabel("depth_m")
    axes_flat[8].set_ylabel("phi_lab")
    axes_flat[8].grid(alpha=0.2)
    fig.suptitle("Feature-target screening (k_lab, phi_lab)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    path = os.path.join(out_figures, "06_scatter_feature_target_screen.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(path)
    return saved


def _build_recommendations(df: pd.DataFrame, cfg: EDAConfig) -> Dict[str, object]:
    numeric = [
        "density",
        "gr",
        "res_deep",
        "res_shallow",
        "phi_neutron",
        "phi_sonic",
        "phi_nd",
        "phi_lab",
        "k_lab",
        "rqi",
        "fzi_lab",
    ]
    pearson = df[numeric].corr(method="pearson")
    target_corr = pearson[cfg.target].drop(cfg.target).sort_values(key=lambda x: x.abs(), ascending=False)

    # RQI and FZI_lab are kept in reports but flagged as leakage-prone model inputs.
    candidate_u_sets = {
        "u_petrophysical_full": [
            "density",
            "gr",
            "res_deep",
            "res_shallow",
            "phi_neutron",
            "phi_sonic",
            "phi_nd",
        ],
        "u_logs_no_phi": ["density", "gr", "res_deep", "res_shallow"],
        "u_minimal_res_gr": ["gr", "res_deep", "res_shallow"],
    }

    k_log = np.log10(np.clip(df["k_lab"].to_numpy(dtype=float), 1e-6, None))
    skew_k_log = float(pd.Series(k_log).skew())
    skew_k_raw = float(df["k_lab"].skew())

    rec = {
        "dataset_shape": {"n_rows": int(df.shape[0]), "n_cols": int(df.shape[1])},
        "primary_target": cfg.target,
        "target_skew": {
            "k_lab_raw_skew": round(skew_k_raw, 6),
            "k_lab_log10_skew": round(skew_k_log, 6),
        },
        "target_transform_advice": (
            "Use log10(k_lab) in smoke runs for stability, then back-transform for reporting."
            if cfg.target == "k_lab"
            else "No mandatory transform for phi_lab in first smoke run."
        ),
        "top_abs_pearson_corr_with_target": {
            k: round(float(v), 6) for k, v in target_corr.head(8).items()
        },
        "leakage_guardrails": [
            "Do not use rqi as u-channel in CLP/Direct-UB smoke runs.",
            "Do not use fzi_lab as u-channel in CLP/Direct-UB smoke runs.",
            "Keep rqi/fzi_lab only for descriptive EDA diagnostics.",
        ],
        "candidate_u_sets": candidate_u_sets,
        "runner_alignment": {
            "window_len": cfg.window_len,
            "step": cfg.step,
            "train_frac": cfg.train_frac,
            "val_frac": cfg.val_frac,
            "measurement_kind": "subsample",
            "rhos_suggested_small_data": [0.2, 0.3, 0.4, 0.5, 0.6],
            "measurement_noise_std_suggested": 0.01,
            "csgm_prior_type_initial": "ridge",
            "csgm_lambda_grid_initial": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
        },
    }
    return rec


def _write_protocol(cfg: EDAConfig, paths: Dict[str, str], table_files: Dict[str, str], figure_files: List[str]) -> None:
    lines = [
        "CLP-CSGM-oriented EDA protocol for Auddys dataset",
        "",
        "excel_path: {}".format(cfg.excel_path),
        "target_focus: {}".format(cfg.target),
        "run_id: {}".format(cfg.run_id),
        "",
        "Output structure:",
        "  run_root: {}".format(paths["run_root"]),
        "  tables:   {}".format(paths["tables"]),
        "  figures:  {}".format(paths["figures"]),
        "",
        "Runner-aligned defaults considered for next stage:",
        "  window_len={}".format(cfg.window_len),
        "  step={}".format(cfg.step),
        "  train_frac={}".format(cfg.train_frac),
        "  val_frac={}".format(cfg.val_frac),
        "  measurement_kind=subsample",
        "",
        "Tables generated:",
    ]
    for key, fp in sorted(table_files.items()):
        lines.append("  - {}: {}".format(key, fp))
    lines.append("")
    lines.append("Figures generated:")
    for fp in figure_files:
        lines.append("  - {}".format(fp))
    lines.append("")
    lines.append("Notes:")
    lines.append("  - EDA is descriptive and aligned to CLP-CSGM benchmark preparation.")
    lines.append("  - RQI and FZI_lab are treated as leakage-prone for model input u.")
    with open(os.path.join(paths["run_root"], "PROTOCOL.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_manifest(cfg: EDAConfig, paths: Dict[str, str], df: pd.DataFrame, df_legend: pd.DataFrame) -> None:
    dd = np.diff(df["depth_m"].to_numpy(dtype=float))
    lines = [
        "DATASET MANIFEST: Auddys CLP-CSGM EDA",
        "",
        "excel_path: {}".format(cfg.excel_path),
        "rows: {}".format(int(df.shape[0])),
        "columns: {}".format(int(df.shape[1])),
        "depth_min: {:.6f}".format(float(df["depth_m"].min())),
        "depth_max: {:.6f}".format(float(df["depth_m"].max())),
        "depth_step_median: {:.6f}".format(float(np.median(dd)) if dd.size else float("nan")),
        "depth_monotonic_increasing: {}".format(bool(np.all(dd > 0.0))),
        "",
        "column_names:",
    ]
    for c in df.columns:
        lines.append("  - {}".format(c))
    lines.append("")
    lines.append("legend_rows: {}".format(int(df_legend.shape[0])))
    with open(os.path.join(paths["run_root"], "DATASET_MANIFEST.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    cfg = _parse_args()
    paths = _prepare_dirs(cfg)
    df, df_legend = _load_data(cfg.excel_path)

    table_files = _write_tables(df, paths["tables"])
    figure_files: List[str] = []
    figure_files.extend(_plot_depth_tracks(df, paths["figures"]))
    figure_files.extend(_plot_correlations(df, paths["figures"]))
    figure_files.extend(_plot_class_views(df, paths["figures"]))
    figure_files.extend(_plot_scatter_screen(df, paths["figures"]))

    rec = _build_recommendations(df, cfg)
    rec_path = os.path.join(paths["run_root"], "recommendations_for_runner.json")
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)

    _write_protocol(cfg, paths, table_files, figure_files)
    _write_manifest(cfg, paths, df, df_legend)

    log_path = os.path.join(paths["logs"], "run_console.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Auddys CLP-CSGM EDA completed.\n")
        f.write("run_root={}\n".format(paths["run_root"]))
        f.write("tables={}\n".format(len(table_files)))
        f.write("figures={}\n".format(len(figure_files)))
        f.write("recommendations={}\n".format(rec_path))

    print("EDA_COMPLETE")
    print("RUN_ROOT", paths["run_root"])
    print("TABLE_FILES", len(table_files))
    print("FIGURE_FILES", len(figure_files))
    print("RECOMMENDATIONS", rec_path)


if __name__ == "__main__":
    main()
