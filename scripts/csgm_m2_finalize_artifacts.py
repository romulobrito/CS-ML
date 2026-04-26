#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finalize CSGM M2 grid artifacts using the project layout.

The output follows the same structure used by direct_ub robustness runs:

  outputs/cross_well_vc/csgm/m2_grid/
    tables/
    figures/
    logs/
    PROTOCOL.txt
    DATASET_MANIFEST.txt
    RUN_MANIFEST.txt
    config.json

It also exports paper-ready assets to:

  paper/figures/csgm_m2_grid/
  paper/experiments_csgm_m2.tex

ASCII-only.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))

RUN_ROOT = os.path.join(_REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid")
TABLES_DIR = os.path.join(RUN_ROOT, "tables")
FIGURES_DIR = os.path.join(RUN_ROOT, "figures")
LOGS_DIR = os.path.join(RUN_ROOT, "logs")
PAPER_DIR = os.path.join(_REPO_ROOT, "paper")
PAPER_FIG_DIR = os.path.join(PAPER_DIR, "figures", "csgm_m2_grid")

CSGM_RAW = os.path.join(TABLES_DIR, "csgm_m2_detailed.csv")
BENCH_STEPS: Tuple[int, ...] = (8, 16, 32)
RHOS: Tuple[float, ...] = (0.05, 0.10, 0.20)
SEEDS: Tuple[int, ...] = (7, 23, 41)
TRAIN_SIZE_BY_STEP: Dict[int, int] = {8: 104, 16: 52, 32: 27}

METHOD_ORDER: Tuple[str, ...] = (
    "ridge_prior_csgm",
    "ae_regression_ub",
    "ml_only",
    "mlp_concat_ub",
    "pca_regression_ub",
)

METHOD_LABELS: Dict[str, str] = {
    "ridge_prior_csgm": "CLP-CSGM Ridge",
    "ae_regression_ub": "AE [u,b]",
    "ml_only": "ML only",
    "mlp_concat_ub": "MLP [u,b]",
    "pca_regression_ub": "PCA [u,b]",
}

METHOD_COLORS: Dict[str, str] = {
    "ridge_prior_csgm": "#d62728",
    "ae_regression_ub": "#17becf",
    "ml_only": "#1f77b4",
    "mlp_concat_ub": "#2ca02c",
    "pca_regression_ub": "#9467bd",
}


def ensure_dirs() -> None:
    """Create standard artifact folders."""
    for path in (TABLES_DIR, FIGURES_DIR, LOGS_DIR, PAPER_FIG_DIR):
        os.makedirs(path, exist_ok=True)


def read_csv_required(path: str) -> pd.DataFrame:
    """Read a required CSV file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def benchmark_summary_by_seed(step: int) -> pd.DataFrame:
    """Read the existing low-data benchmark per-seed table for one step."""
    path = os.path.join(
        _REPO_ROOT,
        "outputs",
        "cross_well_vc",
        "sir_cs_stress_lowdata",
        "runs",
        "prod_step{:02d}".format(int(step)),
        "tables",
        "summary_by_seed.csv",
    )
    df = read_csv_required(path)
    df = df.copy()
    df["step"] = int(step)
    df["n_train_approx"] = TRAIN_SIZE_BY_STEP[int(step)]
    return df


def build_per_seed() -> pd.DataFrame:
    """Combine CSGM and benchmark per-seed metrics."""
    csgm = read_csv_required(CSGM_RAW).copy()
    csgm = csgm.rename(columns={"rel_l2": "relative_l2_mean"})
    csgm["rmse_mean"] = csgm["rmse"]
    csgm["mae_mean"] = csgm["mae"]
    csgm["support_f1_mean"] = np.nan
    csgm["n_test_samples"] = csgm["n_test"]
    csgm["n_train_approx"] = csgm["step"].map(TRAIN_SIZE_BY_STEP).astype(int)
    csgm = csgm[
        [
            "step",
            "n_train_approx",
            "seed",
            "measurement_ratio",
            "method",
            "rmse_mean",
            "mae_mean",
            "relative_l2_mean",
            "support_f1_mean",
            "n_test_samples",
            "lambda",
            "val_rmse_selected",
            "ae_recon_train_rmse",
        ]
    ].copy()

    frames: List[pd.DataFrame] = [csgm]
    for step in BENCH_STEPS:
        b = benchmark_summary_by_seed(step)
        b = b[b["method"].isin(METHOD_ORDER)].copy()
        b["lambda"] = np.nan
        b["val_rmse_selected"] = np.nan
        b["ae_recon_train_rmse"] = np.nan
        b = b[
            [
                "step",
                "n_train_approx",
                "seed",
                "measurement_ratio",
                "method",
                "rmse_mean",
                "mae_mean",
                "relative_l2_mean",
                "support_f1_mean",
                "n_test_samples",
                "lambda",
                "val_rmse_selected",
                "ae_recon_train_rmse",
            ]
        ].copy()
        frames.append(b)
    out = pd.concat(frames, ignore_index=True)
    out["method"] = pd.Categorical(out["method"], categories=list(METHOD_ORDER), ordered=True)
    return out.sort_values(["step", "seed", "measurement_ratio", "method"]).reset_index(drop=True)


def sem(values: pd.Series) -> float:
    """Standard error of the mean."""
    if len(values) <= 1:
        return 0.0
    return float(values.std(ddof=1) / np.sqrt(float(len(values))))


def summarize(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed metrics to project-style summary.csv."""
    g = (
        per_seed.groupby(
            ["step", "n_train_approx", "measurement_ratio", "method"],
            dropna=False,
            observed=True,
        )
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std_across_seeds=("rmse_mean", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            rmse_sem=("rmse_mean", sem),
            mae_mean=("mae_mean", "mean"),
            mae_std_across_seeds=("mae_mean", lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0),
            relative_l2_mean=("relative_l2_mean", "mean"),
            relative_l2_std_across_seeds=(
                "relative_l2_mean",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
            support_f1_mean=("support_f1_mean", "mean"),
            support_f1_std_across_seeds=(
                "support_f1_mean",
                lambda s: float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            ),
            lambda_mean=("lambda", "mean"),
            val_rmse_selected_mean=("val_rmse_selected", "mean"),
            n_seeds=("seed", "nunique"),
            n_test_samples_per_run=("n_test_samples", "first"),
        )
        .reset_index()
    )
    g["rmse_ci95_half"] = 1.96 * g["rmse_sem"]
    g["method"] = pd.Categorical(g["method"], categories=list(METHOD_ORDER), ordered=True)
    return g.sort_values(["step", "measurement_ratio", "method"]).reset_index(drop=True)


def build_winners(summary: pd.DataFrame) -> pd.DataFrame:
    """Build per-cell CSGM-vs-AE comparison."""
    rows: List[Dict[str, object]] = []
    for (step, rho), sub in summary.groupby(["step", "measurement_ratio"], observed=True):
        sub2 = sub.sort_values("rmse_mean")
        winner = sub2.iloc[0]
        csgm = sub[sub["method"] == "ridge_prior_csgm"]
        ae = sub[sub["method"] == "ae_regression_ub"]
        if csgm.empty or ae.empty:
            continue
        c_rmse = float(csgm["rmse_mean"].iloc[0])
        ae_rmse = float(ae["rmse_mean"].iloc[0])
        gap = 100.0 * (c_rmse - ae_rmse) / max(ae_rmse, 1e-12)
        rows.append(
            {
                "step": int(step),
                "n_train_approx": TRAIN_SIZE_BY_STEP[int(step)],
                "measurement_ratio": float(rho),
                "winner_method": str(winner["method"]),
                "winner_rmse": float(winner["rmse_mean"]),
                "csgm_rmse": c_rmse,
                "ae_rmse": ae_rmse,
                "csgm_gap_vs_ae_pct": gap,
                "csgm_wins_vs_ae": bool(c_rmse < ae_rmse),
            }
        )
    return pd.DataFrame(rows).sort_values(["step", "measurement_ratio"])


def overall_rank(summary: pd.DataFrame) -> pd.DataFrame:
    """Rank methods across all step x rho cells."""
    out = (
        summary.groupby("method", observed=True)
        .agg(
            rmse_mean_all_cells=("rmse_mean", "mean"),
            relative_l2_mean_all_cells=("relative_l2_mean", "mean"),
            cells=("rmse_mean", "count"),
        )
        .reset_index()
        .sort_values("rmse_mean_all_cells")
    )
    out["method_label"] = out["method"].map(METHOD_LABELS)
    return out


def save_metric_plot(summary: pd.DataFrame, metric: str, std_col: str, ylabel: str, filename: str) -> str:
    """Save project-style metric-vs-rho panels by low-data step."""
    steps = sorted(summary["step"].unique())
    rhos = sorted(summary["measurement_ratio"].unique())
    fig, axes = plt.subplots(1, len(steps), figsize=(4.8 * len(steps), 4.4), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        sstep = summary[summary["step"] == step]
        for method in METHOD_ORDER:
            sub = sstep[sstep["method"] == method].sort_values("measurement_ratio")
            if sub.empty:
                continue
            ax.errorbar(
                sub["measurement_ratio"].values,
                sub[metric].values,
                yerr=sub[std_col].fillna(0.0).values,
                marker="o",
                capsize=3,
                linewidth=2.3 if method == "ridge_prior_csgm" else 1.4,
                alpha=1.0 if method in ("ridge_prior_csgm", "ae_regression_ub", "ml_only") else 0.55,
                color=METHOD_COLORS.get(method),
                label=METHOD_LABELS.get(method, method),
            )
        ax.set_title("step={} (n_train~{})".format(int(step), TRAIN_SIZE_BY_STEP[int(step)]))
        ax.set_xlabel("measurement_ratio rho")
        ax.set_xticks(rhos)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(loc="upper right", fontsize=7)
    fig.suptitle("CLP-CSGM Ridge vs direct [u,b] baselines", y=1.02)
    out = os.path.join(FIGURES_DIR, filename)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_gain_plot(winners: pd.DataFrame) -> str:
    """Save CSGM gain over AE plot."""
    steps = sorted(winners["step"].unique())
    rhos = sorted(winners["measurement_ratio"].unique())
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    width = 0.25
    x = np.arange(len(rhos), dtype=np.float64)
    for idx, step in enumerate(steps):
        sub = winners[winners["step"] == step].sort_values("measurement_ratio")
        gain = -sub["csgm_gap_vs_ae_pct"].values
        ax.bar(
            x + (idx - 1) * width,
            gain,
            width,
            label="step={} (n_train~{})".format(int(step), TRAIN_SIZE_BY_STEP[int(step)]),
        )
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rhos])
    ax.set_xlabel("measurement_ratio rho")
    ax.set_ylabel("RMSE gain over ae_regression_ub (%)")
    ax.set_title("CLP-CSGM Ridge gain over strongest direct AE baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    out = os.path.join(FIGURES_DIR, "04_csgm_gain_vs_ae.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_rank_plot(rank: pd.DataFrame) -> str:
    """Save overall RMSE ranking bar plot."""
    data = rank.sort_values("rmse_mean_all_cells")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    x = np.arange(data.shape[0])
    labels = [METHOD_LABELS.get(str(m), str(m)) for m in data["method"].values]
    colors = [METHOD_COLORS.get(str(m), "#777777") for m in data["method"].values]
    ax.bar(x, data["rmse_mean_all_cells"].values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Mean RMSE across all step x rho cells")
    ax.set_title("Overall CLP-CSGM Ridge ranking")
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(FIGURES_DIR, "05_overall_rmse_ranking.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def save_lambda_plot(summary: pd.DataFrame) -> str:
    """Save validation-selected lambda plot."""
    sub = summary[summary["method"] == "ridge_prior_csgm"].copy()
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    for step in sorted(sub["step"].unique()):
        s = sub[sub["step"] == step].sort_values("measurement_ratio")
        ax.plot(
            s["measurement_ratio"].values,
            s["lambda_mean"].values,
            marker="o",
            linewidth=1.8,
            label="step={} (n_train~{})".format(int(step), TRAIN_SIZE_BY_STEP[int(step)]),
        )
    ax.set_yscale("log")
    ax.set_xlabel("measurement_ratio rho")
    ax.set_ylabel("lambda selected on validation")
    ax.set_title("CLP-CSGM Ridge lambda selection")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=8)
    out = os.path.join(FIGURES_DIR, "06_lambda_vs_measurement_ratio.png")
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def write_protocol(winners: pd.DataFrame) -> str:
    """Write methodological protocol."""
    wins = int(winners["csgm_wins_vs_ae"].sum())
    cells = int(winners.shape[0])
    avg_gap = float(winners["csgm_gap_vs_ae_pct"].mean())
    lines = [
        "Conditional Latent-Prior CSGM (CLP-CSGM) cross-well Vc benchmark protocol",
        "",
        "0) Dataset: F02-1,F03-2,F06-1 train wells and F03-4 held-out test well.",
        "1) Target: Vc windows, L=64, noisy 30dB logs.",
        "2) u channels: sonic,rhob,gr,ai,vp.",
        "3) Low-data axis: step in {8,16,32}; resulting n_train approx {104,52,27}.",
        "4) Measurement model: b = M y + eta, M is coordinate subsampling.",
        "5) measurement_ratio rho in {0.05,0.10,0.20}; seeds {7,23,41}.",
        "6) CLP-CSGM Ridge: AE decoder G(z) trained on standardized Y_train only.",
        "7) Ridge prior h(u) maps u to z0 using encoded train latents z=E(y).",
        "8) For each validation/test row, solve:",
        "   z_hat = argmin_z ||M G(z) - b||_2^2 + lambda ||z - z0(u)||_2^2.",
        "9) Lambda is selected on validation RMSE per (step, seed, rho), then evaluated on test.",
        "10) Benchmarks are the existing low-data direct [u,b]->y summaries:",
        "    ml_only, mlp_concat_ub, pca_regression_ub, ae_regression_ub.",
        "",
        "Decision summary:",
        "  CSGM wins vs ae_regression_ub: {} / {}".format(wins, cells),
        "  Average CSGM gap vs ae_regression_ub: {:+.2f}%".format(avg_gap),
    ]
    path = os.path.join(RUN_ROOT, "PROTOCOL.txt")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_dataset_manifest(summary: pd.DataFrame) -> str:
    """Write dataset manifest."""
    lines = [
        "CLP-CSGM Ridge cross-well Vc dataset manifest",
        "",
        "train_path: data/F02-1,F03-2,F06-1_6logs_30dB.txt",
        "test_path:  data/F03-4_6logs_30dB.txt",
        "target: VC",
        "channels: SONIC, RHOB, GR, AI, VP",
        "window_len: 64",
        "steps: " + ",".join(str(x) for x in sorted(summary["step"].unique())),
        "measurement_ratios: " + ",".join(str(x) for x in sorted(summary["measurement_ratio"].unique())),
        "seeds: " + ",".join(str(x) for x in SEEDS),
        "",
        "Note: each step defines a different sliding-window density and therefore a distinct low-data regime.",
    ]
    path = os.path.join(RUN_ROOT, "DATASET_MANIFEST.txt")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
    return path


def write_config(summary: pd.DataFrame) -> str:
    """Write JSON config for reproducibility."""
    cfg = {
        "run_id": "m2_grid",
        "method": "ridge_prior_csgm",
        "target": "vc",
        "channels": ["sonic", "rhob", "gr", "ai", "vp"],
        "window_len": 64,
        "steps": [int(x) for x in sorted(summary["step"].unique())],
        "measurement_ratios": [float(x) for x in sorted(summary["measurement_ratio"].unique())],
        "seeds": [int(x) for x in SEEDS],
        "measurement_kind": "subsample",
        "measurement_noise_std": 0.005,
        "ae_latent_dim": 16,
        "ae_hidden": 128,
        "ae_epochs": 200,
        "csgm_iters": 400,
        "csgm_lr": 0.05,
        "csgm_restarts": 3,
        "lambda_grid": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1],
        "tables_dir": TABLES_DIR,
        "figures_dir": FIGURES_DIR,
        "paper_figures_dir": PAPER_FIG_DIR,
    }
    path = os.path.join(RUN_ROOT, "config.json")
    with open(path, "w", encoding="ascii") as f:
        json.dump(cfg, f, indent=2)
    return path


def copy_paper_figures(figures: List[str]) -> List[str]:
    """Copy generated figures to paper/figures/csgm_m2_grid."""
    copied: List[str] = []
    for fig in figures:
        dst = os.path.join(PAPER_FIG_DIR, os.path.basename(fig))
        shutil.copy2(fig, dst)
        copied.append(dst)
    return copied


def write_paper_tex(winners: pd.DataFrame, rank: pd.DataFrame) -> str:
    """Write a paper-ready LaTeX section fragment."""
    csgm = rank[rank["method"] == "ridge_prior_csgm"].iloc[0]
    ae = rank[rank["method"] == "ae_regression_ub"].iloc[0]
    rel_gap = 100.0 * (
        float(csgm["rmse_mean_all_cells"]) - float(ae["rmse_mean_all_cells"])
    ) / max(float(ae["rmse_mean_all_cells"]), 1e-12)
    rows: List[str] = []
    for _, row in winners.iterrows():
        rows.append(
            "    ${}$ & ${:.2f}$ & ${:.5f}$ & ${:.5f}$ & ${:+.2f}\\%$ \\\\".format(
                int(row["step"]),
                float(row["measurement_ratio"]),
                float(row["csgm_rmse"]),
                float(row["ae_rmse"]),
                float(row["csgm_gap_vs_ae_pct"]),
            )
        )
    table_body = "\n".join(rows)
    text = r"""% CLP-CSGM Ridge cross-well Vc section fragment.
% Generated by scripts/csgm_m2_finalize_artifacts.py.

\section{Conditional CSGM for sparse core assimilation}
\label{sec:csgm_m2}

We evaluate a conditional compressed-sensing-with-generative-models (CSGM)
variant on the cross-well Vc low-data benchmark. The decoder $G(z)$ is trained
as an autoencoder prior over Vc windows, while a ridge model predicts a latent
prior $z_0=h(u)$ from the wireline logs. Sparse measurements enter through a
test-time inverse problem,
\[
  \hat z = \arg\min_z \|M G(z)-b\|_2^2
  + \lambda \|z-z_0(u)\|_2^2,
  \qquad \hat y = G(\hat z).
\]
This differs from direct \([u,b]\to y\) baselines because \(b\) is enforced as
a measurement-consistency constraint rather than treated only as an input
feature.

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/csgm_m2_grid/01_rmse_vs_measurement_ratio.png}
  \caption{CLP-CSGM Ridge versus direct baselines on cross-well Vc low-data regimes.
  Each panel fixes the sliding-window step and varies the measurement ratio.}
  \label{fig:csgm_m2_rmse}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=0.82\linewidth]{figures/csgm_m2_grid/04_csgm_gain_vs_ae.png}
  \caption{RMSE gain of CLP-CSGM Ridge over the strongest previous baseline,
  \texttt{ae\_regression\_ub}. Positive values indicate CLP-CSGM improvement.}
  \label{fig:csgm_m2_gain}
\end{figure}

\begin{table}[t]
  \centering
  \caption{CLP-CSGM Ridge versus \texttt{ae\_regression\_ub} by low-data step and
  measurement ratio. Gap is relative RMSE difference of CLP-CSGM against AE; negative
  values mean CLP-CSGM is better.}
  \label{tab:csgm_m2_vs_ae}
  \small
  \begin{tabular}{@{}ccccc@{}}
    \toprule
    step & $\rho$ & CSGM RMSE & AE RMSE & gap vs AE \\
    \midrule
""" + table_body + r"""
    \bottomrule
  \end{tabular}
\end{table}

Across all step--rho cells, CLP-CSGM Ridge obtains mean RMSE """ + "{:.5f}".format(float(csgm["rmse_mean_all_cells"])) + r""",
compared with """ + "{:.5f}".format(float(ae["rmse_mean_all_cells"])) + r""" for
\texttt{ae\_regression\_ub}, a relative gap of """ + "{:+.2f}".format(rel_gap) + r"""\%.
"""
    path = os.path.join(PAPER_DIR, "experiments_csgm_m2.tex")
    with open(path, "w", encoding="ascii") as f:
        f.write(text)
    return path


def write_manifest(elapsed: float, artifacts: List[str]) -> str:
    """Write run manifest in the same style as prior runs."""
    lines = [
        "CLP-CSGM Ridge cross-well Vc benchmark.",
        "run_id: m2_grid",
        "elapsed_seconds: {:.1f}".format(elapsed),
        "measurement_kind: subsample",
        "measurement_noise_std: 0.005",
        "",
        "tables/",
    ]
    for name in (
        "detailed_results.csv",
        "summary_by_seed.csv",
        "summary.csv",
        "summary_focus_csgm_vs_ub.csv",
        "csgm_vs_ae_winners.csv",
        "overall_rank.csv",
        "pivot_rmse.csv",
        "pivot_relative_l2.csv",
        "pivot_rmse.tex",
    ):
        lines.append("  " + name)
    lines.append("")
    lines.append("figures/")
    for fig in sorted(os.listdir(FIGURES_DIR)):
        if fig.endswith(".png"):
            lines.append("  figures/" + fig)
    lines.append("")
    lines.append("paper_assets/")
    lines.append("  paper/figures/csgm_m2_grid/")
    lines.append("  paper/experiments_csgm_m2.tex")
    lines.append("")
    lines.append("all_artifacts:")
    for path in artifacts:
        lines.append("  " + os.path.relpath(path, RUN_ROOT))
    path = os.path.join(RUN_ROOT, "RUN_MANIFEST.txt")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    """Finalize all CLP-CSGM Ridge artifacts."""
    t0 = time.time()
    ensure_dirs()
    per_seed = build_per_seed()
    summary = summarize(per_seed)
    focus = summary[summary["method"].isin(METHOD_ORDER)].copy()
    winners = build_winners(summary)
    rank = overall_rank(summary)

    artifacts: List[str] = []
    detailed_path = os.path.join(TABLES_DIR, "detailed_results.csv")
    per_seed.to_csv(detailed_path, index=False)
    artifacts.append(detailed_path)

    by_seed_path = os.path.join(TABLES_DIR, "summary_by_seed.csv")
    per_seed.to_csv(by_seed_path, index=False)
    artifacts.append(by_seed_path)

    summary_path = os.path.join(TABLES_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    artifacts.append(summary_path)

    focus_path = os.path.join(TABLES_DIR, "summary_focus_clp_csgm_vs_ub.csv")
    focus.to_csv(focus_path, index=False)
    artifacts.append(focus_path)

    winners_path = os.path.join(TABLES_DIR, "csgm_vs_ae_winners.csv")
    winners.to_csv(winners_path, index=False)
    artifacts.append(winners_path)

    rank_path = os.path.join(TABLES_DIR, "overall_rank.csv")
    rank.to_csv(rank_path, index=False)
    artifacts.append(rank_path)

    pivot_rmse = summary.pivot_table(
        index=["step", "measurement_ratio"],
        columns="method",
        values="rmse_mean",
        aggfunc="mean",
        observed=True,
    ).round(5)
    pivot_rmse_path = os.path.join(TABLES_DIR, "pivot_rmse.csv")
    pivot_rmse.to_csv(pivot_rmse_path)
    artifacts.append(pivot_rmse_path)

    pivot_rel = summary.pivot_table(
        index=["step", "measurement_ratio"],
        columns="method",
        values="relative_l2_mean",
        aggfunc="mean",
        observed=True,
    ).round(5)
    pivot_rel_path = os.path.join(TABLES_DIR, "pivot_relative_l2.csv")
    pivot_rel.to_csv(pivot_rel_path)
    artifacts.append(pivot_rel_path)

    tex_path = os.path.join(TABLES_DIR, "pivot_rmse.tex")
    with open(tex_path, "w", encoding="ascii") as f:
        f.write("% CLP-CSGM Ridge RMSE pivot table.\n")
        f.write(pivot_rmse.to_latex(float_format="%.5f"))
    artifacts.append(tex_path)

    figures = [
        save_metric_plot(
            summary,
            "rmse_mean",
            "rmse_std_across_seeds",
            "RMSE on held-out F03-4",
            "01_rmse_vs_measurement_ratio.png",
        ),
        save_metric_plot(
            summary,
            "mae_mean",
            "mae_std_across_seeds",
            "MAE on held-out F03-4",
            "02_mae_vs_measurement_ratio.png",
        ),
        save_metric_plot(
            summary,
            "relative_l2_mean",
            "relative_l2_std_across_seeds",
            "Relative L2 on held-out F03-4",
            "03_relative_l2_vs_measurement_ratio.png",
        ),
        save_gain_plot(winners),
        save_rank_plot(rank),
        save_lambda_plot(summary),
    ]
    artifacts.extend(figures)

    protocol = write_protocol(winners)
    dataset_manifest = write_dataset_manifest(summary)
    config_path = write_config(summary)
    artifacts.extend([protocol, dataset_manifest, config_path])

    copied = copy_paper_figures(figures)
    paper_tex = write_paper_tex(winners, rank)
    artifacts.extend(copied + [paper_tex])

    manifest = write_manifest(time.time() - t0, artifacts)
    artifacts.append(manifest)

    print("CLP-CSGM Ridge artifacts finalized.")
    print("Run root:", RUN_ROOT)
    print("Figures:", FIGURES_DIR)
    print("Paper figures:", PAPER_FIG_DIR)
    print("Paper tex:", paper_tex)
    print("")
    print("CSGM vs AE:")
    print(winners.to_string(index=False))
    print("")
    print("Overall rank:")
    print(rank.to_string(index=False))
    print("")
    print("Manifest:", manifest)


if __name__ == "__main__":
    main()
