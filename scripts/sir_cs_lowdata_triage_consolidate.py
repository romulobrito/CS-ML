#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIR-CS low-data triage consolidator (Teste A).

Reads the three step-stratified production runs of the cross-well Vc benchmark
(prod_step08, prod_step16, prod_step32) and produces decision artifacts:

  - tables/lowdata_long.csv:   long-format with step tag and all methods.
  - tables/lowdata_focus.csv:  focus methods (ml_only, ae_regression_ub,
                               hybrid_lfista_joint).
  - tables/lowdata_pivot_rmse.csv:        RMSE pivot (step x rho x method).
  - tables/lowdata_pivot_relL2.csv:       Relative L2 pivot.
  - tables/lowdata_winners.csv:           per-cell winner (lowest RMSE).
  - tables/lowdata_summary_rank.csv:      mean RMSE per (step, method).
  - tables/lowdata_decision.txt:          go/no-go for JRM with rationale.
  - figures/lowdata_rmse_vs_rho.png:      3 subplots, one per step.
  - figures/lowdata_relL2_vs_rho.png:     same in Relative L2.
  - figures/lowdata_lfista_gap_to_winner.png: SIR-CS gap to the cell winner.

Run from repo root:
  python scripts/sir_cs_lowdata_triage_consolidate.py

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
    _REPO_ROOT, "outputs", "cross_well_vc", "sir_cs_stress_lowdata", "runs"
)
RUN_IDS: Tuple[Tuple[int, str], ...] = (
    (8, "prod_step08"),
    (16, "prod_step16"),
    (32, "prod_step32"),
)
OUT_DIR = os.path.join(RUNS_DIR, "_aggregate")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")

FOCUS_METHODS: Tuple[str, ...] = (
    "ml_only",
    "ae_regression_ub",
    "hybrid_lfista_joint",
)
ALL_METHODS: Tuple[str, ...] = (
    "ml_only",
    "ae_regression_ub",
    "mlp_concat_ub",
    "pca_regression_ub",
    "hybrid_lfista_joint",
)
SIR_CS_NAME = "hybrid_lfista_joint"
TRAIN_SIZE_BY_STEP: Dict[int, int] = {8: 104, 16: 52, 32: 27}


def _read_summary(run_id: str) -> pd.DataFrame:
    path = os.path.join(RUNS_DIR, run_id, "tables", "summary.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_pivot(
    df: pd.DataFrame, value_col: str
) -> pd.DataFrame:
    return df.pivot_table(
        index=["step", "measurement_ratio"],
        columns="method",
        values=value_col,
        aggfunc="mean",
    ).round(5)


def _winners_table(focus: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for (step, rho), sub in focus.groupby(["step", "measurement_ratio"]):
        sub2 = sub.sort_values("rmse_mean")
        if sub2.empty:
            continue
        winner = sub2.iloc[0]
        sir_cs = sub[sub["method"] == SIR_CS_NAME]
        sir_rmse = float(sir_cs["rmse_mean"].iloc[0]) if not sir_cs.empty else float("nan")
        gap_abs = float(sir_rmse - float(winner["rmse_mean"]))
        gap_rel_pct = (
            100.0 * gap_abs / float(winner["rmse_mean"])
            if float(winner["rmse_mean"]) > 0.0
            else float("nan")
        )
        rows.append(
            {
                "step": int(step),
                "n_train_approx": TRAIN_SIZE_BY_STEP.get(int(step), -1),
                "measurement_ratio": float(rho),
                "winner_method": str(winner["method"]),
                "winner_rmse": float(winner["rmse_mean"]),
                "sir_cs_rmse": sir_rmse,
                "sir_cs_gap_abs": gap_abs,
                "sir_cs_gap_pct": gap_rel_pct,
                "sir_cs_is_winner": bool(str(winner["method"]) == SIR_CS_NAME),
            }
        )
    return pd.DataFrame(rows).sort_values(["step", "measurement_ratio"])


def _decision_text(winners: pd.DataFrame) -> str:
    n_cells = int(winners.shape[0])
    n_sir_wins = int(winners["sir_cs_is_winner"].sum())
    n_sir_within_5 = int((winners["sir_cs_gap_pct"] <= 5.0).sum())
    n_sir_within_10 = int((winners["sir_cs_gap_pct"] <= 10.0).sum())
    avg_gap = float(winners["sir_cs_gap_pct"].mean())
    min_gap = float(winners["sir_cs_gap_pct"].min())

    if n_sir_wins >= 1:
        verdict = "GO for JRM (SIR-CS already wins in {} cells).".format(n_sir_wins)
    elif n_sir_within_5 >= 2:
        verdict = (
            "GO for JRM (SIR-CS is within 5% of winner in {} cells; "
            "JRM may close the gap).".format(n_sir_within_5)
        )
    elif n_sir_within_10 >= 2:
        verdict = (
            "WEAK GO for JRM (SIR-CS is within 10% in {} cells; "
            "JRM gain uncertain, consider Tier 2 first).".format(n_sir_within_10)
        )
    else:
        verdict = (
            "NO-GO for JRM in this task family (SIR-CS gap too large "
            "in all cells; pivot to a different y or scenario)."
        )

    lines = [
        "SIR-CS low-data triage decision report",
        "=" * 60,
        "",
        "Total cells (step x rho): {}".format(n_cells),
        "Cells where SIR-CS is the winner: {}".format(n_sir_wins),
        "Cells where SIR-CS is within 5% of winner: {}".format(n_sir_within_5),
        "Cells where SIR-CS is within 10% of winner: {}".format(n_sir_within_10),
        "Average SIR-CS gap to winner (%): {:.2f}".format(avg_gap),
        "Minimum SIR-CS gap to winner (%): {:.2f}".format(min_gap),
        "",
        "VERDICT: {}".format(verdict),
        "",
        "Per-cell breakdown:",
    ]
    for _, r in winners.iterrows():
        lines.append(
            "  step={:>2} (n_tr~{:>3}) rho={:.2f} | winner={:<22} "
            "rmse_win={:.4f} sir_cs={:.4f} gap={:.2f}%".format(
                int(r["step"]),
                int(r["n_train_approx"]),
                float(r["measurement_ratio"]),
                str(r["winner_method"]),
                float(r["winner_rmse"]),
                float(r["sir_cs_rmse"]),
                float(r["sir_cs_gap_pct"]),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    rows: List[pd.DataFrame] = []
    missing: List[str] = []
    for step, run_id in RUN_IDS:
        try:
            df = _read_summary(run_id)
        except FileNotFoundError as ex:
            missing.append(str(ex))
            continue
        df = df.copy()
        df["step"] = int(step)
        df["run_id"] = run_id
        df["n_train_approx"] = TRAIN_SIZE_BY_STEP.get(int(step), -1)
        rows.append(df)

    if not rows:
        print("ERROR: no run summaries found. Missing files:")
        for m in missing:
            print("  " + m)
        sys.exit(1)
    if missing:
        print("Warning: some runs missing:")
        for m in missing:
            print("  " + m)

    long_df = pd.concat(rows, ignore_index=True)
    long_df = long_df[long_df["method"].isin(ALL_METHODS)].copy()
    long_path = os.path.join(TABLES_DIR, "lowdata_long.csv")
    long_df.to_csv(long_path, index=False)
    print("Saved:", long_path)

    focus = long_df[long_df["method"].isin(FOCUS_METHODS)].copy()
    focus_path = os.path.join(TABLES_DIR, "lowdata_focus.csv")
    focus.to_csv(focus_path, index=False)
    print("Saved:", focus_path)

    pv_rmse = _safe_pivot(long_df, "rmse_mean")
    pv_rmse_path = os.path.join(TABLES_DIR, "lowdata_pivot_rmse.csv")
    pv_rmse.to_csv(pv_rmse_path)
    print("Saved:", pv_rmse_path)

    pv_rel = _safe_pivot(long_df, "relative_l2_mean")
    pv_rel_path = os.path.join(TABLES_DIR, "lowdata_pivot_relL2.csv")
    pv_rel.to_csv(pv_rel_path)
    print("Saved:", pv_rel_path)

    winners = _winners_table(focus)
    win_path = os.path.join(TABLES_DIR, "lowdata_winners.csv")
    winners.to_csv(win_path, index=False)
    print("Saved:", win_path)

    rank_rows: List[Dict[str, object]] = []
    for step in sorted(long_df["step"].unique()):
        for m in FOCUS_METHODS:
            sub = long_df[(long_df["step"] == step) & (long_df["method"] == m)]
            if sub.empty:
                continue
            rank_rows.append(
                {
                    "step": int(step),
                    "n_train_approx": TRAIN_SIZE_BY_STEP.get(int(step), -1),
                    "method": m,
                    "rmse_mean_over_rhos": float(sub["rmse_mean"].mean()),
                    "relL2_mean_over_rhos": float(sub["relative_l2_mean"].mean()),
                    "rmse_std_avg": float(sub["rmse_std_across_seeds"].fillna(0.0).mean()),
                }
            )
    rank_df = pd.DataFrame(rank_rows).sort_values(["step", "rmse_mean_over_rhos"])
    rank_path = os.path.join(TABLES_DIR, "lowdata_summary_rank.csv")
    rank_df.to_csv(rank_path, index=False)
    print("Saved:", rank_path)

    decision = _decision_text(winners)
    dec_path = os.path.join(TABLES_DIR, "lowdata_decision.txt")
    with open(dec_path, "w", encoding="ascii") as f:
        f.write(decision)
    print("Saved:", dec_path)
    print()
    print(decision)

    rhos = sorted(long_df["measurement_ratio"].unique())
    steps = sorted(long_df["step"].unique())
    method_colors = {
        "ml_only": "#1f77b4",
        "ae_regression_ub": "#2ca02c",
        "hybrid_lfista_joint": "#d62728",
        "mlp_concat_ub": "#9467bd",
        "pca_regression_ub": "#ff7f0e",
    }
    method_label = {
        "ml_only": "ml_only",
        "ae_regression_ub": "ae_regression_ub",
        "hybrid_lfista_joint": "SIR-CS (hybrid_lfista_joint)",
        "mlp_concat_ub": "mlp_concat_ub",
        "pca_regression_ub": "pca_regression_ub",
    }

    fig, axes = plt.subplots(1, len(steps), figsize=(4.6 * len(steps), 4.4), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        sub_step = long_df[long_df["step"] == step]
        for m in ALL_METHODS:
            r = sub_step[sub_step["method"] == m].sort_values("measurement_ratio")
            if r.empty:
                continue
            ax.errorbar(
                r["measurement_ratio"].values,
                r["rmse_mean"].values,
                yerr=r["rmse_std_across_seeds"].fillna(0.0).values,
                marker="o",
                linewidth=1.6 if m in FOCUS_METHODS else 1.0,
                capsize=3,
                color=method_colors.get(m, "#777"),
                label=method_label.get(m, m),
                alpha=1.0 if m in FOCUS_METHODS else 0.55,
            )
        ax.set_title(
            "step={} (n_train ~ {})".format(step, TRAIN_SIZE_BY_STEP.get(int(step), -1))
        )
        ax.set_xlabel("measurement_ratio rho")
        ax.set_xticks(rhos)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("RMSE on test well (mean across seeds)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Teste A | SIR-CS low-data triage (cross-well Vc, F03-4 held out, DCT)",
        y=1.02,
    )
    p1 = os.path.join(FIGURES_DIR, "lowdata_rmse_vs_rho.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p1)

    fig, axes = plt.subplots(1, len(steps), figsize=(4.6 * len(steps), 4.4), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        sub_step = long_df[long_df["step"] == step]
        for m in ALL_METHODS:
            r = sub_step[sub_step["method"] == m].sort_values("measurement_ratio")
            if r.empty:
                continue
            ax.errorbar(
                r["measurement_ratio"].values,
                r["relative_l2_mean"].values,
                yerr=r["relative_l2_std_across_seeds"].fillna(0.0).values,
                marker="o",
                linewidth=1.6 if m in FOCUS_METHODS else 1.0,
                capsize=3,
                color=method_colors.get(m, "#777"),
                label=method_label.get(m, m),
                alpha=1.0 if m in FOCUS_METHODS else 0.55,
            )
        ax.set_title(
            "step={} (n_train ~ {})".format(step, TRAIN_SIZE_BY_STEP.get(int(step), -1))
        )
        ax.set_xlabel("measurement_ratio rho")
        ax.set_xticks(rhos)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Relative L2 (mean across seeds)")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Teste A | Relative L2 - SIR-CS low-data triage (cross-well Vc)",
        y=1.02,
    )
    p2 = os.path.join(FIGURES_DIR, "lowdata_relL2_vs_rho.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p2)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    width = 0.25
    x = np.arange(len(rhos), dtype=np.float64)
    for k, step in enumerate(steps):
        gaps: List[float] = []
        for rho in rhos:
            row = winners[
                (winners["step"] == step) & (winners["measurement_ratio"] == rho)
            ]
            if row.empty:
                gaps.append(float("nan"))
            else:
                gaps.append(float(row["sir_cs_gap_pct"].iloc[0]))
        ax.bar(
            x + (k - 1) * width,
            gaps,
            width,
            label="step={} (n_tr~{})".format(step, TRAIN_SIZE_BY_STEP.get(int(step), -1)),
        )
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.axhline(5.0, color="g", linestyle="--", linewidth=1.0, label="5% threshold")
    ax.axhline(10.0, color="orange", linestyle="--", linewidth=1.0, label="10% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(r) for r in rhos])
    ax.set_xlabel("measurement_ratio rho")
    ax.set_ylabel("SIR-CS RMSE gap to cell winner (%)")
    ax.set_title("Teste A | SIR-CS gap to best baseline per cell (lower is better)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    p3 = os.path.join(FIGURES_DIR, "lowdata_lfista_gap_to_winner.png")
    fig.tight_layout()
    fig.savefig(p3, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p3)

    print("\nDone.")


if __name__ == "__main__":
    main()
