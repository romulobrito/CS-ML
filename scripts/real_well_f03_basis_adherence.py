#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis adherence study for F03-4 porosity sliding windows.

For each candidate orthonormal basis Psi (dct, haar, db4, sym4, fd1), measure
the effective sparsity of two residuals obtained from the same sliding-window
dataset used by the real-well direct [u,b] benchmark:

  (A) y_i - mean(y_train)        : residual against global training mean.
  (B) y_i - bg_u(u_i)             : residual against sklearn MLP trained on u.

For each residual and basis, compute:
  - cumulative normalized energy curve (mean over windows);
  - effective k95 and k99 per window (samples needed to reach 95/99% energy);
  - Gini coefficient of |alpha| per window (concentration measure).

Outputs (under a timestamped run folder):
  tables/basis_adherence_summary.csv
  tables/basis_adherence_percentiles.csv
  figures/basis_adherence_cum_energy.png
  figures/basis_adherence_k95_box.png
  logs/run_console.log

The u-channel set used in residual (B) is configurable via --u-channels (default 'gr').
This matches the ablation scenario where u is weakened.

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import pandas as pd

import real_well_f03 as rwf
from bases_extra import orthonormality_error
from sir_cs_benchmark_direct_ub import MultiOutputMLP
from sir_cs_pipeline_optimized import get_basis


DEFAULT_BASES: Tuple[str, ...] = ("dct", "haar", "db4", "sym4", "fd1")


def gini(arr: np.ndarray) -> float:
    """Gini coefficient of |arr| along last axis (returns scalar). 0=uniform, 1=sparse."""
    x = np.abs(np.asarray(arr, dtype=np.float64).ravel())
    if x.size == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    denom = float(n) * float(x.sum() + 1e-18)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x)) / denom - (n + 1.0) / float(n))


def effective_k(alpha: np.ndarray, frac: float) -> np.ndarray:
    """Per-window smallest k such that top-k |alpha|^2 covers frac of energy."""
    a2 = np.asarray(alpha, dtype=np.float64) ** 2
    s = np.sort(a2, axis=1)[:, ::-1]
    total = s.sum(axis=1, keepdims=True) + 1e-18
    c = np.cumsum(s, axis=1) / total
    idx = np.argmax(c >= float(frac), axis=1)
    return (idx + 1).astype(np.int64)


def cumulative_energy_profile(alpha: np.ndarray) -> np.ndarray:
    """Mean cumulative energy curve of sorted |alpha|^2, length L."""
    a2 = np.asarray(alpha, dtype=np.float64) ** 2
    s = np.sort(a2, axis=1)[:, ::-1]
    total = s.sum(axis=1, keepdims=True) + 1e-18
    c = np.cumsum(s, axis=1) / total
    return c.mean(axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Basis adherence study on F03-4.")
    p.add_argument("--data-path", type=str, default="data/F03-4_AC+GR+Porosity.txt")
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--train-frac", type=float, default=0.6)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--u-channels",
        type=str,
        default="gr",
        help="Channels for residual (B). Valid: ac, gr. Default 'gr' (ablation scenario).",
    )
    p.add_argument(
        "--bases",
        type=str,
        default=",".join(DEFAULT_BASES),
        help="Comma-separated basis names. Default: dct,haar,db4,sym4,fd1.",
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default="outputs/real_well_f03/basis_adherence",
    )
    p.add_argument("--run-id", type=str, default="")
    return p.parse_args()


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self) -> None:
        for st in self.streams:
            st.flush()


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id).strip() or time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(str(args.base_dir).strip())
    run_root = os.path.join(base_dir, "runs", run_id)
    tables_dir = os.path.join(run_root, "tables")
    figures_dir = os.path.join(run_root, "figures")
    logs_dir = os.path.join(run_root, "logs")
    for d in (tables_dir, figures_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    log_path = os.path.join(logs_dir, "run_console.log")
    lf = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, lf)

    print("Run root:", run_root)
    data_path = os.path.abspath(str(args.data_path).strip())
    print("data_path:", data_path)

    tab = rwf.load_f03_table(data_path)
    u_channels = rwf.normalize_channels(
        tuple(c.strip() for c in str(args.u_channels).split(",") if c.strip())
    )
    print("u_channels:", u_channels)

    L = int(args.window_len)
    st = int(args.step)
    x_all, y_all, _, _ = rwf.build_sliding_windows(tab, L, st, channels=u_channels)
    n_win = int(x_all.shape[0])
    sl_tr, sl_va, sl_te, n_tr, n_va, n_te = rwf.contiguous_split(
        n_win, float(args.train_frac), float(args.val_frac)
    )
    print(f"n_windows={n_win} L={L} n_train={n_tr} n_val={n_va} n_test={n_te}")

    y_train = y_all[sl_tr]
    y_val = y_all[sl_va]
    y_test = y_all[sl_te]
    x_train = x_all[sl_tr]
    x_val = x_all[sl_va]
    x_test = x_all[sl_te]

    mean_y_train = y_train.mean(axis=0, keepdims=True)
    res_A_train = y_train - mean_y_train
    res_A_val = y_val - mean_y_train
    res_A_test = y_test - mean_y_train
    print(
        "[residual A = y - mean(y_train)] std train={:.5f} test={:.5f}".format(
            float(res_A_train.std()), float(res_A_test.std())
        )
    )

    print("Fitting background MLP on u ...")
    bg = MultiOutputMLP(
        hidden_layer_sizes=(128, 128),
        max_iter=500,
        random_state=7,
        early_stopping=True,
        learning_rate_init=1e-3,
    )
    bg.fit(x_train, y_train)
    yhat_train = bg.predict(x_train)
    yhat_val = bg.predict(x_val)
    yhat_test = bg.predict(x_test)
    res_B_train = y_train - yhat_train
    res_B_val = y_val - yhat_val
    res_B_test = y_test - yhat_test
    print(
        "[residual B = y - bg_u(u)] std train={:.5f} test={:.5f} | bg RMSE test={:.5f}".format(
            float(res_B_train.std()),
            float(res_B_test.std()),
            float(np.sqrt((res_B_test ** 2).mean())),
        )
    )

    bases = tuple(b.strip().lower() for b in str(args.bases).split(",") if b.strip())
    print("bases to evaluate:", bases)

    summary_rows: List[Dict[str, object]] = []
    cum_curves: Dict[str, Dict[str, np.ndarray]] = {}
    k95_dist: Dict[str, Dict[str, np.ndarray]] = {}

    for basis in bases:
        try:
            Psi = get_basis(L, basis)
        except ValueError as ex:
            print(f"Skipping basis '{basis}': {ex}")
            continue
        orth_err = orthonormality_error(Psi)
        print(f"\n=== basis={basis} orth_err={orth_err:.2e} ===")
        cum_curves[basis] = {}
        k95_dist[basis] = {}
        for tag, res_all in [("A", res_A_test), ("B", res_B_test)]:
            alpha = res_all @ Psi
            k95 = effective_k(alpha, 0.95)
            k99 = effective_k(alpha, 0.99)
            g = np.asarray([gini(alpha[i]) for i in range(alpha.shape[0])], dtype=np.float64)
            cumc = cumulative_energy_profile(alpha)
            cum_curves[basis][tag] = cumc
            k95_dist[basis][tag] = k95
            print(
                "  residual {}: k95 mean={:.2f} median={} p90={}"
                "  k99 mean={:.2f} median={} p90={} gini_mean={:.3f}".format(
                    tag,
                    float(k95.mean()),
                    int(np.median(k95)),
                    int(np.percentile(k95, 90)),
                    float(k99.mean()),
                    int(np.median(k99)),
                    int(np.percentile(k99, 90)),
                    float(g.mean()),
                )
            )
            summary_rows.append(
                {
                    "basis": basis,
                    "residual": tag,
                    "orth_err": float(orth_err),
                    "k95_mean": float(k95.mean()),
                    "k95_median": float(np.median(k95)),
                    "k95_p90": float(np.percentile(k95, 90)),
                    "k99_mean": float(k99.mean()),
                    "k99_median": float(np.median(k99)),
                    "k99_p90": float(np.percentile(k99, 90)),
                    "gini_mean": float(g.mean()),
                    "alpha_energy_top1_mean": float(cumc[0]),
                    "alpha_energy_top4_mean": float(cumc[3]),
                    "alpha_energy_top8_mean": float(cumc[7]),
                    "alpha_energy_top16_mean": float(cumc[15]),
                }
            )

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(tables_dir, "basis_adherence_summary.csv")
    df.to_csv(csv_path, index=False)
    print("\nSaved:", csv_path)

    rank_cols = ["basis", "residual", "k95_mean", "k99_mean", "gini_mean"]
    print("\n=== ranking by k95_mean (lower = more sparse) ===")
    for tag in ("A", "B"):
        sub = df[df["residual"] == tag].sort_values("k95_mean").copy()
        if not sub.empty:
            print(f"[residual {tag}]")
            print(sub[rank_cols].to_string(index=False))

    perc_rows: List[Dict[str, object]] = []
    for basis in bases:
        if basis not in k95_dist:
            continue
        for tag, arr in k95_dist[basis].items():
            qs = [10, 25, 50, 75, 90]
            row: Dict[str, object] = {"basis": basis, "residual": tag}
            for q in qs:
                row["k95_p{}".format(q)] = float(np.percentile(arr, q))
            perc_rows.append(row)
    pd.DataFrame(perc_rows).to_csv(
        os.path.join(tables_dir, "basis_adherence_percentiles.csv"), index=False
    )

    try:
        import matplotlib.pyplot as plt
    except Exception as ex:
        print("matplotlib unavailable, skipping figures:", ex)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    x = np.arange(1, L + 1)
    for tag, ax in zip(("A", "B"), axes):
        for basis in bases:
            if basis not in cum_curves or tag not in cum_curves[basis]:
                continue
            c = cum_curves[basis][tag]
            ax.plot(x, c, linewidth=1.4, label=basis)
        ax.axhline(0.95, color="#d62728", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.99, color="#9467bd", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("top-k coefficients")
        ax.set_xscale("log")
        ax.set_xlim(1, L)
        ax.grid(True, alpha=0.3)
        res_name = {"A": "A: y - mean(y_train)", "B": "B: y - bg_u(u)"}[tag]
        ax.set_title("Residual " + res_name)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("mean cumulative energy")
    fig.suptitle(
        "F03-4 basis adherence (test windows, L={}, u_channels={})".format(L, u_channels),
        y=1.02,
    )
    p1 = os.path.join(figures_dir, "basis_adherence_cum_energy.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for tag, ax in zip(("A", "B"), axes):
        data_box = []
        labels = []
        for basis in bases:
            if basis not in k95_dist:
                continue
            data_box.append(k95_dist[basis][tag])
            labels.append(basis)
        ax.boxplot(data_box, labels=labels, showfliers=False)
        ax.set_ylabel("k95 per window")
        ax.grid(True, alpha=0.3)
        res_name = {"A": "A: y - mean(y_train)", "B": "B: y - bg_u(u)"}[tag]
        ax.set_title("k95 distribution | residual " + res_name)
    fig.suptitle("F03-4 effective sparsity k95 per window (lower = more sparse)", y=1.02)
    p2 = os.path.join(figures_dir, "basis_adherence_k95_box.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", p2)

    sys.stdout = sys.__stdout__
    lf.close()
    print("Done. Artifacts:", run_root)


if __name__ == "__main__":
    main()
