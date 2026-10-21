#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basis adherence study for cross-well Vc prediction.

Trains a sklearn MultiOutputMLP background on u from training wells (no GR by
default), then measures the effective sparsity of two residuals on the held-out
test well, across several orthonormal bases:

  (A) y_i - mean(y_train)   : residual against global training mean.
  (B) y_i - bg_u(u_i)       : residual against the fitted background MLP.

Per basis and residual, computes k95/k99 (top-k energy coverage) and Gini of
|alpha|, plus cumulative-energy curves. Saves CSV + two PNG figures.

Example (from repo root):
  python scripts/cross_well_vc_basis_adherence.py \
    --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
    --test-path  data/F03-4_6logs_30dB.txt \
    --channels sonic,rhob,ai,vp --target vc

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

import multi_well_vc as mwv
from bases_extra import orthonormality_error
from sir_cs_benchmark_direct_ub import MultiOutputMLP
from sir_cs_pipeline_optimized import get_basis


DEFAULT_BASES: Tuple[str, ...] = ("dct", "haar", "db4", "sym4", "fd1")


def gini(arr: np.ndarray) -> float:
    x = np.abs(np.asarray(arr, dtype=np.float64).ravel())
    if x.size == 0:
        return 0.0
    x = np.sort(x)
    n = x.size
    denom = float(n) * float(x.sum() + 1e-18)
    ranks = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(ranks * x)) / denom - (n + 1.0) / float(n))


def effective_k(alpha: np.ndarray, frac: float) -> np.ndarray:
    a2 = np.asarray(alpha, dtype=np.float64) ** 2
    s = np.sort(a2, axis=1)[:, ::-1]
    total = s.sum(axis=1, keepdims=True) + 1e-18
    c = np.cumsum(s, axis=1) / total
    idx = np.argmax(c >= float(frac), axis=1)
    return (idx + 1).astype(np.int64)


def cumulative_energy_profile(alpha: np.ndarray) -> np.ndarray:
    a2 = np.asarray(alpha, dtype=np.float64) ** 2
    s = np.sort(a2, axis=1)[:, ::-1]
    total = s.sum(axis=1, keepdims=True) + 1e-18
    c = np.cumsum(s, axis=1) / total
    return c.mean(axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-well Vc basis adherence study.")
    p.add_argument(
        "--train-path",
        type=str,
        default="data/F02-1,F03-2,F06-1_6logs_30dB.txt",
    )
    p.add_argument("--test-path", type=str, default="data/F03-4_6logs_30dB.txt")
    p.add_argument("--target", type=str, default="vc")
    p.add_argument("--channels", type=str, default="sonic,rhob,ai,vp")
    p.add_argument("--window-len", type=int, default=64)
    p.add_argument("--step", type=int, default=4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--bases",
        type=str,
        default=",".join(DEFAULT_BASES),
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default="outputs/cross_well_vc/basis_adherence",
    )
    p.add_argument("--run-id", type=str, default="")
    p.add_argument(
        "--bg-type",
        type=str,
        default="mlp2",
        choices=("mlp2", "shallow", "linear"),
        help=(
            "Background capacity used to build residual B. "
            "mlp2: 2 hidden x 128 (sklearn MLP). "
            "shallow: 1 hidden x 32 (sklearn MLP). "
            "linear: Ridge regression on standardized u, y."
        ),
    )
    return p.parse_args()


def _fit_background(bg_type: str, x_train: np.ndarray, y_train: np.ndarray):
    """Return a fitted object with .predict(X) -> Y, consistent with bg_type."""
    bgt = str(bg_type).strip().lower()
    if bgt == "linear":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        xs = StandardScaler()
        ys = StandardScaler()
        xt = xs.fit_transform(x_train)
        yt = ys.fit_transform(y_train)
        base = Ridge(alpha=1.0, random_state=7).fit(xt, yt)

        class _RidgeWrap:
            def __init__(self, xs, ys, m):
                self.xs = xs
                self.ys = ys
                self.m = m

            def predict(self, x):
                z = self.xs.transform(x)
                yh = self.m.predict(z)
                return self.ys.inverse_transform(yh)

        return _RidgeWrap(xs, ys, base)
    if bgt == "shallow":
        m = MultiOutputMLP(
            hidden_layer_sizes=(32,),
            max_iter=500,
            random_state=7,
            early_stopping=True,
            learning_rate_init=1e-3,
        )
        m.fit(x_train, y_train)
        return m
    m = MultiOutputMLP(
        hidden_layer_sizes=(128, 128),
        max_iter=500,
        random_state=7,
        early_stopping=True,
        learning_rate_init=1e-3,
    )
    m.fit(x_train, y_train)
    return m


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

    train_path = os.path.abspath(str(args.train_path).strip())
    test_path = os.path.abspath(str(args.test_path).strip())
    channels = tuple(c.strip().lower() for c in str(args.channels).split(",") if c.strip())
    target_name = str(args.target).strip().lower()
    bases = tuple(b.strip().lower() for b in str(args.bases).split(",") if b.strip())

    print("Run root:", run_root)
    print("train_path:", train_path)
    print("test_path :", test_path)
    print("channels  :", channels)
    print("target    :", target_name)
    print("bases     :", bases)
    print("bg_type   :", str(args.bg_type))

    data = mwv.build_cross_well_data_dict(
        train_path=train_path,
        test_path=test_path,
        target_name=target_name,
        channels=channels,
        window_len=int(args.window_len),
        step=int(args.step),
        val_frac=float(args.val_frac),
        residual_basis="dct",
    )
    meta = data["meta"]
    L = int(meta["n_output"])
    print(
        "n_train={} n_val={} n_test={} p_input={} L={}".format(
            meta["n_train"], meta["n_val"], meta["n_test"], meta["p_input"], L
        )
    )

    y_train = data["Y_train"]
    y_val = data["Y_val"]
    y_test = data["Y_test"]
    x_train = data["X_train"]
    x_val = data["X_val"]
    x_test = data["X_test"]

    mean_y_train = y_train.mean(axis=0, keepdims=True)
    res_A_test = y_test - mean_y_train
    print(
        "[residual A = y - mean(y_train)] std_train={:.5f} std_test={:.5f}".format(
            float((y_train - mean_y_train).std()), float(res_A_test.std())
        )
    )

    bg_type = str(args.bg_type).strip().lower()
    print("Fitting background (bg_type={}) on u ...".format(bg_type))
    bg = _fit_background(bg_type, x_train, y_train)
    yhat_test = bg.predict(x_test)
    res_B_test = y_test - yhat_test
    print(
        "[residual B = y - bg_u(u)] bg_type={} std_test={:.5f} | bg RMSE test={:.5f}".format(
            bg_type,
            float(res_B_test.std()),
            float(np.sqrt((res_B_test ** 2).mean())),
        )
    )

    summary_rows: List[Dict[str, object]] = []
    cum_curves: Dict[str, Dict[str, np.ndarray]] = {}
    k95_dist: Dict[str, Dict[str, np.ndarray]] = {}

    for basis in bases:
        try:
            Psi = get_basis(L, basis)
        except ValueError as ex:
            print("Skipping basis '{}': {}".format(basis, ex))
            continue
        orth_err = orthonormality_error(Psi)
        print("\n=== basis={} orth_err={:.2e} ===".format(basis, orth_err))
        cum_curves[basis] = {}
        k95_dist[basis] = {}
        for tag, res_all in [("A", res_A_test), ("B", res_B_test)]:
            alpha = res_all @ Psi
            k95 = effective_k(alpha, 0.95)
            k99 = effective_k(alpha, 0.99)
            g = np.asarray(
                [gini(alpha[i]) for i in range(alpha.shape[0])], dtype=np.float64
            )
            cumc = cumulative_energy_profile(alpha)
            cum_curves[basis][tag] = cumc
            k95_dist[basis][tag] = k95
            print(
                "  residual {}: k95 mean={:.2f} median={} p90={}  "
                "k99 mean={:.2f} median={} p90={}  gini_mean={:.3f}".format(
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
            print("[residual {}]".format(tag))
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
        sys.stdout = sys.__stdout__
        lf.close()
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
        "Cross-well {} basis adherence (test={}, L={}, u={}, bg={}) ".format(
            target_name.upper(),
            ",".join(meta["test_wells"]),
            L,
            ",".join(channels),
            bg_type,
        ),
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
    fig.suptitle(
        "Cross-well {} effective sparsity k95 per window (lower = sparser, bg={})".format(
            target_name.upper(), bg_type
        ),
        y=1.02,
    )
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
