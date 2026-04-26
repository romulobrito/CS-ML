#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSGM M2 production grid for the ridge latent-prior variant.

This script expands the successful smoke test to:
  step in {8, 16, 32}
  rho  in {0.05, 0.10, 0.20}
  seed in {7, 23, 41}

For each (step, seed), the AE and ridge latent prior are trained once. For
each rho, lambda is selected on validation data, then evaluated on the
held-out test well. Results are compared against the previously generated
low-data benchmark summaries.

Run from repo root:
  python scripts/csgm_m2_grid.py

ASCII-only.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import multi_well_vc as mwv
from csgm_smoke_m2_conditional import (
    BASIS,
    CHANNELS,
    CSGM_ITERS,
    CSGM_LR,
    CSGM_RESTARTS,
    LAMBDA_GRID,
    NOISE_STD,
    TEST_PATH,
    TRAIN_PATH,
    VAL_FRAC,
    WINDOW_LEN,
    RidgePrior,
    build_subsample_m,
    csgm_recover_with_prior,
    encode_y,
    make_b,
    metrics,
    rmse,
    train_ae,
)


OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "cross_well_vc", "csgm", "m2_grid")
TABLES_DIR = os.path.join(OUT_DIR, "tables")
LOG_PATH = os.path.join(OUT_DIR, "grid.log")

STEPS: Tuple[int, ...] = (8, 16, 32)
RHOS: Tuple[float, ...] = (0.05, 0.10, 0.20)
SEEDS: Tuple[int, ...] = (7, 23, 41)


def load_benchmark_rows(step: int) -> pd.DataFrame:
    """Load benchmark summary for one low-data step."""
    path = os.path.join(
        _REPO_ROOT,
        "outputs",
        "cross_well_vc",
        "sir_cs_stress_lowdata",
        "runs",
        "prod_step{:02d}".format(int(step)),
        "tables",
        "summary.csv",
    )
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.copy()
    df["step"] = int(step)
    return df


def summarize_csgm(detailed: pd.DataFrame) -> pd.DataFrame:
    """Aggregate CSGM rows across seeds by step and rho."""
    agg = (
        detailed.groupby(["step", "measurement_ratio", "method"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std_across_seeds=("rmse", "std"),
            mae_mean=("mae", "mean"),
            relative_l2_mean=("rel_l2", "mean"),
            relative_l2_std_across_seeds=("rel_l2", "std"),
            lambda_mean=("lambda", "mean"),
            n_seeds=("seed", "nunique"),
            n_test_samples_per_run=("n_test", "first"),
        )
    )
    agg["rmse_std_across_seeds"] = agg["rmse_std_across_seeds"].fillna(0.0)
    agg["relative_l2_std_across_seeds"] = agg["relative_l2_std_across_seeds"].fillna(0.0)
    return agg


def build_decision_report(csgm_summary: pd.DataFrame, bench_long: pd.DataFrame) -> str:
    """Create a compact go/no-go report against ae_regression_ub."""
    rows: List[str] = []
    rows.append("CSGM M2 grid decision report")
    rows.append("=" * 60)
    rows.append("")
    rows.append("Method: ridge_prior_csgm")
    rows.append("Grid: step in {} | rho in {} | seeds in {}".format(
        list(STEPS), list(RHOS), list(SEEDS)
    ))
    rows.append("")

    wins = 0
    cells = 0
    gap_values: List[float] = []
    for step in STEPS:
        for rho in RHOS:
            c = csgm_summary[
                (csgm_summary["step"] == step)
                & (csgm_summary["measurement_ratio"] == rho)
            ]
            a = bench_long[
                (bench_long["step"] == step)
                & (bench_long["measurement_ratio"] == rho)
                & (bench_long["method"] == "ae_regression_ub")
            ]
            m = bench_long[
                (bench_long["step"] == step)
                & (bench_long["measurement_ratio"] == rho)
                & (bench_long["method"] == "ml_only")
            ]
            s = bench_long[
                (bench_long["step"] == step)
                & (bench_long["measurement_ratio"] == rho)
                & (bench_long["method"] == "hybrid_lfista_joint")
            ]
            if c.empty or a.empty:
                continue
            cells += 1
            c_rmse = float(c["rmse_mean"].iloc[0])
            ae_rmse = float(a["rmse_mean"].iloc[0])
            ml_rmse = float(m["rmse_mean"].iloc[0]) if not m.empty else float("nan")
            sir_rmse = float(s["rmse_mean"].iloc[0]) if not s.empty else float("nan")
            gap = 100.0 * (c_rmse - ae_rmse) / max(ae_rmse, 1e-12)
            gap_values.append(gap)
            if c_rmse < ae_rmse:
                wins += 1
            rows.append(
                "step={:>2} rho={:.2f} | csgm={:.5f} ae={:.5f} "
                "ml={:.5f} sir={:.5f} gap_vs_ae={:+.2f}%".format(
                    int(step), float(rho), c_rmse, ae_rmse, ml_rmse, sir_rmse, gap
                )
            )

    avg_gap = float(np.mean(gap_values)) if gap_values else float("nan")
    min_gap = float(np.min(gap_values)) if gap_values else float("nan")
    rows.append("")
    rows.append("Cells evaluated: {}".format(cells))
    rows.append("CSGM wins vs ae_regression_ub: {} / {}".format(wins, cells))
    rows.append("Average gap vs ae_regression_ub: {:+.2f}%".format(avg_gap))
    rows.append("Best gap vs ae_regression_ub: {:+.2f}%".format(min_gap))
    if wins >= max(1, cells // 2):
        verdict = "GO: CSGM M2 wins in enough cells to justify full method integration."
    elif wins > 0:
        verdict = "WEAK GO: CSGM M2 has local wins but needs refinement."
    else:
        verdict = "NO-GO: smoke win did not generalize across the grid."
    rows.append("VERDICT: " + verdict)
    return "\n".join(rows) + "\n"


def main() -> None:
    """Run the CSGM M2 grid."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    device = "cpu"
    t0 = time.time()
    detailed_rows: List[Dict[str, object]] = []

    for step in STEPS:
        data = mwv.build_cross_well_data_dict(
            train_path=TRAIN_PATH,
            test_path=TEST_PATH,
            target_name="vc",
            channels=CHANNELS,
            window_len=WINDOW_LEN,
            step=int(step),
            val_frac=VAL_FRAC,
            residual_basis=BASIS,
        )
        x_train = np.asarray(data["X_train"], dtype=np.float64)
        x_val = np.asarray(data["X_val"], dtype=np.float64)
        x_test = np.asarray(data["X_test"], dtype=np.float64)
        y_train = np.asarray(data["Y_train"], dtype=np.float64)
        y_val = np.asarray(data["Y_val"], dtype=np.float64)
        y_test = np.asarray(data["Y_test"], dtype=np.float64)
        meta = data["meta"]
        n_train = int(meta["n_train"])
        n_val = int(meta["n_val"])
        n_test = int(meta["n_test"])
        n_output = int(meta["n_output"])
        print("=== step={} | n_train={} n_val={} n_test={} ===".format(
            step, n_train, n_val, n_test
        ))

        y_scaler = StandardScaler().fit(y_train)
        y_train_n = y_scaler.transform(y_train)
        y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
        y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)

        for seed in SEEDS:
            print("--- seed={} ---".format(seed))
            ae = train_ae(y_train_n, seed=int(seed), device=device)
            z_train = encode_y(ae, y_train_n, device=device)
            prior = RidgePrior().fit(x_train, z_train)
            z0_val = prior.predict(x_val)
            z0_test = prior.predict(x_test)

            with torch.no_grad():
                rec_train_n = ae(
                    torch.tensor(y_train_n, dtype=torch.float32, device=device)
                ).cpu().numpy()
            rec_train = rec_train_n * y_scale[None, :] + y_mean[None, :]
            rec_train_rmse = rmse(rec_train, y_train)

            for rho in RHOS:
                m_meas = max(2, int(round(float(rho) * float(n_output))))
                rng = np.random.default_rng(int(seed))
                mat = build_subsample_m(m_meas, n_output, rng)
                b_val = make_b(y_val, mat, NOISE_STD, rng)
                b_test = make_b(y_test, mat, NOISE_STD, rng)

                val_scores: List[Tuple[float, float]] = []
                for lam in LAMBDA_GRID:
                    y_val_hat = csgm_recover_with_prior(
                        ae=ae,
                        mat=mat,
                        b_orig=b_val,
                        z0_np=z0_val,
                        y_mean=y_mean,
                        y_scale=y_scale,
                        lam=float(lam),
                        n_iters=CSGM_ITERS,
                        lr=CSGM_LR,
                        n_restarts=CSGM_RESTARTS,
                        device=device,
                        seed=int(seed),
                    )
                    val_scores.append((float(lam), rmse(y_val_hat, y_val)))
                best_lam, best_val_rmse = min(val_scores, key=lambda item: item[1])
                y_test_hat = csgm_recover_with_prior(
                    ae=ae,
                    mat=mat,
                    b_orig=b_test,
                    z0_np=z0_test,
                    y_mean=y_mean,
                    y_scale=y_scale,
                    lam=float(best_lam),
                    n_iters=CSGM_ITERS,
                    lr=CSGM_LR,
                    n_restarts=CSGM_RESTARTS,
                    device=device,
                    seed=int(seed),
                )
                mt = metrics(y_test_hat, y_test)
                detailed_rows.append({
                    "step": int(step),
                    "seed": int(seed),
                    "measurement_ratio": float(rho),
                    "method": "ridge_prior_csgm",
                    "lambda": float(best_lam),
                    "val_rmse_selected": float(best_val_rmse),
                    "ae_recon_train_rmse": float(rec_train_rmse),
                    "n_train": n_train,
                    "n_val": n_val,
                    "n_test": n_test,
                    **mt,
                })
                print(
                    "  rho={:.2f} lam={:.4g} val_rmse={:.5f} test_rmse={:.5f}".format(
                        float(rho), float(best_lam), float(best_val_rmse), mt["rmse"]
                    )
                )

    detailed = pd.DataFrame(detailed_rows)
    detailed_path = os.path.join(TABLES_DIR, "csgm_m2_detailed.csv")
    detailed.to_csv(detailed_path, index=False)

    summary = summarize_csgm(detailed)
    summary_path = os.path.join(TABLES_DIR, "csgm_m2_summary.csv")
    summary.to_csv(summary_path, index=False)

    bench_frames = [load_benchmark_rows(step) for step in STEPS]
    bench_frames = [df for df in bench_frames if not df.empty]
    bench_long = pd.concat(bench_frames, ignore_index=True) if bench_frames else pd.DataFrame()
    if not bench_long.empty:
        bench_path = os.path.join(TABLES_DIR, "benchmark_lowdata_summary.csv")
        bench_long.to_csv(bench_path, index=False)

    decision = build_decision_report(summary, bench_long)
    decision += "\nElapsed: {:.1f}s\n".format(time.time() - t0)
    decision_path = os.path.join(TABLES_DIR, "csgm_m2_decision.txt")
    with open(decision_path, "w", encoding="ascii") as f:
        f.write(decision)
    with open(LOG_PATH, "w", encoding="ascii") as f:
        f.write(decision)

    print()
    print(decision)
    print("Saved:", detailed_path)
    print("Saved:", summary_path)
    print("Saved:", decision_path)


if __name__ == "__main__":
    main()
