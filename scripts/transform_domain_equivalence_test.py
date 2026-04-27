#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform-domain learning equivalence test.

Goal: verify empirically the theoretical claim that, with orthonormal Psi
and L2 loss, learning u -> y and learning u -> alpha (then reconstructing
y_hat = alpha_hat @ Psi.T) yield essentially the same test error.

For each (step, seed) pair we train four models on the cross-well Vc data:
  - Ridge u -> y                                (analytic baseline)
  - Ridge u -> alpha (recovered via Psi.T)      (analytic baseline, alpha space)
  - MLP   u -> y                                (sklearn MultiOutputMLP)
  - MLP   u -> alpha (recovered via Psi.T)      (sklearn MultiOutputMLP)

Predictions:
  - For "y" target:     y_hat = m.predict(X_test)
  - For "alpha" target: y_hat = m.predict(X_test) @ Psi.T

Metrics: RMSE, MAE, Relative L2 (per-window aggregated to mean).

Output:
  outputs/cross_well_vc/transform_domain_equivalence/
    tables/equivalence_long.csv      (all rows)
    tables/equivalence_pivot.csv     (pivot model x target_space)
    tables/equivalence_decision.txt  (verdict)

Run from repo root:
  python scripts/transform_domain_equivalence_test.py

ASCII-only.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multi_well_vc as mwv
from sir_cs_pipeline_optimized import MultiOutputMLP


TRAIN_PATH = os.path.join(_REPO_ROOT, "data", "F02-1,F03-2,F06-1_6logs_30dB.txt")
TEST_PATH = os.path.join(_REPO_ROOT, "data", "F03-4_6logs_30dB.txt")
OUT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "cross_well_vc", "transform_domain_equivalence"
)
TABLES_DIR = os.path.join(OUT_DIR, "tables")

CHANNELS: Tuple[str, ...] = ("sonic", "rhob", "gr", "ai", "vp")
WINDOW_LEN = 64
VAL_FRAC = 0.1
BASIS = "dct"

STEPS: Tuple[int, ...] = (8, 16, 32)
SEEDS: Tuple[int, ...] = (7, 23, 41)

MLP_HIDDEN = (128, 128)
MLP_MAX_ITER = 500
MLP_LR = 1e-3
RIDGE_ALPHA = 1.0


def _rmse(yh: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yh - y) ** 2)))


def _mae(yh: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(yh - y)))


def _rel_l2(yh: np.ndarray, y: np.ndarray) -> float:
    num = np.linalg.norm(yh - y, axis=1)
    den = np.linalg.norm(y, axis=1)
    den = np.where(den < 1e-12, 1e-12, den)
    return float(np.mean(num / den))


def _ridge_y(x_tr: np.ndarray, y_tr: np.ndarray) -> Ridge:
    m = Ridge(alpha=RIDGE_ALPHA, random_state=0).fit(x_tr, y_tr)
    return m


def _train_mlp(
    x_tr: np.ndarray, y_tr: np.ndarray, seed: int
) -> MultiOutputMLP:
    m = MultiOutputMLP(
        hidden_layer_sizes=MLP_HIDDEN,
        max_iter=MLP_MAX_ITER,
        learning_rate_init=MLP_LR,
        early_stopping=True,
        random_state=int(seed),
    )
    m.fit(x_tr, y_tr)
    return m


def _eval(
    yh: np.ndarray, y_te: np.ndarray
) -> Dict[str, float]:
    return {
        "rmse": _rmse(yh, y_te),
        "mae": _mae(yh, y_te),
        "rel_l2": _rel_l2(yh, y_te),
    }


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    os.makedirs(TABLES_DIR, exist_ok=True)

    if not os.path.isfile(TRAIN_PATH):
        print("Missing train file: " + TRAIN_PATH, file=sys.stderr)
        sys.exit(2)
    if not os.path.isfile(TEST_PATH):
        print("Missing test file: " + TEST_PATH, file=sys.stderr)
        sys.exit(2)

    rows: List[Dict[str, object]] = []
    t_global = time.time()

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
        x_tr = np.asarray(data["X_train"], dtype=np.float64)
        x_te = np.asarray(data["X_test"], dtype=np.float64)
        y_tr = np.asarray(data["Y_train"], dtype=np.float64)
        y_te = np.asarray(data["Y_test"], dtype=np.float64)
        a_tr = np.asarray(data["Alpha_train"], dtype=np.float64)
        psi = np.asarray(data["Psi"], dtype=np.float64)
        psi_t = psi.T
        meta = data["meta"]
        n_tr = int(meta["n_train"])
        n_te = int(meta["n_test"])
        l = int(meta["n_output"])

        ortho_err = float(np.max(np.abs(psi @ psi_t - np.eye(l))))
        print(
            "step={:>2} | n_train={:>4} n_test={:>4} | L={} | "
            "Psi*Psi.T - I, max abs err = {:.2e}".format(
                step, n_tr, n_te, l, ortho_err
            )
        )

        for seed in SEEDS:
            t0 = time.time()
            r_y = _ridge_y(x_tr, y_tr)
            yh = r_y.predict(x_te)
            m_ridge_y = _eval(yh, y_te)

            r_a = _ridge_y(x_tr, a_tr)
            ah = r_a.predict(x_te)
            yh_a = ah @ psi_t
            m_ridge_a = _eval(yh_a, y_te)

            mlp_y = _train_mlp(x_tr, y_tr, seed)
            yh = mlp_y.predict(x_te)
            m_mlp_y = _eval(yh, y_te)

            mlp_a = _train_mlp(x_tr, a_tr, seed)
            ah = mlp_a.predict(x_te)
            yh_a = ah @ psi_t
            m_mlp_a = _eval(yh_a, y_te)

            dt = time.time() - t0

            for tag, mt in (
                ("ridge_y", m_ridge_y),
                ("ridge_alpha", m_ridge_a),
                ("mlp_y", m_mlp_y),
                ("mlp_alpha", m_mlp_a),
            ):
                rows.append({
                    "step": int(step),
                    "n_train": n_tr,
                    "n_test": n_te,
                    "seed": int(seed),
                    "method": tag,
                    "target_space": "y" if tag.endswith("_y") else "alpha",
                    "model_family": "ridge" if tag.startswith("ridge") else "mlp",
                    "rmse": mt["rmse"],
                    "mae": mt["mae"],
                    "rel_l2": mt["rel_l2"],
                })

            print(
                "  seed={:>2} | "
                "ridge_y rmse={:.5f}  ridge_alpha rmse={:.5f}  "
                "mlp_y rmse={:.5f}  mlp_alpha rmse={:.5f}  | t={:.1f}s".format(
                    seed,
                    m_ridge_y["rmse"], m_ridge_a["rmse"],
                    m_mlp_y["rmse"], m_mlp_a["rmse"], dt
                )
            )

    long_df = pd.DataFrame(rows)
    long_path = os.path.join(TABLES_DIR, "equivalence_long.csv")
    long_df.to_csv(long_path, index=False)
    print("\nSaved:", long_path)

    agg = (
        long_df.groupby(["step", "method", "model_family", "target_space"])
        .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
             mae_mean=("mae", "mean"), rel_l2_mean=("rel_l2", "mean"))
        .reset_index()
    )
    agg_path = os.path.join(TABLES_DIR, "equivalence_aggregated.csv")
    agg.to_csv(agg_path, index=False)
    print("Saved:", agg_path)

    pv = long_df.pivot_table(
        index=["step", "model_family"],
        columns="target_space",
        values="rmse",
        aggfunc=["mean", "std"],
    ).round(6)
    pv_path = os.path.join(TABLES_DIR, "equivalence_pivot.csv")
    pv.to_csv(pv_path)
    print("Saved:", pv_path)

    decision_lines: List[str] = []
    decision_lines.append("Transform-domain equivalence test")
    decision_lines.append("=" * 60)
    decision_lines.append("")
    decision_lines.append("Hypothesis: with orthonormal Psi and L2 loss, ")
    decision_lines.append("  learning u -> y and learning u -> alpha (with ")
    decision_lines.append("  reconstruction y_hat = alpha_hat @ Psi.T) yield ")
    decision_lines.append("  essentially the same test RMSE.")
    decision_lines.append("")
    decision_lines.append("Threshold for 'equivalent': |delta_rmse| / rmse_y < 5%.")
    decision_lines.append("")

    diffs: List[Dict[str, object]] = []
    for (step, fam), grp in long_df.groupby(["step", "model_family"]):
        ry = grp[grp["target_space"] == "y"]["rmse"].mean()
        ra = grp[grp["target_space"] == "alpha"]["rmse"].mean()
        if ry <= 0:
            rel = float("nan")
        else:
            rel = 100.0 * (ra - ry) / ry
        diffs.append({
            "step": int(step),
            "family": fam,
            "rmse_y": float(ry),
            "rmse_alpha": float(ra),
            "delta_pct": float(rel),
        })
        decision_lines.append(
            "step={:>2} family={:<6} | "
            "y_target rmse={:.5f}  alpha_target rmse={:.5f}  "
            "delta={:+.2f}%".format(int(step), fam, ry, ra, rel)
        )

    diffs_df = pd.DataFrame(diffs)
    n_within_5 = int((diffs_df["delta_pct"].abs() <= 5.0).sum())
    n_total = int(diffs_df.shape[0])
    decision_lines.append("")
    decision_lines.append("Cells with |delta| <= 5%: {} / {}".format(n_within_5, n_total))

    if n_within_5 == n_total:
        verdict = (
            "CONFIRMED: equivalence holds in all cells. Learning in "
            "alpha space gives no advantage over learning in y space "
            "(modulo init/SGD noise). Formulations A/B/C with orthonormal "
            "Psi and L2 cannot beat the corresponding y-space baselines."
        )
    elif n_within_5 >= n_total - 1:
        verdict = (
            "MOSTLY CONFIRMED: equivalence holds in {}/{} cells. "
            "The single outlier is likely a SGD trajectory artifact, not "
            "a genuine advantage. Conclusion stands.".format(n_within_5, n_total)
        )
    else:
        verdict = (
            "NOT CONFIRMED: alpha space differs from y space in {} cells. "
            "Worth investigating further (could indicate optimization "
            "advantage, regularization interaction, or implementation "
            "issue).".format(n_total - n_within_5)
        )
    decision_lines.append("")
    decision_lines.append("VERDICT: " + verdict)
    decision_lines.append("")
    decision_lines.append("Total run time: {:.1f}s".format(time.time() - t_global))

    dec_text = "\n".join(decision_lines) + "\n"
    dec_path = os.path.join(TABLES_DIR, "equivalence_decision.txt")
    with open(dec_path, "w", encoding="ascii") as f:
        f.write(dec_text)
    print("\n" + dec_text)
    print("Saved:", dec_path)


if __name__ == "__main__":
    main()
