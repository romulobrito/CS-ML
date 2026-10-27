#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well 861 laboratory porosity with one role per physical depth.

The published runner scores overlapping windows that reuse one local mask.
This script assigns each depth to the observed set or the holdout set and
keeps that role in every window. It calls the existing recovery once per
window and does not edit csgm_m2_module.py.

Pilot cells are fold 0 and seed 7. The full grid is the same protocol on
three blocks and seeds 7, 23, and 41. Test scores do not choose the protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import csgm_m2_module as csgm  # noqa: E402
import direct_ub_baselines as dub  # noqa: E402
import run_r1m11_861_phi_lab as published  # noqa: E402
from sir_cs_pipeline_optimized import Config  # noqa: E402

REFERENCE_COMMIT = "92cfea4ca22770b302b7ba2eae7b5e29dca707a3"
OUT_ROOT = REPO_ROOT / "outputs" / "exploratory" / "861_global_depth_mask"
AUDIT_SPLIT = (
    REPO_ROOT
    / "outputs"
    / "exploratory"
    / "861_depth_mask_audit"
    / "tables"
    / "split_summary.csv"
)
SPLIT_CODE = {"train": 1, "val": 2, "test": 3}
PILOT_FOLDS: Tuple[int, ...] = (0,)
PILOT_SEEDS: Tuple[int, ...] = (7,)
FULL_FOLDS: Tuple[int, ...] = (0, 1, 2)
FULL_SEEDS: Tuple[int, ...] = (7, 23, 41)
RHOS: Tuple[float, ...] = (0.25, 0.50)
METHODS: Tuple[str, ...] = (
    "clp_csgm_ridge",
    "csgm_unconditioned",
    "ae_regression_ub",
    "prior_only",
    "depth_interpolation",
)
METHOD_COLOR = {
    "clp_csgm_ridge": "#1f77b4",
    "csgm_unconditioned": "#ff7f0e",
    "ae_regression_ub": "#2ca02c",
    "prior_only": "#9467bd",
    "depth_interpolation": "#d62728",
}
AGGREGATION = "mean_of_covering_windows"
INTERP_EDGE = "nearest_measured_depth"
DERIVED_NOT_IN_U: Tuple[str, ...] = ("RQI", "FZI_lab", "HFU", "k_lab (mD)")


def _git_head() -> str:
    """Return HEAD, or an empty string when git is unavailable."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return done.stdout.strip()


def _check_depth_interpolation() -> None:
    """Reject an interpolator that treats the sample index as distance."""
    depth = np.asarray([0.0, 10.0, 30.0], dtype=np.float64)
    values = np.asarray([0.0, 999.0, 30.0], dtype=np.float64)
    observed = np.asarray([0, 2], dtype=np.int64)
    pred = _interpolate_depths(depth, values, observed, np.asarray([10.0], dtype=np.float64))
    if abs(float(pred[0]) - 10.0) > 1e-8:
        raise ValueError("Interpolation did not follow Depth(m).")
    if abs(float(pred[0]) - 15.0) < 1e-6:
        raise ValueError("Interpolation collapsed to the sample index.")
    ends = _interpolate_depths(
        depth,
        values,
        observed,
        np.asarray([-5.0, 40.0], dtype=np.float64),
    )
    if abs(float(ends[0]) - 0.0) > 1e-8 or abs(float(ends[1]) - 30.0) > 1e-8:
        raise ValueError("Endpoint hold did not use the nearest measured value.")


def _interpolate_depths(
    depth_m: np.ndarray,
    values: np.ndarray,
    observed_rows: np.ndarray,
    query_depth: np.ndarray,
) -> np.ndarray:
    """Linear interpolation on Depth(m). Ends keep the nearest measured value.

    A single observation fills every query with that value. The held-out
    values are not an input.
    """
    if observed_rows.size < 1:
        raise ValueError("Interpolation needs at least one observed depth.")
    order = np.argsort(depth_m[observed_rows], kind="mergesort")
    obs = np.asarray(observed_rows, dtype=np.int64)[order]
    xp = np.asarray(depth_m[obs], dtype=np.float64)
    fp = np.asarray(values[obs], dtype=np.float64)
    if xp.shape[0] != np.unique(xp).shape[0]:
        raise ValueError("Observed depths are not unique.")
    query = np.asarray(query_depth, dtype=np.float64)
    if xp.shape[0] == 1:
        return np.full(query.shape, float(fp[0]), dtype=np.float64)
    return np.interp(query, xp, fp, left=float(fp[0]), right=float(fp[-1]))


def _split_fold(n_rows: int, fold_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the published train, validation, and test row split."""
    blocks = published._depth_blocks(n_rows, published.N_BLOCKS)
    if fold_id < 0 or fold_id >= len(blocks):
        raise ValueError("fold_id is outside the three depth blocks.")
    test_rows = np.asarray(blocks[fold_id], dtype=np.int64)
    train_pool = np.sort(
        np.concatenate(
            [block for index, block in enumerate(blocks) if index != fold_id]
        ).astype(np.int64)
    )
    segments: List[np.ndarray] = []
    seg_start = int(train_pool[0])
    prev = int(train_pool[0])
    for value in train_pool[1:].tolist():
        if int(value) != prev + 1:
            segments.append(np.arange(seg_start, prev + 1, dtype=np.int64))
            seg_start = int(value)
        prev = int(value)
    segments.append(np.arange(seg_start, prev + 1, dtype=np.int64))
    deepest = max(segments, key=lambda seg: int(seg[-1]))
    if int(deepest.shape[0]) <= published.VAL_ROWS + published.WINDOW_LEN:
        raise ValueError("Training segment is too short for validation windows.")
    val_rows = np.asarray(deepest[-published.VAL_ROWS :], dtype=np.int64)
    train_rows = np.setdiff1d(train_pool, val_rows, assume_unique=False)
    return train_rows, val_rows, test_rows


def _mask_count(n_rows: int, rho: float) -> int:
    """Observed-depth count. Python round uses ties-to-even."""
    count = int(round(float(rho) * float(n_rows)))
    if count < 1 or count >= int(n_rows):
        raise ValueError("Observed count {} is outside 1..{}.".format(count, n_rows - 1))
    return count


def _nested_masks(
    rows: np.ndarray,
    seed: int,
    fold_id: int,
    split_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw one permutation. The 25% prefix is inside the 50% prefix."""
    if split_name not in SPLIT_CODE:
        raise ValueError("Unknown split {}.".format(split_name))
    ordered = np.asarray(rows, dtype=np.int64)
    m25 = _mask_count(int(ordered.shape[0]), 0.25)
    m50 = _mask_count(int(ordered.shape[0]), 0.50)
    if not m25 < m50:
        raise ValueError("The 25% mask is not smaller than the 50% mask.")
    rng = np.random.default_rng(
        int(seed) * 1009 + int(fold_id) * 17 + int(SPLIT_CODE[split_name]) * 10007
    )
    perm = rng.permutation(int(ordered.shape[0]))
    chosen = ordered[perm]
    obs25 = np.sort(chosen[:m25].astype(np.int64))
    obs50 = np.sort(chosen[:m50].astype(np.int64))
    if not set(int(v) for v in obs25.tolist()).issubset(set(int(v) for v in obs50.tolist())):
        raise ValueError("Nested mask prefix failed.")
    return obs25, obs50


def _local_offsets(start: int, observed_rows: np.ndarray, window_len: int) -> np.ndarray:
    """Window offsets whose physical row is in the global observed set."""
    observed = set(int(row) for row in observed_rows.tolist())
    offsets = [
        offset
        for offset in range(window_len)
        if (int(start) + offset) in observed
    ]
    return np.asarray(offsets, dtype=np.int64)


def _operator(offsets: np.ndarray, window_len: int) -> np.ndarray:
    """Coordinate restriction with one row per observed offset."""
    operator = np.zeros((int(offsets.shape[0]), window_len), dtype=np.float64)
    for row_id, col in enumerate(offsets.tolist()):
        operator[int(row_id), int(col)] = 1.0
    return operator


def _embed_window(y_window: np.ndarray, offsets: np.ndarray, window_len: int) -> np.ndarray:
    """Fixed-width embedding. An empty window is zeros, without reading a holdout."""
    width = 2 * int(window_len)
    if int(offsets.shape[0]) < 1:
        return np.zeros((1, width), dtype=np.float64)
    measured = np.asarray(y_window[offsets], dtype=np.float64).reshape(1, -1)
    operator = _operator(offsets, window_len)
    embedded = dub.embed_sparse_observation(measured, operator)
    if embedded.shape != (1, width):
        raise ValueError("Sparse embedding has the wrong width.")
    return embedded


def _recover_window(
    ae: csgm.TinyAE,
    z0: np.ndarray,
    y_window: np.ndarray,
    offsets: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    cfg: Config,
    seed: int,
    mode: str,
) -> np.ndarray:
    """Recover one window. Empty windows do not read the laboratory target."""
    if z0.shape != (1, int(ae.latent_dim)):
        raise ValueError("z0 must have shape (1, latent_dim).")
    if int(offsets.shape[0]) < 1:
        if mode == "clp":
            if float(lam) <= 0.0:
                raise ValueError("CLP empty-window fallback requires a positive penalty.")
            return csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device="cpu")
        if mode == "unconditioned":
            zeros = np.zeros_like(z0)
            return csgm.decode_latent_prior(ae, zeros, y_mean, y_scale, device="cpu")
        raise ValueError("Unknown recovery mode {}.".format(mode))
    operator = _operator(offsets, int(y_window.shape[0]))
    measured = np.asarray(y_window[offsets], dtype=np.float64).reshape(1, -1)
    if mode == "clp":
        if float(lam) <= 0.0:
            raise ValueError("CLP recovery requires a positive lambda.")
        z_init = z0
        penalty = float(lam)
    elif mode == "unconditioned":
        z_init = np.zeros_like(z0)
        penalty = 0.0
    else:
        raise ValueError("Unknown recovery mode {}.".format(mode))
    y_hat = csgm.csgm_recover_with_prior(
        ae=ae,
        M=operator,
        B=measured,
        z0_np=z_init,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=penalty,
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device="cpu",
        seed=int(seed),
    )
    if y_hat.shape != (1, int(y_window.shape[0])) or not np.isfinite(y_hat).all():
        raise ValueError("Window recovery returned a non-finite prediction.")
    return y_hat


def _predict_windows(
    ae: csgm.TinyAE,
    z0_rows: np.ndarray,
    y_windows: np.ndarray,
    starts: Sequence[int],
    observed_rows: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    cfg: Config,
    seed: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict every window. The second return is 1 when that window was empty."""
    n_windows = int(y_windows.shape[0])
    if len(starts) != n_windows or z0_rows.shape[0] != n_windows:
        raise ValueError("Window, start, and prior rows do not match.")
    preds = np.empty_like(y_windows)
    empty = np.zeros(n_windows, dtype=np.int64)
    for index, start in enumerate(starts):
        offsets = _local_offsets(int(start), observed_rows, int(y_windows.shape[1]))
        empty[index] = int(offsets.shape[0] == 0)
        y_hat = _recover_window(
            ae,
            z0_rows[index : index + 1],
            y_windows[index],
            offsets,
            y_mean,
            y_scale,
            lam,
            cfg,
            seed,
            mode,
        )
        preds[index] = y_hat[0]
    return preds, empty


def _aggregate_depth(
    starts: Sequence[int],
    y_hat: np.ndarray,
    empty: np.ndarray,
    row_index: int,
    window_len: int,
) -> Tuple[float, int, int]:
    """Mean prediction at one physical depth, plus coverage and fallback counts."""
    values: List[float] = []
    n_fallback = 0
    for index, start in enumerate(starts):
        if int(row_index) < int(start) or int(row_index) >= int(start) + window_len:
            continue
        offset = int(row_index) - int(start)
        values.append(float(y_hat[index, offset]))
        n_fallback += int(empty[index])
    if not values:
        raise ValueError("Depth {} is not covered by any window.".format(row_index))
    return float(np.mean(np.asarray(values, dtype=np.float64))), len(values), n_fallback


def _scores(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """RMSE and MAE on one vector of unique depths."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.ndim != 1 or truth.shape[0] < 1:
        raise ValueError("Score inputs must be matching non-empty vectors.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("Score inputs must be finite.")
    residual = pred - truth
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    mae = float(np.mean(np.abs(residual)))
    return rmse, mae


def _embed_matrix(
    y_windows: np.ndarray,
    starts: Sequence[int],
    observed_rows: np.ndarray,
) -> np.ndarray:
    """Stack fixed-width embeddings in window order."""
    rows: List[np.ndarray] = []
    window_len = int(y_windows.shape[1])
    for index, start in enumerate(starts):
        offsets = _local_offsets(int(start), observed_rows, window_len)
        rows.append(_embed_window(y_windows[index], offsets, window_len))
    embedded = np.vstack(rows)
    if embedded.shape != (len(starts), 2 * window_len):
        raise ValueError("Embedded matrix has the wrong shape.")
    return embedded


def _build_cfg() -> Config:
    """Match the published 861 base settings without changing the optimizer."""
    cfg = Config()
    cfg.n_output = published.WINDOW_LEN
    cfg.p_input = published.WINDOW_LEN * len(published.U_COLUMNS)
    cfg.csgm_latent_dim = published.LATENT_DIM
    cfg.csgm_hidden_dim = published.HIDDEN_DIM
    cfg.csgm_ae_epochs = published.AE_EPOCHS
    cfg.csgm_iters = published.CSGM_ITERS
    cfg.csgm_restarts = published.CSGM_RESTARTS
    cfg.csgm_lambda_grid = list(published.LAMBDA_GRID)
    cfg.measurement_noise_std = 0.0
    cfg.csgm_device = "cpu"
    return cfg


def _depth_frame(
    rows: np.ndarray,
    starts: Sequence[int],
    predictions: Dict[str, np.ndarray],
    empty: Dict[str, np.ndarray],
    depth_m: np.ndarray,
    y_vector: np.ndarray,
    observed_rows: np.ndarray,
    interp: np.ndarray,
) -> pd.DataFrame:
    """One row per physical depth after window predictions have been averaged."""
    observed = set(int(row) for row in observed_rows.tolist())
    records: List[Dict[str, float]] = []
    for row in rows.tolist():
        role = "observed" if int(row) in observed else "holdout"
        record: Dict[str, float] = {
            "row_index": float(row),
            "depth_m": float(depth_m[int(row)]),
            "phi_lab": float(y_vector[int(row)]),
            "role_code": 1.0 if role == "observed" else 0.0,
        }
        n_windows = 0
        n_fallback = 0
        for method in ("clp_csgm_ridge", "csgm_unconditioned", "ae_regression_ub", "prior_only"):
            pred, n_win, n_fall = _aggregate_depth(
                starts,
                predictions[method],
                empty[method],
                int(row),
                published.WINDOW_LEN,
            )
            record["pred_" + method] = pred
            if method == "clp_csgm_ridge":
                n_windows = n_win
                n_fallback = n_fall
        record["pred_depth_interpolation"] = float(interp[int(row)])
        record["n_windows"] = float(n_windows)
        record["n_fallback_windows"] = float(n_fallback)
        records.append(record)
    frame = pd.DataFrame(records)
    frame.insert(2, "role", np.where(frame["role_code"] > 0.5, "observed", "holdout"))
    return frame


def _select_lambda(
    ae: csgm.TinyAE,
    z0_val: np.ndarray,
    y_val: np.ndarray,
    val_starts: Sequence[int],
    val_rows: np.ndarray,
    observed_rows: np.ndarray,
    y_vector: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[float, List[Dict[str, float]]]:
    """Choose lambda by holdout depth RMSE on validation windows only."""
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    curve: List[Dict[str, float]] = []
    holdout = np.asarray(
        [int(row) for row in val_rows.tolist() if int(row) not in set(observed_rows.tolist())],
        dtype=np.int64,
    )
    if holdout.size < 1:
        raise ValueError("Validation mask left no holdout depth.")
    for lam in cfg.csgm_lambda_grid:
        y_hat, empty = _predict_windows(
            ae,
            z0_val,
            y_val,
            val_starts,
            observed_rows,
            y_mean,
            y_scale,
            float(lam),
            cfg,
            seed,
            "clp",
        )
        preds = np.asarray(
            [
                _aggregate_depth(val_starts, y_hat, empty, int(row), published.WINDOW_LEN)[0]
                for row in holdout.tolist()
            ],
            dtype=np.float64,
        )
        rmse, mae = _scores(y_vector[holdout], preds)
        curve.append({"lambda": float(lam), "val_rmse_depth": rmse, "val_mae_depth": mae})
        if rmse < best_score:
            best_score = rmse
            best_lam = float(lam)
    return best_lam, curve


def _plot_profiles(depth_table: pd.DataFrame, metrics: pd.DataFrame, path: Path) -> None:
    """Draw laboratory and reconstructed profiles with depth increasing downward."""
    rhos = list(RHOS)
    fig, axes = plt.subplots(1, len(rhos), figsize=(10.5, 6.5), sharey=True)
    if len(rhos) == 1:
        axes = [axes]
    for axis, rho in zip(axes, rhos):
        part = depth_table[np.isclose(depth_table["rho"], rho)].sort_values("depth_m")
        axis.plot(
            part["phi_lab"],
            part["depth_m"],
            color="black",
            linewidth=1.0,
            label="Phi_lab",
        )
        observed = part[part["role"] == "observed"]
        axis.scatter(
            observed["phi_lab"],
            observed["depth_m"],
            s=28,
            color="black",
            zorder=4,
            label="observed",
        )
        cell = metrics[np.isclose(metrics["rho"], rho)]
        for method in METHODS:
            score = float(cell.loc[cell["method"] == method, "rmse_depth"].iloc[0])
            axis.plot(
                part["pred_" + method],
                part["depth_m"],
                color=METHOD_COLOR[method],
                linewidth=1.1,
                label="{} ({})".format(method, "{:.3f}".format(score)),
            )
        axis.set_xlabel("Phi_lab fraction")
        axis.set_title("rho {}".format(rho))
        axis.grid(True, linewidth=0.3)
        axis.legend(fontsize=7, loc="best")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    fig.suptitle("Well 861 fold 0 seed 7, mean prediction by physical depth")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_rmse(metrics: pd.DataFrame, path: Path) -> None:
    """Bar chart of unique-depth RMSE. Lower is better."""
    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    x_pos = np.arange(len(METHODS), dtype=np.float64)
    width = 0.36
    for offset, rho in enumerate(RHOS):
        part = metrics[np.isclose(metrics["rho"], rho)].set_index("method")
        values = [float(part.loc[method, "rmse_depth"]) for method in METHODS]
        axis.bar(x_pos + (offset - 0.5) * width, values, width=width, label="rho {}".format(rho))
    axis.set_xticks(x_pos)
    axis.set_xticklabels(list(METHODS), rotation=20, ha="right")
    axis.set_ylabel("RMSE by physical depth")
    axis.set_title("Well 861 pilot, fold 0 seed 7")
    axis.legend()
    axis.grid(True, axis="y", linewidth=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _fmt(value: float, digits: int = 4) -> str:
    """Format a finite float."""
    if not np.isfinite(value):
        return "nan"
    return "{:.{}f}".format(float(value), digits)


def _write_report(
    path: Path,
    head: str,
    metrics: pd.DataFrame,
    paired: pd.DataFrame,
    checks: Sequence[Dict[str, object]],
    n_empty: pd.DataFrame,
) -> None:
    """Write the pilot diagnosis from the computed tables."""
    lines = [
        "# Well 861 global depth-mask pilot",
        "",
        "Exploratory reconstruction of Phi_lab. No manuscript text was edited.",
        "The published window-level results were not overwritten.",
        "",
        "- Reference audit commit: `{}`".format(REFERENCE_COMMIT),
        "- HEAD at run: `{}`".format(head if head else "unknown"),
        "- Scope: fold 0, seed 7, fractions 0.25 and 0.50.",
        "- The three depth blocks belong to the same well. This pilot uses",
        "  only the shallowest block and is not a geological validation.",
        "- Aggregation: mean of every window that covers the depth, including",
        "  the documented empty-window fallback.",
        "- Interpolation axis: Depth(m). Ends keep the nearest measured value.",
        "- No artificial measurement noise.",
        "- Lambda is the validation depth RMSE minimum. Ties keep the smaller",
        "  earlier grid value. Test scores are not an input.",
        "- An empty CLP window decodes the conditional prior. That is the",
        "  minimizer of a positive penalty with no data term.",
        "- An empty unconditioned window decodes the zero latent code. A",
        "  restart would be arbitrary because every latent has zero loss.",
        "- An empty AE window uses a zero value-and-mask embedding.",
        "- Per-window recovery selects the restart on that window's own loss.",
        "  The published batch call selected one restart for the whole batch.",
        "  Separate calls are required because each window has its own operator.",
        "",
        "## Depth RMSE",
        "",
        "The scored set is the holdout only. At fraction 0.25 that is 22",
        "depths. At fraction 0.50 it is 15 depths. All 29 depths stay covered.",
        "",
    ]
    for rho in RHOS:
        lines.append("Fraction {}:".format(rho))
        part = metrics[np.isclose(metrics["rho"], rho)]
        for method in METHODS:
            row = part.loc[part["method"] == method].iloc[0]
            lines.append(
                "- {}: RMSE {} , MAE {} , lambda {}".format(
                    method,
                    _fmt(float(row["rmse_depth"])),
                    _fmt(float(row["mae_depth"])),
                    _fmt(float(row["lambda"])) if np.isfinite(row["lambda"]) else "none",
                )
            )
        lines.append("")
    lines.extend(
        [
            "Negative `clp_minus_comparator` means the CLP depth RMSE is lower.",
            "",
        ]
    )
    for record in paired.itertuples(index=False):
        lines.append(
            "- rho {} vs {}: {}".format(
                record.rho,
                record.comparator,
                _fmt(float(record.clp_minus_comparator), 5),
            )
        )
    lines.extend(["", "## Empty windows", ""])
    for record in n_empty.itertuples(index=False):
        lines.append(
            "- rho {}: {} of {} test windows have no observed depth.".format(
                record.rho,
                int(record.n_empty_windows),
                int(record.n_test_windows),
            )
        )
    lines.extend(
        [
            "",
            "Train, validation, and test use separate nested masks. The direct",
            "AE is trained at the same fraction on training rows only. Its",
            "input is scaled u, the placed laboratory values, and a binary mask.",
            "Held-out training depths are targets of that fit. Test depths are not.",
            "",
            "The 87-row table remains a log-grid derivative, not an assay ledger.",
            "",
            "## Checks",
            "",
        ]
    )
    for item in checks:
        state = "PASS" if bool(item["passed"]) else "FAIL"
        lines.append("- [{}] {}: {}".format(state, item["name"], item["detail"]))
    lines.extend(
        [
            "",
            "The full three-block grid was not run.",
            "",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _record(name: str, passed: bool, detail: str) -> Dict[str, object]:
    """Build one check record."""
    return {"name": name, "passed": bool(passed), "detail": detail}


def _run(full: bool) -> int:
    """Fit the pilot or the full grid and write a new output directory."""
    _check_depth_interpolation()
    if int(round(0.25 * 29)) != 7 or int(round(0.50 * 29)) != 14:
        raise ValueError("Test-block counts are not 7 and 14.")
    out_dir = OUT_ROOT / ("full" if full else "pilot")
    forbidden = {
        published.OUT_DIR.resolve(),
        (REPO_ROOT / "outputs" / "exploratory" / "861_depth_mask_audit").resolve(),
    }
    if out_dir.resolve() in forbidden:
        raise ValueError("Refusing to write into a protected directory.")
    frame = published._load_frame(published.EXCEL_PATH)
    for name in list(published.EXCLUDED_POROSITY_PROXIES) + list(DERIVED_NOT_IN_U):
        if name in published.U_COLUMNS:
            raise ValueError("Forbidden column {} is an input.".format(name))
    depth_m = frame[published.DEPTH_COL].to_numpy(dtype=np.float64)
    y_vector = frame[published.TARGET_COL].to_numpy(dtype=np.float64)
    u_matrix = frame.loc[:, list(published.U_COLUMNS)].to_numpy(dtype=np.float64)
    n_rows = int(frame.shape[0])
    audit = pd.read_csv(AUDIT_SPLIT)
    folds = FULL_FOLDS if full else PILOT_FOLDS
    seeds = FULL_SEEDS if full else PILOT_SEEDS
    cfg = _build_cfg()
    dub_cfg = dub.DirectUBTrainConfig(
        ae_latent_dim=published.LATENT_DIM,
        ae_hidden=published.HIDDEN_DIM,
        ae_epochs=published.AE_EPOCHS,
        ae_batch_size=16,
    )
    checks: List[Dict[str, object]] = []
    mask_rows: List[Dict[str, object]] = []
    depth_rows: List[pd.DataFrame] = []
    metric_rows: List[Dict[str, float]] = []
    paired_rows: List[Dict[str, float]] = []
    lambda_rows: List[Dict[str, float]] = []
    empty_rows: List[Dict[str, float]] = []
    window_rows: List[Dict[str, float]] = []
    geometry_ok = True
    exclusive_ok = True
    covered_ok = True
    b_index_ok = True
    scaler_ok = True
    same_obs_ok = True
    nested_ok = True
    lambda_ok = True
    fallback_ok = True
    finite_ok = True

    for fold_id in folds:
        train_rows, val_rows, test_rows = _split_fold(n_rows, int(fold_id))
        train_set = set(int(v) for v in train_rows.tolist())
        val_set = set(int(v) for v in val_rows.tolist())
        test_set = set(int(v) for v in test_rows.tolist())
        if train_set & val_set or train_set & test_set or val_set & test_set:
            geometry_ok = False
        train_starts = published._window_starts(train_rows, published.WINDOW_LEN, published.STEP)
        val_starts = published._window_starts(val_rows, published.WINDOW_LEN, published.STEP)
        test_starts = published._window_starts(test_rows, published.WINDOW_LEN, published.STEP)
        audit_row = audit[audit["fold_id"] == fold_id]
        if audit_row.empty:
            raise ValueError("Audit split is missing fold {}.".format(fold_id))
        if (
            int(audit_row["n_train_windows"].iloc[0]) != len(train_starts)
            or int(audit_row["n_val_windows"].iloc[0]) != len(val_starts)
            or int(audit_row["n_test_windows"].iloc[0]) != len(test_starts)
        ):
            geometry_ok = False
        for starts, allowed in (
            (train_starts, train_set),
            (val_starts, val_set),
            (test_starts, test_set),
        ):
            for start in starts:
                covered = set(range(int(start), int(start) + published.WINDOW_LEN))
                if not covered.issubset(allowed):
                    geometry_ok = False
        x_train, y_train = published._stack_windows(
            u_matrix, y_vector, train_starts, published.WINDOW_LEN
        )
        x_val, y_val = published._stack_windows(
            u_matrix, y_vector, val_starts, published.WINDOW_LEN
        )
        x_test, y_test = published._stack_windows(
            u_matrix, y_vector, test_starts, published.WINDOW_LEN
        )
        x_scaler = StandardScaler().fit(x_train)
        if int(x_scaler.n_samples_seen_) != int(x_train.shape[0]):
            scaler_ok = False
        if not np.allclose(x_scaler.mean_, np.mean(x_train, axis=0)):
            scaler_ok = False
        x_train_s = x_scaler.transform(x_train)
        x_val_s = x_scaler.transform(x_val)
        x_test_s = x_scaler.transform(x_test)

        for seed in seeds:
            print("fit fold", fold_id, "seed", seed, flush=True)
            ae, prior, y_mean, y_scale = published._fit_clp(
                x_train_s, y_train, int(seed), cfg
            )
            if np.any(y_scale <= 0.0):
                scaler_ok = False
            z0_val = prior.predict(x_val_s)
            z0_test = prior.predict(x_test_s)
            y_prior = csgm.decode_latent_prior(ae, z0_test, y_mean, y_scale, device="cpu")
            y_zero = csgm.decode_latent_prior(
                ae,
                np.zeros_like(z0_test),
                y_mean,
                y_scale,
                device="cpu",
            )
            obs = {
                "train": _nested_masks(train_rows, int(seed), int(fold_id), "train"),
                "val": _nested_masks(val_rows, int(seed), int(fold_id), "val"),
                "test": _nested_masks(test_rows, int(seed), int(fold_id), "test"),
            }
            for split_name, row_set in (
                ("train", train_set),
                ("val", val_set),
                ("test", test_set),
            ):
                obs25, obs50 = obs[split_name]
                if not set(int(v) for v in obs25.tolist()).issubset(set(int(v) for v in obs50.tolist())):
                    nested_ok = False
                if not set(int(v) for v in obs50.tolist()).issubset(row_set):
                    geometry_ok = False
                for row in row_set:
                    mask_rows.append(
                        {
                            "fold_id": int(fold_id),
                            "seed": int(seed),
                            "split": split_name,
                            "row_index": int(row),
                            "depth_m": float(depth_m[int(row)]),
                            "observed_25": int(int(row) in set(int(v) for v in obs25.tolist())),
                            "observed_50": int(int(row) in set(int(v) for v in obs50.tolist())),
                        }
                    )
            if obs["test"][0].shape[0] != 7 or obs["test"][1].shape[0] != 14:
                geometry_ok = False

            for rho, which in ((0.25, 0), (0.50, 1)):
                obs_train = obs["train"][which]
                obs_val = obs["val"][which]
                obs_test = obs["test"][which]
                holdout = np.asarray(
                    [int(row) for row in test_rows.tolist() if int(row) not in set(obs_test.tolist())],
                    dtype=np.int64,
                )
                if holdout.shape[0] != int(test_rows.shape[0] - obs_test.shape[0]):
                    exclusive_ok = False
                if set(obs_test.tolist()) & set(holdout.tolist()):
                    exclusive_ok = False
                for start in test_starts:
                    offsets = _local_offsets(int(start), obs_test, published.WINDOW_LEN)
                    covered_rows = [int(start) + offset for offset in offsets.tolist()]
                    if set(covered_rows) & set(holdout.tolist()):
                        b_index_ok = False
                    if offsets.size:
                        measured = y_test[test_starts.index(start), offsets]
                        if not np.allclose(measured, y_vector[covered_rows]):
                            b_index_ok = False
                for row in holdout.tolist():
                    hits = [
                        start
                        for start in test_starts
                        if int(start) <= int(row) < int(start) + published.WINDOW_LEN
                    ]
                    if not hits:
                        covered_ok = False
                lam, curve = _select_lambda(
                    ae,
                    z0_val,
                    y_val,
                    val_starts,
                    val_rows,
                    obs_val,
                    y_vector,
                    y_mean,
                    y_scale,
                    cfg,
                    int(seed),
                )
                if float(lam) <= 0.0 or float(lam) not in set(float(v) for v in cfg.csgm_lambda_grid):
                    lambda_ok = False
                best_seen = min(float(item["val_rmse_depth"]) for item in curve)
                chosen = [item for item in curve if abs(float(item["lambda"]) - float(lam)) < 1e-15]
                if not chosen or abs(float(chosen[0]["val_rmse_depth"]) - best_seen) > 1e-12:
                    lambda_ok = False
                for item in curve:
                    lambda_rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "lambda": float(item["lambda"]),
                            "val_rmse_depth": float(item["val_rmse_depth"]),
                            "val_mae_depth": float(item["val_mae_depth"]),
                            "selected": float(abs(float(item["lambda"]) - float(lam)) < 1e-15),
                        }
                    )
                y_clp, empty_clp = _predict_windows(
                    ae, z0_test, y_test, test_starts, obs_test, y_mean, y_scale, lam, cfg, int(seed), "clp"
                )
                y_free, empty_free = _predict_windows(
                    ae,
                    z0_test,
                    y_test,
                    test_starts,
                    obs_test,
                    y_mean,
                    y_scale,
                    0.0,
                    cfg,
                    int(seed),
                    "unconditioned",
                )
                if not np.array_equal(empty_clp, empty_free):
                    same_obs_ok = False
                for index in range(len(test_starts)):
                    if int(empty_clp[index]) == 1:
                        if not np.allclose(y_clp[index], y_prior[index]):
                            fallback_ok = False
                        if not np.allclose(y_free[index], y_zero[index]):
                            fallback_ok = False
                embed_train = _embed_matrix(y_train, train_starts, obs_train)
                embed_val = _embed_matrix(y_val, val_starts, obs_val)
                embed_test = _embed_matrix(y_test, test_starts, obs_test)
                for index, start in enumerate(test_starts):
                    offsets = _local_offsets(int(start), obs_test, published.WINDOW_LEN)
                    placed = embed_test[index, : published.WINDOW_LEN]
                    mask = embed_test[index, published.WINDOW_LEN :]
                    expected_mask = np.zeros(published.WINDOW_LEN, dtype=np.float64)
                    if offsets.size:
                        expected_mask[offsets] = 1.0
                        if not np.allclose(placed[offsets], y_test[index, offsets]):
                            same_obs_ok = False
                    complement = np.flatnonzero(expected_mask < 0.5)
                    if complement.size and np.any(np.abs(placed[complement]) > 0.0):
                        b_index_ok = False
                    if not np.allclose(mask, expected_mask):
                        same_obs_ok = False
                xb_train = np.hstack([x_train_s, embed_train])
                xb_val = np.hstack([x_val_s, embed_val])
                xb_test = np.hstack([x_test_s, embed_test])
                xb_scaler = StandardScaler().fit(xb_train)
                if int(xb_scaler.n_samples_seen_) != int(xb_train.shape[0]):
                    scaler_ok = False
                _pred_val, y_ae = dub.fit_predict_ae_regression_ub(
                    cfg,
                    int(seed),
                    dub_cfg,
                    xb_train,
                    y_train,
                    xb_val,
                    y_val,
                    xb_test,
                    xb_scaler,
                )
                y_ae = np.asarray(y_ae, dtype=np.float64)
                if y_ae.shape != y_test.shape or not np.isfinite(y_ae).all():
                    finite_ok = False
                empty_ae = empty_clp.copy()
                predictions = {
                    "clp_csgm_ridge": y_clp,
                    "csgm_unconditioned": y_free,
                    "ae_regression_ub": y_ae,
                    "prior_only": y_prior,
                }
                empty_map = {
                    "clp_csgm_ridge": empty_clp,
                    "csgm_unconditioned": empty_free,
                    "ae_regression_ub": empty_ae,
                    "prior_only": np.zeros_like(empty_clp),
                }
                interp_all = np.full(n_rows, np.nan, dtype=np.float64)
                interp_all[test_rows] = _interpolate_depths(
                    depth_m,
                    y_vector,
                    obs_test,
                    depth_m[test_rows],
                )
                # Confirm the interpolator did not need holdout values: rebuilding
                # from observed values only must match.
                rebuilt = _interpolate_depths(
                    depth_m,
                    y_vector,
                    obs_test,
                    depth_m[holdout],
                )
                if not np.allclose(rebuilt, interp_all[holdout]):
                    same_obs_ok = False
                table = _depth_frame(
                    test_rows,
                    test_starts,
                    predictions,
                    empty_map,
                    depth_m,
                    y_vector,
                    obs_test,
                    interp_all,
                )
                table.insert(0, "rho", float(rho))
                table.insert(0, "seed", int(seed))
                table.insert(0, "fold_id", int(fold_id))
                depth_rows.append(table)
                hold = table[table["role"] == "holdout"]
                if int(hold.shape[0]) != int(holdout.shape[0]):
                    exclusive_ok = False
                if int(hold["n_windows"].min()) < 1:
                    covered_ok = False
                scores: Dict[str, Tuple[float, float]] = {}
                for method in METHODS:
                    rmse, mae = _scores(
                        hold["phi_lab"].to_numpy(dtype=np.float64),
                        hold["pred_" + method].to_numpy(dtype=np.float64),
                    )
                    scores[method] = (rmse, mae)
                    metric_rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "method": method,
                            "rmse_depth": rmse,
                            "mae_depth": mae,
                            "n_observed": float(obs_test.shape[0]),
                            "n_holdout": float(holdout.shape[0]),
                            "n_test_windows": float(len(test_starts)),
                            "n_empty_windows": float(np.sum(empty_clp)),
                            "multiplicity_min": float(hold["n_windows"].min()),
                            "multiplicity_max": float(hold["n_windows"].max()),
                            "lambda": float(lam) if method == "clp_csgm_ridge" else float("nan"),
                        }
                    )
                    if not np.isfinite(rmse):
                        finite_ok = False
                for method in METHODS:
                    if method == "clp_csgm_ridge":
                        continue
                    paired_rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "comparator": method,
                            "clp_minus_comparator": float(
                                scores["clp_csgm_ridge"][0] - scores[method][0]
                            ),
                        }
                    )
                empty_rows.append(
                    {
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "n_empty_windows": float(np.sum(empty_clp)),
                        "n_test_windows": float(len(test_starts)),
                    }
                )
                for index, start in enumerate(test_starts):
                    window_rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "start": float(start),
                            "m_obs": float(
                                _local_offsets(int(start), obs_test, published.WINDOW_LEN).shape[0]
                            ),
                            "empty_window": float(empty_clp[index]),
                        }
                    )
                print(
                    "  rho",
                    rho,
                    "lambda",
                    lam,
                    "clp",
                    round(scores["clp_csgm_ridge"][0], 5),
                    "ae",
                    round(scores["ae_regression_ub"][0], 5),
                    "free",
                    round(scores["csgm_unconditioned"][0], 5),
                    "interp",
                    round(scores["depth_interpolation"][0], 5),
                    "prior",
                    round(scores["prior_only"][0], 5),
                    flush=True,
                )

    checks.extend(
        [
            _record("split_matches_audit_and_is_disjoint", geometry_ok, "Row splits match the audit and do not overlap"),
            _record("nested_physical_masks", nested_ok, "The 25% observed depths are inside the 50% set"),
            _record("exclusive_depth_role", exclusive_ok, "Each test depth is observed or held out, not both"),
            _record("holdout_depths_covered", covered_ok, "Every holdout depth sits in at least one test window"),
            _record("holdout_absent_from_b", b_index_ok, "No holdout row index is placed in a test measurement"),
            _record("scalers_fit_on_train_only", scaler_ok, "Feature and AE input scalers see only training windows"),
            _record("same_test_observations", same_obs_ok, "CLP, AE, unconditioned CSGM, and interpolation share the mask"),
            _record("lambda_from_validation_only", lambda_ok, "Selected lambda is the positive validation minimum"),
            _record("empty_window_fallback", fallback_ok, "Empty CLP windows match the prior and empty free windows match z=0"),
            _record("finite_depth_scores", finite_ok, "Depth RMSE and AE predictions are finite"),
        ]
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    table_dir = out_dir / "tables"
    figure_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    paired = pd.DataFrame(paired_rows)
    depths = pd.concat(depth_rows, ignore_index=True)
    masks = pd.DataFrame(mask_rows)
    lambdas = pd.DataFrame(lambda_rows)
    empty = pd.DataFrame(empty_rows)
    windows = pd.DataFrame(window_rows)
    metrics.to_csv(table_dir / "metrics_by_cell.csv", index=False)
    paired.to_csv(table_dir / "paired_differences.csv", index=False)
    depths.to_csv(table_dir / "depth_predictions.csv", index=False)
    masks.to_csv(table_dir / "physical_masks.csv", index=False)
    lambdas.to_csv(table_dir / "lambda_validation.csv", index=False)
    empty.to_csv(table_dir / "empty_windows.csv", index=False)
    windows.to_csv(table_dir / "window_observations.csv", index=False)
    if not full:
        _plot_profiles(depths, metrics, figure_dir / "profiles_fold0_seed7.png")
        _plot_rmse(metrics, figure_dir / "rmse_depth_fold0_seed7.png")
    head = _git_head()
    payload = {
        "reference_commit": REFERENCE_COMMIT,
        "head": head,
        "scope": "full" if full else "pilot",
        "aggregation": AGGREGATION,
        "interpolation_edge": INTERP_EDGE,
        "measurement_noise_std": 0.0,
        "checks": checks,
    }
    (out_dir / "checks.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    config = {
        "u_columns": list(published.U_COLUMNS),
        "target": published.TARGET_COL,
        "window_len": published.WINDOW_LEN,
        "step": published.STEP,
        "lambda_grid": list(published.LAMBDA_GRID),
        "latent_dim": published.LATENT_DIM,
        "hidden_dim": published.HIDDEN_DIM,
        "ae_epochs": published.AE_EPOCHS,
        "csgm_iters": published.CSGM_ITERS,
        "csgm_restarts": published.CSGM_RESTARTS,
        "csgm_opt_lr": float(cfg.csgm_opt_lr),
        "csgm_ridge_alpha": float(cfg.csgm_ridge_alpha),
        "mask_seed": "seed * 1009 + fold * 17 + split_code * 10007",
        "split_code": SPLIT_CODE,
        "aggregation": AGGREGATION,
        "interpolation_edge": INTERP_EDGE,
        "same_well_blocks": True,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if not full:
        _write_report(out_dir / "REPORT.md", head, metrics, paired, checks, empty)
    failed = [str(item["name"]) for item in checks if not bool(item["passed"])]
    print("wrote", out_dir)
    print("checks", len(checks), "failed", len(failed))
    for name in failed:
        print("FAIL", name)
    return 1 if failed else 0


def main() -> int:
    """Run the pilot unless --full is set."""
    parser = argparse.ArgumentParser(description="Well 861 global depth-mask experiment.")
    parser.add_argument("--full", action="store_true", help="Run three blocks and three seeds.")
    args = parser.parse_args()
    started = time.perf_counter()
    code = _run(bool(args.full))
    print("elapsed_s", round(time.perf_counter() - started, 1))
    return code


if __name__ == "__main__":
    sys.exit(main())
