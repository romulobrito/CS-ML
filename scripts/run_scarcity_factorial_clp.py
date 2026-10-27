#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestrate a training-size by observation-fraction pilot.

The script does not reimplement CLP-CSGM. It calls the published window
builder, measurement operator, neural decoder, ridge prior, latent recovery,
and direct autoencoder. One mask and one noisy measurement draw are shared
by every training budget at a fixed seed and observed fraction.

ASCII-only source.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import csgm_m2_module as csgm
import direct_ub_baselines as dub
import multi_well_vc as mwv
from sir_cs_pipeline_optimized import Config, build_measurement_matrix

OUT_DIR = os.path.join(REPO_ROOT, "outputs", "exploratory", "scarcity_factorial")
TRAIN_PATH = os.path.join(
    REPO_ROOT, "data", "F02-1,F03-2,F06-1_6logs_30dB.txt"
)
TEST_PATH = os.path.join(REPO_ROOT, "data", "F03-4_6logs_30dB.txt")
CHANNELS: Tuple[str, ...] = ("sonic", "rhob", "ai", "vp")
TARGET_NAME = "vc"
WINDOW_LEN = 64
# Step 8 is the only published cross-well step with at least 88 windows
# in F02-1 and F03-2 together. Step 16 has 54.
WINDOW_STEP = 8
TRAIN_WELLS: Tuple[str, ...] = ("F02-1", "F03-2")
VAL_WELL = "F06-1"
TEST_WELL = "F03-4"
NOISE_STD = 0.01
PILOT_SEEDS: Tuple[int, ...] = (7,)
FULL_SEEDS: Tuple[int, ...] = (7, 23, 41)
PILOT_SIZES: Tuple[int, ...] = (24, 88)
# 56 is the nested middle budget used only by --full.
FULL_SIZES: Tuple[int, ...] = (24, 56, 88)
RHOS: Tuple[float, ...] = (0.05, 0.20)
METHODS: Tuple[str, ...] = (
    "clp_csgm_ridge",
    "ae_direct",
    "prior_only",
    "interpolation",
)


def fingerprint(M: np.ndarray, B_val: np.ndarray, B_test: np.ndarray) -> str:
    """Identify one mask and the paired validation and test measurements."""
    digest = hashlib.sha256()
    for array in (M, B_val, B_test):
        contiguous = np.ascontiguousarray(array, dtype=np.float64)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def verify_nested_subsets(idx_small: Sequence[int], idx_large: Sequence[int]) -> None:
    """The larger training index set must contain the smaller one."""
    if len(set(idx_small)) != len(idx_small):
        raise AssertionError("Small training indices are not unique.")
    if len(set(idx_large)) != len(idx_large):
        raise AssertionError("Large training indices are not unique.")
    if not set(idx_small).issubset(set(idx_large)):
        raise AssertionError("Small training indices are not nested in the large set.")


def verify_paired_measurements(
    run_small: Mapping[str, object],
    run_large: Mapping[str, object],
) -> None:
    """Distinct budgets must share the same observation problem."""
    if run_small["seed"] != run_large["seed"] or run_small["rho"] != run_large["rho"]:
        raise AssertionError("Paired runs do not share seed and rho.")
    for name in ("M", "B_val", "B_test", "Y_test"):
        np.testing.assert_array_equal(
            run_small[name],
            run_large[name],
            err_msg="Pairing violated: {}".format(name),
        )
    left = fingerprint(run_small["M"], run_small["B_val"], run_small["B_test"])
    right = fingerprint(run_large["M"], run_large["B_val"], run_large["B_test"])
    if left != right:
        raise AssertionError("Measurement fingerprint differs across budgets.")


def verify_training_and_test(
    train_wells: Sequence[str],
    val_wells: Sequence[str],
    test_wells: Sequence[str],
) -> None:
    """Train, validation, and test wells must be disjoint."""
    train_set = set(train_wells)
    val_set = set(val_wells)
    test_set = set(test_wells)
    if not train_set.isdisjoint(val_set):
        raise AssertionError("A training window belongs to the validation well.")
    if not train_set.isdisjoint(test_set):
        raise AssertionError("A training window belongs to the test well.")
    if not val_set.isdisjoint(test_set):
        raise AssertionError("A validation window belongs to the test well.")


def rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed: np.ndarray | None = None,
) -> float:
    """Full-window RMSE, or RMSE after dropping measured coordinates."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape:
        raise AssertionError("Prediction and target shapes differ.")
    if observed is not None:
        keep = np.ones(truth.shape[1], dtype=bool)
        keep[np.asarray(observed, dtype=int)] = False
        if not bool(np.any(keep)):
            raise AssertionError("No unobserved coordinate left to score.")
        truth = truth[:, keep]
        pred = pred[:, keep]
    return float(np.sqrt(np.mean((truth - pred) ** 2)))


def verify_predictions(
    predictions: Mapping[str, np.ndarray],
    Y_test: np.ndarray,
    observed: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Require finite predictions and return both RMSE definitions."""
    scores: Dict[str, Dict[str, float]] = {}
    for method, prediction in predictions.items():
        if prediction.shape != Y_test.shape:
            raise AssertionError("Shape mismatch for {}.".format(method))
        if not bool(np.isfinite(prediction).all()):
            raise AssertionError("Non-finite prediction for {}.".format(method))
        full = rmse(Y_test, prediction)
        held = rmse(Y_test, prediction, observed)
        print(
            method,
            "RMSE full:",
            "{:.6f}".format(full),
            "RMSE held-out:",
            "{:.6f}".format(held),
        )
        scores[method] = {"rmse_full": full, "rmse_unobserved": held}
    return scores


def verify_scaler_on_train(scaler: StandardScaler, train_rows: np.ndarray, name: str) -> None:
    """A scaler may be fit on the training rows and nowhere else."""
    seen = int(getattr(scaler, "n_samples_seen_", -1))
    if seen != int(train_rows.shape[0]):
        raise AssertionError(
            "{} saw {} rows, training has {}.".format(name, seen, train_rows.shape[0])
        )
    np.testing.assert_allclose(
        np.asarray(scaler.mean_, dtype=np.float64),
        np.mean(train_rows, axis=0),
        err_msg="{} mean is not the training mean.".format(name),
    )


def _even_positions(n_items: int, quota: int) -> np.ndarray:
    """Spread `quota` unique positions across `n_items`, in increasing order."""
    if quota < 1 or quota > n_items:
        raise ValueError("Quota {} is outside 1..{}.".format(quota, n_items))
    if quota == n_items:
        return np.arange(n_items, dtype=np.int64)
    raw = np.linspace(0.0, float(n_items - 1), int(quota))
    used = set()
    chosen: List[int] = []
    for value in raw.tolist():
        guess = int(np.rint(value))
        guess = min(max(guess, 0), n_items - 1)
        if guess in used:
            placed = None
            for offset in range(1, n_items):
                left = guess - offset
                right = guess + offset
                if left >= 0 and left not in used:
                    placed = left
                    break
                if right < n_items and right not in used:
                    placed = right
                    break
            if placed is None:
                raise RuntimeError("Could not place a unique window index.")
            guess = int(placed)
        used.add(guess)
        chosen.append(guess)
    out = np.array(sorted(chosen), dtype=np.int64)
    if int(out.size) != int(quota):
        raise RuntimeError("Even sample did not keep the requested quota.")
    return out


def _nested_positions(parent: np.ndarray, quota: int) -> np.ndarray:
    """Take an even subset of an already selected index vector."""
    local = _even_positions(int(parent.size), int(quota))
    return np.sort(parent[local].astype(np.int64))


def _paper_config() -> Config:
    """Published cross-well CLP settings. The architecture is not changed."""
    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.measurement_kind = "subsample"
    cfg.measurement_noise_std = NOISE_STD
    cfg.model_selection_metric = "rmse"
    cfg.csgm_prior_type = "ridge"
    cfg.csgm_latent_dim = 16
    cfg.csgm_hidden_dim = 128
    cfg.csgm_ae_epochs = 200
    cfg.csgm_batch_size = 64
    cfg.csgm_ae_lr = 1e-3
    cfg.csgm_weight_decay = 1e-5
    cfg.csgm_iters = 400
    cfg.csgm_opt_lr = 0.05
    cfg.csgm_restarts = 3
    cfg.csgm_lambda_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    cfg.csgm_ridge_alpha = 1.0
    cfg.csgm_device = "cpu"
    cfg.baseline_early_stopping = True
    return cfg


def _load_role_windows() -> Tuple[mwv.MultiWellWindows, mwv.MultiWellWindows, mwv.MultiWellWindows]:
    """Build step-8 windows and keep wells in separate roles."""
    development = mwv.load_6log_file(
        TRAIN_PATH,
        target_name=TARGET_NAME,
        channels=CHANNELS,
    )
    test_segments = mwv.load_6log_file(
        TEST_PATH,
        target_name=TARGET_NAME,
        channels=CHANNELS,
        well_names=[TEST_WELL],
    )
    by_name = {seg.name: seg for seg in development}
    missing = [name for name in TRAIN_WELLS + (VAL_WELL,) if name not in by_name]
    if missing:
        raise RuntimeError("Missing development wells: {}.".format(missing))
    if len(test_segments) != 1 or test_segments[0].name != TEST_WELL:
        raise RuntimeError("Test file did not resolve to F03-4.")
    train_batches = [
        mwv.build_windows_for_segment(by_name[name], WINDOW_LEN, WINDOW_STEP, CHANNELS)
        for name in TRAIN_WELLS
    ]
    val_windows = mwv.build_windows_for_segment(
        by_name[VAL_WELL], WINDOW_LEN, WINDOW_STEP, CHANNELS
    )
    test_windows = mwv.build_windows_for_segment(
        test_segments[0], WINDOW_LEN, WINDOW_STEP, CHANNELS
    )
    return mwv.concat_windows(train_batches), val_windows, test_windows


def _well_quotas(counts: Mapping[str, int], size: int) -> Dict[str, int]:
    """Split one budget across training wells by their window share."""
    total = int(sum(counts.values()))
    if size > total:
        raise ValueError("Budget {} exceeds the training pool {}.".format(size, total))
    names = list(TRAIN_WELLS)
    raw = [int(round(float(size) * float(counts[name]) / float(total))) for name in names]
    # The last well absorbs the rounding remainder.
    raw[-1] = int(size - sum(raw[:-1]))
    quotas = {name: int(raw[i]) for i, name in enumerate(names)}
    for name, quota in quotas.items():
        if quota < 1 or quota > int(counts[name]):
            raise ValueError("Quota for {} is {}.".format(name, quota))
    if sum(quotas.values()) != int(size):
        raise RuntimeError("Quotas do not sum to the budget.")
    return quotas


def _select_nested_indices(
    wells: np.ndarray,
    sizes: Sequence[int],
) -> Dict[int, np.ndarray]:
    """Even, nested index sets. Selection does not read the target curve."""
    counts = {name: int(np.sum(wells == name)) for name in TRAIN_WELLS}
    offsets: Dict[str, int] = {}
    cursor = 0
    for name in TRAIN_WELLS:
        offsets[name] = cursor
        cursor += counts[name]
    largest = int(max(sizes))
    per_well: Dict[str, Dict[int, np.ndarray]] = {}
    large_quota = _well_quotas(counts, largest)
    for name in TRAIN_WELLS:
        local_large = _even_positions(counts[name], large_quota[name])
        chosen: Dict[int, np.ndarray] = {largest: local_large}
        parent = local_large
        for size in sorted((int(s) for s in sizes if int(s) != largest), reverse=True):
            quota = _well_quotas(counts, size)[name]
            parent = _nested_positions(parent, quota)
            chosen[size] = parent
        per_well[name] = chosen
    out: Dict[int, np.ndarray] = {}
    ordered = sorted(int(size) for size in sizes)
    for size in ordered:
        pieces = []
        for name in TRAIN_WELLS:
            pieces.append(offsets[name] + per_well[name][size])
        out[size] = np.sort(np.concatenate(pieces).astype(np.int64))
    for small, large in zip(ordered, ordered[1:]):
        verify_nested_subsets(out[small].tolist(), out[large].tolist())
    return out


def _observed_columns(operator: np.ndarray) -> np.ndarray:
    """Coordinates that carry a measurement. Columns are unique."""
    column_sums = np.sum(np.abs(operator), axis=0)
    observed = np.flatnonzero(column_sums > 0.0).astype(np.int64)
    if int(observed.size) != int(operator.shape[0]):
        raise RuntimeError("Measurement columns are not a simple coordinate set.")
    return observed


def _interpolate(observations: np.ndarray, positions: np.ndarray, length: int) -> np.ndarray:
    """Linear fill along the window. Ends keep the nearest measured value.

    This is the same np.interp rule as scripts/run_r3m03_simple_baselines.py.
    """
    observed = np.asarray(observations, dtype=np.float64)
    order = np.argsort(positions)
    grid = np.arange(int(length), dtype=np.float64)
    xp = positions[order].astype(np.float64)
    out = np.empty((int(observed.shape[0]), int(length)), dtype=np.float64)
    for row_id in range(int(observed.shape[0])):
        out[row_id] = np.interp(grid, xp, observed[row_id, order])
    return out


def _select_lambda(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_val: np.ndarray,
    y_val: np.ndarray,
    b_val: np.ndarray,
    operator: np.ndarray,
    observed: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    cfg: Config,
    seed: int,
) -> Tuple[float, List[Tuple[float, float]]]:
    """Choose lambda by unobserved-coordinate RMSE on validation only."""
    z0 = prior.predict(x_val)
    records: List[Tuple[float, float]] = []
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    for lam in cfg.csgm_lambda_grid:
        y_hat = csgm.csgm_recover_with_prior(
            ae=ae,
            M=operator,
            B=b_val,
            z0_np=z0,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device="cpu",
            seed=int(seed),
        )
        score = rmse(y_val, y_hat, observed)
        records.append((float(lam), float(score)))
        if score < best_score:
            best_score = float(score)
            best_lam = float(lam)
    chosen = min(records, key=lambda item: item[1])[0]
    if chosen != best_lam:
        raise AssertionError("Lambda was not the validation minimum.")
    return best_lam, records


def _fit_clp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    cfg: Config,
) -> Tuple[csgm.TinyAE, csgm.RidgePrior, StandardScaler, np.ndarray, np.ndarray]:
    """Train the decoder and the ridge anchor on the training subset only."""
    y_scaler = StandardScaler()
    y_train_n = y_scaler.fit_transform(y_train)
    verify_scaler_on_train(y_scaler, y_train, "clp_y_scaler")
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    if np.any(y_scale <= 0.0):
        raise RuntimeError("A training target column has non-positive scale.")
    ae = csgm.train_ae_generator(y_train_n, cfg, seed=int(seed), device="cpu")
    z_train = csgm.encode_y(ae, y_train_n, device="cpu")
    prior = csgm.RidgePrior(alpha=float(cfg.csgm_ridge_alpha), random_state=int(seed))
    prior.fit(x_train, z_train)
    verify_scaler_on_train(prior.x_scaler, x_train, "ridge_x_scaler")
    return ae, prior, y_scaler, y_mean, y_scale


def _predict_clp_and_prior(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_rows: np.ndarray,
    b_rows: np.ndarray,
    operator: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode the prior and refine it with the already drawn measurements."""
    z0 = prior.predict(x_rows)
    y_clp = csgm.csgm_recover_with_prior(
        ae=ae,
        M=operator,
        B=b_rows,
        z0_np=z0,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=float(lam),
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device="cpu",
        seed=int(seed),
    )
    y_prior = csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device="cpu")
    return y_clp, y_prior


def _window_dependence(windows: mwv.MultiWellWindows) -> Dict[str, int]:
    """Count overlapping validation windows. They are not independent wells."""
    starts = np.asarray(windows.row_start_in_well, dtype=np.int64)
    n_windows = int(starts.size)
    overlapping_pairs = 0
    max_shared = 0
    for left in range(n_windows):
        for right in range(left + 1, n_windows):
            shared = int(WINDOW_LEN - abs(int(starts[left]) - int(starts[right])))
            if shared > 0:
                overlapping_pairs += 1
                max_shared = max(max_shared, shared)
    return {
        "n_windows": n_windows,
        "overlapping_pairs": overlapping_pairs,
        "max_shared_samples": max_shared,
        "adjacent_shared_samples": int(WINDOW_LEN - WINDOW_STEP),
    }


def _run_synthetic_checks() -> None:
    """Exercise the checkers before any model is trained."""
    small = [0, 2, 5]
    large = [0, 1, 2, 5, 8]
    verify_nested_subsets(small, large)
    operator = np.zeros((2, 4), dtype=np.float64)
    operator[0, 1] = 1.0
    operator[1, 3] = 1.0
    b_val = np.array([[0.2, 0.4], [0.3, 0.5]], dtype=np.float64)
    b_test = np.array([[0.1, 0.2]], dtype=np.float64)
    y_test = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    run = {
        "seed": 7,
        "rho": 0.05,
        "M": operator,
        "B_val": b_val,
        "B_test": b_test,
        "Y_test": y_test,
    }
    verify_paired_measurements(run, dict(run))
    verify_training_and_test(["F02-1"], ["F06-1"], ["F03-4"])
    pred = np.array([[0.0, 1.5, 2.0, 4.0]], dtype=np.float64)
    scores = verify_predictions({"toy": pred}, y_test, np.array([1, 3], dtype=np.int64))
    # Unobserved columns are 0 and 2. Errors are 0 and 0.
    if abs(scores["toy"]["rmse_unobserved"]) > 1e-12:
        raise AssertionError("Unobserved RMSE did not drop the measured columns.")
    scaler = StandardScaler().fit(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    verify_scaler_on_train(
        scaler,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        "synthetic_scaler",
    )
    print("Synthetic consistency checks passed.")


def _draw_problem(
    y_pool: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    rho: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Draw one mask and one noise field before the budget loop."""
    rng = np.random.default_rng(int(seed) + int(round(100.0 * float(rho))))
    m_obs = max(4, int(round(float(rho) * WINDOW_LEN)))
    operator = build_measurement_matrix(m_obs, WINDOW_LEN, "subsample", rng)
    b_pool = dub.make_B(y_pool, operator, NOISE_STD, rng)
    b_val = dub.make_B(y_val, operator, NOISE_STD, rng)
    b_test = dub.make_B(y_test, operator, NOISE_STD, rng)
    return operator, _observed_columns(operator), b_pool, b_val, b_test


def _save_table(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write one CSV with a fixed column order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _plot(rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Plot full and unobserved RMSE against the training budget."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharex=True)
    metrics = ("rmse_full", "rmse_unobserved")
    titles = ("Full window", "Unobserved coordinates")
    for axis, metric, title in zip(axes, metrics, titles):
        for rho in RHOS:
            for method in METHODS:
                subset = [
                    row
                    for row in rows
                    if float(row["rho"]) == float(rho) and str(row["method"]) == method
                ]
                subset = sorted(subset, key=lambda row: int(row["n_train"]))
                if not subset:
                    continue
                axis.plot(
                    [int(row["n_train"]) for row in subset],
                    [float(row[metric]) for row in subset],
                    marker="o",
                    label="{} rho={:.2f}".format(method, float(rho)),
                )
        axis.set_title(title)
        axis.set_xlabel("Training windows")
        axis.set_ylabel("RMSE")
        axis.grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Training-size by observation-fraction CLP pilot."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--pilot",
        action="store_true",
        help="Seed 7 and training budgets 24 and 88.",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Seeds 7, 23, 41 and nested budgets 24, 56, and 88.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the paired scarcity grid and write tables, checks, and one figure."""
    args = _parse_args()
    _run_synthetic_checks()
    seeds = FULL_SEEDS if args.full else PILOT_SEEDS
    sizes = FULL_SIZES if args.full else PILOT_SIZES
    mode_name = "full" if args.full else "pilot"
    cfg = _paper_config()
    train, val, test = _load_role_windows()
    verify_training_and_test(
        train.well_of_window.tolist(),
        val.well_of_window.tolist(),
        test.well_of_window.tolist(),
    )
    indices = _select_nested_indices(train.well_of_window, FULL_SIZES)
    for size in sizes:
        chosen_wells = train.well_of_window[indices[size]].tolist()
        verify_training_and_test(chosen_wells, [VAL_WELL], [TEST_WELL])
        if TEST_WELL in set(chosen_wells) or VAL_WELL in set(chosen_wells):
            raise AssertionError("A selected training window left the training wells.")
    verify_nested_subsets(indices[24].tolist(), indices[88].tolist())

    val_dependence = _window_dependence(val)
    print(
        "Validation windows:",
        val_dependence["n_windows"],
        "overlapping pairs:",
        val_dependence["overlapping_pairs"],
        "max shared samples:",
        val_dependence["max_shared_samples"],
    )
    print(
        "F06-1 is one well. Its windows share depth samples and are not "
        "independent evidence."
    )

    result_rows: List[Dict[str, object]] = []
    lambda_rows: List[Dict[str, object]] = []
    index_rows: List[Dict[str, object]] = []
    for size, idx in indices.items():
        for pool_index in idx.tolist():
            index_rows.append(
                {
                    "n_train": int(size),
                    "pool_index": int(pool_index),
                    "well": str(train.well_of_window[pool_index]),
                    "row_start": int(train.row_start_in_well[pool_index]),
                    "depth_start": float(train.depth_range[pool_index, 0]),
                    "depth_end": float(train.depth_range[pool_index, 1]),
                }
            )

    y_test = np.asarray(test.y, dtype=np.float64)
    x_test = np.asarray(test.x, dtype=np.float64)
    for seed in seeds:
        for rho in RHOS:
            operator, observed, b_pool, b_val, b_test = _draw_problem(
                np.asarray(train.y, dtype=np.float64),
                np.asarray(val.y, dtype=np.float64),
                y_test,
                float(rho),
                int(seed),
            )
            bundles: Dict[int, Dict[str, object]] = {}
            interp_ref: np.ndarray | None = None
            for size in sizes:
                print(
                    "seed",
                    int(seed),
                    "rho",
                    float(rho),
                    "n_train",
                    int(size),
                    flush=True,
                )
                idx = indices[int(size)]
                x_train = np.asarray(train.x[idx], dtype=np.float64)
                y_train = np.asarray(train.y[idx], dtype=np.float64)
                b_train = np.asarray(b_pool[idx], dtype=np.float64)
                ub_scaler = dub.fit_scaler_ub(dub.concat_ub(x_train, b_train))
                verify_scaler_on_train(
                    ub_scaler,
                    dub.concat_ub(x_train, b_train),
                    "ae_input_scaler",
                )
                # Transforming validation must not refit the scaler.
                _ = ub_scaler.transform(dub.concat_ub(val.x, b_val))
                verify_scaler_on_train(
                    ub_scaler,
                    dub.concat_ub(x_train, b_train),
                    "ae_input_scaler_after_val_transform",
                )
                ae, prior, _y_scaler, y_mean, y_scale = _fit_clp(
                    x_train, y_train, int(seed), cfg
                )
                lam, lam_records = _select_lambda(
                    ae,
                    prior,
                    np.asarray(val.x, dtype=np.float64),
                    np.asarray(val.y, dtype=np.float64),
                    b_val,
                    operator,
                    observed,
                    y_mean,
                    y_scale,
                    cfg,
                    int(seed),
                )
                for lam_value, val_score in lam_records:
                    lambda_rows.append(
                        {
                            "seed": int(seed),
                            "rho": float(rho),
                            "n_train": int(size),
                            "lambda": float(lam_value),
                            "val_rmse_unobserved": float(val_score),
                            "selected": int(float(lam_value) == float(lam)),
                        }
                    )
                y_clp, y_prior = _predict_clp_and_prior(
                    ae,
                    prior,
                    x_test,
                    b_test,
                    operator,
                    y_mean,
                    y_scale,
                    lam,
                    cfg,
                    int(seed),
                )
                _val_ae, y_ae = dub.fit_predict_ae_regression_ub(
                    cfg,
                    int(seed),
                    dub.DirectUBTrainConfig(),
                    dub.concat_ub(x_train, b_train),
                    y_train,
                    dub.concat_ub(val.x, b_val),
                    np.asarray(val.y, dtype=np.float64),
                    dub.concat_ub(x_test, b_test),
                    ub_scaler,
                )
                y_interp = _interpolate(b_test, observed, WINDOW_LEN)
                if interp_ref is None:
                    interp_ref = np.array(y_interp, copy=True)
                else:
                    np.testing.assert_array_equal(y_interp, interp_ref)
                predictions = {
                    "clp_csgm_ridge": y_clp,
                    "ae_direct": y_ae,
                    "prior_only": y_prior,
                    "interpolation": y_interp,
                }
                scores = verify_predictions(predictions, y_test, observed)
                bundles[int(size)] = {
                    "seed": int(seed),
                    "rho": float(rho),
                    "M": operator,
                    "B_val": b_val,
                    "B_test": b_test,
                    "Y_test": y_test,
                }
                for method, metric in scores.items():
                    result_rows.append(
                        {
                            "mode": mode_name,
                            "seed": int(seed),
                            "rho": float(rho),
                            "n_train": int(size),
                            "n_val": int(val.y.shape[0]),
                            "n_test": int(y_test.shape[0]),
                            "m_obs": int(observed.size),
                            "method": method,
                            "rmse_full": float(metric["rmse_full"]),
                            "rmse_unobserved": float(metric["rmse_unobserved"]),
                            "lambda": float(lam) if method == "clp_csgm_ridge" else "",
                        }
                    )
            ordered_sizes = sorted(bundles)
            for left, right in zip(ordered_sizes, ordered_sizes[1:]):
                verify_paired_measurements(bundles[left], bundles[right])
            np.testing.assert_array_equal(y_test, test.y)

    table_dir = os.path.join(OUT_DIR, "tables")
    figure_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(OUT_DIR, exist_ok=True)
    _save_table(
        os.path.join(table_dir, "results.csv"),
        result_rows,
        (
            "mode",
            "seed",
            "rho",
            "n_train",
            "n_val",
            "n_test",
            "m_obs",
            "method",
            "rmse_full",
            "rmse_unobserved",
            "lambda",
        ),
    )
    _save_table(
        os.path.join(table_dir, "lambda_selection.csv"),
        lambda_rows,
        (
            "seed",
            "rho",
            "n_train",
            "lambda",
            "val_rmse_unobserved",
            "selected",
        ),
    )
    _save_table(
        os.path.join(table_dir, "selected_indices.csv"),
        index_rows,
        (
            "n_train",
            "pool_index",
            "well",
            "row_start",
            "depth_start",
            "depth_end",
        ),
    )
    _plot(result_rows, os.path.join(figure_dir, "rmse_by_budget.png"))
    config = {
        "mode": mode_name,
        "seeds": list(seeds),
        "sizes_trained": list(int(size) for size in sizes),
        "sizes_indexed": list(FULL_SIZES),
        "rhos": list(RHOS),
        "window_len": WINDOW_LEN,
        "window_step": WINDOW_STEP,
        "step_reason": (
            "Step 8 is the only published step with at least 88 windows "
            "in F02-1 and F03-2. Step 16 has 54."
        ),
        "channels": list(CHANNELS),
        "target": TARGET_NAME,
        "noise_std": NOISE_STD,
        "train_wells": list(TRAIN_WELLS),
        "val_well": VAL_WELL,
        "test_well": TEST_WELL,
        "n_train_pool": int(train.y.shape[0]),
        "n_val": int(val.y.shape[0]),
        "n_test": int(test.y.shape[0]),
        "val_dependence": val_dependence,
        "val_interpretation": (
            "F06-1 has few overlapping windows from one well. "
            "Lambda selection is not strong statistical evidence."
        ),
        "lambda_rule": "unobserved validation RMSE, matching run_r1m11",
        "csgm_lambda_grid": list(cfg.csgm_lambda_grid),
        "csgm_latent_dim": int(cfg.csgm_latent_dim),
        "csgm_ae_epochs": int(cfg.csgm_ae_epochs),
        "csgm_iters": int(cfg.csgm_iters),
        "csgm_restarts": int(cfg.csgm_restarts),
        "checks": [
            "train wells disjoint from val and test",
            "24 nested in 56 nested in 88",
            "M, B_val, B_test, Y_test shared across budgets",
            "scalers fit on the training subset only",
            "lambda is the validation minimum",
            "unobserved RMSE drops measured coordinates",
            "one test array for every comparison",
            "interpolation identical across budgets",
        ],
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(OUT_DIR, "checks.json"), "w", encoding="utf-8") as handle:
        json.dump({"passed": config["checks"], "n_result_rows": len(result_rows)}, handle, indent=2)
        handle.write("\n")
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
