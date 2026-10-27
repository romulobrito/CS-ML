#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nested observation masks and a conditional-prior ablation.

This study asks two questions the training-size factorial does not answer:

1. Does the CLP-CSGM advantage depend on how many sparse target
   observations are available?
2. Does that advantage come from the conditional prior, or mainly from
   the generative decoder and latent optimization?

The previous script and its outputs are not modified. Training subsets
are read from that study's selected_indices.csv. Masks are nested
prefixes of one permutation, and one noise field is sliced onto those
prefixes. Models A--D share one decoder inside each seed and budget.

ASCII-only source. Exploratory. Not a manuscript result.
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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import csgm_m2_module as csgm
import direct_ub_baselines as dub
import run_scarcity_factorial_clp as base

OUT_DIR = os.path.join(
    REPO_ROOT, "outputs", "exploratory", "scarcity_nested_masks_ablation"
)
INDEX_CSV = os.path.join(
    REPO_ROOT,
    "outputs",
    "exploratory",
    "scarcity_factorial",
    "tables",
    "selected_indices.csv",
)
REFERENCE_COMMIT = "cc57f2bd2babf4b3744bea53e92209bb92911fc9"
# Published measurement-only test recovery uses this offset.
MEASUREMENT_ONLY_SEED_OFFSET = 9200
M_OBS: Tuple[int, ...] = (4, 6, 13)
# Positive values stay on the published ladder. Smaller values test the
# floor seen in 15 of 18 cells of the previous study. Zero is recorded
# on the same curve but is not eligible for the positive-penalty model.
LAMBDA_GRID: Tuple[float, ...] = (
    0.0,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)
EXPECTED_WELL_COUNTS: Dict[int, Dict[str, int]] = {
    24: {"F02-1": 3, "F03-2": 21},
    56: {"F02-1": 7, "F03-2": 49},
    88: {"F02-1": 11, "F03-2": 77},
}
METHOD_CLP = "clp_csgm_ridge"
METHOD_INIT = "clp_init_only"
METHOD_MEAS = "measurement_only_csgm"
METHOD_PRIOR = "prior_only"
METHOD_AE = "ae_direct"
METHOD_INTERP = "interpolation"
METHODS: Tuple[str, ...] = (
    METHOD_CLP,
    METHOD_INIT,
    METHOD_MEAS,
    METHOD_PRIOR,
    METHOD_AE,
    METHOD_INTERP,
)
PILOT_SEEDS: Tuple[int, ...] = (7,)
FULL_SEEDS: Tuple[int, ...] = (7, 23, 41)
PILOT_SIZES: Tuple[int, ...] = (24, 88)
FULL_SIZES: Tuple[int, ...] = (24, 56, 88)


def _sha256_bytes(payload: bytes) -> str:
    """Hash one byte string."""
    return hashlib.sha256(payload).hexdigest()


def _array_hash(arrays: Sequence[np.ndarray]) -> str:
    """Hash shapes and values of several arrays."""
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _operator_from_positions(positions: np.ndarray) -> np.ndarray:
    """Subsample matrix whose row order follows the nested permutation."""
    cols = np.asarray(positions, dtype=np.int64)
    if int(cols.size) != len(set(int(value) for value in cols.tolist())):
        raise AssertionError("Observed positions are not unique.")
    operator = np.zeros((int(cols.size), base.WINDOW_LEN), dtype=np.float64)
    operator[np.arange(int(cols.size)), cols] = 1.0
    return operator


def _slice_measurements(
    y_rows: np.ndarray,
    noise_rows: np.ndarray,
    positions: np.ndarray,
) -> np.ndarray:
    """Keep y + noise on one nested prefix. Shared columns stay identical."""
    cols = np.asarray(positions, dtype=np.int64)
    signal = np.asarray(y_rows, dtype=np.float64)
    noise = np.asarray(noise_rows, dtype=np.float64)
    if signal.shape != noise.shape:
        raise AssertionError("Noise field does not match the target array.")
    return signal[:, cols] + noise[:, cols]


def _draw_nested_problem(
    y_pool: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One permutation and one noise field per split, before any budget loop."""
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(base.WINDOW_LEN).astype(np.int64)
    noise_pool = float(base.NOISE_STD) * rng.normal(size=np.asarray(y_pool).shape)
    noise_val = float(base.NOISE_STD) * rng.normal(size=np.asarray(y_val).shape)
    noise_test = float(base.NOISE_STD) * rng.normal(size=np.asarray(y_test).shape)
    return (
        order,
        np.asarray(noise_pool, dtype=np.float64),
        np.asarray(noise_val, dtype=np.float64),
        np.asarray(noise_test, dtype=np.float64),
    )


def _fixed_coordinates(order: np.ndarray) -> np.ndarray:
    """Coordinates outside Omega_13. The same set is scored at every fraction."""
    observed = np.asarray(order[:13], dtype=np.int64)
    return np.setdiff1d(np.arange(base.WINDOW_LEN, dtype=np.int64), observed)


def verify_nested_masks(order: np.ndarray) -> None:
    """Omega_4, Omega_6, and Omega_13 are prefixes of one permutation."""
    positions = np.asarray(order, dtype=np.int64)
    if int(positions.size) != base.WINDOW_LEN:
        raise AssertionError("The permutation does not cover the window.")
    if set(int(value) for value in positions.tolist()) != set(range(base.WINDOW_LEN)):
        raise AssertionError("The permutation is not a permutation of 64 positions.")
    omega4 = positions[:4]
    omega6 = positions[:6]
    omega13 = positions[:13]
    if not set(int(v) for v in omega4.tolist()).issubset(set(int(v) for v in omega6.tolist())):
        raise AssertionError("Omega_4 is not nested in Omega_6.")
    if not set(int(v) for v in omega6.tolist()).issubset(set(int(v) for v in omega13.tolist())):
        raise AssertionError("Omega_6 is not nested in Omega_13.")
    if list(omega4.tolist()) != list(omega6[:4].tolist()):
        raise AssertionError("Omega_4 is not the first four permutation entries.")
    if list(omega6.tolist()) != list(omega13[:6].tolist()):
        raise AssertionError("Omega_6 is not the first six permutation entries.")


def verify_shared_noise_prefix(
    y_rows: np.ndarray,
    noise_rows: np.ndarray,
    order: np.ndarray,
) -> None:
    """Adding observations must keep the earlier noisy measurements unchanged."""
    order64 = np.asarray(order, dtype=np.int64)
    small = _slice_measurements(y_rows, noise_rows, order64[:4])
    middle = _slice_measurements(y_rows, noise_rows, order64[:6])
    large = _slice_measurements(y_rows, noise_rows, order64[:13])
    np.testing.assert_array_equal(middle[:, :4], small)
    np.testing.assert_array_equal(large[:, :6], middle)
    np.testing.assert_array_equal(large[:, :4], small)


def _choose_positive_lambda(records: Sequence[Tuple[float, float]]) -> float:
    """Minimum unobserved validation RMSE among strictly positive lambdas.

    Lambda 0 stays on the sensitivity curve and is model B, not model A.
    Ties keep the earlier, smaller grid value.
    """
    positive = [(float(lam), float(score)) for lam, score in records if float(lam) > 0.0]
    if not positive:
        raise AssertionError("The lambda grid has no positive value.")
    best_lam = positive[0][0]
    best_score = positive[0][1]
    for lam, score in positive[1:]:
        if score < best_score:
            best_score = score
            best_lam = lam
    return float(best_lam)


def _assert_conditioned_call(
    z0: np.ndarray,
    z0_from_prior: np.ndarray,
    lam: float,
    positive: bool,
) -> None:
    """Model A and model B both start at h(u). Only the penalty differs."""
    np.testing.assert_array_equal(z0, z0_from_prior)
    if positive and not float(lam) > 0.0:
        raise AssertionError("Full CLP received a non-positive penalty.")
    if (not positive) and float(lam) != 0.0:
        raise AssertionError("Init-only CLP received a nonzero penalty.")


def _assert_measurement_only_call(z0: np.ndarray, lam: float, latent_dim: int) -> None:
    """Measurement-only recovery must not receive h(u) or a prior penalty."""
    if float(lam) != 0.0:
        raise AssertionError("Measurement-only CSGM received a nonzero penalty.")
    if int(z0.shape[1]) != int(latent_dim):
        raise AssertionError("Measurement-only latent width is not the decoder width.")
    if not bool(np.all(np.asarray(z0) == 0.0)):
        raise AssertionError("Measurement-only CSGM received a nonzero latent start.")


def _weight_fingerprint(ae: csgm.TinyAE) -> str:
    """Identify one trained decoder without using it to choose a model."""
    digest = hashlib.sha256()
    for name, tensor in ae.state_dict().items():
        digest.update(name.encode("ascii"))
        array = np.ascontiguousarray(tensor.detach().cpu().numpy())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _score_trio(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    observed: np.ndarray,
    fixed: np.ndarray,
) -> Dict[str, float]:
    """Full window, current unobserved set, and the Omega_13 complement."""
    if y_pred.shape != y_true.shape:
        raise AssertionError("Prediction and target shapes differ.")
    if not bool(np.isfinite(y_pred).all()):
        raise AssertionError("A prediction contains a non-finite value.")
    full = base.rmse(y_true, y_pred)
    held = base.rmse(y_true, y_pred, observed)
    fixed_score = base.rmse(y_true[:, fixed], y_pred[:, fixed])
    return {
        "rmse_full": float(full),
        "rmse_unobserved": float(held),
        "rmse_fixed": float(fixed_score),
    }


def _load_recorded_indices(train: object) -> Dict[int, np.ndarray]:
    """Use the training subsets already stored by the factorial study."""
    if not os.path.isfile(INDEX_CSV):
        raise FileNotFoundError("Missing recorded training indices: {}".format(INDEX_CSV))
    grouped: Dict[int, List[int]] = {24: [], 56: [], 88: []}
    wells: Dict[int, List[str]] = {24: [], 56: [], 88: []}
    with open(INDEX_CSV, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected = {
            "n_train",
            "pool_index",
            "well",
            "row_start",
            "depth_start",
            "depth_end",
        }
        if set(reader.fieldnames or []) != expected:
            raise AssertionError("selected_indices.csv does not have the expected columns.")
        for row in reader:
            size = int(row["n_train"])
            if size not in grouped:
                raise AssertionError("Unexpected training budget {}.".format(size))
            pool_index = int(row["pool_index"])
            well = str(train.well_of_window[pool_index])
            if well != str(row["well"]):
                raise AssertionError("Recorded well does not match the rebuilt window.")
            if int(train.row_start_in_well[pool_index]) != int(row["row_start"]):
                raise AssertionError("Recorded row start does not match the rebuilt window.")
            depth_start = float(train.depth_range[pool_index, 0])
            depth_end = float(train.depth_range[pool_index, 1])
            if abs(depth_start - float(row["depth_start"])) > 1e-4:
                raise AssertionError("Recorded depth start does not match the rebuilt window.")
            if abs(depth_end - float(row["depth_end"])) > 1e-4:
                raise AssertionError("Recorded depth end does not match the rebuilt window.")
            grouped[size].append(pool_index)
            wells[size].append(well)
    out: Dict[int, np.ndarray] = {}
    for size, values in grouped.items():
        if len(values) != int(size):
            raise AssertionError("Budget {} has {} indices.".format(size, len(values)))
        if len(set(values)) != int(size):
            raise AssertionError("Budget {} has a repeated index.".format(size))
        counts = {name: wells[size].count(name) for name in ("F02-1", "F03-2")}
        if counts != EXPECTED_WELL_COUNTS[size]:
            raise AssertionError("Budget {} well counts are {}.".format(size, counts))
        out[size] = np.asarray(values, dtype=np.int64)
    base.verify_nested_subsets(out[24].tolist(), out[56].tolist())
    base.verify_nested_subsets(out[56].tolist(), out[88].tolist())
    base.verify_nested_subsets(out[24].tolist(), out[88].tolist())
    return out


def _select_lambda_on_validation(
    ae: csgm.TinyAE,
    z0_val: np.ndarray,
    y_val: np.ndarray,
    b_val: np.ndarray,
    operator: np.ndarray,
    observed: np.ndarray,
    fixed: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    cfg: object,
    seed: int,
    n_val: int,
) -> Tuple[float, List[Dict[str, float]]]:
    """Score the lambda grid on F06-1 only. F03-4 is not an argument."""
    if int(y_val.shape[0]) != int(n_val):
        raise AssertionError("Lambda selection did not receive the validation well.")
    if int(n_val) == 97:
        raise AssertionError("Lambda selection received the test-sized array.")
    records: List[Dict[str, float]] = []
    pairs: List[Tuple[float, float]] = []
    for lam in LAMBDA_GRID:
        _assert_conditioned_call(z0_val, z0_val, float(lam), positive=float(lam) > 0.0)
        y_hat = csgm.csgm_recover_with_prior(
            ae=ae,
            M=operator,
            B=b_val,
            z0_np=z0_val,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device="cpu",
            seed=int(seed),
        )
        scores = _score_trio(y_val, y_hat, observed, fixed)
        # The choice uses only the current unobserved validation RMSE.
        pairs.append((float(lam), float(scores["rmse_unobserved"])))
        records.append(
            {
                "lambda": float(lam),
                "val_rmse_unobserved": float(scores["rmse_unobserved"]),
                "val_rmse_full": float(scores["rmse_full"]),
                "val_rmse_fixed": float(scores["rmse_fixed"]),
            }
        )
    chosen = _choose_positive_lambda(pairs)
    positive_scores = [score for lam, score in pairs if lam > 0.0]
    zero_score = [score for lam, score in pairs if lam == 0.0]
    if len(zero_score) != 1:
        raise AssertionError("The sensitivity curve does not contain lambda 0.")
    best_positive = min(positive_scores)
    if not any(abs(float(row["lambda"]) - chosen) < 1e-15 and abs(float(row["val_rmse_unobserved"]) - best_positive) < 1e-12 for row in records):
        raise AssertionError("Selected lambda is not the positive-grid validation minimum.")
    return chosen, records


def _run_synthetic_checks() -> None:
    """Check nesting, shared noise, and the positive-lambda rule before training."""
    rng = np.random.default_rng(0)
    order = rng.permutation(8)
    # The production check uses 64. This toy check only covers the prefix rule.
    padded = np.concatenate([order, np.arange(8, 64)])
    # Force a real permutation for the 64-length checker.
    padded = np.random.default_rng(1).permutation(64).astype(np.int64)
    verify_nested_masks(padded)
    y_rows = np.arange(12, dtype=np.float64).reshape(2, 6)
    # Local prefix check on a short window, independent of WINDOW_LEN.
    noise = np.array(
        [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [0.0, -0.1, 0.2, -0.2, 0.3, -0.3]],
        dtype=np.float64,
    )
    short_order = np.array([4, 1, 5, 0, 2, 3], dtype=np.int64)
    b4 = y_rows[:, short_order[:4]] + noise[:, short_order[:4]]
    b6 = y_rows[:, short_order[:6]] + noise[:, short_order[:6]]
    np.testing.assert_array_equal(b6[:, :4], b4)
    records = [
        (0.0, 0.01),
        (1e-6, 0.20),
        (1e-4, 0.05),
        (1e-1, 0.05),
    ]
    chosen = _choose_positive_lambda(records)
    if chosen != 1e-4:
        raise AssertionError("Positive lambda selection did not keep the smaller tie.")
    if chosen == 0.0:
        raise AssertionError("Model A accepted lambda 0.")
    base.verify_nested_subsets([0, 2], [0, 1, 2, 5])
    y_true = np.array([[0.0, 1.0, 2.0, 3.0]], dtype=np.float64)
    y_pred = np.array([[0.0, 9.0, 2.0, 3.0]], dtype=np.float64)
    held = base.rmse(y_true, y_pred, np.array([1], dtype=np.int64))
    if held > 1e-12:
        raise AssertionError("Unobserved RMSE kept a measured coordinate.")
    _assert_measurement_only_call(np.zeros((2, 16), dtype=np.float64), 0.0, 16)
    print("Synthetic consistency checks passed.")


def _save_table(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write one CSV with a fixed column order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _lookup(
    rows: Sequence[Mapping[str, object]],
    seed: int,
    n_train: int,
    m_obs: int,
    method: str,
    metric: str,
) -> float:
    """Read one descriptive RMSE. This does not average test windows."""
    matches = [
        row
        for row in rows
        if int(row["seed"]) == int(seed)
        and int(row["n_train"]) == int(n_train)
        and int(row["m_obs"]) == int(m_obs)
        and str(row["method"]) == method
    ]
    if len(matches) != 1:
        raise AssertionError("Expected one row for {} {} {}.".format(n_train, m_obs, method))
    return float(matches[0][metric])


def _plot_lines(
    rows: Sequence[Mapping[str, object]],
    x_key: str,
    panel_key: str,
    panel_values: Sequence[object],
    path: str,
    xlabel: str,
    title: str,
) -> None:
    """One panel per budget or observation count, with one line per method."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, len(panel_values), figsize=(4.4 * len(panel_values), 4.2), sharey=True)
    if len(panel_values) == 1:
        axes = [axes]
    for axis, panel in zip(axes, panel_values):
        for method in METHODS:
            subset = [
                row
                for row in rows
                if str(row["method"]) == method and row[panel_key] == panel
            ]
            subset = sorted(subset, key=lambda row: float(row[x_key]))
            axis.plot(
                [float(row[x_key]) for row in subset],
                [float(row["rmse_full"]) for row in subset],
                marker="o",
                label=method,
            )
        axis.set_title("{}={}".format(panel_key, panel))
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Full-window RMSE")
        axis.grid(True, alpha=0.3)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_gains(summary_rows: Sequence[Mapping[str, object]], sizes: Sequence[int], path: str) -> None:
    """CLP minus each comparator. A negative value means CLP has lower RMSE."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = (
        ("clp_minus_ae", "AE"),
        ("clp_minus_interpolation", "interpolation"),
        ("clp_minus_measurement_only", "measurement-only"),
    )
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.4 * len(sizes), 4.2), sharey=True)
    if len(sizes) == 1:
        axes = [axes]
    for axis, size in zip(axes, sizes):
        subset = [row for row in summary_rows if int(row["n_train"]) == int(size)]
        subset = sorted(subset, key=lambda row: int(row["m_obs"]))
        x_pos = np.arange(len(subset), dtype=np.float64)
        width = 0.25
        for offset, (key, label) in enumerate(keys):
            axis.bar(
                x_pos + (offset - 1) * width,
                [float(row[key]) for row in subset],
                width=width,
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x_pos)
        axis.set_xticklabels([str(int(row["m_obs"])) for row in subset])
        axis.set_xlabel("Observed positions")
        axis.set_title("{} training windows".format(int(size)))
        axis.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("RMSE_CLP - RMSE_comparator")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Negative favors CLP. One pooled test score per cell, not 97 replicates.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_lambda(lambda_rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Validation curve. The test well is not on this figure."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cells = sorted(
        {
            (int(row["n_train"]), int(row["m_obs"]))
            for row in lambda_rows
        }
    )
    fig, axes = plt.subplots(1, len(cells), figsize=(3.6 * max(len(cells), 1), 4.0), sharey=True)
    if len(cells) == 1:
        axes = [axes]
    for axis, (size, m_obs) in zip(axes, cells):
        subset = [
            row
            for row in lambda_rows
            if int(row["n_train"]) == size and int(row["m_obs"]) == m_obs
        ]
        subset = sorted(subset, key=lambda row: float(row["lambda"]))
        axis.plot(
            [float(row["lambda"]) for row in subset],
            [float(row["val_rmse_unobserved"]) for row in subset],
            marker="o",
        )
        chosen = [row for row in subset if int(row["selected_for_clp"]) == 1]
        if len(chosen) == 1:
            axis.scatter(
                [float(chosen[0]["lambda"])],
                [float(chosen[0]["val_rmse_unobserved"])],
                s=40,
                zorder=3,
            )
        axis.set_xscale("symlog", linthresh=1e-7)
        axis.set_title("n={} m={}".format(size, m_obs))
        axis.set_xlabel("lambda")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Validation unobserved RMSE")
    fig.suptitle("F06-1 only. Marker: positive lambda used by full CLP.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_diagnostics(diag_rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Decoder reconstruction and prior-only RMSE against the training budget."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0), sharex=True)
    series = (
        ("decoder_train", "G(E(y)) train"),
        ("decoder_test", "G(E(y)) test"),
        ("prior_full", "prior-only full"),
        ("prior_fixed", "prior-only fixed complement"),
    )
    for name, label in series:
        subset = [row for row in diag_rows if str(row["name"]) == name]
        subset = sorted(subset, key=lambda row: int(row["n_train"]))
        target = axes[0] if name.startswith("decoder") else axes[1]
        target.plot(
            [int(row["n_train"]) for row in subset],
            [float(row["rmse"]) for row in subset],
            marker="o",
            label=label,
        )
    axes[0].set_title("Decoder reconstruction")
    axes[1].set_title("Conditional prior, no assimilation")
    for axis in axes:
        axis.set_xlabel("Training windows")
        axis.set_ylabel("RMSE")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_profiles(
    predictions: Mapping[Tuple[int, int, int, str], np.ndarray],
    y_test: np.ndarray,
    windows: Sequence[Mapping[str, object]],
    settings: Sequence[Tuple[int, int]],
    seed: int,
    path: str,
) -> None:
    """Profiles chosen by a fixed rule, including the largest CLP error."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n_rows = len(windows)
    n_cols = len(settings)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 2.8 * n_rows), sharex=True, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[axis] for axis in axes])
    grid = np.arange(base.WINDOW_LEN)
    for row_id, window in enumerate(windows):
        index = int(window["test_index"])
        for col_id, (size, m_obs) in enumerate(settings):
            axis = axes[row_id, col_id]
            axis.plot(grid, y_test[index], color="black", linewidth=1.4, label="truth")
            for method in (METHOD_CLP, METHOD_INIT, METHOD_MEAS, METHOD_AE, METHOD_INTERP):
                curve = predictions[(int(seed), int(size), int(m_obs), method)][index]
                axis.plot(grid, curve, linewidth=1.0, label=method)
            if row_id == 0:
                axis.set_title("n={} m={}".format(int(size), int(m_obs)))
            if col_id == 0:
                axis.set_ylabel(str(window["criterion"]))
            axis.grid(True, alpha=0.25)
    axes[0, -1].legend(fontsize=6, loc="best")
    fig.suptitle("Predefined windows. The last row is the largest CLP error at n=24, m=4.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_readme(path: str, mode_name: str, sizes: Sequence[int], seeds: Sequence[int]) -> None:
    """Record the protocol and what this process did not run."""
    pending = "not run" if mode_name == "pilot" else "this directory is the full run"
    lines = [
        "# Nested masks and conditional-prior ablation",
        "",
        "Exploratory study. These numbers are not manuscript results.",
        "",
        "Reference commit for the previous factorial: {}.".format(REFERENCE_COMMIT),
        "Previous outputs were read and were not overwritten.",
        "",
        "This run: mode={}, seeds={}, budgets={}, observed positions={}.".format(
            mode_name,
            list(seeds),
            list(sizes),
            list(M_OBS),
        ),
        "Full grid (seeds 7, 23, 41 and budgets 24, 56, 88): {}.".format(pending),
        "",
        "Training subsets are the recorded nested indices, not a new draw.",
        "For each seed, one permutation defines Omega_4, Omega_6, and Omega_13.",
        "One noise field per split is sliced onto those prefixes.",
        "The same mask and the same measurements are reused across budgets.",
        "",
        "Models inside one seed and budget share one decoder:",
        "- clp_csgm_ridge: start at h(u), positive lambda chosen on F06-1.",
        "- clp_init_only: same start and same restart seed, lambda 0.",
        "- measurement_only_csgm: zero latent start, lambda 0, seed+9200.",
        "- prior_only: G(h(u)), no assimilation.",
        "ae_direct and interpolation see the same b as the latent models.",
        "",
        "Lambda selection uses unobserved RMSE on F06-1. F03-4 is not used.",
        "Lambda 0 can win the validation curve and still is not assigned to full CLP.",
        "F06-1 has eight overlapping windows from one well. They are not independent replicates.",
        "The 97 test windows are one pooled score. They are not 97 independent trials.",
        "",
        "clp_minus_* columns are RMSE_CLP minus RMSE_comparator.",
        "Negative favors CLP. This sign is the opposite of the previous study's Delta.",
        "",
        "A difference among full CLP, init-only, and measurement-only is not, by itself,",
        "an isolated causal estimate. Latent recovery is non-convex, and measurement-only",
        "uses the original seed offset.",
        "",
        "G(E(y)) is diagnostic. It does not select lambda, a budget, or a method.",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested-mask observation scarcity and conditional-prior ablation."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Seed 7 and budgets 24 and 88.")
    mode.add_argument("--full", action="store_true", help="Seeds 7, 23, 41 and budgets 24, 56, 88.")
    return parser.parse_args()


def main() -> None:
    """Run the nested-mask grid and write tables, checks, and figures."""
    if os.path.abspath(OUT_DIR) == os.path.abspath(base.OUT_DIR):
        raise RuntimeError("Refusing to write into the previous study directory.")
    args = _parse_args()
    _run_synthetic_checks()
    seeds = FULL_SEEDS if args.full else PILOT_SEEDS
    sizes = FULL_SIZES if args.full else PILOT_SIZES
    mode_name = "full" if args.full else "pilot"
    cfg = base._paper_config()
    cfg.csgm_lambda_grid = list(LAMBDA_GRID)
    train, val, test = base._load_role_windows()
    base.verify_training_and_test(
        train.well_of_window.tolist(),
        val.well_of_window.tolist(),
        test.well_of_window.tolist(),
    )
    if int(val.y.shape[0]) != 8 or int(test.y.shape[0]) != 97:
        raise AssertionError("Validation or test window count is not the published split.")
    indices = _load_recorded_indices(train)
    for size in FULL_SIZES:
        chosen_wells = train.well_of_window[indices[size]].tolist()
        base.verify_training_and_test(chosen_wells, [base.VAL_WELL], [base.TEST_WELL])
    val_dependence = base._window_dependence(val)
    print(
        "Validation windows:",
        val_dependence["n_windows"],
        "overlapping pairs:",
        val_dependence["overlapping_pairs"],
        "F06-1 windows are not independent evidence.",
        flush=True,
    )

    y_pool = np.asarray(train.y, dtype=np.float64)
    y_val = np.asarray(val.y, dtype=np.float64)
    y_test = np.asarray(test.y, dtype=np.float64)
    x_test = np.asarray(test.x, dtype=np.float64)
    x_val = np.asarray(val.x, dtype=np.float64)

    result_rows: List[Dict[str, object]] = []
    lambda_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    mask_rows: List[Dict[str, object]] = []
    predictions: Dict[Tuple[int, int, str], np.ndarray] = {}
    passed: List[str] = [
        "synthetic nesting, shared-noise prefix, and positive-lambda rule",
        "recorded 24 subset of 56 subset of 88, with the previous well quotas",
        "train, validation, and test wells are disjoint",
    ]

    for seed in seeds:
        order, noise_pool, noise_val, noise_test = _draw_nested_problem(
            y_pool, y_val, y_test, int(seed)
        )
        redraw, redraw_pool, redraw_val, redraw_test = _draw_nested_problem(
            y_pool, y_val, y_test, int(seed)
        )
        np.testing.assert_array_equal(order, redraw)
        np.testing.assert_array_equal(noise_pool, redraw_pool)
        np.testing.assert_array_equal(noise_val, redraw_val)
        np.testing.assert_array_equal(noise_test, redraw_test)
        verify_nested_masks(order)
        verify_shared_noise_prefix(y_pool, noise_pool, order)
        verify_shared_noise_prefix(y_val, noise_val, order)
        verify_shared_noise_prefix(y_test, noise_test, order)
        fixed = _fixed_coordinates(order)
        if int(fixed.size) != base.WINDOW_LEN - 13:
            raise AssertionError("The fixed complement does not have 51 coordinates.")
        for rank, position in enumerate(order.tolist()):
            mask_rows.append(
                {
                    "seed": int(seed),
                    "position_rank": int(rank),
                    "window_index": int(position),
                    "in_omega4": int(rank < 4),
                    "in_omega6": int(rank < 6),
                    "in_omega13": int(rank < 13),
                }
            )
        mask_fingerprint = _array_hash((order, noise_pool, noise_val, noise_test))
        first_bundle: Dict[int, Dict[str, np.ndarray]] = {}
        interp_ref: Dict[int, np.ndarray] = {}
        # Same pool window and same mask must keep the same training measurement.
        train_measurement_ref: Dict[Tuple[int, int], np.ndarray] = {}

        for size in sizes:
            print("seed", int(seed), "n_train", int(size), flush=True)
            idx = indices[int(size)]
            x_train = np.asarray(train.x[idx], dtype=np.float64)
            y_train = np.asarray(train.y[idx], dtype=np.float64)
            ae, prior, _y_scaler, y_mean, y_scale = base._fit_clp(
                x_train, y_train, int(seed), cfg
            )
            decoder_id = _weight_fingerprint(ae)
            z0_val = prior.predict(x_val)
            z0_test = prior.predict(x_test)
            y_prior = csgm.decode_latent_prior(ae, z0_test, y_mean, y_scale, device="cpu")
            diag_rows.append(
                {
                    "seed": int(seed),
                    "n_train": int(size),
                    "name": "decoder_train",
                    "rmse": base._decoder_reconstruction_rmse(ae, y_train, y_mean, y_scale),
                }
            )
            diag_rows.append(
                {
                    "seed": int(seed),
                    "n_train": int(size),
                    "name": "decoder_test",
                    "rmse": base._decoder_reconstruction_rmse(ae, y_test, y_mean, y_scale),
                }
            )
            prior_full = base.rmse(y_test, y_prior)
            prior_fixed = base.rmse(y_test[:, fixed], y_prior[:, fixed])
            diag_rows.append(
                {"seed": int(seed), "n_train": int(size), "name": "prior_full", "rmse": prior_full}
            )
            diag_rows.append(
                {"seed": int(seed), "n_train": int(size), "name": "prior_fixed", "rmse": prior_fixed}
            )
            prior_scores_ref: Dict[str, float] | None = None

            for m_obs in M_OBS:
                print("  m_obs", int(m_obs), flush=True)
                positions = order[: int(m_obs)]
                operator = _operator_from_positions(positions)
                observed = np.asarray(positions, dtype=np.int64)
                b_train = _slice_measurements(y_train, noise_pool[idx], positions)
                for local_row, pool_index in enumerate(idx.tolist()):
                    measure_key = (int(m_obs), int(pool_index))
                    current_measure = np.asarray(b_train[local_row], dtype=np.float64)
                    if measure_key in train_measurement_ref:
                        np.testing.assert_array_equal(
                            current_measure,
                            train_measurement_ref[measure_key],
                        )
                    else:
                        train_measurement_ref[measure_key] = current_measure.copy()
                b_val = _slice_measurements(y_val, noise_val, positions)
                b_test = _slice_measurements(y_test, noise_test, positions)
                # Prefix rows of the larger measurement must reproduce the smaller one.
                if int(m_obs) == 13:
                    b4 = _slice_measurements(y_test, noise_test, order[:4])
                    b6 = _slice_measurements(y_test, noise_test, order[:6])
                    np.testing.assert_array_equal(b_test[:, :4], b4)
                    np.testing.assert_array_equal(b_test[:, :6], b6)
                bundle = {"M": operator, "B_val": b_val, "B_test": b_test}
                if int(size) == int(sizes[0]):
                    first_bundle[int(m_obs)] = bundle
                else:
                    for name, array in bundle.items():
                        np.testing.assert_array_equal(array, first_bundle[int(m_obs)][name])
                ub_scaler = dub.fit_scaler_ub(dub.concat_ub(x_train, b_train))
                base.verify_scaler_on_train(
                    ub_scaler,
                    dub.concat_ub(x_train, b_train),
                    "ae_input_scaler",
                )
                _ = ub_scaler.transform(dub.concat_ub(x_val, b_val))
                base.verify_scaler_on_train(
                    ub_scaler,
                    dub.concat_ub(x_train, b_train),
                    "ae_input_scaler_after_val_transform",
                )

                lam, curve = _select_lambda_on_validation(
                    ae,
                    z0_val,
                    y_val,
                    b_val,
                    operator,
                    observed,
                    fixed,
                    y_mean,
                    y_scale,
                    cfg,
                    int(seed),
                    int(y_val.shape[0]),
                )
                zero_rows = [row for row in curve if float(row["lambda"]) == 0.0]
                if len(zero_rows) != 1:
                    raise AssertionError("The validation curve is missing lambda 0.")
                zero_score = float(zero_rows[0]["val_rmse_unobserved"])
                positive_best = min(
                    float(row["val_rmse_unobserved"]) for row in curve if float(row["lambda"]) > 0.0
                )
                at_lower = int(abs(lam - min(value for value in LAMBDA_GRID if value > 0.0)) < 1e-15)
                at_upper = int(abs(lam - max(LAMBDA_GRID)) < 1e-15)
                zero_beats = int(zero_score < positive_best)
                for row in curve:
                    lambda_rows.append(
                        {
                            "seed": int(seed),
                            "n_train": int(size),
                            "m_obs": int(m_obs),
                            "lambda": float(row["lambda"]),
                            "init": "conditioned_z0",
                            "val_rmse_unobserved": float(row["val_rmse_unobserved"]),
                            "val_rmse_full": float(row["val_rmse_full"]),
                            "val_rmse_fixed": float(row["val_rmse_fixed"]),
                            "selected_for_clp": int(abs(float(row["lambda"]) - lam) < 1e-15),
                            "at_lower_boundary": at_lower,
                            "at_upper_boundary": at_upper,
                            "zero_beats_all_positive": zero_beats,
                        }
                    )

                _assert_conditioned_call(z0_test, z0_test, lam, positive=True)
                y_clp = csgm.csgm_recover_with_prior(
                    ae=ae,
                    M=operator,
                    B=b_test,
                    z0_np=z0_test,
                    y_mean=y_mean,
                    y_scale=y_scale,
                    lam=float(lam),
                    n_iters=int(cfg.csgm_iters),
                    opt_lr=float(cfg.csgm_opt_lr),
                    n_restarts=int(cfg.csgm_restarts),
                    device="cpu",
                    seed=int(seed),
                )
                _assert_conditioned_call(z0_test, prior.predict(x_test), 0.0, positive=False)
                y_init = csgm.csgm_recover_with_prior(
                    ae=ae,
                    M=operator,
                    B=b_test,
                    z0_np=z0_test,
                    y_mean=y_mean,
                    y_scale=y_scale,
                    lam=0.0,
                    n_iters=int(cfg.csgm_iters),
                    opt_lr=float(cfg.csgm_opt_lr),
                    n_restarts=int(cfg.csgm_restarts),
                    device="cpu",
                    seed=int(seed),
                )
                z0_zero = np.zeros_like(z0_test)
                _assert_measurement_only_call(z0_zero, 0.0, int(cfg.csgm_latent_dim))
                # Same iteration budget. The start is independent of h(u).
                # The seed offset matches the original measurement-only test call,
                # so this comparison also includes a different restart draw.
                y_meas = csgm.csgm_recover_with_prior(
                    ae=ae,
                    M=operator,
                    B=b_test,
                    z0_np=z0_zero,
                    y_mean=y_mean,
                    y_scale=y_scale,
                    lam=0.0,
                    n_iters=int(cfg.csgm_iters),
                    opt_lr=float(cfg.csgm_opt_lr),
                    n_restarts=int(cfg.csgm_restarts),
                    device="cpu",
                    seed=int(seed) + MEASUREMENT_ONLY_SEED_OFFSET,
                )
                _unused_val_ae, y_ae = dub.fit_predict_ae_regression_ub(
                    cfg,
                    int(seed),
                    dub.DirectUBTrainConfig(),
                    dub.concat_ub(x_train, b_train),
                    y_train,
                    dub.concat_ub(x_val, b_val),
                    y_val,
                    dub.concat_ub(x_test, b_test),
                    ub_scaler,
                )
                y_interp = base._interpolate(b_test, observed, base.WINDOW_LEN)
                if int(m_obs) not in interp_ref:
                    interp_ref[int(m_obs)] = np.array(y_interp, copy=True)
                elif int(size) != int(sizes[0]):
                    np.testing.assert_array_equal(y_interp, interp_ref[int(m_obs)])
                if _weight_fingerprint(ae) != decoder_id:
                    raise AssertionError("The shared decoder changed across fractions or methods.")
                np.testing.assert_array_equal(y_test, test.y)

                packed = {
                    METHOD_CLP: (y_clp, lam, "conditioned_z0_positive_lambda"),
                    METHOD_INIT: (y_init, 0.0, "conditioned_z0_lambda_zero"),
                    METHOD_MEAS: (y_meas, 0.0, "zero_latent_lambda_zero_seed_offset"),
                    METHOD_PRIOR: (y_prior, "", "decode_h_u_no_assimilation"),
                    METHOD_AE: (y_ae, "", "direct_ub"),
                    METHOD_INTERP: (y_interp, "", "linear_np_interp"),
                }
                cell_scores: Dict[str, Dict[str, float]] = {}
                for method, (prediction, lam_value, policy) in packed.items():
                    scores = _score_trio(y_test, prediction, observed, fixed)
                    cell_scores[method] = scores
                    predictions[(int(seed), int(size), int(m_obs), method)] = prediction
                    print(
                        "   ",
                        method,
                        "full",
                        "{:.6f}".format(scores["rmse_full"]),
                        "unobserved",
                        "{:.6f}".format(scores["rmse_unobserved"]),
                        "fixed",
                        "{:.6f}".format(scores["rmse_fixed"]),
                        flush=True,
                    )
                    result_rows.append(
                        {
                            "mode": mode_name,
                            "seed": int(seed),
                            "n_train": int(size),
                            "m_obs": int(m_obs),
                            "n_val": int(y_val.shape[0]),
                            "n_test": int(y_test.shape[0]),
                            "method": method,
                            "rmse_full": scores["rmse_full"],
                            "rmse_unobserved": scores["rmse_unobserved"],
                            "rmse_fixed": scores["rmse_fixed"],
                            "lambda": lam_value,
                            "init_policy": policy,
                            "lambda_at_lower_boundary": at_lower if method == METHOD_CLP else "",
                            "lambda_at_upper_boundary": at_upper if method == METHOD_CLP else "",
                            "zero_beats_positive_on_val": zero_beats if method == METHOD_CLP else "",
                            "mask_fingerprint": mask_fingerprint,
                        }
                    )
                if int(m_obs) == 13:
                    if abs(cell_scores[METHOD_CLP]["rmse_unobserved"] - cell_scores[METHOD_CLP]["rmse_fixed"]) > 1e-12:
                        raise AssertionError("At 13 observations, unobserved RMSE is not the fixed complement.")
                if prior_scores_ref is None:
                    prior_scores_ref = cell_scores[METHOD_PRIOR]
                else:
                    for metric in ("rmse_full", "rmse_fixed"):
                        if abs(cell_scores[METHOD_PRIOR][metric] - prior_scores_ref[metric]) > 1e-12:
                            raise AssertionError("Prior-only changed across observation fractions.")
            if prior_scores_ref is None:
                raise AssertionError("Prior-only was not scored.")

        passed.append("seed {} mask redraw, nesting, and shared noise prefix".format(int(seed)))

    passed.extend(
        [
            "operators and measurements are identical across training budgets",
            "interpolation is identical across budgets at a fixed mask",
            "prior-only full and fixed RMSE are identical across fractions",
            "decoder weights are unchanged across fractions and ablations",
            "scalers are fit on the training subset only",
            "positive lambda is the F06-1 unobserved minimum; lambda 0 is not assigned to full CLP",
            "measurement-only call uses a zero code and lambda 0",
            "unobserved RMSE drops measured coordinates; the fixed score uses the Omega_13 complement",
            "one test array is used for every comparison",
            "G(E(y)) is stored and is not an argument of lambda selection",
        ]
    )

    summary_rows: List[Dict[str, object]] = []
    for seed in seeds:
        for size in sizes:
            for m_obs in M_OBS:
                clp_full = _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_CLP, "rmse_full")
                ae_full = _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_AE, "rmse_full")
                interp_full = _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_INTERP, "rmse_full")
                meas_full = _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_MEAS, "rmse_full")
                init_full = _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_INIT, "rmse_full")
                summary_rows.append(
                    {
                        "mode": mode_name,
                        "seed": int(seed),
                        "n_train": int(size),
                        "m_obs": int(m_obs),
                        "rmse_clp": clp_full,
                        "rmse_init_only": init_full,
                        "rmse_measurement_only": meas_full,
                        "rmse_prior": _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_PRIOR, "rmse_full"),
                        "rmse_ae": ae_full,
                        "rmse_interpolation": interp_full,
                        "clp_minus_ae": clp_full - ae_full,
                        "clp_minus_interpolation": clp_full - interp_full,
                        "clp_minus_measurement_only": clp_full - meas_full,
                        "clp_minus_init_only": clp_full - init_full,
                        "rmse_clp_fixed": _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_CLP, "rmse_fixed"),
                        "rmse_ae_fixed": _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_AE, "rmse_fixed"),
                        "rmse_interpolation_fixed": _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_INTERP, "rmse_fixed"),
                        "rmse_measurement_only_fixed": _lookup(result_rows, int(seed), int(size), int(m_obs), METHOD_MEAS, "rmse_fixed"),
                    }
                )

    # Display rule fixed before looking at the curves: depth order, plus the
    # worst full-window CLP error in the scarcest pilot cell.
    per_window = np.sqrt(np.mean((predictions[(int(seeds[0]), 24, 4, METHOD_CLP)] - y_test) ** 2, axis=1))
    worst = int(np.argmax(per_window))
    n_test = int(y_test.shape[0])
    profile_windows = [
        {"criterion": "depth_order_first", "test_index": 0},
        {"criterion": "depth_order_middle", "test_index": n_test // 2},
        {"criterion": "depth_order_last", "test_index": n_test - 1},
        {"criterion": "max_clp_full_rmse_at_n24_m4", "test_index": worst},
    ]
    for window in profile_windows:
        index = int(window["test_index"])
        window["depth_start"] = float(test.depth_range[index, 0])
        window["depth_end"] = float(test.depth_range[index, 1])
        window["clp_rmse_n24_m4"] = float(per_window[index])

    table_dir = os.path.join(OUT_DIR, "tables")
    figure_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(OUT_DIR, exist_ok=True)
    _save_table(
        os.path.join(table_dir, "results_by_seed.csv"),
        result_rows,
        (
            "mode",
            "seed",
            "n_train",
            "m_obs",
            "n_val",
            "n_test",
            "method",
            "rmse_full",
            "rmse_unobserved",
            "rmse_fixed",
            "lambda",
            "init_policy",
            "lambda_at_lower_boundary",
            "lambda_at_upper_boundary",
            "zero_beats_positive_on_val",
            "mask_fingerprint",
        ),
    )
    _save_table(
        os.path.join(table_dir, "summary.csv"),
        summary_rows,
        (
            "mode",
            "seed",
            "n_train",
            "m_obs",
            "rmse_clp",
            "rmse_init_only",
            "rmse_measurement_only",
            "rmse_prior",
            "rmse_ae",
            "rmse_interpolation",
            "clp_minus_ae",
            "clp_minus_interpolation",
            "clp_minus_measurement_only",
            "clp_minus_init_only",
            "rmse_clp_fixed",
            "rmse_ae_fixed",
            "rmse_interpolation_fixed",
            "rmse_measurement_only_fixed",
        ),
    )
    _save_table(
        os.path.join(table_dir, "lambda_sensitivity.csv"),
        lambda_rows,
        (
            "seed",
            "n_train",
            "m_obs",
            "lambda",
            "init",
            "val_rmse_unobserved",
            "val_rmse_full",
            "val_rmse_fixed",
            "selected_for_clp",
            "at_lower_boundary",
            "at_upper_boundary",
            "zero_beats_all_positive",
        ),
    )
    _save_table(
        os.path.join(table_dir, "decoder_diagnostics.csv"),
        diag_rows,
        ("seed", "n_train", "name", "rmse"),
    )
    _save_table(
        os.path.join(table_dir, "masks.csv"),
        mask_rows,
        (
            "seed",
            "position_rank",
            "window_index",
            "in_omega4",
            "in_omega6",
            "in_omega13",
        ),
    )
    _plot_lines(
        result_rows,
        "m_obs",
        "n_train",
        list(sizes),
        os.path.join(figure_dir, "rmse_vs_observations.png"),
        "Observed positions",
        "Full-window RMSE against the nested observation count",
    )
    _plot_lines(
        result_rows,
        "n_train",
        "m_obs",
        list(M_OBS),
        os.path.join(figure_dir, "rmse_vs_training.png"),
        "Training windows",
        "Full-window RMSE against the training budget",
    )
    _plot_gains(summary_rows, sizes, os.path.join(figure_dir, "paired_gain_clp_minus_comparators.png"))
    ablation_rows = [
        row
        for row in result_rows
        if str(row["method"]) in (METHOD_CLP, METHOD_INIT, METHOD_MEAS, METHOD_PRIOR)
    ]
    _plot_lines(
        ablation_rows,
        "m_obs",
        "n_train",
        list(sizes),
        os.path.join(figure_dir, "conditioning_ablation.png"),
        "Observed positions",
        "Same decoder: full CLP, init-only, measurement-only, prior-only",
    )
    _plot_lambda(lambda_rows, os.path.join(figure_dir, "lambda_validation_curve.png"))
    _plot_diagnostics(diag_rows, os.path.join(figure_dir, "decoder_prior_diagnostic.png"))
    _plot_profiles(
        predictions,
        y_test,
        profile_windows,
        ((24, 4), (88, 13)),
        int(seeds[0]),
        os.path.join(figure_dir, "profiles_predefined.png"),
    )
    with open(INDEX_CSV, "rb") as handle:
        index_hash = _sha256_bytes(handle.read())
    config = {
        "mode": mode_name,
        "reference_commit": REFERENCE_COMMIT,
        "index_csv": INDEX_CSV,
        "index_sha256": index_hash,
        "seeds": list(int(seed) for seed in seeds),
        "sizes_trained": list(int(size) for size in sizes),
        "sizes_indexed": [24, 56, 88],
        "m_obs": list(M_OBS),
        "window_len": base.WINDOW_LEN,
        "window_step": base.WINDOW_STEP,
        "noise_std": base.NOISE_STD,
        "noise_protocol": (
            "One N(0, 0.01^2) field per split is drawn after the permutation. "
            "Each mask slices that field. Shared coordinates keep the same noise."
        ),
        "mask_protocol": "Omega_4, Omega_6, and Omega_13 are prefixes of one permutation per seed.",
        "train_wells": list(base.TRAIN_WELLS),
        "val_well": base.VAL_WELL,
        "test_well": base.TEST_WELL,
        "n_val": int(y_val.shape[0]),
        "n_test": int(y_test.shape[0]),
        "val_dependence": val_dependence,
        "val_interpretation": (
            "F06-1 has eight overlapping windows from one well. "
            "Lambda selection is not strong statistical evidence."
        ),
        "lambda_grid": list(LAMBDA_GRID),
        "lambda_rule": (
            "Unobserved validation RMSE. Full CLP uses the best strictly positive lambda. "
            "Lambda 0 with h(u) is clp_init_only. Lambda 0 with a zero code is measurement-only."
        ),
        "optimizer": {
            "iters": int(cfg.csgm_iters),
            "lr": float(cfg.csgm_opt_lr),
            "restarts": int(cfg.csgm_restarts),
            "measurement_only_seed_offset": MEASUREMENT_ONLY_SEED_OFFSET,
        },
        "decoder_sharing": "A, B, C, and D use one decoder per seed and training budget.",
        "nonconvex_caveat": (
            "Differences among A, B, and C are not isolated causal effects. "
            "Recovery is non-convex, and measurement-only uses a different restart seed."
        ),
        "paired_gain_sign": "clp_minus_comparator = RMSE_CLP - RMSE_comparator. Negative favors CLP.",
        "fixed_metric": "RMSE on the complement of Omega_13, common to all three fractions.",
        "profile_rule": (
            "Depth-order first, middle, and last test windows, plus the test window "
            "with the largest CLP full-window RMSE at 24 training windows and 4 observations."
        ),
        "profile_windows": profile_windows,
        "decoder_reconstruction": "G(E(y)) does not select lambda, a budget, or a method.",
        "test_windows_are_not_independent_replicates": True,
        "manuscript": "not updated",
        "previous_study_outputs": "not overwritten",
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(OUT_DIR, "checks.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "passed": passed,
                "n_result_rows": len(result_rows),
                "n_lambda_rows": len(lambda_rows),
                "mode": mode_name,
                "full_run_executed": bool(args.full),
                "index_sha256": index_hash,
                "profile_windows": profile_windows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    _write_readme(os.path.join(OUT_DIR, "README.md"), mode_name, sizes, seeds)
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
