#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score held-out plugs for the well 9-BRSA-1254-RJS study.

Reads the frozen split. Writes the corrected comparator tables under
corrected_protocol/ and does not replace the tables from the first run.
Does not edit article runners and does not build a dense permeability curve.
ASCII only.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import csgm_m2_module as csgm
import direct_ub_baselines as dub
from sir_cs_pipeline_optimized import Config

from common import (  # noqa: E402
    AE_EPOCHS,
    CSGM_ITERS,
    CSGM_RESTARTS,
    HIDDEN_DIM,
    LAMBDA_GRID,
    LATENT_DIM,
    OUT_DIR,
    SMOOTH_ALPHAS,
    SMOOTH_LINEAR_LIMIT,
    U_COLUMNS,
    WINDOW_LEN,
    WINDOW_STEP,
    mask_seed,
    observation_count,
)

CORRECTED_DIR = OUT_DIR / "corrected_protocol"


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error. One entry per reserved plug."""
    truth = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    if truth.shape != pred.shape or truth.size < 1:
        raise ValueError("RMSE inputs must be non-empty and aligned.")
    if not np.isfinite(truth).all() or not np.isfinite(pred).all():
        raise ValueError("RMSE inputs must be finite.")
    gap = truth - pred
    return float(np.sqrt(np.mean(gap * gap)))


def _interp_depth(
    depth_obs: np.ndarray,
    y_obs: np.ndarray,
    depth_query: np.ndarray,
) -> np.ndarray:
    """Linear interpolation in meters. Ends keep the nearest measured value."""
    order = np.argsort(depth_obs)
    return np.interp(
        np.asarray(depth_query, dtype=np.float64),
        np.asarray(depth_obs, dtype=np.float64)[order],
        np.asarray(y_obs, dtype=np.float64)[order],
    )


def _second_derivative(depth: np.ndarray) -> np.ndarray:
    """Second-derivative matrix along an increasing depth vector, shape (n-2, n)."""
    grid = np.asarray(depth, dtype=np.float64)
    n_depth = int(grid.shape[0])
    if n_depth < 3:
        raise ValueError("Smoothing needs at least 3 depths.")
    matrix = np.zeros((n_depth - 2, n_depth), dtype=np.float64)
    for index in range(n_depth - 2):
        left = float(grid[index + 1] - grid[index])
        right = float(grid[index + 2] - grid[index + 1])
        if left <= 0.0 or right <= 0.0:
            raise ValueError("Depth must be strictly increasing.")
        span = 0.5 * (left + right)
        matrix[index, index] = (1.0 / left) / span
        matrix[index, index + 1] = (-1.0 / left - 1.0 / right) / span
        matrix[index, index + 2] = (1.0 / right) / span
    return matrix


def _smooth_depth(
    depth: np.ndarray,
    y_values: np.ndarray,
    observed_index: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit ||M y - b||^2 + alpha ||D2 y||^2 on physical depth.

    A straight line is the alpha -> infinity member. Only observed entries of
    y_values enter the data term.
    """
    grid = np.asarray(depth, dtype=np.float64)
    values = np.asarray(y_values, dtype=np.float64)
    observed = np.asarray(observed_index, dtype=np.int64)
    n_depth = int(grid.shape[0])
    if values.shape != (n_depth,):
        raise ValueError("Smoothing target must match the depth vector.")
    if observed.size < 2:
        raise ValueError("Smoothing needs at least two observations.")
    if np.any(np.diff(grid) <= 0.0):
        raise ValueError("Depth must be strictly increasing.")
    center = float(np.mean(grid))
    scale = float(np.std(grid))
    if scale <= 0.0:
        raise ValueError("Depth has no spread.")
    standardized = (grid - center) / scale
    null_space = np.column_stack([np.ones(n_depth), standardized])
    if math.isinf(float(alpha)):
        coeff, _residual, rank, _singular = np.linalg.lstsq(
            null_space[observed, :],
            values[observed],
            rcond=None,
        )
        if int(rank) < 2:
            raise RuntimeError("Observed depths do not identify a straight line.")
        return np.asarray(null_space @ coeff, dtype=np.float64)
    if float(alpha) <= 0.0:
        raise ValueError("Smoothing alpha must be positive.")
    penalty = _second_derivative(grid)
    null_penalty = penalty @ null_space
    if float(np.max(np.abs(null_penalty))) > 1.0e-6:
        raise RuntimeError("A straight line is penalized by the depth operator.")
    orthogonal, _factor = np.linalg.qr(null_space, mode="complete")
    free = orthogonal[:, 2:]
    basis = np.hstack([null_space, free])
    n_obs = int(observed.size)
    design = np.zeros((n_obs + n_depth - 2, n_depth), dtype=np.float64)
    design[:n_obs, :] = basis[observed, :]
    design[n_obs:, :] = math.sqrt(float(alpha)) * (penalty @ basis)
    target = np.zeros(design.shape[0], dtype=np.float64)
    target[:n_obs] = values[observed]
    coeff, _residual, rank, _singular = np.linalg.lstsq(design, target, rcond=None)
    if int(rank) < n_depth:
        raise RuntimeError("Smoothing system lost rank at alpha {0}.".format(alpha))
    return np.asarray(basis @ coeff, dtype=np.float64)


def _hold_fitted_ends(curve: np.ndarray, observed_index: np.ndarray) -> np.ndarray:
    """Outside the observed span, hold the fitted value at the nearest observed plug.

    The held value is the smoother at that plug, which can differ from the raw
    measurement kept by interpolation.
    """
    values = np.asarray(curve, dtype=np.float64)
    observed = np.asarray(observed_index, dtype=np.int64)
    if values.ndim != 1 or observed.size < 1:
        raise ValueError("Constant extension needs a curve and an observed plug.")
    if int(np.min(observed)) < 0 or int(np.max(observed)) >= int(values.shape[0]):
        raise ValueError("Observed index is outside the smoothed curve.")
    left = int(np.min(observed))
    right = int(np.max(observed))
    held = np.array(values, copy=True)
    if left > 0:
        held[:left] = float(values[left])
    if right + 1 < int(held.shape[0]):
        held[right + 1 :] = float(values[right])
    return held


def _select_smooth_alpha(
    depth: np.ndarray,
    y_values: np.ndarray,
    observed_index: np.ndarray,
    scored_index: np.ndarray,
    hold_ends: bool,
) -> float:
    """Alpha with the lowest validation RMSE, including the straight-line limit.

    When hold_ends is true, the score uses the constant-extension curve.
    """
    best_alpha = float(SMOOTH_ALPHAS[0])
    best_score = float("inf")
    for alpha in tuple(SMOOTH_ALPHAS) + (SMOOTH_LINEAR_LIMIT,):
        curve = _smooth_depth(depth, y_values, observed_index, float(alpha))
        if hold_ends:
            curve = _hold_fitted_ends(curve, observed_index)
        score = _rmse(y_values[scored_index], curve[scored_index])
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def _segments(indices: np.ndarray) -> List[np.ndarray]:
    """Split a plug-index set into contiguous depth runs."""
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    if ordered.size < 1:
        raise ValueError("Empty plug set.")
    groups: List[np.ndarray] = []
    start = int(ordered[0])
    previous = int(ordered[0])
    for value in ordered[1:].tolist():
        if int(value) != previous + 1:
            groups.append(np.arange(start, previous + 1, dtype=np.int64))
            start = int(value)
        previous = int(value)
    groups.append(np.arange(start, previous + 1, dtype=np.int64))
    return groups


def _window_starts(indices: np.ndarray) -> List[int]:
    """Starts whose full window lies inside one contiguous train or test run."""
    starts: List[int] = []
    for segment in _segments(indices):
        if int(segment.shape[0]) < WINDOW_LEN:
            continue
        for offset in range(0, int(segment.shape[0]) - WINDOW_LEN + 1, WINDOW_STEP):
            starts.append(int(segment[offset]))
    if not starts:
        raise ValueError("No window fits inside the plug set.")
    return starts


def _stack_windows(
    u_matrix: np.ndarray,
    y_vector: np.ndarray,
    starts: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build context windows and log10(k) windows."""
    x_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    for start in starts:
        stop = int(start) + WINDOW_LEN
        y_rows.append(y_vector[start:stop])
        x_rows.append(u_matrix[start:stop, :].reshape(WINDOW_LEN * int(u_matrix.shape[1])))
    return np.vstack(x_rows), np.vstack(y_rows)


def _roles(masks: pd.DataFrame, fold_id: int, seed: int, rho: float, split: str) -> Dict[str, np.ndarray]:
    """Observed and scored plug indices for one frozen mask."""
    part = masks[
        (masks["fold_id"] == fold_id)
        & (masks["seed"] == seed)
        & np.isclose(masks["rho"], rho)
        & (masks["split"] == split)
    ]
    observed = part.loc[part["role"] == "observed", "plug_index"].to_numpy(dtype=np.int64)
    scored = part.loc[part["role"] == "scored", "plug_index"].to_numpy(dtype=np.int64)
    if observed.size < 1 or scored.size < 1:
        raise ValueError("Mask for fold {0} is missing a role.".format(fold_id))
    if set(observed.tolist()) & set(scored.tolist()):
        raise ValueError("Observed and scored plugs overlap.")
    return {"observed": np.sort(observed), "scored": np.sort(scored)}


def _draw_training_observed(train_index: np.ndarray, n_observed: int, seed: int) -> set:
    """Plug-level availability on the training plugs. Same count rule as evaluation."""
    indices = np.asarray(train_index, dtype=np.int64)
    if n_observed < 1 or n_observed >= int(indices.shape[0]):
        raise ValueError("Training availability must leave at least one plug hidden.")
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(indices, size=int(n_observed), replace=False)
    return set(int(value) for value in np.asarray(picked, dtype=np.int64).tolist())


def _embed_windows(
    x_windows: np.ndarray,
    y_windows: np.ndarray,
    starts: Sequence[int],
    observed_plugs: set,
) -> np.ndarray:
    """Place sparse log10(k) on the plugs covered by each window and append the mask.

    A plug is observed or not for the whole split. Sliding windows therefore
    receive different local patterns. The supervised target stays the full window.
    """
    n_context = int(x_windows.shape[1])
    rows: List[np.ndarray] = []
    for row_id, start in enumerate(starts):
        placed = np.zeros(WINDOW_LEN, dtype=np.float64)
        mask = np.zeros(WINDOW_LEN, dtype=np.float64)
        for offset in range(WINDOW_LEN):
            if int(start) + offset in observed_plugs:
                placed[offset] = float(y_windows[row_id, offset])
                mask[offset] = 1.0
        context = x_windows[row_id]
        if int(context.shape[0]) != n_context:
            raise ValueError("Context window length changed.")
        rows.append(np.concatenate([context, placed, mask]))
    embedded = np.vstack(rows)
    mask_part = embedded[:, -WINDOW_LEN:]
    if not np.all((mask_part == 0.0) | (mask_part == 1.0)):
        raise RuntimeError("Observation mask is not binary.")
    return embedded


def _assert_varied_training_masks(embedded: np.ndarray) -> int:
    """Stop if every training window was given the same local observation pattern."""
    mask_part = np.asarray(embedded[:, -WINDOW_LEN:], dtype=np.float64)
    unique = np.unique(mask_part, axis=0)
    if int(unique.shape[0]) < 2:
        raise RuntimeError("Training windows all received the same observation pattern.")
    return int(unique.shape[0])


def _local_groups(
    starts: Sequence[int],
    observed_plugs: set,
) -> Dict[Tuple[int, ...], List[int]]:
    """Group windows that share the same local observation pattern."""
    groups: Dict[Tuple[int, ...], List[int]] = {}
    for row_id, start in enumerate(starts):
        local = tuple(
            offset
            for offset in range(WINDOW_LEN)
            if int(start) + offset in observed_plugs
        )
        groups.setdefault(local, []).append(row_id)
    return groups


def _operator(local: Sequence[int]) -> np.ndarray:
    """Coordinate restriction with one selected plug per row."""
    matrix = np.zeros((len(local), WINDOW_LEN), dtype=np.float64)
    for row_id, column in enumerate(local):
        matrix[row_id, int(column)] = 1.0
    return matrix


def _recover_grouped(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_windows: np.ndarray,
    y_windows: np.ndarray,
    starts: Sequence[int],
    observed_plugs: set,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    cfg: Config,
    seed: int,
    z0_np: np.ndarray | None = None,
) -> np.ndarray:
    """Recover every window.

    The default start is the conditional prior z0(u). Pass z0_np to replace
    that start. An empty observation set returns the decoder at the start,
    because there is no measurement term to optimize.
    """
    predicted = np.zeros_like(y_windows)
    if z0_np is None:
        z0 = np.asarray(prior.predict(x_windows), dtype=np.float64)
    else:
        z0 = np.asarray(z0_np, dtype=np.float64)
    if z0.shape[0] != int(x_windows.shape[0]):
        raise ValueError("Latent start does not match the number of windows.")
    for local, rows in _local_groups(starts, observed_plugs).items():
        row_index = np.asarray(rows, dtype=np.int64)
        if len(local) == 0:
            predicted[row_index] = csgm.decode_latent_prior(
                ae,
                z0[row_index],
                y_mean,
                y_scale,
                "cpu",
            )
            continue
        recovered = csgm.csgm_recover_with_prior(
            ae=ae,
            M=_operator(local),
            B=y_windows[row_index][:, list(local)],
            z0_np=z0[row_index],
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device="cpu",
            seed=int(seed),
        )
        predicted[row_index] = recovered
    return predicted


def _mean_at_plugs(
    starts: Sequence[int],
    y_hat: np.ndarray,
    scored: np.ndarray,
) -> Dict[int, float]:
    """Average window predictions so each reserved plug is scored once."""
    buckets: Dict[int, List[float]] = {int(plug): [] for plug in scored.tolist()}
    for row_id, start in enumerate(starts):
        for offset in range(WINDOW_LEN):
            plug = int(start) + offset
            if plug in buckets:
                buckets[plug].append(float(y_hat[row_id, offset]))
    missing = [plug for plug, values in buckets.items() if not values]
    if missing:
        raise RuntimeError("A scored plug is absent from every window.")
    return {plug: float(np.mean(np.asarray(values, dtype=np.float64))) for plug, values in buckets.items()}


def _append_plug_rows(
    rows: List[Dict[str, float]],
    fold_id: int,
    seed: int,
    rho: float,
    method: str,
    depth: np.ndarray,
    y_vector: np.ndarray,
    prediction: Dict[int, float],
    extra: float,
    span_min: float,
    span_max: float,
) -> None:
    """Store one row per reserved plug, with a flag for the observed span."""
    for plug, y_hat in prediction.items():
        depth_m = float(depth[plug])
        outside = 1.0 if depth_m < span_min or depth_m > span_max else 0.0
        rows.append(
            {
                "fold_id": float(fold_id),
                "seed": float(seed),
                "rho": float(rho),
                "method": method,
                "plug_index": float(plug),
                "depth_m": depth_m,
                "y_true": float(y_vector[plug]),
                "y_pred": float(y_hat),
                "parameter": float(extra),
                "outside_span": outside,
            }
        )


def _fit_clp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    cfg: Config,
) -> Tuple[csgm.TinyAE, csgm.RidgePrior, np.ndarray, np.ndarray]:
    """Train the decoder on training windows and the ridge anchor on u."""
    y_scaler = StandardScaler()
    y_train_n = y_scaler.fit_transform(y_train)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    if np.any(y_scale <= 0.0):
        raise ValueError("Target scale is not positive.")
    ae = csgm.train_ae_generator(y_train_n, cfg, seed=int(seed), device="cpu")
    latent = csgm.encode_y(ae, y_train_n, device="cpu")
    prior = csgm.RidgePrior(alpha=float(cfg.csgm_ridge_alpha), random_state=int(seed))
    prior.fit(x_train, latent)
    return ae, prior, y_mean, y_scale


def _select_lambda(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_starts: Sequence[int],
    observed_plugs: set,
    scored: np.ndarray,
    y_vector: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    cfg: Config,
    seed: int,
) -> float:
    """Choose lambda by RMSE on scored validation plugs."""
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    for lam in cfg.csgm_lambda_grid:
        y_hat = _recover_grouped(
            ae,
            prior,
            x_val,
            y_val,
            val_starts,
            observed_plugs,
            y_mean,
            y_scale,
            float(lam),
            cfg,
            int(seed),
        )
        averaged = _mean_at_plugs(val_starts, y_hat, scored)
        ordered = np.asarray([averaged[int(plug)] for plug in scored.tolist()], dtype=np.float64)
        score = _rmse(y_vector[scored], ordered)
        if score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam


def _write_comparator_protocol() -> None:
    """Declare comparator rules before any corrected score is written."""
    CORRECTED_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Corrected comparator protocol for the 9-BRSA-1254-RJS plug holdout.",
        "The depth blocks, validation plugs, and test masks stay those in the parent folder.",
        "This is a retrospective score: laboratory permeability is real and availability is simulated.",
        "Primary aggregate, fixed before this run: unweighted mean of the nine fold-by-seed RMSE values.",
        "Each of those RMSE values uses the reserved plugs of that fold and seed once.",
        "Per-fold means are reported beside the primary aggregate.",
        "A pooled RMSE, stacking the nine cells, is stored and is not the ranking metric.",
        "No dense permeability curve is built.",
        "",
        "Measurement rule:",
        "- Training plugs fit the decoder, the conditional prior, the direct regression, and the direct AE and MLP.",
        "- Validation plugs select smoothing alpha and CLP lambda. Their targets are not test measurements.",
        "- Interpolation, both smoothers, and residual correction may use only the observed plugs of the block being predicted.",
        "- Residual correction adds the residual at those observed plugs to the regression. It does not anchor residuals on training or validation depths.",
        "- Direct AE and MLP inputs use one plug-level availability draw on the training plugs, at the same rho.",
        "- That draw uses mask_seed + 90 and the same observation count as evaluation. Windows inherit the plugs they cover.",
        "- The draw is not one shared set of window offsets. The supervised target remains the full training window.",
        "",
        "Method names:",
        "- clp: conditional start z0(u), lambda chosen on scored validation plugs.",
        "- clp_no_penalty: same conditional start, penalty weight 0. The label csgm in the first run was this ablation.",
        "- csgm_uncond: penalty weight 0 and latent start at the origin, independent of the conditioning logs.",
        "- Empty windows of csgm_uncond decode G(0). Restarts add the recovery routine's 0.05 noise around the origin.",
        "- prior_only: G(z0), with no measurement update.",
        "- smooth: depth smoother free to leave the observed span. Alpha includes the straight-line limit.",
        "- smooth_constant: same fit, then depths outside the span hold the fitted value at the nearest observed plug.",
        "- smooth_constant selects its own alpha on validation after that hold is applied.",
        "Inside the span means a reserved plug lies between the shallowest and deepest observed plug of that test mask.",
        "Observed plugs are not scored.",
        "",
    ]
    (CORRECTED_DIR / "PROTOCOL_COMPARATORS.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _span_metrics(part: pd.DataFrame) -> Dict[str, float]:
    """RMSE inside and outside the observed span for one fold, seed, and method."""
    outside = part["outside_span"].to_numpy(dtype=np.float64) > 0.5
    inside = np.logical_not(outside)
    truth = part["y_true"].to_numpy(dtype=np.float64)
    pred = part["y_pred"].to_numpy(dtype=np.float64)
    gap2 = (truth - pred) * (truth - pred)
    total = float(np.sum(gap2))
    if total <= 0.0:
        share = float("nan")
    else:
        share = float(np.sum(gap2[outside]) / total)
    if int(np.sum(inside)) > 0:
        rmse_inside = _rmse(truth[inside], pred[inside])
    else:
        rmse_inside = float("nan")
    if int(np.sum(outside)) > 0:
        rmse_outside = _rmse(truth[outside], pred[outside])
    else:
        rmse_outside = float("nan")
    return {
        "n_inside": float(np.sum(inside)),
        "n_outside": float(np.sum(outside)),
        "rmse_inside": rmse_inside,
        "rmse_outside": rmse_outside,
        "sse_share_outside": share,
    }


def main() -> None:
    """Fit every comparator and write plug-level RMSE tables."""
    config_path = OUT_DIR / "config_frozen.json"
    protocol_path = OUT_DIR / "PROTOCOL.txt"
    if not config_path.is_file() or not protocol_path.is_file():
        raise FileNotFoundError("Run freeze_split.py before scoring.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if bool(config.get("models_trained_by_this_script", False)):
        raise ValueError("Frozen config was already marked as a trained run.")
    aligned = pd.read_csv(OUT_DIR / "tables" / "plug_log_aligned.csv")
    split = pd.read_csv(OUT_DIR / "tables" / "split_by_plug.csv")
    masks = pd.read_csv(OUT_DIR / "tables" / "masks.csv")
    if int(aligned.shape[0]) != int(config["n_kept"]):
        raise ValueError("Aligned table does not match the frozen config.")
    depth = aligned["depth_m"].to_numpy(dtype=np.float64)
    y_vector = aligned["log10_k"].to_numpy(dtype=np.float64)
    u_matrix = aligned.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(u_matrix).all() or not np.isfinite(y_vector).all():
        raise ValueError("Aligned table has non-finite values.")
    _write_comparator_protocol()

    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_lambda_grid = list(LAMBDA_GRID)
    cfg.csgm_device = "cpu"
    cfg.measurement_noise_std = 0.0
    cfg.baseline_hidden = (32, 32)
    cfg.baseline_max_iter = 400
    dub_cfg = dub.DirectUBTrainConfig(
        ae_latent_dim=LATENT_DIM,
        ae_hidden=HIDDEN_DIM,
        ae_epochs=AE_EPOCHS,
        ae_batch_size=16,
    )
    torch.manual_seed(0)

    rows: List[Dict[str, float]] = []
    for fold_id in range(3):
        fold = split[split["fold_id"] == fold_id]
        train_index = np.sort(fold.loc[fold["split"] == "train", "plug_index"].to_numpy(dtype=np.int64))
        val_index = np.sort(fold.loc[fold["split"] == "val", "plug_index"].to_numpy(dtype=np.int64))
        test_index = np.sort(fold.loc[fold["split"] == "test", "plug_index"].to_numpy(dtype=np.int64))
        if set(train_index.tolist()) & set(test_index.tolist()):
            raise ValueError("Frozen train and test sets overlap.")
        train_starts = _window_starts(train_index)
        val_starts = _window_starts(val_index)
        test_starts = _window_starts(test_index)
        x_train, y_train = _stack_windows(u_matrix, y_vector, train_starts)
        x_val, y_val = _stack_windows(u_matrix, y_vector, val_starts)
        x_test, y_test = _stack_windows(u_matrix, y_vector, test_starts)
        reg_scaler = StandardScaler().fit(u_matrix[train_index])
        regressor = LinearRegression()
        regressor.fit(reg_scaler.transform(u_matrix[train_index]), y_vector[train_index])

        def _regression_at(index: np.ndarray) -> np.ndarray:
            return np.asarray(
                regressor.predict(reg_scaler.transform(u_matrix[index])),
                dtype=np.float64,
            )

        print(
            "fold",
            fold_id,
            "train_windows",
            len(train_starts),
            "val_windows",
            len(val_starts),
            "test_windows",
            len(test_starts),
            flush=True,
        )
        for seed in config["seeds"]:
            ae, prior, y_mean, y_scale = _fit_clp(x_train, y_train, int(seed), cfg)
            for rho in config["rhos"]:
                val_roles = _roles(masks, fold_id, int(seed), float(rho), "val")
                test_roles = _roles(masks, fold_id, int(seed), float(rho), "test")
                val_obs = set(int(value) for value in val_roles["observed"].tolist())
                test_obs = set(int(value) for value in test_roles["observed"].tolist())
                val_local_obs = _positions(val_index, val_roles["observed"])
                test_local_obs = _positions(test_index, test_roles["observed"])
                alpha = _select_smooth_alpha(
                    depth[val_index],
                    y_vector[val_index],
                    val_local_obs,
                    _positions(val_index, val_roles["scored"]),
                    hold_ends=False,
                )
                alpha_constant = _select_smooth_alpha(
                    depth[val_index],
                    y_vector[val_index],
                    val_local_obs,
                    _positions(val_index, val_roles["scored"]),
                    hold_ends=True,
                )
                smooth_test = _smooth_depth(
                    depth[test_index],
                    y_vector[test_index],
                    test_local_obs,
                    alpha,
                )
                smooth_constant_test = _hold_fitted_ends(
                    _smooth_depth(
                        depth[test_index],
                        y_vector[test_index],
                        test_local_obs,
                        alpha_constant,
                    ),
                    test_local_obs,
                )
                interp_test = _interp_depth(
                    depth[test_roles["observed"]],
                    y_vector[test_roles["observed"]],
                    depth[test_roles["scored"]],
                )
                direct_test = _regression_at(test_roles["scored"])
                # Residual anchors are the test observations only, the same set used by interpolation.
                if int(test_roles["observed"].shape[0]) < 2:
                    raise ValueError("Residual correction needs two observed test plugs.")
                residual_known = y_vector[test_roles["observed"]] - _regression_at(test_roles["observed"])
                residual_query = _interp_depth(
                    depth[test_roles["observed"]],
                    residual_known,
                    depth[test_roles["scored"]],
                )
                residual_test = _regression_at(test_roles["scored"]) + residual_query

                train_obs = _draw_training_observed(
                    train_index,
                    observation_count(int(train_index.shape[0]), float(rho)),
                    mask_seed(int(seed), fold_id, float(rho)) + 90,
                )
                if not train_obs.issubset(set(int(value) for value in train_index.tolist())):
                    raise RuntimeError("Training availability left the training plugs.")
                xb_train = _embed_windows(x_train, y_train, train_starts, train_obs)
                n_patterns = _assert_varied_training_masks(xb_train)
                xb_val = _embed_windows(x_val, y_val, val_starts, val_obs)
                xb_test = _embed_windows(x_test, y_test, test_starts, test_obs)
                xb_scaler = StandardScaler().fit(xb_train)
                _ae_val, ae_test = dub.fit_predict_ae_regression_ub(
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
                _mlp_val, mlp_test = dub.fit_predict_mlp_concat(
                    cfg,
                    int(seed),
                    xb_train,
                    y_train,
                    xb_val,
                    y_val,
                    xb_test,
                    xb_scaler,
                )
                lam = _select_lambda(
                    ae,
                    prior,
                    x_val,
                    y_val,
                    val_starts,
                    val_obs,
                    val_roles["scored"],
                    y_vector,
                    y_mean,
                    y_scale,
                    cfg,
                    int(seed),
                )
                clp_windows = _recover_grouped(
                    ae,
                    prior,
                    x_test,
                    y_test,
                    test_starts,
                    test_obs,
                    y_mean,
                    y_scale,
                    lam,
                    cfg,
                    int(seed),
                )
                no_penalty_windows = _recover_grouped(
                    ae,
                    prior,
                    x_test,
                    y_test,
                    test_starts,
                    test_obs,
                    y_mean,
                    y_scale,
                    0.0,
                    cfg,
                    int(seed),
                )
                origin = np.zeros((int(x_test.shape[0]), LATENT_DIM), dtype=np.float64)
                uncond_windows = _recover_grouped(
                    ae,
                    prior,
                    x_test,
                    y_test,
                    test_starts,
                    test_obs,
                    y_mean,
                    y_scale,
                    0.0,
                    cfg,
                    int(seed),
                    z0_np=origin,
                )
                prior_windows = csgm.decode_latent_prior(
                    ae,
                    np.asarray(prior.predict(x_test), dtype=np.float64),
                    y_mean,
                    y_scale,
                    "cpu",
                )
                predictions = {
                    "interp": {
                        int(plug): float(value)
                        for plug, value in zip(test_roles["scored"].tolist(), interp_test.tolist())
                    },
                    "smooth": _mean_from_block(test_index, smooth_test, test_roles["scored"]),
                    "smooth_constant": _mean_from_block(
                        test_index,
                        smooth_constant_test,
                        test_roles["scored"],
                    ),
                    "direct_regression": {
                        int(plug): float(value)
                        for plug, value in zip(test_roles["scored"].tolist(), direct_test.tolist())
                    },
                    "residual_correction": {
                        int(plug): float(value)
                        for plug, value in zip(test_roles["scored"].tolist(), residual_test.tolist())
                    },
                    "ae_ub": _mean_at_plugs(
                        test_starts,
                        np.asarray(ae_test, dtype=np.float64),
                        test_roles["scored"],
                    ),
                    "mlp_ub": _mean_at_plugs(
                        test_starts,
                        np.asarray(mlp_test, dtype=np.float64),
                        test_roles["scored"],
                    ),
                    "clp_no_penalty": _mean_at_plugs(
                        test_starts,
                        no_penalty_windows,
                        test_roles["scored"],
                    ),
                    "csgm_uncond": _mean_at_plugs(
                        test_starts,
                        uncond_windows,
                        test_roles["scored"],
                    ),
                    "prior_only": _mean_at_plugs(
                        test_starts,
                        prior_windows,
                        test_roles["scored"],
                    ),
                    "clp": _mean_at_plugs(test_starts, clp_windows, test_roles["scored"]),
                }
                parameters = {
                    "interp": float("nan"),
                    "smooth": alpha,
                    "smooth_constant": alpha_constant,
                    "direct_regression": float("nan"),
                    "residual_correction": float("nan"),
                    "ae_ub": float("nan"),
                    "mlp_ub": float("nan"),
                    "clp_no_penalty": 0.0,
                    "csgm_uncond": 0.0,
                    "prior_only": float("nan"),
                    "clp": lam,
                }
                span_min = float(np.min(depth[test_roles["observed"]]))
                span_max = float(np.max(depth[test_roles["observed"]]))
                for method, prediction in predictions.items():
                    _append_plug_rows(
                        rows,
                        fold_id,
                        int(seed),
                        float(rho),
                        method,
                        depth,
                        y_vector,
                        prediction,
                        parameters[method],
                        span_min,
                        span_max,
                    )
                print(
                    "  seed",
                    int(seed),
                    "rho",
                    float(rho),
                    "train_patterns",
                    n_patterns,
                    "smooth_alpha",
                    alpha,
                    "smooth_constant_alpha",
                    alpha_constant,
                    "lambda",
                    lam,
                    flush=True,
                )

    detail = pd.DataFrame(rows)
    summary_rows: List[Dict[str, float]] = []
    for keys, part in detail.groupby(["fold_id", "seed", "rho", "method"], sort=True):
        summary_rows.append(
            {
                "fold_id": float(keys[0]),
                "seed": float(keys[1]),
                "rho": float(keys[2]),
                "method": str(keys[3]),
                "n_scored": float(part.shape[0]),
                "rmse": _rmse(
                    part["y_true"].to_numpy(dtype=np.float64),
                    part["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    if int(summary.groupby(["fold_id", "seed", "rho", "method"]).size().max()) != 1:
        raise RuntimeError("A fold, seed, fraction, and method was scored twice.")
    primary_rows: List[Dict[str, float]] = []
    for keys, part in summary.groupby(["rho", "method"], sort=True):
        if int(part.shape[0]) != 9:
            raise RuntimeError("Primary mean expects nine fold-seed cells.")
        primary_rows.append(
            {
                "rho": float(keys[0]),
                "method": str(keys[1]),
                "n_cells": float(part.shape[0]),
                "mean_rmse": float(part["rmse"].mean()),
                "std_rmse": float(part["rmse"].std(ddof=1)),
            }
        )
    fold_rows: List[Dict[str, float]] = []
    for keys, part in summary.groupby(["fold_id", "rho", "method"], sort=True):
        fold_rows.append(
            {
                "fold_id": float(keys[0]),
                "rho": float(keys[1]),
                "method": str(keys[2]),
                "n_seeds": float(part.shape[0]),
                "mean_rmse": float(part["rmse"].mean()),
                "std_rmse": float(part["rmse"].std(ddof=1)),
            }
        )
    pooled_rows: List[Dict[str, float]] = []
    for keys, part in detail.groupby(["rho", "method"], sort=True):
        pooled_rows.append(
            {
                "rho": float(keys[0]),
                "method": str(keys[1]),
                "n_scored": float(part.shape[0]),
                "pooled_rmse": _rmse(
                    part["y_true"].to_numpy(dtype=np.float64),
                    part["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    span_rows: List[Dict[str, float]] = []
    for keys, part in detail.groupby(["fold_id", "seed", "rho", "method"], sort=True):
        metrics = _span_metrics(part)
        span_rows.append(
            {
                "fold_id": float(keys[0]),
                "seed": float(keys[1]),
                "rho": float(keys[2]),
                "method": str(keys[3]),
                "n_inside": metrics["n_inside"],
                "n_outside": metrics["n_outside"],
                "rmse_inside": metrics["rmse_inside"],
                "rmse_outside": metrics["rmse_outside"],
                "sse_share_outside": metrics["sse_share_outside"],
            }
        )
    table_dir = CORRECTED_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(table_dir / "detail_by_plug.csv", index=False)
    summary.to_csv(table_dir / "summary_by_seed.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(table_dir / "summary_by_fold.csv", index=False)
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(table_dir / "summary_primary.csv", index=False)
    pd.DataFrame(pooled_rows).to_csv(table_dir / "summary_pooled.csv", index=False)
    pd.DataFrame(span_rows).to_csv(table_dir / "span_by_seed.csv", index=False)
    print(primary.to_string(index=False))
    print("wrote", table_dir)


def _positions(block_index: np.ndarray, plugs: np.ndarray) -> np.ndarray:
    """Local offsets of plugs inside a depth-sorted block."""
    location = {int(plug): offset for offset, plug in enumerate(block_index.tolist())}
    offsets: List[int] = []
    for plug in plugs.tolist():
        if int(plug) not in location:
            raise ValueError("Plug {0} is outside the block.".format(int(plug)))
        offsets.append(location[int(plug)])
    return np.asarray(offsets, dtype=np.int64)


def _mean_from_block(
    block_index: np.ndarray,
    curve: np.ndarray,
    scored: np.ndarray,
) -> Dict[int, float]:
    """Map a block curve back to scored plug indices."""
    position = {int(plug): offset for offset, plug in enumerate(block_index.tolist())}
    return {int(plug): float(curve[position[int(plug)]]) for plug in scored.tolist()}


if __name__ == "__main__":
    main()
