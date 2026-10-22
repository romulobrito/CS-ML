#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prior transfer diagnostic for the 9-BRSA-1254-RJS plug holdout.

The first part reads frozen predictions. The second rebuilds G(z0) on every
plug in the train, validation, and test blocks, then checks those values
against the stored scored-plug predictions. Residual corrections are fixed
rules. They are not chosen from the test score.

No manuscript edit and no change to the CLP architecture.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, wasserstein_distance
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import csgm_m2_module as csgm
from sir_cs_pipeline_optimized import Config

from common import (  # noqa: E402
    AE_EPOCHS,
    CSGM_ITERS,
    CSGM_RESTARTS,
    HIDDEN_DIM,
    LATENT_DIM,
    OUT_DIR,
    U_COLUMNS,
    WINDOW_LEN,
)
from run_plug_holdout import (  # noqa: E402
    _fit_clp,
    _mean_at_plugs,
    _roles,
    _stack_windows,
    _window_starts,
)

BASE = OUT_DIR
AUDIT22 = BASE / "restart_oracle_diagnostic"
CORRECTED = BASE / "corrected_protocol"
OUT = BASE / "prior_transfer_diagnostic"
CHANNELS = ("gr", "rhob", "nphi", "rt")
SEEDS = (7, 23, 41)
RHOS = (0.25, 0.50)
BASE_SHA = "14759285a6ab0462e4f2b215ddd9e32f39e34b5a"
RMSE_TOLERANCE = 1.0e-4
PREDICTION_TOLERANCE = 1.0e-4


def _sha256(path: Path) -> str:
    """Hash a frozen input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error of two aligned vectors."""
    gap = np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean(gap * gap)))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman coefficient. The p-value is not used."""
    if int(left.shape[0]) < 3 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return float("nan")
    coefficient = spearmanr(left, right).statistic
    return float(coefficient)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    """Tenth, median, and ninetieth percentiles."""
    grid = np.quantile(np.asarray(values, dtype=np.float64), [0.1, 0.5, 0.9])
    return {"p10": float(grid[0]), "p50": float(grid[1]), "p90": float(grid[2])}


def _load_frozen() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the artifacts written by the restart-oracle audit."""
    paths = {
        "aligned": BASE / "tables" / "plug_log_aligned.csv",
        "split": BASE / "tables" / "split_by_plug.csv",
        "masks": BASE / "tables" / "masks.csv",
        "pred": AUDIT22 / "tables" / "detail_by_plug.csv",
        "cells": AUDIT22 / "tables" / "results_by_cell.csv",
        "config": AUDIT22 / "config.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: {0}".format(", ".join(missing)))
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    if _sha256(paths["split"]) != str(config["split_sha256"]):
        raise RuntimeError("Frozen split hash mismatch.")
    if _sha256(paths["masks"]) != str(config["masks_sha256"]):
        raise RuntimeError("Frozen mask hash mismatch.")
    aligned = pd.read_csv(paths["aligned"])
    split = pd.read_csv(paths["split"])
    masks = pd.read_csv(paths["masks"])
    pred = pd.read_csv(paths["pred"])
    cells = pd.read_csv(paths["cells"])
    if int(aligned.shape[0]) != 161:
        raise RuntimeError("Expected 161 aligned plugs.")
    for frame in (split, masks, pred, cells):
        frame["fold_id"] = frame["fold_id"].astype(int)
    for frame in (masks, pred, cells):
        frame["seed"] = frame["seed"].astype(int)
    return aligned, split, masks, pred, cells, config


def _distribution_shift(aligned: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Training-standardized shift of each auxiliary channel."""
    rows: List[dict] = []
    table = aligned.set_index("plug_index")
    for fold_id in range(3):
        partition = split[split["fold_id"] == fold_id]
        selected = {}
        for role in ("train", "val", "test"):
            index = partition.loc[partition["split"] == role, "plug_index"].astype(int)
            selected[role] = table.loc[index]
        reference = selected["train"]
        for feature in CHANNELS:
            center = float(reference[feature].mean())
            scale = float(reference[feature].std(ddof=0))
            if scale <= 0.0 or not np.isfinite(scale):
                raise RuntimeError("Invalid training scale for {0}.".format(feature))
            reference_values = reference[feature].to_numpy(dtype=np.float64)
            reference_quantiles = _quantiles(reference_values)
            for role in ("val", "test"):
                target = selected[role][feature].to_numpy(dtype=np.float64)
                target_quantiles = _quantiles(target)
                outside = np.mean((target < reference_values.min()) | (target > reference_values.max()))
                rows.append(
                    {
                        "fold_id": fold_id,
                        "split": role,
                        "feature": feature,
                        "n_reference": int(reference.shape[0]),
                        "n_comparison": int(target.shape[0]),
                        "standardized_mean_shift": float((target.mean() - center) / scale),
                        "standardized_wasserstein": float(
                            wasserstein_distance(reference_values / scale, target / scale)
                        ),
                        "outside_reference_range": float(outside),
                        "reference_p10": reference_quantiles["p10"],
                        "reference_p50": reference_quantiles["p50"],
                        "reference_p90": reference_quantiles["p90"],
                        "comparison_p10": target_quantiles["p10"],
                        "comparison_p50": target_quantiles["p50"],
                        "comparison_p90": target_quantiles["p90"],
                    }
                )
    return pd.DataFrame(rows)


def _correlation_shift(aligned: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Change in pairwise channel correlation from training to test or validation."""
    rows: List[dict] = []
    table = aligned.set_index("plug_index")
    pairs = [(left, right) for index, left in enumerate(CHANNELS) for right in CHANNELS[index + 1 :]]
    for fold_id in range(3):
        partition = split[split["fold_id"] == fold_id]
        selected = {}
        for role in ("train", "val", "test"):
            index = partition.loc[partition["split"] == role, "plug_index"].astype(int)
            selected[role] = table.loc[index, list(CHANNELS)]
        reference = selected["train"].corr(method="pearson")
        for role in ("val", "test"):
            comparison = selected[role].corr(method="pearson")
            for left, right in pairs:
                rows.append(
                    {
                        "fold_id": fold_id,
                        "split": role,
                        "feature_left": left,
                        "feature_right": right,
                        "train_correlation": float(reference.loc[left, right]),
                        "comparison_correlation": float(comparison.loc[left, right]),
                        "correlation_shift": float(comparison.loc[left, right] - reference.loc[left, right]),
                    }
                )
    return pd.DataFrame(rows)


def _check_saved_scores(pred: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Confirm the stored plug table still has the stored cell RMSE."""
    rows: List[dict] = []
    for fold_id in range(3):
        for seed in SEEDS:
            for rho in RHOS:
                reference = cells[
                    (cells["fold_id"] == fold_id)
                    & (cells["seed"] == seed)
                    & np.isclose(cells["rho"], rho)
                ]
                if int(reference.shape[0]) != 1:
                    raise RuntimeError("Missing a stored cell.")
                reference_row = reference.iloc[0]
                for method, column in (("prior_only", "prior_rmse"), ("clp_collective", "clp_original_rmse")):
                    subset = pred[
                        (pred["fold_id"] == fold_id)
                        & (pred["seed"] == seed)
                        & np.isclose(pred["rho"], rho)
                        & (pred["method"] == method)
                    ]
                    score = _rmse(
                        subset["y_true"].to_numpy(dtype=np.float64),
                        subset["y_pred"].to_numpy(dtype=np.float64),
                    )
                    if abs(score - float(reference_row[column])) > RMSE_TOLERANCE:
                        raise RuntimeError("Stored {0} RMSE does not match the cell table.".format(method))
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": seed,
                            "rho": rho,
                            "method": method,
                            "recomputed_rmse": score,
                            "stored_rmse": float(reference_row[column]),
                        }
                    )
    return pd.DataFrame(rows)


def _direct_field(aligned: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Ordinary least squares from the four logs, fit on training plugs only."""
    rows: List[dict] = []
    for fold_id in range(3):
        partition = split[split["fold_id"] == fold_id]
        train_index = partition.loc[partition["split"] == "train", "plug_index"].astype(int).to_numpy()
        scaler = StandardScaler().fit(aligned.loc[train_index, list(CHANNELS)])
        model = LinearRegression()
        model.fit(
            scaler.transform(aligned.loc[train_index, list(CHANNELS)]),
            aligned.loc[train_index, "log10_k"].to_numpy(dtype=np.float64),
        )
        for role in ("train", "val", "test"):
            index = partition.loc[partition["split"] == role, "plug_index"].astype(int).to_numpy()
            prediction = model.predict(scaler.transform(aligned.loc[index, list(CHANNELS)]))
            for plug, value in zip(index.tolist(), prediction.tolist()):
                rows.append(
                    {
                        "fold_id": fold_id,
                        "split": role,
                        "plug_index": int(plug),
                        "direct_pred": float(value),
                    }
                )
    return pd.DataFrame(rows)


def _prior_fields(aligned: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    """Rebuild G(z0) on every plug. The field does not depend on the mask."""
    depth = aligned["depth_m"].to_numpy(dtype=np.float64)
    y_vector = aligned["log10_k"].to_numpy(dtype=np.float64)
    u_matrix = aligned.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_device = "cpu"
    cfg.measurement_noise_std = 0.0
    torch.manual_seed(0)
    rows: List[dict] = []
    for fold_id in range(3):
        partition = split[split["fold_id"] == fold_id]
        train_index = np.sort(partition.loc[partition["split"] == "train", "plug_index"].to_numpy(dtype=np.int64))
        train_starts = _window_starts(train_index)
        x_train, y_train = _stack_windows(u_matrix, y_vector, train_starts)
        print("fold", fold_id, "train_windows", len(train_starts), flush=True)
        for seed in SEEDS:
            _ae, prior, mean, scale = _fit_clp(x_train, y_train, int(seed), cfg)
            for role in ("train", "val", "test"):
                index = np.sort(partition.loc[partition["split"] == role, "plug_index"].to_numpy(dtype=np.int64))
                starts = _window_starts(index)
                x_windows, _y_windows = _stack_windows(u_matrix, y_vector, starts)
                latent = np.asarray(prior.predict(x_windows), dtype=np.float64)
                decoded = csgm.decode_latent_prior(_ae, latent, mean, scale, "cpu")
                prediction = _mean_at_plugs(starts, decoded, index)
                for plug in index.tolist():
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": int(seed),
                            "split": role,
                            "plug_index": int(plug),
                            "depth_m": float(depth[int(plug)]),
                            "y_true": float(y_vector[int(plug)]),
                            "prior_pred": float(prediction[int(plug)]),
                        }
                    )
            print("  seed", int(seed), "prior field written", flush=True)
    return pd.DataFrame(rows)


def _match_saved_prior(fields: pd.DataFrame, pred: pd.DataFrame) -> float:
    """Maximum absolute gap between rebuilt and stored prior predictions."""
    stored = pred[pred["method"] == "prior_only"].copy()
    stored["plug_index"] = stored["plug_index"].astype(int)
    # The prior field does not depend on rho. Both saved rhos must match it.
    gaps: List[float] = []
    for rho in RHOS:
        part = stored[np.isclose(stored["rho"], rho)]
        merged = part.merge(
            fields[fields["split"] == "test"],
            on=["fold_id", "seed", "plug_index"],
            how="inner",
            validate="many_to_one",
        )
        if int(merged.shape[0]) != int(part.shape[0]):
            raise RuntimeError("Rebuilt prior field missed a stored scored plug.")
        gap = np.abs(merged["prior_pred"].to_numpy(dtype=np.float64) - merged["y_pred"].to_numpy(dtype=np.float64))
        gaps.append(float(np.max(gap)))
        score_gap = abs(
            _rmse(merged["y_true_x"].to_numpy(dtype=np.float64), merged["prior_pred"].to_numpy(dtype=np.float64))
            - _rmse(merged["y_true_x"].to_numpy(dtype=np.float64), merged["y_pred"].to_numpy(dtype=np.float64))
        )
        if score_gap > RMSE_TOLERANCE:
            raise RuntimeError("Rebuilt prior RMSE left the stored score.")
    maximum = float(np.max(np.asarray(gaps, dtype=np.float64)))
    if maximum > PREDICTION_TOLERANCE:
        raise RuntimeError("Rebuilt prior predictions differ by {0}.".format(maximum))
    return maximum


def _nearest_residual(depth: np.ndarray, residual: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Residual of the nearest observed depth. No tunable parameter."""
    observed = np.asarray(depth, dtype=np.float64)
    values = np.asarray(residual, dtype=np.float64)
    order = np.argsort(observed)
    observed = observed[order]
    values = values[order]
    position = np.searchsorted(observed, np.asarray(query, dtype=np.float64))
    position = np.clip(position, 1, observed.shape[0] - 1)
    left = position - 1
    choose_right = np.abs(observed[position] - query) < np.abs(query - observed[left])
    chosen = np.where(choose_right, position, left)
    return values[chosen]


def _interpolated_residual(depth: np.ndarray, residual: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Depth interpolation of observed residuals. Ends hold the nearest value."""
    order = np.argsort(np.asarray(depth, dtype=np.float64))
    return np.interp(
        np.asarray(query, dtype=np.float64),
        np.asarray(depth, dtype=np.float64)[order],
        np.asarray(residual, dtype=np.float64)[order],
    )


def _transfer_table(
    fields: pd.DataFrame,
    direct: pd.DataFrame,
    masks: pd.DataFrame,
    pred: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compare fixed residual rules on reserved plugs. Rules are not tuned on test."""
    cell_rows: List[dict] = []
    pair_rows: List[dict] = []
    clp = pred[pred["method"] == "clp_collective"].copy()
    clp["plug_index"] = clp["plug_index"].astype(int)
    for fold_id in range(3):
        for seed in SEEDS:
            for role in ("val", "test"):
                block = fields[
                    (fields["fold_id"] == fold_id) & (fields["seed"] == seed) & (fields["split"] == role)
                ].copy()
                block["plug_index"] = block["plug_index"].astype(int)
                block = block.set_index("plug_index")
                direct_block = direct[
                    (direct["fold_id"] == fold_id) & (direct["split"] == role)
                ].set_index("plug_index")
                for rho in RHOS:
                    roles = _roles(masks, fold_id, int(seed), float(rho), role)
                    observed = np.sort(roles["observed"])
                    scored = np.sort(roles["scored"])
                    observed_residual = block.loc[observed, "y_true"].to_numpy() - block.loc[observed, "prior_pred"].to_numpy()
                    reserved_truth = block.loc[scored, "y_true"].to_numpy(dtype=np.float64)
                    reserved_prior = block.loc[scored, "prior_pred"].to_numpy(dtype=np.float64)
                    reserved_residual = reserved_truth - reserved_prior
                    nearest = _nearest_residual(
                        block.loc[observed, "depth_m"].to_numpy(),
                        observed_residual,
                        block.loc[scored, "depth_m"].to_numpy(),
                    )
                    interpolated = _interpolated_residual(
                        block.loc[observed, "depth_m"].to_numpy(),
                        observed_residual,
                        block.loc[scored, "depth_m"].to_numpy(),
                    )
                    direct_pred = direct_block.loc[scored, "direct_pred"].to_numpy(dtype=np.float64)
                    clp_score = float("nan")
                    if role == "test":
                        clp_part = clp[
                            (clp["fold_id"] == fold_id)
                            & (clp["seed"] == seed)
                            & np.isclose(clp["rho"], rho)
                        ].set_index("plug_index")
                        clp_pred = clp_part.loc[scored, "y_pred"].to_numpy(dtype=np.float64)
                        clp_score = _rmse(reserved_truth, clp_pred)
                    observed_depth = block.loc[observed, "depth_m"].to_numpy(dtype=np.float64)
                    scored_depth = block.loc[scored, "depth_m"].to_numpy(dtype=np.float64)
                    gaps = np.min(np.abs(scored_depth[:, None] - observed_depth[None, :]), axis=1)
                    cell_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": int(seed),
                            "split": role,
                            "rho": float(rho),
                            "n_observed": int(observed.shape[0]),
                            "n_scored": int(scored.shape[0]),
                            "prior_rmse": _rmse(reserved_truth, reserved_prior),
                            "prior_bias": float(np.mean(reserved_residual)),
                            "prior_mae": float(np.mean(np.abs(reserved_residual))),
                            "nearest_rmse": _rmse(reserved_truth, reserved_prior + nearest),
                            "interp_rmse": _rmse(reserved_truth, reserved_prior + interpolated),
                            "direct_rmse": _rmse(reserved_truth, direct_pred),
                            "clp_rmse": clp_score,
                            "nearest_residual_spearman": _spearman(reserved_residual, nearest),
                            "residual_wasserstein": float(wasserstein_distance(observed_residual, reserved_residual)),
                            "mean_nearest_observation_m": float(np.mean(gaps)),
                            "max_nearest_observation_m": float(np.max(gaps)),
                        }
                    )
                    for plug, residual, borrowed, gap in zip(scored.tolist(), reserved_residual, nearest, gaps):
                        pair_rows.append(
                            {
                                "fold_id": fold_id,
                                "seed": int(seed),
                                "split": role,
                                "rho": float(rho),
                                "plug_index": int(plug),
                                "depth_m": float(block.loc[int(plug), "depth_m"]),
                                "reserved_residual": float(residual),
                                "nearest_observed_residual": float(borrowed),
                                "nearest_observation_m": float(gap),
                            }
                        )
    return pd.DataFrame(cell_rows), pd.DataFrame(pair_rows)


def _variogram(fields: pd.DataFrame) -> pd.DataFrame:
    """Exploratory semivariogram of the mask-independent prior residual on test blocks."""
    edges = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
    rows: List[dict] = []
    test = fields[fields["split"] == "test"]
    for fold_id in range(3):
        for seed in SEEDS:
            block = test[(test["fold_id"] == fold_id) & (test["seed"] == seed)].sort_values("depth_m")
            depth = block["depth_m"].to_numpy(dtype=np.float64)
            residual = (block["y_true"] - block["prior_pred"]).to_numpy(dtype=np.float64)
            distance = np.abs(depth[:, None] - depth[None, :])
            squared = 0.5 * (residual[:, None] - residual[None, :]) ** 2
            upper = np.triu_indices(depth.shape[0], k=1)
            pair_distance = distance[upper]
            pair_gamma = squared[upper]
            for left, right in zip(edges[:-1], edges[1:]):
                chosen = (pair_distance >= left) & (pair_distance < right)
                if int(np.sum(chosen)) < 5:
                    gamma = float("nan")
                else:
                    gamma = float(np.mean(pair_gamma[chosen]))
                rows.append(
                    {
                        "fold_id": fold_id,
                        "seed": int(seed),
                        "lag_left_m": float(left),
                        "lag_right_m": float(right),
                        "n_pairs": int(np.sum(chosen)),
                        "semivariance": gamma,
                    }
                )
    return pd.DataFrame(rows)


def _feature_residual(fields: pd.DataFrame, aligned: pd.DataFrame) -> pd.DataFrame:
    """Spearman of the prior residual with each auxiliary channel, on test plugs."""
    rows: List[dict] = []
    merged = fields[fields["split"] == "test"].merge(
        aligned[["plug_index"] + list(CHANNELS)],
        on="plug_index",
        how="left",
        validate="many_to_one",
    )
    merged["residual"] = merged["y_true"] - merged["prior_pred"]
    for fold_id in range(3):
        for seed in SEEDS:
            block = merged[(merged["fold_id"] == fold_id) & (merged["seed"] == seed)]
            for feature in CHANNELS:
                rows.append(
                    {
                        "fold_id": fold_id,
                        "seed": int(seed),
                        "feature": feature,
                        "spearman": _spearman(
                            block["residual"].to_numpy(dtype=np.float64),
                            block[feature].to_numpy(dtype=np.float64),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _write_figures(
    shifts: pd.DataFrame,
    fields: pd.DataFrame,
    transfer: pd.DataFrame,
    variogram: pd.DataFrame,
    pairs: pd.DataFrame,
) -> None:
    """Exploratory figures. They do not choose a method."""
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 3.8))
    for feature in CHANNELS:
        part = shifts[(shifts["feature"] == feature) & (shifts["split"] == "test")]
        axis.plot(part["fold_id"], part["standardized_wasserstein"], marker="o", label=feature)
    axis.set_xlabel("held-out block")
    axis.set_ylabel("training-standardized Wasserstein distance")
    axis.set_xticks([0, 1, 2])
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "feature_shift.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.0), sharex=False)
    sample = fields[(fields["split"] == "test") & (fields["seed"] == 7)]
    for axis, fold_id in zip(axes, range(3)):
        block = sample[sample["fold_id"] == fold_id].sort_values("depth_m")
        axis.plot(block["depth_m"], block["y_true"] - block["prior_pred"], linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_ylabel("residual")
        axis.set_title("block {0}, seed 7".format(fold_id))
    axes[-1].set_xlabel("depth (m)")
    figure.tight_layout()
    figure.savefig(figure_dir / "prior_residual_depth.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    centers = variogram.groupby(["fold_id", "lag_left_m"], as_index=False)["semivariance"].mean()
    for fold_id, part in centers.groupby("fold_id"):
        axis.plot(part["lag_left_m"], part["semivariance"], marker="o", label="block {0}".format(int(fold_id)))
    axis.set_xlabel("lag lower edge (m)")
    axis.set_ylabel("mean semivariance")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "prior_residual_variogram.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    test = transfer[transfer["split"] == "test"]
    for axis, rho in zip(axes, RHOS):
        part = test[np.isclose(test["rho"], rho)]
        summary = part.groupby("fold_id")[["prior_rmse", "nearest_rmse", "interp_rmse", "direct_rmse", "clp_rmse"]].mean()
        offset = np.arange(summary.shape[0])
        width = 0.15
        names = ["prior_rmse", "nearest_rmse", "interp_rmse", "direct_rmse", "clp_rmse"]
        labels = ["prior", "nearest residual", "interp residual", "direct", "CLP"]
        for index, name in enumerate(names):
            axis.bar(offset + (index - 2) * width, summary[name], width=width, label=labels[index])
        axis.set_xticks(offset)
        axis.set_xticklabels(["0", "1", "2"])
        axis.set_title("rho {0:.2f}".format(rho))
        axis.set_xlabel("block")
    axes[0].set_ylabel("mean reserved RMSE")
    axes[0].legend(fontsize=6, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "reserved_rmse_by_block.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharex=True, sharey=True)
    for axis, rho in zip(axes, RHOS):
        part = pairs[(pairs["split"] == "test") & np.isclose(pairs["rho"], rho)]
        for fold_id, sample_pairs in part.groupby("fold_id"):
            axis.scatter(
                sample_pairs["nearest_observed_residual"],
                sample_pairs["reserved_residual"],
                s=8,
                alpha=0.35,
                label="block {0:.0f}".format(fold_id),
            )
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.axvline(0.0, color="black", linewidth=0.6)
        axis.set_title("rho {0:.2f}".format(rho))
        axis.set_xlabel("nearest observed residual")
    axes[0].set_ylabel("reserved residual")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "observed_versus_reserved_residual.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    for role, marker in (("val", "o"), ("test", "s")):
        part = transfer[transfer["split"] == role]
        gain = part["prior_rmse"] - part["interp_rmse"]
        axis.scatter(part["fold_id"] + (0.0 if role == "val" else 0.08), gain, marker=marker, label=role)
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xticks([0, 1, 2])
    axis.set_xlabel("block")
    axis.set_ylabel("prior RMSE minus interpolated-residual RMSE")
    axis.legend(fontsize=8, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "validation_versus_test_residual_gain.png", dpi=120)
    plt.close(figure)


def _write_environment(prediction_gap: float) -> None:
    """Record versions and the reproduction gap."""
    import scipy
    import sklearn

    lines = [
        "python {0}".format(sys.version.replace("\n", " ")),
        "platform {0}".format(platform.platform()),
        "torch {0}".format(torch.__version__),
        "numpy {0}".format(np.__version__),
        "pandas {0}".format(pd.__version__),
        "scipy {0}".format(scipy.__version__),
        "sklearn {0}".format(sklearn.__version__),
        "base_sha {0}".format(BASE_SHA),
        "max_abs_prior_prediction_gap {0:.6g}".format(prediction_gap),
    ]
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the read-only checks, rebuild G(z0), and score fixed residual rules."""
    aligned, split, masks, pred, cells, config = _load_frozen()
    saved_check = _check_saved_scores(pred, cells)
    shifts = _distribution_shift(aligned, split)
    correlations = _correlation_shift(aligned, split)
    direct = _direct_field(aligned, split)
    fields = _prior_fields(aligned, split)
    prediction_gap = _match_saved_prior(fields, pred)
    print("prior_prediction_max_abs", prediction_gap, flush=True)
    transfer, pairs = _transfer_table(fields, direct, masks, pred)
    variogram = _variogram(fields)
    feature_link = _feature_residual(fields, aligned)
    # Direct regression on the stored corrected-protocol plugs, same formula.
    stored_direct = pd.read_csv(CORRECTED / "tables" / "detail_by_plug.csv")
    stored_direct = stored_direct[stored_direct["method"] == "direct_regression"].copy()
    stored_direct["plug_index"] = stored_direct["plug_index"].astype(int)
    stored_direct["fold_id"] = stored_direct["fold_id"].astype(int)
    stored_direct["seed"] = stored_direct["seed"].astype(int)
    direct_gap = stored_direct.merge(
        direct[direct["split"] == "test"],
        on=["fold_id", "plug_index"],
        how="inner",
    )
    direct_abs = float(np.max(np.abs(direct_gap["direct_pred"] - direct_gap["y_pred"])))
    if direct_abs > PREDICTION_TOLERANCE:
        raise RuntimeError("Recomputed direct regression differs from the stored table by {0}.".format(direct_abs))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    shifts.to_csv(OUT / "tables" / "distribution_shift.csv", index=False)
    correlations.to_csv(OUT / "tables" / "correlation_shift.csv", index=False)
    saved_check.to_csv(OUT / "tables" / "stored_score_check.csv", index=False)
    fields.to_csv(OUT / "tables" / "prior_field.csv", index=False)
    direct.to_csv(OUT / "tables" / "direct_field.csv", index=False)
    transfer.to_csv(OUT / "tables" / "transfer_by_cell.csv", index=False)
    pairs.to_csv(OUT / "tables" / "residual_pairs.csv", index=False)
    variogram.to_csv(OUT / "tables" / "variogram.csv", index=False)
    feature_link.to_csv(OUT / "tables" / "residual_versus_channel.csv", index=False)
    transfer.groupby(["split", "fold_id", "rho"], as_index=False).mean(numeric_only=True).to_csv(
        OUT / "tables" / "summary_by_block.csv",
        index=False,
    )
    _write_figures(shifts, fields, transfer, variogram, pairs)
    _write_environment(prediction_gap)
    payload = {
        "base_sha": BASE_SHA,
        "split_sha256": config["split_sha256"],
        "masks_sha256": config["masks_sha256"],
        "max_abs_prior_prediction_gap": prediction_gap,
        "max_abs_direct_prediction_gap": direct_abs,
        "residual_rules": ["nearest observed residual", "depth interpolation of observed residual"],
        "rules_tuned_on_test": False,
        "prior_field_depends_on_mask": False,
    }
    (OUT / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "logs" / "run_log.txt").write_text(
        "prior_prediction_max_abs {0:.6g}\ndirect_prediction_max_abs {1:.6g}\n".format(prediction_gap, direct_abs),
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Prior transfer diagnostic",
                "",
                "Well 9-BRSA-1254-RJS.",
                "Run: python scripts/brsa1254/audit_prior_transfer.py",
                "G(z0) is rebuilt with the frozen training rule and checked against Investigation 22.",
                "Nearest and interpolated residual corrections are fixed rules.",
                "See REPORT.md after the scientific reading of this run.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("saved", OUT)


if __name__ == "__main__":
    main()
