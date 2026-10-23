#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contiguous permeability-gap diagnostic for well 9-BRSA-1254-RJS.

One gap is placed in each validation and test block. Its position depends on
the seed, the block, and the observed fraction. It does not depend on
permeability. Lambda is chosen on the validation gap only.

The script first rebuilds one frozen random-mask cell. The contiguous-gap
scores are written only after that cell matches the stored predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    RHOS,
    SEEDS,
    U_COLUMNS,
    WINDOW_LEN,
    observation_count,
)
from run_lambda_extension import _assert_grids, _predict, _score_prediction  # noqa: E402
from run_plug_holdout import (  # noqa: E402
    _fit_clp,
    _interp_depth,
    _rmse,
    _roles,
    _stack_windows,
    _window_starts,
)

OUT = OUT_DIR / "contiguous_gap_diagnostic"
BASE_SHA = "6e98a85d5c66e57410803eada2a88e63bbe8f338"
REPRODUCTION_TOLERANCE = 1.0e-4
SPLIT_CODE = {"val": 11, "test": 29}
CHECK_FOLD = 0
CHECK_SEED = 7
CHECK_RHO = 0.25


def _sha256(path: Path) -> str:
    """Hash one frozen input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gap_kind(indices: np.ndarray, scored: np.ndarray) -> str:
    """Tell an internal gap from a gap that starts or ends at the block edge."""
    touches_start = int(scored[0]) == int(indices[0])
    touches_end = int(scored[-1]) == int(indices[-1])
    if touches_start and touches_end:
        raise RuntimeError("A gap cannot touch both block edges while one plug is observed.")
    if touches_start:
        return "edge_shallow"
    if touches_end:
        return "edge_deep"
    return "interior"


def contiguous_mask(
    indices: np.ndarray,
    rho: float,
    seed: int,
    fold: int,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """Place one missing interval. Permeability is not an argument."""
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    if ordered.size < 2:
        raise ValueError("A block needs at least two plugs.")
    expected = np.arange(int(ordered[0]), int(ordered[-1]) + 1, dtype=np.int64)
    if not np.array_equal(ordered, expected):
        raise ValueError("Contiguous gap requires contiguous plug indices.")
    n_plugs = int(ordered.shape[0])
    n_observed = observation_count(n_plugs, float(rho))
    n_missing = n_plugs - n_observed
    if n_observed < 1 or n_missing < 1:
        raise ValueError("Observation count leaves no gap or no measurement.")
    if split not in SPLIT_CODE:
        raise ValueError("Unknown split {0}.".format(split))
    stream = np.random.SeedSequence(
        [int(seed), int(fold), int(round(float(rho) * 100.0)), int(SPLIT_CODE[split])]
    )
    offset = int(np.random.default_rng(stream).integers(0, n_observed + 1))
    scored = ordered[offset : offset + n_missing]
    observed = np.setdiff1d(ordered, scored)
    if int(observed.shape[0]) != n_observed or int(scored.shape[0]) != n_missing:
        raise RuntimeError("Gap counts do not match the observation rule.")
    if np.intersect1d(observed, scored).size != 0:
        raise RuntimeError("Observed and scored plugs overlap.")
    return observed, scored, offset, _gap_kind(ordered, scored)


def _empty_window_count(starts: Sequence[int], observed: np.ndarray) -> int:
    """Windows whose eight plugs contain no permeability measurement."""
    observed_set = set(int(value) for value in observed.tolist())
    empty = 0
    for start in starts:
        covered = range(int(start), int(start) + WINDOW_LEN)
        if not any(plug in observed_set for plug in covered):
            empty += 1
    return int(empty)


def _reachable(starts: Sequence[int], observed: np.ndarray, plug: int) -> bool:
    """True when some window covering the plug also covers an observation."""
    observed_set = set(int(value) for value in observed.tolist())
    for start in starts:
        origin = int(start)
        if origin <= int(plug) < origin + WINDOW_LEN:
            covered = range(origin, origin + WINDOW_LEN)
            if any(position in observed_set for position in covered):
                return True
    return False


def _fit_direct(
    u_matrix: np.ndarray,
    y_vector: np.ndarray,
    train_index: np.ndarray,
) -> Tuple[StandardScaler, LinearRegression]:
    """Same training-plug regression as the corrected protocol."""
    scaler = StandardScaler().fit(u_matrix[train_index])
    model = LinearRegression()
    model.fit(scaler.transform(u_matrix[train_index]), y_vector[train_index])
    return scaler, model


def _direct_at(
    scaler: StandardScaler,
    model: LinearRegression,
    u_matrix: np.ndarray,
    plugs: np.ndarray,
) -> Dict[int, float]:
    """Predict log10(k) from the four logs. The mask is not an input."""
    values = np.asarray(model.predict(scaler.transform(u_matrix[plugs])), dtype=np.float64)
    return {int(plug): float(value) for plug, value in zip(plugs.tolist(), values.tolist())}


def _interpolate(
    depth: np.ndarray,
    y_vector: np.ndarray,
    observed: np.ndarray,
    scored: np.ndarray,
) -> Dict[int, float]:
    """Linear interpolation in meters. Ends hold the nearest measured value."""
    values = _interp_depth(depth[observed], y_vector[observed], depth[scored])
    return {int(plug): float(value) for plug, value in zip(scored.tolist(), values.tolist())}


def _ordered(prediction: Dict[int, float], scored: np.ndarray) -> np.ndarray:
    """Predictions in the depth order of the reserved plugs."""
    return np.asarray([prediction[int(plug)] for plug in scored.tolist()], dtype=np.float64)


def _bias(y_vector: np.ndarray, scored: np.ndarray, prediction: Dict[int, float]) -> float:
    """Mean of true minus predicted on the reserved plugs."""
    gap = y_vector[scored] - _ordered(prediction, scored)
    return float(np.mean(gap))


def _mae(y_vector: np.ndarray, scored: np.ndarray, prediction: Dict[int, float]) -> float:
    """Mean absolute error on the reserved plugs."""
    gap = y_vector[scored] - _ordered(prediction, scored)
    return float(np.mean(np.abs(gap)))


def _make_config() -> Config:
    """Frozen decoder settings and the extended penalty grid."""
    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_device = "cpu"
    cfg.measurement_noise_std = 0.0
    cfg.csgm_lambda_grid = list(_assert_grids())
    return cfg


def _load_frozen() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Load the plug table, the split, and the original random masks."""
    aligned_path = OUT_DIR / "tables" / "plug_log_aligned.csv"
    split_path = OUT_DIR / "tables" / "split_by_plug.csv"
    masks_path = OUT_DIR / "tables" / "masks.csv"
    config_path = OUT_DIR / "restart_oracle_diagnostic" / "config.json"
    for path in (aligned_path, split_path, masks_path, config_path):
        if not path.is_file():
            raise FileNotFoundError("Missing frozen input {0}.".format(path))
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    if _sha256(split_path) != str(frozen["split_sha256"]):
        raise RuntimeError("Frozen split hash mismatch.")
    if _sha256(masks_path) != str(frozen["masks_sha256"]):
        raise RuntimeError("Frozen mask hash mismatch.")
    aligned = pd.read_csv(aligned_path)
    split = pd.read_csv(split_path)
    masks = pd.read_csv(masks_path)
    if int(aligned.shape[0]) != 161:
        raise RuntimeError("Expected 161 plugs.")
    aligned = aligned.sort_values("plug_index")
    identity = np.arange(161, dtype=np.int64)
    if not np.array_equal(aligned["plug_index"].to_numpy(dtype=np.int64), identity):
        raise RuntimeError("Plug indices are not 0..160 in depth order.")
    for frame in (split, masks):
        frame["fold_id"] = frame["fold_id"].astype(int)
    masks["seed"] = masks["seed"].astype(int)
    return aligned, split, masks, frozen


def _partitions(split: pd.DataFrame, fold_id: int) -> Dict[str, np.ndarray]:
    """Train, validation, and test plug indices for one block."""
    part = split[split["fold_id"] == fold_id]
    selected = {}
    for role in ("train", "val", "test"):
        selected[role] = np.sort(part.loc[part["split"] == role, "plug_index"].to_numpy(dtype=np.int64))
        if selected[role].size < 1:
            raise RuntimeError("Fold {0} has an empty {1} split.".format(fold_id, role))
    return selected


def _audit_geometry(split: pd.DataFrame, depth: np.ndarray) -> pd.DataFrame:
    """Check every contiguous mask before any model is trained."""
    rows: List[dict] = []
    for fold_id in range(3):
        parts = _partitions(split, fold_id)
        if int(parts["val"].shape[0]) != 16:
            raise RuntimeError("Validation must contain 16 plugs.")
        if int(parts["test"].shape[0]) not in (53, 54):
            raise RuntimeError("Unexpected test block size.")
        for seed in SEEDS:
            for rho in RHOS:
                for role in ("val", "test"):
                    observed, scored, offset, kind = contiguous_mask(
                        parts[role], float(rho), int(seed), fold_id, role
                    )
                    starts = _window_starts(parts[role])
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": int(seed),
                            "rho": float(rho),
                            "split": role,
                            "n_block": int(parts[role].shape[0]),
                            "n_observed": int(observed.shape[0]),
                            "n_scored": int(scored.shape[0]),
                            "offset": int(offset),
                            "gap_kind": kind,
                            "gap_span_m": float(depth[int(scored[-1])] - depth[int(scored[0])]),
                            "n_windows": int(len(starts)),
                            "n_empty_windows": _empty_window_count(starts, observed),
                            "n_unreachable_plugs": int(
                                sum(
                                    1
                                    for plug in scored.tolist()
                                    if not _reachable(starts, observed, int(plug))
                                )
                            ),
                        }
                    )
    geometry = pd.DataFrame(rows)
    if int(geometry.shape[0]) != 36:
        raise RuntimeError("Expected 36 validation and test gaps.")
    return geometry


def _reproduce_reference(
    aligned: pd.DataFrame,
    split: pd.DataFrame,
    masks: pd.DataFrame,
    cfg: Config,
) -> Dict[str, float]:
    """Rebuild one stored random-mask cell before the new masks are scored."""
    depth = aligned["depth_m"].to_numpy(dtype=np.float64)
    y_vector = aligned["log10_k"].to_numpy(dtype=np.float64)
    u_matrix = aligned.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    parts = _partitions(split, CHECK_FOLD)
    starts = {role: _window_starts(parts[role]) for role in parts}
    windows = {role: _stack_windows(u_matrix, y_vector, starts[role]) for role in parts}
    torch.manual_seed(0)
    ae, prior, mean, scale = _fit_clp(windows["train"][0], windows["train"][1], CHECK_SEED, cfg)
    roles = _roles(masks, CHECK_FOLD, CHECK_SEED, CHECK_RHO, "test")
    observed = set(int(value) for value in roles["observed"].tolist())
    scored = roles["scored"]
    stored = pd.read_csv(OUT_DIR / "lambda_extension" / "tables" / "detail_by_plug.csv")
    stored["plug_index"] = stored["plug_index"].astype(int)
    stored["fold_id"] = stored["fold_id"].astype(int)
    stored["seed"] = stored["seed"].astype(int)
    cell = stored[
        (stored["fold_id"] == CHECK_FOLD)
        & (stored["seed"] == CHECK_SEED)
        & np.isclose(stored["rho"], CHECK_RHO)
    ]
    extended = cell[cell["method"] == "clp_extended"]
    if int(extended.shape[0]) < 1:
        raise RuntimeError("Stored extended CLP rows are missing.")
    parameter = extended["parameter"].to_numpy(dtype=np.float64)
    if not np.allclose(parameter, parameter[0]):
        raise RuntimeError("Stored extended lambda is not unique.")
    lam = float(parameter[0])
    x_test, y_test = windows["test"]
    prior_pred = _predict(
        ae, prior, x_test, y_test, starts["test"], observed, scored, mean, scale, cfg, CHECK_SEED, None
    )
    clp_pred = _predict(
        ae, prior, x_test, y_test, starts["test"], observed, scored, mean, scale, cfg, CHECK_SEED, lam
    )
    scaler, model = _fit_direct(u_matrix, y_vector, parts["train"])
    direct_pred = _direct_at(scaler, model, u_matrix, scored)
    corrected = pd.read_csv(OUT_DIR / "corrected_protocol" / "tables" / "detail_by_plug.csv")
    corrected["plug_index"] = corrected["plug_index"].astype(int)
    corrected["fold_id"] = corrected["fold_id"].astype(int)
    corrected["seed"] = corrected["seed"].astype(int)
    direct_stored = corrected[
        (corrected["fold_id"] == CHECK_FOLD)
        & (corrected["seed"] == CHECK_SEED)
        & np.isclose(corrected["rho"], CHECK_RHO)
        & (corrected["method"] == "direct_regression")
    ]
    gaps = {
        "prior_only": _prediction_gap(prior_pred, cell[cell["method"] == "prior_only"]),
        "clp_extended": _prediction_gap(clp_pred, extended),
        "direct_regression": _prediction_gap(direct_pred, direct_stored),
    }
    for name, gap in gaps.items():
        if gap > REPRODUCTION_TOLERANCE:
            raise RuntimeError("Reference {0} differs by {1}.".format(name, gap))
    print(
        "reference_ok prior {0:.3e} clp {1:.3e} direct {2:.3e}".format(
            gaps["prior_only"], gaps["clp_extended"], gaps["direct_regression"]
        ),
        flush=True,
    )
    del depth
    return gaps


def _prediction_gap(prediction: Dict[int, float], stored: pd.DataFrame) -> float:
    """Maximum absolute difference against one stored method."""
    if int(stored.shape[0]) != len(prediction):
        raise RuntimeError("Stored prediction count does not match the scored plugs.")
    merged = stored.copy()
    merged["rebuilt"] = merged["plug_index"].map(prediction)
    if merged["rebuilt"].isna().any():
        raise RuntimeError("A stored plug is absent from the rebuilt prediction.")
    gap = np.abs(merged["rebuilt"].to_numpy(dtype=np.float64) - merged["y_pred"].to_numpy(dtype=np.float64))
    return float(np.max(gap))


def _select_lambda(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_val: np.ndarray,
    y_val: np.ndarray,
    starts: Sequence[int],
    observed: np.ndarray,
    scored: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
    cfg: Config,
    seed: int,
    y_vector: np.ndarray,
) -> Tuple[Optional[float], float, List[dict]]:
    """Walk no_update first, then the extended grid. A tie keeps the earlier candidate."""
    observed_set = set(int(value) for value in observed.tolist())
    choices: List[Optional[float]] = [None]
    choices.extend(float(value) for value in cfg.csgm_lambda_grid)
    best: Optional[float] = None
    best_score = float("inf")
    rows: List[dict] = []
    for lam in choices:
        prediction = _predict(
            ae, prior, x_val, y_val, starts, observed_set, scored, mean, scale, cfg, int(seed), lam
        )
        score = _score_prediction(prediction, scored, y_vector)
        rows.append(
            {
                "lambda": "no_update" if lam is None else float(lam),
                "validation_rmse": float(score),
            }
        )
        if score < best_score:
            best = lam
            best_score = float(score)
    return best, best_score, rows


def _lambda_label(lam: Optional[float]) -> str:
    """Store the validation choice without using it as a test decision."""
    if lam is None:
        return "no_update"
    return "{0:.8g}".format(float(lam))


def _region_tables(predictions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """RMSE on the influence rim, the unreachable interior, and distance bins."""
    frame = predictions.copy()
    region_rows: List[dict] = []
    for keys, part in frame.groupby(["fold_id", "seed", "rho", "method", "coverage"], sort=True):
        region_rows.append(
            {
                "fold_id": int(keys[0]),
                "seed": int(keys[1]),
                "rho": float(keys[2]),
                "method": str(keys[3]),
                "coverage": str(keys[4]),
                "n_plugs": int(part.shape[0]),
                "rmse": _rmse(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
                "mae": float(np.mean(np.abs(part["y_true"] - part["y_pred"]))),
            }
        )
    edges = (0.0, 1.0, 2.0, 4.0, 8.0, 1.0e9)
    distance_rows: List[dict] = []
    for keys, part in frame.groupby(["fold_id", "rho", "method"], sort=True):
        for left, right in zip(edges[:-1], edges[1:]):
            chosen = part[
                (part["distance_to_observation_m"] >= left) & (part["distance_to_observation_m"] < right)
            ]
            if int(chosen.shape[0]) < 1:
                score = float("nan")
                mae = float("nan")
            else:
                score = _rmse(chosen["y_true"].to_numpy(), chosen["y_pred"].to_numpy())
                mae = float(np.mean(np.abs(chosen["y_true"] - chosen["y_pred"])))
            distance_rows.append(
                {
                    "fold_id": int(keys[0]),
                    "rho": float(keys[1]),
                    "method": str(keys[2]),
                    "distance_left_m": float(left),
                    "distance_right_m": float(right if right < 1.0e8 else -1.0),
                    "n_plugs": int(chosen.shape[0]),
                    "rmse": score,
                    "mae": mae,
                }
            )
    return pd.DataFrame(region_rows), pd.DataFrame(distance_rows)


def _random_reference() -> pd.DataFrame:
    """Frozen random-mask scores. They stay in their own rows."""
    extended = pd.read_csv(OUT_DIR / "lambda_extension" / "tables" / "selection_by_cell.csv")
    direct = pd.read_csv(OUT_DIR / "corrected_protocol" / "tables" / "detail_by_plug.csv")
    direct = direct[direct["method"] == "direct_regression"].copy()
    rows: List[dict] = []
    for keys, part in direct.groupby(["fold_id", "seed", "rho"], sort=True):
        rows.append(
            {
                "mask_family": "random",
                "fold_id": int(keys[0]),
                "seed": int(keys[1]),
                "rho": float(keys[2]),
                "direct_rmse": _rmse(part["y_true"].to_numpy(), part["y_pred"].to_numpy()),
            }
        )
    reference = pd.DataFrame(rows)
    extended = extended.copy()
    extended["fold_id"] = extended["fold_id"].astype(int)
    extended["seed"] = extended["seed"].astype(int)
    reference = reference.merge(
        extended[["fold_id", "seed", "rho", "prior_test_rmse", "extended_test_rmse"]],
        on=["fold_id", "seed", "rho"],
        how="left",
        validate="one_to_one",
    )
    reference = reference.rename(columns={"extended_test_rmse": "clp_rmse", "prior_test_rmse": "prior_rmse"})
    return reference


def _write_figures(scores: pd.DataFrame, predictions: pd.DataFrame, geometry: pd.DataFrame) -> None:
    """Cell-level figures. They do not choose a method."""
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = ["prior_only", "clp", "interpolation", "direct_regression"]
    labels = ["prior", "CLP", "interpolation", "direct"]
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for axis, rho in zip(axes, RHOS):
        part = scores[np.isclose(scores["rho"], rho)]
        summary = part.groupby(["fold_id", "method"])["test_rmse"].mean().unstack("method")
        offset = np.arange(summary.shape[0])
        width = 0.18
        for index, method in enumerate(methods):
            axis.bar(offset + (index - 1.5) * width, summary[method], width=width, label=labels[index])
        axis.set_xticks(offset)
        axis.set_xticklabels(["0", "1", "2"])
        axis.set_title("rho {0:.2f}".format(rho))
        axis.set_xlabel("block")
    axes[0].set_ylabel("mean reserved RMSE")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "rmse_by_block.png", dpi=120)
    plt.close(figure)

    wide = scores.pivot_table(index=["fold_id", "seed", "rho"], columns="method", values="test_rmse")
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for fold_id, part in wide.groupby("fold_id"):
        axis.scatter(
            np.full(part.shape[0], int(fold_id)),
            part["clp"] - part["prior_only"],
            label="CLP - prior" if int(fold_id) == 0 else None,
            marker="o",
        )
        axis.scatter(
            np.full(part.shape[0], int(fold_id) + 0.15),
            part["clp"] - part["interpolation"],
            label="CLP - interpolation" if int(fold_id) == 0 else None,
            marker="s",
        )
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xticks([0, 1, 2])
    axis.set_xlabel("block")
    axis.set_ylabel("paired RMSE difference")
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "paired_differences.png", dpi=120)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    bins = predictions.copy()
    bins["bin_m"] = pd.cut(
        bins["distance_to_observation_m"],
        [0.0, 1.0, 2.0, 4.0, 8.0, 100.0],
        right=False,
        include_lowest=True,
    )
    for method, label in zip(methods, labels):
        part = bins[bins["method"] == method]
        grouped = part.groupby("bin_m", observed=False)[["y_true", "y_pred"]]
        centers = []
        values = []
        for index, sample in grouped:
            if int(sample.shape[0]) < 1:
                continue
            centers.append(float(index.left))
            values.append(_rmse(sample["y_true"].to_numpy(), sample["y_pred"].to_numpy()))
        axis.plot(centers, values, marker="o", label=label)
    axis.set_xlabel("distance to nearest observation, bin edge (m)")
    axis.set_ylabel("RMSE in bin")
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "error_versus_distance.png", dpi=120)
    plt.close(figure)

    test_geometry = geometry[geometry["split"] == "test"]
    deltas = wide.reset_index()
    deltas["delta_prior"] = deltas["clp"] - deltas["prior_only"]
    merged = deltas.merge(test_geometry, on=["fold_id", "seed", "rho"], how="left")
    figure, axis = plt.subplots(figsize=(6.2, 3.8))
    for kind, part in merged.groupby("gap_kind"):
        axis.scatter(part["n_empty_windows"], part["delta_prior"], label=str(kind))
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.set_xlabel("test windows with no observation")
    axis.set_ylabel("CLP RMSE minus prior RMSE")
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "empty_windows_versus_delta.png", dpi=120)
    plt.close(figure)


def _write_environment(reference_gaps: Dict[str, float]) -> None:
    """Record versions and the random-mask reproduction gap."""
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
    ]
    for name, gap in reference_gaps.items():
        lines.append("reference_max_abs_{0} {1:.6g}".format(name, gap))
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Check the frozen cell, then score one contiguous gap per block and seed."""
    aligned, split, masks, frozen = _load_frozen()
    y_vector = aligned["log10_k"].to_numpy(dtype=np.float64)
    depth = aligned["depth_m"].to_numpy(dtype=np.float64)
    u_matrix = aligned.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(u_matrix).all() or not np.isfinite(y_vector).all() or not np.isfinite(depth).all():
        raise RuntimeError("Aligned values must be finite.")
    if np.any(np.diff(depth) <= 0.0):
        raise RuntimeError("Depth is not strictly increasing with plug index.")
    geometry = _audit_geometry(split, depth)
    print("geometry_ok", int(geometry.shape[0]), "gaps", flush=True)
    cfg = _make_config()
    reference_gaps = _reproduce_reference(aligned, split, masks, cfg)

    score_rows: List[dict] = []
    mask_rows: List[dict] = []
    selection_rows: List[dict] = []
    prediction_rows: List[dict] = []
    torch.manual_seed(0)
    for fold_id in range(3):
        parts = _partitions(split, fold_id)
        starts = {role: _window_starts(parts[role]) for role in parts}
        windows = {role: _stack_windows(u_matrix, y_vector, starts[role]) for role in parts}
        scaler, regressor = _fit_direct(u_matrix, y_vector, parts["train"])
        print("fold", fold_id, "train_windows", len(starts["train"]), flush=True)
        for seed in SEEDS:
            ae, prior, mean, scale = _fit_clp(windows["train"][0], windows["train"][1], int(seed), cfg)
            for rho in RHOS:
                placed = {}
                for role in ("val", "test"):
                    observed, scored, offset, kind = contiguous_mask(
                        parts[role], float(rho), int(seed), fold_id, role
                    )
                    placed[role] = (observed, scored, offset, kind)
                    for plug in observed.tolist():
                        mask_rows.append(
                            {
                                "fold_id": fold_id,
                                "seed": int(seed),
                                "rho": float(rho),
                                "split": role,
                                "plug_index": int(plug),
                                "role": "observed",
                            }
                        )
                    for plug in scored.tolist():
                        mask_rows.append(
                            {
                                "fold_id": fold_id,
                                "seed": int(seed),
                                "rho": float(rho),
                                "split": role,
                                "plug_index": int(plug),
                                "role": "scored",
                            }
                        )
                val_observed, val_scored, _val_offset, _val_kind = placed["val"]
                chosen, val_rmse, candidates = _select_lambda(
                    ae,
                    prior,
                    windows["val"][0],
                    windows["val"][1],
                    starts["val"],
                    val_observed,
                    val_scored,
                    mean,
                    scale,
                    cfg,
                    int(seed),
                    y_vector,
                )
                for candidate in candidates:
                    selection_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": int(seed),
                            "rho": float(rho),
                            "lambda": candidate["lambda"],
                            "validation_rmse": candidate["validation_rmse"],
                            "selected": int(
                                (chosen is None and candidate["lambda"] == "no_update")
                                or (chosen is not None and candidate["lambda"] == float(chosen))
                            ),
                        }
                    )
                test_observed, test_scored, test_offset, test_kind = placed["test"]
                observed_set = set(int(value) for value in test_observed.tolist())
                x_test, y_test = windows["test"]
                predictions = {
                    "prior_only": _predict(
                        ae, prior, x_test, y_test, starts["test"], observed_set, test_scored,
                        mean, scale, cfg, int(seed), None,
                    ),
                    "clp": _predict(
                        ae, prior, x_test, y_test, starts["test"], observed_set, test_scored,
                        mean, scale, cfg, int(seed), chosen,
                    ),
                    "interpolation": _interpolate(depth, y_vector, test_observed, test_scored),
                    "direct_regression": _direct_at(scaler, regressor, u_matrix, test_scored),
                }
                unreachable = [
                    int(plug)
                    for plug in test_scored.tolist()
                    if not _reachable(starts["test"], test_observed, int(plug))
                ]
                unreachable_gap = 0.0
                if unreachable and chosen is not None:
                    unreachable_gap = max(
                        abs(predictions["clp"][plug] - predictions["prior_only"][plug]) for plug in unreachable
                    )
                    # Subset decodes differ from a full-batch decode at float32 scale.
                    if unreachable_gap > 1.0e-5:
                        raise RuntimeError(
                            "CLP left the prior on a window with no observation by {0}.".format(unreachable_gap)
                        )
                span_min = float(np.min(depth[test_observed]))
                span_max = float(np.max(depth[test_observed]))
                endpoints = {int(test_scored[0]), int(test_scored[-1])}
                empty_windows = _empty_window_count(starts["test"], test_observed)
                for method, prediction in predictions.items():
                    score_rows.append(
                        {
                            "fold_id": fold_id,
                            "seed": int(seed),
                            "rho": float(rho),
                            "method": method,
                            "test_rmse": _score_prediction(prediction, test_scored, y_vector),
                            "test_mae": _mae(y_vector, test_scored, prediction),
                            "test_bias": _bias(y_vector, test_scored, prediction),
                            "chosen_lambda": _lambda_label(chosen),
                            "validation_clp_rmse": float(val_rmse),
                            "gap_n_plugs": int(test_scored.shape[0]),
                            "gap_span_m": float(depth[int(test_scored[-1])] - depth[int(test_scored[0])]),
                            "gap_offset": int(test_offset),
                            "gap_kind": test_kind,
                            "n_windows": int(len(starts["test"])),
                            "n_empty_windows": int(empty_windows),
                            "n_unreachable_plugs": int(len(unreachable)),
                            "max_abs_clp_prior_unreachable": float(unreachable_gap),
                        }
                    )
                    for plug in test_scored.tolist():
                        plug_id = int(plug)
                        prediction_rows.append(
                            {
                                "fold_id": fold_id,
                                "seed": int(seed),
                                "rho": float(rho),
                                "method": method,
                                "plug_index": plug_id,
                                "depth_m": float(depth[plug_id]),
                                "y_true": float(y_vector[plug_id]),
                                "y_pred": float(prediction[plug_id]),
                                "distance_to_observation_m": float(
                                    np.min(np.abs(depth[test_observed] - depth[plug_id]))
                                ),
                                "coverage": "unreachable" if plug_id in unreachable else "rim",
                                "gap_endpoint": int(plug_id in endpoints),
                                "outside_observed_span": int(
                                    float(depth[plug_id]) < span_min or float(depth[plug_id]) > span_max
                                ),
                            }
                        )
                print(
                    "fold {0} seed {1} rho {2:.2f} lambda {3} val {4:.5f} empty {5}".format(
                        fold_id, int(seed), float(rho), _lambda_label(chosen), float(val_rmse), empty_windows
                    ),
                    flush=True,
                )

    scores = pd.DataFrame(score_rows)
    if int(scores.shape[0]) != 72:
        raise RuntimeError("Expected 72 method rows.")
    predictions = pd.DataFrame(prediction_rows)
    region, distance = _region_tables(predictions)
    reference = _random_reference()
    wide = scores.pivot_table(
        index=["fold_id", "seed", "rho", "gap_kind", "gap_span_m", "n_empty_windows", "n_unreachable_plugs"],
        columns="method",
        values="test_rmse",
    ).reset_index()
    wide["clp_minus_prior"] = wide["clp"] - wide["prior_only"]
    wide["clp_minus_interpolation"] = wide["clp"] - wide["interpolation"]
    wide["clp_minus_direct"] = wide["clp"] - wide["direct_regression"]
    OUT.mkdir(parents=True, exist_ok=True)
    table_dir = OUT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    scores.to_csv(table_dir / "scores_by_cell.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(table_dir / "contiguous_masks.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(table_dir / "validation_selection.csv", index=False)
    predictions.to_csv(table_dir / "predictions_by_plug.csv", index=False)
    geometry.to_csv(table_dir / "gap_geometry.csv", index=False)
    wide.to_csv(table_dir / "paired_differences.csv", index=False)
    region.to_csv(table_dir / "error_by_coverage.csv", index=False)
    distance.to_csv(table_dir / "error_by_distance.csv", index=False)
    reference.to_csv(table_dir / "reference_random_masks.csv", index=False)
    scores.groupby(["rho", "method"], as_index=False).agg(
        mean_rmse=("test_rmse", "mean"),
        sd_rmse=("test_rmse", "std"),
        mean_mae=("test_mae", "mean"),
        mean_bias=("test_bias", "mean"),
    ).to_csv(table_dir / "summary.csv", index=False)
    scores.groupby(["fold_id", "rho", "method"], as_index=False).agg(
        mean_rmse=("test_rmse", "mean"),
        mean_mae=("test_mae", "mean"),
        mean_bias=("test_bias", "mean"),
    ).to_csv(table_dir / "summary_by_block.csv", index=False)
    _write_figures(scores, predictions, geometry)
    _write_environment(reference_gaps)
    payload = {
        "base_sha": BASE_SHA,
        "split_sha256": frozen["split_sha256"],
        "random_masks_sha256": frozen["masks_sha256"],
        "gap_rule": "one interval; offset from SeedSequence(seed, fold, round(100*rho), split code)",
        "permeability_used_to_place_gap": False,
        "lambda_selected_on": "validation contiguous gap",
        "tie_rule": "no_update first, then increasing lambda; strict improvement",
        "lambda_grid": [float(value) for value in cfg.csgm_lambda_grid],
        "reference_cell": {"fold_id": CHECK_FOLD, "seed": CHECK_SEED, "rho": CHECK_RHO},
        "reference_max_abs": reference_gaps,
        "random_mask_scores_mixed_into_summary": False,
    }
    (OUT / "config.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (OUT / "logs" / "run_log.txt").write_text(
        "\n".join(
            "reference_max_abs_{0} {1:.6g}".format(name, gap) for name, gap in reference_gaps.items()
        )
        + "\n",
        encoding="utf-8",
    )
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Contiguous permeability-gap diagnostic",
                "",
                "Well 9-BRSA-1254-RJS.",
                "Run: python scripts/brsa1254/run_contiguous_gap_audit.py",
                "One gap per block, seed, and observed fraction.",
                "Lambda is selected on the validation gap.",
                "Random-mask scores are stored separately and are not averaged with these scores.",
                "See REPORT.md.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(scores.groupby(["rho", "method"])["test_rmse"].mean().to_string())
    print("saved", OUT)


if __name__ == "__main__":
    main()
