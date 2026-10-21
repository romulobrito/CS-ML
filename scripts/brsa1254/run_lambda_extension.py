#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extend the CLP penalty grid for the well 9-BRSA-1254-RJS plug holdout.

The corrected comparator tables stay in place. This script retrains only the
conditional decoder and prior, then chooses the penalty on validation plugs.
ASCII only.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

import csgm_m2_module as csgm
from sir_cs_pipeline_optimized import Config

from common import (  # noqa: E402
    AE_EPOCHS,
    CSGM_ITERS,
    CSGM_RESTARTS,
    HIDDEN_DIM,
    LAMBDA_GRID,
    LATENT_DIM,
    OUT_DIR,
    U_COLUMNS,
    WINDOW_LEN,
)
from run_plug_holdout import (  # noqa: E402
    _fit_clp,
    _mean_at_plugs,
    _recover_grouped,
    _rmse,
    _roles,
    _stack_windows,
    _window_starts,
)

# Values added after the corrected run hit the old maximum in 10 of 18 cells.
EXTRA_LAMBDAS: Tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
EXPECTED_ORIGINAL: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
EXTENSION_DIR = OUT_DIR / "lambda_extension"
NO_UPDATE = "no_update"
LAMBDA_KIND = "lambda"


def _assert_grids() -> Tuple[float, ...]:
    """Keep the previous grid and append the larger penalties in increasing order."""
    if tuple(float(value) for value in LAMBDA_GRID) != EXPECTED_ORIGINAL:
        raise RuntimeError("The stored lambda grid is not the grid diagnosed at 0.1.")
    extended = tuple(EXPECTED_ORIGINAL) + tuple(EXTRA_LAMBDAS)
    if list(extended) != sorted(extended):
        raise RuntimeError("The extended lambda grid is not sorted.")
    if any(float(value) <= 0.0 for value in extended):
        raise RuntimeError("Every lambda must be positive.")
    return extended


def _select_strict(names: Sequence[str], scores: Sequence[float]) -> Tuple[str, float]:
    """Keep the earliest candidate unless a later score is strictly smaller.

    Callers put no_update first when it is allowed, then lambdas from small
    to large. A tie therefore stays at no update, or at the smaller lambda.
    """
    if len(names) != len(scores) or len(names) < 1:
        raise ValueError("Selection needs one score per candidate.")
    best_name = str(names[0])
    best_score = float(scores[0])
    if not math.isfinite(best_score):
        raise ValueError("Validation RMSE is not finite.")
    for name, score in zip(names[1:], scores[1:]):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError("Validation RMSE is not finite.")
        if value < best_score:
            best_name = str(name)
            best_score = value
    return best_name, best_score


def _lambda_name(lam: float) -> str:
    """Stable label for one positive penalty."""
    return "lambda_{0:.8g}".format(float(lam))


def _choice_fields(name: str) -> Tuple[str, float]:
    """Split a selected label into a rule name and a penalty value."""
    if name == NO_UPDATE:
        return NO_UPDATE, float("nan")
    prefix = "lambda_"
    if not name.startswith(prefix):
        raise ValueError("Unknown candidate label {0}.".format(name))
    return LAMBDA_KIND, float(name[len(prefix) :])


def _write_protocol(extended: Sequence[float]) -> None:
    """Record the extension rule before any score from this run is written."""
    EXTENSION_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "Lambda extension for the corrected 9-BRSA-1254-RJS plug holdout.",
        "Blocks, masks, seeds, and the corrected_protocol tables are unchanged.",
        "Other comparators are not refit.",
        "This file is written before the extension is scored.",
        "Motivation, recorded because the corrected test tables had already been examined:",
        "Validation selected lambda 0.1 in 10 of 18 fold-seed-rho cells.",
        "0.1 was the largest value on the grid {0}.".format(
            ", ".join("{0:.8g}".format(float(value)) for value in EXPECTED_ORIGINAL)
        ),
        "On those tables the mean test RMSE of G(z0) was lower than full CLP at both fractions.",
        "That boundary count does not by itself prove that a larger lambda improves the test score.",
        "Selection in this extension uses validation RMSE only.",
        "The test score is not an input to the choice.",
        "Added lambda values: {0}.".format(
            ", ".join("{0:.8g}".format(float(value)) for value in EXTRA_LAMBDAS)
        ),
        "Full search list: {0}.".format(
            ", ".join("{0:.8g}".format(float(value)) for value in extended)
        ),
        "no_update decodes G(z0) and does not run recovery. It is a validation candidate.",
        "A later candidate replaces the incumbent only when its validation RMSE is strictly smaller.",
        "The extended search starts at no_update, then walks lambdas from smallest to largest.",
        "A tie therefore stays at no_update. A tie between lambdas keeps the smaller lambda.",
        "clp_original_grid repeats the previous seven-value rule on the decoder trained here.",
        "clp_extended is the validation choice on the extended list plus no_update.",
        "prior_only scores G(z0) on the reserved test plugs and is not chosen by the test score.",
        "Primary aggregate: unweighted mean of the nine fold-by-seed RMSE values.",
        "If validation chooses no_update, the test rows for clp_extended use that rule.",
        "",
    ]
    (EXTENSION_DIR / "PROTOCOL_LAMBDA_EXTENSION.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def _saved_lambdas() -> pd.DataFrame:
    """Penalty chosen in the corrected run, one row per fold, seed, and fraction."""
    path = OUT_DIR / "corrected_protocol" / "tables" / "detail_by_plug.csv"
    if not path.is_file():
        raise FileNotFoundError("The corrected plug table is required as the reference.")
    detail = pd.read_csv(path)
    clp = detail[detail["method"] == "clp"]
    rows: List[Dict[str, float]] = []
    for keys, part in clp.groupby(["fold_id", "seed", "rho"], sort=True):
        values = part["parameter"].to_numpy(dtype=np.float64)
        if values.size < 1 or not np.allclose(values, values[0]):
            raise RuntimeError("The corrected run stored more than one lambda in a cell.")
        rows.append(
            {
                "fold_id": float(keys[0]),
                "seed": float(keys[1]),
                "rho": float(keys[2]),
                "saved_lambda": float(values[0]),
            }
        )
    saved = pd.DataFrame(rows)
    if int(saved.shape[0]) != 18:
        raise RuntimeError("Expected 18 corrected CLP cells, found {0}.".format(saved.shape[0]))
    return saved


def _predict(
    ae: csgm.TinyAE,
    prior: csgm.RidgePrior,
    x_windows: np.ndarray,
    y_windows: np.ndarray,
    starts: Sequence[int],
    observed_plugs: set,
    scored: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    cfg: Config,
    seed: int,
    lam: Optional[float],
) -> Dict[int, float]:
    """Map reserved plugs to one prediction. None means decode the prior."""
    if lam is None:
        latent = np.asarray(prior.predict(x_windows), dtype=np.float64)
        windows = csgm.decode_latent_prior(ae, latent, y_mean, y_scale, "cpu")
    else:
        windows = _recover_grouped(
            ae,
            prior,
            x_windows,
            y_windows,
            starts,
            observed_plugs,
            y_mean,
            y_scale,
            float(lam),
            cfg,
            int(seed),
        )
    return _mean_at_plugs(starts, np.asarray(windows, dtype=np.float64), scored)


def _score_prediction(
    prediction: Dict[int, float],
    scored: np.ndarray,
    y_vector: np.ndarray,
) -> float:
    """RMSE on the reserved plugs, in the depth order of the mask."""
    ordered = np.asarray(
        [prediction[int(plug)] for plug in scored.tolist()],
        dtype=np.float64,
    )
    return _rmse(y_vector[scored], ordered)


def _append_method(
    rows: List[Dict[str, object]],
    fold_id: int,
    seed: int,
    rho: float,
    method: str,
    choice: str,
    lam: float,
    depth: np.ndarray,
    y_vector: np.ndarray,
    prediction: Dict[int, float],
) -> None:
    """Store one reserved plug per row."""
    for plug, y_hat in prediction.items():
        rows.append(
            {
                "fold_id": float(fold_id),
                "seed": float(seed),
                "rho": float(rho),
                "method": method,
                "choice": choice,
                "parameter": float(lam),
                "plug_index": float(plug),
                "depth_m": float(depth[int(plug)]),
                "y_true": float(y_vector[int(plug)]),
                "y_pred": float(y_hat),
            }
        )


def main() -> None:
    """Train the conditional model and score the validation-chosen penalty."""
    extended = _assert_grids()
    reference = OUT_DIR / "corrected_protocol" / "PROTOCOL_COMPARATORS.txt"
    if not reference.is_file():
        raise FileNotFoundError("Run the corrected comparator script before this extension.")
    config = json.loads((OUT_DIR / "config_frozen.json").read_text(encoding="utf-8"))
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
    saved = _saved_lambdas()
    _write_protocol(extended)

    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_lambda_grid = [float(value) for value in extended]
    cfg.csgm_device = "cpu"
    cfg.measurement_noise_std = 0.0
    torch.manual_seed(0)

    candidate_rows: List[Dict[str, object]] = []
    selection_rows: List[Dict[str, object]] = []
    detail_rows: List[Dict[str, object]] = []
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
                names: List[str] = [NO_UPDATE]
                scores: List[float] = []
                prior_val = _predict(
                    ae,
                    prior,
                    x_val,
                    y_val,
                    val_starts,
                    val_obs,
                    val_roles["scored"],
                    y_mean,
                    y_scale,
                    cfg,
                    int(seed),
                    None,
                )
                scores.append(_score_prediction(prior_val, val_roles["scored"], y_vector))
                candidate_rows.append(
                    {
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "candidate": NO_UPDATE,
                        "parameter": float("nan"),
                        "val_rmse": scores[0],
                    }
                )
                for lam in extended:
                    label = _lambda_name(float(lam))
                    names.append(label)
                    predicted = _predict(
                        ae,
                        prior,
                        x_val,
                        y_val,
                        val_starts,
                        val_obs,
                        val_roles["scored"],
                        y_mean,
                        y_scale,
                        cfg,
                        int(seed),
                        float(lam),
                    )
                    score = _score_prediction(predicted, val_roles["scored"], y_vector)
                    scores.append(score)
                    candidate_rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "candidate": label,
                            "parameter": float(lam),
                            "val_rmse": score,
                        }
                    )
                # The choice is fixed here, before any test prediction.
                extended_name, extended_val = _select_strict(names, scores)
                original_names = [_lambda_name(float(lam)) for lam in EXPECTED_ORIGINAL]
                original_scores = [
                    float(scores[names.index(label)])
                    for label in original_names
                ]
                original_name, original_val = _select_strict(original_names, original_scores)
                extended_choice, extended_lam = _choice_fields(extended_name)
                original_choice, original_lam = _choice_fields(original_name)
                if original_choice != LAMBDA_KIND:
                    raise RuntimeError("The original grid has no no_update candidate.")

                def _test_for(lam: Optional[float]) -> Dict[int, float]:
                    return _predict(
                        ae,
                        prior,
                        x_test,
                        y_test,
                        test_starts,
                        test_obs,
                        test_roles["scored"],
                        y_mean,
                        y_scale,
                        cfg,
                        int(seed),
                        lam,
                    )

                prior_test = _test_for(None)
                original_test = _test_for(float(original_lam))
                if extended_choice == NO_UPDATE:
                    extended_test = prior_test
                    extended_parameter = float("nan")
                else:
                    extended_parameter = float(extended_lam)
                    if math.isclose(extended_parameter, float(original_lam)):
                        extended_test = original_test
                    else:
                        extended_test = _test_for(extended_parameter)
                prior_test_rmse = _score_prediction(prior_test, test_roles["scored"], y_vector)
                original_test_rmse = _score_prediction(original_test, test_roles["scored"], y_vector)
                extended_test_rmse = _score_prediction(extended_test, test_roles["scored"], y_vector)
                if extended_choice == NO_UPDATE and not math.isclose(extended_test_rmse, prior_test_rmse):
                    raise RuntimeError("no_update test score differs from G(z0).")
                saved_row = saved[
                    (saved["fold_id"] == float(fold_id))
                    & (saved["seed"] == float(seed))
                    & np.isclose(saved["rho"], float(rho))
                ]
                if int(saved_row.shape[0]) != 1:
                    raise RuntimeError("Saved lambda is missing for this cell.")
                saved_lambda = float(saved_row["saved_lambda"].iloc[0])
                selection_rows.append(
                    {
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "extended_choice": extended_choice,
                        "extended_lambda": extended_parameter,
                        "extended_val_rmse": float(extended_val),
                        "extended_test_rmse": float(extended_test_rmse),
                        "extended_at_max_lambda": float(
                            extended_choice == LAMBDA_KIND and math.isclose(extended_parameter, 100.0)
                        ),
                        "original_lambda": float(original_lam),
                        "original_val_rmse": float(original_val),
                        "original_test_rmse": float(original_test_rmse),
                        "original_at_0_1": float(math.isclose(float(original_lam), 0.1)),
                        "prior_val_rmse": float(scores[0]),
                        "prior_test_rmse": float(prior_test_rmse),
                        "saved_lambda": saved_lambda,
                        "matches_saved_lambda": float(math.isclose(float(original_lam), saved_lambda)),
                    }
                )
                _append_method(
                    detail_rows,
                    fold_id,
                    int(seed),
                    float(rho),
                    "prior_only",
                    NO_UPDATE,
                    float("nan"),
                    depth,
                    y_vector,
                    prior_test,
                )
                _append_method(
                    detail_rows,
                    fold_id,
                    int(seed),
                    float(rho),
                    "clp_original_grid",
                    LAMBDA_KIND,
                    float(original_lam),
                    depth,
                    y_vector,
                    original_test,
                )
                _append_method(
                    detail_rows,
                    fold_id,
                    int(seed),
                    float(rho),
                    "clp_extended",
                    extended_choice,
                    extended_parameter,
                    depth,
                    y_vector,
                    extended_test,
                )
                print(
                    "  seed",
                    int(seed),
                    "rho",
                    float(rho),
                    "choice",
                    extended_choice,
                    "lambda",
                    extended_parameter,
                    "val",
                    round(float(extended_val), 5),
                    flush=True,
                )

    detail = pd.DataFrame(detail_rows)
    summary_rows: List[Dict[str, object]] = []
    for keys, part in detail.groupby(["fold_id", "seed", "rho", "method"], sort=True):
        summary_rows.append(
            {
                "fold_id": float(keys[0]),
                "seed": float(keys[1]),
                "rho": float(keys[2]),
                "method": str(keys[3]),
                "choice": str(part["choice"].iloc[0]),
                "parameter": float(part["parameter"].iloc[0]),
                "n_scored": float(part.shape[0]),
                "rmse": _rmse(
                    part["y_true"].to_numpy(dtype=np.float64),
                    part["y_pred"].to_numpy(dtype=np.float64),
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    primary_rows: List[Dict[str, object]] = []
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
    fold_rows: List[Dict[str, object]] = []
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
    table_dir = EXTENSION_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(candidate_rows).to_csv(table_dir / "validation_candidates.csv", index=False)
    selection = pd.DataFrame(selection_rows)
    selection.to_csv(table_dir / "selection_by_cell.csv", index=False)
    detail.to_csv(table_dir / "detail_by_plug.csv", index=False)
    summary.to_csv(table_dir / "summary_by_seed.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(table_dir / "summary_by_fold.csv", index=False)
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(table_dir / "summary_primary.csv", index=False)
    print(selection.to_string(index=False))
    print(primary.to_string(index=False))
    print("wrote", table_dir)


if __name__ == "__main__":
    main()
