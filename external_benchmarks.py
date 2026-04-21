#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
external_benchmarks.py

Sklearn-based sparse recovery baselines on the same (A, rhs) problems as
sir_cs_pipeline_optimized.py, for fair external comparison (Etapa 1 benchmark).

Scenarios (protocol labels):
    S1: hybrid residual CS, rhs = z = b - M y_ml, y_hat = y_ml + Psi @ alpha.
    S2: cs_only, rhs = b, y_hat = Psi @ alpha.
    S3: hybrid OMP with true sparsity level k = cfg.residual_k (oracle-k OMP).

ASCII-only source.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Union

import numpy as np
from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit
from sklearn.metrics import mean_absolute_error

from sir_cs_pipeline_optimized import (
    Config,
    evaluate_metric,
    relative_l2,
    rmse,
    select_regularization_for_cs_method,
    solve_sparse_alpha,
    support_f1,
)


def sklearn_lasso_alpha_grid(cfg: Config, m: int) -> List[float]:
    """
    Map pipeline FISTA-style grid to sklearn Lasso `alpha` scale.

    Pipeline FISTA minimizes 0.5 * ||A x - b||_2^2 + lam * ||x||_1.
    sklearn.Lasso minimizes (1 / (2 * n_samples)) * ||A x - b||_2^2 + alpha * ||x||_1
    with n_samples = m (rows of A). Equating L1 weights: alpha ~= lam / m.
    """
    return [float(lam) / float(max(m, 1)) for lam in cfg.l1_lambda_grid]


def default_lasso_max_iter(cfg: Config) -> int:
    """Sklearn coordinate descent may need more iterations than FISTA."""
    return int(max(4000, cfg.fista_max_iter * 10))


def fit_lasso_coeffs(
    A: np.ndarray,
    b: np.ndarray,
    alpha: float,
    max_iter: int,
) -> np.ndarray:
    """Return alpha_hat for one (A, b) pair."""
    m, _k = A.shape
    model = Lasso(
        alpha=float(alpha),
        max_iter=int(max_iter),
        tol=1e-4,
        fit_intercept=False,
        selection="random",
        random_state=0,
    )
    model.fit(A, b)
    return np.asarray(model.coef_, dtype=float).reshape(-1)


def select_best_sklearn_lasso_alpha(
    cfg: Config,
    A: np.ndarray,
    Psi: np.ndarray,
    y_sel: np.ndarray,
    ybg_sel: np.ndarray,
    b_sel: np.ndarray,
    z_sel: np.ndarray,
    scenario: Literal["S1_hybrid", "S2_cs_only"],
    alpha_grid: List[float],
    L_A: Optional[float],
) -> float:
    """Pick alpha minimizing cfg.model_selection_metric on the val subset."""
    if len(alpha_grid) == 0:
        raise ValueError("alpha_grid empty")
    best_alpha = float(alpha_grid[0])
    best_score = float("inf")
    n_samples = len(y_sel)
    for alpha in alpha_grid:
        preds = np.zeros_like(y_sel)
        for i in range(n_samples):
            if scenario == "S1_hybrid":
                rhs = z_sel[i]
            else:
                rhs = b_sel[i]
            alpha_hat = fit_lasso_coeffs(A, rhs, alpha, max_iter=default_lasso_max_iter(cfg))
            if scenario == "S1_hybrid":
                y_hat = ybg_sel[i] + Psi @ alpha_hat
            else:
                y_hat = Psi @ alpha_hat
            preds[i] = y_hat
        score = evaluate_metric(y_sel, preds, cfg.model_selection_metric)
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def fit_omp_coeffs(A: np.ndarray, b: np.ndarray, n_nonzero_coefs: int) -> np.ndarray:
    """Orthogonal matching pursuit; coefficient length = A.shape[1]."""
    k = int(max(1, min(n_nonzero_coefs, A.shape[1])))
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=False)
    omp.fit(A, b)
    return np.asarray(omp.coef_, dtype=float).reshape(-1)


def hybrid_fista_best_lambda(
    cfg: Config,
    A: np.ndarray,
    Psi: np.ndarray,
    y_sel: np.ndarray,
    ybg_sel: np.ndarray,
    b_sel: np.ndarray,
    z_sel: np.ndarray,
    L_A: Optional[float],
) -> float:
    """Same hybrid FISTA lambda selection as the main pipeline (reference)."""
    return float(
        select_regularization_for_cs_method(
            "hybrid_fista",
            "hybrid",
            "fista",
            cfg,
            A,
            Psi,
            y_sel,
            ybg_sel,
            b_sel,
            z_sel,
            None,
            L_A,
        )
    )


def hybrid_fista_predict_one(
    cfg: Config,
    A: np.ndarray,
    Psi: np.ndarray,
    y_ml: np.ndarray,
    z_i: np.ndarray,
    lam: float,
    L_A: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    """One test sample: hybrid FISTA alpha and y_hat."""
    alpha_hat = solve_sparse_alpha(
        A=A,
        b=z_i,
        cs_engine="fista",
        regularization=float(lam),
        weights=None,
        L_A=L_A,
        cfg=cfg,
        x0=None,
    )
    y_hat = y_ml + Psi @ alpha_hat
    return alpha_hat, y_hat


def per_sample_metrics_row(
    seed: int,
    measurement_ratio: float,
    method: str,
    sample_id: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha_true: np.ndarray,
    alpha_hat: np.ndarray,
    reg_value: float,
    cs_engine: str,
    m: int,
    support_f1_override: Optional[float] = None,
) -> dict[str, Union[float, int, str]]:
    sf: float
    if support_f1_override is not None:
        sf = float(support_f1_override)
    else:
        sf = support_f1(alpha_true, alpha_hat)
    return {
        "seed": seed,
        "measurement_ratio": measurement_ratio,
        "method": method,
        "sample_id": sample_id,
        "rmse": rmse(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "relative_l2": relative_l2(y_true, y_pred),
        "support_f1": sf,
        "lambda": float(reg_value),
        "cs_engine": cs_engine,
        "m": m,
    }
