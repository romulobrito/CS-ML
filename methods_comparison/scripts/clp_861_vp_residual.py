#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLP-CSGM residual Vp after Gassmann physics (Well 861 MOGNO).

Hybrid: Vp_hybrid = Vp_gassmann + G(z_hat)

Methods (depth-block OOF, stitched windows):
  clp_ridge_prior_m0   -- Ridge h(u) -> z0, m=0 at test (comparable to RF OOF)
  clp_zero_residual_m0 -- encode(zero residual) -> z0, m=0 (variant B, trust Gassmann)
  clp_plug_sparse_b    -- encode(zero) + sparse b at 10 plug depths (Etapa 1f style)

Planning: methods_comparison/planning/etapa3b_clp_csgm_vp_residual_poco861.md
ASCII-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from auddys_smoke_direct_ub import build_windows  # noqa: E402

import csgm_m2_module as csgm  # noqa: E402
import real_well_f03 as rwf  # noqa: E402
from clp_861_plug_fixed_runner import (  # noqa: E402
    _make_b_row,
    build_coordinate_M,
    classify_windows_by_rows,
    plug_local_indices,
    resolve_torch_device,
)
from clp_861_protocol import load_plug_measurement_rows, plug_row_indices_unique  # noqa: E402
from ml_861_data import (  # noqa: E402
    DEPTH_COL,
    LEAKAGE_FOR_VP_RESIDUAL,
    LOG_FEATURE_COLUMNS,
    RESIDUAL_VP_FEATURE_EXTRA,
    RESIDUAL_VP_TARGET,
    depth_block_splits,
)
from sir_cs_pipeline_optimized import Config, apply_config_profile  # noqa: E402

VP_CLP_METHOD_RIDGE_PRIOR = "clp_ridge_prior_m0"
VP_CLP_METHOD_ZERO_RESIDUAL = "clp_zero_residual_m0"
VP_CLP_METHOD_PLUG_SPARSE = "clp_plug_sparse_b"

VP_CLP_METHODS_ALL: Tuple[str, ...] = (
    VP_CLP_METHOD_RIDGE_PRIOR,
    VP_CLP_METHOD_ZERO_RESIDUAL,
    VP_CLP_METHOD_PLUG_SPARSE,
)


@dataclass(frozen=True)
class VpClpRunConfig:
    """Knobs for CLP-CSGM Vp residual depth-block CV."""

    window_len: int = 16
    step: int = 1
    seed: int = 7
    n_depth_blocks: int = 3
    measurement_noise_std: float = 0.01
    csgm_latent_dim: int = 16
    csgm_hidden_dim: int = 128
    csgm_ae_epochs: int = 200
    csgm_iters: int = 400
    csgm_restarts: int = 3
    csgm_opt_lr: float = 0.05
    csgm_ridge_alpha: float = 1.0
    csgm_lambda_grid: Tuple[float, ...] = (
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    )
    methods: Tuple[str, ...] = VP_CLP_METHODS_ALL
    device: Optional[str] = None


@dataclass
class VpClpFoldResult:
    """One fold OOF stitched residual profile for each method."""

    fold_id: int
    test_idx: np.ndarray
    depth_min_m: float
    depth_max_m: float
    delta_oof_by_method: Dict[str, np.ndarray]
    lambda_by_method: Dict[str, float]


@dataclass
class VpClpCvResult:
    """Full depth-block CV output."""

    n_rows: int
    window_len: int
    methods: Tuple[str, ...]
    delta_oof: Dict[str, np.ndarray]
    fold_results: List[VpClpFoldResult]
    selected_lambda: Dict[str, float]


def vp_clp_u_channels() -> Tuple[str, ...]:
    """Dense u channels: wireline logs plus Vp Gassmann, minus leakage columns."""
    cols = tuple(LOG_FEATURE_COLUMNS) + tuple(RESIDUAL_VP_FEATURE_EXTRA)
    leak = set(LEAKAGE_FOR_VP_RESIDUAL)
    return tuple(c for c in cols if c not in leak)


def encode_zero_residual_z0(
    ae: csgm.TinyAE,
    y_scaler: StandardScaler,
    device: str,
) -> np.ndarray:
    """Latent prior for zero residual (trust Gassmann baseline)."""
    n_out = int(y_scaler.mean_.shape[0])
    zeros = np.zeros((1, n_out), dtype=np.float64)
    zeros_n = y_scaler.transform(zeros)
    return csgm.encode_y(ae, zeros_n, device=device)


def _build_csgm_cfg(cfg_run: VpClpRunConfig, n_u: int, window_len: int) -> Config:
    """Map VpClpRunConfig to sir_cs Config."""
    cfg = Config(config_profile="real_well_f03_direct_ub")
    apply_config_profile(cfg)
    cfg.n_output = int(window_len)
    cfg.p_input = int(n_u)
    cfg.measurement_noise_std = float(cfg_run.measurement_noise_std)
    cfg.csgm_latent_dim = int(cfg_run.csgm_latent_dim)
    cfg.csgm_hidden_dim = int(cfg_run.csgm_hidden_dim)
    cfg.csgm_ae_epochs = int(cfg_run.csgm_ae_epochs)
    cfg.csgm_iters = int(cfg_run.csgm_iters)
    cfg.csgm_restarts = int(cfg_run.csgm_restarts)
    cfg.csgm_opt_lr = float(cfg_run.csgm_opt_lr)
    cfg.csgm_ridge_alpha = float(cfg_run.csgm_ridge_alpha)
    cfg.csgm_prior_type = "ridge"
    cfg.csgm_lambda_grid = list(cfg_run.csgm_lambda_grid)
    return cfg


def _train_fold_generator(
    cfg_run: VpClpRunConfig,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    device: str,
    seed: int,
) -> Tuple[csgm.TinyAE, StandardScaler, np.ndarray, np.ndarray, csgm.RidgePrior]:
    """Train AE on residual windows and Ridge prior h(u) -> z."""
    y_scaler = StandardScaler()
    y_tr_n = y_scaler.fit_transform(y_tr)
    y_mean = y_scaler.mean_.astype(np.float64)
    y_scale = y_scaler.scale_.astype(np.float64)

    cfg = _build_csgm_cfg(cfg_run, int(x_tr.shape[1]), int(y_tr.shape[1]))
    ae = csgm.train_ae_generator(y_tr_n, cfg, seed=int(seed), device=device)
    z_tr = csgm.encode_y(ae, y_tr_n, device=device)
    ridge_prior = csgm.RidgePrior(
        alpha=float(cfg_run.csgm_ridge_alpha),
        random_state=int(seed),
    )
    ridge_prior.fit(x_tr, z_tr)
    return ae, y_scaler, y_mean, y_scale, ridge_prior


def _predict_delta_ridge_prior_m0(
    ae: csgm.TinyAE,
    ridge_prior: csgm.RidgePrior,
    x_window: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    device: str,
) -> np.ndarray:
    """Decode residual window from Ridge latent prior only (m=0)."""
    z0 = ridge_prior.predict(x_window.reshape(1, -1))
    return csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device=device)[0]


def _predict_delta_zero_m0(
    ae: csgm.TinyAE,
    y_scaler: StandardScaler,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    device: str,
) -> np.ndarray:
    """Decode zero-residual latent prior (m=0)."""
    z0 = encode_zero_residual_z0(ae, y_scaler, device)
    return csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device=device)[0]


def _predict_delta_plug_sparse(
    ae: csgm.TinyAE,
    y_true_row: np.ndarray,
    plug_local: Sequence[int],
    y_scaler: StandardScaler,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    noise_std: float,
    rng: np.random.Generator,
    seed: int,
    device: str,
    window_len: int,
    cfg: Config,
) -> np.ndarray:
    """Recover residual window with sparse plug observations (variant B)."""
    z0 = encode_zero_residual_z0(ae, y_scaler, device)
    loc = list(plug_local)
    M = build_coordinate_M(loc, int(window_len))
    if M.shape[0] == 0:
        return _predict_delta_zero_m0(ae, y_scaler, y_mean, y_scale, device)
    B = _make_b_row(np.asarray(y_true_row, dtype=np.float64), M, noise_std, rng)
    return csgm.csgm_recover_with_prior(
        ae=ae,
        M=M,
        B=B,
        z0_np=z0,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=float(lam),
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device=device,
        seed=int(seed),
    )[0]


def cal_local_indices(
    window_start: int,
    cal_rows: frozenset,
    window_len: int,
) -> List[int]:
    """Local window indices where global calibration rows fall."""
    start = int(window_start)
    l = int(window_len)
    out: List[int] = []
    for j in range(l):
        if (start + j) in cal_rows:
            out.append(j)
    return out


def select_lambda_sparse_b(
    ae: csgm.TinyAE,
    y_val: np.ndarray,
    starts_val: np.ndarray,
    cal_rows: frozenset,
    cfg_run: VpClpRunConfig,
    y_scaler: StandardScaler,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    seed: int,
    device: str,
    rng: np.random.Generator,
    window_len: int,
) -> Tuple[float, float]:
    """Grid search lambda on validation windows with sparse b at cal_rows."""
    cfg = _build_csgm_cfg(cfg_run, 1, window_len)
    best_lam = float(cfg_run.csgm_lambda_grid[0])
    best_score = float("inf")
    l = int(window_len)
    for lam in cfg_run.csgm_lambda_grid:
        preds: List[np.ndarray] = []
        for j in range(int(y_val.shape[0])):
            start_j = int(starts_val[j])
            loc = cal_local_indices(start_j, cal_rows, l)
            y_hat = _predict_delta_plug_sparse(
                ae,
                y_val[j],
                loc,
                y_scaler,
                y_mean,
                y_scale,
                float(lam),
                float(cfg_run.measurement_noise_std),
                rng,
                int(seed),
                device,
                l,
                cfg,
            )
            preds.append(y_hat)
        pred = np.stack(preds, axis=0)
        score = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        if score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam, best_score


def _select_lambda_plug_sparse(
    ae: csgm.TinyAE,
    y_val: np.ndarray,
    starts_val: np.ndarray,
    plug_rows: Sequence[int],
    cfg_run: VpClpRunConfig,
    y_scaler: StandardScaler,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    seed: int,
    device: str,
    rng: np.random.Generator,
    window_len: int,
) -> Tuple[float, float]:
    """Grid search lambda on validation residual windows (RMSE on delta)."""
    cfg = _build_csgm_cfg(cfg_run, 1, window_len)
    best_lam = float(cfg_run.csgm_lambda_grid[0])
    best_score = float("inf")
    l = int(window_len)
    for lam in cfg_run.csgm_lambda_grid:
        preds: List[np.ndarray] = []
        for j in range(int(y_val.shape[0])):
            start_j = int(starts_val[j])
            loc = plug_local_indices(start_j, plug_rows, l)
            y_hat = _predict_delta_plug_sparse(
                ae,
                y_val[j],
                loc,
                y_scaler,
                y_mean,
                y_scale,
                float(lam),
                float(cfg_run.measurement_noise_std),
                rng,
                int(seed),
                device,
                l,
                cfg,
            )
            preds.append(y_hat)
        pred = np.stack(preds, axis=0)
        score = float(np.sqrt(np.mean((pred - y_val) ** 2)))
        if score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam, best_score


def _stitch_window_predictions(
    y_pred_windows: np.ndarray,
    starts: np.ndarray,
    window_len: int,
    n_rows: int,
) -> np.ndarray:
    """Uniform-mean stitch of overlapping window predictions."""
    prof, _ = rwf.reconstruct_depth_profile(
        y_pred_windows,
        starts,
        int(window_len),
        int(n_rows),
        overlap_agg=rwf.PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
    )
    return prof


def run_vp_clp_depth_block_cv(
    df: pd.DataFrame,
    cfg_run: VpClpRunConfig,
) -> VpClpCvResult:
    """
  Depth-block CV for CLP-CSGM Vp residual methods.

  Returns stitched OOF residual profiles (length n_rows) per method.
  """
    channels = vp_clp_u_channels()
    target = RESIDUAL_VP_TARGET
    for col in channels + (target, "vp_gassmann_km_s"):
        if col not in df.columns:
            raise ValueError("Missing column for CLP Vp: {}".format(col))

    work = df.sort_values(DEPTH_COL).reset_index(drop=True)
    n_rows = int(work.shape[0])
    plug_rows = plug_row_indices_unique(load_plug_measurement_rows())

    x_all, y_all, starts = build_windows(
        work,
        channels,
        target,
        int(cfg_run.window_len),
        int(cfg_run.step),
    )
    folds = depth_block_splits(work, n_blocks=int(cfg_run.n_depth_blocks))
    device = resolve_torch_device(cfg_run.device)
    rng = np.random.default_rng(int(cfg_run.seed))

    methods = tuple(m for m in cfg_run.methods if m in VP_CLP_METHODS_ALL)
    if not methods:
        raise ValueError("No valid CLP Vp methods requested.")

    delta_oof: Dict[str, np.ndarray] = {
        m: np.full(n_rows, np.nan, dtype=np.float64) for m in methods
    }
    fold_results: List[VpClpFoldResult] = []
    selected_lambda: Dict[str, float] = {}

    for fold in folds:
        test_rows = fold.test_idx
        train_pool = fold.train_idx
        n_tr_pool = len(train_pool)
        n_va = max(1, int(round(0.2 * n_tr_pool)))
        val_rows = train_pool[-n_va:]
        train_rows = train_pool[:-n_va]

        tr_w, va_w, te_w = classify_windows_by_rows(
            starts,
            int(cfg_run.window_len),
            train_rows,
            val_rows,
            test_rows,
        )
        if not te_w or not tr_w:
            raise ValueError(
                "Fold {} empty windows (train={}, test={}).".format(
                    fold.fold_id, len(tr_w), len(te_w)
                )
            )

        x_tr = x_all[tr_w]
        y_tr = y_all[tr_w]
        x_va = x_all[va_w] if va_w else x_all[tr_w[-max(1, len(tr_w) // 5) :]]
        y_va = y_all[va_w] if va_w else y_all[tr_w[-max(1, len(tr_w) // 5) :]]
        starts_va = starts[va_w] if va_w else starts[tr_w[-max(1, len(tr_w) // 5) :]]
        y_te = y_all[te_w]
        starts_te = starts[te_w]

        ae, y_scaler, y_mean, y_scale, ridge_prior = _train_fold_generator(
            cfg_run,
            x_tr,
            y_tr,
            device,
            int(cfg_run.seed) + int(fold.fold_id),
        )

        fold_delta: Dict[str, np.ndarray] = {}
        fold_lam: Dict[str, float] = {}

        if VP_CLP_METHOD_RIDGE_PRIOR in methods:
            preds_ridge: List[np.ndarray] = []
            for j in range(int(y_te.shape[0])):
                wi = int(te_w[j])
                delta_hat = _predict_delta_ridge_prior_m0(
                    ae,
                    ridge_prior,
                    x_all[wi],
                    y_mean,
                    y_scale,
                    device,
                )
                preds_ridge.append(delta_hat)
            prof_ridge = _stitch_window_predictions(
                np.stack(preds_ridge, axis=0),
                starts_te,
                int(cfg_run.window_len),
                n_rows,
            )
            fold_delta[VP_CLP_METHOD_RIDGE_PRIOR] = prof_ridge
            fold_lam[VP_CLP_METHOD_RIDGE_PRIOR] = float("nan")

        if VP_CLP_METHOD_ZERO_RESIDUAL in methods:
            preds_zero: List[np.ndarray] = []
            for _j in range(int(y_te.shape[0])):
                delta_hat = _predict_delta_zero_m0(ae, y_scaler, y_mean, y_scale, device)
                preds_zero.append(delta_hat)
            prof_zero = _stitch_window_predictions(
                np.stack(preds_zero, axis=0),
                starts_te,
                int(cfg_run.window_len),
                n_rows,
            )
            fold_delta[VP_CLP_METHOD_ZERO_RESIDUAL] = prof_zero
            fold_lam[VP_CLP_METHOD_ZERO_RESIDUAL] = float("nan")

        if VP_CLP_METHOD_PLUG_SPARSE in methods:
            lam_plug, _val_score = _select_lambda_plug_sparse(
                ae,
                y_va,
                starts_va,
                plug_rows,
                cfg_run,
                y_scaler,
                y_mean,
                y_scale,
                int(cfg_run.seed) + int(fold.fold_id),
                device,
                rng,
                int(cfg_run.window_len),
            )
            cfg = _build_csgm_cfg(cfg_run, int(x_tr.shape[1]), int(cfg_run.window_len))
            preds_plug: List[np.ndarray] = []
            for j in range(int(y_te.shape[0])):
                start_j = int(starts_te[j])
                loc = plug_local_indices(start_j, plug_rows, int(cfg_run.window_len))
                y_hat = _predict_delta_plug_sparse(
                    ae,
                    y_te[j],
                    loc,
                    y_scaler,
                    y_mean,
                    y_scale,
                    lam_plug,
                    float(cfg_run.measurement_noise_std),
                    rng,
                    int(cfg_run.seed) + int(fold.fold_id),
                    device,
                    int(cfg_run.window_len),
                    cfg,
                )
                preds_plug.append(y_hat)
            prof_plug = _stitch_window_predictions(
                np.stack(preds_plug, axis=0),
                starts_te,
                int(cfg_run.window_len),
                n_rows,
            )
            fold_delta[VP_CLP_METHOD_PLUG_SPARSE] = prof_plug
            fold_lam[VP_CLP_METHOD_PLUG_SPARSE] = lam_plug
            selected_lambda[VP_CLP_METHOD_PLUG_SPARSE] = lam_plug

        test_idx = np.asarray(test_rows, dtype=np.int64)
        for method, prof in fold_delta.items():
            delta_oof[method][test_idx] = prof[test_idx]

        fold_results.append(
            VpClpFoldResult(
                fold_id=int(fold.fold_id),
                test_idx=test_idx,
                depth_min_m=float(fold.depth_min_m),
                depth_max_m=float(fold.depth_max_m),
                delta_oof_by_method=fold_delta,
                lambda_by_method=fold_lam,
            )
        )

    return VpClpCvResult(
        n_rows=n_rows,
        window_len=int(cfg_run.window_len),
        methods=methods,
        delta_oof=delta_oof,
        fold_results=fold_results,
        selected_lambda=selected_lambda,
    )


def smoke_vp_clp_config() -> VpClpRunConfig:
    """Fast settings for smoke tests."""
    return VpClpRunConfig(
        window_len=12,
        csgm_ae_epochs=15,
        csgm_iters=50,
        csgm_restarts=1,
        n_depth_blocks=2,
        methods=(VP_CLP_METHOD_RIDGE_PRIOR, VP_CLP_METHOD_ZERO_RESIDUAL),
    )
