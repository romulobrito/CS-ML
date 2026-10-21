#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rho subsample sweep: CLP sparse b vs RF sparse vs RF oracle (Vp residual, Well 861).

At each rho, a fraction rho of train/test rows per fold is used as calibration
points (known delta_vp). CLP uses b at those depths; RF sparse trains only on
calibration train rows. RF oracle uses full train (reference).

Planning: methods_comparison/planning/etapa3c_vp_rho_subsample_poco861.md
ASCII-only.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from auddys_smoke_direct_ub import build_windows  # noqa: E402

from clp_861_protocol import load_plug_measurement_rows, plug_row_indices_unique  # noqa: E402
from clp_861_plug_fixed_runner import classify_windows_by_rows  # noqa: E402
from clp_861_vp_residual import (  # noqa: E402
    VpClpRunConfig,
    _build_csgm_cfg,
    _predict_delta_plug_sparse,
    _predict_delta_zero_m0,
    _stitch_window_predictions,
    _train_fold_generator,
    cal_local_indices,
    resolve_torch_device,
    select_lambda_sparse_b,
    vp_clp_u_channels,
)
from ml_861_data import (  # noqa: E402
    RESIDUAL_VP_TARGET,
    XYBundle,
    build_residual_feature_columns,
    build_xy_from_columns,
    depth_block_splits,
)

METHOD_CLP_SPARSE = "clp_sparse_b"
METHOD_RF_SPARSE = "rf_sparse"
METHOD_RF_ORACLE = "rf_oracle"
METHOD_GASSMANN = "gassmann_physics"

DEFAULT_RHOS: Tuple[float, ...] = (
    0.0,
    0.1,
    10.0 / 87.0,
    0.2,
    0.3,
    0.5,
    0.7,
    1.0,
)


@dataclass(frozen=True)
class RhoMetricsRow:
    """One rho x method metrics row."""

    rho: float
    method: str
    repeat_id: int
    n_cal_train_mean: float
    n_cal_test_mean: float
    mape_vp_pct: float
    rmse_vp_km_s: float
    bias_vp_km_s: float
    lambda_clp: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rho": float(self.rho),
            "method": str(self.method),
            "repeat_id": int(self.repeat_id),
            "n_cal_train_mean": float(self.n_cal_train_mean),
            "n_cal_test_mean": float(self.n_cal_test_mean),
            "mape_vp_pct": float(self.mape_vp_pct),
            "rmse_vp_km_s": float(self.rmse_vp_km_s),
            "bias_vp_km_s": float(self.bias_vp_km_s),
            "lambda_clp": float(self.lambda_clp),
        }


def subsample_row_indices(
    row_indices: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random subsample of row indices at fraction rho (at least 1 if rho > 0)."""
    rows = np.asarray(row_indices, dtype=np.int64).ravel()
    if rows.size == 0:
        return np.array([], dtype=np.int64)
    if float(rho) <= 0.0:
        return np.array([], dtype=np.int64)
    if float(rho) >= 1.0:
        return rows.copy()
    k = max(1, int(round(float(rho) * float(rows.size))))
    k = min(k, int(rows.size))
    chosen = rng.choice(rows, size=k, replace=False)
    return np.sort(chosen.astype(np.int64, copy=False))


def fixed_row_indices(row_indices: np.ndarray, fixed: Sequence[int]) -> np.ndarray:
    """Intersection of fold rows with a fixed global calibration set."""
    row_set = set(int(r) for r in row_indices.tolist())
    pick = sorted(int(r) for r in fixed if int(r) in row_set)
    return np.asarray(pick, dtype=np.int64)


def vp_metrics_from_delta(
    vp_gassmann: np.ndarray,
    delta_pred: np.ndarray,
    vp_sonic: np.ndarray,
) -> Tuple[float, float, float]:
    """MAPE, RMSE, bias for Vp hybrid vs sonic."""
    vp_pred = vp_gassmann + delta_pred
    err = vp_pred - vp_sonic
    mape = float(np.mean(np.abs(err / vp_sonic)) * 100.0)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    return mape, rmse, bias


def _predict_clp_fold_test(
    ae,
    y_scaler,
    y_mean,
    y_scale,
    x_all: np.ndarray,
    y_te: np.ndarray,
    starts_te: np.ndarray,
    te_w: List[int],
    cal_test: frozenset,
    lam: float,
    cfg_run: VpClpRunConfig,
    rng: np.random.Generator,
    seed: int,
    device: str,
) -> np.ndarray:
    """Stitched delta profile on test windows with sparse b at cal_test."""
    cfg = _build_csgm_cfg(cfg_run, int(x_all.shape[1]), int(cfg_run.window_len))
    preds: List[np.ndarray] = []
    l = int(cfg_run.window_len)
    for j in range(int(y_te.shape[0])):
        wi = int(te_w[j])
        start_j = int(starts_te[j])
        loc = cal_local_indices(start_j, cal_test, l)
        if not loc:
            delta_hat = _predict_delta_zero_m0(ae, y_scaler, y_mean, y_scale, device)
        else:
            delta_hat = _predict_delta_plug_sparse(
                ae,
                y_te[j],
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
        preds.append(delta_hat)
    return np.stack(preds, axis=0)


def run_rf_oracle_oof(
    bundle: XYBundle,
    n_blocks: int,
    rf_n_estimators: int,
    random_state: int,
) -> np.ndarray:
    """RF depth-block OOF on full train rows (oracle reference)."""
    from run_861_ml_residual import evaluate_depth_blocks_oof

    def factory() -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=int(rf_n_estimators),
            random_state=int(random_state),
        )

    _summary, oof = evaluate_depth_blocks_oof(factory, bundle, n_blocks=n_blocks)
    return oof


def run_rho_sweep(
    df: pd.DataFrame,
    bundle: XYBundle,
    cfg_run: VpClpRunConfig,
    rhos: Sequence[float],
    rf_n_estimators: int = 200,
    n_repeats: int = 1,
    plug_rows: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Sweep rho with depth-block CV.

    Returns metrics table and oof delta arrays keyed by 'clp_rho_{rho}' etc.
    """
    channels = vp_clp_u_channels()
    target = RESIDUAL_VP_TARGET
    work = df.sort_values("Depth(m)").reset_index(drop=True)
    n_rows = int(work.shape[0])
    vp_gass = work["vp_gassmann_km_s"].to_numpy(dtype=np.float64)
    vp_sonic = work["vp_sonic_km_s"].to_numpy(dtype=np.float64)

    x_all, y_all, starts = build_windows(
        work,
        channels,
        target,
        int(cfg_run.window_len),
        int(cfg_run.step),
    )
    folds = depth_block_splits(work, n_blocks=int(cfg_run.n_depth_blocks))
    device = resolve_torch_device(cfg_run.device)

    oof_store: Dict[str, np.ndarray] = {}
    rows: List[RhoMetricsRow] = []

    rf_oracle = run_rf_oracle_oof(
        bundle,
        int(cfg_run.n_depth_blocks),
        int(rf_n_estimators),
        int(cfg_run.seed),
    )
    oof_store[METHOD_RF_ORACLE] = rf_oracle
    mape, rmse, bias = vp_metrics_from_delta(vp_gass, rf_oracle, vp_sonic)
    rows.append(
        RhoMetricsRow(
            rho=float("nan"),
            method=METHOD_RF_ORACLE,
            repeat_id=0,
            n_cal_train_mean=float("nan"),
            n_cal_test_mean=float("nan"),
            mape_vp_pct=mape,
            rmse_vp_km_s=rmse,
            bias_vp_km_s=bias,
            lambda_clp=float("nan"),
        )
    )

    zeros = np.zeros(n_rows, dtype=np.float64)
    oof_store[METHOD_GASSMANN] = zeros
    mape, rmse, bias = vp_metrics_from_delta(vp_gass, zeros, vp_sonic)
    rows.append(
        RhoMetricsRow(
            rho=0.0,
            method=METHOD_GASSMANN,
            repeat_id=0,
            n_cal_train_mean=0.0,
            n_cal_test_mean=0.0,
            mape_vp_pct=mape,
            rmse_vp_km_s=rmse,
            bias_vp_km_s=bias,
            lambda_clp=float("nan"),
        )
    )

    plug_list = list(plug_rows) if plug_rows is not None else []

    for rho in rhos:
        rho_f = float(rho)
        for repeat_id in range(int(n_repeats)):
            oof_clp = np.full(n_rows, np.nan, dtype=np.float64)
            oof_rf = np.full(n_rows, np.nan, dtype=np.float64)
            n_cal_tr: List[int] = []
            n_cal_te: List[int] = []
            lam_vals: List[float] = []

            for fold in folds:
                test_rows = np.asarray(fold.test_idx, dtype=np.int64)
                train_pool = np.asarray(fold.train_idx, dtype=np.int64)
                n_tr_pool = int(train_pool.size)
                n_va = max(1, int(round(0.2 * n_tr_pool)))
                val_rows = train_pool[-n_va:]
                train_rows = train_pool[:-n_va]

                fold_seed = (
                    int(cfg_run.seed)
                    + int(fold.fold_id) * 1000
                    + int(repeat_id) * 10000
                    + int(round(rho_f * 1000.0))
                )
                rng = np.random.default_rng(fold_seed)

                if plug_list and abs(rho_f - 10.0 / 87.0) < 1.0e-6:
                    cal_tr = fixed_row_indices(train_rows, plug_list)
                    cal_va = fixed_row_indices(val_rows, plug_list)
                    cal_te = fixed_row_indices(test_rows, plug_list)
                else:
                    cal_tr = subsample_row_indices(train_rows, rho_f, rng)
                    cal_va = subsample_row_indices(val_rows, rho_f, rng)
                    cal_te = subsample_row_indices(test_rows, rho_f, rng)

                cal_tr_set = frozenset(int(r) for r in cal_tr.tolist())
                cal_va_set = frozenset(int(r) for r in cal_va.tolist())
                cal_te_set = frozenset(int(r) for r in cal_te.tolist())
                n_cal_tr.append(int(cal_tr.size))
                n_cal_te.append(int(cal_te.size))

                tr_w, va_w, te_w = classify_windows_by_rows(
                    starts,
                    int(cfg_run.window_len),
                    train_rows,
                    val_rows,
                    test_rows,
                )
                if not te_w or not tr_w:
                    raise ValueError("Fold {} empty windows.".format(fold.fold_id))

                x_tr = x_all[tr_w]
                y_tr = y_all[tr_w]
                x_va = x_all[va_w] if va_w else x_all[tr_w[-max(1, len(tr_w) // 5) :]]
                y_va = y_all[va_w] if va_w else y_all[tr_w[-max(1, len(tr_w) // 5) :]]
                starts_va = starts[va_w] if va_w else starts[tr_w[-max(1, len(tr_w) // 5) :]]
                y_te = y_all[te_w]
                starts_te = starts[te_w]

                ae, y_scaler, y_mean, y_scale, _ridge = _train_fold_generator(
                    cfg_run,
                    x_tr,
                    y_tr,
                    device,
                    fold_seed,
                )

                if cal_va_set:
                    lam, _score = select_lambda_sparse_b(
                        ae,
                        y_va,
                        starts_va,
                        cal_va_set,
                        cfg_run,
                        y_scaler,
                        y_mean,
                        y_scale,
                        fold_seed,
                        device,
                        rng,
                        int(cfg_run.window_len),
                    )
                else:
                    lam = float(cfg_run.csgm_lambda_grid[0])
                lam_vals.append(lam)

                pred_windows = _predict_clp_fold_test(
                    ae,
                    y_scaler,
                    y_mean,
                    y_scale,
                    x_all,
                    y_te,
                    starts_te,
                    te_w,
                    cal_te_set,
                    lam,
                    cfg_run,
                    rng,
                    fold_seed,
                    device,
                )
                prof = _stitch_window_predictions(
                    pred_windows,
                    starts_te,
                    int(cfg_run.window_len),
                    n_rows,
                )
                oof_clp[test_rows] = prof[test_rows]

                if cal_tr.size >= 2:
                    rf = RandomForestRegressor(
                        n_estimators=int(rf_n_estimators),
                        random_state=int(cfg_run.seed),
                    )
                    x_cal = bundle.X[cal_tr]
                    y_cal = bundle.y[cal_tr]
                    rf.fit(x_cal, y_cal)
                    pred_rf = rf.predict(bundle.X[test_rows])
                elif cal_tr.size == 1:
                    rf = RandomForestRegressor(
                        n_estimators=int(rf_n_estimators),
                        random_state=int(cfg_run.seed),
                    )
                    rf.fit(bundle.X[cal_tr], bundle.y[cal_tr])
                    pred_rf = rf.predict(bundle.X[test_rows])
                else:
                    pred_rf = np.zeros(int(test_rows.size), dtype=np.float64)
                oof_rf[test_rows] = pred_rf

            lam_mean = float(np.mean(lam_vals)) if lam_vals else float("nan")
            key_clp = "clp_rho_{:.3f}_r{}".format(rho_f, repeat_id)
            key_rf = "rf_sparse_rho_{:.3f}_r{}".format(rho_f, repeat_id)
            oof_store[key_clp] = oof_clp
            oof_store[key_rf] = oof_rf

            mape_c, rmse_c, bias_c = vp_metrics_from_delta(vp_gass, oof_clp, vp_sonic)
            rows.append(
                RhoMetricsRow(
                    rho=rho_f,
                    method=METHOD_CLP_SPARSE,
                    repeat_id=int(repeat_id),
                    n_cal_train_mean=float(np.mean(n_cal_tr)),
                    n_cal_test_mean=float(np.mean(n_cal_te)),
                    mape_vp_pct=mape_c,
                    rmse_vp_km_s=rmse_c,
                    bias_vp_km_s=bias_c,
                    lambda_clp=lam_mean,
                )
            )
            mape_r, rmse_r, bias_r = vp_metrics_from_delta(vp_gass, oof_rf, vp_sonic)
            rows.append(
                RhoMetricsRow(
                    rho=rho_f,
                    method=METHOD_RF_SPARSE,
                    repeat_id=int(repeat_id),
                    n_cal_train_mean=float(np.mean(n_cal_tr)),
                    n_cal_test_mean=float(np.mean(n_cal_te)),
                    mape_vp_pct=mape_r,
                    rmse_vp_km_s=rmse_r,
                    bias_vp_km_s=bias_r,
                    lambda_clp=float("nan"),
                )
            )

    return pd.DataFrame([r.to_dict() for r in rows]), oof_store


def aggregate_rho_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Mean metrics per rho x method across repeats."""
    sweep = metrics_df[metrics_df["method"].isin((METHOD_CLP_SPARSE, METHOD_RF_SPARSE))].copy()
    if sweep.empty:
        return sweep
    agg = (
        sweep.groupby(["rho", "method"], as_index=False)
        .agg(
            mape_vp_pct=("mape_vp_pct", "mean"),
            rmse_vp_km_s=("rmse_vp_km_s", "mean"),
            bias_vp_km_s=("bias_vp_km_s", "mean"),
            n_cal_train_mean=("n_cal_train_mean", "mean"),
            n_cal_test_mean=("n_cal_test_mean", "mean"),
            lambda_clp=("lambda_clp", "mean"),
            n_repeats=("repeat_id", "count"),
        )
        .sort_values(["rho", "method"])
        .reset_index(drop=True)
    )
    return agg


def smoke_rho_config() -> VpClpRunConfig:
    """Fast config for smoke tests."""
    return VpClpRunConfig(
        window_len=12,
        csgm_ae_epochs=15,
        csgm_iters=50,
        csgm_restarts=1,
        n_depth_blocks=2,
        methods=(),
    )
