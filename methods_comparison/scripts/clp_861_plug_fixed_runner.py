#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLP-CSGM with fixed plug-depth measurement mask (Well 861 MOGNO).

b uses Phi_lab only at global plug row indices inside each window; M is coordinate
subsampling at those depths (not random rho).

Planning: methods_comparison/planning/etapa1f_clp_csgm_phi_lab_poco861.md
ASCII-only.
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clp_861_protocol import (  # noqa: E402
    Clp861RunPaths,
    compare_rf_baseline_dir,
    load_plug_measurement_rows,
    plug_row_indices_unique,
    u_channels_csv,
)
from ml_861_data import (  # noqa: E402
    CLP_861_SCENARIO_PLUG_SPARSE,
    depth_block_splits,
)

import csgm_m2_module as csgm  # noqa: E402
import external_benchmarks as extb  # noqa: E402
import real_well_f03 as rwf  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sir_cs_pipeline_optimized import (  # noqa: E402
    Config,
    apply_config_profile,
)

# auddys helpers (same Excel -> windows pipeline)
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from auddys_smoke_direct_ub import (  # noqa: E402
    _depth_profile_figure_name,
    _profile_x_label_for_target,
    apply_depth_bounds,
    build_windows,
    load_logs_table,
)


@dataclass(frozen=True)
class PlugFixedRunConfig:
    """Knobs for plug-fixed CLP run."""

    excel_path: Path
    run_paths: Clp861RunPaths
    window_len: int = 16
    step: int = 1
    seeds: Tuple[int, ...] = (7,)
    prior_types: Tuple[str, ...] = ("ridge",)
    measurement_noise_std: float = 0.01
    csgm_latent_dim: int = 16
    csgm_hidden_dim: int = 128
    csgm_ae_epochs: int = 200
    csgm_iters: int = 400
    csgm_restarts: int = 3
    csgm_opt_lr: float = 0.05
    csgm_lambda_grid: Tuple[float, ...] = (
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    )
    depth_min_m: float = 5205.91
    depth_max_m: float = 5233.72
    n_depth_blocks: int = 3
    rf_n_estimators: int = 200
    device: Optional[str] = None


def resolve_torch_device(explicit: Optional[str] = None) -> str:
    """
    Pick torch device: explicit override, else CUDA if available, else CPU.

    Honor CUDA_VISIBLE_DEVICES when set (empty string forces CPU).
    """
    if explicit is not None and str(explicit).strip():
        dev = str(explicit).strip().lower()
        if dev not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda', got {}.".format(dev))
        if dev == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device=cuda requested but torch.cuda.is_available() is False.")
        return dev
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def torch_device_label(device: str) -> str:
    """Human-readable device string for logs and PROTOCOL.txt."""
    dev = str(device).strip().lower()
    if dev == "cuda" and torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "cuda"
        return "cuda ({})".format(name)
    return "cpu"


PLUG_FIXED_METHOD_COLORS: Dict[str, str] = {
    "ridge_prior_csgm_plug_fixed": "#1565C0",
    "mlp_prior_csgm_plug_fixed": "#E65100",
    "rf_prior_csgm_plug_fixed": "#6A1B9A",
    "rf_residual_csgm_plug_fixed": "#00897B",
}

RF_PRIOR_FEATURE_COLS: Tuple[str, ...] = (
    "gr",
    "density",
    "res_deep",
    "res_shallow",
    "phi_neutron",
    "phi_sonic",
    "phi_nd",
    "lithotype",
)

OBSERVED_PROFILE_COLOR = "#212121"
PLUG_MARKER_COLOR = "#2E7D32"


def _method_short_label(method: str) -> str:
    """Display label for plug-fixed method keys."""
    if method.startswith("ridge_"):
        return "ridge CSGM (plug-fixed b)"
    if method.startswith("mlp_"):
        return "mlp CSGM (plug-fixed b)"
    if method.startswith("rf_residual_"):
        return "RF + residual CSGM (plug-fixed b)"
    if method.startswith("rf_"):
        return "RF prior CSGM (plug-fixed b)"
    return method


class RfWindowPrior:
    """Pointwise wireline RF; window curve encoded to CSGM latent z0."""

    def __init__(self, n_estimators: int = 200, random_state: int = 0) -> None:
        self.model = RandomForestRegressor(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame, train_row_indices: np.ndarray) -> "RfWindowPrior":
        """Fit RF on train rows only (no lab leakage in u)."""
        idx = np.asarray(train_row_indices, dtype=np.int64)
        x = df.iloc[idx][list(RF_PRIOR_FEATURE_COLS)].to_numpy(dtype=np.float64)
        y = df.iloc[idx]["phi_lab"].to_numpy(dtype=np.float64)
        self.model.fit(x, y)
        self._fitted = True
        return self

    def predict_rows(self, df: pd.DataFrame, row_indices: np.ndarray) -> np.ndarray:
        """Predict phi_lab at explicit row indices."""
        if not self._fitted:
            raise RuntimeError("RfWindowPrior is not fitted.")
        idx = np.asarray(row_indices, dtype=np.int64)
        x = df.iloc[idx][list(RF_PRIOR_FEATURE_COLS)].to_numpy(dtype=np.float64)
        return self.model.predict(x).astype(np.float64)

    def predict_window_curve(
        self,
        df: pd.DataFrame,
        window_start: int,
        window_len: int,
    ) -> np.ndarray:
        """RF curve of length window_len for one sliding window."""
        rows = np.arange(int(window_start), int(window_start) + int(window_len), dtype=np.int64)
        return self.predict_rows(df, rows)

    def encode_z0(
        self,
        ae: csgm.TinyAE,
        df: pd.DataFrame,
        window_start: int,
        window_len: int,
        y_scaler: StandardScaler,
        device: str,
    ) -> np.ndarray:
        """Map RF window curve to AE latent prior code."""
        y_rf = self.predict_window_curve(df, window_start, window_len)
        y_rf_n = y_scaler.transform(y_rf.reshape(1, -1))
        return csgm.encode_y(ae, y_rf_n, device=device)


def subtract_rf_from_windows(
    Y: np.ndarray,
    starts: np.ndarray,
    rf_prior: RfWindowPrior,
    df: pd.DataFrame,
    window_len: int,
) -> np.ndarray:
    """Per-window residual delta = phi_lab_window - RF_window_curve."""
    y_out = np.asarray(Y, dtype=np.float64).copy()
    for j in range(int(y_out.shape[0])):
        start_j = int(starts[j])
        y_rf = rf_prior.predict_window_curve(df, start_j, int(window_len))
        y_out[j] = y_out[j] - y_rf
    return y_out


def encode_zero_residual_z0(
    ae: csgm.TinyAE,
    y_scaler: StandardScaler,
    device: str,
) -> np.ndarray:
    """Latent prior for zero residual (trust RF baseline when b is empty)."""
    n_out = int(y_scaler.mean_.shape[0])
    zeros = np.zeros((1, n_out), dtype=np.float64)
    zeros_n = y_scaler.transform(zeros)
    return csgm.encode_y(ae, zeros_n, device=device)


def _recover_rf_residual_window(
    ae: csgm.TinyAE,
    rf_prior: RfWindowPrior,
    df: pd.DataFrame,
    window_start: int,
    y_true_row: np.ndarray,
    M: np.ndarray,
    cfg: Config,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    y_scaler: StandardScaler,
    lam: float,
    noise_std: float,
    rng: np.random.Generator,
    seed: int,
    device: str,
    window_len: int,
) -> np.ndarray:
    """
    CSGM on residual delta; final curve = RF_window + delta_hat.

    b observes delta at plug coordinates only (M @ (y_true - y_rf)).
  """
    y_rf = rf_prior.predict_window_curve(df, int(window_start), int(window_len))
    delta_row = np.asarray(y_true_row, dtype=np.float64).ravel() - y_rf
    z0 = encode_zero_residual_z0(ae, y_scaler, device)
    if M.shape[0] == 0:
        delta_hat = csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device=device)[0]
    else:
        B = _make_b_row(delta_row, M, noise_std, rng)
        delta_hat = csgm.csgm_recover_with_prior(
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
    return y_rf + delta_hat


def _select_lambda_rf_residual(
    ae: csgm.TinyAE,
    rf_prior: RfWindowPrior,
    df: pd.DataFrame,
    Y_val: np.ndarray,
    starts_val: np.ndarray,
    plug_rows: Sequence[int],
    cfg: Config,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    y_scaler: StandardScaler,
    seed: int,
    device: str,
    rng: np.random.Generator,
    window_len: int,
) -> Tuple[float, float]:
    """Grid search lambda on validation windows (metric on phi_lab, not delta)."""
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    l = int(Y_val.shape[1])
    for lam in cfg.csgm_lambda_grid:
        preds: List[np.ndarray] = []
        for j in range(int(Y_val.shape[0])):
            start_j = int(starts_val[j])
            loc = plug_local_indices(start_j, plug_rows, l)
            M = build_coordinate_M(loc, l)
            y_hat = _recover_rf_residual_window(
                ae,
                rf_prior,
                df,
                start_j,
                Y_val[j],
                M,
                cfg,
                y_mean,
                y_scale,
                y_scaler,
                float(lam),
                float(cfg.measurement_noise_std),
                rng,
                int(seed),
                device,
                int(window_len),
            )
            preds.append(y_hat)
        pred = np.stack(preds, axis=0)
        score = float(np.sqrt(np.mean((pred - Y_val) ** 2)))
        if score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam, best_score


def plot_plug_fixed_depth_profile(
    depth_axis: np.ndarray,
    profiles: Dict[str, np.ndarray],
    save_path: Path,
    methods: Sequence[str],
    title: str,
    profile_x_label: str,
    plug_row_indices: Optional[Sequence[int]] = None,
) -> None:
    """
    Single-panel depth profile with high-contrast colors for plug-fixed CLP.

    observed = black; ridge = blue; mlp = orange; plug measurement depths = green rings.
    """
    import matplotlib.pyplot as plt

    d = np.asarray(depth_axis, dtype=np.float64).ravel()
    obs = np.asarray(profiles["observed"], dtype=np.float64).ravel()
    if obs.shape[0] != d.shape[0]:
        raise ValueError("depth_axis and observed must have equal length.")

    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    mask_obs = np.isfinite(obs)
    if bool(np.any(mask_obs)):
        ax.plot(
            obs[mask_obs],
            d[mask_obs],
            color=OBSERVED_PROFILE_COLOR,
            linewidth=1.6,
            linestyle="-",
            label="observed (phi_lab)",
            zorder=2,
        )

    for method in methods:
        if method not in profiles:
            continue
        pr = np.asarray(profiles[method], dtype=np.float64).ravel()
        if pr.shape[0] != d.shape[0]:
            continue
        m_pr = np.isfinite(pr)
        if not bool(np.any(m_pr)):
            continue
        col = PLUG_FIXED_METHOD_COLORS.get(method, "#7f7f7f")
        ax.plot(
            pr[m_pr],
            d[m_pr],
            color=col,
            linewidth=2.2,
            linestyle="-",
            label=_method_short_label(method),
            zorder=3,
        )

    if plug_row_indices is not None:
        idx = [int(i) for i in plug_row_indices if 0 <= int(i) < d.shape[0]]
        if idx:
            plug_d = d[idx]
            plug_y = obs[idx]
            m_plug = np.isfinite(plug_y)
            if bool(np.any(m_plug)):
                ax.scatter(
                    plug_y[m_plug],
                    plug_d[m_plug],
                    s=55,
                    facecolors="none",
                    edgecolors=PLUG_MARKER_COLOR,
                    linewidths=1.8,
                    label="plug b (fixed mask)",
                    zorder=4,
                )

    ax.set_xlabel(str(profile_x_label))
    ax.set_ylabel("depth (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.invert_yaxis()
    fig.suptitle(str(title), fontsize=12, y=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_plug_fixed_depth_profile_panels(
    depth_axis: np.ndarray,
    profiles: Dict[str, np.ndarray],
    save_path: Path,
    methods: Sequence[str],
    title: str,
    profile_x_label: str,
) -> None:
    """Optional side-by-side panels (observed dashed gray vs colored prediction)."""
    import matplotlib.pyplot as plt

    d = np.asarray(depth_axis, dtype=np.float64).ravel()
    obs = np.asarray(profiles["observed"], dtype=np.float64).ravel()
    mask_obs = np.isfinite(obs)
    n_cols = 1 + len(methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 7.5), sharey=True)
    if n_cols == 1:
        axes = np.array([axes])

    ax0 = axes[0]
    ax0.plot(obs[mask_obs], d[mask_obs], color=OBSERVED_PROFILE_COLOR, linewidth=1.6, label="observed")
    ax0.set_title("observed", fontsize=11)
    ax0.set_xlabel(str(profile_x_label))
    ax0.set_ylabel("depth (m)")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best", fontsize=9)

    for ax_i, method in enumerate(methods, start=1):
        ax = axes[ax_i]
        pr = np.asarray(profiles[method], dtype=np.float64).ravel()
        m_pr = np.isfinite(pr) & mask_obs
        col = PLUG_FIXED_METHOD_COLORS.get(method, "#7f7f7f")
        ax.plot(
            obs[mask_obs],
            d[mask_obs],
            color="#9E9E9E",
            linewidth=1.0,
            linestyle="--",
            alpha=0.9,
            label="observed",
        )
        ax.plot(
            pr[m_pr],
            d[m_pr],
            color=col,
            linewidth=2.2,
            label=_method_short_label(method),
        )
        ax.set_title(_method_short_label(method), fontsize=10)
        ax.set_xlabel(str(profile_x_label))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for ax in axes:
        ax.invert_yaxis()
    fig.suptitle(str(title), fontsize=12, y=1.0)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)

@dataclass
class PlugFixedFigureContext:
    """Inputs for depth profile and parity figures after CV."""

    depth_axis: np.ndarray
    observed: np.ndarray
    window_len: int
    plug_row_indices: List[int]
    fold_depth_bounds: List[Tuple[float, float]]
    parity_parts: Dict[Tuple[int, str], Dict[str, List[np.ndarray]]]


def _parity_key(seed: int, method: str) -> Tuple[int, str]:
    return int(seed), str(method)


def _append_parity_part(
    parity_parts: DefaultDict[Tuple[int, str], Dict[str, List[np.ndarray]]],
    seed: int,
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    start_row: int,
) -> None:
    """Accumulate one test window for OOF stitching."""
    key = _parity_key(seed, method)
    parity_parts[key]["y_true"].append(np.asarray(y_true, dtype=np.float64).ravel())
    parity_parts[key]["y_pred"].append(np.asarray(y_pred, dtype=np.float64).ravel())
    parity_parts[key]["starts"].append(int(start_row))


def _stack_parity_part(data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack lists into arrays for profile reconstruction."""
    y_true = np.stack(data["y_true"], axis=0)
    y_pred = np.stack(data["y_pred"], axis=0)
    starts = np.asarray(data["starts"], dtype=np.int64)
    return {"y_true": y_true, "y_pred": y_pred, "starts": starts}


def save_plug_fixed_rmse_by_fold_figure(
    summary_by_fold: pd.DataFrame,
    figures_dir: Path,
    primary_seed: int = 7,
) -> Path:
    """Bar chart of RMSE by depth-block fold (one seed)."""
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    sub = summary_by_fold.loc[summary_by_fold["seed"] == int(primary_seed)].copy()
    if sub.empty:
        sub = summary_by_fold.copy()
    methods = sorted(sub["method"].unique().tolist())
    fold_ids = sorted(sub["fold_id"].unique().tolist())
    x = np.arange(len(fold_ids), dtype=np.float64)
    width = 0.35
    colors = PLUG_FIXED_METHOD_COLORS

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for i, method in enumerate(methods):
        msub = sub.loc[sub["method"] == method].set_index("fold_id").reindex(fold_ids)
        offset = (float(i) - 0.5 * (len(methods) - 1)) * width
        vals = msub["rmse_mean"].to_numpy(dtype=np.float64)
        ax.bar(x + offset, vals, width=width, label=_method_short_label(method),
               color=colors.get(method, "#7f7f7f"))
    ax.set_xticks(x)
    ax.set_xticklabels(["fold {}".format(int(f)) for f in fold_ids])
    ax.set_ylabel("RMSE (pu)")
    ax.set_xlabel("depth-block fold")
    ax.set_title("Plug-fixed CLP: RMSE by fold (seed={})".format(int(primary_seed)))
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out = figures_dir / "11_rmse_by_depth_fold.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_plug_fixed_parity_figure(
    parity_parts: Dict[Tuple[int, str], Dict[str, List[np.ndarray]]],
    figures_dir: Path,
    primary_seed: int = 7,
) -> Path:
    """Parity scatter: ground truth vs prediction (pooled test windows, one seed)."""
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    methods = sorted({k[1] for k in parity_parts if k[0] == int(primary_seed)})
    if not methods:
        methods = sorted({k[1] for k in parity_parts})
        primary_seed = sorted({k[0] for k in parity_parts})[0]

    y_true_ref: Optional[np.ndarray] = None
    pred_by_method: Dict[str, np.ndarray] = {}
    for method in methods:
        key = _parity_key(int(primary_seed), method)
        if key not in parity_parts:
            continue
        stacked = _stack_parity_part(parity_parts[key])
        yt = stacked["y_true"].reshape(-1)
        yp = stacked["y_pred"].reshape(-1)
        if y_true_ref is None:
            y_true_ref = yt
        pred_by_method[method] = yp

    if y_true_ref is None or not pred_by_method:
        raise ValueError("No parity data for seed {}".format(primary_seed))

    n_m = len(pred_by_method)
    ncols = 2
    nrows = int(np.ceil(n_m / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    lo = float(np.min(y_true_ref))
    hi = float(np.max(y_true_ref))
    colors = PLUG_FIXED_METHOD_COLORS
    for ax_idx, method in enumerate(sorted(pred_by_method.keys())):
        r, c = divmod(ax_idx, ncols)
        ax = axes[r][c]
        yp = pred_by_method[method]
        m = np.isfinite(y_true_ref) & np.isfinite(yp)
        ax.scatter(y_true_ref[m], yp[m], s=8, alpha=0.35,
                   color=colors.get(method, "#7f7f7f"), rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.0)
        if int(np.sum(m)) > 2:
            cc = float(np.corrcoef(y_true_ref[m], yp[m])[0, 1])
            ax.set_title("{} (corr={:.3f})".format(_method_short_label(method), cc))
        else:
            ax.set_title(_method_short_label(method))
        ax.set_xlabel("phi_lab observed (pu)")
        ax.set_ylabel("phi_lab predicted (pu)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
    for ax_idx in range(n_m, nrows * ncols):
        r, c = divmod(ax_idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle(
        "Parity OOF (depth-block CV, seed={}, plug-fixed b)".format(int(primary_seed)),
        fontsize=12,
    )
    plt.tight_layout()
    out = figures_dir / "09_parity_ground_truth_vs_prediction.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def save_plug_fixed_depth_profile_figure(
    fig_ctx: PlugFixedFigureContext,
    run_paths: Clp861RunPaths,
    primary_seed: int = 7,
    overlap_agg: str = rwf.PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
) -> Path:
    """OOF depth profile stitched from all depth-block test windows."""
    run_paths.figures.mkdir(parents=True, exist_ok=True)
    d = np.asarray(fig_ctx.depth_axis, dtype=np.float64).ravel()
    nrows = int(d.shape[0])
    l = int(fig_ctx.window_len)
    agg = rwf.validate_profile_overlap_agg(str(overlap_agg))

    profiles: Dict[str, np.ndarray] = {"observed": np.asarray(fig_ctx.observed, dtype=np.float64).ravel()}
    parity_npz: Dict[str, np.ndarray] = {"depth_axis": d}

    methods_for_seed = sorted(
        {k[1] for k in fig_ctx.parity_parts if k[0] == int(primary_seed)}
    )
    if not methods_for_seed:
        primary_seed = sorted({k[0] for k in fig_ctx.parity_parts})[0]
        methods_for_seed = sorted(
            {k[1] for k in fig_ctx.parity_parts if k[0] == int(primary_seed)}
        )

    for method in methods_for_seed:
        key = _parity_key(int(primary_seed), method)
        stacked = _stack_parity_part(fig_ctx.parity_parts[key])
        prof, _ = rwf.reconstruct_depth_profile(
            stacked["y_pred"],
            stacked["starts"],
            l,
            nrows,
            overlap_agg=agg,
        )
        profiles[method] = prof
        parity_npz["profile_{}".format(method)] = prof
        parity_npz["row_starts_{}".format(method)] = stacked["starts"]
        parity_npz["y_true_windows_{}".format(method)] = stacked["y_true"]

    prof_obs, _ = rwf.reconstruct_depth_profile(
        _stack_parity_part(
            fig_ctx.parity_parts[_parity_key(int(primary_seed), methods_for_seed[0])]
        )["y_true"],
        _stack_parity_part(
            fig_ctx.parity_parts[_parity_key(int(primary_seed), methods_for_seed[0])]
        )["starts"],
        l,
        nrows,
        overlap_agg=agg,
    )
    profiles["observed_stitched"] = prof_obs
    parity_npz["profile_observed_stitched"] = prof_obs
    parity_npz["profile_observed_full"] = profiles["observed"]

    depth_name = _depth_profile_figure_name("phi_lab")
    out = run_paths.figures / depth_name
    title = "861 MOGNO phi_lab OOF (plug-fixed b, seed={})".format(int(primary_seed))
    x_label = _profile_x_label_for_target("phi_lab")
    plot_plug_fixed_depth_profile(
        d,
        profiles,
        out,
        methods_for_seed,
        title,
        x_label,
        plug_row_indices=fig_ctx.plug_row_indices,
    )
    panels_out = run_paths.figures / "10_depth_profile_porosity_panels.png"
    plot_plug_fixed_depth_profile_panels(
        d,
        profiles,
        panels_out,
        methods_for_seed,
        title,
        x_label,
    )

    npz_path = run_paths.tables / "depth_profile_oof.npz"
    np.savez_compressed(npz_path, **parity_npz)
    return out


def regenerate_plug_fixed_depth_profile_from_npz(
    run_paths: Clp861RunPaths,
    primary_seed: int = 7,
) -> List[Path]:
    """Rebuild depth-profile figures from saved depth_profile_oof.npz (no re-training)."""
    npz_path = run_paths.tables / "depth_profile_oof.npz"
    meta_path = run_paths.run_root / "plug_fixed_meta.json"
    if not npz_path.is_file():
        raise FileNotFoundError(str(npz_path))
    z = np.load(npz_path)
    d = np.asarray(z["depth_axis"], dtype=np.float64).ravel()
    if "profile_observed_full" in z.files:
        observed = np.asarray(z["profile_observed_full"], dtype=np.float64).ravel()
    else:
        observed = np.asarray(z["profile_observed_stitched"], dtype=np.float64).ravel()
    profiles: Dict[str, np.ndarray] = {"observed": observed}
    methods: List[str] = []
    for key in sorted(z.files):
        if not key.startswith("profile_"):
            continue
        name = key[len("profile_") :]
        if name in ("observed_stitched", "observed_full"):
            continue
        profiles[name] = np.asarray(z[key], dtype=np.float64).ravel()
        methods.append(name)
    plug_rows: List[int] = []
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        plug_rows = [int(x) for x in meta.get("plug_global_row_indices", [])]
    title = "861 MOGNO phi_lab OOF (plug-fixed b, seed={})".format(int(primary_seed))
    x_label = _profile_x_label_for_target("phi_lab")
    run_paths.figures.mkdir(parents=True, exist_ok=True)
    out = run_paths.figures / _depth_profile_figure_name("phi_lab")
    plot_plug_fixed_depth_profile(
        d, profiles, out, methods, title, x_label, plug_row_indices=plug_rows
    )
    panels_out = run_paths.figures / "10_depth_profile_porosity_panels.png"
    plot_plug_fixed_depth_profile_panels(d, profiles, panels_out, methods, title, x_label)
    return [out, panels_out]


def save_plug_fixed_figures(
    run_paths: Clp861RunPaths,
    fig_ctx: PlugFixedFigureContext,
    summary_by_fold: pd.DataFrame,
    primary_seed: int = 7,
) -> List[Path]:
    """Write RMSE, parity, and depth-profile figures under run_paths/figures/."""
    paths: List[Path] = []
    paths.append(save_plug_fixed_rmse_by_fold_figure(summary_by_fold, run_paths.figures, primary_seed))
    paths.append(save_plug_fixed_parity_figure(fig_ctx.parity_parts, run_paths.figures, primary_seed))
    paths.append(
        save_plug_fixed_depth_profile_figure(fig_ctx, run_paths, primary_seed=primary_seed)
    )
    return paths


def save_plug_fixed_figures_from_tables(
    run_paths: Clp861RunPaths,
    primary_seed: int = 7,
) -> List[Path]:
    """Regenerate figures from saved tables/npz (no full re-training)."""
    paths: List[Path] = []
    summary_path = run_paths.tables / "summary_plug_fixed_by_fold.csv"
    if summary_path.is_file():
        summary = pd.read_csv(summary_path)
        paths.append(
            save_plug_fixed_rmse_by_fold_figure(summary, run_paths.figures, primary_seed)
        )
    npz_path = run_paths.tables / "depth_profile_oof.npz"
    if npz_path.is_file():
        paths.extend(regenerate_plug_fixed_depth_profile_from_npz(run_paths, primary_seed))
    if not paths:
        raise FileNotFoundError("No summary or depth_profile_oof.npz under {}".format(run_paths.tables))
    return paths


def _parse_channels_csv() -> Tuple[str, ...]:
    return tuple(u_channels_csv().split(","))


def plug_local_indices(
    window_start: int,
    plug_global_rows: Sequence[int],
    window_len: int,
) -> List[int]:
    """Local indices in [0, window_len) where plug lab measurements exist."""
    out: List[int] = []
    end = int(window_start) + int(window_len)
    for row in plug_global_rows:
        r = int(row)
        if int(window_start) <= r < end:
            loc = r - int(window_start)
            if loc not in out:
                out.append(loc)
    return sorted(out)


def build_coordinate_M(local_indices: Sequence[int], window_len: int) -> np.ndarray:
    """M with one row per observed coordinate."""
    idx = list(local_indices)
    m = len(idx)
    if m == 0:
        return np.zeros((0, int(window_len)), dtype=np.float64)
    mat = np.zeros((m, int(window_len)), dtype=np.float64)
    for i, j in enumerate(idx):
        mat[i, int(j)] = 1.0
    return mat


def window_fully_in_rows(window_start: int, window_len: int, row_set: set) -> bool:
    """True if every row index in the window belongs to row_set."""
    for r in range(int(window_start), int(window_start) + int(window_len)):
        if r not in row_set:
            return False
    return True


def classify_windows_by_rows(
    starts: np.ndarray,
    window_len: int,
    train_rows: np.ndarray,
    val_rows: np.ndarray,
    test_rows: np.ndarray,
) -> Tuple[List[int], List[int], List[int]]:
    """Assign window indices to train/val/test (must lie fully inside row sets)."""
    tr_set = set(int(x) for x in train_rows.tolist())
    va_set = set(int(x) for x in val_rows.tolist())
    te_set = set(int(x) for x in test_rows.tolist())
    train_w: List[int] = []
    val_w: List[int] = []
    test_w: List[int] = []
    for wi, s in enumerate(starts.tolist()):
        s_int = int(s)
        if window_fully_in_rows(s_int, window_len, te_set):
            test_w.append(wi)
        elif window_fully_in_rows(s_int, window_len, va_set):
            val_w.append(wi)
        elif window_fully_in_rows(s_int, window_len, tr_set):
            train_w.append(wi)
    return train_w, val_w, test_w


def _make_b_row(y: np.ndarray, M: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """b = M y + noise; returns shape (1, m)."""
    if M.shape[0] == 0:
        return np.zeros((1, 0), dtype=np.float64)
    b = (M @ y.reshape(-1)).reshape(1, -1)
    if noise_std > 0.0:
        b = b + noise_std * rng.normal(size=b.shape)
    return b


def _recover_one_window(
    ae: csgm.TinyAE,
    prior: Any,
    X_row: np.ndarray,
    y_row: np.ndarray,
    M: np.ndarray,
    cfg: Config,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    noise_std: float,
    rng: np.random.Generator,
    seed: int,
    device: str,
    z0_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    """CSGM recovery for one window; prior-only if M is empty."""
    if z0_override is not None:
        z0 = z0_override
    else:
        if prior is None:
            raise ValueError("prior is required when z0_override is None.")
        z0 = prior.predict(X_row.reshape(1, -1))
    if M.shape[0] == 0:
        return csgm.decode_latent_prior(ae, z0, y_mean, y_scale, device=device)[0]
    B = _make_b_row(y_row, M, noise_std, rng)
    y_hat = csgm.csgm_recover_with_prior(
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
    )
    return y_hat[0]


def _resolve_z0_for_window(
    prior_type: str,
    prior: Any,
    rf_prior: Optional[RfWindowPrior],
    ae: csgm.TinyAE,
    df: pd.DataFrame,
    x_row: np.ndarray,
    window_start: int,
    window_len: int,
    y_scaler: StandardScaler,
    device: str,
) -> np.ndarray:
    """Latent prior z0 for one window (ridge/mlp or RF curve)."""
    pt = str(prior_type).strip().lower()
    if pt == "rf":
        if rf_prior is None:
            raise ValueError("rf prior requested but rf_prior is None.")
        return rf_prior.encode_z0(
            ae, df, int(window_start), int(window_len), y_scaler, device
        )
    if prior is None:
        raise ValueError("prior is required for prior_type={}.".format(pt))
    return prior.predict(x_row.reshape(1, -1))


def _select_lambda_plug_fixed(
    ae: csgm.TinyAE,
    prior: Any,
    prior_type: str,
    rf_prior: Optional[RfWindowPrior],
    df: pd.DataFrame,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    starts_val: np.ndarray,
    plug_rows: Sequence[int],
    cfg: Config,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    y_scaler: StandardScaler,
    seed: int,
    device: str,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Grid search lambda on validation windows with plug-fixed M."""
    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    l = int(Y_val.shape[1])
    for lam in cfg.csgm_lambda_grid:
        preds: List[np.ndarray] = []
        for j in range(int(Y_val.shape[0])):
            start_j = int(starts_val[j])
            loc = plug_local_indices(start_j, plug_rows, l)
            M = build_coordinate_M(loc, l)
            z0 = _resolve_z0_for_window(
                prior_type,
                prior,
                rf_prior,
                ae,
                df,
                X_val[j],
                start_j,
                l,
                y_scaler,
                device,
            )
            y_hat = _recover_one_window(
                ae,
                prior,
                X_val[j],
                Y_val[j],
                M,
                cfg,
                y_mean,
                y_scale,
                float(lam),
                float(cfg.measurement_noise_std),
                rng,
                seed,
                device,
                z0_override=z0,
            )
            preds.append(y_hat)
        pred = np.stack(preds, axis=0)
        score = float(np.sqrt(np.mean((pred - Y_val) ** 2)))
        if score < best_score:
            best_score = score
            best_lam = float(lam)
    return best_lam, best_score


def run_plug_fixed_depth_block_cv(
    cfg_run: PlugFixedRunConfig,
) -> Tuple[pd.DataFrame, PlugFixedFigureContext]:
    """3-fold depth-block CV with plug-fixed b; returns metrics and figure inputs."""
    channels = _parse_channels_csv()
    target = "phi_lab"
    df = load_logs_table(str(cfg_run.excel_path), "Logs")
    df = apply_depth_bounds(df, cfg_run.depth_min_m, cfg_run.depth_max_m)
    plug_rows = plug_row_indices_unique(load_plug_measurement_rows())

    x_all, y_all, starts = build_windows(
        df, channels, target, int(cfg_run.window_len), int(cfg_run.step)
    )
    n_rows = int(df.shape[0])
    folds = depth_block_splits(df.assign(**{"Depth(m)": df["depth_m"]}), cfg_run.n_depth_blocks)

    all_rows: List[dict] = []
    plug_window_stats: List[dict] = []
    parity_parts: DefaultDict[Tuple[int, str], Dict[str, List[np.ndarray]]] = defaultdict(
        lambda: {"y_true": [], "y_pred": [], "starts": []}
    )
    fold_depth_bounds: List[Tuple[float, float]] = []

    device = resolve_torch_device(cfg_run.device)
    print("CLP_PLUG_FIXED_DEVICE", torch_device_label(device))

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
                "Fold {} has empty train/test windows (train_w={}, test_w={}).".format(
                    fold.fold_id, len(tr_w), len(te_w)
                )
            )

        n_plugs_in_test = 0
        for wi in te_w:
            loc = plug_local_indices(int(starts[wi]), plug_rows, int(cfg_run.window_len))
            n_plugs_in_test += len(loc)
        plug_window_stats.append(
            {
                "fold_id": int(fold.fold_id),
                "n_train_windows": len(tr_w),
                "n_val_windows": len(va_w),
                "n_test_windows": len(te_w),
                "n_plug_obs_test": int(n_plugs_in_test),
                "depth_min_m": float(fold.depth_min_m),
                "depth_max_m": float(fold.depth_max_m),
            }
        )
        fold_depth_bounds.append((float(fold.depth_min_m), float(fold.depth_max_m)))

        X_tr = x_all[tr_w]
        Y_tr = y_all[tr_w]
        X_va = x_all[va_w] if va_w else x_all[tr_w[-max(1, len(tr_w) // 5) :]]
        Y_va = y_all[va_w] if va_w else y_all[tr_w[-max(1, len(tr_w) // 5) :]]
        starts_va = starts[va_w] if va_w else starts[tr_w[-max(1, len(tr_w) // 5) :]]
        X_te = x_all[te_w]
        Y_te = y_all[te_w]
        starts_te = starts[te_w]

        cfg = Config(config_profile="real_well_f03_direct_ub")
        apply_config_profile(cfg)
        cfg.n_output = int(cfg_run.window_len)
        cfg.p_input = int(X_tr.shape[1])
        cfg.measurement_noise_std = float(cfg_run.measurement_noise_std)
        cfg.csgm_latent_dim = int(cfg_run.csgm_latent_dim)
        cfg.csgm_hidden_dim = int(cfg_run.csgm_hidden_dim)
        cfg.csgm_ae_epochs = int(cfg_run.csgm_ae_epochs)
        cfg.csgm_iters = int(cfg_run.csgm_iters)
        cfg.csgm_restarts = int(cfg_run.csgm_restarts)
        cfg.csgm_opt_lr = float(cfg_run.csgm_opt_lr)
        cfg.csgm_lambda_grid = list(cfg_run.csgm_lambda_grid)
        cfg.model_selection_metric = "rmse"

        for seed in cfg_run.seeds:
            rng = np.random.default_rng(int(seed))
            for prior_type in cfg_run.prior_types:
                pt = str(prior_type).strip().lower()
                rf_prior: Optional[RfWindowPrior] = None
                prior = None
                if pt in ("rf", "rf_residual"):
                    rf_prior = RfWindowPrior(
                        n_estimators=int(cfg_run.rf_n_estimators),
                        random_state=int(seed),
                    )
                    rf_prior.fit(df, train_rows)
                if pt == "rf_residual":
                    Y_tr_fit = subtract_rf_from_windows(
                        Y_tr, starts[tr_w], rf_prior, df, int(cfg_run.window_len)
                    )
                    y_scaler = StandardScaler().fit(Y_tr_fit)
                    y_train_n = y_scaler.transform(Y_tr_fit)
                    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
                    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
                    ae = csgm.train_ae_generator(y_train_n, cfg, seed=int(seed), device=device)
                    best_lam, val_score = _select_lambda_rf_residual(
                        ae,
                        rf_prior,
                        df,
                        Y_va,
                        starts_va,
                        plug_rows,
                        cfg,
                        y_mean,
                        y_scale,
                        y_scaler,
                        int(seed),
                        device,
                        rng,
                        int(cfg_run.window_len),
                    )
                    method = "rf_residual_csgm_plug_fixed"
                    for j in range(int(Y_te.shape[0])):
                        start_j = int(starts_te[j])
                        loc = plug_local_indices(start_j, plug_rows, int(cfg_run.window_len))
                        M = build_coordinate_M(loc, int(cfg_run.window_len))
                        y_hat = _recover_rf_residual_window(
                            ae,
                            rf_prior,
                            df,
                            start_j,
                            Y_te[j],
                            M,
                            cfg,
                            y_mean,
                            y_scale,
                            y_scaler,
                            best_lam,
                            float(cfg.measurement_noise_std),
                            rng,
                            int(seed),
                            device,
                            int(cfg_run.window_len),
                        )
                        m_obs = int(M.shape[0])
                        nan_f = float("nan")
                        row = extb.per_sample_metrics_row(
                            int(seed),
                            float("nan"),
                            method,
                            int(j),
                            Y_te[j],
                            y_hat,
                            np.zeros(int(cfg_run.window_len), dtype=np.float64),
                            np.zeros(int(cfg_run.window_len), dtype=np.float64),
                            best_lam,
                            "csgm_plug_fixed_residual",
                            m_obs,
                            support_f1_override=nan_f,
                        )
                        row["fold_id"] = int(fold.fold_id)
                        row["measurement_kind"] = "plug_fixed"
                        row["n_plugs_in_window"] = m_obs
                        row["window_start_row"] = int(starts_te[j])
                        row["val_score"] = val_score
                        all_rows.append(row)
                        _append_parity_part(
                            parity_parts,
                            int(seed),
                            method,
                            Y_te[j],
                            y_hat,
                            int(starts_te[j]),
                        )
                    continue
                if pt not in ("rf",):
                    cfg.csgm_prior_type = pt
                    prior = csgm._make_prior(cfg, seed=int(seed) + 7000)

                y_scaler = StandardScaler().fit(Y_tr)
                y_train_n = y_scaler.transform(Y_tr)
                y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
                y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)

                ae = csgm.train_ae_generator(y_train_n, cfg, seed=int(seed), device=device)
                z_train = csgm.encode_y(ae, y_train_n, device=device)
                if prior is not None:
                    prior.fit(X_tr, z_train)

                best_lam, val_score = _select_lambda_plug_fixed(
                    ae,
                    prior,
                    pt,
                    rf_prior,
                    df,
                    X_va,
                    Y_va,
                    starts_va,
                    plug_rows,
                    cfg,
                    y_mean,
                    y_scale,
                    y_scaler,
                    int(seed),
                    device,
                    rng,
                )
                method = "{}_prior_csgm_plug_fixed".format(pt)

                for j in range(int(Y_te.shape[0])):
                    start_j = int(starts_te[j])
                    loc = plug_local_indices(start_j, plug_rows, int(cfg_run.window_len))
                    M = build_coordinate_M(loc, int(cfg_run.window_len))
                    z0 = _resolve_z0_for_window(
                        pt,
                        prior,
                        rf_prior,
                        ae,
                        df,
                        X_te[j],
                        start_j,
                        int(cfg_run.window_len),
                        y_scaler,
                        device,
                    )
                    y_hat = _recover_one_window(
                        ae,
                        prior,
                        X_te[j],
                        Y_te[j],
                        M,
                        cfg,
                        y_mean,
                        y_scale,
                        best_lam,
                        float(cfg.measurement_noise_std),
                        rng,
                        int(seed),
                        device,
                        z0_override=z0,
                    )
                    m_obs = int(M.shape[0])
                    nan_f = float("nan")
                    row = extb.per_sample_metrics_row(
                        int(seed),
                        float("nan"),
                        method,
                        int(j),
                        Y_te[j],
                        y_hat,
                        np.zeros(int(cfg_run.window_len), dtype=np.float64),
                        np.zeros(int(cfg_run.window_len), dtype=np.float64),
                        best_lam,
                        "csgm_plug_fixed",
                        m_obs,
                        support_f1_override=nan_f,
                    )
                    row["fold_id"] = int(fold.fold_id)
                    row["measurement_kind"] = "plug_fixed"
                    row["n_plugs_in_window"] = m_obs
                    row["window_start_row"] = int(starts_te[j])
                    row["val_score"] = val_score
                    all_rows.append(row)
                    _append_parity_part(
                        parity_parts,
                        int(seed),
                        method,
                        Y_te[j],
                        y_hat,
                        int(starts_te[j]),
                    )

    stats_path = cfg_run.run_paths.tables / "plug_window_stats_by_fold.csv"
    pd.DataFrame(plug_window_stats).to_csv(stats_path, index=False)

    depth_axis = df["depth_m"].to_numpy(dtype=np.float64)
    observed = df[target].to_numpy(dtype=np.float64)
    fig_ctx = PlugFixedFigureContext(
        depth_axis=depth_axis,
        observed=observed,
        window_len=int(cfg_run.window_len),
        plug_row_indices=list(plug_rows),
        fold_depth_bounds=fold_depth_bounds,
        parity_parts=dict(parity_parts),
    )
    return pd.DataFrame(all_rows), fig_ctx


def summarize_plug_fixed(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate RMSE by method, fold, and seed."""
    if df.empty:
        return df
    agg = (
        df.groupby(["fold_id", "method", "seed"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            n_test_windows=("rmse", "count"),
            n_plugs_mean=("n_plugs_in_window", "mean"),
            lambda_selected=("lambda", "first"),
        )
    )
    return agg


def _point_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE on finite pairs."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if not bool(np.any(m)):
        return float("nan")
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def _point_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R2 on finite pairs."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(m)) < 2:
        return float("nan")
    return float(r2_score(y_true[m], y_pred[m]))


def pointwise_fidelity_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    depth_m: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """RMSE, R2, MAE, std ratio and diff-correlation for aligned arrays."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    if not bool(np.any(m)):
        return {
            "rmse": float("nan"),
            "r2": float("nan"),
            "mae": float("nan"),
            "corr": float("nan"),
            "std_ratio": float("nan"),
            "corr_diff": float("nan"),
            "n_points": 0.0,
        }
    yt_m = yt[m]
    yp_m = yp[m]
    rmse = _point_rmse(yt_m, yp_m)
    r2 = _point_r2(yt_m, yp_m)
    mae = float(np.mean(np.abs(yp_m - yt_m)))
    corr = float(np.corrcoef(yt_m, yp_m)[0, 1])
    std_obs = float(np.std(yt_m))
    std_ratio = float(np.std(yp_m) / std_obs) if std_obs > 0.0 else float("nan")
    corr_diff = float("nan")
    if depth_m is not None:
        d = np.asarray(depth_m, dtype=np.float64).ravel()[m]
        if int(d.shape[0]) > 2:
            order = np.argsort(d)
            dy = np.diff(yt_m[order])
            dp = np.diff(yp_m[order])
            if float(np.std(dy)) > 0.0 and float(np.std(dp)) > 0.0:
                corr_diff = float(np.corrcoef(dy, dp)[0, 1])
    return {
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "corr": corr,
        "std_ratio": std_ratio,
        "corr_diff": corr_diff,
        "n_points": float(int(np.sum(m))),
    }


def evaluate_plug_fixed_rf_residual(
    cfg_run: PlugFixedRunConfig,
    primary_seed: int = 7,
) -> Dict[str, float]:
    """Run plug-fixed rf_residual CV and return global fidelity metrics."""
    _detailed, fig_ctx = run_plug_fixed_depth_block_cv(cfg_run)
    method = "rf_residual_csgm_plug_fixed"
    prof = stitch_method_profile(fig_ctx, method, int(primary_seed))
    obs = np.asarray(fig_ctx.observed, dtype=np.float64)
    depth = np.asarray(fig_ctx.depth_axis, dtype=np.float64)
    out = pointwise_fidelity_metrics(obs, prof, depth_m=depth)
    out["method"] = method
    out["seed"] = float(primary_seed)
    return out


def stitch_method_profile(
    fig_ctx: PlugFixedFigureContext,
    method: str,
    seed: int,
    overlap_agg: str = rwf.PROFILE_OVERLAP_AGG_UNIFORM_MEAN,
) -> np.ndarray:
    """Stitch OOF window predictions to one value per depth row."""
    key = _parity_key(int(seed), str(method))
    if key not in fig_ctx.parity_parts:
        raise KeyError("No parity data for seed={} method={}".format(seed, method))
    stacked = _stack_parity_part(fig_ctx.parity_parts[key])
    nrows = int(fig_ctx.depth_axis.shape[0])
    prof, _ = rwf.reconstruct_depth_profile(
        stacked["y_pred"],
        stacked["starts"],
        int(fig_ctx.window_len),
        nrows,
        overlap_agg=overlap_agg,
    )
    return prof


def collect_rf_oof_profiles(
    df: pd.DataFrame,
    n_blocks: int,
    rf_n_estimators: int,
    seed: int,
    plug_rows: Sequence[int],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """RF OOF pointwise predictions (same depth-block splits as CLP)."""
    folds = depth_block_splits(df.assign(**{"Depth(m)": df["depth_m"]}), n_blocks)
    n = int(df.shape[0])
    rf_oof = np.full(n, np.nan, dtype=np.float64)
    plug_set = set(int(r) for r in plug_rows)
    fold_rows: List[dict] = []
    for fold in folds:
        rf = RfWindowPrior(n_estimators=int(rf_n_estimators), random_state=int(seed))
        rf.fit(df, fold.train_idx)
        test_idx = np.asarray(fold.test_idx, dtype=np.int64)
        pred = rf.predict_rows(df, test_idx)
        rf_oof[test_idx] = pred
        y_true = df.iloc[test_idx]["phi_lab"].to_numpy(dtype=np.float64)
        n_plugs = int(sum(1 for r in test_idx if int(r) in plug_set))
        fold_rows.append(
            {
                "fold_id": int(fold.fold_id),
                "depth_min_m": float(fold.depth_min_m),
                "depth_max_m": float(fold.depth_max_m),
                "n_test": int(len(test_idx)),
                "n_plugs_in_test_block": n_plugs,
                "rmse_rf": _point_rmse(y_true, pred),
                "r2_rf": _point_r2(y_true, pred),
            }
        )
    return rf_oof, pd.DataFrame(fold_rows)


def export_clp_vs_rf_artifacts(
    cfg_run: PlugFixedRunConfig,
    fig_ctx: PlugFixedFigureContext,
    plug_rows: Sequence[int],
    run_paths: Clp861RunPaths,
    primary_seed: int = 7,
) -> Dict[str, Path]:
    """
    Pointwise OOF comparison: CLP stitched profile vs RF (aligned depth-block CV).

    Writes run tables and compare_rf_baseline/ copies.
    """
    df = load_logs_table(str(cfg_run.excel_path), "Logs")
    df = apply_depth_bounds(df, cfg_run.depth_min_m, cfg_run.depth_max_m)
    depth_m = df["depth_m"].to_numpy(dtype=np.float64)
    y_true = df["phi_lab"].to_numpy(dtype=np.float64)
    nrows = int(y_true.shape[0])

    methods = sorted(
        {k[1] for k in fig_ctx.parity_parts if k[0] == int(primary_seed)}
    )
    if not methods:
        raise ValueError("No CLP methods for seed {}.".format(primary_seed))

    rf_oof, rf_fold_df = collect_rf_oof_profiles(
        df,
        int(cfg_run.n_depth_blocks),
        int(cfg_run.rf_n_estimators),
        int(primary_seed),
        plug_rows,
    )

    folds = depth_block_splits(df.assign(**{"Depth(m)": df["depth_m"]}), cfg_run.n_depth_blocks)
    fold_id_by_row = np.full(nrows, -1, dtype=np.int64)
    for fold in folds:
        fold_id_by_row[np.asarray(fold.test_idx, dtype=np.int64)] = int(fold.fold_id)

    oof_df = pd.DataFrame(
        {
            "row_index": np.arange(nrows, dtype=np.int64),
            "depth_m": depth_m,
            "phi_lab": y_true,
            "fold_id_oof": fold_id_by_row,
            "rf_oof": rf_oof,
        }
    )

    cmp_fold_rows: List[dict] = []
    for method in methods:
        clp_oof = stitch_method_profile(fig_ctx, method, int(primary_seed))
        col = "clp_oof_{}".format(method.replace("_prior_csgm_plug_fixed", ""))
        oof_df[col] = clp_oof

        for fold in folds:
            test_idx = np.asarray(fold.test_idx, dtype=np.int64)
            yt = y_true[test_idx]
            rf_p = rf_oof[test_idx]
            clp_p = clp_oof[test_idx]
            cmp_fold_rows.append(
                {
                    "fold_id": int(fold.fold_id),
                    "depth_min_m": float(fold.depth_min_m),
                    "depth_max_m": float(fold.depth_max_m),
                    "method": str(method),
                    "seed": int(primary_seed),
                    "n_test": int(len(test_idx)),
                    "rmse_rf": _point_rmse(yt, rf_p),
                    "r2_rf": _point_r2(yt, rf_p),
                    "rmse_clp": _point_rmse(yt, clp_p),
                    "r2_clp": _point_r2(yt, clp_p),
                    "rmse_delta_clp_minus_rf": _point_rmse(yt, clp_p)
                    - _point_rmse(yt, rf_p),
                    "n_plugs_in_test_block": int(
                        rf_fold_df.loc[
                            rf_fold_df["fold_id"] == int(fold.fold_id),
                            "n_plugs_in_test_block",
                        ].iloc[0]
                    ),
                }
            )

    cmp_fold_df = pd.DataFrame(cmp_fold_rows)
    global_rows: List[dict] = []
    for method in methods:
        clp_col = "clp_oof_{}".format(method.replace("_prior_csgm_plug_fixed", ""))
        clp_p = oof_df[clp_col].to_numpy(dtype=np.float64)
        m = np.isfinite(y_true) & np.isfinite(rf_oof) & np.isfinite(clp_p)
        global_rows.append(
            {
                "method": str(method),
                "seed": int(primary_seed),
                "rmse_rf": _point_rmse(y_true[m], rf_oof[m]),
                "r2_rf": _point_r2(y_true[m], rf_oof[m]),
                "rmse_clp": _point_rmse(y_true[m], clp_p[m]),
                "r2_clp": _point_r2(y_true[m], clp_p[m]),
                "rmse_delta_clp_minus_rf": _point_rmse(y_true[m], clp_p[m])
                - _point_rmse(y_true[m], rf_oof[m]),
                "n_points": int(np.sum(m)),
            }
        )
    cmp_global_df = pd.DataFrame(global_rows)

    run_paths.tables.mkdir(parents=True, exist_ok=True)
    oof_path = run_paths.tables / "oof_profile_predictions.csv"
    oof_df.to_csv(oof_path, index=False)

    cmp_fold_path = run_paths.tables / "clp_vs_rf_phi_lab_depth_block.csv"
    cmp_fold_df.to_csv(cmp_fold_path, index=False)

    cmp_global_path = run_paths.tables / "clp_vs_rf_phi_lab_global.csv"
    cmp_global_df.to_csv(cmp_global_path, index=False)

    rf_fold_path = run_paths.tables / "rf_oof_by_fold.csv"
    rf_fold_df.to_csv(rf_fold_path, index=False)

    cmp_root = compare_rf_baseline_dir()
    cmp_root.mkdir(parents=True, exist_ok=True)
    run_tag = run_paths.run_root.name
    cmp_fold_copy = cmp_root / "clp_vs_rf_phi_lab_depth_block_{}.csv".format(run_tag)
    cmp_global_copy = cmp_root / "clp_vs_rf_phi_lab_global_{}.csv".format(run_tag)
    cmp_fold_df.to_csv(cmp_fold_copy, index=False)
    cmp_global_df.to_csv(cmp_global_copy, index=False)

    note_path = cmp_root / "notes_{}.md".format(run_tag)
    best = cmp_global_df.sort_values("rmse_clp").iloc[0]
    note_path.write_text(
        "\n".join(
            [
                "# CLP vs RF (aligned depth-block OOF)",
                "",
                "run_id: `{}`".format(run_tag),
                "prior_types: {}".format(list(cfg_run.prior_types)),
                "primary_seed: {}".format(int(primary_seed)),
                "",
                "Global RMSE RF: {:.6f} pu".format(float(best["rmse_rf"])),
                "Global RMSE CLP (best): {:.6f} pu ({})".format(
                    float(best["rmse_clp"]), str(best["method"])
                ),
                "Delta CLP-RF: {:.6f} pu".format(float(best["rmse_delta_clp_minus_rf"])),
                "",
                "Artifacts:",
                "- `{}`".format(oof_path),
                "- `{}`".format(cmp_fold_path),
                "- `{}`".format(cmp_global_path),
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "oof_profile_predictions": oof_path,
        "clp_vs_rf_by_fold": cmp_fold_path,
        "clp_vs_rf_global": cmp_global_path,
        "rf_oof_by_fold": rf_fold_path,
        "compare_dir_fold": cmp_fold_copy,
        "compare_dir_global": cmp_global_copy,
        "compare_dir_notes": note_path,
    }


def execute_plug_fixed_run(
    run_paths: Clp861RunPaths,
    excel_path: Path,
    seeds: Sequence[int],
    prior_types: Sequence[str],
    csgm_ae_epochs: int = 200,
    compare_rf: bool = False,
    primary_seed: int = 7,
    device: Optional[str] = None,
) -> Path:
    """Run plug-fixed CLP and write artifacts."""
    run_paths.ensure_dirs()
    resolved_device = resolve_torch_device(device)
    plug_rows = plug_row_indices_unique(load_plug_measurement_rows())
    cfg = PlugFixedRunConfig(
        excel_path=excel_path,
        run_paths=run_paths,
        seeds=tuple(int(s) for s in seeds),
        prior_types=tuple(str(p) for p in prior_types),
        csgm_ae_epochs=int(csgm_ae_epochs),
        device=resolved_device,
    )
    meta = {
        "measurement_kind": "plug_fixed",
        "plug_global_row_indices": plug_rows,
        "n_plugs": len(load_plug_measurement_rows()),
        "n_unique_plug_rows": len(plug_rows),
        "window_len": cfg.window_len,
        "seeds": list(cfg.seeds),
        "prior_types": list(cfg.prior_types),
        "rf_prior": "rf" in [str(p).lower() for p in cfg.prior_types],
        "rf_residual": "rf_residual" in [str(p).lower() for p in cfg.prior_types],
        "rf_n_estimators": int(cfg.rf_n_estimators),
        "torch_device": resolved_device,
        "torch_device_label": torch_device_label(resolved_device),
    }
    (run_paths.run_root / "plug_fixed_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    detailed, fig_ctx = run_plug_fixed_depth_block_cv(cfg)
    detailed_path = run_paths.tables / "plug_fixed_detailed.csv"
    detailed.to_csv(detailed_path, index=False)

    summary = summarize_plug_fixed(detailed)
    summary_path = run_paths.tables / "summary_plug_fixed_by_fold.csv"
    summary.to_csv(summary_path, index=False)

    global_summary = (
        summary.groupby(["method", "seed"], as_index=False)
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std=("rmse_mean", "std"),
            n_folds=("fold_id", "nunique"),
        )
    )
    global_path = run_paths.tables / "summary_plug_fixed_global.csv"
    global_summary.to_csv(global_path, index=False)

    figure_paths = save_plug_fixed_figures(run_paths, fig_ctx, summary, primary_seed=int(primary_seed))
    figures_note = ", ".join(str(p.name) for p in figure_paths)

    compare_note = ""
    if compare_rf:
        cmp_paths = export_clp_vs_rf_artifacts(
            cfg,
            fig_ctx,
            plug_rows,
            run_paths,
            primary_seed=int(primary_seed),
        )
        compare_note = "compare_rf: {}".format(
            ", ".join(str(v.name) for v in cmp_paths.values())
        )

    proto_lines = [
        "CLP-861 plug-fixed b protocol",
        "measurement_kind: plug_fixed",
        "plug global row indices: {}".format(plug_rows),
        "validation: depth-block {} folds".format(cfg.n_depth_blocks),
        "prior_types: {}".format(list(cfg.prior_types)),
        "torch_device: {}".format(torch_device_label(resolved_device)),
        "detailed: {}".format(detailed_path),
        "summary: {}".format(global_path),
        "figures: {}".format(figures_note),
    ]
    if compare_note:
        proto_lines.append(compare_note)
    proto_lines.append("")
    (run_paths.run_root / "PROTOCOL.txt").write_text("\n".join(proto_lines), encoding="utf-8")
    return global_path
