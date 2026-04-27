#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
csgm_m2_module.py

Conditional CSGM M2 branch for direct-UB/cross-well benchmarks.

The method trains an AE decoder G(z) on standardized Y_train, fits a prior
h(u) -> z0, and performs test-time latent recovery:

    z_hat = argmin_z ||M_eff G(z) - b_eff||_2^2
            + lambda ||z - z0(u)||_2^2
    y_hat = inverse_scale(G(z_hat)).

ASCII-only source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import external_benchmarks as extb


@dataclass
class CSGMM2Result:
    """Container returned by the CSGM M2 benchmark branch."""

    df: pd.DataFrame
    predictions: np.ndarray
    selected_lambda: float
    val_score: float
    ae_recon_train_rmse: float
    prior_type: str


class TinyAE(nn.Module):
    """Small MLP autoencoder used as the CSGM generator."""

    def __init__(self, n_output: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        h = int(hidden_dim)
        self.enc = nn.Sequential(
            nn.Linear(int(n_output), h),
            nn.ReLU(),
            nn.Linear(h, int(latent_dim)),
        )
        self.dec = nn.Sequential(
            nn.Linear(int(latent_dim), h),
            nn.ReLU(),
            nn.Linear(h, int(n_output)),
        )

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        """Map standardized y to latent z."""
        return self.enc(y)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent z to standardized y."""
        return self.dec(z)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Autoencode standardized y."""
        return self.decode(self.encode(y))


class RidgePrior:
    """Scaled ridge prior for X -> z0."""

    def __init__(self, alpha: float = 1.0, random_state: int = 0) -> None:
        self.x_scaler = StandardScaler()
        self.z_scaler = StandardScaler()
        self.model = Ridge(alpha=float(alpha), random_state=int(random_state))

    def fit(self, x: np.ndarray, z: np.ndarray) -> "RidgePrior":
        """Fit the prior on train features and encoded train latents."""
        x_s = self.x_scaler.fit_transform(x)
        z_s = self.z_scaler.fit_transform(z)
        self.model.fit(x_s, z_s)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict latent prior codes."""
        x_s = self.x_scaler.transform(x)
        z_s = self.model.predict(x_s)
        return self.z_scaler.inverse_transform(z_s)


class MLPPrior:
    """Scaled MLP prior for X -> z0."""

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, int],
        max_iter: int,
        learning_rate_init: float,
        alpha: float,
        early_stopping: bool,
        random_state: int,
    ) -> None:
        self.x_scaler = StandardScaler()
        self.z_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=int(max_iter),
            learning_rate_init=float(learning_rate_init),
            alpha=float(alpha),
            early_stopping=bool(early_stopping),
            random_state=int(random_state),
        )

    def fit(self, x: np.ndarray, z: np.ndarray) -> "MLPPrior":
        """Fit the prior on train features and encoded train latents."""
        x_s = self.x_scaler.fit_transform(x)
        z_s = self.z_scaler.fit_transform(z)
        self.model.fit(x_s, z_s)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict latent prior codes."""
        x_s = self.x_scaler.transform(x)
        z_s = self.model.predict(x_s)
        return self.z_scaler.inverse_transform(z_s)


def train_ae_generator(
    y_train_n: np.ndarray,
    cfg: Any,
    seed: int,
    device: str,
) -> TinyAE:
    """Train the AE generator in standardized y space."""
    torch.manual_seed(int(seed))
    n_output = int(y_train_n.shape[1])
    ae = TinyAE(
        n_output=n_output,
        latent_dim=int(cfg.csgm_latent_dim),
        hidden_dim=int(cfg.csgm_hidden_dim),
    ).to(device)
    opt = torch.optim.Adam(
        ae.parameters(),
        lr=float(cfg.csgm_ae_lr),
        weight_decay=float(cfg.csgm_weight_decay),
    )
    yt = torch.tensor(y_train_n, dtype=torch.float32, device=device)
    n_samples = int(yt.shape[0])
    batch_size = min(int(cfg.csgm_batch_size), n_samples)
    ae.train()
    for _epoch in range(int(cfg.csgm_ae_epochs)):
        perm = torch.randperm(n_samples, device=device)
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            yb = yt[idx]
            opt.zero_grad()
            recon = ae(yb)
            loss = torch.mean((recon - yb) ** 2)
            loss.backward()
            opt.step()
    return ae


def encode_y(ae: TinyAE, y_n: np.ndarray, device: str) -> np.ndarray:
    """Encode standardized y rows to latent codes."""
    ae.eval()
    with torch.no_grad():
        z = ae.encode(torch.tensor(y_n, dtype=torch.float32, device=device))
    return z.cpu().numpy()


def csgm_recover_with_prior(
    ae: TinyAE,
    M: np.ndarray,
    B: np.ndarray,
    z0_np: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    n_iters: int,
    opt_lr: float,
    n_restarts: int,
    device: str,
    seed: int,
) -> np.ndarray:
    """Recover Y from B by optimizing latent z around z0."""
    ae.eval()
    n_rows = int(B.shape[0])
    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    scale_t = torch.tensor(y_scale, dtype=torch.float32, device=device)
    B_t = torch.tensor(B, dtype=torch.float32, device=device)
    z0 = torch.tensor(z0_np, dtype=torch.float32, device=device)

    M_eff = M_t * scale_t.unsqueeze(0)
    B_eff = B_t - (mean_t.unsqueeze(0) @ M_t.T)

    best_loss: Optional[float] = None
    best_y_hat: Optional[np.ndarray] = None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for restart in range(int(n_restarts)):
        if restart == 0:
            z = z0.clone().detach()
        else:
            z = z0.clone().detach() + 0.05 * torch.randn(
                z0.shape, generator=gen, device=device
            )
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=float(opt_lr))
        for _ in range(int(n_iters)):
            opt.zero_grad()
            y_n = ae.decode(z)
            residual = (y_n @ M_eff.T) - B_eff
            data_term = torch.sum(residual ** 2) / float(n_rows)
            prior_term = float(lam) * torch.sum((z - z0) ** 2) / float(n_rows)
            loss = data_term + prior_term
            loss.backward()
            opt.step()
        with torch.no_grad():
            y_n = ae.decode(z)
            residual = (y_n @ M_eff.T) - B_eff
            data_term = torch.sum(residual ** 2) / float(n_rows)
            prior_term = float(lam) * torch.sum((z - z0) ** 2) / float(n_rows)
            final_loss = float((data_term + prior_term).item())
            if best_loss is None or final_loss < best_loss:
                best_loss = final_loss
                y_hat_n = y_n.detach().cpu().numpy()
                best_y_hat = y_hat_n * y_scale[None, :] + y_mean[None, :]
    if best_y_hat is None:
        raise RuntimeError("CSGM M2 recovery failed.")
    return best_y_hat


def decode_latent_prior(
    ae: TinyAE,
    z0_np: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    device: str,
) -> np.ndarray:
    """Decode latent prior predictions without sparse-measurement refinement."""
    ae.eval()
    with torch.no_grad():
        z0 = torch.tensor(z0_np, dtype=torch.float32, device=device)
        y_n = ae.decode(z0).cpu().numpy()
    return y_n * y_scale[None, :] + y_mean[None, :]


def _rmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _select_score(y_pred: np.ndarray, y_true: np.ndarray, metric: str) -> float:
    m = str(metric).strip().lower()
    if m == "mae":
        return _mae(y_pred, y_true)
    return _rmse(y_pred, y_true)


def _make_prior(cfg: Any, seed: int) -> Any:
    prior_type = str(cfg.csgm_prior_type).strip().lower()
    if prior_type == "ridge":
        return RidgePrior(alpha=float(cfg.csgm_ridge_alpha), random_state=int(seed))
    if prior_type == "mlp":
        return MLPPrior(
            hidden_layer_sizes=tuple(cfg.csgm_prior_hidden),
            max_iter=int(cfg.csgm_prior_max_iter),
            learning_rate_init=float(cfg.csgm_prior_learning_rate_init),
            alpha=float(cfg.csgm_prior_alpha),
            early_stopping=bool(cfg.csgm_prior_early_stopping),
            random_state=int(seed),
        )
    raise ValueError("Unknown csgm_prior_type: {}".format(prior_type))


def run_csgm_m2_experiment_dataframe(
    cfg: Any,
    seed: int,
    measurement_ratio: float,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    Y_test: np.ndarray,
    Alpha_test: np.ndarray,
    M: np.ndarray,
    B_val: np.ndarray,
    B_test: np.ndarray,
) -> CSGMM2Result:
    """Run CSGM M2 and return per-sample metrics plus predictions."""
    device_raw = str(getattr(cfg, "csgm_device", "")).strip()
    device = device_raw if device_raw else "cpu"
    y_scaler = StandardScaler().fit(Y_train)
    y_train_n = y_scaler.transform(Y_train)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)

    ae = train_ae_generator(y_train_n, cfg, seed=seed, device=device)
    z_train = encode_y(ae, y_train_n, device=device)
    prior = _make_prior(cfg, seed=int(seed) + 7000).fit(X_train, z_train)
    z0_val = prior.predict(X_val)
    z0_test = prior.predict(X_test)

    with torch.no_grad():
        rec_n = ae(torch.tensor(y_train_n, dtype=torch.float32, device=device)).cpu().numpy()
    rec_train = rec_n * y_scale[None, :] + y_mean[None, :]
    ae_recon_train_rmse = _rmse(rec_train, Y_train)

    best_lam = float(cfg.csgm_lambda_grid[0])
    best_score = float("inf")
    for lam in list(cfg.csgm_lambda_grid):
        y_val_hat = csgm_recover_with_prior(
            ae=ae,
            M=M,
            B=B_val,
            z0_np=z0_val,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device=device,
            seed=int(seed),
        )
        score = _select_score(y_val_hat, Y_val, str(cfg.model_selection_metric))
        if score < best_score:
            best_score = float(score)
            best_lam = float(lam)

    y_test_hat = csgm_recover_with_prior(
        ae=ae,
        M=M,
        B=B_test,
        z0_np=z0_test,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=best_lam,
        n_iters=int(cfg.csgm_iters),
        opt_lr=float(cfg.csgm_opt_lr),
        n_restarts=int(cfg.csgm_restarts),
        device=device,
        seed=int(seed),
    )

    rows: List[Dict[str, Any]] = []
    nan_f = float("nan")
    m = int(M.shape[0])
    prior_type = str(cfg.csgm_prior_type).strip().lower()
    method = "{}_prior_csgm".format(prior_type)
    for i in range(int(Y_test.shape[0])):
        rows.append(
            extb.per_sample_metrics_row(
                int(seed),
                float(measurement_ratio),
                method,
                int(i),
                Y_test[i],
                y_test_hat[i],
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                best_lam,
                "csgm_m2",
                m,
                support_f1_override=nan_f,
            )
        )
        rows[-1]["val_score"] = best_score
        rows[-1]["ae_recon_train_rmse"] = ae_recon_train_rmse

    if bool(getattr(cfg, "run_csgm_ablations", False)):
        prior_val_hat = decode_latent_prior(
            ae=ae,
            z0_np=z0_val,
            y_mean=y_mean,
            y_scale=y_scale,
            device=device,
        )
        prior_test_hat = decode_latent_prior(
            ae=ae,
            z0_np=z0_test,
            y_mean=y_mean,
            y_scale=y_scale,
            device=device,
        )
        prior_score = _select_score(prior_val_hat, Y_val, str(cfg.model_selection_metric))
        prior_method = "{}_prior_only_decoder".format(prior_type)
        for i in range(int(Y_test.shape[0])):
            rows.append(
                extb.per_sample_metrics_row(
                    int(seed),
                    float(measurement_ratio),
                    prior_method,
                    int(i),
                    Y_test[i],
                    prior_test_hat[i],
                    Alpha_test[i],
                    np.zeros_like(Alpha_test[i]),
                    nan_f,
                    "csgm_prior_only",
                    m,
                    support_f1_override=nan_f,
                )
            )
            rows[-1]["val_score"] = float(prior_score)
            rows[-1]["ae_recon_train_rmse"] = ae_recon_train_rmse

        zero_val = np.zeros_like(z0_val)
        zero_test = np.zeros_like(z0_test)
        meas_val_hat = csgm_recover_with_prior(
            ae=ae,
            M=M,
            B=B_val,
            z0_np=zero_val,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=0.0,
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device=device,
            seed=int(seed) + 9100,
        )
        meas_score = _select_score(meas_val_hat, Y_val, str(cfg.model_selection_metric))
        meas_test_hat = csgm_recover_with_prior(
            ae=ae,
            M=M,
            B=B_test,
            z0_np=zero_test,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=0.0,
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device=device,
            seed=int(seed) + 9200,
        )
        for i in range(int(Y_test.shape[0])):
            rows.append(
                extb.per_sample_metrics_row(
                    int(seed),
                    float(measurement_ratio),
                    "measurement_only_csgm",
                    int(i),
                    Y_test[i],
                    meas_test_hat[i],
                    Alpha_test[i],
                    np.zeros_like(Alpha_test[i]),
                    0.0,
                    "csgm_measurement_only",
                    m,
                    support_f1_override=nan_f,
                )
            )
            rows[-1]["val_score"] = float(meas_score)
            rows[-1]["ae_recon_train_rmse"] = ae_recon_train_rmse
    return CSGMM2Result(
        df=pd.DataFrame(rows),
        predictions=np.asarray(y_test_hat, dtype=np.float64),
        selected_lambda=best_lam,
        val_score=best_score,
        ae_recon_train_rmse=ae_recon_train_rmse,
        prior_type=prior_type,
    )
