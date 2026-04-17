#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_lfista.py

Etapa 2 do roadmap SIR-CS: versão mínima em PyTorch com bloco LFISTA unrolled.

Objetivo
--------
Implementar uma versão *end-to-end trainable* do SIR-CS, substituindo o
solver clássico desacoplado por um bloco proximal desenrolado (LFISTA).

Modelo
------
    y = f_theta(u) + Psi alpha + xi
    b = M y + eta
    z = b - M f_theta(u)
    alpha^{k+1} = S_{tau_k}( v^k - eta_k A^T(A v^k - z) )
    y_hat = f_theta(u) + Psi alpha^K

onde A = M Psi e S_{tau} é o soft-thresholding.

Este script entrega:
1) gerador sintético compatível com o pipeline atual;
2) background MLP em PyTorch;
3) bloco LFISTA unrolled;
4) treino em duas fases:
   - fase A: background congelado, treina só o bloco LFISTA;
   - fase B: treino conjunto end-to-end;
5) avaliação contra:
   - ml_only_torch
   - hybrid_lfista_frozen
   - hybrid_lfista_joint

Observações
-----------
- Mantém o caso sintético básico do roadmap: measurement_kind="gaussian",
  residual_basis in {identity, dct}.
- Não implementa weighted nem TV/física extra.
- Foi desenhado para ser um ponto de partida limpo e legível.

Dependências
------------
- numpy
- pandas
- matplotlib
- torch
- scikit-learn (apenas para train/val/test scalers opcionais, aqui não usado)

Uso
---
    python sir_cs_lfista.py
    python sir_cs_lfista.py --profile explore
    python sir_cs_lfista.py --profile phase2_lfista
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 1) Configuração
# ============================================================


@dataclass
class LFISTAConfig:
    # perfis
    profile: Literal["phase2_lfista", "explore"] = "phase2_lfista"

    # reprodutibilidade
    seeds: List[int] = field(default_factory=lambda: [7, 13, 23])

    # dimensões
    n_train: int = 1200
    n_val: int = 300
    n_test: int = 300
    p_input: int = 12
    n_output: int = 128

    # residual sintético
    residual_basis: str = "identity"  # identity | dct
    residual_k: int = 6
    residual_amplitude: float = 1.2
    residual_mode: str = "support_from_u"  # support_from_u | random

    # medição
    measurement_kind: str = "gaussian"  # gaussian | subsample
    measurement_ratios: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])
    measurement_noise_std: float = 0.02
    output_noise_std: float = 0.01

    # background network
    bg_hidden: Tuple[int, int] = (128, 128)
    bg_dropout: float = 0.0

    # LFISTA
    lfista_steps: int = 8
    learn_step_sizes: bool = True
    learn_thresholds: bool = True
    init_step_scale: float = 1.0
    init_threshold: float = 1e-2
    use_momentum: bool = True

    # treino
    batch_size: int = 128
    num_epochs_bg: int = 80
    num_epochs_frozen: int = 60
    num_epochs_joint: int = 80
    lr_bg: float = 1e-3
    lr_frozen: float = 1e-3
    lr_joint: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 12

    # losses
    loss_alpha_weight: float = 0.0  # use >0 no sintético se quiser supervisionar alpha
    loss_l1_alpha_weight: float = 0.0

    # saída
    save_dir: str = "outputs/lfista_baseline"
    plots_subdir: str = "../paper/figures/lfista_baseline"
    n_example_plots: int = 3
    max_gt_scatter_points: int = 50000
    log_progress: bool = True

    # dispositivo
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def apply_profile(cfg: LFISTAConfig) -> None:
    if cfg.profile == "explore":
        cfg.seeds = [7]
        cfg.measurement_ratios = [0.3, 0.5]
        cfg.n_train = 600
        cfg.n_val = 80
        cfg.n_test = 100
        cfg.num_epochs_bg = 25
        cfg.num_epochs_frozen = 20
        cfg.num_epochs_joint = 25
        cfg.lfista_steps = 5
        cfg.save_dir = "outputs/lfista_explore"
        cfg.plots_subdir = "../paper/figures/lfista_explore"


# ============================================================
# 2) Utilidades matemáticas e sintéticas
# ============================================================


def log(msg: str, cfg: LFISTAConfig) -> None:
    if cfg.log_progress:
        print(msg, flush=True)


def orthonormal_dct_matrix(n: int) -> np.ndarray:
    Psi = np.zeros((n, n), dtype=float)
    factor0 = math.sqrt(1.0 / n)
    factor = math.sqrt(2.0 / n)
    k = np.arange(n)
    for i in range(n):
        if i == 0:
            Psi[:, i] = factor0
        else:
            Psi[:, i] = factor * np.cos(np.pi * (k + 0.5) * i / n)
    return Psi


def get_basis(n: int, basis_name: str) -> np.ndarray:
    if basis_name == "identity":
        return np.eye(n)
    if basis_name == "dct":
        return orthonormal_dct_matrix(n)
    raise ValueError(f"Base desconhecida: {basis_name}")


def build_measurement_matrix(m: int, n: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "gaussian":
        return rng.normal(0.0, 1.0 / math.sqrt(m), size=(m, n))
    if kind == "subsample":
        idx = rng.choice(n, size=m, replace=False)
        M = np.zeros((m, n))
        M[np.arange(m), idx] = 1.0
        return M
    raise ValueError(f"measurement_kind desconhecido: {kind}")


def random_feature_background(X: np.ndarray, n_output: int, rng: np.random.Generator) -> np.ndarray:
    p = X.shape[1]
    h = 32
    W1 = rng.normal(scale=0.8, size=(p, h))
    c1 = rng.normal(scale=0.2, size=(h,))
    H = np.tanh(X @ W1 + c1)
    W2 = rng.normal(scale=0.7 / math.sqrt(h), size=(h, n_output))
    y_bg = H @ W2

    grid = np.linspace(0.0, 1.0, n_output)
    coeff1 = (X[:, 0:1] + 0.5 * X[:, 1:2])
    coeff2 = (0.7 * X[:, 2:3] - 0.3 * X[:, 3:4])
    y_bg += 0.4 * np.sin(2 * np.pi * coeff1 * grid[None, :])
    y_bg += 0.25 * np.cos(2 * np.pi * coeff2 * grid[None, :])
    return y_bg


def choose_support_from_u(u: np.ndarray, n_output: int, k: int, rng: np.random.Generator, mode: str) -> np.ndarray:
    if mode == "random":
        return rng.choice(n_output, size=k, replace=False)

    score = 0.9 * u[0] + 0.7 * u[1] - 0.4 * u[2] + 0.3 * u[3]
    center = int(((math.tanh(score) + 1.0) / 2.0) * (n_output - 1))
    offsets = np.array([-12, -6, -3, 0, 4, 9, 14, 18])
    candidates = np.clip(center + offsets, 0, n_output - 1)
    base = list(np.unique(candidates))
    rng.shuffle(base)
    picked = base[: min(k, len(base))]
    while len(picked) < k:
        cand = int(rng.integers(0, n_output))
        if cand not in picked:
            picked.append(cand)
    return np.array(picked, dtype=int)


def generate_sparse_alpha(X: np.ndarray, n_output: int, k: int, amplitude: float, mode: str, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    Alpha = np.zeros((n, n_output))
    for i in range(n):
        supp = choose_support_from_u(X[i], n_output, k, rng, mode=mode)
        amps = amplitude * (
            0.6 * rng.choice([-1.0, 1.0], size=k)
            + 0.6 * np.tanh(X[i, 4:4 + min(k, X.shape[1] - 4)].mean() if X.shape[1] > 4 else X[i, 0])
        )
        amps = np.asarray(amps).reshape(-1)
        if amps.size < k:
            amps = np.pad(amps, (0, k - amps.size), mode="edge")
        amps = amps[:k] + 0.15 * rng.normal(size=k)
        Alpha[i, supp] = amps
    return Alpha


def make_dataset(cfg: LFISTAConfig, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_total = cfg.n_train + cfg.n_val + cfg.n_test
    X = rng.normal(size=(n_total, cfg.p_input))
    y_bg = random_feature_background(X, cfg.n_output, rng)
    Alpha = generate_sparse_alpha(
        X=X,
        n_output=cfg.n_output,
        k=cfg.residual_k,
        amplitude=cfg.residual_amplitude,
        mode=cfg.residual_mode,
        rng=rng,
    )
    Psi = get_basis(cfg.n_output, cfg.residual_basis)
    residual = Alpha @ Psi.T
    y = y_bg + residual + cfg.output_noise_std * rng.normal(size=y_bg.shape)
    return {
        "X_train": X[:cfg.n_train],
        "X_val": X[cfg.n_train:cfg.n_train + cfg.n_val],
        "X_test": X[cfg.n_train + cfg.n_val:],
        "Y_train": y[:cfg.n_train],
        "Y_val": y[cfg.n_train:cfg.n_train + cfg.n_val],
        "Y_test": y[cfg.n_train + cfg.n_val:],
        "Ybg_train": y_bg[:cfg.n_train],
        "Ybg_val": y_bg[cfg.n_train:cfg.n_train + cfg.n_val],
        "Ybg_test": y_bg[cfg.n_train + cfg.n_val:],
        "Alpha_train": Alpha[:cfg.n_train],
        "Alpha_val": Alpha[cfg.n_train:cfg.n_train + cfg.n_val],
        "Alpha_test": Alpha[cfg.n_train + cfg.n_val:],
        "Psi": Psi,
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.linalg.norm(y_true - y_pred) / (np.linalg.norm(y_true) + 1e-12))


def support_f1(alpha_true: np.ndarray, alpha_hat: np.ndarray, threshold: float = 1e-6) -> float:
    true_support = np.abs(alpha_true) > threshold
    pred_support = np.abs(alpha_hat) > threshold
    tp = np.sum(true_support & pred_support)
    fp = np.sum(~true_support & pred_support)
    fn = np.sum(true_support & ~pred_support)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


# ============================================================
# 3) Módulos PyTorch
# ============================================================


class BackgroundMLP(nn.Module):
    def __init__(self, p_input: int, n_output: int, hidden: Tuple[int, int], dropout: float = 0.0):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(p_input, h1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, n_output),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LFISTAUnrolled(nn.Module):
    """Bloco mínimo de LFISTA unrolled com parâmetros aprendíveis por camada.

    Resolve aproximadamente o residual sparse via K iterações:
        alpha^{k+1} = S_{tau_k}(v^k - eta_k A^T(A v^k - z))
    com aceleração tipo FISTA opcional.
    """

    def __init__(
        self,
        A: torch.Tensor,
        K: int,
        init_step_scale: float = 1.0,
        init_threshold: float = 1e-2,
        learn_step_sizes: bool = True,
        learn_thresholds: bool = True,
        use_momentum: bool = True,
    ):
        super().__init__()
        if A.dim() != 2:
            raise ValueError("A deve ser 2D")
        self.register_buffer("A", A)
        self.register_buffer("AT", A.transpose(0, 1))
        self.K = int(K)
        self.use_momentum = bool(use_momentum)

        # Lipschitz do gradiente de 0.5||A a - z||^2
        with torch.no_grad():
            svals = torch.linalg.svdvals(A)
            L = float((svals.max() ** 2).item()) + 1e-12
        self.register_buffer("L", torch.tensor(L, dtype=A.dtype, device=A.device))

        init_eta = float(init_step_scale) / L
        init_tau = float(init_threshold)

        eta0 = torch.full((self.K,), init_eta, dtype=A.dtype, device=A.device)
        tau0 = torch.full((self.K,), init_tau, dtype=A.dtype, device=A.device)

        if learn_step_sizes:
            self.log_eta = nn.Parameter(torch.log(torch.clamp(eta0, min=1e-8)))
        else:
            self.register_buffer("log_eta", torch.log(torch.clamp(eta0, min=1e-8)))

        if learn_thresholds:
            self.log_tau = nn.Parameter(torch.log(torch.clamp(tau0, min=1e-8)))
        else:
            self.register_buffer("log_tau", torch.log(torch.clamp(tau0, min=1e-8)))

    def _soft(self, x: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.relu(torch.abs(x) - thr)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        n_code = self.A.shape[1]
        alpha_prev = torch.zeros(batch_size, n_code, dtype=z.dtype, device=z.device)
        alpha = torch.zeros_like(alpha_prev)
        t_prev = torch.ones(batch_size, 1, dtype=z.dtype, device=z.device)

        for k in range(self.K):
            eta_k = torch.exp(self.log_eta[k]).clamp(min=1e-8, max=10.0)
            tau_k = torch.exp(self.log_tau[k]).clamp(min=1e-8, max=10.0)

            if self.use_momentum and k > 0:
                t_new = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * t_prev * t_prev))
                beta = (t_prev - 1.0) / t_new
                v = alpha + beta * (alpha - alpha_prev)
            else:
                t_new = t_prev
                v = alpha

            # grad = A^T (A v - z)
            Av = v @ self.A.transpose(0, 1)
            grad = (Av - z) @ self.A
            alpha_new = self._soft(v - eta_k * grad, eta_k * tau_k)

            alpha_prev = alpha
            alpha = alpha_new
            t_prev = t_new

        return alpha


class HybridLFISTA(nn.Module):
    def __init__(self, background: BackgroundMLP, sparse_block: LFISTAUnrolled, Psi: torch.Tensor, M: torch.Tensor):
        super().__init__()
        self.background = background
        self.sparse_block = sparse_block
        self.register_buffer("Psi", Psi)
        self.register_buffer("M", M)

    def forward(self, x: torch.Tensor, b: torch.Tensor) -> Dict[str, torch.Tensor]:
        y_bg = self.background(x)
        z = b - y_bg @ self.M.transpose(0, 1)
        alpha = self.sparse_block(z)
        y_hat = y_bg + alpha @ self.Psi.transpose(0, 1)
        return {"y_bg": y_bg, "z": z, "alpha": alpha, "y_hat": y_hat}


# ============================================================
# 4) Treino e avaliação
# ============================================================


def make_torch_loader(X: np.ndarray, Y: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def make_measurements(M: np.ndarray, Y: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    b = Y @ M.T
    if noise_std > 0.0:
        b = b + noise_std * rng.normal(size=b.shape)
    return b


def evaluate_bg_model(model: BackgroundMLP, X: np.ndarray, Y: np.ndarray, device: str) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        y = torch.tensor(Y, dtype=torch.float32, device=device)
        y_hat = model(x)
    y_np = y.cpu().numpy()
    y_hat_np = y_hat.cpu().numpy()
    mae = float(np.mean(np.abs(y_np - y_hat_np)))
    return {
        "rmse": rmse(y_np, y_hat_np),
        "mae": mae,
        "relative_l2": relative_l2(y_np, y_hat_np),
    }


def train_background(
    cfg: LFISTAConfig,
    model: BackgroundMLP,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
) -> BackgroundMLP:
    device = cfg.device
    model.to(device)
    train_loader = make_torch_loader(X_train, Y_train, cfg.batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_bg, weight_decay=cfg.weight_decay)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, cfg.num_epochs_bg + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yhat = model(xb)
            loss = F.mse_loss(yhat, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val = evaluate_bg_model(model, X_val, Y_val, device)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            log(f"[bg] epoch={epoch:03d} val_rmse={val['rmse']:.6f}", cfg)
        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def lfista_loss(
    out: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    alpha_true: Optional[torch.Tensor],
    cfg: LFISTAConfig,
) -> torch.Tensor:
    loss = F.mse_loss(out["y_hat"], y_true)
    if alpha_true is not None and cfg.loss_alpha_weight > 0.0:
        loss = loss + cfg.loss_alpha_weight * F.l1_loss(out["alpha"], alpha_true)
    if cfg.loss_l1_alpha_weight > 0.0:
        loss = loss + cfg.loss_l1_alpha_weight * torch.mean(torch.abs(out["alpha"]))
    return loss


def evaluate_hybrid(
    model: HybridLFISTA,
    X: np.ndarray,
    Y: np.ndarray,
    Alpha: np.ndarray,
    B: np.ndarray,
    cfg: LFISTAConfig,
) -> Dict[str, float]:
    device = cfg.device
    model.eval()
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        b = torch.tensor(B, dtype=torch.float32, device=device)
        out = model(x, b)
    y_hat = out["y_hat"].cpu().numpy()
    alpha_hat = out["alpha"].cpu().numpy()
    n = len(Y)
    sf1 = np.mean([support_f1(Alpha[i], alpha_hat[i]) for i in range(n)])
    return {
        "rmse": rmse(Y, y_hat),
        "mae": float(np.mean(np.abs(Y - y_hat))),
        "relative_l2": relative_l2(Y, y_hat),
        "support_f1": float(sf1),
    }


def train_lfista_stage(
    cfg: LFISTAConfig,
    hybrid: HybridLFISTA,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    Alpha_train: np.ndarray,
    B_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    Alpha_val: np.ndarray,
    B_val: np.ndarray,
    num_epochs: int,
    lr: float,
    train_background_flag: bool,
) -> HybridLFISTA:
    device = cfg.device
    hybrid.to(device)

    for p in hybrid.background.parameters():
        p.requires_grad = bool(train_background_flag)

    params = [p for p in hybrid.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.weight_decay)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train, dtype=torch.float32),
        torch.tensor(Alpha_train, dtype=torch.float32),
        torch.tensor(B_train, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    stage_name = "joint" if train_background_flag else "frozen"

    for epoch in range(1, num_epochs + 1):
        hybrid.train()
        for xb, yb, ab, bb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            ab = ab.to(device)
            bb = bb.to(device)
            out = hybrid(xb, bb)
            loss = lfista_loss(out, yb, ab, cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()

        val = evaluate_hybrid(hybrid, X_val, Y_val, Alpha_val, B_val, cfg)
        if val["rmse"] < best_val:
            best_val = val["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in hybrid.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0:
            log(f"[{stage_name}] epoch={epoch:03d} val_rmse={val['rmse']:.6f} val_f1={val['support_f1']:.4f}", cfg)
        if bad_epochs >= cfg.patience:
            break

    if best_state is not None:
        hybrid.load_state_dict(best_state)
    return hybrid


# ============================================================
# 5) Figuras
# ============================================================


def plots_dir(cfg: LFISTAConfig) -> str:
    p = os.path.join(cfg.save_dir, cfg.plots_subdir)
    os.makedirs(p, exist_ok=True)
    return p


def plot_metric(summary: pd.DataFrame, metric: str, save_path: str) -> None:
    colors = {
        "ml_only_torch": "#1f77b4",
        "hybrid_lfista_frozen": "#ff7f0e",
        "hybrid_lfista_joint": "#2ca02c",
    }
    plt.figure(figsize=(8.5, 5))
    for method in ["ml_only_torch", "hybrid_lfista_frozen", "hybrid_lfista_joint"]:
        sdf = summary[summary["method"] == method].sort_values("measurement_ratio")
        if len(sdf) == 0:
            continue
        plt.errorbar(
            sdf["measurement_ratio"].values,
            sdf[f"{metric}_mean"].values,
            yerr=sdf[f"{metric}_std_across_seeds"].values,
            marker="o",
            capsize=4,
            label=method,
            color=colors.get(method),
        )
    plt.xlabel("Measurement ratio m / N")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs measurement ratio")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


# ============================================================
# 6) Execução por (seed, rho)
# ============================================================


def summarize_per_seed(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["seed", "measurement_ratio", "method"])
        .agg(
            rmse_mean=("rmse", "mean"),
            mae_mean=("mae", "mean"),
            relative_l2_mean=("relative_l2", "mean"),
            support_f1_mean=("support_f1", "mean"),
            n_test_samples=("rmse", "size"),
        )
        .reset_index()
    )


def summarize_across_seeds(per_seed: pd.DataFrame) -> pd.DataFrame:
    def _std(s: pd.Series) -> float:
        return float(s.std(ddof=1)) if len(s) > 1 else 0.0
    return (
        per_seed.groupby(["measurement_ratio", "method"])
        .agg(
            rmse_mean=("rmse_mean", "mean"),
            rmse_std_across_seeds=("rmse_mean", _std),
            mae_mean=("mae_mean", "mean"),
            mae_std_across_seeds=("mae_mean", _std),
            relative_l2_mean=("relative_l2_mean", "mean"),
            relative_l2_std_across_seeds=("relative_l2_mean", _std),
            support_f1_mean=("support_f1_mean", "mean"),
            support_f1_std_across_seeds=("support_f1_mean", _std),
            n_seeds=("seed", "nunique"),
            n_test_samples_per_run=("n_test_samples", "first"),
        )
        .reset_index()
        .sort_values(["measurement_ratio", "method"])
        .reset_index(drop=True)
    )


def run_single_setting(cfg: LFISTAConfig, seed: int, measurement_ratio: float) -> pd.DataFrame:
    log(f"--- seed={seed} rho={measurement_ratio:.2f} ---", cfg)
    rng = np.random.default_rng(seed)
    data = make_dataset(cfg, seed)

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    Y_train, Y_val, Y_test = data["Y_train"], data["Y_val"], data["Y_test"]
    Alpha_train, Alpha_val, Alpha_test = data["Alpha_train"], data["Alpha_val"], data["Alpha_test"]
    Psi_np = data["Psi"]

    m = max(4, int(round(measurement_ratio * cfg.n_output)))
    M_np = build_measurement_matrix(m, cfg.n_output, cfg.measurement_kind, rng)
    B_train = make_measurements(M_np, Y_train, cfg.measurement_noise_std, rng)
    B_val = make_measurements(M_np, Y_val, cfg.measurement_noise_std, rng)
    B_test = make_measurements(M_np, Y_test, cfg.measurement_noise_std, rng)

    device = cfg.device
    Psi = torch.tensor(Psi_np, dtype=torch.float32, device=device)
    M = torch.tensor(M_np, dtype=torch.float32, device=device)
    A = M @ Psi

    # 1) background
    bg = BackgroundMLP(cfg.p_input, cfg.n_output, cfg.bg_hidden, cfg.bg_dropout)
    bg = train_background(cfg, bg, X_train, Y_train, X_val, Y_val)

    # 2) hybrid_lfista_frozen
    sparse_frozen = LFISTAUnrolled(
        A=A,
        K=cfg.lfista_steps,
        init_step_scale=cfg.init_step_scale,
        init_threshold=cfg.init_threshold,
        learn_step_sizes=cfg.learn_step_sizes,
        learn_thresholds=cfg.learn_thresholds,
        use_momentum=cfg.use_momentum,
    )
    hybrid_frozen = HybridLFISTA(bg, sparse_frozen, Psi, M)
    hybrid_frozen = train_lfista_stage(
        cfg,
        hybrid_frozen,
        X_train,
        Y_train,
        Alpha_train,
        B_train,
        X_val,
        Y_val,
        Alpha_val,
        B_val,
        num_epochs=cfg.num_epochs_frozen,
        lr=cfg.lr_frozen,
        train_background_flag=False,
    )

    # 3) hybrid_lfista_joint (inicializa com o frozen)
    sparse_joint = LFISTAUnrolled(
        A=A,
        K=cfg.lfista_steps,
        init_step_scale=cfg.init_step_scale,
        init_threshold=cfg.init_threshold,
        learn_step_sizes=cfg.learn_step_sizes,
        learn_thresholds=cfg.learn_thresholds,
        use_momentum=cfg.use_momentum,
    )
    hybrid_joint = HybridLFISTA(bg, sparse_joint, Psi, M)
    hybrid_joint.load_state_dict(hybrid_frozen.state_dict(), strict=False)
    hybrid_joint = train_lfista_stage(
        cfg,
        hybrid_joint,
        X_train,
        Y_train,
        Alpha_train,
        B_train,
        X_val,
        Y_val,
        Alpha_val,
        B_val,
        num_epochs=cfg.num_epochs_joint,
        lr=cfg.lr_joint,
        train_background_flag=True,
    )

    # avaliação por amostra
    rows = []

    # ml_only_torch
    bg.eval()
    with torch.no_grad():
        y_bg_test = bg(torch.tensor(X_test, dtype=torch.float32, device=device)).cpu().numpy()

    frozen_eval = evaluate_hybrid(hybrid_frozen, X_test, Y_test, Alpha_test, B_test, cfg)
    joint_eval = evaluate_hybrid(hybrid_joint, X_test, Y_test, Alpha_test, B_test, cfg)
    ml_eval = {
        "rmse": rmse(Y_test, y_bg_test),
        "mae": float(np.mean(np.abs(Y_test - y_bg_test))),
        "relative_l2": relative_l2(Y_test, y_bg_test),
        "support_f1": np.nan,
    }

    # detalhado por amostra
    with torch.no_grad():
        out_frozen = hybrid_frozen(
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(B_test, dtype=torch.float32, device=device),
        )
        out_joint = hybrid_joint(
            torch.tensor(X_test, dtype=torch.float32, device=device),
            torch.tensor(B_test, dtype=torch.float32, device=device),
        )
    yh_frozen = out_frozen["y_hat"].cpu().numpy()
    ah_frozen = out_frozen["alpha"].cpu().numpy()
    yh_joint = out_joint["y_hat"].cpu().numpy()
    ah_joint = out_joint["alpha"].cpu().numpy()

    for i in range(len(X_test)):
        rows.append({
            "seed": seed,
            "measurement_ratio": measurement_ratio,
            "method": "ml_only_torch",
            "sample_id": i,
            "rmse": rmse(Y_test[i], y_bg_test[i]),
            "mae": float(np.mean(np.abs(Y_test[i] - y_bg_test[i]))),
            "relative_l2": relative_l2(Y_test[i], y_bg_test[i]),
            "support_f1": np.nan,
            "m": m,
        })
        rows.append({
            "seed": seed,
            "measurement_ratio": measurement_ratio,
            "method": "hybrid_lfista_frozen",
            "sample_id": i,
            "rmse": rmse(Y_test[i], yh_frozen[i]),
            "mae": float(np.mean(np.abs(Y_test[i] - yh_frozen[i]))),
            "relative_l2": relative_l2(Y_test[i], yh_frozen[i]),
            "support_f1": support_f1(Alpha_test[i], ah_frozen[i]),
            "m": m,
        })
        rows.append({
            "seed": seed,
            "measurement_ratio": measurement_ratio,
            "method": "hybrid_lfista_joint",
            "sample_id": i,
            "rmse": rmse(Y_test[i], yh_joint[i]),
            "mae": float(np.mean(np.abs(Y_test[i] - yh_joint[i]))),
            "relative_l2": relative_l2(Y_test[i], yh_joint[i]),
            "support_f1": support_f1(Alpha_test[i], ah_joint[i]),
            "m": m,
        })

    log(
        f"[done] seed={seed} rho={measurement_ratio:.2f} | "
        f"ml_rmse={ml_eval['rmse']:.4f} | frozen_rmse={frozen_eval['rmse']:.4f} | joint_rmse={joint_eval['rmse']:.4f}",
        cfg,
    )
    return pd.DataFrame(rows)


# ============================================================
# 7) main
# ============================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SIR-CS LFISTA minimal baseline")
    ap.add_argument("--profile", type=str, default="phase2_lfista", choices=["phase2_lfista", "explore"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LFISTAConfig(profile=args.profile)
    apply_profile(cfg)
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    log(
        f"=== LFISTA start | profile={cfg.profile} | seeds={cfg.seeds} | "
        f"measurement_ratios={cfg.measurement_ratios} | device={cfg.device} ===",
        cfg,
    )

    t0 = time.time()
    dfs = []
    for seed in cfg.seeds:
        for rho in cfg.measurement_ratios:
            dfs.append(run_single_setting(cfg, seed, rho))

    detailed = pd.concat(dfs, ignore_index=True)
    per_seed = summarize_per_seed(detailed)
    summary = summarize_across_seeds(per_seed)

    detailed.to_csv(os.path.join(cfg.save_dir, "detailed_results.csv"), index=False)
    per_seed.to_csv(os.path.join(cfg.save_dir, "summary_by_seed.csv"), index=False)
    summary.to_csv(os.path.join(cfg.save_dir, "summary.csv"), index=False)

    pdir = plots_dir(cfg)
    plot_metric(summary, "rmse", os.path.join(pdir, "01_rmse_vs_measurement_ratio.png"))
    plot_metric(summary, "mae", os.path.join(pdir, "02_mae_vs_measurement_ratio.png"))
    plot_metric(summary, "relative_l2", os.path.join(pdir, "03_relative_l2_vs_measurement_ratio.png"))

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print("RESUMO LFISTA")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    print(f"\nArquivos salvos em: {os.path.abspath(cfg.save_dir)}")
    print(f"Tempo total: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
