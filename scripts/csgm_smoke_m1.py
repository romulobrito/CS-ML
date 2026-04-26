#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSGM M1 smoke: pure CSGM, AE prior trained on Y_train only.

At test time, ignore u and recover y from b using AE decoder as generative
prior:

  z* = argmin_z || M_eff G(z) - b_eff ||^2 + lam ||z||^2
  y_hat = G(z*) (denormalized)

with M_eff = M * scale[None,:] and b_eff = b_orig - M @ mean (so the
optimization runs in the normalized y space, consistent with the AE).

Single cell smoke: step=16, rho=0.1, seed=7. Compares against ml_only,
ae_regression_ub, and hybrid_lfista_joint stored numbers from the
production runs of the cross-well Vc benchmark.

Run from repo root:
  python scripts/csgm_smoke_m1.py

ASCII-only.
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multi_well_vc as mwv


TRAIN_PATH = os.path.join(_REPO_ROOT, "data", "F02-1,F03-2,F06-1_6logs_30dB.txt")
TEST_PATH = os.path.join(_REPO_ROOT, "data", "F03-4_6logs_30dB.txt")
OUT_DIR = os.path.join(
    _REPO_ROOT, "outputs", "cross_well_vc", "csgm", "smoke_m1"
)
TABLES_DIR = os.path.join(OUT_DIR, "tables")
LOG_PATH = os.path.join(OUT_DIR, "smoke.log")

CHANNELS: Tuple[str, ...] = ("sonic", "rhob", "gr", "ai", "vp")
WINDOW_LEN = 64
VAL_FRAC = 0.1
BASIS = "dct"

STEP = 16
RHO = 0.1
SEED = 7
NOISE_STD = 0.005

AE_LATENT = 16
AE_HIDDEN = 128
AE_EPOCHS = 200
AE_BATCH = 64
AE_LR = 1e-3
AE_WD = 1e-5

CSGM_ITERS = 400
CSGM_LR = 0.05
CSGM_LAM = 1e-3
CSGM_RESTARTS = 3


class _TinyAE(nn.Module):
    """MLP autoencoder consistent with direct_ub_baselines._TinyAE."""

    def __init__(self, n_output: int, h_dim: int, hidden: int) -> None:
        super().__init__()
        self.h_dim = int(h_dim)
        h = int(hidden)
        self.enc = nn.Sequential(
            nn.Linear(n_output, h),
            nn.ReLU(),
            nn.Linear(h, h_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(h_dim, h),
            nn.ReLU(),
            nn.Linear(h, n_output),
        )

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return self.enc(y)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(y))


def build_subsample_M(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(n, size=m, replace=False)
    M = np.zeros((m, n), dtype=np.float64)
    M[np.arange(m), idx] = 1.0
    return M


def make_b(
    Y: np.ndarray, M: np.ndarray, noise_std: float, rng: np.random.Generator
) -> np.ndarray:
    b_clean = Y @ M.T
    if noise_std > 0.0:
        return b_clean + noise_std * rng.normal(size=b_clean.shape)
    return b_clean


def train_ae(
    Y_train_n: np.ndarray,
    seed: int,
    latent: int,
    hidden: int,
    epochs: int,
    batch: int,
    lr: float,
    weight_decay: float,
    device: str,
) -> _TinyAE:
    torch.manual_seed(int(seed))
    n_out = int(Y_train_n.shape[1])
    ae = _TinyAE(n_out, latent, hidden).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr, weight_decay=weight_decay)
    yt = torch.tensor(Y_train_n, dtype=torch.float32, device=device)
    n = yt.shape[0]
    b = min(int(batch), n)
    ae.train()
    for epoch in range(int(epochs)):
        perm = torch.randperm(n, device=device)
        running = 0.0
        n_batches = 0
        for s in range(0, n, b):
            idx = perm[s : s + b]
            yb = yt[idx]
            opt.zero_grad()
            recon = ae(yb)
            loss = torch.mean((recon - yb) ** 2)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n_batches += 1
        if (epoch + 1) % 50 == 0:
            print("  AE epoch {:>3}/{} | mse={:.5f}".format(
                epoch + 1, epochs, running / max(n_batches, 1)
            ))
    return ae


def csgm_recover(
    ae: _TinyAE,
    M: np.ndarray,
    b_orig: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    n_iters: int,
    lr: float,
    n_restarts: int,
    device: str,
    seed: int,
) -> np.ndarray:
    """Batched CSGM recovery in normalized y space.

    Solves (in batched form for n_test samples):
      z* = argmin_z || M_eff G(z) - b_eff ||^2 + lam ||z||^2
      y_hat_n = G(z*)
      y_hat = y_hat_n * y_scale + y_mean

    Inputs:
      M       : (m, L) shared measurement matrix
      b_orig  : (n_test, m)
      y_mean  : (L,)
      y_scale : (L,)
    """
    ae.eval()
    n_test, m = b_orig.shape
    L = M.shape[1]
    d = ae.h_dim

    M_t = torch.tensor(M, dtype=torch.float32, device=device)
    mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    scale_t = torch.tensor(y_scale, dtype=torch.float32, device=device)
    b_t = torch.tensor(b_orig, dtype=torch.float32, device=device)
    M_eff = M_t * scale_t.unsqueeze(0)
    b_eff = b_t - (mean_t.unsqueeze(0) @ M_t.T)

    best_loss = None
    best_y_hat = None
    g = torch.Generator(device=device).manual_seed(int(seed))
    for r in range(int(n_restarts)):
        z = torch.zeros(n_test, d, device=device)
        if r > 0:
            z = z + 0.1 * torch.randn(z.shape, generator=g, device=device)
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=lr)
        for it in range(int(n_iters)):
            opt.zero_grad()
            y_n = ae.decode(z)
            residual = (y_n @ M_eff.T) - b_eff
            data_term = torch.sum(residual ** 2) / float(n_test)
            reg = lam * torch.sum(z ** 2) / float(n_test)
            loss = data_term + reg
            loss.backward()
            opt.step()
        with torch.no_grad():
            y_n = ae.decode(z)
            residual = (y_n @ M_eff.T) - b_eff
            final_data = torch.sum(residual ** 2) / float(n_test)
        cur_loss = float(final_data.item())
        if (best_loss is None) or (cur_loss < best_loss):
            best_loss = cur_loss
            with torch.no_grad():
                y_hat_n = ae.decode(z).detach().cpu().numpy()
                y_hat = y_hat_n * y_scale[None, :] + y_mean[None, :]
                best_y_hat = y_hat.copy()
    if best_y_hat is None:
        raise RuntimeError("CSGM recover failed.")
    print("  CSGM best data_term (normalized): {:.5f}".format(best_loss))
    return best_y_hat


def _rmse(yh: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((yh - y) ** 2)))


def _mae(yh: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.abs(yh - y)))


def _rel_l2(yh: np.ndarray, y: np.ndarray) -> float:
    num = np.linalg.norm(yh - y, axis=1)
    den = np.linalg.norm(y, axis=1)
    den = np.where(den < 1e-12, 1e-12, den)
    return float(np.mean(num / den))


def main() -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)
    device = "cpu"
    t0 = time.time()

    print("Loading cross-well data: step={}, channels={}".format(STEP, CHANNELS))
    data = mwv.build_cross_well_data_dict(
        train_path=TRAIN_PATH,
        test_path=TEST_PATH,
        target_name="vc",
        channels=CHANNELS,
        window_len=WINDOW_LEN,
        step=STEP,
        val_frac=VAL_FRAC,
        residual_basis=BASIS,
    )
    y_tr = np.asarray(data["Y_train"], dtype=np.float64)
    y_te = np.asarray(data["Y_test"], dtype=np.float64)
    meta = data["meta"]
    n_tr = int(meta["n_train"])
    n_te = int(meta["n_test"])
    L = int(meta["n_output"])
    print("n_train={}, n_test={}, L={}".format(n_tr, n_te, L))

    y_scaler = StandardScaler().fit(y_tr)
    y_tr_n = y_scaler.transform(y_tr)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)

    print("Training AE: latent={}, hidden={}, epochs={}".format(
        AE_LATENT, AE_HIDDEN, AE_EPOCHS
    ))
    ae = train_ae(
        y_tr_n,
        seed=SEED,
        latent=AE_LATENT,
        hidden=AE_HIDDEN,
        epochs=AE_EPOCHS,
        batch=AE_BATCH,
        lr=AE_LR,
        weight_decay=AE_WD,
        device=device,
    )

    ae.eval()
    with torch.no_grad():
        rec_tr = ae(torch.tensor(y_tr_n, dtype=torch.float32, device=device)).cpu().numpy()
        rec_loss_tr = _rmse(rec_tr * y_scale[None, :] + y_mean[None, :], y_tr)
    print("AE recon RMSE on Y_train: {:.5f}".format(rec_loss_tr))

    m_meas = max(2, int(round(RHO * L)))
    rng = np.random.default_rng(int(SEED))
    M = build_subsample_M(m_meas, L, rng)
    b = make_b(y_te, M, noise_std=NOISE_STD, rng=rng)
    print("Measurement: rho={}, m={}, M shape={}, b shape={}, noise_std={}".format(
        RHO, m_meas, M.shape, b.shape, NOISE_STD
    ))

    y_hat = csgm_recover(
        ae=ae,
        M=M,
        b_orig=b,
        y_mean=y_mean,
        y_scale=y_scale,
        lam=CSGM_LAM,
        n_iters=CSGM_ITERS,
        lr=CSGM_LR,
        n_restarts=CSGM_RESTARTS,
        device=device,
        seed=SEED,
    )

    rmse_csgm = _rmse(y_hat, y_te)
    mae_csgm = _mae(y_hat, y_te)
    rel_csgm = _rel_l2(y_hat, y_te)

    bench_path = os.path.join(
        _REPO_ROOT,
        "outputs",
        "cross_well_vc",
        "sir_cs_stress_lowdata",
        "runs",
        "prod_step{:02d}".format(STEP),
        "tables",
        "summary.csv",
    )
    bench: Dict[str, Dict[str, float]] = {}
    if os.path.isfile(bench_path):
        import pandas as pd

        df = pd.read_csv(bench_path)
        sub = df[df["measurement_ratio"] == RHO]
        for _, r in sub.iterrows():
            bench[str(r["method"])] = {
                "rmse": float(r["rmse_mean"]),
                "rel_l2": float(r["relative_l2_mean"]),
            }

    elapsed = time.time() - t0
    lines: List[str] = []
    lines.append("CSGM M1 smoke - Pure CSGM (AE prior on Y, ignores u)")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Cell: step={}, rho={}, seed={}".format(STEP, RHO, SEED))
    lines.append("Channels: {}".format(",".join(CHANNELS)))
    lines.append("AE: latent={}, hidden={}, epochs={}".format(
        AE_LATENT, AE_HIDDEN, AE_EPOCHS
    ))
    lines.append("CSGM: lam={}, lr={}, iters={}, restarts={}".format(
        CSGM_LAM, CSGM_LR, CSGM_ITERS, CSGM_RESTARTS
    ))
    lines.append("Sizes: n_train={}, n_test={}, L={}, m={}".format(
        n_tr, n_te, L, m_meas
    ))
    lines.append("AE recon RMSE on Y_train: {:.5f}".format(rec_loss_tr))
    lines.append("")
    lines.append("CSGM RMSE on test well: {:.5f}".format(rmse_csgm))
    lines.append("CSGM MAE  on test well: {:.5f}".format(mae_csgm))
    lines.append("CSGM Rel.L2 on test well: {:.5f}".format(rel_csgm))
    lines.append("")
    if bench:
        lines.append("Production benchmark for the same (step, rho) cell (3 seeds):")
        order = ["ae_regression_ub", "ml_only", "hybrid_lfista_joint",
                 "mlp_concat_ub", "pca_regression_ub"]
        for k in order:
            if k in bench:
                lines.append("  {:<22} rmse={:.5f}  rel_l2={:.5f}".format(
                    k, bench[k]["rmse"], bench[k]["rel_l2"]
                ))
        if "ae_regression_ub" in bench:
            ae_rmse = bench["ae_regression_ub"]["rmse"]
            gap_pct = 100.0 * (rmse_csgm - ae_rmse) / max(ae_rmse, 1e-12)
            lines.append("")
            lines.append("Gap vs ae_regression_ub: {:+.2f}%".format(gap_pct))
            if gap_pct < 0:
                verdict = (
                    "CSGM (pure) BEATS ae_regression_ub. Strong signal "
                    "to proceed to M2 (conditional CSGM)."
                )
            elif gap_pct < 20:
                verdict = (
                    "CSGM (pure) is competitive (within 20%). Proceed to "
                    "M2: adding u via conditional decoder should close gap."
                )
            elif gap_pct < 100:
                verdict = (
                    "CSGM (pure) is moderate (within 100%). M2 may help "
                    "but no guarantee. Consider before investing more."
                )
            else:
                verdict = (
                    "CSGM (pure) is far behind. AE manifold may not "
                    "generalize, or CSGM optimization is failing. "
                    "Inspect before M2."
                )
            lines.append("VERDICT: " + verdict)
    else:
        lines.append("(production benchmark CSV not found at: {})".format(bench_path))
    lines.append("")
    lines.append("Elapsed: {:.1f}s".format(elapsed))

    text = "\n".join(lines) + "\n"
    print()
    print(text)
    with open(LOG_PATH, "w", encoding="ascii") as f:
        f.write(text)

    np.savez_compressed(
        os.path.join(TABLES_DIR, "smoke_m1_arrays.npz"),
        Y_test=y_te,
        Y_hat_csgm=y_hat,
        M=M,
        b=b,
        y_mean=y_mean,
        y_scale=y_scale,
    )
    print("Saved arrays:", os.path.join(TABLES_DIR, "smoke_m1_arrays.npz"))
    print("Saved log:", LOG_PATH)


if __name__ == "__main__":
    main()
