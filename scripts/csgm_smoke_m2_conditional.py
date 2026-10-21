#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSGM M2 smoke: conditional latent-prior CSGM.

The method keeps the AE decoder as a learned generative prior and uses u only
to predict a latent initial/prior code z0:

  AE:       y_n -> z -> y_n
  prior:    u -> z0
  test:     z* = argmin_z || M_eff G(z) - b_eff ||^2 + lam ||z - z0(u)||^2
  output:   y_hat = inverse_scale(G(z*))

Lambda is selected on validation data, then applied to the held-out test well.
The smoke uses the same cell as M1: step=16, rho=0.1, seed=7.

Run from repo root:
  python scripts/csgm_smoke_m2_conditional.py

ASCII-only.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import multi_well_vc as mwv


TRAIN_PATH = os.path.join(_REPO_ROOT, "data", "F02-1,F03-2,F06-1_6logs_30dB.txt")
TEST_PATH = os.path.join(_REPO_ROOT, "data", "F03-4_6logs_30dB.txt")
OUT_DIR = os.path.join(_REPO_ROOT, "outputs", "cross_well_vc", "csgm", "smoke_m2")
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
CSGM_RESTARTS = 3
LAMBDA_GRID: Tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


class TinyAE(nn.Module):
    """Small MLP autoencoder used as the CSGM generator."""

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
        """Map normalized y to latent z."""
        return self.enc(y)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent z to normalized y."""
        return self.dec(z)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Autoencode normalized y."""
        return self.decode(self.encode(y))


class RidgePrior:
    """Scaled ridge prior for u -> z0."""

    def __init__(self) -> None:
        self.x_scaler = StandardScaler()
        self.z_scaler = StandardScaler()
        self.model = Ridge(alpha=1.0, random_state=SEED)

    def fit(self, x: np.ndarray, z: np.ndarray) -> "RidgePrior":
        x_s = self.x_scaler.fit_transform(x)
        z_s = self.z_scaler.fit_transform(z)
        self.model.fit(x_s, z_s)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_s = self.x_scaler.transform(x)
        z_s = self.model.predict(x_s)
        return self.z_scaler.inverse_transform(z_s)


class MLPPrior:
    """Scaled MLP prior for u -> z0."""

    def __init__(self, seed: int) -> None:
        self.x_scaler = StandardScaler()
        self.z_scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=(128, 128),
            max_iter=500,
            learning_rate_init=1e-3,
            alpha=1e-4,
            early_stopping=True,
            random_state=int(seed),
        )

    def fit(self, x: np.ndarray, z: np.ndarray) -> "MLPPrior":
        x_s = self.x_scaler.fit_transform(x)
        z_s = self.z_scaler.fit_transform(z)
        self.model.fit(x_s, z_s)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_s = self.x_scaler.transform(x)
        z_s = self.model.predict(x_s)
        return self.z_scaler.inverse_transform(z_s)


def build_subsample_m(m: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """Build a shared row-subsampling matrix."""
    idx = rng.choice(n, size=m, replace=False)
    mat = np.zeros((m, n), dtype=np.float64)
    mat[np.arange(m), idx] = 1.0
    return mat


def make_b(y: np.ndarray, mat: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Create sparse measurements b = M y + eta."""
    b_clean = y @ mat.T
    if noise_std > 0.0:
        return b_clean + noise_std * rng.normal(size=b_clean.shape)
    return b_clean


def train_ae(y_train_n: np.ndarray, seed: int, device: str) -> TinyAE:
    """Train the AE generator in normalized y space."""
    torch.manual_seed(int(seed))
    n_out = int(y_train_n.shape[1])
    ae = TinyAE(n_out, AE_LATENT, AE_HIDDEN).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=AE_LR, weight_decay=AE_WD)
    yt = torch.tensor(y_train_n, dtype=torch.float32, device=device)
    n = int(yt.shape[0])
    batch = min(int(AE_BATCH), n)
    ae.train()
    for epoch in range(int(AE_EPOCHS)):
        perm = torch.randperm(n, device=device)
        total = 0.0
        count = 0
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            yb = yt[idx]
            opt.zero_grad()
            recon = ae(yb)
            loss = torch.mean((recon - yb) ** 2)
            loss.backward()
            opt.step()
            total += float(loss.item())
            count += 1
        if (epoch + 1) % 50 == 0:
            print("  AE epoch {:>3}/{} | mse={:.5f}".format(
                epoch + 1, AE_EPOCHS, total / max(count, 1)
            ))
    return ae


def encode_y(ae: TinyAE, y_n: np.ndarray, device: str) -> np.ndarray:
    """Encode normalized y to latent codes."""
    ae.eval()
    with torch.no_grad():
        z = ae.encode(torch.tensor(y_n, dtype=torch.float32, device=device))
    return z.cpu().numpy()


def csgm_recover_with_prior(
    ae: TinyAE,
    mat: np.ndarray,
    b_orig: np.ndarray,
    z0_np: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    n_iters: int,
    lr: float,
    n_restarts: int,
    device: str,
    seed: int,
) -> np.ndarray:
    """Recover y using measurement fit plus a latent prior around z0."""
    ae.eval()
    n_test = int(b_orig.shape[0])
    mat_t = torch.tensor(mat, dtype=torch.float32, device=device)
    mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)
    scale_t = torch.tensor(y_scale, dtype=torch.float32, device=device)
    b_t = torch.tensor(b_orig, dtype=torch.float32, device=device)
    z0 = torch.tensor(z0_np, dtype=torch.float32, device=device)

    mat_eff = mat_t * scale_t.unsqueeze(0)
    b_eff = b_t - (mean_t.unsqueeze(0) @ mat_t.T)

    best_loss = None
    best_y_hat = None
    gen = torch.Generator(device=device).manual_seed(int(seed))
    for restart in range(int(n_restarts)):
        if restart == 0:
            z = z0.clone().detach()
        else:
            z = z0.clone().detach() + 0.05 * torch.randn(
                z0.shape, generator=gen, device=device
            )
        z.requires_grad_(True)
        opt = torch.optim.Adam([z], lr=float(lr))
        for _ in range(int(n_iters)):
            opt.zero_grad()
            y_n = ae.decode(z)
            residual = (y_n @ mat_eff.T) - b_eff
            data_term = torch.sum(residual ** 2) / float(n_test)
            prior_term = float(lam) * torch.sum((z - z0) ** 2) / float(n_test)
            loss = data_term + prior_term
            loss.backward()
            opt.step()
        with torch.no_grad():
            y_n = ae.decode(z)
            residual = (y_n @ mat_eff.T) - b_eff
            data_term = torch.sum(residual ** 2) / float(n_test)
            prior_term = float(lam) * torch.sum((z - z0) ** 2) / float(n_test)
            final_loss = float((data_term + prior_term).item())
            if (best_loss is None) or (final_loss < best_loss):
                best_loss = final_loss
                y_hat_n = y_n.detach().cpu().numpy()
                best_y_hat = y_hat_n * y_scale[None, :] + y_mean[None, :]
    if best_y_hat is None:
        raise RuntimeError("CSGM recovery failed.")
    return best_y_hat


def rmse(yh: np.ndarray, y: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((yh - y) ** 2)))


def mae(yh: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(yh - y)))


def rel_l2(yh: np.ndarray, y: np.ndarray) -> float:
    """Mean per-window relative L2 error."""
    num = np.linalg.norm(yh - y, axis=1)
    den = np.linalg.norm(y, axis=1)
    den = np.where(den < 1e-12, 1e-12, den)
    return float(np.mean(num / den))


def metrics(yh: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Return the standard metric dict."""
    return {"rmse": rmse(yh, y), "mae": mae(yh, y), "rel_l2": rel_l2(yh, y)}


def load_benchmarks() -> Dict[str, Dict[str, float]]:
    """Load production benchmark values for the same step/rho cell."""
    path = os.path.join(
        _REPO_ROOT,
        "outputs",
        "cross_well_vc",
        "sir_cs_stress_lowdata",
        "runs",
        "prod_step{:02d}".format(STEP),
        "tables",
        "summary.csv",
    )
    out: Dict[str, Dict[str, float]] = {}
    if not os.path.isfile(path):
        return out
    df = pd.read_csv(path)
    sub = df[df["measurement_ratio"] == RHO]
    for _, row in sub.iterrows():
        out[str(row["method"])] = {
            "rmse": float(row["rmse_mean"]),
            "rel_l2": float(row["relative_l2_mean"]),
        }
    return out


def main() -> None:
    """Run the conditional CSGM smoke."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    device = "cpu"
    t0 = time.time()

    print("Loading data: step={}, rho={}, seed={}".format(STEP, RHO, SEED))
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
    x_train = np.asarray(data["X_train"], dtype=np.float64)
    x_val = np.asarray(data["X_val"], dtype=np.float64)
    x_test = np.asarray(data["X_test"], dtype=np.float64)
    y_train = np.asarray(data["Y_train"], dtype=np.float64)
    y_val = np.asarray(data["Y_val"], dtype=np.float64)
    y_test = np.asarray(data["Y_test"], dtype=np.float64)
    meta = data["meta"]
    n_train = int(meta["n_train"])
    n_val = int(meta["n_val"])
    n_test = int(meta["n_test"])
    n_output = int(meta["n_output"])
    print("n_train={}, n_val={}, n_test={}, L={}".format(
        n_train, n_val, n_test, n_output
    ))

    y_scaler = StandardScaler().fit(y_train)
    y_train_n = y_scaler.transform(y_train)
    y_val_n = y_scaler.transform(y_val)
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)

    print("Training AE generator...")
    ae = train_ae(y_train_n, seed=SEED, device=device)
    z_train = encode_y(ae, y_train_n, device=device)
    z_val_enc = encode_y(ae, y_val_n, device=device)

    with torch.no_grad():
        rec_train_n = ae(torch.tensor(y_train_n, dtype=torch.float32, device=device))
        rec_train = rec_train_n.cpu().numpy() * y_scale[None, :] + y_mean[None, :]
    rec_train_rmse = rmse(rec_train, y_train)

    rng = np.random.default_rng(int(SEED))
    m_meas = max(2, int(round(float(RHO) * float(n_output))))
    mat = build_subsample_m(m_meas, n_output, rng)
    b_val = make_b(y_val, mat, NOISE_STD, rng)
    b_test = make_b(y_test, mat, NOISE_STD, rng)

    prior_models = {
        "ridge_prior": RidgePrior().fit(x_train, z_train),
        "mlp_prior": MLPPrior(seed=SEED + 100).fit(x_train, z_train),
    }

    rows: List[Dict[str, object]] = []
    pred_bundle: Dict[str, np.ndarray] = {"Y_test": y_test, "M": mat, "b_test": b_test}
    for prior_name, prior in prior_models.items():
        z0_val = prior.predict(x_val)
        z0_test = prior.predict(x_test)

        # Context-only reconstruction before measurement refinement.
        with torch.no_grad():
            y_prior_n = ae.decode(
                torch.tensor(z0_test, dtype=torch.float32, device=device)
            ).cpu().numpy()
        y_prior = y_prior_n * y_scale[None, :] + y_mean[None, :]
        prior_metrics = metrics(y_prior, y_test)
        rows.append({
            "method": prior_name + "_decode_only",
            "lambda": np.nan,
            "split": "test",
            **prior_metrics,
        })
        pred_bundle[prior_name + "_decode_only"] = y_prior

        val_scores: List[Tuple[float, float]] = []
        for lam in LAMBDA_GRID:
            y_val_hat = csgm_recover_with_prior(
                ae=ae,
                mat=mat,
                b_orig=b_val,
                z0_np=z0_val,
                y_mean=y_mean,
                y_scale=y_scale,
                lam=float(lam),
                n_iters=CSGM_ITERS,
                lr=CSGM_LR,
                n_restarts=CSGM_RESTARTS,
                device=device,
                seed=SEED,
            )
            val_rmse = rmse(y_val_hat, y_val)
            val_scores.append((float(lam), val_rmse))
            rows.append({
                "method": prior_name + "_csgm",
                "lambda": float(lam),
                "split": "val",
                **metrics(y_val_hat, y_val),
            })
            print("  {} lam={:.4g} val_rmse={:.5f}".format(
                prior_name, float(lam), val_rmse
            ))

        best_lam = min(val_scores, key=lambda item: item[1])[0]
        y_test_hat = csgm_recover_with_prior(
            ae=ae,
            mat=mat,
            b_orig=b_test,
            z0_np=z0_test,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(best_lam),
            n_iters=CSGM_ITERS,
            lr=CSGM_LR,
            n_restarts=CSGM_RESTARTS,
            device=device,
            seed=SEED,
        )
        test_metrics = metrics(y_test_hat, y_test)
        rows.append({
            "method": prior_name + "_csgm",
            "lambda": float(best_lam),
            "split": "test",
            **test_metrics,
        })
        pred_bundle[prior_name + "_csgm"] = y_test_hat
        print("  {} best_lam={:.4g} test_rmse={:.5f}".format(
            prior_name, float(best_lam), test_metrics["rmse"]
        ))

    result_df = pd.DataFrame(rows)
    result_path = os.path.join(TABLES_DIR, "smoke_m2_results.csv")
    result_df.to_csv(result_path, index=False)

    np.savez_compressed(
        os.path.join(TABLES_DIR, "smoke_m2_arrays.npz"),
        **pred_bundle,
    )

    bench = load_benchmarks()
    test_rows = result_df[result_df["split"] == "test"].copy()
    best_row = test_rows.sort_values("rmse").iloc[0]
    elapsed = time.time() - t0

    lines: List[str] = []
    lines.append("CSGM M2 smoke - Conditional latent-prior CSGM")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Cell: step={}, rho={}, seed={}".format(STEP, RHO, SEED))
    lines.append("Sizes: n_train={}, n_val={}, n_test={}, L={}, m={}".format(
        n_train, n_val, n_test, n_output, m_meas
    ))
    lines.append("AE recon RMSE on Y_train: {:.5f}".format(rec_train_rmse))
    lines.append("AE latent validation self-code RMSE is diagnostic only.")
    lines.append("Encoded val latent norm mean: {:.5f}".format(
        float(np.mean(np.linalg.norm(z_val_enc, axis=1)))
    ))
    lines.append("")
    lines.append("M2 test results:")
    for _, row in test_rows.sort_values("rmse").iterrows():
        lines.append("  {:<28} lam={} rmse={:.5f} mae={:.5f} rel_l2={:.5f}".format(
            str(row["method"]),
            "n/a" if not np.isfinite(float(row["lambda"])) else "{:.4g}".format(float(row["lambda"])),
            float(row["rmse"]),
            float(row["mae"]),
            float(row["rel_l2"]),
        ))
    lines.append("")
    if bench:
        lines.append("Production benchmark for same step/rho cell (3 seeds):")
        for key in (
            "ae_regression_ub",
            "ml_only",
            "hybrid_lfista_joint",
            "mlp_concat_ub",
            "pca_regression_ub",
        ):
            if key in bench:
                lines.append("  {:<22} rmse={:.5f} rel_l2={:.5f}".format(
                    key, bench[key]["rmse"], bench[key]["rel_l2"]
                ))
        if "ae_regression_ub" in bench:
            ae_rmse = bench["ae_regression_ub"]["rmse"]
            gap_pct = 100.0 * (float(best_row["rmse"]) - ae_rmse) / max(ae_rmse, 1e-12)
            lines.append("")
            lines.append("Best M2 method: {}".format(str(best_row["method"])))
            lines.append("Gap vs ae_regression_ub: {:+.2f}%".format(gap_pct))
            if gap_pct < 0.0:
                verdict = "GO: M2 beats ae_regression_ub in the smoke cell."
            elif gap_pct <= 20.0:
                verdict = "WEAK GO: M2 is close enough to justify a small grid."
            else:
                verdict = "NO-GO: M2 is still too far from ae_regression_ub."
            lines.append("VERDICT: " + verdict)
    lines.append("")
    lines.append("Saved: " + result_path)
    lines.append("Elapsed: {:.1f}s".format(elapsed))
    text = "\n".join(lines) + "\n"
    print()
    print(text)
    with open(LOG_PATH, "w", encoding="ascii") as f:
        f.write(text)
    print("Saved log:", LOG_PATH)


if __name__ == "__main__":
    main()
