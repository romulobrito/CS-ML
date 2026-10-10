#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Noise sweep: interpolation versus CLP on the published F03-4 Vc grid.

Same windows, seeds, and Omega stream as R3-03. Only the measurement noise
scale changes. The decoder and the dense-log starting map are fit once per
(step, seed), because neither uses b. ASCII only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import direct_ub_baselines as dub
import multi_well_vc as mwv
from csgm_m2_module import (
    _make_prior,
    csgm_recover_with_prior,
    decode_latent_prior,
    encode_y,
    train_ae_generator,
)
from scripts.run_r3m03_simple_baselines import (
    CHANNELS,
    EXPECTED_COUNTS,
    TEST_PATH,
    TRAIN_PATH,
    VAL_FRAC,
    WINDOW_LEN,
    _interpolate,
    _mean_window_rmse,
    _omega_positions,
)
from sir_cs_pipeline import build_measurement_matrix
from sir_cs_pipeline_optimized import Config

OUT_DIR = ROOT_DIR / "outputs" / "reviewer_revision" / "r3m03_noise_sweep"
STEPS = (8, 16, 32)
SEEDS = (7, 23, 41)
RHOS = (0.05, 0.10, 0.20)
NOISE_STDS = (0.01, 0.05, 0.10, 0.20)
# Coarse grid: the paper values at the low end, then prior weights large
# enough for validation to ignore a noisy measurement.
LAMBDA_GRID = (
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
    100.0,
)


def _paper_config() -> Config:
    """Published CSGM settings, with the noise-sweep lambda grid."""
    cfg = Config()
    cfg.csgm_device = "cpu"
    cfg.csgm_prior_type = "ridge"
    cfg.csgm_latent_dim = 16
    cfg.csgm_hidden_dim = 128
    cfg.csgm_ae_epochs = 200
    cfg.csgm_batch_size = 64
    cfg.csgm_ae_lr = 1.0e-3
    cfg.csgm_weight_decay = 1.0e-5
    cfg.csgm_iters = 400
    cfg.csgm_restarts = 3
    cfg.csgm_opt_lr = 0.05
    cfg.csgm_lambda_grid = [float(item) for item in LAMBDA_GRID]
    return cfg


def _load_split(step: int) -> Dict[str, np.ndarray]:
    """Cross-well windows for one step. The split does not depend on the seed."""
    data = mwv.build_cross_well_data_dict(
        train_path=str(ROOT_DIR / TRAIN_PATH),
        test_path=str(ROOT_DIR / TEST_PATH),
        target_name="vc",
        channels=CHANNELS,
        window_len=WINDOW_LEN,
        step=int(step),
        val_frac=VAL_FRAC,
        val_embargo_windows=0,
        val_split_mode="tail",
        val_split_seed=0,
        residual_basis="dct",
    )
    expected = EXPECTED_COUNTS[int(step)]
    got = (
        int(data["Y_train"].shape[0]),
        int(data["Y_val"].shape[0]),
        int(data["Y_test"].shape[0]),
    )
    if got != expected:
        raise RuntimeError(
            "Step {0} window counts {1} differ from {2}.".format(step, got, expected)
        )
    return data


def _draw_unit_noise(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    rho: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Published Omega, plus one standard normal draw per split.

    Scaling these draws by the noise level reuses the same coordinates and
    the same noise pattern as make_B at every level.
    """
    rng = np.random.default_rng(int(seed))
    length = int(y_train.shape[1])
    m_obs = max(4, int(round(float(rho) * length)))
    measurement = build_measurement_matrix(m_obs, length, "subsample", rng)
    clean_train = np.asarray(y_train, dtype=np.float64) @ measurement.T
    clean_val = np.asarray(y_val, dtype=np.float64) @ measurement.T
    clean_test = np.asarray(y_test, dtype=np.float64) @ measurement.T
    unit_train = rng.normal(size=clean_train.shape)
    unit_val = rng.normal(size=clean_val.shape)
    unit_test = rng.normal(size=clean_test.shape)
    return (
        measurement,
        clean_train,
        clean_val,
        clean_test,
        unit_train,
        unit_val,
        unit_test,
    )


def _scale_noise(
    clean: np.ndarray, unit: np.ndarray, noise_std: float
) -> np.ndarray:
    """b = M y + noise_std * standard normal."""
    return np.asarray(clean, dtype=np.float64) + float(noise_std) * np.asarray(
        unit, dtype=np.float64
    )


def _fit_generator(
    cfg: Config, seed: int, data: Dict[str, np.ndarray]
) -> Tuple[object, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the decoder and the dense-log map. Both ignore b."""
    y_scaler = StandardScaler().fit(data["Y_train"])
    y_mean = np.asarray(y_scaler.mean_, dtype=np.float64)
    y_scale = np.asarray(y_scaler.scale_, dtype=np.float64)
    y_train_n = y_scaler.transform(data["Y_train"])
    ae = train_ae_generator(y_train_n, cfg, seed=int(seed), device="cpu")
    z_train = encode_y(ae, y_train_n, device="cpu")
    prior = _make_prior(cfg, seed=int(seed) + 7000).fit(data["X_train"], z_train)
    z0_val = np.asarray(prior.predict(data["X_val"]), dtype=np.float64)
    z0_test = np.asarray(prior.predict(data["X_test"]), dtype=np.float64)
    return ae, y_mean, y_scale, z0_val, z0_test


def _select_lambda(
    ae: object,
    measurement: np.ndarray,
    b_val: np.ndarray,
    y_val: np.ndarray,
    z0_val: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    seed: int,
    cfg: Config,
) -> Tuple[float, float]:
    """Lambda with the lowest validation mean-window RMSE."""
    best_lam = float(LAMBDA_GRID[0])
    best_score = float("inf")
    for lam in LAMBDA_GRID:
        y_hat = csgm_recover_with_prior(
            ae=ae,
            M=measurement,
            B=b_val,
            z0_np=z0_val,
            y_mean=y_mean,
            y_scale=y_scale,
            lam=float(lam),
            n_iters=int(cfg.csgm_iters),
            opt_lr=float(cfg.csgm_opt_lr),
            n_restarts=int(cfg.csgm_restarts),
            device="cpu",
            seed=int(seed),
        )
        score = _mean_window_rmse(y_val, y_hat)
        if score < best_score:
            best_score = float(score)
            best_lam = float(lam)
    return best_lam, best_score


def _write_protocol(target_std: float) -> None:
    """Record the sweep so the table can be read later."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "R3-03 noise sweep. Cross-well Vc, held-out F03-4.",
        "Steps {0}, seeds {1}, rho {2}.".format(STEPS, SEEDS, RHOS),
        "Noise std {0}.".format(NOISE_STDS),
        "Omega and the standard normal draws follow the published stream.",
        "Each noise level scales those same draws.",
        "Decoder and ridge prior are fit once per (step, seed).",
        "Lambda grid {0}.".format(LAMBDA_GRID),
        "Lambda is chosen by validation mean-window RMSE.",
        "prior_only is G(z0), with no measurement term.",
        "Test-window target std {0:.6f}.".format(float(target_std)),
        "Metric: mean of per-window RMSE.",
    ]
    (OUT_DIR / "PROTOCOL.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frozen = {
        "steps": list(STEPS),
        "seeds": list(SEEDS),
        "rhos": list(RHOS),
        "noise_stds": list(NOISE_STDS),
        "lambda_grid": list(LAMBDA_GRID),
        "target_std_test": float(target_std),
    }
    (OUT_DIR / "config_frozen.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )


def _summarize(detail: pd.DataFrame) -> pd.DataFrame:
    """Cell means and how many seeds have CLP below interpolation."""
    rows: List[Dict[str, float]] = []
    grouped = detail.groupby(["step", "measurement_ratio", "noise_std"], sort=True)
    for keys, frame in grouped:
        step, rho, noise_std = keys
        delta = frame["interp_rmse"] - frame["clp_rmse"]
        rows.append(
            {
                "step": int(step),
                "measurement_ratio": float(rho),
                "noise_std": float(noise_std),
                "interp_rmse": float(frame["interp_rmse"].mean()),
                "clp_rmse": float(frame["clp_rmse"].mean()),
                "prior_only_rmse": float(frame["prior_only_rmse"].mean()),
                "delta_interp_minus_clp": float(delta.mean()),
                "clp_wins": int(np.sum(frame["clp_rmse"] < frame["interp_rmse"])),
                "n_seeds": int(frame.shape[0]),
                "lambda_min": float(frame["lambda"].min()),
                "lambda_max": float(frame["lambda"].max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the sweep and write the seed-level and cell-level tables."""
    cfg = _paper_config()
    tables = OUT_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    detail_path = tables / "detail_by_seed.csv"
    records: List[Dict[str, float]] = []
    protocol_written = False
    for step in STEPS:
        data = _load_split(int(step))
        if not protocol_written:
            _write_protocol(float(np.std(data["Y_test"])))
            protocol_written = True
            print("target std", round(float(np.std(data["Y_test"])), 5))
        for seed in SEEDS:
            print("fit generator", "step", step, "seed", seed, flush=True)
            ae, y_mean, y_scale, z0_val, z0_test = _fit_generator(cfg, int(seed), data)
            prior_only = decode_latent_prior(
                ae, z0_test, y_mean, y_scale, device="cpu"
            )
            prior_score = _mean_window_rmse(data["Y_test"], prior_only)
            for rho in RHOS:
                (
                    measurement,
                    clean_train,
                    clean_val,
                    clean_test,
                    unit_train,
                    unit_val,
                    unit_test,
                ) = _draw_unit_noise(
                    data["Y_train"],
                    data["Y_val"],
                    data["Y_test"],
                    int(seed),
                    float(rho),
                )
                if clean_test.shape != unit_test.shape or clean_val.shape != unit_val.shape:
                    raise RuntimeError("Noise draw shape does not match M y.")
                del clean_train, unit_train
                positions = _omega_positions(measurement)
                length = int(data["Y_test"].shape[1])
                for noise_std in NOISE_STDS:
                    b_val = _scale_noise(clean_val, unit_val, float(noise_std))
                    b_test = _scale_noise(clean_test, unit_test, float(noise_std))
                    interp = _interpolate(b_test, positions, length)
                    interp_score = _mean_window_rmse(data["Y_test"], interp)
                    lam, val_score = _select_lambda(
                        ae,
                        measurement,
                        b_val,
                        data["Y_val"],
                        z0_val,
                        y_mean,
                        y_scale,
                        int(seed),
                        cfg,
                    )
                    clp = csgm_recover_with_prior(
                        ae=ae,
                        M=measurement,
                        B=b_test,
                        z0_np=z0_test,
                        y_mean=y_mean,
                        y_scale=y_scale,
                        lam=float(lam),
                        n_iters=int(cfg.csgm_iters),
                        opt_lr=float(cfg.csgm_opt_lr),
                        n_restarts=int(cfg.csgm_restarts),
                        device="cpu",
                        seed=int(seed),
                    )
                    clp_score = _mean_window_rmse(data["Y_test"], clp)
                    row = {
                        "step": int(step),
                        "seed": int(seed),
                        "measurement_ratio": float(rho),
                        "noise_std": float(noise_std),
                        "interp_rmse": float(interp_score),
                        "clp_rmse": float(clp_score),
                        "prior_only_rmse": float(prior_score),
                        "lambda": float(lam),
                        "val_rmse": float(val_score),
                        "m_obs": int(measurement.shape[0]),
                    }
                    records.append(row)
                    print(
                        "step",
                        step,
                        "seed",
                        seed,
                        "rho",
                        rho,
                        "noise",
                        noise_std,
                        "interp",
                        round(interp_score, 5),
                        "clp",
                        round(clp_score, 5),
                        "lambda",
                        lam,
                        flush=True,
                    )
            pd.DataFrame(records).to_csv(detail_path, index=False)
    detail = pd.DataFrame(records)
    detail.to_csv(detail_path, index=False)
    summary = _summarize(detail)
    summary.to_csv(tables / "summary_by_cell.csv", index=False)
    print(summary.to_string(index=False))
    print("wrote", detail_path)


if __name__ == "__main__":
    main()
