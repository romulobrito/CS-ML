#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latent observability diagnostic for the 9-BRSA-1254-RJS plug holdout.

The analytical Jacobian is diag(scale) W2 diag(relu'(pre)) W1.
PyTorch ReLU backpropagates 0 at a preactivation of exactly 0, and the
analytical mask uses the same strict inequality.

The recovery objective for a group of windows is

  (||M G(z) - b||^2 + lambda ||z - z0||^2) / n_group,

with G in physical log10(k) units. Eigenvalues sigma(J_M)^2 + lambda match
that objective. A factor of two appears in the Hessian and cancels in the
condition number and in the effective-dimension ratio.

Outputs go to latent_observability_diagnostic and do not replace earlier runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    LATENT_DIM,
    OUT_DIR,
    U_COLUMNS,
    WINDOW_LEN,
)
from run_plug_holdout import (  # noqa: E402
    _fit_clp,
    _local_groups,
    _mean_at_plugs,
    _operator,
    _roles,
    _stack_windows,
    _window_starts,
)

OUT = OUT_DIR / "latent_observability_diagnostic"
BASE_SHA = "a25d94a544827cd906f188828eac7a0318fe4e21"
REPRODUCTION_TOLERANCE = 1.0e-4
RANK_RELATIVE_TOLERANCE = 1.0e-5
JACOBIAN_ABS_TOLERANCE = 1.0e-4


def _digest(path: Path) -> str:
    """SHA-256 of a frozen input file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoder_weights(ae: csgm.TinyAE) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return W1, b1, W2, b2 from the trained standardized decoder."""
    first = ae.dec[0]
    second = ae.dec[2]
    if not isinstance(first, torch.nn.Linear) or not isinstance(second, torch.nn.Linear):
        raise TypeError("Decoder is not the two-linear ReLU generator.")
    if not isinstance(ae.dec[1], torch.nn.ReLU):
        raise TypeError("Decoder hidden activation is not ReLU.")
    weight_1 = first.weight.detach().cpu().numpy().astype(np.float64)
    bias_1 = first.bias.detach().cpu().numpy().astype(np.float64)
    weight_2 = second.weight.detach().cpu().numpy().astype(np.float64)
    bias_2 = second.bias.detach().cpu().numpy().astype(np.float64)
    return weight_1, bias_1, weight_2, bias_2


def jacobian(ae: csgm.TinyAE, z_value: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Analytical Jacobian of physical G at one latent code, shape (window, latent)."""
    weight_1, bias_1, weight_2, _bias_2 = _decoder_weights(ae)
    latent = np.asarray(z_value, dtype=np.float64)
    if latent.shape != (LATENT_DIM,):
        raise ValueError("Jacobian expects one latent code of dimension {0}.".format(LATENT_DIM))
    preactivation = weight_1 @ latent + bias_1
    # Match torch.nn.ReLU.backward: the subgradient at 0 is 0.
    active = (preactivation > 0.0).astype(np.float64)
    hidden = active[:, None] * weight_1
    standardized = weight_2 @ hidden
    physical_scale = np.asarray(scale, dtype=np.float64)
    if physical_scale.shape != (WINDOW_LEN,):
        raise ValueError("Target scale does not match the window.")
    return physical_scale[:, None] * standardized


def _autograd_jacobian(
    ae: csgm.TinyAE,
    z_value: np.ndarray,
    scale: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    """PyTorch Jacobian of physical G, one output coordinate at a time."""
    ae.eval()
    latent = torch.tensor(np.asarray(z_value, dtype=np.float32), dtype=torch.float32, requires_grad=True)
    scale_t = torch.as_tensor(np.asarray(scale, dtype=np.float32), dtype=torch.float32)
    mean_t = torch.as_tensor(np.asarray(mean, dtype=np.float32), dtype=torch.float32)
    physical = ae.decode(latent) * scale_t + mean_t
    rows: List[np.ndarray] = []
    for coordinate in range(int(physical.shape[0])):
        gradient = torch.autograd.grad(physical[coordinate], latent, retain_graph=True)[0]
        rows.append(gradient.detach().cpu().numpy().astype(np.float64))
    return np.vstack(rows)


def _check_jacobian(ae: csgm.TinyAE, scale: np.ndarray, mean: np.ndarray) -> Dict[str, float]:
    """Compare the analytical Jacobian with autograd, including a ReLU kink."""
    weight_1, bias_1, _weight_2, _bias_2 = _decoder_weights(ae)
    generator = np.random.default_rng(0)
    samples = [np.zeros(LATENT_DIM, dtype=np.float64)]
    samples.extend(generator.normal(size=(6, LATENT_DIM)))
    column = int(np.argmax(np.abs(weight_1[0])))
    kink = np.zeros(LATENT_DIM, dtype=np.float64)
    if abs(float(weight_1[0, column])) < 1.0e-12:
        raise RuntimeError("First decoder row has no usable column for a kink test.")
    kink[column] = -float(bias_1[0]) / float(weight_1[0, column])
    samples.append(kink)
    max_abs = 0.0
    max_rel = 0.0
    kink_abs = 0.0
    for sample in samples:
        analytical = jacobian(ae, sample, scale)
        automatic = _autograd_jacobian(ae, sample, scale, mean)
        gap = np.abs(analytical - automatic)
        scale_gap = np.maximum(np.abs(analytical), 1.0e-8)
        max_abs = max(max_abs, float(np.max(gap)))
        max_rel = max(max_rel, float(np.max(gap / scale_gap)))
        if np.allclose(sample, kink):
            kink_abs = float(np.max(gap))
            preactivation = weight_1 @ sample + bias_1
            if abs(float(preactivation[0])) > 1.0e-6:
                raise RuntimeError("Kink sample did not land on a zero preactivation.")
    if max_abs > JACOBIAN_ABS_TOLERANCE:
        raise RuntimeError("Analytical Jacobian disagrees with autograd by {0}.".format(max_abs))
    return {
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "kink_abs_error": kink_abs,
        "relu_at_zero": "pytorch_and_analytical_use_zero_subgradient",
        "n_samples": float(len(samples)),
    }


def _singular_values(matrix: np.ndarray) -> np.ndarray:
    """Descending singular values padded to the latent dimension with structural zeros."""
    if int(matrix.shape[0]) == 0:
        return np.zeros(LATENT_DIM, dtype=np.float64)
    values = np.linalg.svd(matrix, compute_uv=False)
    if values.shape[0] > LATENT_DIM:
        raise RuntimeError("Singular spectrum is longer than the latent dimension.")
    padded = np.zeros(LATENT_DIM, dtype=np.float64)
    padded[: values.shape[0]] = values
    return padded


def spectral(jacobian_matrix: np.ndarray, observed: Sequence[int], lam: Optional[float]) -> Dict[str, float]:
    """Local spectrum of M dG/dz. Structural zeros are the missing latent slots."""
    local = [int(index) for index in observed]
    if jacobian_matrix.shape != (WINDOW_LEN, LATENT_DIM):
        raise ValueError("Jacobian shape is not window by latent.")
    if any(index < 0 or index >= WINDOW_LEN for index in local):
        raise ValueError("Observed offset is outside the window.")
    restricted = jacobian_matrix[local, :] if local else np.zeros((0, LATENT_DIM), dtype=np.float64)
    singular = _singular_values(restricted)
    n_observed = len(local)
    n_structural = int(LATENT_DIM - min(n_observed, LATENT_DIM))
    available = singular[: min(n_observed, LATENT_DIM)]
    threshold = max(float(singular[0]), 1.0e-12) * RANK_RELATIVE_TOLERANCE
    n_numerical_small = int(np.count_nonzero(available <= threshold)) if available.size else 0
    rank = int(np.count_nonzero(singular > threshold))
    observable = singular[singular > threshold]
    if observable.size == 0:
        observable_condition = float("nan")
    elif observable.size == 1:
        observable_condition = 1.0
    else:
        observable_condition = float(observable[0] / observable[-1])
    output: Dict[str, float] = {
        "rank": float(rank),
        "n_structural_zeros": float(n_structural),
        "n_numerical_small": float(n_numerical_small),
        "observable_condition": observable_condition,
        "sigma_max": float(singular[0]),
        "sigma_min": float(singular[-1]),
    }
    for index, value in enumerate(singular.tolist()):
        output["sigma_{0}".format(index + 1)] = float(value)
    if lam is None:
        output["effective_dim"] = float("nan")
        output["regularized_condition"] = float("nan")
        for index in range(LATENT_DIM):
            output["eigen_{0}".format(index + 1)] = float("nan")
        return output
    penalty = float(lam)
    if penalty < 0.0:
        raise ValueError("Lambda must be non-negative when recovery runs.")
    curvature = singular * singular + penalty
    output["effective_dim"] = float(np.sum((singular * singular) / curvature))
    output["regularized_condition"] = float(np.max(curvature) / np.min(curvature))
    for index, value in enumerate(curvature.tolist()):
        output["eigen_{0}".format(index + 1)] = float(value)
    return output


def recover_latents(
    ae: csgm.TinyAE,
    z0: np.ndarray,
    y_windows: np.ndarray,
    starts: Sequence[int],
    observed_plugs: set,
    mean: np.ndarray,
    scale: np.ndarray,
    lam: float,
    cfg: Config,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collective restart selection, matching csgm_recover_with_prior per mask group."""
    final_z = np.array(z0, dtype=np.float64, copy=True)
    final_y = csgm.decode_latent_prior(ae, z0, mean, scale, "cpu")
    for local, rows in _local_groups(starts, observed_plugs).items():
        if len(local) == 0:
            continue
        ids = np.asarray(rows, dtype=np.int64)
        operator = torch.tensor(_operator(local), dtype=torch.float32)
        measurements = torch.tensor(y_windows[ids][:, list(local)], dtype=torch.float32)
        origin = torch.tensor(z0[ids], dtype=torch.float32)
        mean_t = torch.tensor(mean, dtype=torch.float32)
        scale_t = torch.tensor(scale, dtype=torch.float32)
        measurement_operator = operator * scale_t.unsqueeze(0)
        target = measurements - (mean_t.unsqueeze(0) @ operator.T)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        best_loss = float("inf")
        best_z = origin.detach().cpu().numpy()
        best_y = final_y[ids]
        for restart in range(int(cfg.csgm_restarts)):
            if restart == 0:
                latent = origin.detach().clone()
            else:
                latent = origin.detach().clone() + 0.05 * torch.randn(
                    origin.shape,
                    generator=generator,
                )
            latent.requires_grad_(True)
            optimizer = torch.optim.Adam([latent], lr=float(cfg.csgm_opt_lr))
            for _step in range(int(cfg.csgm_iters)):
                optimizer.zero_grad()
                decoded = ae.decode(latent)
                residual = decoded @ measurement_operator.T - target
                loss = (residual.square().sum() + float(lam) * (latent - origin).square().sum()) / float(len(ids))
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                decoded = ae.decode(latent)
                residual = decoded @ measurement_operator.T - target
                loss_value = float(
                    (
                        residual.square().sum() + float(lam) * (latent - origin).square().sum()
                    ).item()
                    / float(len(ids))
                )
                if loss_value < best_loss:
                    best_loss = loss_value
                    best_z = latent.detach().cpu().numpy()
                    best_y = decoded.detach().cpu().numpy() * scale + mean
        final_z[ids] = best_z
        final_y[ids] = best_y
    return final_z, final_y


def _score(starts: Sequence[int], windows: np.ndarray, scored: np.ndarray, truth: np.ndarray) -> float:
    """Plug RMSE after averaging windows, the frozen scoring convention."""
    prediction = _mean_at_plugs(starts, windows, scored)
    ordered = np.asarray([prediction[int(plug)] for plug in scored.tolist()], dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)[scored]
    gap = ordered - reference
    return float(np.sqrt(np.mean(gap * gap)))


def _observation_rmse(truth_window: np.ndarray, prediction_window: np.ndarray, local: Sequence[int]) -> float:
    """RMSE on the measured coordinates of one window. Empty masks are undefined."""
    if len(local) == 0:
        return float("nan")
    index = list(local)
    gap = np.asarray(prediction_window, dtype=np.float64)[index] - np.asarray(truth_window, dtype=np.float64)[index]
    return float(np.sqrt(np.mean(gap * gap)))


def _prefix(point: str, spectrum: Dict[str, float]) -> Dict[str, float]:
    """Rename a spectrum so prior and recovered columns can share one row."""
    renamed: Dict[str, float] = {}
    for key, value in spectrum.items():
        renamed["{0}_{1}".format(key, point)] = float(value)
    return renamed


def _write_figures(windows: pd.DataFrame, cells: pd.DataFrame) -> None:
    """Figures use test windows. Block means are the dependent summary."""
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    test_windows = windows[windows["split"] == "test"]
    test_cells = cells[cells["split"] == "test"]
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    for axis, rho in zip(axes, (0.25, 0.50)):
        part = test_windows[np.isclose(test_windows["rho"], rho)]
        for component in range(1, LATENT_DIM + 1):
            column = "sigma_{0}_prior".format(component)
            grouped = [part.loc[part["fold_id"] == fold, column].to_numpy() for fold in range(3)]
            position = np.arange(3) + (component - 2.5) * 0.15
            axis.boxplot(grouped, positions=position, widths=0.12, manage_ticks=False)
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("block")
        axis.set_title("rho {0:.2f}".format(rho))
    axes[0].set_ylabel("singular value at z0")
    figure.tight_layout()
    figure.savefig(figure_dir / "singular_values_by_block.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for axis, rho in zip(axes, (0.25, 0.50)):
        part = test_windows[np.isclose(test_windows["rho"], rho)]
        for fold_id, sample in part.groupby("fold_id"):
            axis.scatter(sample["n_observed"], sample["rank_prior"], s=10, alpha=0.35, label="block {0:.0f}".format(fold_id))
        axis.set_xlabel("observations in the window")
        axis.set_title("rho {0:.2f}".format(rho))
    axes[0].set_ylabel("numerical rank at z0")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "rank_versus_observations.png", dpi=120)
    plt.close(figure)

    updated = test_windows[test_windows["updated"] > 0.5]
    figure, axis = plt.subplots(figsize=(5.4, 3.6))
    for fold_id, sample in updated.groupby("fold_id"):
        axis.scatter(
            sample["lambda_value"],
            sample["regularized_condition_prior"],
            s=12,
            alpha=0.4,
            label="block {0:.0f}".format(fold_id),
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("lambda")
    axis.set_ylabel("regularized condition at z0")
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "condition_versus_lambda.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for axis, rho in zip(axes, (0.25, 0.50)):
        part = test_cells[np.isclose(test_cells["rho"], rho)]
        axis.scatter(part["mean_latent_displacement"], part["delta_rmse"], s=28)
        for _, row in part.iterrows():
            axis.annotate("{0:.0f}".format(row["fold_id"]), (row["mean_latent_displacement"], row["delta_rmse"]), fontsize=7)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xlabel("mean ||zhat - z0||")
        axis.set_title("rho {0:.2f}".format(rho))
    axes[0].set_ylabel("CLP RMSE minus prior RMSE")
    figure.tight_layout()
    figure.savefig(figure_dir / "displacement_versus_delta_rmse.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    recovered_cells = test_cells[test_cells["updated"] > 0.5]
    axes[0].scatter(recovered_cells["mean_obs_rmse_prior"], recovered_cells["mean_obs_rmse_recovered"], s=28)
    limits = [
        float(np.nanmin(recovered_cells[["mean_obs_rmse_prior", "mean_obs_rmse_recovered"]].to_numpy())),
        float(np.nanmax(recovered_cells[["mean_obs_rmse_prior", "mean_obs_rmse_recovered"]].to_numpy())),
    ]
    axes[0].plot(limits, limits, color="black", linewidth=0.6)
    axes[0].set_xlabel("observation RMSE at z0")
    axes[0].set_ylabel("observation RMSE at zhat")
    axes[1].scatter(recovered_cells["mean_obs_rmse_recovered"] - recovered_cells["mean_obs_rmse_prior"], recovered_cells["delta_rmse"], s=28)
    axes[1].axhline(0.0, color="black", linewidth=0.6)
    axes[1].axvline(0.0, color="black", linewidth=0.6)
    axes[1].set_xlabel("change in observation RMSE")
    axes[1].set_ylabel("change in reserved RMSE")
    figure.tight_layout()
    figure.savefig(figure_dir / "observation_versus_reserved.png", dpi=120)
    plt.close(figure)

    summary = test_cells.groupby(["fold_id", "rho"], as_index=False)[["delta_rmse", "mean_effective_dim"]].mean()
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
    for axis, rho in zip(axes, (0.25, 0.50)):
        part = summary[np.isclose(summary["rho"], rho)]
        axis.bar(part["fold_id"] - 0.15, part["delta_rmse"], width=0.3, label="delta RMSE")
        twin = axis.twinx()
        twin.bar(part["fold_id"] + 0.15, part["mean_effective_dim"], width=0.3, label="effective dim", color="C1")
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("block")
        axis.set_title("rho {0:.2f}".format(rho))
    axes[0].set_ylabel("CLP minus prior")
    figure.tight_layout()
    figure.savefig(figure_dir / "blocks_delta_and_effective_dim.png", dpi=120)
    plt.close(figure)


def _write_environment() -> None:
    """Record the package versions used for this diagnostic."""
    import sklearn

    lines = [
        "python {0}".format(sys.version.replace("\n", " ")),
        "platform {0}".format(platform.platform()),
        "torch {0}".format(torch.__version__),
        "numpy {0}".format(np.__version__),
        "pandas {0}".format(pd.__version__),
        "sklearn {0}".format(sklearn.__version__),
        "base_sha {0}".format(BASE_SHA),
    ]
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Train the frozen decoders, reproduce CLP, and record latent spectra."""
    inputs = {
        "aligned": OUT_DIR / "tables" / "plug_log_aligned.csv",
        "split": OUT_DIR / "tables" / "split_by_plug.csv",
        "masks": OUT_DIR / "tables" / "masks.csv",
        "choices": OUT_DIR / "lambda_extension" / "tables" / "selection_by_cell.csv",
        "reference": OUT_DIR / "restart_oracle_diagnostic" / "config.json",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: {0}".format(", ".join(missing)))
    reference = json.loads(inputs["reference"].read_text(encoding="utf-8"))
    split_hash = _digest(inputs["split"])
    mask_hash = _digest(inputs["masks"])
    if split_hash != str(reference["split_sha256"]) or mask_hash != str(reference["masks_sha256"]):
        raise RuntimeError("Frozen split or masks do not match the restart-oracle audit.")
    aligned = pd.read_csv(inputs["aligned"])
    split = pd.read_csv(inputs["split"])
    masks = pd.read_csv(inputs["masks"])
    choices = pd.read_csv(inputs["choices"])
    depth = aligned["depth_m"].to_numpy(dtype=np.float64)
    y_vector = aligned["log10_k"].to_numpy(dtype=np.float64)
    u_matrix = aligned.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    if int(y_vector.shape[0]) != 161:
        raise ValueError("Expected 161 plugs.")
    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_device = "cpu"
    cfg.measurement_noise_std = 0.0
    torch.manual_seed(0)
    window_rows: List[Dict[str, float]] = []
    cell_rows: List[Dict[str, float]] = []
    log_lines = ["base_sha {0}".format(BASE_SHA), "split_sha256 {0}".format(split_hash), "masks_sha256 {0}".format(mask_hash)]
    jacobian_report: Optional[Dict[str, float]] = None
    for fold_id in range(3):
        fold = split[split["fold_id"] == fold_id]
        train_index = np.sort(fold.loc[fold["split"] == "train", "plug_index"].to_numpy(dtype=np.int64))
        val_index = np.sort(fold.loc[fold["split"] == "val", "plug_index"].to_numpy(dtype=np.int64))
        test_index = np.sort(fold.loc[fold["split"] == "test", "plug_index"].to_numpy(dtype=np.int64))
        train_starts = _window_starts(train_index)
        x_train, y_train = _stack_windows(u_matrix, y_vector, train_starts)
        print("fold", fold_id, "train_windows", len(train_starts), flush=True)
        for seed in (7, 23, 41):
            ae, prior, mean, scale = _fit_clp(x_train, y_train, int(seed), cfg)
            ae.eval()
            if jacobian_report is None:
                jacobian_report = _check_jacobian(ae, scale, mean)
                log_lines.append(
                    "jacobian_max_abs {0:.6g} kink {1:.6g}".format(
                        jacobian_report["max_abs_error"],
                        jacobian_report["kink_abs_error"],
                    )
                )
                print("jacobian_max_abs", jacobian_report["max_abs_error"], flush=True)
            for split_name, index in (("val", val_index), ("test", test_index)):
                starts = _window_starts(index)
                x_windows, y_windows = _stack_windows(u_matrix, y_vector, starts)
                z0 = np.asarray(prior.predict(x_windows), dtype=np.float64)
                prior_y = csgm.decode_latent_prior(ae, z0, mean, scale, "cpu")
                for rho in (0.25, 0.50):
                    selected = choices[
                        (choices["fold_id"] == fold_id)
                        & (choices["seed"] == seed)
                        & np.isclose(choices["rho"], rho)
                    ]
                    if int(selected.shape[0]) != 1:
                        raise RuntimeError("Missing a stored lambda choice.")
                    choice = selected.iloc[0]
                    roles = _roles(masks, fold_id, int(seed), float(rho), split_name)
                    observed = set(int(value) for value in roles["observed"].tolist())
                    scored = roles["scored"]
                    no_update = str(choice["extended_choice"]) == "no_update"
                    lam = None if no_update else float(choice["extended_lambda"])
                    if no_update:
                        z_hat = np.array(z0, copy=True)
                        recovered = np.array(prior_y, copy=True)
                    else:
                        z_hat, recovered = recover_latents(
                            ae,
                            z0,
                            y_windows,
                            starts,
                            observed,
                            mean,
                            scale,
                            float(lam),
                            cfg,
                            int(seed),
                        )
                    prior_rmse = _score(starts, prior_y, scored, y_vector)
                    clp_rmse = _score(starts, recovered, scored, y_vector)
                    if split_name == "test":
                        stored = float(choice["extended_test_rmse"])
                        if abs(clp_rmse - stored) > REPRODUCTION_TOLERANCE:
                            raise RuntimeError(
                                "Reproduction failed for fold {0} seed {1} rho {2}: {3} vs {4}".format(
                                    fold_id,
                                    seed,
                                    rho,
                                    clp_rmse,
                                    stored,
                                )
                            )
                    groups = _local_groups(starts, observed)
                    local_by_row: Dict[int, Tuple[int, ...]] = {}
                    for local, ids in groups.items():
                        for row_id in ids:
                            local_by_row[int(row_id)] = tuple(int(offset) for offset in local)
                    obs_prior: List[float] = []
                    obs_after: List[float] = []
                    for row_id, start in enumerate(starts):
                        local = local_by_row[row_id]
                        prior_spectrum = spectral(jacobian(ae, z0[row_id], scale), local, lam)
                        recovered_spectrum = spectral(jacobian(ae, z_hat[row_id], scale), local, lam)
                        obs_before = _observation_rmse(y_windows[row_id], prior_y[row_id], local)
                        obs_after_value = _observation_rmse(y_windows[row_id], recovered[row_id], local)
                        obs_prior.append(obs_before)
                        obs_after.append(obs_after_value)
                        row: Dict[str, float] = {
                            "split": split_name,
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "window_index": float(row_id),
                            "window_start": float(start),
                            "depth_m": float(depth[int(start)]),
                            "n_observed": float(len(local)),
                            "observed_offsets": ";".join(str(offset) for offset in local),
                            "updated": float(not no_update),
                            "lambda_value": float("nan") if lam is None else float(lam),
                            "latent_displacement": float(np.linalg.norm(z_hat[row_id] - z0[row_id])),
                            "obs_rmse_prior": obs_before,
                            "obs_rmse_recovered": obs_after_value,
                        }
                        row.update(_prefix("prior", prior_spectrum))
                        row.update(_prefix("recovered", recovered_spectrum))
                        window_rows.append(row)
                    cell_rows.append(
                        {
                            "split": split_name,
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "prior_rmse": prior_rmse,
                            "clp_rmse": clp_rmse,
                            "delta_rmse": clp_rmse - prior_rmse,
                            "updated": float(not no_update),
                            "lambda_value": float("nan") if lam is None else float(lam),
                            "mean_obs_rmse_prior": float(np.nanmean(np.asarray(obs_prior, dtype=np.float64))),
                            "mean_obs_rmse_recovered": float(np.nanmean(np.asarray(obs_after, dtype=np.float64))),
                            "stored_test_rmse": float(choice["extended_test_rmse"]) if split_name == "test" else float("nan"),
                        }
                    )
                    if split_name == "test":
                        message = "fold {0} seed {1} rho {2} prior {3:.5f} clp {4:.5f}".format(
                            fold_id,
                            seed,
                            rho,
                            prior_rmse,
                            clp_rmse,
                        )
                        print(message, flush=True)
                        log_lines.append(message)
    if jacobian_report is None:
        raise RuntimeError("Jacobian check did not run.")
    windows = pd.DataFrame(window_rows)
    cells = pd.DataFrame(cell_rows)
    test_cells = cells[cells["split"] == "test"]
    if int(test_cells.shape[0]) != 18:
        raise RuntimeError("Expected 18 test cells.")
    if not np.isfinite(test_cells["clp_rmse"]).all():
        raise RuntimeError("A test RMSE is not finite.")
    grouped_windows = windows.groupby(["split", "fold_id", "seed", "rho"], as_index=False).agg(
        underdetermined_fraction=("n_observed", lambda values: float(np.mean(np.asarray(values) < 4))),
        mean_rank=("rank_prior", "mean"),
        mean_effective_dim=("effective_dim_prior", "mean"),
        median_observable_condition=("observable_condition_prior", "median"),
        median_regularized_condition=("regularized_condition_prior", "median"),
        mean_latent_displacement=("latent_displacement", "mean"),
        fraction_numerical_deficiency=(
            "n_numerical_small_prior",
            lambda values: float(np.mean(np.asarray(values) > 0)),
        ),
    )
    enough = windows[windows["n_observed"] >= 4]
    if enough.shape[0] > 0:
        deficiency = enough.groupby(["split", "fold_id", "seed", "rho"], as_index=False).agg(
            fraction_rank_below_count=("rank_prior", lambda values: float(np.mean(np.asarray(values) < LATENT_DIM)))
        )
        grouped_windows = grouped_windows.merge(
            deficiency,
            on=["split", "fold_id", "seed", "rho"],
            how="left",
            validate="one_to_one",
        )
    cells = cells.merge(
        grouped_windows,
        on=["split", "fold_id", "seed", "rho"],
        how="left",
        validate="one_to_one",
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    windows.to_csv(OUT / "tables" / "window_metrics.csv", index=False)
    cells.to_csv(OUT / "tables" / "cell_metrics.csv", index=False)
    cells[cells["split"] == "test"].groupby(["fold_id", "rho"]).mean(numeric_only=True).to_csv(
        OUT / "tables" / "summary_by_block.csv"
    )
    cells[cells["split"] == "val"].groupby(["fold_id", "rho"]).mean(numeric_only=True).to_csv(
        OUT / "tables" / "summary_by_block_validation.csv"
    )
    metric_names = [
        "underdetermined_fraction",
        "mean_rank",
        "mean_effective_dim",
        "median_observable_condition",
        "median_regularized_condition",
        "mean_latent_displacement",
        "fraction_numerical_deficiency",
    ]
    correlations: List[Dict[str, float]] = []
    exploratory = cells[cells["split"] == "test"]
    for metric in metric_names:
        subset = exploratory[["delta_rmse", metric]].dropna()
        correlation = float(subset["delta_rmse"].corr(subset[metric])) if int(subset.shape[0]) >= 3 else float("nan")
        correlations.append(
            {
                "scope": "eighteen_cells_exploratory",
                "metric": metric,
                "correlation": correlation,
                "n_rows": float(subset.shape[0]),
                "pre_recovery": float(metric != "mean_latent_displacement"),
            }
        )
    block_means = exploratory.groupby(["fold_id", "rho"], as_index=False)[["delta_rmse"] + metric_names].mean()
    for metric in metric_names:
        subset = block_means[["delta_rmse", metric]].dropna()
        correlation = float(subset["delta_rmse"].corr(subset[metric])) if int(subset.shape[0]) >= 3 else float("nan")
        correlations.append(
            {
                "scope": "six_block_rho_means",
                "metric": metric,
                "correlation": correlation,
                "n_rows": float(subset.shape[0]),
                "pre_recovery": float(metric != "mean_latent_displacement"),
            }
        )
    pd.DataFrame(correlations).to_csv(OUT / "tables" / "correlations_exploratory.csv", index=False)
    reproduction_gap = float(np.max(np.abs(test_cells["clp_rmse"] - test_cells["stored_test_rmse"])))
    config = {
        "base_sha": BASE_SHA,
        "split_sha256": split_hash,
        "masks_sha256": mask_hash,
        "reproduction_max_abs_rmse": reproduction_gap,
        "jacobian_check": jacobian_report,
        "objective": "(||M G(z) - b||^2 + lambda ||z - z0||^2) / n_group",
        "curvature": "sigma(J_M)^2 + lambda",
        "relu_kink": "subgradient 0, matching torch.nn.ReLU",
        "rank_relative_tolerance": RANK_RELATIVE_TOLERANCE,
        "lambda_source": "lambda_extension validation choice",
        "oracle_not_used": True,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (OUT / "tables" / "jacobian_check.json").write_text(json.dumps(jacobian_report, indent=2) + "\n", encoding="utf-8")
    (OUT / "logs" / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# Latent observability diagnostic",
                "",
                "Well 9-BRSA-1254-RJS, branch research/brsa1254-latent-observability.",
                "Run: python scripts/brsa1254/audit_latent_observability.py",
                "See REPORT.md for the interpretation of this run.",
                "The oracle from the previous audit is not used to choose lambda or the recovery.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    _write_environment()
    _write_figures(windows, cells)
    print("reproduction_max_abs", reproduction_gap)
    print("saved", OUT)


if __name__ == "__main__":
    main()
