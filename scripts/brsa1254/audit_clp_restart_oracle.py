#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Diagnostic audit of CLP restart selection and decoder capacity.

Run from the repository root:

  python scripts/brsa1254/audit_clp_restart_oracle.py

The collective recovery is the repository function. The individual rule is
local to this file and does not edit csgm_m2_module.py. Lambda comes from the
validation choice already stored by the lambda extension. The oracle uses the
true test windows and is not a baseline.

Config is taken from sir_cs_pipeline_optimized, the same class used by
run_plug_holdout.py. sir_cs_benchmark_direct_ub re-exports that class, and
importing the benchmark module is not required for this audit.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
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
    _mean_at_plugs,
    _recover_grouped,
    _roles,
    _stack_windows,
    _window_starts,
)

OUT = OUT_DIR / "restart_oracle_diagnostic"
ORACLE_STARTS = 12
ORACLE_ITERS = 600
REPRODUCTION_TOLERANCE = 1.0e-4
BASE_SHA = "f6b6159e910c2529a9f750057fcab676b6bcf769"

# One entry per recovery call. Empty windows never reach this function.
CALL_LOG: List[Dict[str, float]] = []


def _sha256(path: Path) -> str:
    """Hash a frozen input so the audit can show it was not rewritten."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def select_per_window(
    ae: csgm.TinyAE,
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
    """Same restarts and optimizer as the repository, then one choice per row.

    The batch loss matches csgm_recover_with_prior. The collective restart is
    the one with the smaller sum of row losses. The returned prediction keeps,
    for each window, the restart with the smaller row loss.
    """
    ae.eval()
    n_rows = int(B.shape[0])
    if n_rows < 1 or int(M.shape[0]) < 1:
        raise ValueError("Individual selection received an empty observation set.")
    dev = torch.device(device)
    operator = torch.as_tensor(M, dtype=torch.float32, device=dev)
    mean = torch.as_tensor(y_mean, dtype=torch.float32, device=dev)
    scale = torch.as_tensor(y_scale, dtype=torch.float32, device=dev)
    observed = torch.as_tensor(B, dtype=torch.float32, device=dev)
    z0 = torch.as_tensor(z0_np, dtype=torch.float32, device=dev)
    measurement = operator * scale.unsqueeze(0)
    target = observed - (mean.unsqueeze(0) @ operator.T)
    generator = torch.Generator(device=dev).manual_seed(int(seed))
    best_row_loss = np.full(n_rows, np.inf)
    best_pred = np.zeros((n_rows, int(mean.numel())), dtype=np.float64)
    best_data = np.full(n_rows, np.inf)
    total_losses: List[float] = []
    row_losses: List[np.ndarray] = []
    data_losses: List[np.ndarray] = []
    for restart in range(int(n_restarts)):
        if restart == 0:
            latent = z0.detach().clone()
        else:
            latent = z0.detach().clone() + 0.05 * torch.randn(
                z0.shape,
                generator=generator,
                device=dev,
            )
        latent.requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=float(opt_lr))
        for _step in range(int(n_iters)):
            optimizer.zero_grad()
            decoded = ae.decode(latent)
            residual = decoded @ measurement.T - target
            terms = residual.square().sum(dim=1) + float(lam) * (latent - z0).square().sum(dim=1)
            loss = terms.sum() / float(n_rows)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            decoded = ae.decode(latent)
            residual = decoded @ measurement.T - target
            data_row = residual.square().sum(dim=1).detach().cpu().numpy()
            terms_np = (
                data_row
                + float(lam) * (latent - z0).square().sum(dim=1).detach().cpu().numpy()
            )
            prediction = decoded.detach().cpu().numpy() * y_scale[None, :] + y_mean[None, :]
            better = terms_np < best_row_loss
            best_row_loss[better] = terms_np[better]
            best_data[better] = data_row[better]
            best_pred[better] = prediction[better]
            total_losses.append(float(np.mean(terms_np)))
            row_losses.append(np.asarray(terms_np, dtype=np.float64).copy())
            data_losses.append(np.asarray(data_row, dtype=np.float64).copy())
    collective_index = int(np.argmin([float(np.sum(values)) for values in row_losses]))
    # Per-window choice cannot raise the sum of the optimized row losses.
    if float(best_row_loss.sum()) > float(np.sum(row_losses[collective_index])) + 1.0e-5:
        raise RuntimeError("Individual selection increased the optimized objective.")
    CALL_LOG.append(
        {
            "n_rows": float(n_rows),
            "individual_objective": float(best_row_loss.sum()),
            "collective_objective": float(np.sum(row_losses[collective_index])),
            "individual_data": float(best_data.sum()),
            "collective_data": float(np.sum(data_losses[collective_index])),
        }
    )
    return best_pred


def oracle_windows(
    ae: csgm.TinyAE,
    y_true_windows: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    seed: int,
    starts: int = ORACLE_STARTS,
    iters: int = ORACLE_ITERS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Approximate full-target latent fit. Diagnostic only: it sees test targets."""
    ae.eval()
    truth = np.asarray(y_true_windows, dtype=np.float64)
    standardized = torch.as_tensor((truth - y_mean) / y_scale, dtype=torch.float32)
    best_cost = np.full(truth.shape[0], np.inf)
    best_pred = np.zeros_like(truth)
    winner = np.full(truth.shape[0], -1, dtype=np.int64)
    generator = np.random.default_rng(int(seed))
    for start in range(int(starts)):
        if start == 0:
            with torch.no_grad():
                initial = ae.encode(standardized).detach()
        elif start == 1:
            initial = torch.zeros((truth.shape[0], int(ae.latent_dim)))
        else:
            draw = generator.standard_normal((truth.shape[0], int(ae.latent_dim)))
            initial = torch.tensor(draw, dtype=torch.float32)
        latent = initial.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=0.03)
        mean_t = torch.as_tensor(y_mean, dtype=torch.float32)
        scale_t = torch.as_tensor(y_scale, dtype=torch.float32)
        truth_t = torch.as_tensor(truth, dtype=torch.float32)
        for _step in range(int(iters)):
            optimizer.zero_grad()
            prediction = ae.decode(latent) * scale_t + mean_t
            cost = (prediction - truth_t).square().sum(dim=1)
            cost.sum().backward()
            optimizer.step()
        with torch.no_grad():
            prediction_np = (ae.decode(latent) * scale_t + mean_t).numpy()
            cost_np = ((prediction_np - truth) ** 2).sum(axis=1)
            update = cost_np < best_cost
            best_cost[update] = cost_np[update]
            best_pred[update] = prediction_np[update]
            winner[update] = int(start)
    return best_pred, best_cost, winner


def _score(starts: Sequence[int], windows: np.ndarray, scored: np.ndarray, truth: np.ndarray) -> float:
    """RMSE after averaging every window that covers a reserved plug."""
    prediction = _mean_at_plugs(starts, windows, scored)
    ordered = np.asarray([prediction[int(plug)] for plug in scored.tolist()], dtype=np.float64)
    gap = ordered - np.asarray(truth, dtype=np.float64)
    return float(np.sqrt(np.mean(gap * gap)))


def _append_plugs(
    rows: List[Dict[str, float]],
    fold_id: int,
    seed: int,
    rho: float,
    method: str,
    starts: Sequence[int],
    windows: np.ndarray,
    scored: np.ndarray,
    depth: np.ndarray,
    y_vector: np.ndarray,
) -> None:
    """Store one reserved plug after window aggregation."""
    prediction = _mean_at_plugs(starts, windows, scored)
    for plug in scored.tolist():
        rows.append(
            {
                "fold_id": float(fold_id),
                "seed": float(seed),
                "rho": float(rho),
                "method": method,
                "plug_index": float(plug),
                "depth_m": float(depth[int(plug)]),
                "y_true": float(y_vector[int(plug)]),
                "y_pred": float(prediction[int(plug)]),
            }
        )


def _write_figures(detail: pd.DataFrame, cells: pd.DataFrame) -> None:
    """Write the five diagnostic figures from the scored plugs."""
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = ["prior_only", "clp_collective", "clp_individual", "oracle_approx"]
    labels = {
        "prior_only": "prior",
        "clp_collective": "collective",
        "clp_individual": "individual",
        "oracle_approx": "oracle",
    }
    for rho in (0.25, 0.50):
        part = detail[np.isclose(detail["rho"], rho)]
        figure, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharex=True, sharey=True)
        for fold_id, axis in enumerate(axes):
            block = part[part["fold_id"] == fold_id]
            for method in methods:
                sample = block[block["method"] == method]
                axis.scatter(
                    sample["y_true"],
                    sample["y_pred"],
                    s=8,
                    alpha=0.45,
                    label=labels[method],
                )
            limits = [
                float(np.min(block["y_true"])),
                float(np.max(block["y_true"])),
            ]
            axis.plot(limits, limits, color="black", linewidth=0.6)
            axis.set_title("block {0}".format(fold_id))
            axis.set_xlabel("true log10(k)")
        axes[0].set_ylabel("predicted log10(k)")
        axes[0].legend(fontsize=7, frameon=False)
        figure.suptitle("rho {0:.2f}".format(rho))
        figure.tight_layout()
        figure.savefig(figure_dir / "reconstruction_rho_{0:.2f}.png".format(rho), dpi=120)
        plt.close(figure)

        figure, axes = plt.subplots(3, 1, figsize=(8.0, 7.2), sharex=False)
        for fold_id, axis in enumerate(axes):
            block = part[part["fold_id"] == fold_id]
            for method in methods:
                sample = block[block["method"] == method]
                axis.scatter(
                    sample["depth_m"],
                    sample["y_pred"] - sample["y_true"],
                    s=8,
                    alpha=0.4,
                    label=labels[method],
                )
            axis.axhline(0.0, color="black", linewidth=0.6)
            axis.set_ylabel("error")
            axis.set_title("block {0}".format(fold_id))
        axes[0].legend(fontsize=7, frameon=False, ncol=4)
        axes[-1].set_xlabel("depth (m)")
        figure.suptitle("rho {0:.2f}".format(rho))
        figure.tight_layout()
        figure.savefig(figure_dir / "error_by_depth_rho_{0:.2f}.png".format(rho), dpi=120)
        plt.close(figure)

    means = (
        cells.groupby(["fold_id", "rho"], as_index=False)[
            ["prior_rmse", "clp_original_rmse", "clp_individual_rmse", "oracle_approx_rmse"]
        ]
        .mean()
    )
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), sharey=True)
    names = ["prior_rmse", "clp_original_rmse", "clp_individual_rmse", "oracle_approx_rmse"]
    tick_labels = ["prior", "collective", "individual", "oracle"]
    for axis, rho in zip(axes, (0.25, 0.50)):
        part = means[np.isclose(means["rho"], rho)]
        width = 0.18
        for offset, name in enumerate(names):
            axis.bar(
                part["fold_id"] + (offset - 1.5) * width,
                part[name],
                width=width,
                label=tick_labels[offset],
            )
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("block")
        axis.set_title("rho {0:.2f}".format(rho))
    axes[0].set_ylabel("mean RMSE over seeds")
    axes[0].legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "rmse_by_block.png", dpi=120)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    for axis, rho in zip(axes, (0.25, 0.50)):
        part = cells[np.isclose(cells["rho"], rho)]
        for fold_id, sample in part.groupby("fold_id"):
            axis.scatter(
                np.full(sample.shape[0], fold_id),
                sample["individual_minus_original"],
                s=28,
            )
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_xticks([0, 1, 2])
        axis.set_xlabel("block")
        axis.set_title("rho {0:.2f}".format(rho))
    axes[0].set_ylabel("individual RMSE minus collective RMSE")
    figure.tight_layout()
    figure.savefig(figure_dir / "paired_restart_difference.png", dpi=120)
    plt.close(figure)

    ran = cells[cells["recovery_ran"] > 0.5]
    figure, axis = plt.subplots(figsize=(5.2, 3.8))
    for rho, sample in ran.groupby("rho"):
        axis.scatter(
            sample["individual_data"] - sample["collective_data"],
            sample["individual_minus_original"],
            s=28,
            label="rho {0:.2f}".format(float(rho)),
        )
    axis.axhline(0.0, color="black", linewidth=0.6)
    axis.axvline(0.0, color="black", linewidth=0.6)
    axis.set_xlabel("individual minus collective observation loss")
    axis.set_ylabel("individual minus collective test RMSE")
    axis.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "objective_versus_reserved_rmse.png", dpi=120)
    plt.close(figure)


def _write_report(cells: pd.DataFrame, checks: pd.DataFrame, oracle_diag: pd.DataFrame, passed: bool) -> None:
    """Write the scientific report from the tables of this run."""
    lines = [
        "# CLP restart and decoder-capacity audit, well 9-BRSA-1254-RJS",
        "",
        "Base commit: {0}.".format(BASE_SHA),
        "Remote main matched that commit at the start of this audit. A later fetch is not implied.",
        "Lambda is the validation choice stored in lambda_extension/tables/selection_by_cell.csv.",
        "Collective recovery calls csgm_recover_with_prior. Individual selection is local to this audit.",
        "The repository groups windows that share an observation pattern before recovery.",
        "Collective selection is therefore the best restart inside each group, not one restart for the whole block.",
        "Individual selection chooses inside that same group, one window at a time, with the same objective.",
        "Empty observation windows stay at G(z0) in both rules.",
        "The oracle fits latent codes to the true test windows. It is not a method and it does not choose lambda.",
        "Oracle search: {0} starts, {1} Adam steps, step size 0.03. The first start encodes the true window.".format(
            ORACLE_STARTS,
            ORACLE_ITERS,
        ),
        "Reproduction tolerance: {0} RMSE against the stored extended-rule test score.".format(REPRODUCTION_TOLERANCE),
        "Reproduction status: {0}.".format("passed" if passed else "failed"),
        "Maximum absolute reproduction gap: {0:.6g}.".format(float(checks["difference"].abs().max())),
        "",
        "## Primary mean of the nine fold-seed RMSE values",
        "",
    ]
    primary = cells.groupby("rho")[
        ["prior_rmse", "clp_original_rmse", "clp_individual_rmse", "oracle_approx_rmse"]
    ].mean()
    lines.append("| rho | prior | collective | individual | oracle |")
    lines.append("|---:|---:|---:|---:|---:|")
    for rho, row in primary.iterrows():
        lines.append(
            "| {0:.2f} | {1:.3f} | {2:.3f} | {3:.3f} | {4:.3f} |".format(
                float(rho),
                float(row["prior_rmse"]),
                float(row["clp_original_rmse"]),
                float(row["clp_individual_rmse"]),
                float(row["oracle_approx_rmse"]),
            )
        )
    lines.extend(["", "## By block, mean over the three seeds", ""])
    lines.append("| block | rho | prior | collective | individual | oracle |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    grouped = cells.groupby(["fold_id", "rho"])[
        ["prior_rmse", "clp_original_rmse", "clp_individual_rmse", "oracle_approx_rmse"]
    ].mean()
    for key, row in grouped.iterrows():
        lines.append(
            "| {0:.0f} | {1:.2f} | {2:.3f} | {3:.3f} | {4:.3f} | {5:.3f} |".format(
                float(key[0]),
                float(key[1]),
                float(row["prior_rmse"]),
                float(row["clp_original_rmse"]),
                float(row["clp_individual_rmse"]),
                float(row["oracle_approx_rmse"]),
            )
        )
    ran = cells[cells["recovery_ran"] > 0.5]
    lines.extend(
        [
            "",
            "## Restart selection, recovery cells only",
            "",
            "Cells whose validation choice was no_update do not run recovery. Count: {0}.".format(
                int((cells["recovery_ran"] < 0.5).sum())
            ),
            "Recovery cells: {0}.".format(int(ran.shape[0])),
            "Individual objective sum is lower or equal in every recovery call by the script check.",
            "Mean objective sum, individual minus collective: {0:.6g}.".format(
                float((ran["individual_objective"] - ran["collective_objective"]).mean())
            ),
            "Mean observation-loss sum, individual minus collective: {0:.6g}.".format(
                float((ran["individual_data"] - ran["collective_data"]).mean())
            ),
            "Test RMSE lower for the individual rule: {0} of {1}.".format(
                int((ran["individual_minus_original"] < -1.0e-7).sum()),
                int(ran.shape[0]),
            ),
            "Test RMSE higher for the individual rule: {0} of {1}.".format(
                int((ran["individual_minus_original"] > 1.0e-7).sum()),
                int(ran.shape[0]),
            ),
            "Mean test RMSE, individual minus collective: {0:.6g}.".format(
                float(ran["individual_minus_original"].mean())
            ),
            "",
            "## Oracle",
            "",
            "Mean fraction of windows whose best start was the encoded truth: {0:.3f}.".format(
                float(oracle_diag["fraction_encode_start"].mean())
            ),
            "Mean fraction of windows updated by a start after the first: {0:.3f}.".format(
                float(oracle_diag["fraction_after_encode"].mean())
            ),
            "This is a multistart approximation, not a certified minimum.",
            "",
        ]
    )
    reference = OUT_DIR / "corrected_protocol" / "tables" / "summary_primary.csv"
    if reference.is_file():
        lines.extend(
            [
                "## Previously scored comparators, not refit",
                "",
                "Source: corrected_protocol/tables/summary_primary.csv.",
                "",
            ]
        )
        frame = pd.read_csv(reference)
        lines.append("| rho | method | mean RMSE |")
        lines.append("|---:|---|---:|")
        keep = {
            "prior_only",
            "clp",
            "interp",
            "smooth_constant",
            "direct_regression",
            "ae_ub",
            "mlp_ub",
        }
        for _, row in frame.iterrows():
            if str(row["method"]) not in keep:
                continue
            lines.append(
                "| {0:.2f} | {1} | {2:.3f} |".format(
                    float(row["rho"]),
                    str(row["method"]),
                    float(row["mean_rmse"]),
                )
            )
        lines.append("")
    lines.extend(
        [
            "## Reading for the manuscript",
            "",
            "The nine cells are three seeds inside three depth blocks. The block is the dependent unit.",
            "A lower observation loss from per-window restart selection is not, by itself, a test improvement.",
            "The oracle bounds only the decoder fit found by this search. It does not justify a claim in the main grid.",
            "This audit does not edit the manuscript.",
            "",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_environment(passed: bool) -> None:
    """Record package versions and the frozen-input hashes."""
    import numpy
    import sklearn

    lines = [
        "python {0}".format(sys.version.replace("\n", " ")),
        "platform {0}".format(platform.platform()),
        "torch {0}".format(torch.__version__),
        "numpy {0}".format(numpy.__version__),
        "pandas {0}".format(pd.__version__),
        "sklearn {0}".format(sklearn.__version__),
        "base_sha {0}".format(BASE_SHA),
        "reproduction_passed {0}".format(passed),
    ]
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = "unavailable"
    lines.append("git_head {0}".format(head))
    (OUT / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Retrain the conditional decoder and compare restart rules plus the oracle."""
    required = [
        OUT_DIR / "config_frozen.json",
        OUT_DIR / "tables" / "plug_log_aligned.csv",
        OUT_DIR / "tables" / "split_by_plug.csv",
        OUT_DIR / "tables" / "masks.csv",
        OUT_DIR / "lambda_extension" / "tables" / "selection_by_cell.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen inputs: {0}".format(", ".join(missing)))
    aligned = pd.read_csv(OUT_DIR / "tables" / "plug_log_aligned.csv")
    split = pd.read_csv(OUT_DIR / "tables" / "split_by_plug.csv")
    masks = pd.read_csv(OUT_DIR / "tables" / "masks.csv")
    choices = pd.read_csv(OUT_DIR / "lambda_extension" / "tables" / "selection_by_cell.csv")
    if int(aligned.shape[0]) != 161:
        raise ValueError("Expected 161 aligned plugs.")
    depth = aligned["depth_m"].to_numpy(dtype=np.float64)
    y_vector = aligned["log10_k"].to_numpy(dtype=np.float64)
    u_matrix = aligned.loc[:, list(U_COLUMNS)].to_numpy(dtype=np.float64)
    cfg = Config()
    cfg.n_output = WINDOW_LEN
    cfg.csgm_latent_dim = LATENT_DIM
    cfg.csgm_hidden_dim = HIDDEN_DIM
    cfg.csgm_ae_epochs = AE_EPOCHS
    cfg.csgm_iters = CSGM_ITERS
    cfg.csgm_restarts = CSGM_RESTARTS
    cfg.csgm_device = "cpu"
    cfg.measurement_noise_std = 0.0
    if int(cfg.csgm_restarts) != 2 or int(cfg.csgm_iters) != 200:
        raise RuntimeError("Recovery settings differ from the frozen experiment.")
    torch.manual_seed(0)
    original = csgm.csgm_recover_with_prior
    rows: List[Dict[str, float]] = []
    plugs: List[Dict[str, float]] = []
    checks: List[Dict[str, float]] = []
    oracle_rows: List[Dict[str, float]] = []
    log_lines = [
        "base_sha {0}".format(BASE_SHA),
        "masks_sha256 {0}".format(_sha256(OUT_DIR / "tables" / "masks.csv")),
        "split_sha256 {0}".format(_sha256(OUT_DIR / "tables" / "split_by_plug.csv")),
    ]
    try:
        for fold_id in range(3):
            fold = split[split["fold_id"] == fold_id]
            train_index = np.sort(fold.loc[fold["split"] == "train", "plug_index"].to_numpy(dtype=np.int64))
            test_index = np.sort(fold.loc[fold["split"] == "test", "plug_index"].to_numpy(dtype=np.int64))
            if set(train_index.tolist()) & set(test_index.tolist()):
                raise ValueError("Train and test plugs overlap.")
            train_starts = _window_starts(train_index)
            test_starts = _window_starts(test_index)
            x_train, y_train = _stack_windows(u_matrix, y_vector, train_starts)
            x_test, y_test = _stack_windows(u_matrix, y_vector, test_starts)
            print("fold", fold_id, "train_windows", len(train_starts), "test_windows", len(test_starts), flush=True)
            for seed in (7, 23, 41):
                ae, prior, y_mean, y_scale = _fit_clp(x_train, y_train, int(seed), cfg)
                oracle, oracle_cost, oracle_winner = oracle_windows(ae, y_test, y_mean, y_scale, int(seed))
                oracle_rows.append(
                    {
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "n_windows": float(y_test.shape[0]),
                        "mean_window_sse": float(np.mean(oracle_cost)),
                        "fraction_encode_start": float(np.mean(oracle_winner == 0)),
                        "fraction_after_encode": float(np.mean(oracle_winner > 0)),
                        "max_winner_start": float(np.max(oracle_winner)),
                    }
                )
                for rho in (0.25, 0.50):
                    selected = choices[
                        (choices["fold_id"] == fold_id)
                        & (choices["seed"] == seed)
                        & np.isclose(choices["rho"], rho)
                    ]
                    if int(selected.shape[0]) != 1:
                        raise RuntimeError("Expected one stored lambda choice for this cell.")
                    choice = selected.iloc[0]
                    roles = _roles(masks, fold_id, int(seed), float(rho), "test")
                    observed = set(int(value) for value in roles["observed"].tolist())
                    scored = roles["scored"]
                    truth = y_vector[scored]
                    prior_windows = csgm.decode_latent_prior(
                        ae,
                        np.asarray(prior.predict(x_test), dtype=np.float64),
                        y_mean,
                        y_scale,
                        "cpu",
                    )
                    prior_rmse = _score(test_starts, prior_windows, scored, truth)
                    oracle_rmse = _score(test_starts, oracle, scored, truth)
                    recovery_ran = 0.0
                    objective = {
                        "individual_objective": float("nan"),
                        "collective_objective": float("nan"),
                        "individual_data": float("nan"),
                        "collective_data": float("nan"),
                    }
                    if str(choice["extended_choice"]) == "no_update":
                        collective = prior_windows
                        individual = prior_windows
                        collective_rmse = prior_rmse
                        individual_rmse = prior_rmse
                    else:
                        lam = float(choice["extended_lambda"])
                        csgm.csgm_recover_with_prior = original
                        collective = _recover_grouped(
                            ae,
                            prior,
                            x_test,
                            y_test,
                            test_starts,
                            observed,
                            y_mean,
                            y_scale,
                            lam,
                            cfg,
                            int(seed),
                        )
                        CALL_LOG.clear()
                        csgm.csgm_recover_with_prior = select_per_window
                        individual = _recover_grouped(
                            ae,
                            prior,
                            x_test,
                            y_test,
                            test_starts,
                            observed,
                            y_mean,
                            y_scale,
                            lam,
                            cfg,
                            int(seed),
                        )
                        if not CALL_LOG:
                            raise RuntimeError("Individual selection was not called.")
                        recovery_ran = 1.0
                        objective = {
                            "individual_objective": float(sum(item["individual_objective"] for item in CALL_LOG)),
                            "collective_objective": float(sum(item["collective_objective"] for item in CALL_LOG)),
                            "individual_data": float(sum(item["individual_data"] for item in CALL_LOG)),
                            "collective_data": float(sum(item["collective_data"] for item in CALL_LOG)),
                        }
                        collective_rmse = _score(test_starts, collective, scored, truth)
                        individual_rmse = _score(test_starts, individual, scored, truth)
                    stored = float(choice["extended_test_rmse"])
                    checks.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "stored_rmse": stored,
                            "repeated_rmse": collective_rmse,
                            "difference": collective_rmse - stored,
                        }
                    )
                    rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "recovery_ran": recovery_ran,
                            "prior_rmse": prior_rmse,
                            "clp_original_rmse": collective_rmse,
                            "clp_individual_rmse": individual_rmse,
                            "oracle_approx_rmse": oracle_rmse,
                            "individual_minus_original": individual_rmse - collective_rmse,
                            "oracle_minus_prior": oracle_rmse - prior_rmse,
                            "individual_objective": objective["individual_objective"],
                            "collective_objective": objective["collective_objective"],
                            "individual_data": objective["individual_data"],
                            "collective_data": objective["collective_data"],
                        }
                    )
                    _append_plugs(plugs, fold_id, int(seed), float(rho), "prior_only", test_starts, prior_windows, scored, depth, y_vector)
                    _append_plugs(plugs, fold_id, int(seed), float(rho), "clp_collective", test_starts, collective, scored, depth, y_vector)
                    _append_plugs(plugs, fold_id, int(seed), float(rho), "clp_individual", test_starts, individual, scored, depth, y_vector)
                    _append_plugs(plugs, fold_id, int(seed), float(rho), "oracle_approx", test_starts, oracle, scored, depth, y_vector)
                    message = "fold {0} seed {1} rho {2} collective {3:.4f} individual {4:.4f} oracle {5:.4f}".format(
                        fold_id,
                        int(seed),
                        float(rho),
                        collective_rmse,
                        individual_rmse,
                        oracle_rmse,
                    )
                    print(message, flush=True)
                    log_lines.append(message)
    finally:
        csgm.csgm_recover_with_prior = original

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    cell_frame = pd.DataFrame(rows)
    check_frame = pd.DataFrame(checks)
    detail = pd.DataFrame(plugs)
    oracle_frame = pd.DataFrame(oracle_rows)
    if int(cell_frame.shape[0]) != 18:
        raise RuntimeError("Expected 18 cells, found {0}.".format(cell_frame.shape[0]))
    cell_frame.to_csv(OUT / "results_by_cell.csv", index=False)
    check_frame.to_csv(OUT / "original_reproduction_check.csv", index=False)
    cell_frame.to_csv(OUT / "tables" / "results_by_cell.csv", index=False)
    check_frame.to_csv(OUT / "tables" / "original_reproduction_check.csv", index=False)
    detail.to_csv(OUT / "tables" / "detail_by_plug.csv", index=False)
    oracle_frame.to_csv(OUT / "tables" / "oracle_window_diagnostic.csv", index=False)
    summary = cell_frame.groupby("rho")[
        ["prior_rmse", "clp_original_rmse", "clp_individual_rmse", "oracle_approx_rmse", "individual_minus_original"]
    ].agg(["mean", "std"])
    summary.to_csv(OUT / "summary.csv")
    summary.to_csv(OUT / "tables" / "summary.csv")
    by_block = cell_frame.groupby(["fold_id", "rho"])[
        ["prior_rmse", "clp_original_rmse", "clp_individual_rmse", "oracle_approx_rmse", "individual_minus_original"]
    ].mean()
    by_block.to_csv(OUT / "tables" / "summary_by_block.csv")
    ran = cell_frame[cell_frame["recovery_ran"] > 0.5].copy()
    ran["individual_wins"] = ran["clp_individual_rmse"] < ran["clp_original_rmse"] - 1.0e-7
    ran["individual_loses"] = ran["clp_individual_rmse"] > ran["clp_original_rmse"] + 1.0e-7
    ran.groupby("rho")[["individual_wins", "individual_loses"]].sum().to_csv(OUT / "winner_counts.csv")
    ran.groupby(["fold_id", "rho"])[["individual_wins", "individual_loses"]].sum().to_csv(
        OUT / "tables" / "paired_by_block.csv"
    )
    config = {
        "base_sha": BASE_SHA,
        "n_plugs": 161,
        "seeds": [7, 23, 41],
        "rhos": [0.25, 0.50],
        "window_len": WINDOW_LEN,
        "latent_dim": LATENT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "ae_epochs": AE_EPOCHS,
        "csgm_iters": int(cfg.csgm_iters),
        "csgm_restarts": int(cfg.csgm_restarts),
        "csgm_opt_lr": float(cfg.csgm_opt_lr),
        "oracle_starts": ORACLE_STARTS,
        "oracle_iters": ORACLE_ITERS,
        "lambda_source": "lambda_extension/tables/selection_by_cell.csv extended_choice",
        "masks_sha256": _sha256(OUT_DIR / "tables" / "masks.csv"),
        "split_sha256": _sha256(OUT_DIR / "tables" / "split_by_plug.csv"),
        "oracle_uses_test_targets": True,
        "oracle_selects_clp": False,
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    readme = "\n".join(
        [
            "# Restart and oracle diagnostic",
            "",
            "Well 9-BRSA-1254-RJS plug holdout.",
            "See REPORT.md for the numbers from this run.",
            "The oracle uses true test windows and is not an operational baseline.",
            "Earlier holdout tables are left in the parent directory.",
            "",
        ]
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    (OUT / "logs" / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    max_difference = float(check_frame["difference"].abs().max())
    passed = bool(max_difference <= REPRODUCTION_TOLERANCE)
    _write_environment(passed)
    _write_report(cell_frame, check_frame, oracle_frame, passed)
    if passed:
        _write_figures(detail, cell_frame)
    print(summary)
    print("Maximum original reproduction difference", max_difference)
    if not passed:
        raise RuntimeError("REPRODUCTION FAILED (>{0} RMSE).".format(REPRODUCTION_TOLERANCE))
    print("REPRODUCTION PASSED within 1e-4 RMSE tolerance.")


if __name__ == "__main__":
    main()
