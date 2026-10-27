#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate latent start from the penalty anchor under shared restarts.

The nested-mask pilot kept one decoder, but measurement-only used another
restart seed. This extension does not overwrite that pilot. It compares
four recoveries that add the same perturbations:

- zero start, no penalty
- h(u) start, no penalty
- zero start, penalty around h(u)
- h(u) start, the same penalty

The original measurement-only call, with its seed offset, is kept as a
historical reference and is not part of the shared-noise contrast.

ASCII-only source. Exploratory. Seed 7 only. Not a manuscript result.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import csgm_m2_module as csgm
import run_scarcity_factorial_clp as base
import run_scarcity_nested_masks_ablation as nested

OUT_DIR = os.path.join(
    REPO_ROOT, "outputs", "exploratory", "scarcity_init_penalty_control"
)
PREVIOUS_RESULTS = os.path.join(
    nested.OUT_DIR, "tables", "results_by_seed.csv"
)
PREVIOUS_MASKS = os.path.join(nested.OUT_DIR, "tables", "masks.csv")
PILOT_SEED = 7
PILOT_SIZES: Tuple[int, ...] = (24, 88)
METHOD_ZERO = "zero_init_no_penalty"
METHOD_COND = "conditioned_init_no_penalty"
METHOD_ZERO_PEN = "zero_init_anchored_penalty"
METHOD_COND_PEN = "conditioned_init_anchored_penalty"
METHOD_HIST = "measurement_only_historical"
SHARED_METHODS: Tuple[str, ...] = (
    METHOD_ZERO,
    METHOD_COND,
    METHOD_ZERO_PEN,
    METHOD_COND_PEN,
)


def _restart_perturbations(
    shape: Tuple[int, ...],
    n_restarts: int,
    seed: int,
) -> List[Optional[torch.Tensor]]:
    """Match the original generator: restart 0 is exact, later restarts add noise.

    The tensors do not depend on the start or on the penalty anchor.
    """
    if int(n_restarts) < 1:
        raise AssertionError("Recovery needs at least one restart.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    perturbations: List[Optional[torch.Tensor]] = []
    for restart in range(int(n_restarts)):
        if restart == 0:
            perturbations.append(None)
        else:
            draw = torch.randn(shape, generator=generator, dtype=torch.float32)
            perturbations.append(0.05 * draw)
    return perturbations


def _tensor_list_hash(perturbations: Sequence[Optional[torch.Tensor]]) -> str:
    """Identify one shared perturbation list."""
    digest = hashlib.sha256()
    for item in perturbations:
        if item is None:
            digest.update(b"exact-start")
            continue
        array = np.ascontiguousarray(item.detach().cpu().numpy())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def recover_with_separate_anchor(
    ae: csgm.TinyAE,
    measurement: np.ndarray,
    observations: np.ndarray,
    z_init: np.ndarray,
    z_anchor: np.ndarray,
    y_mean: np.ndarray,
    y_scale: np.ndarray,
    lam: float,
    n_iters: int,
    opt_lr: float,
    perturbations: Sequence[Optional[torch.Tensor]],
) -> np.ndarray:
    """Same latent update as csgm_recover_with_prior, with two latent roles.

    z_init is the restart center. z_anchor is the penalty center.
    The original function uses one vector for both roles.
    """
    ae.eval()
    n_rows = int(observations.shape[0])
    if int(z_init.shape[0]) != n_rows or int(z_anchor.shape[0]) != n_rows:
        raise AssertionError("Latent start and anchor must have one row per window.")
    if int(z_init.shape[1]) != int(z_anchor.shape[1]):
        raise AssertionError("Latent start and anchor have different widths.")
    operator_t = torch.tensor(measurement, dtype=torch.float32)
    mean_t = torch.tensor(y_mean, dtype=torch.float32)
    scale_t = torch.tensor(y_scale, dtype=torch.float32)
    observations_t = torch.tensor(observations, dtype=torch.float32)
    init_t = torch.tensor(z_init, dtype=torch.float32)
    anchor_t = torch.tensor(z_anchor, dtype=torch.float32)
    effective_operator = operator_t * scale_t.unsqueeze(0)
    effective_observations = observations_t - (mean_t.unsqueeze(0) @ operator_t.T)
    best_loss: Optional[float] = None
    best_prediction: Optional[np.ndarray] = None
    for restart, noise in enumerate(perturbations):
        if noise is None:
            if restart != 0:
                raise AssertionError("Only restart 0 may omit the perturbation.")
            latent = init_t.clone().detach()
        else:
            latent = init_t.clone().detach() + noise
        latent.requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=float(opt_lr))
        for _ in range(int(n_iters)):
            optimizer.zero_grad()
            decoded = ae.decode(latent)
            residual = (decoded @ effective_operator.T) - effective_observations
            data_term = torch.sum(residual ** 2) / float(n_rows)
            prior_term = float(lam) * torch.sum((latent - anchor_t) ** 2) / float(n_rows)
            loss = data_term + prior_term
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            decoded = ae.decode(latent)
            residual = (decoded @ effective_operator.T) - effective_observations
            data_term = torch.sum(residual ** 2) / float(n_rows)
            prior_term = float(lam) * torch.sum((latent - anchor_t) ** 2) / float(n_rows)
            final_loss = float((data_term + prior_term).item())
            if best_loss is None or final_loss < best_loss:
                best_loss = final_loss
                restored = decoded.detach().cpu().numpy() * y_scale[None, :] + y_mean[None, :]
                best_prediction = restored
    if best_prediction is None:
        raise RuntimeError("Split-anchor recovery failed.")
    return best_prediction


def _synthetic_equivalence() -> None:
    """The split call must reproduce the original function when the roles coincide."""
    torch.manual_seed(0)
    ae = csgm.TinyAE(n_output=8, latent_dim=3, hidden_dim=5)
    generator = np.random.default_rng(0)
    measurement = np.zeros((2, 8), dtype=np.float64)
    measurement[0, 1] = 1.0
    measurement[1, 4] = 1.0
    observations = generator.normal(size=(4, 2))
    y_mean = np.linspace(0.1, 0.8, 8)
    y_scale = np.linspace(0.5, 1.2, 8)
    z_anchor = generator.normal(size=(4, 3))
    z_zero = np.zeros_like(z_anchor)
    perturbations = _restart_perturbations((4, 3), 3, seed=7)
    common = {
        "ae": ae,
        "M": measurement,
        "B": observations,
        "y_mean": y_mean,
        "y_scale": y_scale,
        "n_iters": 2,
        "opt_lr": 0.05,
        "n_restarts": 3,
        "device": "cpu",
        "seed": 7,
    }
    original_penalty = csgm.csgm_recover_with_prior(
        z0_np=z_anchor, lam=1e-3, **common
    )
    split_penalty = recover_with_separate_anchor(
        ae, measurement, observations, z_anchor, z_anchor, y_mean, y_scale,
        1e-3, 2, 0.05, perturbations,
    )
    if float(np.max(np.abs(original_penalty - split_penalty))) > 1e-6:
        raise AssertionError("Split recovery drifted from the original penalty call.")
    original_zero = csgm.csgm_recover_with_prior(
        z0_np=z_zero, lam=0.0, **common
    )
    split_zero_anchor_ignored = recover_with_separate_anchor(
        ae, measurement, observations, z_zero, z_anchor, y_mean, y_scale,
        0.0, 2, 0.05, perturbations,
    )
    split_zero_anchor_zero = recover_with_separate_anchor(
        ae, measurement, observations, z_zero, z_zero, y_mean, y_scale,
        0.0, 2, 0.05, perturbations,
    )
    if float(np.max(np.abs(original_zero - split_zero_anchor_ignored))) > 1e-6:
        raise AssertionError("Lambda 0 still depended on the penalty anchor.")
    if float(np.max(np.abs(split_zero_anchor_ignored - split_zero_anchor_zero))) > 1e-6:
        raise AssertionError("Two lambda-0 anchors did not match.")
    redraw = _restart_perturbations((4, 3), 3, seed=7)
    if _tensor_list_hash(perturbations) != _tensor_list_hash(redraw):
        raise AssertionError("Shared perturbations were not reproducible.")
    print("Synthetic split-anchor checks passed.")


def _load_previous_cells() -> Dict[Tuple[int, int, str], Dict[str, str]]:
    """Read the versioned pilot rows. Test labels are not used to choose lambda."""
    if not os.path.isfile(PREVIOUS_RESULTS):
        raise FileNotFoundError("Missing nested-mask pilot results: {}".format(PREVIOUS_RESULTS))
    found: Dict[Tuple[int, int, str], Dict[str, str]] = {}
    with open(PREVIOUS_RESULTS, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["seed"]) != PILOT_SEED:
                raise AssertionError("The preserved pilot is not seed 7.")
            key = (int(row["n_train"]), int(row["m_obs"]), str(row["method"]))
            found[key] = row
    return found


def _selected_lambda(previous: Mapping[Tuple[int, int, str], Mapping[str, str]], n_train: int, m_obs: int) -> float:
    """Positive lambda already chosen on F06-1 in the preserved pilot."""
    row = previous[(int(n_train), int(m_obs), "clp_csgm_ridge")]
    lam = float(row["lambda"])
    if not lam > 0.0:
        raise AssertionError("Preserved CLP lambda is not positive.")
    return lam


def _previous_rmse(
    previous: Mapping[Tuple[int, int, str], Mapping[str, str]],
    n_train: int,
    m_obs: int,
    method: str,
) -> float:
    """Pooled test RMSE stored by the preserved pilot."""
    return float(previous[(int(n_train), int(m_obs), method)]["rmse_full"])


def _load_previous_order() -> np.ndarray:
    """Permutation versioned with the nested-mask pilot."""
    order = np.full(base.WINDOW_LEN, -1, dtype=np.int64)
    with open(PREVIOUS_MASKS, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != base.WINDOW_LEN:
        raise AssertionError("Preserved mask file does not have 64 positions.")
    for row in rows:
        if int(row["seed"]) != PILOT_SEED:
            raise AssertionError("Preserved mask is not seed 7.")
        order[int(row["position_rank"])] = int(row["window_index"])
    if int(np.min(order)) < 0:
        raise AssertionError("Preserved permutation has a missing rank.")
    nested.verify_nested_masks(order)
    return order


def _save_table(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write one CSV with a fixed column order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _plot(rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Full-window RMSE for the shared-noise quartet and the historical call."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    methods = SHARED_METHODS + (METHOD_HIST,)
    fig, axes = plt.subplots(1, len(PILOT_SIZES), figsize=(9.2, 4.2), sharey=True)
    for axis, size in zip(axes, PILOT_SIZES):
        for method in methods:
            subset = [
                row
                for row in rows
                if int(row["n_train"]) == int(size) and str(row["method"]) == method
            ]
            subset = sorted(subset, key=lambda row: int(row["m_obs"]))
            axis.plot(
                [int(row["m_obs"]) for row in subset],
                [float(row["rmse_full"]) for row in subset],
                marker="o",
                linestyle="--" if method == METHOD_HIST else "-",
                label=method,
            )
        axis.set_title("{} training windows".format(int(size)))
        axis.set_xlabel("Observed positions")
        axis.set_xticks(list(nested.M_OBS))
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Full-window RMSE")
    axes[-1].legend(fontsize=6, loc="best")
    fig.suptitle("Shared perturbations, except the dashed historical seed offset")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _write_readme(path: str) -> None:
    """State the contrast and that the three-seed grid was not run."""
    lines = [
        "# Shared-restart init and penalty control",
        "",
        "Exploratory extension. These numbers are not manuscript results.",
        "",
        "Preserved pilot: commit feef3a7, directory scarcity_nested_masks_ablation.",
        "That directory was not overwritten. Its measurement-only call remains the historical reference.",
        "",
        "This run: seed 7, budgets 24 and 88, observation counts 4, 6, and 13.",
        "Not run: seeds 23 and 41, and the 56-window budget.",
        "",
        "The four shared-noise recoveries use one perturbation list drawn with seed 7.",
        "Restart 0 is the exact start. Later restarts add the same N(0, 0.05^2) tensor.",
        "Zero start with no penalty uses that shared list, not seed+9200.",
        "The historical measurement-only call still uses seed+9200 and the original function.",
        "",
        "The penalty weight is the positive lambda already chosen on F06-1 in the preserved pilot.",
        "F03-4 does not choose lambda, the start, or the anchor.",
        "The same lambda and the same h(u) are used for both penalized starts.",
        "",
        "A remaining gap is non-convex optimization. Shared noise removes the restart-seed confound;",
        "it does not turn the four losses into an isolated causal estimate.",
        "The 97 test windows remain one pooled score.",
        "",
    ]
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    """Run the seed-7 mechanistic control and write a separate output directory."""
    protected = {os.path.abspath(nested.OUT_DIR), os.path.abspath(base.OUT_DIR)}
    if os.path.abspath(OUT_DIR) in protected:
        raise RuntimeError("Refusing to write into a previous study directory.")
    _synthetic_equivalence()
    previous = _load_previous_cells()
    recorded_order = _load_previous_order()
    cfg = base._paper_config()
    train, val, test = base._load_role_windows()
    base.verify_training_and_test(
        train.well_of_window.tolist(),
        val.well_of_window.tolist(),
        test.well_of_window.tolist(),
    )
    indices = nested._load_recorded_indices(train)
    y_pool = np.asarray(train.y, dtype=np.float64)
    y_val = np.asarray(val.y, dtype=np.float64)
    y_test = np.asarray(test.y, dtype=np.float64)
    x_test = np.asarray(test.x, dtype=np.float64)
    order, noise_pool, noise_val, noise_test = nested._draw_nested_problem(
        y_pool, y_val, y_test, PILOT_SEED
    )
    np.testing.assert_array_equal(order, recorded_order)
    nested.verify_shared_noise_prefix(y_test, noise_test, order)
    fixed = nested._fixed_coordinates(order)
    result_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    reproduction: List[Dict[str, float]] = []

    for size in PILOT_SIZES:
        print("seed", PILOT_SEED, "n_train", int(size), flush=True)
        idx = indices[int(size)]
        x_train = np.asarray(train.x[idx], dtype=np.float64)
        y_train = np.asarray(train.y[idx], dtype=np.float64)
        ae, prior, _scaler, y_mean, y_scale = base._fit_clp(
            x_train, y_train, PILOT_SEED, cfg
        )
        z_conditioned = np.asarray(prior.predict(x_test), dtype=np.float64)
        z_zero = np.zeros_like(z_conditioned)
        perturbations = _restart_perturbations(
            tuple(int(value) for value in z_conditioned.shape),
            int(cfg.csgm_restarts),
            PILOT_SEED,
        )
        noise_hash = _tensor_list_hash(perturbations)
        for m_obs in nested.M_OBS:
            print("  m_obs", int(m_obs), flush=True)
            positions = order[: int(m_obs)]
            operator = nested._operator_from_positions(positions)
            b_test = nested._slice_measurements(y_test, noise_test, positions)
            lam = _selected_lambda(previous, int(size), int(m_obs))
            calls = (
                (METHOD_ZERO, z_zero, z_zero, 0.0),
                (METHOD_COND, z_conditioned, z_zero, 0.0),
                (METHOD_ZERO_PEN, z_zero, z_conditioned, lam),
                (METHOD_COND_PEN, z_conditioned, z_conditioned, lam),
            )
            scored: Dict[str, Dict[str, float]] = {}
            for method, z_init, z_anchor, penalty in calls:
                if method in (METHOD_ZERO, METHOD_COND) and float(penalty) != 0.0:
                    raise AssertionError("An unpenalized configuration received a penalty.")
                if method in (METHOD_ZERO_PEN, METHOD_COND_PEN) and not float(penalty) > 0.0:
                    raise AssertionError("A penalized configuration received lambda 0.")
                prediction = recover_with_separate_anchor(
                    ae,
                    operator,
                    b_test,
                    z_init,
                    z_anchor,
                    y_mean,
                    y_scale,
                    float(penalty),
                    int(cfg.csgm_iters),
                    float(cfg.csgm_opt_lr),
                    perturbations,
                )
                scores = nested._score_trio(y_test, prediction, positions, fixed)
                scored[method] = scores
                print(
                    "   ",
                    method,
                    "full",
                    "{:.6f}".format(scores["rmse_full"]),
                    flush=True,
                )
                result_rows.append(
                    {
                        "seed": PILOT_SEED,
                        "n_train": int(size),
                        "m_obs": int(m_obs),
                        "method": method,
                        "rmse_full": scores["rmse_full"],
                        "rmse_unobserved": scores["rmse_unobserved"],
                        "rmse_fixed": scores["rmse_fixed"],
                        "lambda": float(penalty),
                        "restart_seed": PILOT_SEED,
                        "shared_noise": 1,
                        "noise_hash": noise_hash,
                    }
                )
            historical = csgm.csgm_recover_with_prior(
                ae=ae,
                M=operator,
                B=b_test,
                z0_np=z_zero,
                y_mean=y_mean,
                y_scale=y_scale,
                lam=0.0,
                n_iters=int(cfg.csgm_iters),
                opt_lr=float(cfg.csgm_opt_lr),
                n_restarts=int(cfg.csgm_restarts),
                device="cpu",
                seed=PILOT_SEED + nested.MEASUREMENT_ONLY_SEED_OFFSET,
            )
            historical_scores = nested._score_trio(y_test, historical, positions, fixed)
            stored_historical = _previous_rmse(
                previous, int(size), int(m_obs), "measurement_only_csgm"
            )
            stored_clp = _previous_rmse(previous, int(size), int(m_obs), "clp_csgm_ridge")
            stored_init = _previous_rmse(previous, int(size), int(m_obs), "clp_init_only")
            historical_gap = abs(historical_scores["rmse_full"] - stored_historical)
            clp_gap = abs(scored[METHOD_COND_PEN]["rmse_full"] - stored_clp)
            init_gap = abs(scored[METHOD_COND]["rmse_full"] - stored_init)
            print(
                "    reproduction gaps historical",
                "{:.3e}".format(historical_gap),
                "clp",
                "{:.3e}".format(clp_gap),
                "init",
                "{:.3e}".format(init_gap),
                flush=True,
            )
            if max(historical_gap, clp_gap, init_gap) > 1e-5:
                raise AssertionError(
                    "Seed-7 recovery did not reproduce the preserved pilot RMSE."
                )
            reproduction.append(
                {
                    "n_train": float(size),
                    "m_obs": float(m_obs),
                    "historical_abs_gap": historical_gap,
                    "conditioned_penalty_abs_gap": clp_gap,
                    "conditioned_init_abs_gap": init_gap,
                }
            )
            result_rows.append(
                {
                    "seed": PILOT_SEED,
                    "n_train": int(size),
                    "m_obs": int(m_obs),
                    "method": METHOD_HIST,
                    "rmse_full": historical_scores["rmse_full"],
                    "rmse_unobserved": historical_scores["rmse_unobserved"],
                    "rmse_fixed": historical_scores["rmse_fixed"],
                    "lambda": 0.0,
                    "restart_seed": PILOT_SEED + nested.MEASUREMENT_ONLY_SEED_OFFSET,
                    "shared_noise": 0,
                    "noise_hash": "",
                }
            )
            summary_rows.append(
                {
                    "seed": PILOT_SEED,
                    "n_train": int(size),
                    "m_obs": int(m_obs),
                    "lambda": lam,
                    "rmse_zero_init_no_penalty": scored[METHOD_ZERO]["rmse_full"],
                    "rmse_conditioned_init_no_penalty": scored[METHOD_COND]["rmse_full"],
                    "rmse_zero_init_anchored_penalty": scored[METHOD_ZERO_PEN]["rmse_full"],
                    "rmse_conditioned_init_anchored_penalty": scored[METHOD_COND_PEN]["rmse_full"],
                    "rmse_measurement_only_historical": historical_scores["rmse_full"],
                    "penalty_effect_at_zero_init": (
                        scored[METHOD_ZERO_PEN]["rmse_full"] - scored[METHOD_ZERO]["rmse_full"]
                    ),
                    "penalty_effect_at_conditioned_init": (
                        scored[METHOD_COND_PEN]["rmse_full"] - scored[METHOD_COND]["rmse_full"]
                    ),
                    "init_effect_without_penalty": (
                        scored[METHOD_COND]["rmse_full"] - scored[METHOD_ZERO]["rmse_full"]
                    ),
                    "init_effect_with_penalty": (
                        scored[METHOD_COND_PEN]["rmse_full"] - scored[METHOD_ZERO_PEN]["rmse_full"]
                    ),
                }
            )

    table_dir = os.path.join(OUT_DIR, "tables")
    figure_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(OUT_DIR, exist_ok=True)
    _save_table(
        os.path.join(table_dir, "results_by_seed.csv"),
        result_rows,
        (
            "seed",
            "n_train",
            "m_obs",
            "method",
            "rmse_full",
            "rmse_unobserved",
            "rmse_fixed",
            "lambda",
            "restart_seed",
            "shared_noise",
            "noise_hash",
        ),
    )
    _save_table(
        os.path.join(table_dir, "summary.csv"),
        summary_rows,
        (
            "seed",
            "n_train",
            "m_obs",
            "lambda",
            "rmse_zero_init_no_penalty",
            "rmse_conditioned_init_no_penalty",
            "rmse_zero_init_anchored_penalty",
            "rmse_conditioned_init_anchored_penalty",
            "rmse_measurement_only_historical",
            "penalty_effect_at_zero_init",
            "penalty_effect_at_conditioned_init",
            "init_effect_without_penalty",
            "init_effect_with_penalty",
        ),
    )
    _plot(result_rows, os.path.join(figure_dir, "shared_restart_quartet.png"))
    config = {
        "mode": "pilot",
        "seed": PILOT_SEED,
        "sizes_trained": list(PILOT_SIZES),
        "m_obs": list(nested.M_OBS),
        "preserved_commit": "feef3a761c5ac2dfd6d298d56e497a83326e176b",
        "preserved_results": PREVIOUS_RESULTS,
        "lambda_source": "positive lambda selected on F06-1 in the preserved pilot",
        "shared_restart_seed": PILOT_SEED,
        "historical_restart_seed": PILOT_SEED + nested.MEASUREMENT_ONLY_SEED_OFFSET,
        "optimizer": {
            "iters": int(cfg.csgm_iters),
            "lr": float(cfg.csgm_opt_lr),
            "restarts": int(cfg.csgm_restarts),
        },
        "contrast_sign": (
            "effect columns are RMSE_with_factor minus RMSE_without_factor. "
            "Negative means the named factor has lower RMSE."
        ),
        "nonconvex_caveat": (
            "Shared perturbations remove the restart-seed difference. "
            "They do not remove non-convex optimization."
        ),
        "full_grid_executed": False,
        "manuscript": "not updated",
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(os.path.join(OUT_DIR, "checks.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "passed": [
                    "split recovery matches the original function when start and anchor coincide",
                    "lambda 0 does not depend on the anchor",
                    "shared perturbations are reproducible and independent of the start",
                    "seed-7 permutation matches the preserved mask file",
                    "historical, full CLP, and init-only RMSE match the preserved pilot within 1e-5",
                    "the four new contrasts use one perturbation list per training budget",
                ],
                "reproduction_gaps": reproduction,
                "n_result_rows": len(result_rows),
                "full_grid_executed": False,
            },
            handle,
            indent=2,
        )
        handle.write("\n")
    _write_readme(os.path.join(OUT_DIR, "README.md"))
    print("Wrote", OUT_DIR)


if __name__ == "__main__":
    main()
