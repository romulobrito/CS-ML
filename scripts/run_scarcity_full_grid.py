#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Expand the nested-mask and shared-restart controls to three seeds.

The pilot directories are not overwritten. One run writes:

- outputs/exploratory/scarcity_nested_masks_ablation_full
- outputs/exploratory/scarcity_init_penalty_control_full

Seeds are 7, 23, and 41. Budgets are 24, 56, and 88. Masks stay nested.
The four latent recoveries share one decoder, one mask, one measurement
draw, and one perturbation list. Positive lambda is chosen on F06-1 only
and is then used for both penalized starts.

The best restart is the one with the lowest objective on the whole batch.
That rule is unchanged.

ASCII-only source. Exploratory. Not a manuscript result.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

import csgm_m2_module as csgm
import direct_ub_baselines as dub
import run_scarcity_factorial_clp as base
import run_scarcity_init_penalty_control as control
import run_scarcity_nested_masks_ablation as nested

NESTED_FULL = os.path.join(
    REPO_ROOT, "outputs", "exploratory", "scarcity_nested_masks_ablation_full"
)
CONTROL_FULL = os.path.join(
    REPO_ROOT, "outputs", "exploratory", "scarcity_init_penalty_control_full"
)
SEEDS: Tuple[int, ...] = (7, 23, 41)
SIZES: Tuple[int, ...] = (24, 56, 88)
METRICS: Tuple[str, ...] = ("rmse_full", "rmse_unobserved", "rmse_fixed")
METHOD_ZERO = control.METHOD_ZERO
METHOD_COND = control.METHOD_COND
METHOD_ZERO_PEN = control.METHOD_ZERO_PEN
METHOD_COND_PEN = control.METHOD_COND_PEN
METHOD_PRIOR = nested.METHOD_PRIOR
METHOD_AE = nested.METHOD_AE
METHOD_INTERP = nested.METHOD_INTERP
ALL_METHODS: Tuple[str, ...] = (
    METHOD_COND_PEN,
    METHOD_COND,
    METHOD_ZERO_PEN,
    METHOD_ZERO,
    METHOD_PRIOR,
    METHOD_AE,
    METHOD_INTERP,
)
REFERENCE_COMMIT = "1d490d2accccea0a08e65cb4302a72c45f887788"
RESTART_RULE = (
    "The retained restart is the one with the lowest sum of the data term "
    "and the penalty term over the whole batch, divided by the number of "
    "windows. Restarts are not chosen separately for each window. "
    "This is the rule in csgm_recover_with_prior and it was not changed."
)


def _read_rows(path: str) -> List[Dict[str, str]]:
    """Load one CSV written by a preserved pilot."""
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _saved_rmse(rows: Sequence[Mapping[str, str]], n_train: int, m_obs: int, method: str) -> float:
    """One pooled RMSE from a preserved seed-7 table."""
    matches = [
        row
        for row in rows
        if int(row["n_train"]) == int(n_train)
        and int(row["m_obs"]) == int(m_obs)
        and str(row["method"]) == method
    ]
    if len(matches) != 1:
        raise AssertionError("Preserved table is missing {} {} {}.".format(n_train, m_obs, method))
    return float(matches[0]["rmse_full"])


def _saved_lambda(rows: Sequence[Mapping[str, str]], n_train: int, m_obs: int) -> float:
    """Positive lambda stored for full CLP in the nested-mask pilot."""
    matches = [
        row
        for row in rows
        if int(row["n_train"]) == int(n_train)
        and int(row["m_obs"]) == int(m_obs)
        and str(row["method"]) == nested.METHOD_CLP
    ]
    if len(matches) != 1:
        raise AssertionError("Preserved lambda row is missing.")
    lam = float(matches[0]["lambda"])
    if not lam > 0.0:
        raise AssertionError("Preserved lambda is not positive.")
    return lam


def _effects(scores: Mapping[str, Mapping[str, float]], metric: str) -> Dict[str, float]:
    """Paired contrasts. Negative means the named factor has lower RMSE."""
    zero = float(scores[METHOD_ZERO][metric])
    cond = float(scores[METHOD_COND][metric])
    zero_pen = float(scores[METHOD_ZERO_PEN][metric])
    cond_pen = float(scores[METHOD_COND_PEN][metric])
    penalty_at_zero = zero_pen - zero
    penalty_at_cond = cond_pen - cond
    init_without = cond - zero
    init_with = cond_pen - zero_pen
    interaction = init_with - init_without
    other = penalty_at_cond - penalty_at_zero
    if abs(interaction - other) > 1e-12:
        raise AssertionError("Interaction identity failed.")
    return {
        "penalty_effect_at_zero_init": penalty_at_zero,
        "penalty_effect_at_conditioned_init": penalty_at_cond,
        "init_effect_without_penalty": init_without,
        "init_effect_with_penalty": init_with,
        "interaction": interaction,
    }


def _save_table(path: str, rows: Sequence[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    """Write one CSV with a fixed column order."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def _plot_rmse(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    path: str,
    methods: Sequence[str] = ALL_METHODS,
) -> None:
    """One panel per seed and budget. Lines are methods, not windows."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(len(SEEDS), len(SIZES), figsize=(12.5, 9.0), sharex=True, sharey=True)
    for row_id, seed in enumerate(SEEDS):
        for col_id, size in enumerate(SIZES):
            axis = axes[row_id, col_id]
            for method in methods:
                subset = [
                    item
                    for item in rows
                    if int(item["seed"]) == int(seed)
                    and int(item["n_train"]) == int(size)
                    and str(item["method"]) == method
                ]
                subset = sorted(subset, key=lambda item: int(item["m_obs"]))
                axis.plot(
                    [int(item["m_obs"]) for item in subset],
                    [float(item[metric]) for item in subset],
                    marker="o",
                    label=method,
                )
            axis.set_xticks(list(nested.M_OBS))
            axis.grid(True, alpha=0.3)
            if row_id == 0:
                axis.set_title("{} windows".format(int(size)))
            if col_id == 0:
                axis.set_ylabel("seed {}".format(int(seed)))
    axes[0, -1].legend(fontsize=6, loc="best")
    fig.suptitle(metric)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_effects(summary_rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Seed lines for the two main effects and their interaction."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    effect_names = (
        "penalty_effect_at_zero_init",
        "init_effect_without_penalty",
        "interaction",
    )
    fig, axes = plt.subplots(len(effect_names), len(SIZES), figsize=(12.0, 8.0), sharex=True, sharey=True)
    for row_id, effect in enumerate(effect_names):
        for col_id, size in enumerate(SIZES):
            axis = axes[row_id, col_id]
            for seed in SEEDS:
                subset = [
                    item
                    for item in summary_rows
                    if int(item["seed"]) == int(seed) and int(item["n_train"]) == int(size)
                ]
                subset = sorted(subset, key=lambda item: int(item["m_obs"]))
                axis.plot(
                    [int(item["m_obs"]) for item in subset],
                    [float(item[effect]) for item in subset],
                    marker="o",
                    label="seed {}".format(int(seed)),
                )
            axis.axhline(0.0, color="black", linewidth=0.8)
            axis.set_xticks(list(nested.M_OBS))
            axis.grid(True, alpha=0.3)
            if row_id == 0:
                axis.set_title("{} windows".format(int(size)))
            if col_id == 0:
                axis.set_ylabel(effect)
    axes[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle("Negative means the named factor has lower full-window RMSE")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_lambda(lambda_rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Validation curves. Each panel is one seed and one budget."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(len(SEEDS), len(SIZES), figsize=(12.0, 8.0), sharey=True)
    for row_id, seed in enumerate(SEEDS):
        for col_id, size in enumerate(SIZES):
            axis = axes[row_id, col_id]
            for m_obs in nested.M_OBS:
                subset = [
                    item
                    for item in lambda_rows
                    if int(item["seed"]) == int(seed)
                    and int(item["n_train"]) == int(size)
                    and int(item["m_obs"]) == int(m_obs)
                ]
                subset = sorted(subset, key=lambda item: float(item["lambda"]))
                axis.plot(
                    [float(item["lambda"]) for item in subset],
                    [float(item["val_rmse_unobserved"]) for item in subset],
                    marker="o",
                    label="m={}".format(int(m_obs)),
                )
            axis.set_xscale("symlog", linthresh=1e-7)
            axis.grid(True, alpha=0.3)
            if row_id == 0:
                axis.set_title("{} windows".format(int(size)))
            if col_id == 0:
                axis.set_ylabel("seed {}".format(int(seed)))
    axes[0, -1].legend(fontsize=7, loc="best")
    fig.suptitle("F06-1 unobserved RMSE. Test well is not on this figure.")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_decoder(diag_rows: Sequence[Mapping[str, object]], path: str) -> None:
    """Decoder reconstruction and prior-only RMSE by seed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharex=True)
    for axis, name in zip(axes, ("decoder_test", "prior_full")):
        for seed in SEEDS:
            subset = [
                item
                for item in diag_rows
                if int(item["seed"]) == int(seed) and str(item["name"]) == name
            ]
            subset = sorted(subset, key=lambda item: int(item["n_train"]))
            axis.plot(
                [int(item["n_train"]) for item in subset],
                [float(item["rmse"]) for item in subset],
                marker="o",
                label="seed {}".format(int(seed)),
            )
        axis.set_title(name)
        axis.set_xlabel("Training windows")
        axis.set_ylabel("RMSE")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _mean_rows(summary_rows: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    """Mean of the three seed-level scores. Windows stay inside each seed score."""
    numeric = (
        "rmse_conditioned_init_anchored_penalty",
        "rmse_conditioned_init_no_penalty",
        "rmse_zero_init_anchored_penalty",
        "rmse_zero_init_no_penalty",
        "rmse_prior",
        "rmse_ae",
        "rmse_interpolation",
        "clp_minus_ae",
        "clp_minus_interpolation",
        "clp_minus_zero_init",
        "penalty_effect_at_zero_init",
        "penalty_effect_at_conditioned_init",
        "init_effect_without_penalty",
        "init_effect_with_penalty",
        "interaction",
        "penalty_effect_at_zero_init_unobserved",
        "init_effect_without_penalty_unobserved",
        "interaction_unobserved",
        "penalty_effect_at_zero_init_fixed",
        "init_effect_without_penalty_fixed",
        "interaction_fixed",
        "clp_minus_ae_fixed",
        "clp_minus_interpolation_fixed",
    )
    out: List[Dict[str, object]] = []
    for size in SIZES:
        for m_obs in nested.M_OBS:
            subset = [
                row
                for row in summary_rows
                if int(row["n_train"]) == int(size) and int(row["m_obs"]) == int(m_obs)
            ]
            if len(subset) != len(SEEDS):
                raise AssertionError("A cell does not have three seeds.")
            record: Dict[str, object] = {"n_train": int(size), "m_obs": int(m_obs), "n_seeds": len(SEEDS)}
            for key in numeric:
                values = [float(row[key]) for row in subset]
                record["mean_{}".format(key)] = float(np.mean(values))
                if key.startswith("penalty_") or key.startswith("init_") or key.startswith("interaction") or key.startswith("clp_minus_"):
                    record["n_negative_{}".format(key)] = int(sum(value < 0.0 for value in values))
                for row, seed in zip(sorted(subset, key=lambda item: int(item["seed"])), SEEDS):
                    record["seed{}_{}".format(int(seed), key)] = float(row[key])
            out.append(record)
    return out


def _limitations_text() -> str:
    """Method limits that stay in force after the three-seed expansion."""
    return "\n".join(
        [
            "# Limitations",
            "",
            "Exploratory full grid. These numbers are not manuscript results.",
            "",
            "F06-1 has eight overlapping windows from one well. All 28 pairs overlap.",
            "Lambda selection on that well is not strong statistical evidence.",
            "F03-4 does not choose lambda, a mask, a budget, or a method.",
            "",
            "Each reported RMSE pools the test windows and the scored coordinates.",
            "The 97 windows are not 97 independent trials. The paired unit is the seed.",
            "Means across seeds are means of three seed-level scores.",
            "",
            RESTART_RULE,
            "A later sensitivity check could score restarts inside each window.",
            "That check is not part of this run.",
            "",
            "Shared perturbations remove the restart-seed confound among the four latent starts.",
            "They do not remove non-convex optimization.",
            "The interaction is the difference of those paired effects, not a causal estimate.",
            "",
            "Masks are nested prefixes of one permutation per seed.",
            "Observation count is separated from a new noise draw.",
            "The three seeds still use three different permutations.",
            "",
            "The historical measurement-only call with seed+9200 stays in the seed-7 pilot.",
            "This grid uses the shared-noise zero start instead of that offset.",
            "",
            "The target is the well-log curve. This run does not use independent lab assays.",
            "G(E(y)) is diagnostic and does not select lambda, a budget, or a method.",
            "",
        ]
    )


def _write_text(path: str, text: str) -> None:
    """Write one ASCII report."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _guard_outputs() -> None:
    """Refuse to overwrite a preserved pilot directory."""
    protected = {
        os.path.abspath(nested.OUT_DIR),
        os.path.abspath(base.OUT_DIR),
        os.path.abspath(control.OUT_DIR),
    }
    for path in (NESTED_FULL, CONTROL_FULL):
        if os.path.abspath(path) in protected:
            raise RuntimeError("Refusing to write into a preserved pilot directory.")


def main() -> None:
    """Run the three-seed grid once and write both full directories."""
    _guard_outputs()
    nested._run_synthetic_checks()
    control._synthetic_equivalence()
    cfg = base._paper_config()
    cfg.csgm_lambda_grid = list(nested.LAMBDA_GRID)
    train, val, test = base._load_role_windows()
    base.verify_training_and_test(
        train.well_of_window.tolist(),
        val.well_of_window.tolist(),
        test.well_of_window.tolist(),
    )
    if int(val.y.shape[0]) != 8 or int(test.y.shape[0]) != 97:
        raise AssertionError("Validation or test window count changed.")
    indices = nested._load_recorded_indices(train)
    recorded_order = control._load_previous_order()
    nested_rows = _read_rows(os.path.join(nested.OUT_DIR, "tables", "results_by_seed.csv"))
    control_rows = _read_rows(os.path.join(control.OUT_DIR, "tables", "results_by_seed.csv"))
    y_pool = np.asarray(train.y, dtype=np.float64)
    y_val = np.asarray(val.y, dtype=np.float64)
    y_test = np.asarray(test.y, dtype=np.float64)
    x_test = np.asarray(test.x, dtype=np.float64)
    x_val = np.asarray(val.x, dtype=np.float64)
    val_dependence = base._window_dependence(val)

    result_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    lambda_rows: List[Dict[str, object]] = []
    diag_rows: List[Dict[str, object]] = []
    mask_rows: List[Dict[str, object]] = []

    for seed in SEEDS:
        order, noise_pool, _noise_val, noise_test = nested._draw_nested_problem(
            y_pool, y_val, y_test, int(seed)
        )
        redraw, _, _, _ = nested._draw_nested_problem(y_pool, y_val, y_test, int(seed))
        np.testing.assert_array_equal(order, redraw)
        nested.verify_nested_masks(order)
        nested.verify_shared_noise_prefix(y_test, noise_test, order)
        if int(seed) == control.PILOT_SEED:
            np.testing.assert_array_equal(order, recorded_order)
        fixed = nested._fixed_coordinates(order)
        for rank, position in enumerate(order.tolist()):
            mask_rows.append(
                {
                    "seed": int(seed),
                    "position_rank": int(rank),
                    "window_index": int(position),
                    "in_omega4": int(rank < 4),
                    "in_omega6": int(rank < 6),
                    "in_omega13": int(rank < 13),
                }
            )
        first_bundle: Dict[int, Dict[str, np.ndarray]] = {}
        interp_ref: Dict[int, np.ndarray] = {}
        for size in SIZES:
            print("seed", int(seed), "n_train", int(size), flush=True)
            idx = indices[int(size)]
            x_train = np.asarray(train.x[idx], dtype=np.float64)
            y_train = np.asarray(train.y[idx], dtype=np.float64)
            ae, prior, _scaler, y_mean, y_scale = base._fit_clp(
                x_train, y_train, int(seed), cfg
            )
            decoder_id = nested._weight_fingerprint(ae)
            z_val = prior.predict(x_val)
            z_cond = np.asarray(prior.predict(x_test), dtype=np.float64)
            z_zero = np.zeros_like(z_cond)
            y_prior = csgm.decode_latent_prior(ae, z_cond, y_mean, y_scale, device="cpu")
            perturbations = control._restart_perturbations(
                tuple(int(value) for value in z_cond.shape),
                int(cfg.csgm_restarts),
                int(seed),
            )
            noise_hash = control._tensor_list_hash(perturbations)
            diag_rows.append(
                {
                    "seed": int(seed),
                    "n_train": int(size),
                    "name": "decoder_train",
                    "rmse": base._decoder_reconstruction_rmse(ae, y_train, y_mean, y_scale),
                }
            )
            diag_rows.append(
                {
                    "seed": int(seed),
                    "n_train": int(size),
                    "name": "decoder_test",
                    "rmse": base._decoder_reconstruction_rmse(ae, y_test, y_mean, y_scale),
                }
            )
            diag_rows.append(
                {
                    "seed": int(seed),
                    "n_train": int(size),
                    "name": "prior_full",
                    "rmse": base.rmse(y_test, y_prior),
                }
            )
            diag_rows.append(
                {
                    "seed": int(seed),
                    "n_train": int(size),
                    "name": "prior_fixed",
                    "rmse": base.rmse(y_test[:, fixed], y_prior[:, fixed]),
                }
            )
            prior_full_ref: Optional[float] = None
            for m_obs in nested.M_OBS:
                print("  m_obs", int(m_obs), flush=True)
                positions = order[: int(m_obs)]
                operator = nested._operator_from_positions(positions)
                b_train = nested._slice_measurements(y_train, noise_pool[idx], positions)
                b_val = nested._slice_measurements(y_val, _noise_val, positions)
                b_test = nested._slice_measurements(y_test, noise_test, positions)
                bundle = {"M": operator, "B_val": b_val, "B_test": b_test}
                if int(size) == int(SIZES[0]):
                    first_bundle[int(m_obs)] = bundle
                else:
                    for name, array in bundle.items():
                        np.testing.assert_array_equal(array, first_bundle[int(m_obs)][name])
                lam, curve = nested._select_lambda_on_validation(
                    ae,
                    z_val,
                    y_val,
                    b_val,
                    operator,
                    positions,
                    fixed,
                    y_mean,
                    y_scale,
                    cfg,
                    int(seed),
                    int(y_val.shape[0]),
                )
                if int(seed) == control.PILOT_SEED and int(size) in control.PILOT_SIZES:
                    expected = _saved_lambda(nested_rows, int(size), int(m_obs))
                    if abs(float(lam) - expected) > 1e-15:
                        raise AssertionError(
                            "Seed-7 lambda drifted from the preserved pilot at n={} m={}.".format(size, m_obs)
                        )
                zero_score = min(
                    float(row["val_rmse_unobserved"]) for row in curve if float(row["lambda"]) == 0.0
                )
                positive_best = min(
                    float(row["val_rmse_unobserved"]) for row in curve if float(row["lambda"]) > 0.0
                )
                at_lower = int(abs(lam - min(value for value in nested.LAMBDA_GRID if value > 0.0)) < 1e-15)
                at_upper = int(abs(lam - max(nested.LAMBDA_GRID)) < 1e-15)
                for row in curve:
                    lambda_rows.append(
                        {
                            "seed": int(seed),
                            "n_train": int(size),
                            "m_obs": int(m_obs),
                            "lambda": float(row["lambda"]),
                            "val_rmse_unobserved": float(row["val_rmse_unobserved"]),
                            "val_rmse_full": float(row["val_rmse_full"]),
                            "val_rmse_fixed": float(row["val_rmse_fixed"]),
                            "selected_for_both_penalized_starts": int(
                                abs(float(row["lambda"]) - float(lam)) < 1e-15
                            ),
                            "at_lower_boundary": at_lower,
                            "at_upper_boundary": at_upper,
                            "zero_beats_all_positive": int(zero_score < positive_best),
                        }
                    )
                calls = (
                    (METHOD_ZERO, z_zero, z_zero, 0.0),
                    (METHOD_COND, z_cond, z_zero, 0.0),
                    (METHOD_ZERO_PEN, z_zero, z_cond, float(lam)),
                    (METHOD_COND_PEN, z_cond, z_cond, float(lam)),
                )
                scored: Dict[str, Dict[str, float]] = {}
                for method, z_init, z_anchor, penalty in calls:
                    prediction = control.recover_with_separate_anchor(
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
                    if int(seed) == control.PILOT_SEED and int(size) in control.PILOT_SIZES:
                        expected_rmse = _saved_rmse(control_rows, int(size), int(m_obs), method)
                        if abs(scores["rmse_full"] - expected_rmse) > 1e-5:
                            raise AssertionError(
                                "Seed-7 {} drifted at n={} m={}.".format(method, size, m_obs)
                            )
                ub_scaler = dub.fit_scaler_ub(dub.concat_ub(x_train, b_train))
                base.verify_scaler_on_train(
                    ub_scaler, dub.concat_ub(x_train, b_train), "ae_input_scaler"
                )
                _unused, y_ae = dub.fit_predict_ae_regression_ub(
                    cfg,
                    int(seed),
                    dub.DirectUBTrainConfig(),
                    dub.concat_ub(x_train, b_train),
                    y_train,
                    dub.concat_ub(x_val, b_val),
                    y_val,
                    dub.concat_ub(x_test, b_test),
                    ub_scaler,
                )
                y_interp = base._interpolate(b_test, positions, base.WINDOW_LEN)
                if int(m_obs) not in interp_ref:
                    interp_ref[int(m_obs)] = np.array(y_interp, copy=True)
                else:
                    np.testing.assert_array_equal(y_interp, interp_ref[int(m_obs)])
                scored[METHOD_PRIOR] = nested._score_trio(y_test, y_prior, positions, fixed)
                scored[METHOD_AE] = nested._score_trio(y_test, y_ae, positions, fixed)
                scored[METHOD_INTERP] = nested._score_trio(y_test, y_interp, positions, fixed)
                if prior_full_ref is None:
                    prior_full_ref = float(scored[METHOD_PRIOR]["rmse_full"])
                elif abs(float(scored[METHOD_PRIOR]["rmse_full"]) - prior_full_ref) > 1e-12:
                    raise AssertionError("Prior-only changed across observation counts.")
                if nested._weight_fingerprint(ae) != decoder_id:
                    raise AssertionError("Decoder weights changed inside a budget.")
                if int(seed) == control.PILOT_SEED and int(size) in control.PILOT_SIZES:
                    for method, preserved_name in (
                        (METHOD_AE, nested.METHOD_AE),
                        (METHOD_INTERP, nested.METHOD_INTERP),
                        (METHOD_PRIOR, nested.METHOD_PRIOR),
                    ):
                        expected_rmse = _saved_rmse(nested_rows, int(size), int(m_obs), preserved_name)
                        if abs(scored[method]["rmse_full"] - expected_rmse) > 1e-5:
                            raise AssertionError(
                                "Seed-7 {} drifted at n={} m={}.".format(method, size, m_obs)
                            )
                for method in ALL_METHODS:
                    penalty = float(lam) if method in (METHOD_COND_PEN, METHOD_ZERO_PEN) else 0.0
                    if method in (METHOD_PRIOR, METHOD_AE, METHOD_INTERP):
                        penalty_value: object = ""
                    else:
                        penalty_value = penalty
                    print(
                        "   ",
                        method,
                        "{:.6f}".format(scored[method]["rmse_full"]),
                        flush=True,
                    )
                    result_rows.append(
                        {
                            "seed": int(seed),
                            "n_train": int(size),
                            "m_obs": int(m_obs),
                            "method": method,
                            "rmse_full": scored[method]["rmse_full"],
                            "rmse_unobserved": scored[method]["rmse_unobserved"],
                            "rmse_fixed": scored[method]["rmse_fixed"],
                            "lambda": penalty_value,
                            "shared_noise_hash": noise_hash if method in control.SHARED_METHODS else "",
                            "restart_rule": "batch_objective",
                        }
                    )
                full_effects = _effects(scored, "rmse_full")
                unobs_effects = _effects(scored, "rmse_unobserved")
                fixed_effects = _effects(scored, "rmse_fixed")
                summary_rows.append(
                    {
                        "seed": int(seed),
                        "n_train": int(size),
                        "m_obs": int(m_obs),
                        "lambda": float(lam),
                        "lambda_at_lower_boundary": at_lower,
                        "lambda_at_upper_boundary": at_upper,
                        "zero_beats_all_positive": int(zero_score < positive_best),
                        "rmse_conditioned_init_anchored_penalty": scored[METHOD_COND_PEN]["rmse_full"],
                        "rmse_conditioned_init_no_penalty": scored[METHOD_COND]["rmse_full"],
                        "rmse_zero_init_anchored_penalty": scored[METHOD_ZERO_PEN]["rmse_full"],
                        "rmse_zero_init_no_penalty": scored[METHOD_ZERO]["rmse_full"],
                        "rmse_prior": scored[METHOD_PRIOR]["rmse_full"],
                        "rmse_ae": scored[METHOD_AE]["rmse_full"],
                        "rmse_interpolation": scored[METHOD_INTERP]["rmse_full"],
                        "clp_minus_ae": scored[METHOD_COND_PEN]["rmse_full"] - scored[METHOD_AE]["rmse_full"],
                        "clp_minus_interpolation": (
                            scored[METHOD_COND_PEN]["rmse_full"] - scored[METHOD_INTERP]["rmse_full"]
                        ),
                        "clp_minus_zero_init": (
                            scored[METHOD_COND_PEN]["rmse_full"] - scored[METHOD_ZERO]["rmse_full"]
                        ),
                        "penalty_effect_at_zero_init": full_effects["penalty_effect_at_zero_init"],
                        "penalty_effect_at_conditioned_init": full_effects["penalty_effect_at_conditioned_init"],
                        "init_effect_without_penalty": full_effects["init_effect_without_penalty"],
                        "init_effect_with_penalty": full_effects["init_effect_with_penalty"],
                        "interaction": full_effects["interaction"],
                        "penalty_effect_at_zero_init_unobserved": unobs_effects["penalty_effect_at_zero_init"],
                        "init_effect_without_penalty_unobserved": unobs_effects["init_effect_without_penalty"],
                        "interaction_unobserved": unobs_effects["interaction"],
                        "penalty_effect_at_zero_init_fixed": fixed_effects["penalty_effect_at_zero_init"],
                        "init_effect_without_penalty_fixed": fixed_effects["init_effect_without_penalty"],
                        "interaction_fixed": fixed_effects["interaction"],
                        "clp_minus_ae_fixed": (
                            scored[METHOD_COND_PEN]["rmse_fixed"] - scored[METHOD_AE]["rmse_fixed"]
                        ),
                        "clp_minus_interpolation_fixed": (
                            scored[METHOD_COND_PEN]["rmse_fixed"] - scored[METHOD_INTERP]["rmse_fixed"]
                        ),
                        "rmse_clp_unobserved": scored[METHOD_COND_PEN]["rmse_unobserved"],
                        "rmse_clp_fixed": scored[METHOD_COND_PEN]["rmse_fixed"],
                        "rmse_ae_fixed": scored[METHOD_AE]["rmse_fixed"],
                        "rmse_interpolation_fixed": scored[METHOD_INTERP]["rmse_fixed"],
                    }
                )
            if prior_full_ref is None:
                raise AssertionError("Prior-only was not scored.")

    across = _mean_rows(summary_rows)
    summary_fields = list(summary_rows[0].keys())
    result_fields = (
        "seed",
        "n_train",
        "m_obs",
        "method",
        "rmse_full",
        "rmse_unobserved",
        "rmse_fixed",
        "lambda",
        "shared_noise_hash",
        "restart_rule",
    )
    lambda_fields = (
        "seed",
        "n_train",
        "m_obs",
        "lambda",
        "val_rmse_unobserved",
        "val_rmse_full",
        "val_rmse_fixed",
        "selected_for_both_penalized_starts",
        "at_lower_boundary",
        "at_upper_boundary",
        "zero_beats_all_positive",
    )
    quartet_rows = [row for row in result_rows if str(row["method"]) in control.SHARED_METHODS]
    limitations = _limitations_text()
    readme = "\n".join(
        [
            "# Three-seed nested-mask and shared-restart grid",
            "",
            "Exploratory. Not a manuscript result.",
            "Reference commit: {}.".format(REFERENCE_COMMIT),
            "Preserved pilots were not overwritten.",
            "",
            "Seeds 7, 23, and 41. Budgets 24, 56, and 88. Observations 4, 6, and 13.",
            "Positive lambda is chosen on F06-1 and shared by the two penalized starts.",
            "The four latent configurations share decoder, mask, measurements, and perturbations.",
            "Direct AE, prior-only, and interpolation use those same measurements.",
            "",
            "clp_minus_* and effect columns are RMSE differences.",
            "Negative means the first named method, or the named factor, has lower RMSE.",
            "interaction = init_effect_with_penalty - init_effect_without_penalty.",
            "",
            RESTART_RULE,
            "",
            "See LIMITATIONS.md.",
            "",
        ]
    )
    for directory in (NESTED_FULL, CONTROL_FULL):
        os.makedirs(directory, exist_ok=True)
        _write_text(os.path.join(directory, "README.md"), readme)
        _write_text(os.path.join(directory, "LIMITATIONS.md"), limitations)
    nested_tables = os.path.join(NESTED_FULL, "tables")
    control_tables = os.path.join(CONTROL_FULL, "tables")
    nested_figures = os.path.join(NESTED_FULL, "figures")
    control_figures = os.path.join(CONTROL_FULL, "figures")
    _save_table(os.path.join(nested_tables, "results_by_seed.csv"), result_rows, result_fields)
    _save_table(os.path.join(nested_tables, "summary_by_seed.csv"), summary_rows, summary_fields)
    _save_table(
        os.path.join(nested_tables, "summary_across_seeds.csv"),
        across,
        list(across[0].keys()),
    )
    _save_table(os.path.join(nested_tables, "lambda_sensitivity.csv"), lambda_rows, lambda_fields)
    _save_table(
        os.path.join(nested_tables, "decoder_diagnostics.csv"),
        diag_rows,
        ("seed", "n_train", "name", "rmse"),
    )
    _save_table(
        os.path.join(nested_tables, "masks.csv"),
        mask_rows,
        ("seed", "position_rank", "window_index", "in_omega4", "in_omega6", "in_omega13"),
    )
    _save_table(os.path.join(control_tables, "results_by_seed.csv"), quartet_rows, result_fields)
    _save_table(os.path.join(control_tables, "summary_by_seed.csv"), summary_rows, summary_fields)
    _save_table(
        os.path.join(control_tables, "summary_across_seeds.csv"),
        across,
        list(across[0].keys()),
    )
    _plot_rmse(result_rows, "rmse_full", os.path.join(nested_figures, "rmse_full_by_seed_budget.png"))
    _plot_rmse(result_rows, "rmse_fixed", os.path.join(nested_figures, "rmse_fixed_by_seed_budget.png"))
    _plot_rmse(
        result_rows,
        "rmse_unobserved",
        os.path.join(nested_figures, "rmse_unobserved_by_seed_budget.png"),
    )
    _plot_lambda(lambda_rows, os.path.join(nested_figures, "lambda_validation_by_seed.png"))
    _plot_decoder(diag_rows, os.path.join(nested_figures, "decoder_prior_by_seed.png"))
    _plot_effects(summary_rows, os.path.join(control_figures, "effects_by_seed_budget.png"))
    _plot_rmse(
        quartet_rows,
        "rmse_full",
        os.path.join(control_figures, "quartet_rmse_by_seed_budget.png"),
        control.SHARED_METHODS,
    )
    config = {
        "reference_commit": REFERENCE_COMMIT,
        "seeds": list(SEEDS),
        "sizes": list(SIZES),
        "m_obs": list(nested.M_OBS),
        "lambda_rule": (
            "Unobserved RMSE on F06-1. The best positive lambda is used for both "
            "penalized starts. Lambda 0 is not assigned to those starts."
        ),
        "shared_factors": [
            "decoder within each seed and budget",
            "nested mask and noise field within each seed",
            "measurements across budgets",
            "restart perturbations across the four latent configurations",
        ],
        "restart_rule": RESTART_RULE,
        "effect_sign": "Negative means the named factor has lower RMSE.",
        "interaction": "init_effect_with_penalty minus init_effect_without_penalty.",
        "paired_unit": "seed",
        "val_dependence": val_dependence,
        "preserved_pilots_overwritten": False,
        "manuscript": "not updated",
        "historical_seed_offset_rerun": False,
    }
    checks = {
        "passed": [
            "synthetic nested-mask and split-anchor checks",
            "recorded training indices remain nested",
            "seed 7 permutation matches the preserved pilot",
            "seed 7 lambda and RMSE match the preserved pilots at 24 and 88 windows",
            "masks are nested and shared noise prefixes match",
            "measurements and interpolation match across budgets",
            "four latent configurations share one perturbation hash per budget",
            "positive lambda is the F06-1 minimum and is shared by both penalized starts",
            "F03-4 is not an argument of lambda selection",
            "prior-only full RMSE is identical across observation counts",
            "decoder weights are unchanged across observation counts",
            "scalers are fit on the training subset only",
            "restart choice remains the batch objective",
        ],
        "n_result_rows": len(result_rows),
        "n_seeds": len(SEEDS),
        "full_grid_executed": True,
    }
    for directory in (NESTED_FULL, CONTROL_FULL):
        with open(os.path.join(directory, "config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, sort_keys=True)
            handle.write("\n")
        with open(os.path.join(directory, "checks.json"), "w", encoding="utf-8") as handle:
            json.dump(checks, handle, indent=2, sort_keys=True)
            handle.write("\n")
    print("Wrote", NESTED_FULL)
    print("Wrote", CONTROL_FULL)


if __name__ == "__main__":
    main()
