#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_benchmark_stage1.py

Etapa 1: external sklearn baselines (S1/S2/S3) on the same synthetic protocol as
sir_cs_pipeline_optimized.py, plus optional hybrid_fista reference.

Artifacts layout (each run):
    <base_dir>/runs/<run_id>/
        tables/     detailed_results.csv, summary*.csv, summary_focus_stage1.csv
        figures/    comparison PNGs (same naming as main pipeline)
        logs/       run_console.log
        config.json
        RUN_MANIFEST.txt

ASCII-only source.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from typing import List, Optional, TextIO

from sklearn.exceptions import ConvergenceWarning

import numpy as np
import pandas as pd

import external_benchmarks as extb
from sir_cs_pipeline_optimized import (
    METHOD_ORDER_STAGE1_EXTERNAL,
    Config,
    MultiOutputMLP,
    apply_config_profile,
    build_lambda_selection_arrays,
    build_measurement_matrix,
    make_dataset,
    power_iteration_lipschitz,
    save_all_comparison_plots,
    summarize_results_across_seeds,
    summarize_results_per_seed,
)


class _Tee:
    """Write to real stdout and optional log file (must not replace sys.stdout with self)."""

    def __init__(self, real_stdout: TextIO, path: Optional[str]) -> None:
        self._real = real_stdout
        self._path = path
        self._f: Optional[TextIO] = None
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self._f = open(path, "w", encoding="utf-8")

    def write(self, s: str) -> None:
        self._real.write(s)
        if self._f:
            self._f.write(s)
            self._f.flush()

    def flush(self) -> None:
        self._real.flush()
        if self._f:
            self._f.flush()

    def close(self) -> None:
        if self._f:
            self._f.close()
            self._f = None


def _log(tee: Optional[_Tee], msg: str) -> None:
    line = f"{msg}\n"
    if tee:
        tee.write(line)
    else:
        print(msg, flush=True)


def run_stage1_single_setting(
    cfg: Config,
    seed: int,
    measurement_ratio: float,
    include_hybrid_fista: bool,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = make_dataset(cfg, seed=seed)
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    Y_train = data["Y_train"]
    Y_val = data["Y_val"]
    Y_test = data["Y_test"]
    Alpha_test = data["Alpha_test"]
    Psi = data["Psi"]

    m = max(4, int(round(measurement_ratio * cfg.n_output)))
    M = build_measurement_matrix(m, cfg.n_output, cfg.measurement_kind, rng)
    A = M @ Psi
    L_A = power_iteration_lipschitz(A, n_iter=cfg.power_iteration_n_iter)

    baseline = MultiOutputMLP(
        hidden_layer_sizes=cfg.baseline_hidden,
        max_iter=cfg.baseline_max_iter,
        learning_rate_init=cfg.baseline_learning_rate_init,
        alpha=cfg.baseline_alpha,
        early_stopping=cfg.baseline_early_stopping,
        random_state=seed,
    )
    baseline.fit(X_train, Y_train)

    alpha_model = None
    if cfg.use_alpha_predictor:
        alpha_model = MultiOutputMLP(
            hidden_layer_sizes=cfg.alpha_hidden,
            max_iter=cfg.alpha_max_iter,
            learning_rate_init=cfg.alpha_learning_rate_init,
            alpha=cfg.alpha_alpha,
            early_stopping=cfg.alpha_early_stopping,
            random_state=seed + 999,
        )
        alpha_model.fit(X_train, data["Alpha_train"])

    y_sel, ybg_sel, b_sel, z_sel, _alpha_pred_sel = build_lambda_selection_arrays(
        cfg=cfg,
        M=M,
        X_val=X_val,
        Y_val=Y_val,
        baseline_model=baseline,
        alpha_model=alpha_model,
        rng=rng,
    )

    alpha_grid = extb.sklearn_lasso_alpha_grid(cfg, m)
    best_alpha_s1 = extb.select_best_sklearn_lasso_alpha(
        cfg, A, Psi, y_sel, ybg_sel, b_sel, z_sel, "S1_hybrid", alpha_grid, L_A
    )
    best_alpha_s2 = extb.select_best_sklearn_lasso_alpha(
        cfg, A, Psi, y_sel, ybg_sel, b_sel, z_sel, "S2_cs_only", alpha_grid, L_A
    )

    lam_hf = None
    if include_hybrid_fista:
        lam_hf = extb.hybrid_fista_best_lambda(
            cfg, A, Psi, y_sel, ybg_sel, b_sel, z_sel, L_A
        )

    Ybg_test = baseline.predict(X_test)
    rows: List[dict[str, float | int | str]] = []
    n_test = len(X_test)
    nan_f = float("nan")
    for i in range(n_test):
        noise = cfg.measurement_noise_std * rng.normal(size=m)
        b_i = M @ Y_test[i] + noise
        y_ml = Ybg_test[i]
        z_i = b_i - M @ y_ml

        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "ml_only",
                i,
                Y_test[i],
                y_ml,
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                nan_f,
                "none",
                m,
                support_f1_override=nan_f,
            )
        )

        ah_s1 = extb.fit_lasso_coeffs(
            A, z_i, best_alpha_s1, max_iter=extb.default_lasso_max_iter(cfg)
        )
        y_s1 = y_ml + Psi @ ah_s1
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "ext_sklearn_lasso_S1_hybrid",
                i,
                Y_test[i],
                y_s1,
                Alpha_test[i],
                ah_s1,
                best_alpha_s1,
                "sklearn_lasso",
                m,
            )
        )

        ah_s2 = extb.fit_lasso_coeffs(
            A, b_i, best_alpha_s2, max_iter=extb.default_lasso_max_iter(cfg)
        )
        y_s2 = Psi @ ah_s2
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "ext_sklearn_lasso_S2_cs_only",
                i,
                Y_test[i],
                y_s2,
                Alpha_test[i],
                ah_s2,
                best_alpha_s2,
                "sklearn_lasso",
                m,
            )
        )

        ah_s3 = extb.fit_omp_coeffs(A, z_i, cfg.residual_k)
        y_s3 = y_ml + Psi @ ah_s3
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "ext_sklearn_omp_S3_hybrid_oracle_k",
                i,
                Y_test[i],
                y_s3,
                Alpha_test[i],
                ah_s3,
                float(cfg.residual_k),
                "sklearn_omp",
                m,
            )
        )

        if include_hybrid_fista and lam_hf is not None:
            ah_hf, y_hf = extb.hybrid_fista_predict_one(
                cfg, A, Psi, y_ml, z_i, lam_hf, L_A
            )
            rows.append(
                extb.per_sample_metrics_row(
                    seed,
                    measurement_ratio,
                    "hybrid_fista",
                    i,
                    Y_test[i],
                    y_hf,
                    Alpha_test[i],
                    ah_hf,
                    lam_hf,
                    "fista",
                    m,
                )
            )

    return pd.DataFrame(rows)


def save_focus_tables_stage1(
    run_root: str,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    include_hybrid_fista: bool,
) -> List[str]:
    """Filtered CSVs for Stage-1 methods only."""
    focus = list(METHOD_ORDER_STAGE1_EXTERNAL)
    if not include_hybrid_fista:
        focus = [m for m in focus if m != "hybrid_fista"]
    sub_s = summary[summary["method"].isin(focus)].sort_values(["measurement_ratio", "method"])
    sub_p = per_seed[per_seed["method"].isin(focus)].sort_values(
        ["seed", "measurement_ratio", "method"]
    )
    tables = os.path.join(run_root, "tables")
    p1 = os.path.join(tables, "summary_focus_stage1.csv")
    p2 = os.path.join(tables, "summary_by_seed_focus_stage1.csv")
    sub_s.to_csv(p1, index=False)
    sub_p.to_csv(p2, index=False)
    return [p1, p2]


def write_run_manifest(
    run_root: str,
    run_id: str,
    elapsed_s: float,
    plot_paths: List[str],
    focus_paths: List[str],
) -> str:
    lines = [
        "SIR-CS external benchmark Etapa 1 (sklearn Lasso S1/S2, OMP S3, optional hybrid_fista).",
        f"run_id: {run_id}",
        f"elapsed_seconds: {elapsed_s:.1f}",
        "",
        "tables/",
        "  detailed_results.csv",
        "  summary_by_seed.csv",
        "  summary.csv",
        "  summary_focus_stage1.csv",
        "  summary_by_seed_focus_stage1.csv",
        "",
        "figures/",
    ]
    for p in sorted(plot_paths):
        rel = os.path.relpath(p, run_root)
        lines.append(f"  {rel}")
    lines.append("")
    lines.append("focus_tables:")
    for p in focus_paths:
        lines.append(f"  {os.path.relpath(p, run_root)}")
    path = os.path.join(run_root, "RUN_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="External sklearn benchmark Etapa 1 (S1/S2/S3).")
    p.add_argument(
        "--explore",
        action="store_true",
        help="Fast settings: one seed, two ratios, smaller nets.",
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default="outputs/external_benchmark_stage1",
        help="Base directory; each run goes under base_dir/runs/<run_id>/.",
    )
    p.add_argument(
        "--run-id",
        type=str,
        default="",
        help="Run folder name (default: timestamp YYYYMMDD_HHMMSS).",
    )
    p.add_argument(
        "--no-hybrid-fista",
        action="store_true",
        help="Skip pipeline hybrid_fista reference rows.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation (CSVs only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = Config()
    cfg.log_progress = False

    if args.explore:
        cfg.config_profile = "external_benchmark_stage1_explore"
    else:
        cfg.config_profile = "external_benchmark_stage1"
    apply_config_profile(cfg)

    run_id = args.run_id.strip() or time.strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.abspath(args.base_dir)
    run_root = os.path.join(base_dir, "runs", run_id)
    tables_dir = os.path.join(run_root, "tables")
    logs_dir = os.path.join(run_root, "logs")
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    cfg.save_dir = run_root
    cfg.plots_subdir = "figures"

    log_path = os.path.join(logs_dir, "run_console.log")
    old_stdout = sys.stdout
    tee = _Tee(old_stdout, log_path)
    sys.stdout = tee  # type: ignore[assignment]

    include_hf = not bool(args.no_hybrid_fista)
    all_dfs: List[pd.DataFrame] = []
    t0 = time.time()
    job_idx = 0
    total = len(cfg.seeds) * len(cfg.measurement_ratios)
    try:
        _log(tee, f"Run root: {run_root}")
        _log(tee, f"Profile: {cfg.config_profile} | jobs: {total} | hybrid_fista: {include_hf}")
        for seed in cfg.seeds:
            for mr in cfg.measurement_ratios:
                job_idx += 1
                _log(tee, f"--- job {job_idx}/{total} seed={seed} measurement_ratio={mr:.2f} ---")
                df = run_stage1_single_setting(cfg, seed, mr, include_hf)
                all_dfs.append(df)

        detailed = pd.concat(all_dfs, ignore_index=True)
        per_seed = summarize_results_per_seed(detailed)
        summary = summarize_results_across_seeds(per_seed)

        detailed.to_csv(os.path.join(tables_dir, "detailed_results.csv"), index=False)
        per_seed.to_csv(os.path.join(tables_dir, "summary_by_seed.csv"), index=False)
        summary.to_csv(os.path.join(tables_dir, "summary.csv"), index=False)

        focus_paths = save_focus_tables_stage1(run_root, summary, per_seed, include_hf)

        plot_paths: List[str] = []
        if not args.no_plots:
            plot_paths = save_all_comparison_plots(cfg, summary, per_seed)
            _log(tee, f"Figures: {len(plot_paths)} files under {os.path.join(run_root, 'figures')}")

        cfg_dump = asdict(cfg)
        cfg_dump["benchmark_stage1"] = {
            "run_id": run_id,
            "scenarios": {
                "S1": "ext_sklearn_lasso_S1_hybrid: rhs=z, sklearn Lasso, alpha by val grid",
                "S2": "ext_sklearn_lasso_S2_cs_only: rhs=b, sklearn Lasso, alpha by val grid",
                "S3": "ext_sklearn_omp_S3_hybrid_oracle_k: rhs=z, OMP with n_nonzero=cfg.residual_k",
            },
            "include_hybrid_fista": include_hf,
            "tables_dir": tables_dir,
            "figures_dir": os.path.join(run_root, "figures"),
        }
        with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2)

        elapsed = time.time() - t0
        manifest_path = write_run_manifest(run_root, run_id, elapsed, plot_paths, focus_paths)
        _log(tee, f"Done in {elapsed:.1f}s | manifest: {manifest_path}")
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"Artifacts: {run_root}", flush=True)


if __name__ == "__main__":
    main()
