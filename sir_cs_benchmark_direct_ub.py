#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_benchmark_direct_ub.py

Benchmark: direct [u,b] -> y baselines (MLP concat, PCA+MLP, AE+MLP) vs hybrid_fista
and optional LFISTA branch (ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint)
on the same synthetic protocol as sir_cs_pipeline_optimized.py.

Artifacts per run:
    <base_dir>/runs/<run_id>/
        tables/           CSVs + summary_focus_direct_ub.csv + parity_pooled.npz (pooled y vs y_hat)
        figures/          01--07 PNGs (same dashboard as other benchmarks) + 09_parity_ground_truth_vs_prediction.png
        logs/run_console.log
        PROTOCOL.txt
        config.json
        RUN_MANIFEST.txt

Psi ablation (pilot): use --residual-basis identity|dct; same alpha rule, only Psi changes.
Organize with separate --base-dir per axis, e.g. outputs/direct_ub_psi_ablation/identity vs .../dct.
See docs/direct_ub_psi_ablation.txt.

M-axis (with Psi fixed, usually dct): use --measurement-kind gaussian|subsample.
Reference Gaussian + DCT run: see docs/direct_ub_m_ablation.txt.

Measurement-noise robustness (DCT + subsample, joint-only): see docs/direct_ub_robustness_measurement_noise.txt.
Use --measurement-noise-std FLOAT and optional --robustness-lite (three seeds, full rho grid).

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
from typing import Dict, List, Optional, TextIO, Tuple

from sklearn.exceptions import ConvergenceWarning

import numpy as np
import pandas as pd

import direct_ub_baselines as dub
import external_benchmarks as extb
from sir_cs_pipeline_optimized import (
    METHOD_ORDER_DIRECT_UB,
    METHOD_ORDER_DIRECT_UB_JOINT_FOCUS,
    Config,
    MultiOutputMLP,
    apply_config_profile,
    build_lambda_selection_arrays,
    build_measurement_matrix,
    make_dataset,
    merge_gt_pred_bundles,
    plot_parity_ground_truth_vs_predictions,
    power_iteration_lipschitz,
    run_lfista_branch,
    save_all_comparison_plots,
    summarize_results_across_seeds,
    summarize_results_per_seed,
)


class _Tee:
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


def build_direct_ub_parity_fragment(
    Y_test: np.ndarray,
    Ybg_test: np.ndarray,
    pred_test_mlp: np.ndarray,
    pred_test_pca: np.ndarray,
    pred_test_ae: Optional[np.ndarray],
    run_ae: bool,
    y_hf_stack: Optional[np.ndarray],
    lf_gt: Optional[Dict[str, np.ndarray]],
    joint_only: bool,
) -> Dict[str, np.ndarray]:
    """Flatten test tensors for parity scatter (pooled over samples and output coordinates)."""
    out: Dict[str, np.ndarray] = {
        "y_true": np.asarray(Y_test, dtype=np.float64).ravel(),
        "ml_only": np.asarray(Ybg_test, dtype=np.float64).ravel(),
        "mlp_concat_ub": np.asarray(pred_test_mlp, dtype=np.float64).ravel(),
        "pca_regression_ub": np.asarray(pred_test_pca, dtype=np.float64).ravel(),
    }
    if run_ae and pred_test_ae is not None:
        out["ae_regression_ub"] = np.asarray(pred_test_ae, dtype=np.float64).ravel()
    if y_hf_stack is not None and y_hf_stack.size > 0:
        out["hybrid_fista"] = np.asarray(y_hf_stack, dtype=np.float64).ravel()
    if lf_gt:
        if joint_only:
            k = "hybrid_lfista_joint"
            if k in lf_gt:
                out[k] = np.asarray(lf_gt[k], dtype=np.float64).ravel()
        else:
            for k, v in lf_gt.items():
                out[str(k)] = np.asarray(v, dtype=np.float64).ravel()
    return out


def run_direct_ub_single_setting(
    cfg: Config,
    dub_cfg: dub.DirectUBTrainConfig,
    seed: int,
    measurement_ratio: float,
    include_hybrid_fista: bool,
    run_ae: bool,
    include_lfista: bool,
    joint_only: bool,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
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

    B_train = dub.make_B(Y_train, M, cfg.measurement_noise_std, rng)
    B_val = dub.make_B(Y_val, M, cfg.measurement_noise_std, rng)
    B_test = dub.make_B(Y_test, M, cfg.measurement_noise_std, rng)

    Xb_train = dub.concat_ub(X_train, B_train)
    Xb_val = dub.concat_ub(X_val, B_val)
    Xb_test = dub.concat_ub(X_test, B_test)
    ub_scaler = dub.fit_scaler_ub(Xb_train)

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

    y_sel, ybg_sel, b_sel, z_sel, _ap = build_lambda_selection_arrays(
        cfg=cfg,
        M=M,
        X_val=X_val,
        Y_val=Y_val,
        baseline_model=baseline,
        alpha_model=alpha_model,
        rng=rng,
    )

    lam_hf = None
    if include_hybrid_fista:
        lam_hf = extb.hybrid_fista_best_lambda(
            cfg, A, Psi, y_sel, ybg_sel, b_sel, z_sel, L_A
        )

    pred_val_mlp, pred_test_mlp = dub.fit_predict_mlp_concat(
        cfg, seed, Xb_train, Y_train, Xb_val, Y_val, Xb_test, ub_scaler
    )
    pred_val_pca, pred_test_pca, best_r = dub.fit_predict_pca_regression_ub(
        cfg, seed, dub_cfg, Xb_train, Y_train, Xb_val, Y_val, Xb_test, ub_scaler
    )
    pred_test_ae: Optional[np.ndarray] = None
    if run_ae:
        pred_val_ae, pred_test_ae = dub.fit_predict_ae_regression_ub(
            cfg, seed, dub_cfg, Xb_train, Y_train, Xb_val, Y_val, Xb_test, ub_scaler
        )

    Ybg_test = baseline.predict(X_test)
    nan_f = float("nan")
    rows: List[dict[str, float | int | str]] = []
    n_test = len(X_test)
    y_hf_rows: List[np.ndarray] = []

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
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "mlp_concat_ub",
                i,
                Y_test[i],
                pred_test_mlp[i],
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                nan_f,
                "direct_ub",
                m,
                support_f1_override=nan_f,
            )
        )
        rows.append(
            extb.per_sample_metrics_row(
                seed,
                measurement_ratio,
                "pca_regression_ub",
                i,
                Y_test[i],
                pred_test_pca[i],
                Alpha_test[i],
                np.zeros_like(Alpha_test[i]),
                float(best_r),
                "pca_mlp",
                m,
                support_f1_override=nan_f,
            )
        )
        if run_ae:
            rows.append(
                extb.per_sample_metrics_row(
                    seed,
                    measurement_ratio,
                    "ae_regression_ub",
                    i,
                    Y_test[i],
                    pred_test_ae[i],
                    Alpha_test[i],
                    np.zeros_like(Alpha_test[i]),
                    float(dub_cfg.ae_latent_dim),
                    "ae_mlp",
                    m,
                    support_f1_override=nan_f,
                )
            )
        if include_hybrid_fista and lam_hf is not None:
            ah_hf, y_hf = extb.hybrid_fista_predict_one(
                cfg, A, Psi, y_ml, z_i, lam_hf, L_A
            )
            y_hf_rows.append(np.asarray(y_hf, dtype=np.float64))
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

    df = pd.DataFrame(rows)
    y_hf_stack: Optional[np.ndarray]
    if y_hf_rows:
        y_hf_stack = np.stack(y_hf_rows, axis=0)
    else:
        y_hf_stack = None

    lf_gt: Optional[Dict[str, np.ndarray]] = None
    if include_lfista:

        def _lf_log(_msg: str) -> None:
            return None

        lf_df, lf_gt = run_lfista_branch(
            cfg,
            seed,
            measurement_ratio,
            data,
            M,
            _lf_log,
        )
        if joint_only:
            lf_df = lf_df[lf_df["method"] == "hybrid_lfista_joint"].copy()
        df = pd.concat([df, lf_df], ignore_index=True)

    parity_fragment = build_direct_ub_parity_fragment(
        Y_test,
        Ybg_test,
        pred_test_mlp,
        pred_test_pca,
        pred_test_ae,
        run_ae,
        y_hf_stack,
        lf_gt,
        joint_only,
    )
    return df, parity_fragment


def save_focus_tables(
    run_root: str,
    summary: pd.DataFrame,
    per_seed: pd.DataFrame,
    include_hybrid_fista: bool,
    run_ae: bool,
    include_lfista: bool,
    joint_only: bool,
) -> List[str]:
    if joint_only:
        focus = list(METHOD_ORDER_DIRECT_UB_JOINT_FOCUS)
        if not run_ae:
            focus = [x for x in focus if x != "ae_regression_ub"]
    else:
        focus = list(METHOD_ORDER_DIRECT_UB)
        if not include_hybrid_fista:
            focus = [x for x in focus if x != "hybrid_fista"]
        if not run_ae:
            focus = [x for x in focus if x != "ae_regression_ub"]
        if not include_lfista:
            focus = [
                x
                for x in focus
                if x not in ("ml_only_torch", "hybrid_lfista_frozen", "hybrid_lfista_joint")
            ]
    sub_s = summary[summary["method"].isin(focus)].sort_values(["measurement_ratio", "method"])
    sub_p = per_seed[per_seed["method"].isin(focus)].sort_values(
        ["seed", "measurement_ratio", "method"]
    )
    tables = os.path.join(run_root, "tables")
    if joint_only:
        p1 = os.path.join(tables, "summary_focus_lfista_joint_vs_ub.csv")
        p2 = os.path.join(tables, "summary_by_seed_focus_lfista_joint_vs_ub.csv")
    else:
        p1 = os.path.join(tables, "summary_focus_direct_ub.csv")
        p2 = os.path.join(tables, "summary_by_seed_focus_direct_ub.csv")
    sub_s.to_csv(p1, index=False)
    sub_p.to_csv(p2, index=False)
    return [p1, p2]


def write_protocol(
    run_root: str,
    joint_only: bool,
    residual_basis: str,
    measurement_kind: str,
    measurement_noise_std: float,
) -> str:
    common = [
        "Direct [u,b] -> y benchmark (methodological protocol)",
        "",
        f"0) residual_basis={residual_basis} (Psi = get_basis(N, residual_basis); y_res = Psi @ alpha per row).",
        "   Alpha support/amplitudes use the same generator rule as identity runs; only Psi differs in Psi-axis studies.",
        f"0b) measurement_kind={measurement_kind} (M = build_measurement_matrix: gaussian = i.i.d. N(0,1/sqrt(m)) rows;",
        "    subsample = m distinct coordinate indicators, one per row). M-axis studies change only this, keeping other knobs fixed when configured.",
        f"0c) measurement_noise_std={measurement_noise_std} (eta in b = M y + eta on train/val/test and per-sample test rows).",
        "",
        "1) Same synthetic generator and splits as sir_cs_pipeline_optimized (make_dataset).",
        "2) Same M, noise on b, per (seed, measurement_ratio) as hybrid CS evaluation.",
        "3) Train features: [u, b] with b = M y + eta on train/val/test (independent noise per row).",
        "4) StandardScaler fit on train [u,b] only; transform val/test.",
        "5) mlp_concat_ub: sklearn MLPRegressor, same hidden/max_iter/alpha as Config baseline MLP.",
        "6) pca_regression_ub: PCA on Y_train only; r chosen by val RMSE/MAE (cfg.model_selection_metric);",
        "   MLP maps scaled [u,b] to PCA scores; inverse transform to y.",
        "7) ae_regression_ub: PyTorch AE on standardized Y_train; MLP maps [u,b] to latent; decode.",
        "8) ml_only: sklearn MLP on u only (reference).",
    ]
    if joint_only:
        extra = [
            "9) Proposed method only: hybrid_lfista_joint (run_lfista_branch, keep joint rows only).",
            "   No hybrid_fista, ml_only_torch, or hybrid_lfista_frozen in this run.",
            "",
        ]
    else:
        extra = [
            "8b) hybrid_fista: same validation lambda selection and per-test residual CS as pipeline.",
            "9) LFISTA branch optional: ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint from run_lfista_branch.",
            "",
        ]
    text = "\n".join(common + extra)
    path = os.path.join(run_root, "PROTOCOL.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def write_run_manifest(
    run_root: str,
    run_id: str,
    elapsed_s: float,
    plot_paths: List[str],
    focus_paths: List[str],
    joint_only: bool,
    residual_basis: str,
    measurement_kind: str,
    measurement_noise_std: float,
) -> str:
    title = (
        "Direct [u,b] benchmark: hybrid_lfista_joint vs mlp_concat / PCA / AE (joint-only mode)."
        if joint_only
        else "Direct [u,b] benchmark (+ optional full LFISTA branch)."
    )
    lines = [
        title,
        f"run_id: {run_id}",
        f"elapsed_seconds: {elapsed_s:.1f}",
        f"residual_basis: {residual_basis}",
        f"measurement_kind: {measurement_kind}",
        f"measurement_noise_std: {measurement_noise_std}",
        "",
        "tables/",
        "  detailed_results.csv",
        "  summary_by_seed.csv",
        "  summary.csv",
    ]
    if joint_only:
        lines.extend(
            [
                "  summary_focus_lfista_joint_vs_ub.csv",
                "  summary_by_seed_focus_lfista_joint_vs_ub.csv",
            ]
        )
    else:
        lines.extend(
            [
                "  summary_focus_direct_ub.csv",
                "  summary_by_seed_focus_direct_ub.csv",
            ]
        )
    lines.extend(
        [
            "  PROTOCOL.txt",
            "",
            "figures/",
        ]
    )
    for p in sorted(plot_paths):
        lines.append(f"  {os.path.relpath(p, run_root)}")
    lines.append("")
    lines.append("focus_tables:")
    for p in focus_paths:
        lines.append(f"  {os.path.relpath(p, run_root)}")
    path = os.path.join(run_root, "RUN_MANIFEST.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Direct [u,b]->y baselines vs hybrid_fista.")
    p.add_argument("--explore", action="store_true", help="Fast profile (one seed, two ratios).")
    p.add_argument(
        "--base-dir",
        type=str,
        default="outputs/direct_ub_benchmark",
        help="Base directory; runs go to base_dir/runs/<run_id>/.",
    )
    p.add_argument("--run-id", type=str, default="", help="Run folder name (default: timestamp).")
    p.add_argument("--no-hybrid-fista", action="store_true", help="Skip hybrid_fista reference.")
    p.add_argument("--no-ae", action="store_true", help="Skip AE baseline (faster).")
    p.add_argument("--no-plots", action="store_true", help="Skip figure generation.")
    p.add_argument(
        "--no-parity",
        action="store_true",
        help="Skip parity scatter (09_parity_...) and tables/parity_pooled.npz export.",
    )
    p.add_argument(
        "--no-lfista",
        action="store_true",
        help="Skip PyTorch LFISTA branch (ml_only_torch, hybrid_lfista_frozen, hybrid_lfista_joint).",
    )
    p.add_argument(
        "--joint-only",
        action="store_true",
        help=(
            "Compare only hybrid_lfista_joint to direct [u,b] baselines (mlp_concat, PCA, AE); "
            "omit hybrid_fista, ml_only_torch, hybrid_lfista_frozen. Implies LFISTA on."
        ),
    )
    p.add_argument(
        "--residual-basis",
        type=str,
        default="identity",
        choices=("identity", "dct"),
        help=(
            "Residual operator Psi for make_dataset and CS (identity vs DCT). "
            "Default identity. For Psi-axis pilot, run twice with separate --base-dir (see docs/direct_ub_psi_ablation.txt)."
        ),
    )
    p.add_argument(
        "--measurement-kind",
        type=str,
        default="gaussian",
        choices=("gaussian", "subsample"),
        help=(
            "How M is built: gaussian (dense) vs subsample (coordinate selection). "
            "Default gaussian. For M-axis with DCT Psi, see docs/direct_ub_m_ablation.txt."
        ),
    )
    p.add_argument(
        "--measurement-noise-std",
        type=float,
        default=None,
        help="Override cfg.measurement_noise_std after profile load (default: keep profile value, usually 0.02).",
    )
    p.add_argument(
        "--robustness-lite",
        action="store_true",
        help=(
            "Joint-only profile with three seeds and full rho grid (direct_ub_lfista_joint_robustness_lite). "
            "Mutually exclusive with --explore; requires --joint-only."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.joint_only) and bool(args.no_lfista):
        print("--joint-only requires LFISTA; remove --no-lfista.", file=sys.stderr)
        sys.exit(2)
    if bool(args.explore) and bool(args.robustness_lite):
        print("Use only one of --explore or --robustness-lite.", file=sys.stderr)
        sys.exit(2)
    if bool(args.robustness_lite) and not bool(args.joint_only):
        print("--robustness-lite requires --joint-only.", file=sys.stderr)
        sys.exit(2)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    cfg = Config()
    cfg.log_progress = False
    joint_only = bool(args.joint_only)
    if joint_only:
        if bool(args.robustness_lite):
            cfg.config_profile = "direct_ub_lfista_joint_robustness_lite"
        elif args.explore:
            cfg.config_profile = "direct_ub_lfista_joint_only_explore"
        else:
            cfg.config_profile = "direct_ub_lfista_joint_only"
    elif args.explore:
        cfg.config_profile = "direct_ub_benchmark_explore"
    else:
        cfg.config_profile = "direct_ub_benchmark"
    apply_config_profile(cfg)
    if args.measurement_noise_std is not None:
        cfg.measurement_noise_std = float(args.measurement_noise_std)
    rb = str(args.residual_basis).strip().lower()
    if rb not in ("identity", "dct"):
        print("--residual-basis must be identity or dct.", file=sys.stderr)
        sys.exit(2)
    cfg.residual_basis = rb  # type: ignore[assignment]
    mk = str(args.measurement_kind).strip().lower()
    if mk not in ("gaussian", "subsample"):
        print("--measurement-kind must be gaussian or subsample.", file=sys.stderr)
        sys.exit(2)
    cfg.measurement_kind = mk  # type: ignore[assignment]

    include_lfista = joint_only or (not bool(args.no_lfista))
    if include_lfista:
        cfg.run_lfista = True

    include_hf = False if joint_only else (not bool(args.no_hybrid_fista))

    dub_cfg = dub.DirectUBTrainConfig()

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

    run_ae = not bool(args.no_ae)
    all_dfs: List[pd.DataFrame] = []
    parity_bundles: List[Dict[str, np.ndarray]] = []
    t0 = time.time()
    job_idx = 0
    total = len(cfg.seeds) * len(cfg.measurement_ratios)
    try:
        _log(tee, f"Run root: {run_root}")
        _log(
            tee,
            f"Profile: {cfg.config_profile} | jobs: {total} | joint_only: {joint_only} | "
            f"hybrid_fista: {include_hf} | ae: {run_ae} | lfista: {include_lfista} | "
            f"residual_basis: {cfg.residual_basis} | measurement_kind: {cfg.measurement_kind} | "
            f"measurement_noise_std: {cfg.measurement_noise_std}",
        )
        for seed in cfg.seeds:
            for mr in cfg.measurement_ratios:
                job_idx += 1
                _log(tee, f"--- job {job_idx}/{total} seed={seed} measurement_ratio={mr:.2f} ---")
                df, pfrag = run_direct_ub_single_setting(
                    cfg, dub_cfg, seed, mr, include_hf, run_ae, include_lfista, joint_only
                )
                all_dfs.append(df)
                parity_bundles.append(pfrag)

        detailed = pd.concat(all_dfs, ignore_index=True)
        per_seed = summarize_results_per_seed(detailed)
        summary = summarize_results_across_seeds(per_seed)

        detailed.to_csv(os.path.join(tables_dir, "detailed_results.csv"), index=False)
        per_seed.to_csv(os.path.join(tables_dir, "summary_by_seed.csv"), index=False)
        summary.to_csv(os.path.join(tables_dir, "summary.csv"), index=False)

        focus_paths = save_focus_tables(
            run_root, summary, per_seed, include_hf, run_ae, include_lfista, joint_only
        )
        proto_path = write_protocol(
            run_root,
            joint_only,
            str(cfg.residual_basis),
            str(cfg.measurement_kind),
            float(cfg.measurement_noise_std),
        )

        plot_paths: List[str] = []
        if not args.no_plots:
            plot_paths = save_all_comparison_plots(cfg, summary, per_seed)
            _log(tee, f"Figures: {len(plot_paths)} files under {os.path.join(run_root, 'figures')}")
        if not bool(args.no_plots) and not bool(args.no_parity) and parity_bundles:
            merged_parity = merge_gt_pred_bundles(parity_bundles)
            npz_path = os.path.join(tables_dir, "parity_pooled.npz")
            np.savez_compressed(npz_path, **merged_parity)
            parity_png = os.path.join(run_root, cfg.plots_subdir, "09_parity_ground_truth_vs_prediction.png")
            plot_parity_ground_truth_vs_predictions(cfg, merged_parity, parity_png)
            plot_paths.append(npz_path)
            plot_paths.append(parity_png)
            _log(tee, f"Parity: {parity_png} | arrays: {npz_path}")

        cfg_dump = asdict(cfg)
        dub_dump = asdict(dub_cfg)
        dub_dump["pca_r_grid"] = list(dub_dump["pca_r_grid"])
        cfg_dump["direct_ub_benchmark"] = {
            "run_id": run_id,
            "joint_only_lfista_joint": joint_only,
            "include_hybrid_fista": include_hf,
            "include_lfista": include_lfista,
            "include_ae": run_ae,
            "residual_basis": str(cfg.residual_basis),
            "measurement_kind": str(cfg.measurement_kind),
            "dub_cfg": dub_dump,
            "protocol_txt": proto_path,
            "tables_dir": tables_dir,
            "figures_dir": os.path.join(run_root, "figures"),
        }
        with open(os.path.join(run_root, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg_dump, f, indent=2)

        elapsed = time.time() - t0
        manifest_path = write_run_manifest(
            run_root,
            run_id,
            elapsed,
            plot_paths,
            focus_paths + [proto_path],
            joint_only,
            str(cfg.residual_basis),
            str(cfg.measurement_kind),
            float(cfg.measurement_noise_std),
        )
        _log(tee, f"Done in {elapsed:.1f}s | manifest: {manifest_path}")
    finally:
        sys.stdout = old_stdout
        tee.close()

    print(f"Artifacts: {run_root}", flush=True)


if __name__ == "__main__":
    main()
