#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sir_cs_lfista.py

Etapa 2 do roadmap SIR-CS: versao minima em PyTorch com bloco LFISTA unrolled.

O nucleo de modelos e treino vive em lfista_module.py; este ficheiro mantem o
laboratorio standalone (perfis, artefactos, figuras).

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
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lfista_module import (
    LFISTATrainConfig,
    build_measurement_matrix,
    run_lfista_experiment_dataframe,
)


# ============================================================
# 1) Configuracao
# ============================================================


@dataclass
class LFISTAConfig:
    profile: Literal["phase2_lfista", "explore"] = "phase2_lfista"

    seeds: List[int] = field(default_factory=lambda: [7, 13, 23])

    n_train: int = 1200
    n_val: int = 300
    n_test: int = 300
    p_input: int = 12
    n_output: int = 128

    residual_basis: str = "identity"
    residual_k: int = 6
    residual_amplitude: float = 1.2
    residual_mode: str = "support_from_u"

    measurement_kind: str = "gaussian"
    measurement_ratios: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])
    measurement_noise_std: float = 0.02
    output_noise_std: float = 0.01

    bg_hidden: Tuple[int, int] = (128, 128)
    bg_dropout: float = 0.0

    lfista_steps: int = 8
    learn_step_sizes: bool = True
    learn_thresholds: bool = True
    init_step_scale: float = 1.0
    init_threshold: float = 1e-2
    use_momentum: bool = True

    batch_size: int = 128
    num_epochs_bg: int = 80
    num_epochs_frozen: int = 60
    num_epochs_joint: int = 80
    lr_bg: float = 1e-3
    lr_frozen: float = 1e-3
    lr_joint: float = 5e-4
    weight_decay: float = 1e-5
    patience: int = 12

    loss_alpha_weight: float = 0.0
    loss_l1_alpha_weight: float = 0.0

    save_dir: str = "outputs/lfista_baseline"
    plots_subdir: str = "../paper/figures/lfista_baseline"
    n_example_plots: int = 3
    max_gt_scatter_points: int = 50000
    log_progress: bool = True
    artifact_log_path: Optional[str] = None
    run_artifact_id: str = ""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def apply_profile(cfg: LFISTAConfig) -> None:
    if cfg.profile == "phase2_lfista":
        cfg.seeds = [7, 13, 23, 29, 31, 37, 41, 43, 47, 53]
        cfg.measurement_ratios = [0.2, 0.3, 0.4, 0.5, 0.6]
        cfg.save_dir = "outputs/lfista_baseline"
        cfg.plots_subdir = "../paper/figures/lfista_baseline"
        return
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
        return


def lfista_config_to_train(cfg: LFISTAConfig) -> LFISTATrainConfig:
    return LFISTATrainConfig(
        device=cfg.device,
        p_input=cfg.p_input,
        n_output=cfg.n_output,
        bg_hidden=cfg.bg_hidden,
        bg_dropout=cfg.bg_dropout,
        lfista_steps=cfg.lfista_steps,
        learn_step_sizes=cfg.learn_step_sizes,
        learn_thresholds=cfg.learn_thresholds,
        init_step_scale=cfg.init_step_scale,
        init_threshold=cfg.init_threshold,
        use_momentum=cfg.use_momentum,
        batch_size=cfg.batch_size,
        num_epochs_bg=cfg.num_epochs_bg,
        num_epochs_frozen=cfg.num_epochs_frozen,
        num_epochs_joint=cfg.num_epochs_joint,
        lr_bg=cfg.lr_bg,
        lr_frozen=cfg.lr_frozen,
        lr_joint=cfg.lr_joint,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        loss_alpha_weight=cfg.loss_alpha_weight,
        loss_l1_alpha_weight=cfg.loss_l1_alpha_weight,
        measurement_noise_std=cfg.measurement_noise_std,
    )


# ============================================================
# 2) Utilidades sinteticas
# ============================================================


def log(msg: str, cfg: LFISTAConfig) -> None:
    if cfg.log_progress:
        print(msg, flush=True)
    if cfg.artifact_log_path:
        with open(cfg.artifact_log_path, "a", encoding="utf-8") as fp:
            fp.write(msg + "\n")


def layout_lfista_run(cfg: LFISTAConfig, run_id: str) -> None:
    base_save = cfg.save_dir
    cfg.save_dir = os.path.join(base_save, "runs", run_id)
    cfg.run_artifact_id = run_id
    cwd = os.getcwd()
    base_name = os.path.basename(os.path.normpath(base_save))
    fig_abs = os.path.abspath(os.path.join(cwd, "paper", "figures", base_name, "runs", run_id))
    save_abs = os.path.abspath(os.path.join(cwd, cfg.save_dir))
    cfg.plots_subdir = os.path.relpath(fig_abs, save_abs)
    os.makedirs(fig_abs, exist_ok=True)
    os.makedirs(save_abs, exist_ok=True)

    latest = os.path.join(cwd, base_save, "LATEST")
    try:
        os.remove(latest)
    except FileNotFoundError:
        pass
    os.symlink(os.path.join("runs", run_id), latest, target_is_directory=False)

    readme = os.path.join(save_abs, "README_RUN.txt")
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    lines = [
        "SIR-CS LFISTA lab (Etapa 2 prototype)",
        "run_id: " + run_id,
        "profile: " + cfg.profile,
        "started_local: " + started,
        "cwd: " + cwd,
        "argv: " + json.dumps(sys.argv),
        "",
        "Artifacts in this folder:",
        "  config.json, PROTOCOL.txt, detailed_results.csv, summary_by_seed.csv, summary.csv",
        "  run_console.log",
        "",
        "Figures (repo): paper/figures/" + base_name + "/runs/" + run_id + "/",
        "",
        "Symlink: " + base_save + "/LATEST -> runs/" + run_id,
        "",
    ]
    with open(readme, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


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
        "X_train": X[: cfg.n_train],
        "X_val": X[cfg.n_train : cfg.n_train + cfg.n_val],
        "X_test": X[cfg.n_train + cfg.n_val :],
        "Y_train": y[: cfg.n_train],
        "Y_val": y[cfg.n_train : cfg.n_train + cfg.n_val],
        "Y_test": y[cfg.n_train + cfg.n_val :],
        "Ybg_train": y_bg[: cfg.n_train],
        "Ybg_val": y_bg[cfg.n_train : cfg.n_train + cfg.n_val],
        "Ybg_test": y_bg[cfg.n_train + cfg.n_val :],
        "Alpha_train": Alpha[: cfg.n_train],
        "Alpha_val": Alpha[cfg.n_train : cfg.n_train + cfg.n_val],
        "Alpha_test": Alpha[cfg.n_train + cfg.n_val :],
        "Psi": Psi,
    }


# ============================================================
# 3) Figuras
# ============================================================


def plots_dir(cfg: LFISTAConfig) -> str:
    p = os.path.join(cfg.save_dir, cfg.plots_subdir)
    os.makedirs(p, exist_ok=True)
    return p


def plot_gain_over_ml(summary: pd.DataFrame, metric: str, save_path: str) -> None:
    col_mean = f"{metric}_mean"
    ml = summary[summary["method"] == "ml_only_torch"][["measurement_ratio", col_mean]].rename(
        columns={col_mean: "ml_val"}
    )
    colors = {"hybrid_lfista_frozen": "#ff7f0e", "hybrid_lfista_joint": "#2ca02c"}
    plt.figure(figsize=(8.5, 5))
    for method in ["hybrid_lfista_frozen", "hybrid_lfista_joint"]:
        sdf = summary[summary["method"] == method].merge(ml, on="measurement_ratio", how="inner")
        if len(sdf) == 0:
            continue
        gain = sdf["ml_val"].values - sdf[col_mean].values
        plt.plot(
            sdf["measurement_ratio"].values,
            gain,
            marker="o",
            label=method,
            color=colors.get(method),
        )
    plt.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("Measurement ratio m / N")
    plt.ylabel("ml_only - method (" + metric + ")")
    plt.title("Gain over ml_only_torch (" + metric + ", higher is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


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


def save_lfista_plots(cfg: LFISTAConfig, summary: pd.DataFrame) -> List[str]:
    pdir = plots_dir(cfg)
    out: List[str] = []
    for fname, metric in [
        ("01_rmse_vs_measurement_ratio.png", "rmse"),
        ("02_mae_vs_measurement_ratio.png", "mae"),
        ("03_relative_l2_vs_measurement_ratio.png", "relative_l2"),
    ]:
        path = os.path.join(pdir, fname)
        plot_metric(summary, metric, path)
        out.append(path)
    g1 = os.path.join(pdir, "04_gain_rmse_over_ml_only_torch.png")
    plot_gain_over_ml(summary, "rmse", g1)
    out.append(g1)
    g2 = os.path.join(pdir, "05_gain_mae_over_ml_only_torch.png")
    plot_gain_over_ml(summary, "mae", g2)
    out.append(g2)
    return out


# ============================================================
# 4) Execucao por (seed, rho)
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
    def _log(msg: str) -> None:
        log(msg, cfg)

    rng = np.random.default_rng(seed)
    data = make_dataset(cfg, seed)
    m = max(4, int(round(measurement_ratio * cfg.n_output)))
    M_np = build_measurement_matrix(m, cfg.n_output, cfg.measurement_kind, rng)
    tc = lfista_config_to_train(cfg)
    df, _ = run_lfista_experiment_dataframe(tc, seed, measurement_ratio, data, M_np, _log)
    return df


# ============================================================
# 5) main
# ============================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SIR-CS LFISTA minimal baseline")
    ap.add_argument("--profile", type=str, default="phase2_lfista", choices=["phase2_lfista", "explore"])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    cfg = LFISTAConfig(profile=args.profile)
    apply_profile(cfg)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    layout_lfista_run(cfg, run_id)
    cfg.artifact_log_path = os.path.join(cfg.save_dir, "run_console.log")

    with open(os.path.join(cfg.save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    base_out = os.path.dirname(os.path.dirname(cfg.save_dir))
    protocol_lines = [
        "LFISTA lab protocol (roadmap Etapa 2 prototype, sir_cs_lfista.py).",
        "PyTorch background MLP + LFISTAUnrolled; frozen then joint training.",
        "",
        "run_id: " + run_id,
        "profile: " + cfg.profile,
        "save_dir (relative): " + cfg.save_dir,
        "plots_subdir (relative to save_dir): " + cfg.plots_subdir,
        "",
        "seeds: " + str(cfg.seeds),
        "measurement_ratios: " + str(cfg.measurement_ratios),
        "lfista_steps (K): " + str(cfg.lfista_steps),
        "device: " + str(cfg.device),
        "",
        "Figures: paper/figures/" + os.path.basename(base_out) + "/runs/" + run_id + "/",
        "Symlink: " + base_out + "/LATEST -> runs/" + run_id,
    ]
    with open(os.path.join(cfg.save_dir, "PROTOCOL.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(protocol_lines) + "\n")

    log(
        f"=== LFISTA start | profile={cfg.profile} | run_id={run_id} | seeds={cfg.seeds} | "
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

    plot_paths = save_lfista_plots(cfg, summary)

    elapsed = time.time() - t0

    readme_p = os.path.join(cfg.save_dir, "README_RUN.txt")
    finished = time.strftime("%Y-%m-%dT%H:%M:%S")
    extra = [
        "",
        "finished_local: " + finished,
        "elapsed_seconds: {:.1f}".format(elapsed),
        "",
        "Plot files:",
    ]
    with open(readme_p, "a", encoding="utf-8") as f:
        f.write("\n".join(extra) + "\n")
        for pp in sorted(plot_paths):
            f.write("  " + os.path.basename(pp) + "\n")

    print("\n" + "=" * 72)
    print("RESUMO LFISTA")
    print("=" * 72)
    print(summary.round(4).to_string(index=False))
    out_abs = os.path.abspath(cfg.save_dir)
    base_name = os.path.basename(base_out)
    print(f"\nArquivos salvos em: {out_abs}")
    print("  detailed_results.csv, summary_by_seed.csv, summary.csv, config.json, PROTOCOL.txt, run_console.log")
    print(f"Figuras em: paper/figures/{base_name}/runs/{run_id}/")
    for pp in sorted(plot_paths):
        print(f"  - {os.path.basename(pp)}")
    print(f"Tempo total: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
