#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Well 861 ML diagnostics for target and feature-set decisions.

Produces organized artifacts under:
  methods_comparison/data/processed/ml_runs/diagnostics_861/

Sections:
  correlations/     feature-target Pearson r
  cv_scenarios/     oracle vs logs vs mean baseline (depth-block CV)
  stratified/       OOF metrics by HFU and Lithotype
  depth_blocks/     geology summary per CV block
  decision/         JSON + markdown recommendation
  figures/          diagnostic plots

Optionally runs missing per-target baselines (k_lab, Phi_lab) via subprocess.

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    DEPTH_COL,
    DEFAULT_ENRICHED,
    LOG_FEATURE_COLUMNS,
    ORACLE_LAB_FEATURES,
    TARGET_COLUMNS,
    XYBundle,
    build_xy,
    build_xy_from_columns,
    compare_out_dir_for_target,
    depth_block_splits,
    iter_fold_arrays,
    load_logs_enriched,
    target_slug,
)
from ml_861_metrics import evaluate_depth_blocks

ROOT = SCRIPT_DIR.parents[1]
ML_RUNS = ROOT / "methods_comparison" / "data" / "processed" / "ml_runs"
DEFAULT_DIAG_OUT = ML_RUNS / "diagnostics_861"
COMPARE_ROOT = ML_RUNS / "compare_861"

PRIMARY_TARGETS: Tuple[str, ...] = ("FZI_lab", "Phi_lab (pu)", "k_lab (mD)")
LAB_CORR_COLUMNS: Tuple[str, ...] = (
    "RQI",
    "FZI_lab",
    "Phi_lab (pu)",
    "k_lab (mD)",
    "HFU",
)
STRAT_COLUMNS: Tuple[str, ...] = ("HFU", "Lithotype")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _make_rf(n_estimators: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def collect_oof_predictions(
    bundle: XYBundle,
    model_factory: Callable[[], Any],
    n_blocks: int = 3,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Out-of-fold predictions aligned to bundle row order."""
    folds_def = depth_block_splits(bundle.df, n_blocks=n_blocks)
    n = len(bundle.y)
    oof = np.full(n, np.nan, dtype=np.float64)
    fold_rows: List[Dict[str, Any]] = []

    for fold_id, x_tr, y_tr, x_te, y_te in iter_fold_arrays(bundle, folds_def):
        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        test_idx = folds_def[fold_id].test_idx
        oof[test_idx] = pred
        fold_rows.append(
            {
                "fold_id": fold_id,
                "rmse": _rmse(y_te, pred),
                "r2": float(r2_score(y_te, pred)),
                "depth_min_m": folds_def[fold_id].depth_min_m,
                "depth_max_m": folds_def[fold_id].depth_max_m,
            }
        )
    return oof, fold_rows


def stratified_oof_metrics(
    bundle: XYBundle,
    oof_pred: np.ndarray,
    strat_col: str,
) -> pd.DataFrame:
    """RMSE/R2/MAE per stratification column from OOF predictions."""
    if strat_col not in bundle.df.columns:
        raise ValueError("Missing strat column: {}".format(strat_col))

    rows: List[Dict[str, Any]] = []
    df = bundle.df
    mask = np.isfinite(oof_pred)
    for level, grp in df.loc[mask].groupby(strat_col):
        idx = grp.index.to_numpy()
        y_true = bundle.y[idx]
        y_pred = oof_pred[idx]
        if len(y_true) < 2:
            r2 = float("nan")
        else:
            r2 = float(r2_score(y_true, y_pred))
        rows.append(
            {
                "target": bundle.target,
                "strat_column": strat_col,
                "strat_value": level,
                "n_samples": int(len(y_true)),
                "rmse": _rmse(y_true, y_pred),
                "mae": float(mean_absolute_error(y_true, y_pred)),
                "r2": r2,
                "y_mean": float(np.mean(y_true)),
                "y_std": float(np.std(y_true)),
            }
        )
    return pd.DataFrame(rows).sort_values(["strat_column", "strat_value"])


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    """Pearson r between log/lab columns and primary targets."""
    feature_cols = [c for c in LOG_FEATURE_COLUMNS if c in df.columns]
    lab_cols = [c for c in LAB_CORR_COLUMNS if c in df.columns]
    all_cols = feature_cols + lab_cols
    target_cols = [t for t in PRIMARY_TARGETS if t in df.columns]

    rows: List[Dict[str, Any]] = []
    for feat in all_cols:
        for tgt in target_cols:
            if feat == tgt:
                continue
            sub = df[[feat, tgt]].dropna()
            if len(sub) < 3:
                r = float("nan")
            else:
                r = float(sub[feat].astype(float).corr(sub[tgt].astype(float)))
            rows.append(
                {
                    "feature": feat,
                    "target": tgt,
                    "pearson_r": r,
                    "abs_r": abs(r) if np.isfinite(r) else float("nan"),
                    "n_pairs": int(len(sub)),
                }
            )
    out = pd.DataFrame(rows)
    return out.sort_values(["target", "abs_r"], ascending=[True, False])


def depth_block_summary(df: pd.DataFrame, n_blocks: int = 3) -> pd.DataFrame:
    """Descriptive stats per depth-block fold."""
    folds = depth_block_splits(df, n_blocks=n_blocks)
    rows: List[Dict[str, Any]] = []

    for fold in folds:
        block = df.iloc[fold.test_idx]
        row: Dict[str, Any] = {
            "fold_id": fold.fold_id,
            "depth_min_m": fold.depth_min_m,
            "depth_max_m": fold.depth_max_m,
            "n_rows": int(len(block)),
        }
        for tgt in PRIMARY_TARGETS:
            if tgt in block.columns:
                row["{}_mean".format(target_slug(tgt))] = float(block[tgt].mean())
                row["{}_std".format(target_slug(tgt))] = float(block[tgt].std())
        if "HFU" in block.columns:
            for hfu, cnt in block["HFU"].value_counts().sort_index().items():
                row["HFU{}_count".format(hfu)] = int(cnt)
        if "Lithotype" in block.columns:
            for lith, cnt in block["Lithotype"].value_counts().sort_index().items():
                row["Lithotype{}_count".format(lith)] = int(cnt)
        rows.append(row)
    return pd.DataFrame(rows)


def cv_scenario_rows(
    df: pd.DataFrame,
    target: str,
    feature_set: str,
    feature_columns: Sequence[str],
    n_estimators: int,
    n_blocks: int,
    random_state: int,
) -> List[Dict[str, Any]]:
    """One row per fold for a CV scenario."""
    bundle = build_xy_from_columns(df, target=target, feature_columns=feature_columns)
    factory = lambda: _make_rf(n_estimators, random_state)
    cv = evaluate_depth_blocks(factory, bundle, n_blocks=n_blocks)
    rows: List[Dict[str, Any]] = []
    for fold in cv.folds:
        rows.append(
            {
                "target": target,
                "feature_set": feature_set,
                "n_features": len(feature_columns),
                "fold_id": fold.fold_id,
                "rmse": fold.rmse,
                "r2": fold.r2,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "depth_min_m": fold.extra.get("depth_min_m"),
                "depth_max_m": fold.extra.get("depth_max_m"),
                "mean_rmse": cv.mean_rmse,
                "mean_r2": cv.mean_r2,
            }
        )
    return rows


def holdout_legacy_row(
    bundle: XYBundle,
    n_estimators: int,
    random_state: int,
) -> Dict[str, Any]:
    """Random 80/20 holdout (legacy comparison only)."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        bundle.X, bundle.y, test_size=0.2, random_state=random_state
    )
    model = _make_rf(n_estimators, random_state)
    model.fit(x_tr, y_tr)
    pred = model.predict(x_te)
    return {
        "target": bundle.target,
        "feature_set": "wireline_only",
        "protocol": "holdout_80_20",
        "rmse": _rmse(y_te, pred),
        "r2": float(r2_score(y_te, pred)),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "note": "legacy comparison only; prefer depth_block_cv",
    }


def run_missing_baselines(
    targets: Sequence[str],
    smoke: bool,
    random_state: int,
) -> None:
    """Invoke run_861_ml_baseline.py for targets without saved metrics."""
    py = sys.executable
    for target in targets:
        out_dir = compare_out_dir_for_target(target, base=COMPARE_ROOT)
        metrics_path = out_dir / "metrics.json"
        if metrics_path.is_file():
            continue
        cmd = [
            py,
            str(SCRIPT_DIR / "run_861_ml_baseline.py"),
            "--target",
            target,
            "--random-state",
            str(random_state),
        ]
        if smoke:
            cmd.append("--smoke")
        print("RUN baseline:", " ".join(cmd))
        subprocess.run(cmd, check=True)


def load_baseline_summaries() -> pd.DataFrame:
    """Load all per-target baseline CSVs under compare_861/by_target/."""
    rows: List[Dict[str, Any]] = []
    by_target = COMPARE_ROOT / "by_target"
    if not by_target.is_dir():
        return pd.DataFrame(rows)

    for sub in sorted(by_target.iterdir()):
        if not sub.is_dir():
            continue
        csv_path = sub / "861_ml_baseline_summary.csv"
        if csv_path.is_file():
            part = pd.read_csv(csv_path)
            rows.extend(part.to_dict(orient="records"))
    return pd.DataFrame(rows)


def build_decision_summary(
    cv_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Rank targets and emit recommendation flags."""
    logs_summary = (
        cv_df[cv_df["feature_set"] == "wireline_only"]
        .groupby("target", as_index=False)
        .agg(mean_r2=("mean_r2", "first"), mean_rmse=("mean_rmse", "first"))
        .sort_values("mean_r2", ascending=False)
    )

    oracle_summary = (
        cv_df[cv_df["feature_set"] == "oracle_phi_k_lab"]
        .groupby("target", as_index=False)
        .agg(oracle_mean_r2=("mean_r2", "first"))
    )

    merged = logs_summary.merge(oracle_summary, on="target", how="left")

    best_rf = pd.DataFrame()
    if not baseline_df.empty and "regressor" in baseline_df.columns:
        rf_rows = baseline_df[baseline_df["regressor"] == "rf"].copy()
        if not rf_rows.empty:
            best_rf = rf_rows[["target", "mean_r2", "mean_rmse", "protocol"]]

    fzi_corr = corr_df[(corr_df["target"] == "FZI_lab") & (corr_df["feature"].isin(LOG_FEATURE_COLUMNS))]
    best_log_r = float(fzi_corr["abs_r"].max()) if not fzi_corr.empty else float("nan")

    recommendations: List[str] = []
    fzi_row = merged[merged["target"] == "FZI_lab"]
    if not fzi_row.empty and float(fzi_row.iloc[0]["mean_r2"]) < 0:
        recommendations.append(
            "Do not use FZI_lab + wireline_only as primary well_profile deliverable (depth-block R2 < 0)."
        )

    positive = merged[merged["mean_r2"] > 0]
    if not positive.empty:
        best = positive.iloc[0]
        recommendations.append(
            "Prefer target {} for well_profile (wireline_only depth-block mean R2={:.3f}).".format(
                best["target"], best["mean_r2"]
            )
        )
    else:
        recommendations.append(
            "No target shows positive depth-block R2 with wireline_only; consider ct_plugs or HFU classification."
        )

    if np.isfinite(best_log_r) and best_log_r < 0.35:
        recommendations.append(
            "Weak log-FZI correlation (best |r|={:.2f}); FZI poor predictability from wireline is expected.".format(
                best_log_r
            )
        )

    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "target_ranking_wireline_only": merged.to_dict(orient="records"),
        "rf_baseline_by_target": best_rf.to_dict(orient="records") if not best_rf.empty else [],
        "best_log_fzi_abs_correlation": best_log_r,
        "recommendations": recommendations,
    }


def write_manifest(out_dir: Path, files: Sequence[str]) -> None:
    """List generated artifact paths."""
    lines = [
        "Well 861 ML diagnostics manifest",
        "Generated: {}".format(datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")),
        "Script: diagnose_861_ml.py",
        "",
        "Artifacts:",
    ]
    lines.extend(["  {}".format(f) for f in sorted(files)])
    (out_dir / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_decision_markdown(out_dir: Path, decision: Dict[str, Any]) -> None:
    """Human-readable decision summary."""
    lines = [
        "# Well 861 ML -- decision summary",
        "",
        "Generated: {}".format(decision["generated_utc"]),
        "",
        "## Target ranking (logs only, RF depth-block CV)",
        "",
    ]
    for row in decision["target_ranking_wireline_only"]:
        lines.append(
            "- **{}**: mean R2={:.3f}, mean RMSE={:.3f}, oracle R2={}".format(
                row["target"],
                row["mean_r2"],
                row["mean_rmse"],
                row.get("oracle_mean_r2", "n/a"),
            )
        )

    lines.extend(["", "## Recommendations", ""])
    for rec in decision["recommendations"]:
        lines.append("- {}".format(rec))

    lines.extend(
        [
            "",
            "## Next steps",
            "",
            "- Report stratified metrics (HFU, Lithotype) before heavy hyperparameter tuning.",
            "- If well_profile target remains weak, run ct_plugs (10 CT samples, leave-one-plug-out).",
            "- Keep depth-block CV as primary protocol; holdout 80/20 is legacy only.",
            "",
        ]
    )
    (out_dir / "TARGET_DECISION_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def plot_feature_target_corr(corr_df: pd.DataFrame, fig_dir: Path) -> None:
    """Bar chart of log feature correlations with FZI_lab."""
    sub = corr_df[
        (corr_df["target"] == "FZI_lab")
        & (corr_df["feature"].isin(LOG_FEATURE_COLUMNS))
    ].sort_values("pearson_r")
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:red" if r < 0 else "tab:blue" for r in sub["pearson_r"]]
    ax.barh(sub["feature"], sub["pearson_r"], color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Pearson r with FZI_lab")
    ax.set_title("Well 861 -- log features vs FZI_lab")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "log_features_vs_FZI_lab_corr.png", dpi=150)
    plt.close(fig)


def plot_lab_properties_vs_depth(df: pd.DataFrame, fig_dir: Path) -> None:
    """Lab phi, permeability and FZI vs depth (87 MOGNO rows, beamer slide)."""
    required = (DEPTH_COL, "Phi_lab (pu)", "k_lab (mD)", "FZI_lab")
    for col in required:
        if col not in df.columns:
            return

    work = df.sort_values(DEPTH_COL).copy()
    depth = work[DEPTH_COL].to_numpy(dtype=np.float64)
    phi = work["Phi_lab (pu)"].to_numpy(dtype=np.float64)
    k_md = np.maximum(work["k_lab (mD)"].to_numpy(dtype=np.float64), 0.1)
    fzi = work["FZI_lab"].to_numpy(dtype=np.float64)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 4.0), sharey=True)
    panels = (
        (axes[0], phi, "Phi_lab (pu)", "#004a7f"),
        (axes[1], k_md, "k_lab (mD)", "#008060"),
        (axes[2], fzi, "FZI_lab", "#c45c00"),
    )
    for ax, values, xlabel, color in panels:
        ax.plot(values, depth, "o", markersize=3.5, color=color, alpha=0.85)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        if xlabel.startswith("k_"):
            ax.set_xscale("log")

    depth_min = float(depth.min())
    depth_max = float(depth.max())
    axes[0].set_ylabel("Depth (m)", fontsize=9)
    fig.suptitle(
        "Well 861 MOGNO -- lab properties ({}--{} m, n={})".format(
            depth_min, depth_max, len(work)
        ),
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "mogno_lab_properties_vs_depth.png", dpi=160)
    plt.close(fig)


def plot_oracle_vs_logs(cv_df: pd.DataFrame, fig_dir: Path) -> None:
    """Grouped bar: mean R2 by fold for FZI oracle vs logs."""
    sub = cv_df[cv_df["target"] == "FZI_lab"].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="fold_id",
        columns="feature_set",
        values="r2",
        aggfunc="first",
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(pivot))
    width = 0.35
    sets = list(pivot.columns)
    for i, fs in enumerate(sets):
        offset = (i - (len(sets) - 1) / 2.0) * width
        ax.bar(x + offset, pivot[fs], width, label=fs)

    ax.set_xticks(x)
    ax.set_xticklabels(["block {}".format(i) for i in pivot.index])
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("R2")
    ax.set_title("FZI_lab depth-block CV -- oracle vs logs (RF)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "FZI_oracle_vs_logs_r2_by_fold.png", dpi=150)
    plt.close(fig)


def plot_target_comparison(cv_df: pd.DataFrame, fig_dir: Path) -> None:
    """Mean R2 across targets for logs vs mean baseline."""
    sub = cv_df[cv_df["feature_set"].isin(["wireline_only", "mean_train_baseline"])]
    if sub.empty:
        return

    agg = sub.groupby(["target", "feature_set"], as_index=False)["mean_r2"].first()
    targets = list(PRIMARY_TARGETS)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(targets))
    width = 0.35
    for i, fs in enumerate(["wireline_only", "mean_train_baseline"]):
        vals = []
        for tgt in targets:
            row = agg[(agg["target"] == tgt) & (agg["feature_set"] == fs)]
            vals.append(float(row["mean_r2"].iloc[0]) if not row.empty else float("nan"))
        offset = (i - 0.5) * width
        ax.bar(x + offset, vals, width, label=fs)

    ax.set_xticks(x)
    ax.set_xticklabels(targets, rotation=15, ha="right")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_ylabel("mean R2 (depth-block CV)")
    ax.set_title("Well 861 -- target comparison (RF vs mean baseline)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "target_comparison_mean_r2.png", dpi=150)
    plt.close(fig)


def run_diagnostics(
    data_path: Path,
    out_dir: Path,
    smoke: bool = False,
    run_baselines: bool = True,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Main diagnostic pipeline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    corr_dir = out_dir / "correlations"
    cv_dir = out_dir / "cv_scenarios"
    strat_dir = out_dir / "stratified"
    block_dir = out_dir / "depth_blocks"
    decision_dir = out_dir / "decision"
    fig_dir = out_dir / "figures"
    for d in (corr_dir, cv_dir, strat_dir, block_dir, decision_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_estimators = 10 if smoke else 200
    n_blocks = 2 if smoke else 3

    if run_baselines:
        run_missing_baselines(PRIMARY_TARGETS, smoke=smoke, random_state=random_state)

    df = load_logs_enriched(data_path)

    corr_df = correlation_table(df)
    corr_path = corr_dir / "feature_target_pearson.csv"
    corr_df.to_csv(corr_path, index=False)

    block_df = depth_block_summary(df, n_blocks=n_blocks)
    block_path = block_dir / "depth_block_geology_summary.csv"
    block_df.to_csv(block_path, index=False)

    cv_rows: List[Dict[str, Any]] = []
    holdout_rows: List[Dict[str, Any]] = []
    strat_frames: List[pd.DataFrame] = []

    log_features = list(LOG_FEATURE_COLUMNS)
    oracle_features = list(ORACLE_LAB_FEATURES)

    for target in PRIMARY_TARGETS:
        if target not in df.columns:
            continue

        cv_rows.extend(
            cv_scenario_rows(
                df,
                target=target,
                feature_set="wireline_only",
                feature_columns=log_features,
                n_estimators=n_estimators,
                n_blocks=n_blocks,
                random_state=random_state,
            )
        )

        if target == "FZI_lab":
            cv_rows.extend(
                cv_scenario_rows(
                    df,
                    target=target,
                    feature_set="oracle_phi_k_lab",
                    feature_columns=oracle_features,
                    n_estimators=n_estimators,
                    n_blocks=n_blocks,
                    random_state=random_state,
                )
            )

        bundle_logs = build_xy(df, target=target, feature_mode="log_only")

        folds_def = depth_block_splits(bundle_logs.df, n_blocks=n_blocks)
        mean_fold_r2: List[float] = []
        mean_fold_rmse: List[float] = []
        for fold_id, _x_tr, y_tr, _x_te, y_te in iter_fold_arrays(bundle_logs, folds_def):
            mu = float(np.mean(y_tr))
            pred = np.full_like(y_te, mu)
            mean_fold_r2.append(float(r2_score(y_te, pred)))
            mean_fold_rmse.append(_rmse(y_te, pred))
            cv_rows.append(
                {
                    "target": target,
                    "feature_set": "mean_train_baseline",
                    "n_features": 0,
                    "fold_id": fold_id,
                    "rmse": mean_fold_rmse[-1],
                    "r2": mean_fold_r2[-1],
                    "n_train": len(y_tr),
                    "n_test": len(y_te),
                    "depth_min_m": folds_def[fold_id].depth_min_m,
                    "depth_max_m": folds_def[fold_id].depth_max_m,
                    "mean_rmse": float(np.mean(mean_fold_rmse)),
                    "mean_r2": float(np.mean(mean_fold_r2)),
                }
            )

        holdout_rows.append(holdout_legacy_row(bundle_logs, n_estimators, random_state))

        oof, _fold_info = collect_oof_predictions(
            bundle_logs,
            lambda: _make_rf(n_estimators, random_state),
            n_blocks=n_blocks,
        )
        for strat_col in STRAT_COLUMNS:
            if strat_col in bundle_logs.df.columns:
                strat_frames.append(
                    stratified_oof_metrics(bundle_logs, oof, strat_col=strat_col)
                )

    cv_df = pd.DataFrame(cv_rows)
    cv_path = cv_dir / "depth_block_scenarios.csv"
    cv_df.to_csv(cv_path, index=False)

    holdout_df = pd.DataFrame(holdout_rows)
    holdout_path = cv_dir / "holdout_80_20_legacy.csv"
    holdout_df.to_csv(holdout_path, index=False)

    if strat_frames:
        strat_df = pd.concat(strat_frames, ignore_index=True)
    else:
        strat_df = pd.DataFrame()
    strat_path = strat_dir / "oof_metrics_by_hfu_lithotype.csv"
    strat_df.to_csv(strat_path, index=False)

    baseline_df = load_baseline_summaries()
    combined_path = COMPARE_ROOT / "861_all_targets_summary.csv"
    if not baseline_df.empty:
        baseline_df.to_csv(combined_path, index=False)

    decision = build_decision_summary(cv_df, baseline_df, corr_df)
    decision_json = decision_dir / "target_recommendation.json"
    decision_json.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    write_decision_markdown(decision_dir, decision)

    plot_feature_target_corr(corr_df, fig_dir)
    plot_lab_properties_vs_depth(df, fig_dir)
    plot_oracle_vs_logs(cv_df, fig_dir)
    plot_target_comparison(cv_df, fig_dir)

    artifact_files = [
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*")
        if p.is_file() and p.name != "MANIFEST.txt"
    ]
    if combined_path.is_file():
        artifact_files.append("../compare_861/861_all_targets_summary.csv")
    write_manifest(out_dir, artifact_files)

    return {
        "out_dir": str(out_dir),
        "decision": decision,
        "n_cv_rows": len(cv_df),
        "combined_baseline_csv": str(combined_path) if combined_path.is_file() else None,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Well 861 ML diagnostics and target decision support")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_DIAG_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Do not run missing per-target baseline scripts",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    result = run_diagnostics(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        smoke=args.smoke,
        run_baselines=not args.skip_baselines,
        random_state=args.random_state,
    )
    print("OK diagnostics out_dir={}".format(result["out_dir"]))
    for rec in result["decision"]["recommendations"]:
        print("  - {}".format(rec))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
