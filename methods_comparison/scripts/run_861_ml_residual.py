#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3: ML residual on Vp after Gassmann physics (Well 861).

Target: vp_residual_km_s = vp_sonic - vp_gassmann
Hybrid: Vp_hybrid = Vp_gassmann + predicted_residual

Planning: methods_comparison/planning/etapa3_ml_residual_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_861_ml_baseline import model_factory as baseline_model_factory
from ml_861_data import (
    DEPTH_COL,
    DLIS_GASSMANN_VALIDATION_CSV,
    ML_RESIDUAL_VP_ROOT,
    RESIDUAL_VP_TARGET,
    XYBundle,
    build_residual_feature_columns,
    build_xy_from_columns,
    depth_block_splits,
    iter_fold_arrays,
    load_logs_enriched,
    residual_vp_rf_dir,
)
from ml_861_metrics import CvSummary, FoldMetrics, _rmse

OUT_TABLES = ML_RESIDUAL_VP_ROOT / "tables"
OUT_FIGURES = ML_RESIDUAL_VP_ROOT / "figures"
HFU_COL = "HFU"

# Etapa 1 five regressors + Ridge (Etapa 1d alternative).
RESIDUAL_REGRESSORS: Tuple[str, ...] = ("rf", "gb", "xgb", "mlp", "lr", "ridge")

REGRESSOR_DISPLAY: Dict[str, str] = {
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "xgb": "XGBoost",
    "mlp": "MLP",
    "lr": "Linear Regression",
    "ridge": "Ridge",
}


def residual_regressor_factory(
    name: str,
    smoke: bool = False,
    random_state: int = 42,
) -> Callable[[], Any]:
    """Build estimator factory for residual-target comparison."""
    if name == "ridge":
        return lambda: Ridge(alpha=1.0, random_state=random_state)
    return baseline_model_factory(
        name,
        smoke=smoke,
        random_state=random_state,
        small_sample=False,
    )


@dataclass(frozen=True)
class VpValidationMetrics:
    """Vp validation vs sonic (km/s)."""

    n: int
    mape_pct: float
    rmse_km_s: float
    bias_km_s: float
    mean_vp_pred_km_s: float
    mean_vp_sonic_km_s: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "n": float(self.n),
            "mape_vp_pct": self.mape_pct,
            "rmse_vp_km_s": self.rmse_km_s,
            "bias_vp_km_s": self.bias_km_s,
            "mean_vp_pred_km_s": self.mean_vp_pred_km_s,
            "mean_vp_sonic_km_s": self.mean_vp_sonic_km_s,
        }


def utc_now_iso() -> str:
    """UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build_residual_dataset(validation_csv: Path) -> pd.DataFrame:
    """Merge enriched logs with Gassmann vs sonic validation table."""
    if not validation_csv.is_file():
        raise FileNotFoundError(
            "Gassmann validation missing: {}. Run run_861_dem_sc_gassmann.py.".format(
                validation_csv
            )
        )
    logs = load_logs_enriched()
    val = pd.read_csv(validation_csv)
    ok = val[val["has_sonic_vp"] == True].copy()  # noqa: E712
    if ok.empty:
        raise RuntimeError("No sonic-matched rows in validation CSV")

    vp_col = "vp_gassmann_km_s" if "vp_gassmann_km_s" in ok.columns else "vp_dem_km_s"
    pick_cols = [
        "Depth(m)",
        vp_col,
        "vp_sonic_km_s",
        "vp_bias_km_s",
        "phi_input",
    ]
    merged = logs.merge(ok[pick_cols], on="Depth(m)", how="inner")
    if vp_col == "vp_dem_km_s":
        merged = merged.rename(columns={"vp_dem_km_s": "vp_gassmann_km_s"})

    merged["vp_residual_km_s"] = merged["vp_sonic_km_s"] - merged["vp_gassmann_km_s"]
    merged = merged.sort_values(DEPTH_COL).reset_index(drop=True)
    if len(merged) != 87:
        raise RuntimeError("Expected 87 merged rows, got {}".format(len(merged)))
    return merged


def vp_metrics_vs_sonic(vp_pred: np.ndarray, vp_sonic: np.ndarray) -> VpValidationMetrics:
    """MAPE, RMSE, bias for Vp predictions vs sonic."""
    err = vp_pred - vp_sonic
    mape = float(np.mean(np.abs(err / vp_sonic)) * 100.0)
    return VpValidationMetrics(
        n=len(vp_sonic),
        mape_pct=mape,
        rmse_km_s=float(np.sqrt(np.mean(err ** 2))),
        bias_km_s=float(np.mean(err)),
        mean_vp_pred_km_s=float(np.mean(vp_pred)),
        mean_vp_sonic_km_s=float(np.mean(vp_sonic)),
    )


def evaluate_depth_blocks_oof(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
    n_blocks: int = 3,
) -> Tuple[CvSummary, np.ndarray]:
    """Depth-block CV with out-of-fold residual predictions."""
    folds_def = depth_block_splits(bundle.df, n_blocks=n_blocks)
    fold_metrics: List[FoldMetrics] = []
    oof_pred = np.full(len(bundle.y), np.nan, dtype=np.float64)

    for fold_id, x_tr, y_tr, x_te, y_te in iter_fold_arrays(bundle, folds_def):
        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        test_idx = folds_def[fold_id].test_idx
        oof_pred[test_idx] = pred
        fold_metrics.append(
            FoldMetrics(
                fold_id=fold_id,
                rmse=_rmse(y_te, pred),
                r2=float(r2_score(y_te, pred)),
                n_train=len(y_tr),
                n_test=len(y_te),
                extra={
                    "depth_min_m": folds_def[fold_id].depth_min_m,
                    "depth_max_m": folds_def[fold_id].depth_max_m,
                },
            )
        )

    rmse_vals = [f.rmse for f in fold_metrics]
    r2_vals = [f.r2 for f in fold_metrics]
    valid_mask = ~np.isnan(oof_pred)
    valid_y = bundle.y[valid_mask]
    valid_oof = oof_pred[valid_mask]
    summary = CvSummary(
        target=bundle.target,
        model_name="",
        folds=fold_metrics,
        mean_rmse=float(np.mean(rmse_vals)),
        std_rmse=float(np.std(rmse_vals)),
        mean_r2=float(np.mean(r2_vals)),
        std_r2=float(np.std(r2_vals)),
        protocol="depth_block_{}".format(n_blocks),
        global_oof_rmse=_rmse(valid_y, valid_oof),
        global_oof_r2=float(r2_score(valid_y, valid_oof)) if len(valid_y) > 1 else float("nan"),
        global_oof_mae=float(np.mean(np.abs(valid_y - valid_oof))),
    )
    return summary, oof_pred


def summary_by_hfu_vp(df: pd.DataFrame, vp_col: str) -> pd.DataFrame:
    """HFU-level Vp metrics vs sonic."""
    rows: List[dict] = []
    for hfu in sorted(df[HFU_COL].unique()):
        sub = df[df[HFU_COL] == hfu]
        m = vp_metrics_vs_sonic(
            sub[vp_col].to_numpy(dtype=np.float64),
            sub["vp_sonic_km_s"].to_numpy(dtype=np.float64),
        )
        rows.append(
            {
                "HFU": int(hfu),
                "n": int(m.n),
                "mape_vp_pct": m.mape_pct,
                "bias_vp_km_s": m.bias_km_s,
                "rmse_vp_km_s": m.rmse_km_s,
                "mean_vp_pred_km_s": m.mean_vp_pred_km_s,
                "mean_vp_sonic_km_s": m.mean_vp_sonic_km_s,
            }
        )
    return pd.DataFrame(rows)


def plot_residual_oof_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Scatter observed vs predicted residual."""
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    ax.scatter(y_true, y_pred, c="#4c72b0", edgecolors="k", linewidths=0.3, alpha=0.85)
    lim = max(float(np.max(np.abs(y_true))), float(np.max(np.abs(y_pred))), 0.01) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Residual observed (km/s)")
    ax.set_ylabel("Residual predicted OOF (km/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.axvline(0.0, color="gray", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


REGRESSOR_LINE_COLORS: Dict[str, str] = {
    "rf": "#d62728",
    "ridge": "#9467bd",
    "mlp": "#ff7f0e",
    "xgb": "#8c564b",
    "gb": "#e377c2",
    "lr": "#7f7f7f",
}


def plot_regressor_depth_tracks(depth_df: pd.DataFrame, out_path: Path) -> None:
    """Depth track: sonic, Gassmann, hybrid OOF per regressor."""
    work = depth_df.sort_values(DEPTH_COL)
    fig, ax = plt.subplots(figsize=(7.5, 9.0))
    ax.plot(
        work["vp_sonic_km_s"],
        work[DEPTH_COL],
        "o-",
        color="#2ca02c",
        label="Vp sonic (DSI)",
        markersize=2.5,
        linewidth=1.2,
        zorder=10,
    )
    ax.plot(
        work["vp_gassmann_km_s"],
        work[DEPTH_COL],
        "s--",
        color="#1f77b4",
        label="Vp Gassmann",
        markersize=2.5,
        linewidth=1.0,
        alpha=0.85,
        zorder=5,
    )
    for reg in RESIDUAL_REGRESSORS:
        col = "vp_hybrid_{}_oof_km_s".format(reg)
        if col not in work.columns:
            continue
        lw = 1.4 if reg == "rf" else 1.0
        alpha = 1.0 if reg in ("rf", "ridge") else 0.75
        zorder = 8 if reg == "rf" else (7 if reg == "ridge" else 4)
        ax.plot(
            work[col],
            work[DEPTH_COL],
            "-",
            color=REGRESSOR_LINE_COLORS.get(reg, "#4c72b0"),
            label="Hybrid OOF ({})".format(REGRESSOR_DISPLAY.get(reg, reg)),
            linewidth=lw,
            alpha=alpha,
            zorder=zorder,
        )
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title("Well 861: sonic vs Gassmann vs hybrid by regressor (OOF)")
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=6.5, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regressor_sensitivity_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: hybrid Vp MAPE OOF by regressor."""
    work = df.sort_values("hybrid_mape_vp_pct")
    labels = [REGRESSOR_DISPLAY.get(r, r) for r in work["regressor"].tolist()]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = ["#d62728" if r == "rf" else "#4c72b0" for r in work["regressor"].tolist()]
    ax.barh(labels, work["hybrid_mape_vp_pct"].to_numpy(dtype=np.float64), color=colors)
    ax.axvline(15.369, color="#1f77b4", linestyle="--", linewidth=1.0, label="Gassmann MAPE")
    ax.set_xlabel("MAPE Vp hybrid OOF vs DSI (%)")
    ax.set_title("Regressor sensitivity (depth-block OOF, 87 rows)")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_regressor_sensitivity(
    bundle: XYBundle,
    n_blocks: int = 3,
    smoke: bool = False,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, List[Dict[str, object]], pd.DataFrame]:
    """Compare Etapa 1 regressors (+ Ridge) on residual target and hybrid Vp."""
    vp_sonic = bundle.df["vp_sonic_km_s"].to_numpy(dtype=np.float64)
    vp_gass = bundle.df["vp_gassmann_km_s"].to_numpy(dtype=np.float64)
    rows: List[Dict[str, object]] = []
    details: List[Dict[str, object]] = []
    hybrid_by_reg: Dict[str, np.ndarray] = {}

    for reg in RESIDUAL_REGRESSORS:
        row: Dict[str, object] = {
            "regressor": reg,
            "display_name": REGRESSOR_DISPLAY.get(reg, reg),
            "skipped": False,
            "skip_reason": "",
        }
        try:
            factory = residual_regressor_factory(reg, smoke=smoke, random_state=random_state)
            cv_summary, oof_residual = evaluate_depth_blocks_oof(factory, bundle, n_blocks=n_blocks)
            cv_summary.model_name = REGRESSOR_DISPLAY.get(reg, reg)
            vp_hybrid = vp_gass + oof_residual
            hybrid_by_reg[reg] = vp_hybrid
            hybrid_m = vp_metrics_vs_sonic(vp_hybrid, vp_sonic)
            row.update(
                {
                    "residual_oof_rmse_km_s": cv_summary.global_oof_rmse,
                    "residual_oof_r2": cv_summary.global_oof_r2,
                    "residual_oof_mae_km_s": cv_summary.global_oof_mae,
                    "hybrid_mape_vp_pct": hybrid_m.mape_pct,
                    "hybrid_rmse_vp_km_s": hybrid_m.rmse_km_s,
                    "hybrid_bias_vp_km_s": hybrid_m.bias_km_s,
                }
            )
            details.append({"regressor": reg, "cv": cv_summary.to_dict()})
        except ImportError as exc:
            row["skipped"] = True
            row["skip_reason"] = str(exc)
            for col in (
                "residual_oof_rmse_km_s",
                "residual_oof_r2",
                "residual_oof_mae_km_s",
                "hybrid_mape_vp_pct",
                "hybrid_rmse_vp_km_s",
                "hybrid_bias_vp_km_s",
            ):
                row[col] = float("nan")
            details.append({"regressor": reg, "skipped": True, "reason": str(exc)})
        except Exception as exc:
            row["skipped"] = True
            row["skip_reason"] = str(exc)
            for col in (
                "residual_oof_rmse_km_s",
                "residual_oof_r2",
                "residual_oof_mae_km_s",
                "hybrid_mape_vp_pct",
                "hybrid_rmse_vp_km_s",
                "hybrid_bias_vp_km_s",
            ):
                row[col] = float("nan")
            details.append({"regressor": reg, "skipped": True, "reason": str(exc)})
        rows.append(row)

    depth_df = bundle.df[
        [DEPTH_COL, HFU_COL, "vp_sonic_km_s", "vp_gassmann_km_s"]
    ].copy()
    for reg, vp_hybrid in hybrid_by_reg.items():
        depth_df["vp_hybrid_{}_oof_km_s".format(reg)] = vp_hybrid

    return pd.DataFrame(rows), details, depth_df


def plot_vp_depth_tracks(
    df: pd.DataFrame,
    out_path: Path,
    hybrid_col: str = "vp_hybrid_oof_km_s",
    hybrid_label: str = "Vp hybrid OOF",
    hybrid_color: str = "#d62728",
    hybrid_marker: str = "d",
    title: str = "Well 861: physics vs hybrid vs sonic (OOF)",
) -> None:
    """Depth track: sonic vs Gassmann vs hybrid OOF (one regressor)."""
    if hybrid_col not in df.columns:
        raise KeyError("Missing hybrid column: {}".format(hybrid_col))
    work = df.sort_values(DEPTH_COL)
    fig, ax = plt.subplots(figsize=(6.0, 8.0))
    ax.plot(
        work["vp_sonic_km_s"],
        work[DEPTH_COL],
        "o-",
        color="#2ca02c",
        label="Vp sonic",
        markersize=3,
        linewidth=1.0,
    )
    ax.plot(
        work["vp_gassmann_km_s"],
        work[DEPTH_COL],
        "s--",
        color="#1f77b4",
        label="Vp Gassmann",
        markersize=3,
        linewidth=1.0,
    )
    ax.plot(
        work[hybrid_col],
        work[DEPTH_COL],
        hybrid_marker + "-",
        color=hybrid_color,
        label=hybrid_label,
        markersize=3,
        linewidth=1.2,
    )
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_residual_pipeline(
    validation_csv: Path,
    out_root: Path,
    smoke: bool = False,
    random_state: int = 42,
) -> Dict[str, object]:
    """Build dataset, train RF/Ridge, compare physics vs hybrid."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    rf_dir = residual_vp_rf_dir()
    for d in (tables_dir, figures_dir, rf_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_estimators = 10 if smoke else 200
    n_blocks = 2 if smoke else 3

    dataset = build_residual_dataset(validation_csv)
    dataset.to_csv(tables_dir / "residual_dataset.csv", index=False, float_format="%.6f")

    feature_cols = build_residual_feature_columns(dataset)
    bundle = build_xy_from_columns(
        dataset,
        target=RESIDUAL_VP_TARGET,
        feature_columns=feature_cols,
    )

    def rf_factory() -> RandomForestRegressor:
        return RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)

    cv_rf, oof_residual = evaluate_depth_blocks_oof(rf_factory, bundle, n_blocks=n_blocks)
    cv_rf.model_name = "RandomForestRegressor"

    def ridge_factory() -> Ridge:
        return Ridge(alpha=1.0, random_state=random_state)

    cv_ridge, _ = evaluate_depth_blocks_oof(ridge_factory, bundle, n_blocks=n_blocks)
    cv_ridge.model_name = "Ridge"

    reg_df, reg_details, reg_depth_df = run_regressor_sensitivity(
        bundle,
        n_blocks=n_blocks,
        smoke=smoke,
        random_state=random_state,
    )
    reg_df.to_csv(
        tables_dir / "regressor_sensitivity.csv",
        index=False,
        float_format="%.6f",
    )
    reg_depth_df.to_csv(
        tables_dir / "regressor_oof_vp_depth.csv",
        index=False,
        float_format="%.6f",
    )
    ok_df = reg_df[reg_df["skipped"] == False].copy()  # noqa: E712
    if not ok_df.empty:
        plot_regressor_sensitivity_bar(
            ok_df,
            figures_dir / "regressor_sensitivity_mape.png",
        )
    if not reg_depth_df.empty:
        plot_regressor_depth_tracks(
            reg_depth_df,
            figures_dir / "regressor_depth_comparison.png",
        )

    work = bundle.df.copy()
    work["residual_pred_oof_km_s"] = oof_residual
    work["vp_hybrid_oof_km_s"] = work["vp_gassmann_km_s"] + work["residual_pred_oof_km_s"]

    physics_metrics = vp_metrics_vs_sonic(
        work["vp_gassmann_km_s"].to_numpy(dtype=np.float64),
        work["vp_sonic_km_s"].to_numpy(dtype=np.float64),
    )
    hybrid_metrics = vp_metrics_vs_sonic(
        work["vp_hybrid_oof_km_s"].to_numpy(dtype=np.float64),
        work["vp_sonic_km_s"].to_numpy(dtype=np.float64),
    )

    comparison = pd.DataFrame(
        [
            {"model": "gassmann_physics", **physics_metrics.to_dict()},
            {"model": "hybrid_rf_oof", **hybrid_metrics.to_dict()},
        ]
    )
    comparison.to_csv(
        tables_dir / "comparison_physics_vs_hybrid.csv",
        index=False,
        float_format="%.6f",
    )

    oof_cols = [
        DEPTH_COL,
        HFU_COL,
        "vp_gassmann_km_s",
        "vp_sonic_km_s",
        "vp_residual_km_s",
        "residual_pred_oof_km_s",
        "vp_hybrid_oof_km_s",
    ]
    work[oof_cols].to_csv(tables_dir / "oof_predictions.csv", index=False, float_format="%.6f")

    by_hfu_physics = summary_by_hfu_vp(work, "vp_gassmann_km_s")
    by_hfu_physics["model"] = "gassmann_physics"
    by_hfu_hybrid = summary_by_hfu_vp(work, "vp_hybrid_oof_km_s")
    by_hfu_hybrid["model"] = "hybrid_rf_oof"
    by_hfu = pd.concat([by_hfu_physics, by_hfu_hybrid], ignore_index=True)
    by_hfu.to_csv(tables_dir / "summary_by_hfu.csv", index=False, float_format="%.6f")

    plot_residual_oof_scatter(
        work["vp_residual_km_s"].to_numpy(dtype=np.float64),
        work["residual_pred_oof_km_s"].to_numpy(dtype=np.float64),
        figures_dir / "residual_oof_scatter.png",
        "Vp residual OOF (RF, depth-block CV)",
    )
    plot_vp_depth_tracks(
        work,
        figures_dir / "vp_physics_vs_hybrid_vs_sonic_depth.png",
        hybrid_col="vp_hybrid_oof_km_s",
        hybrid_label="Vp hybrid OOF (RF)",
        hybrid_color="#d62728",
        title="Well 861: RF hybrid vs Gassmann vs sonic (OOF)",
    )
    ridge_col = "vp_hybrid_ridge_oof_km_s"
    if ridge_col in reg_depth_df.columns:
        plot_vp_depth_tracks(
            reg_depth_df,
            figures_dir / "vp_physics_vs_hybrid_ridge_vs_sonic_depth.png",
            hybrid_col=ridge_col,
            hybrid_label="Vp hybrid OOF (Ridge)",
            hybrid_color="#9467bd",
            hybrid_marker="^",
            title="Well 861: Ridge hybrid vs Gassmann vs sonic (OOF)",
        )
        ridge_oof_cols = [
            DEPTH_COL,
            HFU_COL,
            "vp_gassmann_km_s",
            "vp_sonic_km_s",
            ridge_col,
        ]
        reg_depth_df[ridge_oof_cols].to_csv(
            tables_dir / "ridge_oof_predictions.csv",
            index=False,
            float_format="%.6f",
        )

    model = rf_factory()
    model.fit(bundle.X, bundle.y)
    joblib.dump(model, rf_dir / "model_vp_residual_rf_861.joblib")

    metrics: Dict[str, object] = {
        "well_id": "861",
        "approach": "residual_vp_depth_block_cv",
        "target": RESIDUAL_VP_TARGET,
        "n_rows": int(len(work)),
        "feature_names": feature_cols,
        "residual_cv_rf": cv_rf.to_dict(),
        "residual_cv_ridge": cv_ridge.to_dict(),
        "regressor_sensitivity": reg_details,
        "physics_vs_sonic": physics_metrics.to_dict(),
        "hybrid_vs_sonic": hybrid_metrics.to_dict(),
        "bias_improvement_km_s": float(
            physics_metrics.bias_km_s - hybrid_metrics.bias_km_s
        ),
        "mape_improvement_pct": float(
            physics_metrics.mape_pct - hybrid_metrics.mape_pct
        ),
        "smoke": smoke,
        "generated_utc": utc_now_iso(),
    }
    (out_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = [
        "Well 861 -- Etapa 3 ML residual Vp",
        "Generated: {}".format(utc_now_iso()),
        "Rows: {}".format(len(work)),
        "Physics MAPE={:.1f}% bias={:+.3f} km/s".format(
            physics_metrics.mape_pct, physics_metrics.bias_km_s
        ),
        "Hybrid  MAPE={:.1f}% bias={:+.3f} km/s".format(
            hybrid_metrics.mape_pct, hybrid_metrics.bias_km_s
        ),
        "Regressor sensitivity: tables/regressor_sensitivity.csv",
        "Regressor depth tracks: tables/regressor_oof_vp_depth.csv",
        "Ridge depth track: figures/vp_physics_vs_hybrid_ridge_vs_sonic_depth.png",
    ]
    (out_root / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Well 861 Etapa 3: ML Vp residual")
    p.add_argument("--validation-csv", type=Path, default=DLIS_GASSMANN_VALIDATION_CSV)
    p.add_argument("--out-root", type=Path, default=ML_RESIDUAL_VP_ROOT)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    metrics = run_residual_pipeline(
        validation_csv=args.validation_csv.resolve(),
        out_root=args.out_root.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
    )
    phys = metrics["physics_vs_sonic"]
    hyb = metrics["hybrid_vs_sonic"]
    print(
        "OK residual_vp smoke={} physics MAPE={:.1f}% bias={:+.3f}".format(
            metrics["smoke"],
            phys["mape_vp_pct"],
            phys["bias_vp_km_s"],
        )
    )
    print(
        "   hybrid OOF      MAPE={:.1f}% bias={:+.3f}".format(
            hyb["mape_vp_pct"],
            hyb["bias_vp_km_s"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
