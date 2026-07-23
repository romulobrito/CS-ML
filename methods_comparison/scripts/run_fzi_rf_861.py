#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest baseline for FZI_lab on Well 861 integrated dataset.

Replaces Windows-path legacy script with repo-relative paths, depth-block CV,
and optional SHAP. Protocol: etapa1c_ml_baseline_poco861.md

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import DEFAULT_ENRICHED, build_xy, load_logs_enriched
from ml_861_metrics import evaluate_depth_blocks

ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = ROOT / "methods_comparison" / "data" / "processed" / "ml_runs" / "fzi_rf"


def _make_rf(n_estimators: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def run_fzi_rf(
    data_path: Path,
    out_dir: Path,
    smoke: bool = False,
    random_state: int = 42,
    run_shap: bool = True,
) -> dict:
    """Train RF for FZI_lab, evaluate, save artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_estimators = 10 if smoke else 200
    n_blocks = 2 if smoke else 3
    shap_enabled = run_shap and not smoke

    df = load_logs_enriched(data_path)
    bundle = build_xy(df, target="FZI_lab", feature_mode="log_only")

    cv = evaluate_depth_blocks(
        lambda: _make_rf(n_estimators, random_state),
        bundle,
        n_blocks=n_blocks,
    )
    cv.model_name = "RandomForestRegressor"

    x_train, x_test, y_train, y_test = train_test_split(
        bundle.X, bundle.y, test_size=0.2, random_state=random_state
    )
    model = _make_rf(n_estimators, random_state)
    model.fit(x_train, y_train)
    pred_test = model.predict(x_test)
    holdout_rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
    holdout_r2 = float(r2_score(y_test, pred_test))

    model_path = out_dir / "model_fzi_rf_861.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "well_id": "861",
        "target": "FZI_lab",
        "model": "RandomForestRegressor",
        "n_estimators": n_estimators,
        "smoke": smoke,
        "n_rows": len(bundle.df),
        "n_features": len(bundle.feature_names),
        "feature_names": bundle.feature_names,
        "depth_block_cv": cv.to_dict(),
        "holdout_80_20": {
            "rmse": holdout_rmse,
            "r2": holdout_r2,
            "note": "legacy comparison only; prefer depth_block_cv",
        },
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred_test, y_test, color="tab:blue", alpha=0.8)
    lim_hi = max(float(np.max(y_test)), float(np.max(pred_test)), 1.0) * 1.05
    ax.plot([0, lim_hi], [0, lim_hi], "k--", linewidth=1.5)
    ax.set_xlabel("FZI_predicted")
    ax.set_ylabel("FZI_target")
    ax.set_title("FZI Random Forest Well 861")
    ax.text(0.05 * lim_hi, 0.9 * lim_hi, "RMSE={:.3f}".format(holdout_rmse))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fzi_pred_vs_obs.png", dpi=150)
    plt.close(fig)

    if shap_enabled:
        try:
            import shap

            explainer = shap.Explainer(model, feature_names=bundle.feature_names)
            sample_n = min(50, bundle.X.shape[0])
            x_sample = bundle.X[:sample_n]
            shap_values = explainer(x_sample)
            plt.figure(figsize=(8, 5))
            shap.plots.bar(shap_values, show=False)
            plt.title("SHAP bar -- FZI_lab Well 861")
            plt.tight_layout()
            plt.savefig(out_dir / "shap_bar.png", dpi=150)
            plt.close()
        except Exception as exc:
            metrics["shap_error"] = str(exc)

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FZI_lab Random Forest baseline Well 861")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true", help="Fast run for CI/smoke tests")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metrics = run_fzi_rf(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
        run_shap=not args.no_shap,
    )
    cv = metrics["depth_block_cv"]
    print(
        "OK FZI RF well=861 smoke={} depth_cv_rmse={:.4f} depth_cv_r2={:.4f} holdout_rmse={:.4f}".format(
            metrics["smoke"],
            cv["mean_rmse"],
            cv["mean_r2"],
            metrics["holdout_80_20"]["rmse"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
