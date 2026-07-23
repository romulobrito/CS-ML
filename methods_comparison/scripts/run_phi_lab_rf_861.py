#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Random Forest baseline for Phi_lab (pu) on Well 861 -- well-profile approach.

87 rows, depth-block CV on wireline features.
Protocol: methods_comparison/planning/etapa1d_well_profile_ct_plugs_poco861.md
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

from ml_861_data import (
    DEFAULT_ENRICHED,
    WELL_PROFILE_PRIMARY_TARGET,
    build_xy,
    load_logs_enriched,
    well_profile_phi_rf_dir,
)
from ml_861_metrics import evaluate_depth_blocks

ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = well_profile_phi_rf_dir()


def _make_rf(n_estimators: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def run_phi_lab_rf(
    data_path: Path,
    out_dir: Path,
    smoke: bool = False,
    random_state: int = 42,
    run_shap: bool = True,
) -> dict:
    """Train RF for Phi_lab, evaluate, save artifacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = WELL_PROFILE_PRIMARY_TARGET
    n_estimators = 10 if smoke else 200
    n_blocks = 2 if smoke else 3
    shap_enabled = run_shap and not smoke

    df = load_logs_enriched(data_path)
    bundle = build_xy(df, target=target, feature_mode="log_only")

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

    model_path = out_dir / "model_phi_lab_rf_861.joblib"
    joblib.dump(model, model_path)

    metrics = {
        "well_id": "861",
        "approach": "well_profile_depth_block_cv",
        "target": target,
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
    ax.scatter(pred_test, y_test, color="tab:green", alpha=0.8)
    lim_lo = min(float(np.min(y_test)), float(np.min(pred_test)), 0.0)
    lim_hi = max(float(np.max(y_test)), float(np.max(pred_test)), 0.01) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=1.5)
    ax.set_xlabel("Phi_lab_predicted (pu)")
    ax.set_ylabel("Phi_lab_target (pu)")
    ax.set_title("Phi_lab RF Well 861 (well_profile, depth-block CV)")
    ax.text(0.05 * lim_hi, 0.9 * lim_hi, "RMSE={:.4f}".format(holdout_rmse))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "phi_pred_vs_obs.png", dpi=150)
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
            plt.title("SHAP bar -- Phi_lab Well 861")
            plt.tight_layout()
            plt.savefig(out_dir / "shap_bar.png", dpi=150)
            plt.close()
        except Exception as exc:
            metrics["shap_error"] = str(exc)

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phi_lab RF baseline Well 861 (well_profile depth-block CV)"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    metrics = run_phi_lab_rf(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
        run_shap=not args.no_shap,
    )
    cv = metrics["depth_block_cv"]
    print(
        "OK Phi_lab RF well_profile smoke={} depth_cv_rmse={:.4f} depth_cv_r2={:.4f}".format(
            metrics["smoke"],
            cv["mean_rmse"],
            cv["mean_r2"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
