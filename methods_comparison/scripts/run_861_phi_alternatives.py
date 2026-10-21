#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phi_lab well_profile alternatives: Ridge and GAM vs RF baseline.

Protocol: depth-block CV (same as run_phi_lab_rf_861.py).
GAM: pygam if installed, else SplineTransformer + Ridge (sklearn).

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    DEFAULT_ENRICHED,
    WELL_PROFILE_PRIMARY_TARGET,
    build_xy,
    load_logs_enriched,
    well_profile_phi_alternatives_dir,
    well_profile_phi_rf_dir,
)
from ml_861_metrics import CvSummary, evaluate_depth_blocks

ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = well_profile_phi_alternatives_dir()


def _ridge_factory(alpha: float = 1.0) -> Callable[[], Any]:
    return lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def _gam_factory(n_features: int, smoke: bool) -> tuple[Callable[[], Any], str]:
    """Return GAM estimator factory and backend label."""
    try:
        from pygam import LinearGAM, s

        n_splines = 4 if smoke else 6
        term = s(0, n_splines=n_splines)
        for i in range(1, n_features):
            term = term + s(i, n_splines=n_splines)

        def factory() -> Any:
            return LinearGAM(term)

        return factory, "pygam"
    except ImportError:
        n_knots = 4 if smoke else 6

        def factory() -> Any:
            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "spline",
                        SplineTransformer(
                            n_knots=n_knots,
                            degree=3,
                            include_bias=False,
                        ),
                    ),
                    ("model", Ridge(alpha=1.0)),
                ]
            )

        return factory, "spline_ridge_sklearn"


def _load_rf_baseline_metrics() -> Optional[Dict[str, Any]]:
    path = well_profile_phi_rf_dir() / "metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cv_row(model_name: str, cv: CvSummary, smoke: bool, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "target": cv.target,
        "model": model_name,
        "protocol": cv.protocol,
        "mean_rmse": cv.mean_rmse,
        "std_rmse": cv.std_rmse,
        "mean_r2": cv.mean_r2,
        "std_r2": cv.std_r2,
        "smoke": smoke,
        "approach": "well_profile_depth_block_cv",
    }
    if extra:
        row.update(extra)
    return row


def _plot_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart mean R2 across Phi_lab models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    order = summary_df.sort_values("mean_r2", ascending=True)
    colors = ["tab:green" if m == "rf_baseline" else "tab:blue" for m in order["model"]]
    ax.barh(order["model"], order["mean_r2"], color=colors)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("mean R2 (depth-block CV)")
    ax.set_title("Phi_lab well_profile: RF vs Ridge vs GAM")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_phi_alternatives(
    data_path: Path,
    out_dir: Path,
    smoke: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run Ridge and GAM; compare with existing RF metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_blocks = 2 if smoke else 3
    target = WELL_PROFILE_PRIMARY_TARGET

    df = load_logs_enriched(data_path)
    bundle = build_xy(df, target=target, feature_mode="log_only")
    n_features = bundle.X.shape[1]

    rows: List[Dict[str, Any]] = []
    model_details: Dict[str, Any] = {
        "well_id": "861",
        "approach": "well_profile_depth_block_cv",
        "target": target,
        "smoke": smoke,
        "n_rows": len(bundle.df),
        "feature_names": bundle.feature_names,
        "models": {},
    }

    ridge_cv = evaluate_depth_blocks(_ridge_factory(alpha=1.0), bundle, n_blocks=n_blocks)
    ridge_cv.model_name = "Ridge"
    rows.append(_cv_row("ridge", ridge_cv, smoke))
    model_details["models"]["ridge"] = ridge_cv.to_dict()

    gam_factory, gam_backend = _gam_factory(n_features, smoke=smoke)
    gam_cv = evaluate_depth_blocks(gam_factory, bundle, n_blocks=n_blocks)
    gam_cv.model_name = "GAM"
    rows.append(_cv_row("gam", gam_cv, smoke, extra={"gam_backend": gam_backend}))
    model_details["models"]["gam"] = gam_cv.to_dict()
    model_details["gam_backend"] = gam_backend

    rf_metrics = _load_rf_baseline_metrics()
    if rf_metrics is not None:
        rf_cv = rf_metrics.get("depth_block_cv", {})
        rows.append(
            {
                "target": target,
                "model": "rf_baseline",
                "protocol": rf_cv.get("protocol", "depth_block_3"),
                "mean_rmse": rf_cv.get("mean_rmse"),
                "std_rmse": rf_cv.get("std_rmse"),
                "mean_r2": rf_cv.get("mean_r2"),
                "std_r2": rf_cv.get("std_r2"),
                "smoke": rf_metrics.get("smoke", False),
                "approach": "well_profile_depth_block_cv",
                "source": str(well_profile_phi_rf_dir() / "metrics.json"),
            }
        )
        model_details["models"]["rf_baseline"] = rf_cv
    else:
        model_details["rf_baseline_note"] = "RF metrics.json not found; run run_phi_lab_rf_861.py first"

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "861_phi_model_comparison.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(model_details, indent=2), encoding="utf-8")
    _plot_comparison(summary_df, out_dir / "phi_model_comparison_mean_r2.png")

    return summary_df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phi_lab Ridge/GAM vs RF (well_profile)")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_phi_alternatives(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
    )
    print(summary[["model", "mean_rmse", "mean_r2", "protocol"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
