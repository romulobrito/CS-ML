#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare five ML regressors on Well 861 integrated dataset.

Regressors: RF, Gradient Boosting, XGBoost, MLP, Linear Regression.
Protocol: depth-block CV (well_profile) or leave-one-plug-out (ct_plugs).

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    DEFAULT_CT,
    DEFAULT_ENRICHED,
    DatasetMode,
    FeatureMode,
    build_xy,
    compare_out_dir_for_target,
    load_dataset,
    target_slug,
)
from ml_861_metrics import CvSummary, evaluate_depth_blocks, evaluate_plug_out

ROOT = SCRIPT_DIR.parents[1]
DEFAULT_COMPARE_ROOT = (
    ROOT / "methods_comparison" / "data" / "processed" / "ml_runs" / "compare_861"
)

REGRESSOR_CHOICES = ("rf", "gb", "xgb", "mlp", "lr", "all")


def _scaled_factory(
    base_factory: Callable[[], Any],
    scaler_kind: str,
) -> Callable[[], Any]:
    """Wrap estimator with scaler in a simple callable for CV (fit per fold)."""
    from sklearn.pipeline import Pipeline

    if scaler_kind == "minmax":
        return lambda: Pipeline([("scaler", MinMaxScaler()), ("model", base_factory())])
    if scaler_kind == "standard":
        return lambda: Pipeline([("scaler", StandardScaler()), ("model", base_factory())])
    return base_factory


def model_factory(
    name: str,
    smoke: bool,
    random_state: int = 42,
    small_sample: bool = False,
) -> Callable[[], Any]:
    """Return a zero-arg callable that builds a fresh estimator."""
    if name == "rf":
        n = 10 if smoke else 200
        base = lambda: RandomForestRegressor(n_estimators=n, random_state=random_state)
        return _scaled_factory(base, "none")
    if name == "gb":
        n = 20 if smoke else 200
        base = lambda: GradientBoostingRegressor(n_estimators=n, random_state=random_state)
        return _scaled_factory(base, "none")
    if name == "xgb":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "xgboost not installed; pip install xgboost or exclude regressor xgb"
            ) from exc

        n = 20 if smoke else 200
        base = lambda: xgb.XGBRegressor(
            n_estimators=n,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=1,
        )
        return _scaled_factory(base, "none")
    if name == "mlp":
        use_early_stop = not smoke and not small_sample
        hidden = (32,) if (smoke or small_sample) else (128, 64)
        base = lambda: MLPRegressor(
            hidden_layer_sizes=hidden,
            max_iter=80 if smoke else 500,
            early_stopping=use_early_stop,
            validation_fraction=0.15,
            random_state=random_state,
        )
        return _scaled_factory(base, "standard")
    if name == "lr":
        base = lambda: LinearRegression()
        return _scaled_factory(base, "standard")
    raise ValueError("Unknown regressor: {}".format(name))


def run_compare(
    dataset_mode: DatasetMode,
    data_path: Path,
    target: str,
    feature_mode: FeatureMode,
    regressors: Sequence[str],
    out_dir: Path,
    smoke: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run CV for each regressor and save summary CSV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_dataset(dataset_mode, data_path)
    bundle = build_xy(df, target=target, feature_mode=feature_mode)

    rows: List[Dict[str, Any]] = []
    summaries: Dict[str, Any] = {
        "well_id": "861",
        "dataset_mode": dataset_mode,
        "target": target,
        "feature_mode": feature_mode,
        "smoke": smoke,
        "n_rows": len(bundle.df),
        "feature_names": bundle.feature_names,
        "models": {},
    }

    for reg in regressors:
        try:
            factory = model_factory(
                reg,
                smoke=smoke,
                random_state=random_state,
                small_sample=(dataset_mode == "ct"),
            )
        except ImportError as exc:
            rows.append(
                {
                    "well_id": "861",
                    "dataset_mode": dataset_mode,
                    "target": target,
                    "regressor": reg,
                    "protocol": "skipped",
                    "mean_rmse": float("nan"),
                    "std_rmse": float("nan"),
                    "mean_r2": float("nan"),
                    "std_r2": float("nan"),
                    "smoke": smoke,
                    "skip_reason": str(exc),
                }
            )
            summaries["models"][reg] = {"skipped": True, "reason": str(exc)}
            continue

        if dataset_mode == "ct":
            try:
                cv: CvSummary = evaluate_plug_out(factory, bundle)
            except Exception as exc:
                rows.append(
                    {
                        "well_id": "861",
                        "dataset_mode": dataset_mode,
                        "target": target,
                        "regressor": reg,
                        "protocol": "skipped",
                        "mean_rmse": float("nan"),
                        "std_rmse": float("nan"),
                        "mean_r2": float("nan"),
                        "std_r2": float("nan"),
                        "smoke": smoke,
                        "skip_reason": str(exc),
                    }
                )
                summaries["models"][reg] = {"skipped": True, "reason": str(exc)}
                continue
        else:
            n_blocks = 2 if smoke else 3
            try:
                cv = evaluate_depth_blocks(factory, bundle, n_blocks=n_blocks)
            except Exception as exc:
                rows.append(
                    {
                        "well_id": "861",
                        "dataset_mode": dataset_mode,
                        "target": target,
                        "regressor": reg,
                        "protocol": "skipped",
                        "mean_rmse": float("nan"),
                        "std_rmse": float("nan"),
                        "mean_r2": float("nan"),
                        "std_r2": float("nan"),
                        "smoke": smoke,
                        "skip_reason": str(exc),
                    }
                )
                summaries["models"][reg] = {"skipped": True, "reason": str(exc)}
                continue
        cv.model_name = reg
        summaries["models"][reg] = cv.to_dict()
        rows.append(
            {
                "well_id": "861",
                "dataset_mode": dataset_mode,
                "target": target,
                "regressor": reg,
                "protocol": cv.protocol,
                "mean_rmse": cv.mean_rmse,
                "std_rmse": cv.std_rmse,
                "mean_r2": cv.mean_r2,
                "std_r2": cv.std_r2,
                "smoke": smoke,
            }
        )

    summary_df = pd.DataFrame(rows)
    csv_path = out_dir / "861_ml_baseline_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    per_target_dir = out_dir
    per_target_dir.mkdir(parents=True, exist_ok=True)
    json_path = per_target_dir / "metrics.json"
    json_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    return summary_df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Well 861 five-regressor ML baseline")
    parser.add_argument("--dataset", choices=("enriched", "ct"), default="enriched")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Override default enriched or CT xlsx path",
    )
    parser.add_argument("--target", type=str, default="FZI_lab")
    parser.add_argument(
        "--feature-mode",
        choices=("log_only", "log_plus_ct"),
        default="log_only",
    )
    parser.add_argument(
        "--regressor",
        choices=REGRESSOR_CHOICES,
        default="all",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output dir (default: compare_861/by_target/<target_slug>/)",
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_path = args.data_path
    if data_path is None:
        data_path = DEFAULT_ENRICHED if args.dataset == "enriched" else DEFAULT_CT

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = compare_out_dir_for_target(args.target, base=DEFAULT_COMPARE_ROOT)

    if args.regressor == "all":
        regs = ["rf", "gb", "xgb", "mlp", "lr"]
    else:
        regs = [args.regressor]

    df = run_compare(
        dataset_mode=args.dataset,
        data_path=data_path.resolve(),
        target=args.target,
        feature_mode=args.feature_mode,
        regressors=regs,
        out_dir=out_dir.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
    )
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
