#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare Lithotype encoding for Phi_lab well_profile (Well 861).

Variants:
  - integer: single column 1--4 (current Etapa 1 default)
  - onehot: four binary lith_1..lith_4 columns
  - none: wireline without Lithotype

Models: RandomForestRegressor, Ridge (scaled).
CV: depth-block OOF (3 folds), same as run_phi_lab_rf_861.py.

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (  # noqa: E402
    DEFAULT_ENRICHED,
    LOG_FEATURE_COLUMNS,
    WELL_PROFILE_PRIMARY_TARGET,
    XYBundle,
    load_logs_enriched,
)
from ml_861_metrics import collect_depth_block_oof, evaluate_depth_blocks  # noqa: E402

LITHOTYPE_COL = "Lithotype"
CONTINUOUS_COLS: Tuple[str, ...] = tuple(
    c for c in LOG_FEATURE_COLUMNS if c != LITHOTYPE_COL
)
LITH_ENCODING = Literal["integer", "onehot", "none"]

ROOT = SCRIPT_DIR.parents[1]
OUT_ROOT = (
    ROOT
    / "methods_comparison"
    / "data"
    / "processed"
    / "ml_runs"
    / "diagnostics_861"
    / "lithotype_encoding"
)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _onehot_matrix(lith_values: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Four-column one-hot for lithotype codes 1--4."""
    lith = lith_values.astype(np.int64)
    names = ["lith_{}".format(i) for i in range(1, 5)]
    cols: List[np.ndarray] = []
    for code in range(1, 5):
        cols.append((lith == code).astype(np.float64))
    return np.column_stack(cols), names


def build_xy_lithotype_encoding(
    df: pd.DataFrame,
    encoding: LITH_ENCODING,
    target: str = WELL_PROFILE_PRIMARY_TARGET,
) -> XYBundle:
    """Build (X, y) with chosen Lithotype encoding."""
    if target not in df.columns:
        raise ValueError("Target not found: {}".format(target))

    required = list(CONTINUOUS_COLS)
    if encoding in ("integer", "onehot"):
        required.append(LITHOTYPE_COL)

    work = df.dropna(subset=[target] + required).copy()
    if work.empty:
        raise ValueError("No rows after NaN drop for encoding={}".format(encoding))

    x_cont = work[list(CONTINUOUS_COLS)].to_numpy(dtype=np.float64)
    feature_names = list(CONTINUOUS_COLS)

    if encoding == "integer":
        x_lith = work[[LITHOTYPE_COL]].to_numpy(dtype=np.float64)
        x_all = np.hstack([x_cont, x_lith])
        feature_names.append(LITHOTYPE_COL)
    elif encoding == "onehot":
        x_lith, lith_names = _onehot_matrix(work[LITHOTYPE_COL].to_numpy())
        x_all = np.hstack([x_cont, x_lith])
        feature_names.extend(lith_names)
    elif encoding == "none":
        x_all = x_cont
    else:
        raise ValueError("Unknown encoding: {}".format(encoding))

    y = work[target].to_numpy(dtype=np.float64)
    return XYBundle(
        X=x_all,
        y=y,
        feature_names=feature_names,
        target=target,
        df=work.reset_index(drop=True),
    )


def _make_rf(n_estimators: int, random_state: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
    )


def _make_ridge(alpha: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def run_compare(
    data_path: Path,
    out_dir: Path,
    n_estimators: int = 200,
    n_blocks: int = 3,
    random_state: int = 42,
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    """Run encoding comparison and write tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = load_logs_enriched(data_path)
    rows: List[Dict[str, object]] = []

    encodings: List[LITH_ENCODING] = ["integer", "onehot", "none"]
    models = {
        "RandomForest": lambda: _make_rf(n_estimators, random_state),
        "Ridge": lambda: _make_ridge(ridge_alpha),
    }

    for enc in encodings:
        bundle = build_xy_lithotype_encoding(df, encoding=enc)
        for model_name, factory in models.items():
            cv = evaluate_depth_blocks(factory, bundle, n_blocks=n_blocks)
            oof_pred, _ = collect_depth_block_oof(factory, bundle, n_blocks=n_blocks)
            mask = np.isfinite(oof_pred)
            y_oof = bundle.y[mask]
            pred_oof = oof_pred[mask]
            rows.append(
                {
                    "lithotype_encoding": enc,
                    "model": model_name,
                    "n_features": len(bundle.feature_names),
                    "feature_names": ",".join(bundle.feature_names),
                    "mean_rmse_fold": cv.mean_rmse,
                    "std_rmse_fold": cv.std_rmse,
                    "mean_r2_fold": cv.mean_r2,
                    "std_r2_fold": cv.std_r2,
                    "oof_rmse_global": _rmse(y_oof, pred_oof),
                    "oof_r2_global": float(r2_score(y_oof, pred_oof)),
                    "oof_mae_global": float(mean_absolute_error(y_oof, pred_oof)),
                    "n_points": int(len(y_oof)),
                    "n_blocks": int(n_blocks),
                    "random_state": int(random_state),
                }
            )

    result = pd.DataFrame(rows)
    csv_path = tables_dir / "lithotype_encoding_compare.csv"
    result.to_csv(csv_path, index=False)

    meta = {
        "well_id": "861",
        "target": WELL_PROFILE_PRIMARY_TARGET,
        "protocol": "depth_block_oof_{}".format(n_blocks),
        "n_estimators_rf": int(n_estimators),
        "ridge_alpha": float(ridge_alpha),
        "random_state": int(random_state),
        "encodings": encodings,
        "csv": str(csv_path),
    }
    (out_dir / "lithotype_encoding_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lithotype integer vs one-hot vs none for Phi_lab (Well 861)."
    )
    p.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    p.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    p.add_argument("--n-estimators", type=int, default=200)
    p.add_argument("--n-blocks", type=int, default=3)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    return p.parse_args(argv)


def main() -> int:
    args = parse_args()
    result = run_compare(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        n_estimators=int(args.n_estimators),
        n_blocks=int(args.n_blocks),
        random_state=int(args.random_state),
        ridge_alpha=float(args.ridge_alpha),
    )
    print(result.to_string(index=False))
    print("OUT", args.out_dir / "tables" / "lithotype_encoding_compare.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
