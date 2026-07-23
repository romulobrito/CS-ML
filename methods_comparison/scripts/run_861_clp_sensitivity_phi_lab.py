#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sensitivity and information-ceiling study for CLP phi_lab (Well 861).

Baselines:
  - Phi_ND wireline (no ML)
  - Phi_Neutron wireline
  - RF OOF (depth-block)
  - Linear interpolation of phi_lab at plug depths (OOF)

CLP rf_residual one-factor-at-a-time (OFAT) ablation from default hyperparameters.

Outputs under methods_comparison/data/processed/ml_runs/clp_861/phi_lab/sensitivity/
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from clp_861_plug_fixed_runner import (  # noqa: E402
    PlugFixedRunConfig,
    collect_rf_oof_profiles,
    evaluate_plug_fixed_rf_residual,
    pointwise_fidelity_metrics,
)
from clp_861_protocol import (  # noqa: E402
    load_plug_measurement_rows,
    plug_row_indices_unique,
)
from ml_861_data import CLP_861_ML_ROOT, depth_block_splits  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from auddys_smoke_direct_ub import apply_depth_bounds, load_logs_table  # noqa: E402

DEFAULT_EXCEL = REPO_ROOT / "data" / "Auddys_table.xlsx"
SENSITIVITY_ROOT = CLP_861_ML_ROOT / "phi_lab" / "sensitivity"
PRIMARY_SEED = 7

DEFAULT_CLP: Dict[str, Any] = {
    "window_len": 16,
    "csgm_latent_dim": 16,
    "csgm_hidden_dim": 128,
    "csgm_ae_epochs": 200,
    "csgm_iters": 400,
    "csgm_opt_lr": 0.05,
    "prior_types": ("rf_residual",),
}


def _metrics_row(
    case_id: str,
    family: str,
    knob: str,
    knob_value: str,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Flatten metrics dict into a CSV row."""
    return {
        "case_id": str(case_id),
        "family": str(family),
        "knob": str(knob),
        "knob_value": str(knob_value),
        "rmse": float(metrics.get("rmse", float("nan"))),
        "r2": float(metrics.get("r2", float("nan"))),
        "mae": float(metrics.get("mae", float("nan"))),
        "corr": float(metrics.get("corr", float("nan"))),
        "std_ratio": float(metrics.get("std_ratio", float("nan"))),
        "corr_diff": float(metrics.get("corr_diff", float("nan"))),
        "n_points": int(metrics.get("n_points", 0)),
    }


def load_mogno_phi_table(excel_path: Path) -> pd.DataFrame:
    """Same Logs table and depth window as plug-fixed CLP."""
    df = load_logs_table(str(excel_path), "Logs")
    return apply_depth_bounds(df, 5205.91, 5233.72)


def evaluate_wireline_column(
    df: pd.DataFrame,
    col: str,
    case_id: str,
) -> Dict[str, Any]:
    """Direct wireline log as phi_lab proxy (information ceiling, no CV)."""
    y_true = df["phi_lab"].to_numpy(dtype=np.float64)
    y_pred = df[col].to_numpy(dtype=np.float64)
    depth = df["depth_m"].to_numpy(dtype=np.float64)
    met = pointwise_fidelity_metrics(y_true, y_pred, depth_m=depth)
    return _metrics_row(case_id, "wireline", "column", col, met)


def evaluate_rf_oof(
    df: pd.DataFrame,
    plug_rows: Sequence[int],
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """RF depth-block OOF aligned with CLP protocol."""
    work = df.assign(**{"Depth(m)": df["depth_m"]})
    rf_oof, _ = collect_rf_oof_profiles(work, 3, 200, int(seed), plug_rows)
    y_true = df["phi_lab"].to_numpy(dtype=np.float64)
    depth = df["depth_m"].to_numpy(dtype=np.float64)
    met = pointwise_fidelity_metrics(y_true, rf_oof, depth_m=depth)
    return _metrics_row("rf_oof", "ml_baseline", "model", "rf_depth_block", met)


def evaluate_plug_linear_spline_oof(
    df: pd.DataFrame,
    plug_rows: Sequence[int],
    n_blocks: int = 3,
) -> Dict[str, Any]:
    """
    OOF linear interpolation in depth using phi_lab only at train-fold plugs.
    """
    from scipy.interpolate import interp1d

    plug_set = set(int(r) for r in plug_rows)
    work = df.assign(**{"Depth(m)": df["depth_m"]})
    folds = depth_block_splits(work, n_blocks)
    n = int(df.shape[0])
    oof = np.full(n, np.nan, dtype=np.float64)
    depth_all = df["depth_m"].to_numpy(dtype=np.float64)
    y_all = df["phi_lab"].to_numpy(dtype=np.float64)

    for fold in folds:
        train_set = set(int(i) for i in fold.train_idx)
        test_idx = np.asarray(fold.test_idx, dtype=np.int64)
        plug_train = sorted(
            int(i)
            for i in train_set
            if int(i) in plug_set and np.isfinite(y_all[int(i)])
        )
        if len(plug_train) < 2:
            continue
        d_plug = depth_all[plug_train]
        y_plug = y_all[plug_train]
        order = np.argsort(d_plug)
        d_plug = d_plug[order]
        y_plug = y_plug[order]
        f = interp1d(
            d_plug,
            y_plug,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        oof[test_idx] = f(depth_all[test_idx])

    met = pointwise_fidelity_metrics(y_all, oof, depth_m=depth_all)
    return _metrics_row(
        "plug_linear_oof",
        "mechanical",
        "interpolator",
        "linear_plugs_depth_block",
        met,
    )


def _dummy_run_paths(root: Path) -> Any:
    """Minimal paths object for sensitivity runs (tables only)."""
    from clp_861_protocol import Clp861RunPaths

    root.mkdir(parents=True, exist_ok=True)
    paths = Clp861RunPaths(
        run_root=root,
        tables=root / "tables",
        figures=root / "figures",
        logs=root / "logs",
    )
    paths.ensure_dirs()
    return paths


def run_clp_ofat_case(
    case_id: str,
    knob: str,
    knob_value: Any,
    overrides: Dict[str, Any],
    excel_path: Path,
    out_root: Path,
    seed: int = PRIMARY_SEED,
) -> Dict[str, Any]:
    """Single rf_residual plug-fixed run with hyperparameter overrides."""
    params = dict(DEFAULT_CLP)
    params.update(overrides)
    run_dir = out_root / "runs" / case_id
    cfg = PlugFixedRunConfig(
        excel_path=excel_path,
        run_paths=_dummy_run_paths(run_dir),
        window_len=int(params["window_len"]),
        step=1,
        seeds=(int(seed),),
        prior_types=tuple(params["prior_types"]),
        csgm_latent_dim=int(params["csgm_latent_dim"]),
        csgm_hidden_dim=int(params["csgm_hidden_dim"]),
        csgm_ae_epochs=int(params["csgm_ae_epochs"]),
        csgm_iters=int(params["csgm_iters"]),
        csgm_opt_lr=float(params["csgm_opt_lr"]),
        device=None,
    )
    t0 = time.time()
    met = evaluate_plug_fixed_rf_residual(cfg, primary_seed=int(seed))
    elapsed = time.time() - t0
    row = _metrics_row(case_id, "clp_rf_residual", knob, str(knob_value), met)
    row["elapsed_s"] = float(elapsed)
    row["window_len"] = int(params["window_len"])
    row["csgm_latent_dim"] = int(params["csgm_latent_dim"])
    row["csgm_hidden_dim"] = int(params["csgm_hidden_dim"])
    row["csgm_ae_epochs"] = int(params["csgm_ae_epochs"])
    row["csgm_opt_lr"] = float(params["csgm_opt_lr"])
    return row


def build_ofat_cases() -> List[Tuple[str, str, Any, Dict[str, Any]]]:
    """One-factor-at-a-time cases relative to DEFAULT_CLP."""
    base = dict(DEFAULT_CLP)
    cases: List[Tuple[str, str, Any, Dict[str, Any]]] = [
        ("clp_default", "baseline", "default", base),
    ]
    sweeps: List[Tuple[str, str, List[Any]]] = [
        ("csgm_latent_dim", "latent_dim", [8, 32]),
        ("csgm_hidden_dim", "hidden_dim", [64, 256]),
        ("csgm_ae_epochs", "ae_epochs", [100, 400]),
        ("csgm_opt_lr", "opt_lr", [0.01, 0.1]),
        ("window_len", "window_len", [8, 24]),
    ]
    for key, knob, values in sweeps:
        for val in values:
            over = dict(base)
            over[key] = val
            case_id = "clp_{}_{}".format(knob, val)
            cases.append((case_id, knob, val, over))
    return cases


def run_sensitivity(
    excel_path: Path,
    out_root: Path,
    run_baselines: bool = True,
    run_ablation: bool = True,
    seed: int = PRIMARY_SEED,
) -> pd.DataFrame:
    """Execute full sensitivity battery and write tables."""
    out_root.mkdir(parents=True, exist_ok=True)
    tables_dir = out_root / "tables"
    logs_dir = out_root / "logs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    df = load_mogno_phi_table(excel_path)
    plug_rows = plug_row_indices_unique(load_plug_measurement_rows())
    rows: List[Dict[str, Any]] = []

    if run_baselines:
        rows.append(evaluate_wireline_column(df, "phi_nd", "phi_nd_direct"))
        rows.append(evaluate_wireline_column(df, "phi_neutron", "phi_neutron_direct"))
        rows.append(evaluate_rf_oof(df, plug_rows, seed=int(seed)))
        rows.append(evaluate_plug_linear_spline_oof(df, plug_rows))

    if run_ablation:
        for case_id, knob, val, overrides in build_ofat_cases():
            print("SENSITIVITY_RUN", case_id, flush=True)
            rows.append(
                run_clp_ofat_case(
                    case_id,
                    knob,
                    val,
                    overrides,
                    excel_path,
                    out_root,
                    seed=int(seed),
                )
            )

    result = pd.DataFrame(rows)
    result.to_csv(tables_dir / "sensitivity_all_cases.csv", index=False)

    base_rmse = float("nan")
    sub_clp = result.loc[result["family"] == "clp_rf_residual"].copy()
    if "clp_default" in set(sub_clp["case_id"].tolist()):
        base_rmse = float(
            sub_clp.loc[sub_clp["case_id"] == "clp_default", "rmse"].iloc[0]
        )
    if np.isfinite(base_rmse) and not sub_clp.empty:
        sub_clp["delta_rmse_vs_default"] = sub_clp["rmse"] - base_rmse
        sub_clp.to_csv(tables_dir / "sensitivity_clp_ofat.csv", index=False)

    base_rows = result.loc[result["family"].isin(["wireline", "ml_baseline", "mechanical"])]
    if not base_rows.empty:
        base_rows.to_csv(tables_dir / "sensitivity_baselines.csv", index=False)

    # Preserve prior OFAT/baseline tables when running a partial battery.
    if run_baselines and not run_ablation:
        ofat_path = tables_dir / "sensitivity_clp_ofat.csv"
        if ofat_path.is_file() and int(pd.read_csv(ofat_path).shape[0]) > 0:
            ofat_prev = pd.read_csv(ofat_path)
            merged = pd.concat([base_rows, ofat_prev], ignore_index=True, sort=False)
            merged.to_csv(tables_dir / "sensitivity_all_cases.csv", index=False)
    if run_ablation and not run_baselines:
        base_path = tables_dir / "sensitivity_baselines.csv"
        if base_path.is_file() and int(pd.read_csv(base_path).shape[0]) > 0:
            base_prev = pd.read_csv(base_path)
            merged = pd.concat([base_prev, sub_clp], ignore_index=True, sort=False)
            merged.to_csv(tables_dir / "sensitivity_all_cases.csv", index=False)

    summary = {
        "seed": int(seed),
        "n_cases": int(len(result)),
        "clp_default_rmse": base_rmse,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_root / "sensitivity_meta.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Well 861 CLP phi_lab sensitivity study.")
    p.add_argument("--excel-path", type=Path, default=DEFAULT_EXCEL)
    p.add_argument("--out-root", type=Path, default=SENSITIVITY_ROOT)
    p.add_argument("--seed", type=int, default=PRIMARY_SEED)
    p.add_argument("--baselines-only", action="store_true")
    p.add_argument("--ablation-only", action="store_true")
    return p.parse_args(argv)


def main() -> None:
    """Entry point."""
    args = parse_args()
    run_b = not args.ablation_only
    run_a = not args.baselines_only
    df = run_sensitivity(
        Path(args.excel_path),
        Path(args.out_root),
        run_baselines=run_b,
        run_ablation=run_a,
        seed=int(args.seed),
    )
    print("OK sensitivity n_cases={}".format(int(df.shape[0])))
    print("OUT", Path(args.out_root) / "tables" / "sensitivity_all_cases.csv")


if __name__ == "__main__":
    main()
