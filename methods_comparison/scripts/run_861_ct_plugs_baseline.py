#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CT plugs ML baseline: 10 microCT-integrated samples, leave-one-plug-out CV.

Compares wireline_only vs wireline_plus_ct for Phi_lab and FZI_lab.
Saves artifacts under ml_runs/ct_plugs/.

ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    CT_PLUGS_FEATURE_MODES,
    CT_PLUGS_ML_ROOT,
    CT_PLUGS_TARGETS,
    DEFAULT_CT,
    FeatureMode,
    build_xy,
    ct_plugs_scenario_dir,
    feature_mode_slug,
    load_ct_samples,
    target_slug,
)
from ml_861_metrics import plug_out_predictions_df
from run_861_ml_baseline import model_factory, run_compare

REGRESSOR_CHOICES = ("rf", "gb", "xgb", "mlp", "lr", "all")


def _plot_rf_plug_out(
    pred_df: pd.DataFrame,
    target: str,
    feature_mode: FeatureMode,
    out_path: Path,
) -> None:
    """Pred vs obs for RF leave-one-plug-out."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(pred_df["y_pred"], pred_df["y_true"], color="tab:orange", s=80, alpha=0.9)
    for _, row in pred_df.iterrows():
        ax.annotate(
            str(row["sample_id"]),
            (row["y_pred"], row["y_true"]),
            fontsize=7,
            alpha=0.8,
        )
    lo = min(pred_df["y_true"].min(), pred_df["y_pred"].min())
    hi = max(pred_df["y_true"].max(), pred_df["y_pred"].max()) * 1.05
    if lo == hi:
        hi = lo + 1.0
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.5)
    ax.set_xlabel("predicted")
    ax.set_ylabel("observed")
    ax.set_title(
        "ct_plugs RF LOO: {} / {}".format(target, feature_mode_slug(feature_mode))
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_ct_plugs_scenario(
    target: str,
    feature_mode: FeatureMode,
    regressors: Sequence[str],
    data_path: Path,
    out_dir: Path,
    smoke: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run one ct_plugs scenario (target + feature mode)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = run_compare(
        dataset_mode="ct",
        data_path=data_path,
        target=target,
        feature_mode=feature_mode,
        regressors=regressors,
        out_dir=out_dir,
        smoke=smoke,
        random_state=random_state,
    )

    factory = model_factory(
        "rf",
        smoke=smoke,
        random_state=random_state,
        small_sample=True,
    )
    df = load_ct_samples(data_path)
    bundle = build_xy(df, target=target, feature_mode=feature_mode)
    pred_df = plug_out_predictions_df(factory, bundle)
    pred_df["feature_mode"] = feature_mode_slug(feature_mode)
    pred_path = out_dir / "plug_out_predictions_rf.csv"
    pred_df.to_csv(pred_path, index=False)

    _plot_rf_plug_out(
        pred_df,
        target=target,
        feature_mode=feature_mode,
        out_path=out_dir / "plug_out_pred_vs_obs_rf.png",
    )

    metrics_path = out_dir / "metrics.json"
    if metrics_path.is_file():
        meta = json.loads(metrics_path.read_text(encoding="utf-8"))
        meta["feature_mode"] = feature_mode_slug(feature_mode)
        meta["approach"] = "ct_plugs_leave_one_out"
        meta["plug_out_rf_csv"] = str(pred_path.name)
        metrics_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return summary_df


def build_ct_plugs_summary(out_root: Path) -> pd.DataFrame:
    """Aggregate all ct_plugs scenario CSVs into one comparison table."""
    rows: List[Dict[str, Any]] = []
    by_target = out_root / "by_target"
    if not by_target.is_dir():
        return pd.DataFrame(rows)

    for target_dir in sorted(by_target.iterdir()):
        if not target_dir.is_dir():
            continue
        for mode_dir in sorted(target_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            csv_path = mode_dir / "861_ml_baseline_summary.csv"
            if not csv_path.is_file():
                continue
            part = pd.read_csv(csv_path)
            part["feature_mode"] = mode_dir.name
            part["target_slug"] = target_dir.name
            rows.extend(part.to_dict(orient="records"))

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary_path = out_root / "ct_plugs_scenarios_summary.csv"
        summary.to_csv(summary_path, index=False)

        rf_only = summary[summary["regressor"] == "rf"].copy()
        if not rf_only.empty:
            pivot = rf_only.pivot_table(
                index="target",
                columns="feature_mode",
                values="mean_r2",
                aggfunc="first",
            )
            pivot_path = out_root / "ct_plugs_rf_r2_wireline_vs_ct.csv"
            pivot.to_csv(pivot_path)

    return summary


def write_manifest(out_root: Path) -> None:
    """Write ct_plugs artifact manifest."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = sorted(
        str(p.relative_to(out_root))
        for p in out_root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.txt"
    )
    lines = [
        "Well 861 ct_plugs ML manifest (leave-one-plug-out, 10 samples)",
        "Generated: {}".format(ts),
        "Script: run_861_ct_plugs_baseline.py",
        "CV: leave-one-plug-out",
        "",
        "Artifacts:",
    ]
    lines.extend(["  {}".format(f) for f in files])
    (out_root / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_all_ct_plugs(
    targets: Sequence[str],
    feature_modes: Sequence[FeatureMode],
    regressors: Sequence[str],
    data_path: Path,
    out_root: Path,
    smoke: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run full ct_plugs target x feature-mode grid."""
    out_root.mkdir(parents=True, exist_ok=True)

    for target in targets:
        for feature_mode in feature_modes:
            scenario_dir = ct_plugs_scenario_dir(target, feature_mode, base=out_root)
            print(
                "ct_plugs target={} features={} -> {}".format(
                    target, feature_mode_slug(feature_mode), scenario_dir
                )
            )
            run_ct_plugs_scenario(
                target=target,
                feature_mode=feature_mode,
                regressors=regressors,
                data_path=data_path,
                out_dir=scenario_dir,
                smoke=smoke,
                random_state=random_state,
            )

    summary = build_ct_plugs_summary(out_root)
    write_manifest(out_root)
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Well 861 ct_plugs ML baseline (leave-one-plug-out)"
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_CT)
    parser.add_argument("--out-dir", type=Path, default=CT_PLUGS_ML_ROOT)
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Single target (default: all ct_plugs targets)",
    )
    parser.add_argument(
        "--feature-mode",
        choices=CT_PLUGS_FEATURE_MODES,
        default=None,
        help="Single feature mode (default: wireline_only + wireline_plus_ct)",
    )
    parser.add_argument("--regressor", choices=REGRESSOR_CHOICES, default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.regressor == "all":
        regs = ["rf", "gb", "xgb", "mlp", "lr"]
    else:
        regs = [args.regressor]

    targets = [args.target] if args.target else list(CT_PLUGS_TARGETS)
    modes: List[FeatureMode] = (
        [args.feature_mode] if args.feature_mode else list(CT_PLUGS_FEATURE_MODES)
    )

    summary = run_all_ct_plugs(
        targets=targets,
        feature_modes=modes,
        regressors=regs,
        data_path=args.data_path.resolve(),
        out_root=args.out_dir.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
    )

    if not summary.empty:
        rf_view = summary[summary["regressor"] == "rf"][
            ["target", "feature_mode", "mean_rmse", "mean_r2", "protocol"]
        ]
        print(rf_view.to_string(index=False))
    print("OK ct_plugs complete out_dir={}".format(args.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
