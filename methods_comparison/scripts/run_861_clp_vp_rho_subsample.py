#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3c: rho subsample sweep for CLP vs RF sparse (Vp residual, Well 861).

Planning: methods_comparison/planning/etapa3c_vp_rho_subsample_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LATEX_FIGURES = SCRIPT_DIR.parent / "latex" / "figures"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clp_861_protocol import plug_row_indices_unique, load_plug_measurement_rows  # noqa: E402
from clp_861_vp_residual import VpClpRunConfig  # noqa: E402
from clp_861_vp_rho_subsample import (  # noqa: E402
    DEFAULT_RHOS,
    METHOD_CLP_SPARSE,
    METHOD_GASSMANN,
    METHOD_RF_ORACLE,
    METHOD_RF_SPARSE,
    aggregate_rho_table,
    run_rho_sweep,
    smoke_rho_config,
)
from ml_861_data import (  # noqa: E402
    CLP_861_VP_RHO_ROOT,
    DLIS_GASSMANN_VALIDATION_CSV,
    RESIDUAL_VP_TARGET,
    build_residual_feature_columns,
    build_xy_from_columns,
)
from run_861_ml_residual import (  # noqa: E402
    build_residual_dataset,
    utc_now_iso,
)


def parse_rhos(text: str) -> Tuple[float, ...]:
    """Parse comma-separated rho values."""
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("Empty --rhos list.")
    return tuple(float(p) for p in parts)


def plot_rho_sweep_mape(
    agg: pd.DataFrame,
    references: pd.DataFrame,
    out_path: Path,
) -> None:
    """MAPE vs rho for CLP sparse b and RF sparse."""
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    clp = agg[agg["method"] == METHOD_CLP_SPARSE].sort_values("rho")
    rf = agg[agg["method"] == METHOD_RF_SPARSE].sort_values("rho")
    if not clp.empty:
        ax.plot(
            clp["rho"],
            clp["mape_vp_pct"],
            "o-",
            color="#ff7f0e",
            label="CLP sparse b",
            linewidth=1.5,
            markersize=5,
        )
    if not rf.empty:
        ax.plot(
            rf["rho"],
            rf["mape_vp_pct"],
            "s-",
            color="#d62728",
            label="RF sparse (train on cal only)",
            linewidth=1.5,
            markersize=5,
        )
    ref_oracle = references[references["method"] == METHOD_RF_ORACLE]
    ref_gass = references[references["method"] == METHOD_GASSMANN]
    if not ref_oracle.empty:
        ax.axhline(
            float(ref_oracle["mape_vp_pct"].iloc[0]),
            color="#2ca02c",
            linestyle="--",
            linewidth=1.2,
            label="RF oracle (full train)",
        )
    if not ref_gass.empty:
        ax.axhline(
            float(ref_gass["mape_vp_pct"].iloc[0]),
            color="#1f77b4",
            linestyle=":",
            linewidth=1.2,
            label="Gassmann",
        )
    ax.set_xlabel("rho (fraction of calibration depths per fold)")
    ax.set_ylabel("MAPE Vp vs sonic (%)")
    ax.set_title("Well 861: sparse calibration sweep (depth-block OOF)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_rho_delta_mape(
    agg: pd.DataFrame,
    references: pd.DataFrame,
    out_path: Path,
) -> None:
    """Delta MAPE vs RF oracle (CLP - oracle, RF sparse - oracle)."""
    ref_oracle = references[references["method"] == METHOD_RF_ORACLE]
    if ref_oracle.empty:
        return
    oracle_mape = float(ref_oracle["mape_vp_pct"].iloc[0])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for method, color, marker in (
        (METHOD_CLP_SPARSE, "#ff7f0e", "o"),
        (METHOD_RF_SPARSE, "#d62728", "s"),
    ):
        sub = agg[agg["method"] == method].sort_values("rho")
        if sub.empty:
            continue
        delta = sub["mape_vp_pct"].to_numpy(dtype=np.float64) - oracle_mape
        ax.plot(
            sub["rho"],
            delta,
            marker + "-",
            color=color,
            label="{} vs oracle".format(method),
            linewidth=1.5,
            markersize=5,
        )
    ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
    ax.set_xlabel("rho")
    ax.set_ylabel("MAPE - RF oracle (pp)")
    ax.set_title("Gap to RF oracle (negative = better than oracle)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def find_crossover_rho(agg: pd.DataFrame, references: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Approximate rho where method MAPE crosses RF oracle."""
    ref = references[references["method"] == METHOD_RF_ORACLE]
    if ref.empty:
        return {}
    oracle = float(ref["mape_vp_pct"].iloc[0])
    out: Dict[str, Optional[float]] = {}
    for method in (METHOD_CLP_SPARSE, METHOD_RF_SPARSE):
        sub = agg[agg["method"] == method].sort_values("rho")
        if sub.empty:
            continue
        mape = sub["mape_vp_pct"].to_numpy(dtype=np.float64)
        rhos = sub["rho"].to_numpy(dtype=np.float64)
        better = mape <= oracle
        if bool(np.all(better)):
            out[method] = 1.0
        elif bool(np.any(better)):
            idx = int(np.where(better)[0][0])
            out[method] = float(rhos[idx])
        else:
            out[method] = None
    return out


def copy_latex_figures(figures_dir: Path) -> Dict[str, Path]:
    """Copy rho sweep figures to latex/figures."""
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    mapping = {
        "rho_sweep_mape.png": "fig3_rho_sweep_mape.png",
        "rho_delta_vs_oracle.png": "fig3_rho_delta_vs_oracle.png",
    }
    copied: Dict[str, Path] = {}
    for src_name, dst_name in mapping.items():
        src = figures_dir / src_name
        if not src.is_file():
            continue
        dst = LATEX_FIGURES / dst_name
        shutil.copy2(src, dst)
        copied[src_name] = dst
    return copied


def run_rho_pipeline(
    validation_csv: Path,
    out_root: Path,
    cfg_run: VpClpRunConfig,
    rhos: Sequence[float],
    rf_n_estimators: int,
    n_repeats: int,
    smoke: bool,
) -> Dict[str, object]:
    """Execute rho sweep and write artifacts."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    for d in (tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    dataset = build_residual_dataset(validation_csv)
    feature_cols = build_residual_feature_columns(dataset)
    bundle = build_xy_from_columns(
        dataset,
        target=RESIDUAL_VP_TARGET,
        feature_columns=feature_cols,
    )

    plug_rows = plug_row_indices_unique(load_plug_measurement_rows())
    metrics_df, oof_store = run_rho_sweep(
        dataset,
        bundle,
        cfg_run,
        rhos=rhos,
        rf_n_estimators=int(rf_n_estimators),
        n_repeats=int(n_repeats),
        plug_rows=plug_rows,
    )
    metrics_df.to_csv(tables_dir / "rho_sweep_metrics.csv", index=False, float_format="%.6f")
    agg = aggregate_rho_table(metrics_df)
    agg.to_csv(tables_dir / "rho_sweep_aggregated.csv", index=False, float_format="%.6f")

    refs = metrics_df[metrics_df["method"].isin((METHOD_RF_ORACLE, METHOD_GASSMANN))].copy()
    refs.to_csv(tables_dir / "rho_references.csv", index=False, float_format="%.6f")

    plot_rho_sweep_mape(agg, refs, figures_dir / "rho_sweep_mape.png")
    plot_rho_delta_mape(agg, refs, figures_dir / "rho_delta_vs_oracle.png")
    latex_copied = copy_latex_figures(figures_dir)

    crossover = find_crossover_rho(agg, refs)
    metrics: Dict[str, object] = {
        "well_id": "861",
        "approach": "vp_residual_rho_subsample",
        "rhos": list(rhos),
        "n_repeats": int(n_repeats),
        "crossover_rho_at_oracle": crossover,
        "references": refs.to_dict(orient="records"),
        "aggregated": agg.to_dict(orient="records"),
        "smoke": bool(smoke),
        "generated_utc": utc_now_iso(),
    }
    (out_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = [
        "Well 861 -- Etapa 3c Vp residual rho subsample",
        "Generated: {}".format(utc_now_iso()),
        "Rhos: {}".format(", ".join("{:.3f}".format(r) for r in rhos)),
        "Crossover vs RF oracle: {}".format(crossover),
        "LaTeX figures: {}".format(len(latex_copied)),
    ]
    (out_root / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Well 861 Etapa 3c: rho subsample CLP vs RF sparse")
    p.add_argument("--validation-csv", type=Path, default=DLIS_GASSMANN_VALIDATION_CSV)
    p.add_argument("--out-root", type=Path, default=CLP_861_VP_RHO_ROOT)
    p.add_argument("--rhos", type=str, default=",".join("{:.3f}".format(r) for r in DEFAULT_RHOS))
    p.add_argument("--n-repeats", type=int, default=1)
    p.add_argument("--rf-n-estimators", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--window-len", type=int, default=16)
    p.add_argument("--csgm-ae-epochs", type=int, default=200)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    rhos = parse_rhos(args.rhos)
    if args.smoke:
        cfg_run = smoke_rho_config()
        rhos = (0.0, 0.3, 1.0)
    else:
        cfg_run = VpClpRunConfig(
            window_len=int(args.window_len),
            seed=int(args.seed),
            csgm_ae_epochs=int(args.csgm_ae_epochs),
            methods=(),
        )
    metrics = run_rho_pipeline(
        validation_csv=args.validation_csv.resolve(),
        out_root=args.out_root.resolve(),
        cfg_run=cfg_run,
        rhos=rhos,
        rf_n_estimators=int(args.rf_n_estimators),
        n_repeats=int(args.n_repeats),
        smoke=bool(args.smoke),
    )
    print("OK vp_rho_subsample smoke={}".format(metrics["smoke"]))
    print("Crossover:", metrics["crossover_rho_at_oracle"])
    for row in metrics["aggregated"]:
        print(
            "  rho={:.3f} {} MAPE={:.1f}%".format(
                row["rho"],
                row["method"],
                row["mape_vp_pct"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
