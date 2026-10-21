#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3b: CLP-CSGM residual Vp after Gassmann (Well 861).

Compares CLP window methods vs RF pointwise OOF (Etapa 3).

Planning: methods_comparison/planning/etapa3b_clp_csgm_vp_residual_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LATEX_FIGURES = SCRIPT_DIR.parent / "latex" / "figures"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clp_861_vp_residual import (  # noqa: E402
    VP_CLP_METHOD_PLUG_SPARSE,
    VP_CLP_METHOD_RIDGE_PRIOR,
    VP_CLP_METHOD_ZERO_RESIDUAL,
    VpClpRunConfig,
    run_vp_clp_depth_block_cv,
    smoke_vp_clp_config,
)
from ml_861_data import (  # noqa: E402
    CLP_861_VP_RESIDUAL_ROOT,
    DEPTH_COL,
    DLIS_GASSMANN_VALIDATION_CSV,
    ML_RESIDUAL_VP_ROOT,
)
from run_861_ml_residual import (  # noqa: E402
    HFU_COL,
    build_residual_dataset,
    plot_residual_oof_scatter,
    plot_vp_depth_tracks,
    summary_by_hfu_vp,
    utc_now_iso,
    vp_metrics_vs_sonic,
)

METHOD_LABELS: Dict[str, str] = {
    "gassmann_physics": "Gassmann",
    "hybrid_rf_oof": "RF hybrid OOF",
    VP_CLP_METHOD_RIDGE_PRIOR: "CLP Ridge prior (m=0)",
    VP_CLP_METHOD_ZERO_RESIDUAL: "CLP zero residual (m=0)",
    VP_CLP_METHOD_PLUG_SPARSE: "CLP plug sparse b",
}

METHOD_COLORS: Dict[str, str] = {
    "gassmann_physics": "#1f77b4",
    "hybrid_rf_oof": "#d62728",
    VP_CLP_METHOD_RIDGE_PRIOR: "#ff7f0e",
    VP_CLP_METHOD_ZERO_RESIDUAL: "#9467bd",
    VP_CLP_METHOD_PLUG_SPARSE: "#8c564b",
}


def load_rf_oof_predictions(residual_root: Path) -> Optional[pd.DataFrame]:
    """Load RF OOF table from Etapa 3 if available."""
    path = residual_root / "tables" / "oof_predictions.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


def build_comparison_table(
    work: pd.DataFrame,
    vp_cols: Dict[str, str],
) -> pd.DataFrame:
    """Global Vp metrics vs sonic for each model column."""
    rows: List[dict] = []
    for model, col in vp_cols.items():
        m = vp_metrics_vs_sonic(
            work[col].to_numpy(dtype=np.float64),
            work["vp_sonic_km_s"].to_numpy(dtype=np.float64),
        )
        rows.append({"model": model, **m.to_dict()})
    return pd.DataFrame(rows)


def build_hfu_comparison_table(work: pd.DataFrame, vp_cols: Dict[str, str]) -> pd.DataFrame:
    """HFU breakdown for each model."""
    parts: List[pd.DataFrame] = []
    for model, col in vp_cols.items():
        sub = summary_by_hfu_vp(work, col)
        sub["model"] = model
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def plot_mape_comparison_bar(
    comparison: pd.DataFrame,
    out_path: Path,
    title: str = "Well 861: Vp MAPE vs sonic (OOF)",
) -> None:
    """Bar chart of MAPE by model."""
    work = comparison.sort_values("mape_vp_pct")
    labels = [METHOD_LABELS.get(m, m) for m in work["model"].tolist()]
    colors = [METHOD_COLORS.get(m, "#333333") for m in work["model"].tolist()]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    x = np.arange(len(labels))
    ax.bar(x, work["mape_vp_pct"].to_numpy(dtype=np.float64), color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("MAPE Vp (%)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_depth_tracks_multi(
    df: pd.DataFrame,
    hybrid_cols: Dict[str, str],
    out_path: Path,
    title: str,
) -> None:
    """Depth track with sonic, Gassmann and multiple hybrid curves."""
    work = df.sort_values(DEPTH_COL)
    fig, ax = plt.subplots(figsize=(7.0, 8.0))
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
        color=METHOD_COLORS["gassmann_physics"],
        label="Vp Gassmann",
        markersize=3,
        linewidth=1.0,
    )
    markers = ["d", "^", "v", "P"]
    for i, (model, col) in enumerate(hybrid_cols.items()):
        if col not in work.columns:
            continue
        ax.plot(
            work[col],
            work[DEPTH_COL],
            markers[i % len(markers)] + "-",
            color=METHOD_COLORS.get(model, "#333333"),
            label=METHOD_LABELS.get(model, model),
            markersize=3,
            linewidth=1.1,
        )
    ax.set_xlabel("Vp (km/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="best", fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def copy_latex_figures(figures_dir: Path) -> Dict[str, Path]:
    """Copy pipeline figures to latex/figures with fig3_clp_* names."""
    LATEX_FIGURES.mkdir(parents=True, exist_ok=True)
    mapping = {
        "comparison_mape_bar.png": "fig3_clp_comparison_mape.png",
        "vp_depth_rf_vs_clp.png": "fig3_clp_vp_hybrid_depth.png",
        "clp_ridge_residual_scatter.png": "fig3_clp_residual_scatter.png",
        "vp_depth_three_methods.png": "fig3_clp_three_methods_depth.png",
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


def run_clp_vp_residual_pipeline(
    validation_csv: Path,
    out_root: Path,
    residual_root: Path,
    cfg_run: VpClpRunConfig,
    smoke: bool = False,
) -> Dict[str, object]:
    """Run CLP Vp residual CV, compare with RF, export artifacts."""
    tables_dir = out_root / "tables"
    figures_dir = out_root / "figures"
    for d in (tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    dataset = build_residual_dataset(validation_csv)
    dataset.to_csv(tables_dir / "residual_dataset.csv", index=False, float_format="%.6f")

    cv_result = run_vp_clp_depth_block_cv(dataset, cfg_run)
    work = dataset.copy()
    vp_cols: Dict[str, str] = {"gassmann_physics": "vp_gassmann_km_s"}

    rf_oof = load_rf_oof_predictions(residual_root)
    if rf_oof is not None and "vp_hybrid_oof_km_s" in rf_oof.columns:
        work = work.merge(
            rf_oof[[DEPTH_COL, "vp_hybrid_oof_km_s", "residual_pred_oof_km_s"]],
            on=DEPTH_COL,
            how="left",
        )
        vp_cols["hybrid_rf_oof"] = "vp_hybrid_oof_km_s"
    else:
        work["vp_hybrid_oof_km_s"] = np.nan
        work["residual_pred_oof_km_s"] = np.nan

    for method in cv_result.methods:
        delta_col = "delta_oof_{}".format(method)
        vp_col = "vp_hybrid_{}_km_s".format(method)
        work[delta_col] = cv_result.delta_oof[method]
        work[vp_col] = work["vp_gassmann_km_s"] + work[delta_col]
        vp_cols[method] = vp_col

    comparison = build_comparison_table(work, vp_cols)
    comparison.to_csv(
        tables_dir / "comparison_gassmann_rf_clp.csv",
        index=False,
        float_format="%.6f",
    )

    by_hfu = build_hfu_comparison_table(work, vp_cols)
    by_hfu.to_csv(tables_dir / "summary_by_hfu_clp.csv", index=False, float_format="%.6f")

    oof_export_cols = [DEPTH_COL, HFU_COL, "vp_gassmann_km_s", "vp_sonic_km_s", "vp_residual_km_s"]
    for method in cv_result.methods:
        oof_export_cols.extend(
            [
                "delta_oof_{}".format(method),
                "vp_hybrid_{}_km_s".format(method),
            ]
        )
    if "residual_pred_oof_km_s" in work.columns:
        oof_export_cols.append("residual_pred_oof_km_s")
        oof_export_cols.append("vp_hybrid_oof_km_s")
    work[oof_export_cols].to_csv(
        tables_dir / "oof_predictions_clp.csv",
        index=False,
        float_format="%.6f",
    )

    fold_rows: List[dict] = []
    for fr in cv_result.fold_results:
        row: dict = {
            "fold_id": fr.fold_id,
            "depth_min_m": fr.depth_min_m,
            "depth_max_m": fr.depth_max_m,
            "n_test": int(len(fr.test_idx)),
        }
        for method, lam in fr.lambda_by_method.items():
            row["lambda_{}".format(method)] = lam
        fold_rows.append(row)
    pd.DataFrame(fold_rows).to_csv(
        tables_dir / "fold_summary.csv",
        index=False,
        float_format="%.6f",
    )

    plot_mape_comparison_bar(
        comparison,
        figures_dir / "comparison_mape_bar.png",
    )

    hybrid_plot_cols: Dict[str, str] = {}
    if "hybrid_rf_oof" in vp_cols:
        hybrid_plot_cols["hybrid_rf_oof"] = vp_cols["hybrid_rf_oof"]
    if VP_CLP_METHOD_RIDGE_PRIOR in vp_cols:
        hybrid_plot_cols[VP_CLP_METHOD_RIDGE_PRIOR] = vp_cols[VP_CLP_METHOD_RIDGE_PRIOR]

    if hybrid_plot_cols:
        plot_depth_tracks_multi(
            work,
            hybrid_plot_cols,
            figures_dir / "vp_depth_rf_vs_clp.png",
            "Well 861: RF vs CLP Ridge hybrid vs sonic (OOF)",
        )

    three_cols: Dict[str, str] = {}
    if "hybrid_rf_oof" in vp_cols:
        three_cols["hybrid_rf_oof"] = vp_cols["hybrid_rf_oof"]
    if VP_CLP_METHOD_RIDGE_PRIOR in vp_cols:
        three_cols[VP_CLP_METHOD_RIDGE_PRIOR] = vp_cols[VP_CLP_METHOD_RIDGE_PRIOR]
    if VP_CLP_METHOD_PLUG_SPARSE in vp_cols:
        three_cols[VP_CLP_METHOD_PLUG_SPARSE] = vp_cols[VP_CLP_METHOD_PLUG_SPARSE]
    if three_cols:
        plot_depth_tracks_multi(
            work,
            three_cols,
            figures_dir / "vp_depth_three_methods.png",
            "Well 861: Gassmann + RF + CLP hybrids vs sonic (OOF)",
        )

    if VP_CLP_METHOD_RIDGE_PRIOR in cv_result.methods:
        delta_col = "delta_oof_{}".format(VP_CLP_METHOD_RIDGE_PRIOR)
        plot_residual_oof_scatter(
            work["vp_residual_km_s"].to_numpy(dtype=np.float64),
            work[delta_col].to_numpy(dtype=np.float64),
            figures_dir / "clp_ridge_residual_scatter.png",
            "CLP Ridge prior: residual OOF vs observed (depth-block CV)",
        )
        plot_vp_depth_tracks(
            work.assign(vp_hybrid_clp_oof_km_s=work[vp_cols[VP_CLP_METHOD_RIDGE_PRIOR]]),
            figures_dir / "vp_clp_ridge_depth_track.png",
            hybrid_col="vp_hybrid_clp_oof_km_s",
            hybrid_label="Vp hybrid OOF (CLP Ridge)",
            hybrid_color=METHOD_COLORS[VP_CLP_METHOD_RIDGE_PRIOR],
            title="Well 861: CLP Ridge hybrid vs Gassmann vs sonic (OOF)",
        )

    latex_copied = copy_latex_figures(figures_dir)

    metrics_by_model = {
        row["model"]: {
            "mape_vp_pct": float(row["mape_vp_pct"]),
            "rmse_vp_km_s": float(row["rmse_vp_km_s"]),
            "bias_vp_km_s": float(row["bias_vp_km_s"]),
        }
        for _, row in comparison.iterrows()
    }

    metrics: Dict[str, object] = {
        "well_id": "861",
        "approach": "clp_csgm_vp_residual_depth_block_cv",
        "target": "vp_residual_km_s",
        "n_rows": int(len(work)),
        "window_len": int(cv_result.window_len),
        "methods": list(cv_result.methods),
        "selected_lambda": cv_result.selected_lambda,
        "metrics_by_model": metrics_by_model,
        "smoke": smoke,
        "generated_utc": utc_now_iso(),
    }
    (out_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = [
        "Well 861 -- Etapa 3b CLP-CSGM Vp residual",
        "Generated: {}".format(utc_now_iso()),
        "Rows: {}".format(len(work)),
        "Methods: {}".format(", ".join(cv_result.methods)),
        "Comparison: tables/comparison_gassmann_rf_clp.csv",
        "LaTeX figures copied: {}".format(len(latex_copied)),
    ]
    for model, m in metrics_by_model.items():
        manifest.append(
            "  {} MAPE={:.1f}% bias={:+.3f} km/s".format(
                model,
                m["mape_vp_pct"],
                m["bias_vp_km_s"],
            )
        )
    (out_root / "MANIFEST.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")

    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI."""
    p = argparse.ArgumentParser(description="Well 861 Etapa 3b: CLP-CSGM Vp residual")
    p.add_argument("--validation-csv", type=Path, default=DLIS_GASSMANN_VALIDATION_CSV)
    p.add_argument("--out-root", type=Path, default=CLP_861_VP_RESIDUAL_ROOT)
    p.add_argument(
        "--residual-root",
        type=Path,
        default=ML_RESIDUAL_VP_ROOT,
        help="Etapa 3 RF outputs (for OOF comparison)",
    )
    p.add_argument("--window-len", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-depth-blocks", type=int, default=3)
    p.add_argument("--csgm-ae-epochs", type=int, default=200)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    args = parse_args(argv)
    if args.smoke:
        cfg_run = smoke_vp_clp_config()
    else:
        cfg_run = VpClpRunConfig(
            window_len=int(args.window_len),
            seed=int(args.seed),
            n_depth_blocks=int(args.n_depth_blocks),
            csgm_ae_epochs=int(args.csgm_ae_epochs),
            device=args.device,
        )
    metrics = run_clp_vp_residual_pipeline(
        validation_csv=args.validation_csv.resolve(),
        out_root=args.out_root.resolve(),
        residual_root=args.residual_root.resolve(),
        cfg_run=cfg_run,
        smoke=bool(args.smoke),
    )
    print("OK clp_vp_residual smoke={}".format(metrics["smoke"]))
    for model, m in metrics["metrics_by_model"].items():
        print(
            "  {} MAPE={:.1f}% bias={:+.3f} km/s".format(
                model,
                m["mape_vp_pct"],
                m["bias_vp_km_s"],
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
