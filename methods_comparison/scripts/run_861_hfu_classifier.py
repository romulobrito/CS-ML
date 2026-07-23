#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HFU classification on Well 861 well_profile (depth-block CV).

Models: RandomForestClassifier, LogisticRegression (multinomial).
Metrics: accuracy, balanced accuracy, F1 macro (OOF global + per fold).

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    DEFAULT_ENRICHED,
    WELL_PROFILE_HFU_TARGET,
    build_xy,
    load_logs_enriched,
    well_profile_hfu_classifier_dir,
)
from ml_861_metrics import (
    ClassCvSummary,
    classification_oof_predictions,
    evaluate_depth_blocks_classification,
)

ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUT = well_profile_hfu_classifier_dir()


def _rf_classifier_factory(n_estimators: int, random_state: int) -> Callable[[], Any]:
    return lambda: RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced",
    )


def _logistic_factory(random_state: int) -> Callable[[], Any]:
    return lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def _class_row(model_name: str, cv: ClassCvSummary, smoke: bool) -> Dict[str, Any]:
    return {
        "target": cv.target,
        "model": model_name,
        "protocol": cv.protocol,
        "mean_accuracy": cv.mean_accuracy,
        "mean_balanced_accuracy": cv.mean_balanced_accuracy,
        "mean_f1_macro": cv.mean_f1_macro,
        "global_oof_accuracy": cv.global_oof_accuracy,
        "global_oof_balanced_accuracy": cv.global_oof_balanced_accuracy,
        "global_oof_f1_macro": cv.global_oof_f1_macro,
        "smoke": smoke,
        "approach": "well_profile_depth_block_cv",
    }


def _plot_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[int],
    title: str,
    out_path: Path,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_model_comparison(summary_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    metrics = ["global_oof_accuracy", "global_oof_balanced_accuracy", "global_oof_f1_macro"]
    x = np.arange(len(summary_df))
    width = 0.25
    for i, metric in enumerate(metrics):
        offset = (i - 1) * width
        ax.bar(x + offset, summary_df[metric], width, label=metric.replace("global_oof_", "oof_"))
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("HFU classification well_profile (depth-block CV OOF)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_hfu_classifiers(
    data_path: Path,
    out_dir: Path,
    smoke: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run HFU classifiers with depth-block CV."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_blocks = 2 if smoke else 3
    n_estimators = 10 if smoke else 200
    target = WELL_PROFILE_HFU_TARGET

    df = load_logs_enriched(data_path)
    bundle = build_xy(df, target=target, feature_mode="log_only")

    rows: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {
        "well_id": "861",
        "approach": "well_profile_depth_block_cv",
        "target": target,
        "smoke": smoke,
        "n_rows": len(bundle.df),
        "feature_names": bundle.feature_names,
        "class_distribution": df[target].value_counts().sort_index().to_dict(),
        "models": {},
    }

    configs = [
        ("rf_balanced", _rf_classifier_factory(n_estimators, random_state)),
        ("logistic_multinomial", _logistic_factory(random_state)),
    ]

    for model_name, factory in configs:
        cv = evaluate_depth_blocks_classification(factory, bundle, n_blocks=n_blocks)
        cv.model_name = model_name
        rows.append(_class_row(model_name, cv, smoke))
        details["models"][model_name] = cv.to_dict()

        oof_df = classification_oof_predictions(factory, bundle, n_blocks=n_blocks)
        oof_df["model"] = model_name
        oof_path = out_dir / "oof_predictions_{}.csv".format(model_name)
        oof_df.to_csv(oof_path, index=False)

        labels = cv.class_labels if cv.class_labels is not None else []
        if labels:
            _plot_confusion(
                oof_df["y_true"].to_numpy(dtype=int),
                oof_df["y_pred"].to_numpy(dtype=int),
                labels=labels,
                title="HFU OOF confusion: {}".format(model_name),
                out_path=out_dir / "confusion_matrix_oof_{}.png".format(model_name),
            )

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "861_hfu_model_comparison.csv", index=False)
    (out_dir / "metrics.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    _plot_model_comparison(summary_df, out_dir / "hfu_model_comparison_oof.png")

    return summary_df


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HFU classifier well_profile depth-block CV")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_ENRICHED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = run_hfu_classifiers(
        data_path=args.data_path.resolve(),
        out_dir=args.out_dir.resolve(),
        smoke=args.smoke,
        random_state=args.random_state,
    )
    print(
        summary[
            [
                "model",
                "global_oof_accuracy",
                "global_oof_balanced_accuracy",
                "global_oof_f1_macro",
            ]
        ].to_string(index=False)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
