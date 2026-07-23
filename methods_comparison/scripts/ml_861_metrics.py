#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation metrics and cross-validation helpers for Well 861 ML baseline.

ASCII-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ml_861_data import (
    DEPTH_COL,
    XYBundle,
    depth_block_splits,
    iter_fold_arrays,
    leave_one_plug_out_splits,
    load_ct_samples,
    load_logs_enriched,
    build_xy,
)


@dataclass
class FoldMetrics:
    """Metrics for a single CV fold."""

    fold_id: int
    rmse: float
    r2: float
    n_train: int
    n_test: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CvSummary:
    """Aggregated cross-validation results."""

    target: str
    model_name: str
    folds: List[FoldMetrics]
    mean_rmse: float
    std_rmse: float
    mean_r2: float
    std_r2: float
    protocol: str
    global_oof_rmse: Optional[float] = None
    global_oof_r2: Optional[float] = None
    global_oof_mae: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "target": self.target,
            "model_name": self.model_name,
            "protocol": self.protocol,
            "mean_rmse": self.mean_rmse,
            "std_rmse": self.std_rmse,
            "mean_r2": self.mean_r2,
            "std_r2": self.std_r2,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "rmse": f.rmse,
                    "r2": f.r2,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    **f.extra,
                }
                for f in self.folds
            ],
        }
        if self.global_oof_rmse is not None:
            out["global_oof_rmse"] = self.global_oof_rmse
        if self.global_oof_r2 is not None:
            out["global_oof_r2"] = self.global_oof_r2
        if self.global_oof_mae is not None:
            out["global_oof_mae"] = self.global_oof_mae
        return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_depth_blocks(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
    n_blocks: int = 3,
) -> CvSummary:
    """Depth-block CV on enriched well profile."""
    folds_def = depth_block_splits(bundle.df, n_blocks=n_blocks)
    fold_metrics: List[FoldMetrics] = []

    for fold_id, x_tr, y_tr, x_te, y_te in iter_fold_arrays(bundle, folds_def):
        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        fold_metrics.append(
            FoldMetrics(
                fold_id=fold_id,
                rmse=_rmse(y_te, pred),
                r2=float(r2_score(y_te, pred)),
                n_train=len(y_tr),
                n_test=len(y_te),
                extra={
                    "depth_min_m": folds_def[fold_id].depth_min_m,
                    "depth_max_m": folds_def[fold_id].depth_max_m,
                },
            )
        )

    rmse_vals = [f.rmse for f in fold_metrics]
    r2_vals = [f.r2 for f in fold_metrics]
    return CvSummary(
        target=bundle.target,
        model_name="",
        folds=fold_metrics,
        mean_rmse=float(np.mean(rmse_vals)),
        std_rmse=float(np.std(rmse_vals)),
        mean_r2=float(np.mean(r2_vals)),
        std_r2=float(np.std(r2_vals)),
        protocol="depth_block_{}".format(n_blocks),
    )


def collect_depth_block_oof(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
    n_blocks: int = 3,
) -> tuple[np.ndarray, List[FoldMetrics]]:
    """
    Out-of-fold predictions aligned to bundle row order (depth-block CV).

    Returns (oof_pred, fold_metrics) where oof_pred[i] is NaN if row i was
    never in a test fold.
    """
    folds_def = depth_block_splits(bundle.df, n_blocks=n_blocks)
    n = len(bundle.y)
    oof_pred = np.full(n, np.nan, dtype=np.float64)
    fold_metrics: List[FoldMetrics] = []

    for fold_id, x_tr, y_tr, x_te, y_te in iter_fold_arrays(bundle, folds_def):
        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        test_idx = folds_def[fold_id].test_idx
        oof_pred[test_idx] = pred
        fold_metrics.append(
            FoldMetrics(
                fold_id=fold_id,
                rmse=_rmse(y_te, pred),
                r2=float(r2_score(y_te, pred)),
                n_train=len(y_tr),
                n_test=len(y_te),
                extra={
                    "depth_min_m": folds_def[fold_id].depth_min_m,
                    "depth_max_m": folds_def[fold_id].depth_max_m,
                },
            )
        )
    return oof_pred, fold_metrics


def evaluate_plug_out(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
) -> CvSummary:
    """Leave-one-plug-out CV on CT sample table."""
    folds_def = leave_one_plug_out_splits(bundle.df)
    fold_metrics: List[FoldMetrics] = []
    oof_pred = np.full(len(bundle.y), np.nan, dtype=np.float64)

    for fold_id, (train_idx, test_idx, sample_id) in enumerate(folds_def):
        x_tr, y_tr = bundle.X[train_idx], bundle.y[train_idx]
        x_te, y_te = bundle.X[test_idx], bundle.y[test_idx]
        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = model.predict(x_te)
        oof_pred[test_idx] = pred
        abs_err = float(np.abs(y_te[0] - pred[0]))
        fold_metrics.append(
            FoldMetrics(
                fold_id=fold_id,
                rmse=abs_err,
                r2=float("nan"),
                n_train=len(y_tr),
                n_test=len(y_te),
                extra={
                    "held_out_sample": sample_id,
                    "y_true": float(y_te[0]),
                    "y_pred": float(pred[0]),
                    "abs_error": abs_err,
                },
            )
        )

    rmse_vals = [f.rmse for f in fold_metrics]
    valid_oof = np.isfinite(oof_pred)
    global_rmse = _rmse(bundle.y[valid_oof], oof_pred[valid_oof])
    global_r2 = float(r2_score(bundle.y[valid_oof], oof_pred[valid_oof]))
    global_mae = float(mean_absolute_error(bundle.y[valid_oof], oof_pred[valid_oof]))

    return CvSummary(
        target=bundle.target,
        model_name="",
        folds=fold_metrics,
        mean_rmse=float(np.mean(rmse_vals)),
        std_rmse=float(np.std(rmse_vals)),
        mean_r2=global_r2,
        std_r2=float("nan"),
        protocol="leave_one_plug_out",
        global_oof_rmse=global_rmse,
        global_oof_r2=global_r2,
        global_oof_mae=global_mae,
    )


def plug_out_predictions_df(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
) -> "pd.DataFrame":
    """Leave-one-plug-out predictions as a DataFrame."""
    import pandas as pd

    folds_def = leave_one_plug_out_splits(bundle.df)
    rows: List[Dict[str, Any]] = []

    for fold_id, (train_idx, test_idx, sample_id) in enumerate(folds_def):
        x_tr, y_tr = bundle.X[train_idx], bundle.y[train_idx]
        x_te, y_te = bundle.X[test_idx], bundle.y[test_idx]
        model = model_factory()
        model.fit(x_tr, y_tr)
        pred = float(model.predict(x_te)[0])
        row: Dict[str, Any] = {
            "fold_id": fold_id,
            "sample_id": sample_id,
            "target": bundle.target,
            "y_true": float(y_te[0]),
            "y_pred": pred,
            "abs_error": float(abs(y_te[0] - pred)),
        }
        if "ct_depth_m" in bundle.df.columns:
            row["ct_depth_m"] = float(bundle.df.loc[test_idx[0], "ct_depth_m"])
        if "HFU" in bundle.df.columns:
            row["HFU"] = int(bundle.df.loc[test_idx[0], "HFU"])
        rows.append(row)

    return pd.DataFrame(rows)


def fit_full_and_predict(
    model: Any,
    bundle: XYBundle,
) -> np.ndarray:
    """Fit on all rows and return in-sample predictions."""
    fitted = clone(model)
    fitted.fit(bundle.X, bundle.y)
    return fitted.predict(bundle.X)


@dataclass
class ClassFoldMetrics:
    """Classification metrics for a single CV fold."""

    fold_id: int
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    n_train: int
    n_test: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClassCvSummary:
    """Aggregated depth-block classification CV results."""

    target: str
    model_name: str
    folds: List[ClassFoldMetrics]
    mean_accuracy: float
    std_accuracy: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    mean_f1_macro: float
    std_f1_macro: float
    protocol: str
    global_oof_accuracy: Optional[float] = None
    global_oof_balanced_accuracy: Optional[float] = None
    global_oof_f1_macro: Optional[float] = None
    class_labels: Optional[List[int]] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "target": self.target,
            "model_name": self.model_name,
            "protocol": self.protocol,
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_balanced_accuracy": self.mean_balanced_accuracy,
            "std_balanced_accuracy": self.std_balanced_accuracy,
            "mean_f1_macro": self.mean_f1_macro,
            "std_f1_macro": self.std_f1_macro,
            "folds": [
                {
                    "fold_id": f.fold_id,
                    "accuracy": f.accuracy,
                    "balanced_accuracy": f.balanced_accuracy,
                    "f1_macro": f.f1_macro,
                    "n_train": f.n_train,
                    "n_test": f.n_test,
                    **f.extra,
                }
                for f in self.folds
            ],
        }
        if self.global_oof_accuracy is not None:
            out["global_oof_accuracy"] = self.global_oof_accuracy
        if self.global_oof_balanced_accuracy is not None:
            out["global_oof_balanced_accuracy"] = self.global_oof_balanced_accuracy
        if self.global_oof_f1_macro is not None:
            out["global_oof_f1_macro"] = self.global_oof_f1_macro
        if self.class_labels is not None:
            out["class_labels"] = self.class_labels
        return out


def _classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def evaluate_depth_blocks_classification(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
    n_blocks: int = 3,
) -> ClassCvSummary:
    """Depth-block CV for classification targets (e.g. HFU)."""
    folds_def = depth_block_splits(bundle.df, n_blocks=n_blocks)
    fold_metrics: List[ClassFoldMetrics] = []
    oof_pred = np.full(len(bundle.y), np.nan, dtype=np.float64)

    for fold_id, x_tr, y_tr, x_te, y_te in iter_fold_arrays(bundle, folds_def):
        y_tr_cls = y_tr.astype(int)
        y_te_cls = y_te.astype(int)
        model = model_factory()
        model.fit(x_tr, y_tr_cls)
        pred = model.predict(x_te).astype(int)
        oof_pred[folds_def[fold_id].test_idx] = pred.astype(float)
        scores = _classification_scores(y_te_cls, pred)
        fold_metrics.append(
            ClassFoldMetrics(
                fold_id=fold_id,
                accuracy=scores["accuracy"],
                balanced_accuracy=scores["balanced_accuracy"],
                f1_macro=scores["f1_macro"],
                n_train=len(y_tr_cls),
                n_test=len(y_te_cls),
                extra={
                    "depth_min_m": folds_def[fold_id].depth_min_m,
                    "depth_max_m": folds_def[fold_id].depth_max_m,
                },
            )
        )

    acc_vals = [f.accuracy for f in fold_metrics]
    bal_vals = [f.balanced_accuracy for f in fold_metrics]
    f1_vals = [f.f1_macro for f in fold_metrics]
    valid = np.isfinite(oof_pred)
    y_all = bundle.y[valid].astype(int)
    pred_all = oof_pred[valid].astype(int)
    global_scores = _classification_scores(y_all, pred_all)
    labels = sorted(int(v) for v in np.unique(bundle.y))

    return ClassCvSummary(
        target=bundle.target,
        model_name="",
        folds=fold_metrics,
        mean_accuracy=float(np.mean(acc_vals)),
        std_accuracy=float(np.std(acc_vals)),
        mean_balanced_accuracy=float(np.mean(bal_vals)),
        std_balanced_accuracy=float(np.std(bal_vals)),
        mean_f1_macro=float(np.mean(f1_vals)),
        std_f1_macro=float(np.std(f1_vals)),
        protocol="depth_block_{}".format(n_blocks),
        global_oof_accuracy=global_scores["accuracy"],
        global_oof_balanced_accuracy=global_scores["balanced_accuracy"],
        global_oof_f1_macro=global_scores["f1_macro"],
        class_labels=labels,
    )


def classification_oof_predictions(
    model_factory: Callable[[], Any],
    bundle: XYBundle,
    n_blocks: int = 3,
) -> "pd.DataFrame":
    """Out-of-fold classification predictions."""
    import pandas as pd

    folds_def = depth_block_splits(bundle.df, n_blocks=n_blocks)
    rows: List[Dict[str, Any]] = []

    for fold_id, x_tr, y_tr, x_te, y_te in iter_fold_arrays(bundle, folds_def):
        y_tr_cls = y_tr.astype(int)
        y_te_cls = y_te.astype(int)
        model = model_factory()
        model.fit(x_tr, y_tr_cls)
        pred = model.predict(x_te).astype(int)
        for i, idx in enumerate(folds_def[fold_id].test_idx):
            row: Dict[str, Any] = {
                "fold_id": fold_id,
                "row_idx": int(idx),
                "y_true": int(y_te_cls[i]),
                "y_pred": int(pred[i]),
                "correct": bool(y_te_cls[i] == pred[i]),
            }
            if DEPTH_COL in bundle.df.columns:
                row["depth_m"] = float(bundle.df.loc[idx, DEPTH_COL])
            rows.append(row)

    return pd.DataFrame(rows)
