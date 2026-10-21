#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split smoother error inside and outside the observed depth span.

Uses the same windows, Omega, and noise draws as the extended smoother run.
Does not refit CLP. ASCII only.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.run_r3m03_noise_sweep import (
    NOISE_STDS,
    RHOS,
    SEEDS,
    STEPS,
    _draw_unit_noise,
    _load_split,
    _scale_noise,
)
from scripts.run_r3m03_simple_baselines import (
    _interpolate,
    _mean_window_rmse,
    _omega_positions,
    _select_smooth_alpha,
)
from sir_cs_pipeline_optimized import rmse

DETAIL = (
    ROOT_DIR
    / "outputs"
    / "reviewer_revision"
    / "r3m03_sparse_methods"
    / "interp_smooth_extended"
    / "tables"
    / "detail_by_seed.csv"
)
OUT_DIR = (
    ROOT_DIR
    / "outputs"
    / "reviewer_revision"
    / "r3m03_sparse_methods"
    / "interp_smooth_extended"
    / "edge_diagnostic"
)
# Named case from the extended table, plus the worst smoother cells.
FOCUS = (
    (32, 7, 0.10, 0.20),
)


def _span_masks(positions: np.ndarray, length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Interior is the closed span from the first to the last observation."""
    index = np.asarray(positions, dtype=np.int64)
    if index.size < 2:
        raise ValueError("Edge split needs at least two observed depths.")
    lo = int(np.min(index))
    hi = int(np.max(index))
    grid = np.arange(int(length), dtype=np.int64)
    interior = (grid >= lo) & (grid <= hi)
    exterior = np.logical_not(interior)
    return interior, exterior


def _subset_rmse(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """RMSE on one depth mask. Empty masks are reported as NaN."""
    if int(np.sum(mask)) == 0:
        return float("nan")
    return float(rmse(truth[mask], pred[mask]))


def _sse(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> float:
    """Sum of squared errors on one depth mask."""
    if int(np.sum(mask)) == 0:
        return 0.0
    gap = np.asarray(truth[mask], dtype=np.float64) - np.asarray(pred[mask], dtype=np.float64)
    return float(np.sum(gap * gap))


def _score_split(
    truth: np.ndarray,
    pred: np.ndarray,
    interior: np.ndarray,
    exterior: np.ndarray,
) -> Dict[str, float]:
    """Mean per-window RMSE inside and outside the observed span."""
    n_windows = int(truth.shape[0])
    inside: List[float] = []
    outside: List[float] = []
    sse_in = 0.0
    sse_out = 0.0
    n_out = 0
    for i in range(n_windows):
        inside.append(_subset_rmse(truth[i], pred[i], interior))
        outside.append(_subset_rmse(truth[i], pred[i], exterior))
        sse_in += _sse(truth[i], pred[i], interior)
        sse_out += _sse(truth[i], pred[i], exterior)
        n_out += int(np.sum(exterior))
    full = _mean_window_rmse(truth, pred)
    sse_all = sse_in + sse_out
    share = float(sse_out / sse_all) if sse_all > 0.0 else float("nan")
    return {
        "rmse": float(full),
        "rmse_inside": float(np.nanmean(np.asarray(inside, dtype=np.float64))),
        "rmse_outside": float(np.nanmean(np.asarray(outside, dtype=np.float64))),
        "sse_share_outside": share,
        "n_outside": int(n_out),
        "n_inside_per_window": int(np.sum(interior)),
        "n_outside_per_window": int(np.sum(exterior)),
    }


def _worst_window(
    truth: np.ndarray,
    pred: np.ndarray,
) -> int:
    """Index of the test window with the largest full-window RMSE."""
    scores = [
        rmse(truth[i], pred[i]) for i in range(int(truth.shape[0]))
    ]
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def _plot_window(
    path: Path,
    depth_index: np.ndarray,
    truth: np.ndarray,
    interp: np.ndarray,
    smooth: np.ndarray,
    positions: np.ndarray,
    observations: np.ndarray,
    title: str,
) -> None:
    """One window: target, interpolation, smoother, and the sparse samples."""
    fig, axis = plt.subplots(figsize=(8.2, 3.6))
    axis.plot(depth_index, truth, color="black", linewidth=1.2, label="target")
    axis.plot(depth_index, interp, color="tab:blue", linewidth=1.0, label="interpolation")
    axis.plot(depth_index, smooth, color="tab:orange", linewidth=1.0, label="smoother")
    axis.scatter(
        positions,
        observations,
        s=18,
        color="tab:red",
        zorder=3,
        label="observations",
    )
    lo = int(np.min(positions))
    hi = int(np.max(positions))
    axis.axvspan(depth_index[0], depth_index[lo], color="0.85", zorder=0)
    axis.axvspan(depth_index[hi], depth_index[-1], color="0.85", zorder=0)
    axis.set_xlabel("depth index in the window")
    axis.set_ylabel("Vc")
    axis.set_title(title)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    """Write the inside/outside table and the extreme-window figures."""
    if not DETAIL.is_file():
        raise FileNotFoundError(str(DETAIL))
    saved = pd.read_csv(DETAIL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    window_rows: List[Dict[str, float]] = []
    plotted = 0
    for step in STEPS:
        data = _load_split(int(step))
        length = int(data["Y_test"].shape[1])
        depth_index = np.arange(length, dtype=np.int64)
        for seed in SEEDS:
            for rho in RHOS:
                (
                    measurement,
                    _clean_train,
                    clean_val,
                    clean_test,
                    _unit_train,
                    unit_val,
                    unit_test,
                ) = _draw_unit_noise(
                    data["Y_train"],
                    data["Y_val"],
                    data["Y_test"],
                    int(seed),
                    float(rho),
                )
                positions = _omega_positions(measurement)
                interior, exterior = _span_masks(positions, length)
                for noise_std in NOISE_STDS:
                    b_val = _scale_noise(clean_val, unit_val, float(noise_std))
                    b_test = _scale_noise(clean_test, unit_test, float(noise_std))
                    interp = _interpolate(b_test, positions, length)
                    smooth, alpha, _val_score = _select_smooth_alpha(
                        b_val,
                        data["Y_val"],
                        b_test,
                        positions,
                        length,
                    )
                    truth = np.asarray(data["Y_test"], dtype=np.float64)
                    interp_split = _score_split(truth, interp, interior, exterior)
                    smooth_split = _score_split(truth, smooth, interior, exterior)
                    key = (
                        int(step),
                        int(seed),
                        float(rho),
                        float(noise_std),
                    )
                    match = saved[
                        (saved["step"] == key[0])
                        & (saved["seed"] == key[1])
                        & np.isclose(saved["measurement_ratio"], key[2])
                        & np.isclose(saved["noise_std"], key[3])
                    ]
                    if int(match.shape[0]) != 1:
                        raise RuntimeError("Saved row missing for {0}.".format(key))
                    saved_smooth = float(match["smooth_rmse"].iloc[0])
                    gap = abs(saved_smooth - float(smooth_split["rmse"]))
                    if gap > 1.0e-6:
                        raise RuntimeError(
                            "Smoother RMSE drifted by {0} at {1}.".format(gap, key)
                        )
                    row: Dict[str, float] = {
                        "step": float(key[0]),
                        "seed": float(key[1]),
                        "measurement_ratio": key[2],
                        "noise_std": key[3],
                        "smooth_alpha": float(alpha),
                        "lo": float(np.min(positions)),
                        "hi": float(np.max(positions)),
                        "n_outside_per_window": float(smooth_split["n_outside_per_window"]),
                    }
                    for prefix, split in (
                        ("interp", interp_split),
                        ("smooth", smooth_split),
                    ):
                        row[prefix + "_rmse"] = float(split["rmse"])
                        row[prefix + "_rmse_inside"] = float(split["rmse_inside"])
                        row[prefix + "_rmse_outside"] = float(split["rmse_outside"])
                        row[prefix + "_sse_share_outside"] = float(split["sse_share_outside"])
                    rows.append(row)
                    worst = _worst_window(truth, smooth)
                    window_rows.append(
                        {
                            "step": float(key[0]),
                            "seed": float(key[1]),
                            "measurement_ratio": key[2],
                            "noise_std": key[3],
                            "worst_window": float(worst),
                            "worst_smooth_rmse": float(rmse(truth[worst], smooth[worst])),
                            "worst_interp_rmse": float(rmse(truth[worst], interp[worst])),
                            "worst_smooth_inside": _subset_rmse(
                                truth[worst], smooth[worst], interior
                            ),
                            "worst_smooth_outside": _subset_rmse(
                                truth[worst], smooth[worst], exterior
                            ),
                            "smooth_alpha": float(alpha),
                        }
                    )
                    focus = (key[0], key[1], key[2], key[3]) in FOCUS
                    huge = float(smooth_split["rmse"]) >= 0.3
                    if focus or huge:
                        name = "step{0}_seed{1}_rho{2:.2f}_noise{3:.2f}_w{4}.png".format(
                            key[0], key[1], key[2], key[3], worst
                        )
                        _plot_window(
                            OUT_DIR / "figures" / name,
                            depth_index,
                            truth[worst],
                            interp[worst],
                            smooth[worst],
                            positions,
                            b_test[worst],
                            "step {0}, seed {1}, rho {2}, noise {3}, window {4}".format(
                                key[0], key[1], key[2], key[3], worst
                            ),
                        )
                        plotted += 1
    detail = pd.DataFrame(rows)
    detail_path = OUT_DIR / "tables"
    detail_path.mkdir(parents=True, exist_ok=True)
    detail.to_csv(detail_path / "edge_by_seed.csv", index=False)
    pd.DataFrame(window_rows).to_csv(detail_path / "worst_window_by_cell.csv", index=False)
    summary_rows: List[Dict[str, float]] = []
    grouped = detail.groupby(["step", "measurement_ratio", "noise_std"], sort=True)
    for keys, frame in grouped:
        step, rho, noise_std = keys
        summary_rows.append(
            {
                "step": int(step),
                "measurement_ratio": float(rho),
                "noise_std": float(noise_std),
                "smooth_rmse": float(frame["smooth_rmse"].mean()),
                "smooth_rmse_inside": float(frame["smooth_rmse_inside"].mean()),
                "smooth_rmse_outside": float(frame["smooth_rmse_outside"].mean()),
                "smooth_sse_share_outside": float(frame["smooth_sse_share_outside"].mean()),
                "interp_rmse": float(frame["interp_rmse"].mean()),
                "interp_rmse_inside": float(frame["interp_rmse_inside"].mean()),
                "interp_rmse_outside": float(frame["interp_rmse_outside"].mean()),
                "interp_sse_share_outside": float(frame["interp_sse_share_outside"].mean()),
                "n_outside_per_window": float(frame["n_outside_per_window"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(detail_path / "edge_summary_by_cell.csv", index=False)
    lines = [
        "Inside/outside diagnostic for the extended smoother.",
        "Interior: depths from the first to the last observation, inclusive.",
        "Outside: depths beyond that span. Interpolation holds the nearest sample there.",
        "The smoother does not. CLP curves were not recomputed.",
        "Smoother full-window RMSE was checked against interp_smooth_extended.",
        "Figures: the named step-32 seed-7 rho-0.10 noise-0.20 case,",
        "and every cell whose mean smoother RMSE is at least 0.30.",
        "Figures plotted: {0}.".format(plotted),
    ]
    (OUT_DIR / "PROTOCOL.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT_DIR / "config_frozen.json").write_text(
        json.dumps(
            {
                "source_detail": str(DETAIL),
                "focus": [list(item) for item in FOCUS],
                "huge_rmse": 0.3,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("wrote", detail_path)


if __name__ == "__main__":
    main()
