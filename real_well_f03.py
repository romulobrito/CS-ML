#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load F03-4 well log (AC, GR, Porosity vs depth) and build sliding windows for
direct [u,b]->y benchmark (Path A, real data).

u: [AC window || GR window] in R^{2L}. y: Porosity in R^L. Split is contiguous
along depth (no shuffle) to reduce leakage.

ASCII-only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sir_cs_pipeline_optimized import get_basis


@dataclass
class F03Table:
    """Indexed rows sorted by increasing depth."""

    depth: np.ndarray
    ac: np.ndarray
    gr: np.ndarray
    porosity: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.depth.shape[0])


def load_f03_table(path: str) -> F03Table:
    """Read tab-separated file with header Depth, AC, GR, Porosity."""
    p = os.path.abspath(path.strip())
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p, sep="\t")
    dcol = _find_col(df, "depth")
    acol = _find_col(df, "ac")
    gcol = _find_col(df, "gr")
    pcol = _find_col(df, "porosity")
    depth = np.asarray(df[dcol].values, dtype=np.float64)
    ac = np.asarray(df[acol].values, dtype=np.float64)
    gr = np.asarray(df[gcol].values, dtype=np.float64)
    poro = np.asarray(df[pcol].values, dtype=np.float64)
    order = np.argsort(depth)
    depth = depth[order]
    ac = ac[order]
    gr = gr[order]
    poro = poro[order]
    mask = np.isfinite(depth) & np.isfinite(ac) & np.isfinite(gr) & np.isfinite(poro)
    if not bool(np.all(mask)):
        depth = depth[mask]
        ac = ac[mask]
        gr = gr[mask]
        poro = poro[mask]
    if depth.size < 8:
        raise ValueError("Too few rows after cleaning.")
    return F03Table(depth=depth, ac=ac, gr=gr, porosity=poro)


def _norm_name(c: str) -> str:
    s = str(c).strip().lower()
    for ch in " ()/":
        s = s.replace(ch, "")
    return s


def _find_col(df: pd.DataFrame, key: str) -> str:
    """
    key: 'depth' | 'ac' | 'gr' | 'porosity'
    """
    for c in df.columns:
        n = _norm_name(c)
        if key == "depth" and n.startswith("depth"):
            return str(c)
        if key == "ac" and n.startswith("ac"):
            return str(c)
        if key == "gr" and n.startswith("gr"):
            return str(c)
        if key == "porosity" and (n == "porosity" or n.startswith("porosity") or n == "phi"):
            return str(c)
    names = list(df.columns)
    if len(names) >= 4 and key == "depth":
        return str(names[0])
    if len(names) >= 4 and key == "ac":
        return str(names[1])
    if len(names) >= 4 and key == "gr":
        return str(names[2])
    if len(names) >= 4 and key == "porosity":
        return str(names[3])
    raise KeyError("Could not find column for key={} in {}".format(key, list(df.columns)))


_VALID_CHANNELS = ("ac", "gr")


def _select_channel_series(tab: F03Table, name: str) -> np.ndarray:
    """Return the raw series from tab by canonical channel name."""
    key = name.strip().lower()
    if key == "ac":
        return tab.ac
    if key == "gr":
        return tab.gr
    raise ValueError(
        "Unknown channel '{}'. Valid channels: {}.".format(name, _VALID_CHANNELS)
    )


def normalize_channels(channels: Tuple[str, ...]) -> Tuple[str, ...]:
    """Return a validated, order-preserving tuple of unique channel names."""
    seen: List[str] = []
    for c in channels:
        key = c.strip().lower()
        if key not in _VALID_CHANNELS:
            raise ValueError(
                "Unknown channel '{}'. Valid channels: {}.".format(c, _VALID_CHANNELS)
            )
        if key not in seen:
            seen.append(key)
    if not seen:
        raise ValueError("At least one channel must be provided.")
    return tuple(seen)


def build_sliding_windows(
    tab: F03Table,
    window_len: int,
    step: int,
    channels: Tuple[str, ...] = ("ac", "gr"),
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[float],
    List[Tuple[float, float]],
]:
    """
    Return X (n_win, C*L), Y (n_win, L), center_depths, depth_range per window.
    Here C = len(channels); channels in order define the block structure of each u.
    """
    ch = normalize_channels(channels)
    series = [_select_channel_series(tab, c) for c in ch]
    l = int(window_len)
    st = max(1, int(step))
    n = tab.n_rows
    if n < l:
        raise ValueError(f"Need at least L={l} depth samples, got n={n}.")
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    centers: List[float] = []
    ranges: List[Tuple[float, float]] = []
    t = 0
    while t + l <= n:
        sw = tab.depth[t : t + l]
        centers.append(float(0.5 * (sw[0] + sw[-1])))
        ranges.append((float(sw[0]), float(sw[-1])))
        yseg = tab.porosity[t : t + l]
        segs = [s[t : t + l] for s in series]
        u = np.concatenate(segs, axis=0)
        xs.append(u.astype(np.float64, copy=False))
        ys.append(yseg.astype(np.float64, copy=False))
        t += st
    if not xs:
        raise ValueError("No windows: check window_len and step.")
    x_arr = np.stack(xs, axis=0)
    y_arr = np.stack(ys, axis=0)
    return x_arr, y_arr, centers, ranges


def contiguous_split(
    n_samples: int, train_frac: float, val_frac: float
) -> Tuple[slice, slice, slice, int, int, int]:
    if not (0.0 < train_frac < 1.0) or not (0.0 < val_frac < 1.0):
        raise ValueError("train_frac and val_frac must be in (0,1).")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1.0 to leave a test set.")
    n_tr = int(np.floor(float(train_frac) * float(n_samples)))
    n_va = int(np.floor(float(val_frac) * float(n_samples)))
    n_te = n_samples - n_tr - n_va
    if n_tr < 4 or n_va < 2 or n_te < 2:
        raise ValueError(
            f"Split too small: n={n_samples} n_tr={n_tr} n_va={n_va} n_te={n_te}. "
            "Increase data or adjust fracs / windowing."
        )
    sl_tr = slice(0, n_tr)
    sl_va = slice(n_tr, n_tr + n_va)
    sl_te = slice(n_tr + n_va, n_samples)
    return sl_tr, sl_va, sl_te, n_tr, n_va, n_te


def reconstruct_depth_profile(
    window_preds: np.ndarray,
    window_starts: np.ndarray,
    window_len: int,
    n_rows_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct a point-wise profile of length n_rows_total from overlapping window
    predictions by averaging all windows that cover each row.

    window_preds: array (n_win, L) of predicted y-windows.
    window_starts: array (n_win,) of absolute starting row indices for each window.
    window_len: L (number of rows per window).
    n_rows_total: total number of rows in the full well (depth axis).

    Returns (profile, coverage) both of length n_rows_total. Rows with coverage == 0
    are left as NaN in profile.
    """
    wp = np.asarray(window_preds, dtype=np.float64)
    ws = np.asarray(window_starts, dtype=np.int64).ravel()
    l = int(window_len)
    nr = int(n_rows_total)
    if wp.ndim != 2 or wp.shape[1] != l:
        raise ValueError("window_preds shape must be (n_win, L).")
    if ws.shape[0] != wp.shape[0]:
        raise ValueError("window_starts must align with window_preds.")
    acc = np.zeros(nr, dtype=np.float64)
    cov = np.zeros(nr, dtype=np.int64)
    for j in range(wp.shape[0]):
        t = int(ws[j])
        if t < 0 or t + l > nr:
            continue
        acc[t : t + l] += wp[j]
        cov[t : t + l] += 1
    profile = np.full(nr, np.nan, dtype=np.float64)
    mask = cov > 0
    profile[mask] = acc[mask] / cov[mask].astype(np.float64)
    return profile, cov


def test_window_row_starts(
    n_train: int, n_val: int, n_test: int, step: int
) -> np.ndarray:
    """Absolute starting row indices (into tab.depth) for each test-block window."""
    st = max(1, int(step))
    start = (int(n_train) + int(n_val)) * st
    return np.arange(int(n_test), dtype=np.int64) * st + start


def build_direct_ub_data_dict(
    x_all: np.ndarray,
    y_all: np.ndarray,
    sl_tr: slice,
    sl_va: slice,
    sl_te: slice,
    residual_basis: str,
) -> Dict[str, np.ndarray]:
    """Tensors for run_direct_ub_from_data: Alpha = Y @ Psi (DCT/identity)."""
    y_tr = y_all[sl_tr]
    y_va = y_all[sl_va]
    y_te = y_all[sl_te]
    l = y_tr.shape[1]
    p_in = int(x_all.shape[1])
    if p_in % l != 0 or p_in < l:
        raise ValueError(
            "Expected p_input = C * L with C >= 1 and C integer; got p_in={} L={}.".format(
                p_in, l
            )
        )
    psi = get_basis(l, residual_basis)
    alpha_tr = y_tr @ psi
    alpha_va = y_va @ psi
    alpha_te = y_te @ psi
    return {
        "X_train": x_all[sl_tr].copy(),
        "X_val": x_all[sl_va].copy(),
        "X_test": x_all[sl_te].copy(),
        "Y_train": y_tr.copy(),
        "Y_val": y_va.copy(),
        "Y_test": y_te.copy(),
        "Alpha_train": alpha_tr,
        "Alpha_val": alpha_va,
        "Alpha_test": alpha_te,
        "Psi": psi,
    }
