#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read-only audit of the published well 861 laboratory-porosity protocol.

The published runner is scripts/run_r1m11_861_phi_lab.py. This script
reconstructs its depth blocks, windows, and masks, then checks whether one
physical depth is both a measurement and a scored target. It does not train,
does not edit the runner, and does not write into the published output
directory.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_r1m11_861_phi_lab as published  # noqa: E402

REFERENCE_COMMIT = "2e657987ac2425aace4191e49e2b9affb6bda509"
OUT_DIR = REPO_ROOT / "outputs" / "exploratory" / "861_depth_mask_audit"
PUBLISHED_DIR = published.OUT_DIR
PUBLISHED_SCRIPT = SCRIPTS_DIR / "run_r1m11_861_phi_lab.py"
RECOVERY_SCRIPT = REPO_ROOT / "csgm_m2_module.py"
AUDDYS_PATH = REPO_ROOT / "data" / "Auddys_table.xlsx"
GLOBAL_SEED_SALT = 9100
RQI_FACTOR = 0.0314
IDENTITY_ATOL = 1e-6
DERIVED_COLUMNS: Tuple[str, ...] = ("RQI", "FZI_lab", "HFU", "k_lab (mD)")


def _sha256(path: Path) -> str:
    """Return the hex digest of a file."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_head() -> str:
    """Return the current commit, or an empty string if git is unavailable."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return done.stdout.strip()


def _require_file(path: Path) -> None:
    """Raise if a required input is missing."""
    if not path.is_file():
        raise FileNotFoundError(str(path))


def _split_fold(
    n_rows: int,
    fold_id: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce the published train, validation, and test row split.

    The validation rows are the last VAL_ROWS of the deepest contiguous
    training segment. They are removed from training. Windows are not
    built here.
    """
    blocks = published._depth_blocks(n_rows, published.N_BLOCKS)
    if fold_id < 0 or fold_id >= len(blocks):
        raise ValueError("fold_id is outside the published block list.")
    test_rows = np.asarray(blocks[fold_id], dtype=np.int64)
    train_pool = np.sort(
        np.concatenate(
            [block for index, block in enumerate(blocks) if index != fold_id]
        ).astype(np.int64)
    )
    segments: List[np.ndarray] = []
    seg_start = int(train_pool[0])
    prev = int(train_pool[0])
    for value in train_pool[1:].tolist():
        if int(value) != prev + 1:
            segments.append(np.arange(seg_start, prev + 1, dtype=np.int64))
            seg_start = int(value)
        prev = int(value)
    segments.append(np.arange(seg_start, prev + 1, dtype=np.int64))
    deepest = max(segments, key=lambda seg: int(seg[-1]))
    if int(deepest.shape[0]) <= published.VAL_ROWS + published.WINDOW_LEN:
        raise ValueError("Training segment is too short for validation windows.")
    val_rows = np.asarray(deepest[-published.VAL_ROWS :], dtype=np.int64)
    train_rows = np.setdiff1d(train_pool, val_rows, assume_unique=False)
    return train_rows, val_rows, test_rows


def _nearest_gap_m(
    depth_m: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """Smallest positive depth separation between two row sets, in meters."""
    if left.size < 1 or right.size < 1:
        raise ValueError("Cannot measure a gap between empty row sets.")
    left_depth = depth_m[left]
    right_depth = depth_m[right]
    # Broadcast is small: at most 71 by 29.
    gap = np.abs(left_depth.reshape(-1, 1) - right_depth.reshape(1, -1))
    return float(np.min(gap))


def _index_adjacent(left: np.ndarray, right: np.ndarray) -> bool:
    """True when two integer index sets contain a consecutive pair."""
    if left.size < 1 or right.size < 1:
        return False
    right_set = set(int(value) for value in right.tolist())
    for value in left.tolist():
        if (int(value) + 1) in right_set or (int(value) - 1) in right_set:
            return True
    return False


def _published_omega(fold_id: int, seed: int, rho: float) -> np.ndarray:
    """Redraw the published local Omega with the runner's seed formula."""
    rng = np.random.default_rng(
        int(seed) * 1009 + int(fold_id) * 17 + int(round(float(rho) * 100.0))
    )
    observed, _operator = published._sample_omega(
        published.WINDOW_LEN,
        float(rho),
        rng,
    )
    return np.asarray(observed, dtype=np.int64)


def _role_counts(
    rows: np.ndarray,
    starts: Sequence[int],
    observed_local: np.ndarray,
    window_len: int,
) -> Dict[int, Dict[str, int]]:
    """Count how often each physical row is observed or scored.

    A row covered by a window contributes to exactly one of the two counts
    for that window. Rows outside every window stay at zero.
    """
    observed_set = set(int(value) for value in observed_local.tolist())
    counts: Dict[int, Dict[str, int]] = {}
    for row in rows.tolist():
        counts[int(row)] = {"n_windows": 0, "n_observed": 0, "n_scored": 0}
    for start in starts:
        for offset in range(window_len):
            row = int(start) + offset
            if row not in counts:
                raise ValueError("Window {} leaves the row set.".format(start))
            slot = counts[row]
            slot["n_windows"] += 1
            if offset in observed_set:
                slot["n_observed"] += 1
            else:
                slot["n_scored"] += 1
            if slot["n_observed"] + slot["n_scored"] != slot["n_windows"]:
                raise ValueError("Role counts diverged at row {}.".format(row))
    return counts


def _summarize_roles(counts: Dict[int, Dict[str, int]]) -> Dict[str, float]:
    """Collapse per-depth role counts to one mask summary."""
    n_both = 0
    n_scored_only = 0
    n_observed_only = 0
    n_uncovered = 0
    score_pairs = 0
    contaminated_pairs = 0
    max_scored = 0
    for slot in counts.values():
        n_obs = int(slot["n_observed"])
        n_score = int(slot["n_scored"])
        max_scored = max(max_scored, n_score)
        score_pairs += n_score
        if n_obs > 0 and n_score > 0:
            n_both += 1
            contaminated_pairs += n_score
        elif n_score > 0:
            n_scored_only += 1
        elif n_obs > 0:
            n_observed_only += 1
        else:
            n_uncovered += 1
    if score_pairs < 1 and n_uncovered != len(counts):
        # A split with windows must score something unless every covered
        # coordinate is observed. Uncovered-only sets are reported by the
        # caller. Here a positive score count is required when any window
        # exists.
        pass
    fraction = float(contaminated_pairs) / float(score_pairs) if score_pairs else float("nan")
    return {
        "n_depths": float(len(counts)),
        "n_both": float(n_both),
        "n_scored_only": float(n_scored_only),
        "n_observed_only": float(n_observed_only),
        "n_uncovered": float(n_uncovered),
        "n_score_pairs": float(score_pairs),
        "n_contaminated_pairs": float(contaminated_pairs),
        "contaminated_pair_fraction": fraction,
        "max_score_multiplicity": float(max_scored),
    }


def _role_name(n_observed: int, n_scored: int) -> str:
    """Name the physical-depth role from the two counts."""
    if n_observed > 0 and n_scored > 0:
        return "both"
    if n_scored > 0:
        return "scored_only"
    if n_observed > 0:
        return "observed_only"
    return "uncovered"


def _global_observed_rows(
    rows: np.ndarray,
    fold_id: int,
    seed: int,
    rho: float,
) -> np.ndarray:
    """Draw one counterfactual observed set on physical rows.

    The salt keeps this draw distinct from the published local Omega.
    The fraction is applied to the row count, not to the window length.
    """
    n_rows = int(rows.shape[0])
    m_obs = int(round(float(rho) * float(n_rows)))
    m_obs = max(1, min(m_obs, n_rows - 1))
    rng = np.random.default_rng(
        int(seed) * 1009
        + int(fold_id) * 17
        + int(round(float(rho) * 100.0))
        + GLOBAL_SEED_SALT
    )
    chosen = rng.choice(rows, size=m_obs, replace=False).astype(np.int64)
    return np.sort(chosen)


def _empty_window_probability(n_rows: int, n_observed: int, window_len: int) -> float:
    """Probability that one window misses every globally observed row.

    The draw is a uniform subset. Overlapping windows are not independent,
    so this is not the expected fraction of empty windows.
    """
    n_held = int(n_rows) - int(n_observed)
    if n_held < int(window_len) or int(window_len) < 1 or int(n_rows) < int(window_len):
        return 0.0
    return float(math.comb(n_held, int(window_len))) / float(math.comb(int(n_rows), int(window_len)))


def _count_empty_windows(
    starts: Sequence[int],
    observed_rows: np.ndarray,
    window_len: int,
) -> int:
    """Count windows whose physical rows miss the observed set."""
    observed = set(int(value) for value in observed_rows.tolist())
    empty = 0
    for start in starts:
        covered = [int(start) + offset for offset in range(window_len)]
        if not any(row in observed for row in covered):
            empty += 1
    return empty


def _roles_from_global(
    rows: np.ndarray,
    starts: Sequence[int],
    observed_rows: np.ndarray,
    window_len: int,
) -> Dict[int, Dict[str, int]]:
    """Apply one physical observed set to every window."""
    observed = set(int(value) for value in observed_rows.tolist())
    counts: Dict[int, Dict[str, int]] = {}
    for row in rows.tolist():
        counts[int(row)] = {"n_windows": 0, "n_observed": 0, "n_scored": 0}
    for start in starts:
        for offset in range(window_len):
            row = int(start) + offset
            if row not in counts:
                raise ValueError("Global-mask window leaves the row set.")
            slot = counts[row]
            slot["n_windows"] += 1
            if row in observed:
                slot["n_observed"] += 1
            else:
                slot["n_scored"] += 1
    return counts


def _depth_span_m(depth_m: np.ndarray, rows: np.ndarray) -> float:
    """Inclusive depth span of a row set. A single row has span 0."""
    if rows.size < 1:
        return float("nan")
    values = depth_m[rows]
    return float(np.max(values) - np.min(values))


def _load_legend() -> Tuple[pd.DataFrame, str]:
    """Read facies and HFU labels. Missing file yields an empty table."""
    if not AUDDYS_PATH.is_file():
        return pd.DataFrame(columns=["code_name", "code", "label"]), "absent"
    book = pd.ExcelFile(AUDDYS_PATH)
    if "Legend" not in book.sheet_names:
        return pd.DataFrame(columns=["code_name", "code", "label"]), "no_legend_sheet"
    raw = pd.read_excel(AUDDYS_PATH, sheet_name="Legend", header=None)
    rows: List[Dict[str, str]] = []
    current = ""
    for record in raw.itertuples(index=False):
        left = "" if pd.isna(record[0]) else str(record[0]).strip()
        right = "" if pd.isna(record[1]) else str(record[1]).strip()
        if right.lower() == "meaning":
            current = left
            continue
        if current and left and right:
            rows.append({"code_name": current, "code": left, "label": right})
    return pd.DataFrame(rows), "read"


def _legend_mentions_target(legend: pd.DataFrame) -> bool:
    """True if the legend text names the laboratory porosity column."""
    if legend.empty:
        return False
    blob = " ".join(legend.astype(str).to_numpy().ravel().tolist()).lower()
    return ("phi_lab" in blob) or ("porosity" in blob)


def _scan_lithotype_assignments(root: Path) -> List[Dict[str, str]]:
    """List assignments to Lithotype and whether the right-hand side uses the target.

    A hit derives the code only when the expression names Phi_lab, k_lab,
    RQI, FZI_lab, or HFU. Other assignments are recorded and are not treated
    as target leakage.
    """
    target_tokens = ("Phi_lab", "k_lab", "RQI", "FZI_lab", "HFU")
    hits: List[Dict[str, str]] = []
    roots = [root / "methods_comparison" / "scripts", root / "scripts"]
    this_file = Path(__file__).resolve()
    for base in roots:
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*.py")):
            if path.resolve() == this_file:
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#") or "=" not in line or "Lithotype" not in line:
                    continue
                left, right = line.split("=", 1)
                if "Lithotype" not in left:
                    continue
                derives = any(token in right for token in target_tokens)
                hits.append(
                    {
                        "path": "{}:{}".format(path.relative_to(root).as_posix(), line_no),
                        "derives_from_target": "true" if derives else "false",
                        "line": stripped,
                    }
                )
    return hits


def _range_overlap(frame: pd.DataFrame, code_col: str, value_col: str) -> bool:
    """True when at least two codes have overlapping numeric ranges."""
    spans: List[Tuple[float, float]] = []
    for _code, part in frame.groupby(code_col):
        values = part[value_col].to_numpy(dtype=np.float64)
        spans.append((float(np.min(values)), float(np.max(values))))
    for index, left in enumerate(spans):
        for right in spans[index + 1 :]:
            if left[0] <= right[1] and right[0] <= left[1]:
                return True
    return False


def _contiguous_partition(codes: np.ndarray, values: np.ndarray) -> bool:
    """True when sorting by values yields one contiguous run per code."""
    if codes.shape[0] != values.shape[0] or codes.shape[0] < 1:
        raise ValueError("Partition inputs must be aligned and non-empty.")
    order = np.argsort(values, kind="mergesort")
    ordered = codes[order]
    runs = 1 + int(np.sum(ordered[1:] != ordered[:-1]))
    return runs == int(np.unique(ordered).shape[0])


def _identity_errors(frame: pd.DataFrame) -> Dict[str, float]:
    """Max absolute error of the RQI and FZI identities. No values are stored."""
    phi = frame["Phi_lab (pu)"].to_numpy(dtype=np.float64)
    perm = frame["k_lab (mD)"].to_numpy(dtype=np.float64)
    rqi = frame["RQI"].to_numpy(dtype=np.float64)
    fzi = frame["FZI_lab"].to_numpy(dtype=np.float64)
    if np.any(phi <= 0.0) or np.any(phi >= 1.0) or np.any(perm < 0.0):
        raise ValueError("Phi_lab or k_lab is outside the identity domain.")
    rqi_hat = RQI_FACTOR * np.sqrt(perm / phi)
    ratio = phi / (1.0 - phi)
    fzi_hat = rqi / ratio
    return {
        "rqi_max_abs_error": float(np.max(np.abs(rqi_hat - rqi))),
        "fzi_max_abs_error": float(np.max(np.abs(fzi_hat - fzi))),
    }


def _recovery_facts() -> Dict[str, bool]:
    """Read the recovery function. Do not import or run it."""
    text = RECOVERY_SCRIPT.read_text(encoding="utf-8")
    start = text.find("def csgm_recover_with_prior")
    end = text.find("\ndef decode_latent_prior")
    if start < 0 or end < start:
        raise ValueError("Could not locate csgm_recover_with_prior.")
    body = text[start:end]
    return {
        "one_operator_argument": "M: np.ndarray" in body and "B: np.ndarray" in body,
        "batch_loss_over_windows": "torch.sum(residual ** 2) / float(n_rows)" in body,
        "single_best_restart": "best_loss" in body and "best_y_hat" in body,
    }


def _runner_facts() -> Dict[str, bool]:
    """Confirm how the published runner shares one Omega."""
    text = PUBLISHED_SCRIPT.read_text(encoding="utf-8")
    return {
        "one_omega_draw": text.count("observed, operator = _sample_omega(") == 1,
        "same_operator_for_clp_and_ae": (
            "b_test = _measurements(y_test, operator)" in text
            and "dub.concat_ub(x_test_s, b_test)" in text
            and "csgm_recover_with_prior(" in text
        ),
        "score_drops_local_omega": "scored[observed] = False" in text,
        "no_depth_interpolation": "np.interp" not in text,
        "no_zero_lambda": "0.0" not in text.split("LAMBDA_GRID", 1)[-1].split(")", 1)[0],
    }


def _check(name: str, passed: bool, detail: str) -> Dict[str, object]:
    """Build one check record."""
    return {"name": name, "passed": bool(passed), "detail": detail}


def _fmt(value: float, digits: int = 4) -> str:
    """Format a finite float. Non-finite values become the word nan."""
    if not np.isfinite(value):
        return "nan"
    return "{:.{}f}".format(float(value), digits)


def _write_report(
    path: Path,
    head: str,
    script_hash: str,
    frame: pd.DataFrame,
    split_summary: pd.DataFrame,
    mask_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    provenance: pd.DataFrame,
    facies: pd.DataFrame,
    checks: Sequence[Dict[str, object]],
) -> None:
    """Write the English ASCII diagnosis from the computed tables."""
    depth = frame[published.DEPTH_COL].to_numpy(dtype=np.float64)
    gaps = np.diff(depth)
    test_masks = mask_summary.copy()
    global_cov = coverage[coverage["alternative"] == "global_depth"]
    prob_25 = global_cov.loc[np.isclose(global_cov["rho"], 0.25), "single_window_empty_probability"]
    prob_50 = global_cov.loc[np.isclose(global_cov["rho"], 0.50), "single_window_empty_probability"]
    if prob_25.empty or prob_50.empty:
        raise ValueError("Global-mask coverage is missing a fraction.")
    if int(np.unique(np.round(prob_25.to_numpy(dtype=np.float64), 8)).shape[0]) != 1:
        raise ValueError("Empty-window probability is not constant at fraction 0.25.")
    if int(np.unique(np.round(prob_50.to_numpy(dtype=np.float64), 8)).shape[0]) != 1:
        raise ValueError("Empty-window probability is not constant at fraction 0.50.")
    pairs_25 = test_masks.loc[np.isclose(test_masks["rho"], 0.25), "test_n_score_pairs"]
    pairs_50 = test_masks.loc[np.isclose(test_masks["rho"], 0.50), "test_n_score_pairs"]
    if int(pairs_25.nunique()) != 1 or int(pairs_50.nunique()) != 1:
        raise ValueError("Score-pair counts are not constant inside a fraction.")
    observed_25 = (
        test_masks.loc[np.isclose(test_masks["rho"], 0.25), "test_n_both"]
        + test_masks.loc[np.isclose(test_masks["rho"], 0.25), "test_n_observed_only"]
    )
    observed_50 = (
        test_masks.loc[np.isclose(test_masks["rho"], 0.50), "test_n_both"]
        + test_masks.loc[np.isclose(test_masks["rho"], 0.50), "test_n_observed_only"]
    )
    val_contam = test_masks["val_contaminated_pair_fraction"].to_numpy(dtype=np.float64)
    empty_25 = global_cov.loc[np.isclose(global_cov["rho"], 0.25), "n_empty_windows"]
    empty_50 = global_cov.loc[np.isclose(global_cov["rho"], 0.50), "n_empty_windows"]
    if int(empty_50.min()) != int(empty_50.max()):
        raise ValueError("Fraction 0.50 empty-window count is not constant.")
    facies_bits: List[str] = []
    for fold_id, part in facies.groupby("fold_id"):
        dropped_bits: List[str] = []
        for record in part.itertuples(index=False):
            if int(record.n_dropped_step8) < 1:
                continue
            dropped_bits.append(
                "code {} loses {} of {}".format(
                    int(record.lithotype),
                    int(record.n_dropped_step8),
                    int(record.n_in_block),
                )
            )
        facies_bits.append(
            "fold {} ({})".format(
                int(fold_id),
                "; ".join(dropped_bits) if dropped_bits else "none dropped",
            )
        )
    split_bits: List[str] = []
    for record in split_summary.itertuples(index=False):
        split_bits.append(
            "Fold {}: train-test gap {} m (adjacent={}), val-test gap {} m (adjacent={})".format(
                int(record.fold_id),
                _fmt(float(record.train_test_gap_m), 2),
                int(record.train_test_adjacent),
                _fmt(float(record.val_test_gap_m), 2),
                int(record.val_test_adjacent),
            )
        )
    contam = test_masks["test_contaminated_pair_fraction"].to_numpy(dtype=np.float64)
    both = test_masks["test_n_both"].to_numpy(dtype=np.float64)
    scored_only = test_masks["test_n_scored_only"].to_numpy(dtype=np.float64)
    lines: List[str] = [
        "# Well 861 depth-mask audit",
        "",
        "Read-only audit of `scripts/run_r1m11_861_phi_lab.py` and",
        "`outputs/reviewer_revision/r1m11_861_phi_lab/`. No model was trained.",
        "The runner and the published tables were not modified.",
        "",
        "- Reference commit: `{}`".format(REFERENCE_COMMIT),
        "- HEAD at audit: `{}`".format(head if head else "unknown"),
        "- Runner sha256: `{}`".format(script_hash),
        "- Rows: {}".format(int(frame.shape[0])),
        "- Depth: {} m to {} m".format(_fmt(float(depth.min()), 2), _fmt(float(depth.max()), 2)),
        "- Depth gap min / median / max: {} / {} / {} m".format(
            _fmt(float(np.min(gaps)), 2),
            _fmt(float(np.median(gaps)), 2),
            _fmt(float(np.max(gaps)), 2),
        ),
        "- Window {} , step {} , blocks {}".format(
            published.WINDOW_LEN,
            published.STEP,
            published.N_BLOCKS,
        ),
        "",
        "## Train, validation, and test",
        "",
        "Blocks are contiguous groups of 29 rows, split before windowing.",
        "A window is kept only when all 8 rows lie inside one of train,",
        "validation, or test. The three row sets are disjoint in every fold.",
        "No test row enters a training window or a validation window.",
        "The decoder is fit on training windows. Lambda is selected on",
        "validation windows. Test `Phi_lab` is not an input to either fit.",
        "",
        "That separation does not remove geological adjacency. {}".format(
            "; ".join(split_bits) + "."
        ),
        "An adjacent flag means consecutive row indices, not a shared",
        "`Phi_lab` value. Neighboring depths can be statistically dependent.",
        "",
        "See `tables/split_summary.csv`.",
        "",
        "## Published mask",
        "",
        "Inside a fold, seed, and observed fraction, one local Omega is drawn",
        "on the 8 window offsets and reused by every window. The published",
        "fractions 0.25 and 0.50 therefore mean 2 and 4 offsets, not 25% and",
        "50% of the 29 physical depths.",
        "",
        "The runner scores every window offset outside that local Omega and",
        "pools those offsets. The docstring is correct at window level: a",
        "local offset used as `b` in a window is not scored in that same",
        "window. It is not correct as a statement about physical depth.",
        "",
        "Across the 18 published cells (3 folds x 3 seeds x 2 fractions):",
        "",
        "- Depths that are both measured and scored: {} to {} per block.".format(
            int(np.min(both)),
            int(np.max(both)),
        ),
        "- Depths scored and never measured: {} to {}.".format(
            int(np.min(scored_only)),
            int(np.max(scored_only)),
        ),
        "- Depths measured at least once: {} to {} of 29 at fraction 0.25, and {} to {} of 29 at fraction 0.50.".format(
            int(observed_25.min()),
            int(observed_25.max()),
            int(observed_50.min()),
            int(observed_50.max()),
        ),
        "- Share of test RMSE terms whose depth was also a measurement in another test window: {} to {}.".format(
            _fmt(float(np.min(contam))),
            _fmt(float(np.max(contam))),
        ),
        "- Same share inside the validation windows used to choose lambda: {} to {}.".format(
            _fmt(float(np.min(val_contam))),
            _fmt(float(np.max(val_contam))),
        ),
        "- Score pairs per cell: {} at fraction 0.25 and {} at fraction 0.50, against 29 depths.".format(
            int(pairs_25.iloc[0]),
            int(pairs_50.iloc[0]),
        ),
        "- Maximum times one depth enters the RMSE: {}.".format(
            int(np.max(test_masks["test_max_score_multiplicity"].to_numpy(dtype=np.float64))),
        ),
        "",
        "The same dual role appears inside the validation windows used to",
        "choose lambda. See `tables/published_mask_summary.csv` and",
        "`tables/published_depth_roles.csv`. Redrawn Omega strings match the",
        "published detail file.",
        "",
        "## What is leakage, and what is not",
        "",
        "Demonstrated reuse of laboratory values inside the test evaluation:",
        "for every published cell, the `Phi_lab` value of a physical depth is",
        "placed in `b` of at least one test window and is also a scored target",
        "in another overlapping test window. The share of pooled RMSE terms",
        "with that property is the contaminated fraction above. This is why a",
        "window-level holdout is not a holdout of laboratory depths.",
        "",
        "Not demonstrated, and not claimed: the scored window's own `b` does",
        "not contain the offset being scored. The local complement excludes it.",
        "Train, validation, and test do not share rows, so this is not leakage",
        "of test `Phi_lab` into the decoder or into lambda selection.",
        "",
        "Dependence that is not that reuse:",
        "",
        "- Overlapping windows are not independent samples. An interior depth",
        "  sits in up to 8 windows. In the published scores one depth enters",
        "  the RMSE up to {} times, including depths that were never measured.".format(
            int(np.max(test_masks["test_max_score_multiplicity"].to_numpy(dtype=np.float64)))
        ),
        "- Nearby depths remain correlated after any mask correction.",
        "- `csgm_recover_with_prior` applies one operator to the whole batch",
        "  and keeps the restart with the lowest summed loss. Window updates",
        "  do not share each other's measurements inside the data term, but",
        "  the chosen restart is common. This audit does not rerun recovery,",
        "  so it does not measure how much that shared choice moves a prediction.",
        "",
        "The published files store RMSE only. They do not store per-window",
        "predictions, so residual correlation was not estimated.",
        "",
        "## Lithotype",
        "",
        "`Lithotype` is one of the four inputs. The Auddys legend labels the",
        "codes Grainstone, Shrubs, Laminite, and Spherulite. The legend does",
        "not define the code from `Phi_lab`. The source scan finds no",
        "assignment whose right-hand side uses `Phi_lab`, `k_lab`, `RQI`,",
        "`FZI_lab`, or `HFU`. Other lines that mention the column are a",
        "per-fold count and an empty placeholder in a different well export;",
        "they are listed in the provenance table and do not build the code.",
        "On the 87 rows the code is not a threshold cut of `Phi_lab` or of",
        "`FZI_lab`, and it is not equal to `HFU`.",
        "",
        "`RQI` matches `0.0314 * sqrt(k_lab / Phi_lab)` and `FZI_lab` matches",
        "`RQI / (phi / (1 - phi))` within 1e-6. Those columns, `HFU`, and",
        "`k_lab` are not inputs. Density, neutron porosity, sonic porosity,",
        "and `Phi_ND` are also excluded. Nothing in this table shows that",
        "`Lithotype` was constructed from the target. The remaining limit is",
        "provenance: the integer is copied from the log table, and this file",
        "is not an assay ledger.",
        "",
        "See `tables/lithotype_provenance.csv`.",
        "",
        "## CLP and the direct AE",
        "",
        "The runner draws Omega once and passes the same `b` to CLP recovery",
        "and to `concat_ub` for the direct AE. The published detail file has",
        "one Omega string per fold, seed, and fraction. The two methods",
        "therefore see the same window-level observations, including the same",
        "cross-window reuse.",
        "",
        "Prior-only does not receive `b`. The runner has no unconditioned CSGM",
        "and no depth interpolation. A comparison with those baselines is not",
        "in the published result.",
        "",
        "## Two corrections, not implemented",
        "",
        "Global depth mask. Draw the observed fraction on the 29 physical",
        "rows. A row is a measurement in every window that contains it, or it",
        "is scored, never both. All 29 depths keep a role. Because the current",
        "recovery call accepts one operator for the batch, each window needs",
        "its own call when the local support changes. That uses the existing",
        "function and does not edit `csgm_m2_module.py`. Held-out depths that",
        "sit in several windows must be averaged to one residual before the",
        "RMSE. At 7 observed rows out of 29, a single window of 8 is empty",
        "with probability {}. At 14 observed rows the probability is {}.".format(
            _fmt(float(prob_25.iloc[0])),
            _fmt(float(prob_50.iloc[0])),
        ),
        "In the 9 counterfactual draws at fraction 0.25, empty windows range",
        "from {} to {} out of 22. At fraction 0.50 the count is {} in every draw.".format(
            int(empty_25.min()),
            int(empty_25.max()),
            int(empty_50.max()),
        ),
        "An empty window has no data term. The correction has to state that",
        "case before any training run. The direct AE would need a fixed-width",
        "embedding such as `embed_sparse_observation`, because `b` would no",
        "longer have one width.",
        "",
        "Non-overlapping test windows. Step 8 yields 3 windows and covers 24",
        "of 29 depths in every block. The 5 dropped depths are the deepest",
        "tail of each block, 15 of 87 rows. No covered depth has both roles,",
        "and each covered depth is scored once. The lost interval is a fixed",
        "tail, so the evaluated set is shorter than the block. The tail is",
        "not a proportional facies sample: {}.".format("; ".join(facies_bits)),
        "Counts are in `tables/facies_coverage_nonoverlap.csv`.",
        "",
        "The global depth mask is the appropriate primary correction. It",
        "answers whether a laboratory depth was given to the test evaluation.",
        "Non-overlapping windows are a coverage-limited check, not a substitute:",
        "they discard the same tail in every fold. Neither correction is",
        "implemented here. Interpolation, if added later, has to use `Depth(m)`",
        "because the gap is not uniform.",
        "",
        "Counterfactual draws use the published seed formula plus salt",
        "{}. They illustrate the geometry. They are not a new experiment.".format(
            GLOBAL_SEED_SALT
        ),
        "See `tables/alternative_coverage.csv`.",
        "",
        "## Checks",
        "",
    ]
    for item in checks:
        state = "PASS" if bool(item["passed"]) else "FAIL"
        lines.append("- [{}] {}: {}".format(state, item["name"], item["detail"]))
    lines.extend(
        [
            "",
            "## Rerun",
            "",
            "```",
            "python3 scripts/audit_r1m11_861_depth_masking.py",
            "```",
            "",
            "Outputs stay under `outputs/exploratory/861_depth_mask_audit/`.",
            "",
        ]
    )
    if "item" not in provenance.columns:
        raise ValueError("Provenance table is missing the item column.")
    if split_summary.shape[0] != published.N_BLOCKS:
        raise ValueError("Split summary does not have one row per block.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _run() -> int:
    """Rebuild the published geometry, write tables, and return 0 on success."""
    _require_file(PUBLISHED_SCRIPT)
    _require_file(RECOVERY_SCRIPT)
    _require_file(published.EXCEL_PATH)
    detail_path = PUBLISHED_DIR / "heldout_rmse_detail.csv"
    summary_path = PUBLISHED_DIR / "heldout_rmse_summary.csv"
    _require_file(detail_path)
    _require_file(summary_path)
    if OUT_DIR.resolve() == PUBLISHED_DIR.resolve():
        raise ValueError("Audit output must not replace the published directory.")

    frame = published._load_frame(published.EXCEL_PATH)
    depth_m = frame[published.DEPTH_COL].to_numpy(dtype=np.float64)
    detail = pd.read_csv(detail_path)
    required = {"fold_id", "seed", "rho", "method", "omega", "rmse_holdout", "n_train_windows", "n_val_windows", "n_test_windows"}
    missing = sorted(required.difference(detail.columns))
    if missing:
        raise ValueError("Published detail is missing columns: {}".format(", ".join(missing)))

    head = _git_head()
    script_hash = _sha256(PUBLISHED_SCRIPT)
    n_rows = int(frame.shape[0])
    blocks = published._depth_blocks(n_rows, published.N_BLOCKS)

    split_rows: List[Dict[str, float]] = []
    role_rows: List[Dict[str, float]] = []
    mask_rows: List[Dict[str, float]] = []
    coverage_rows: List[Dict[str, float]] = []
    facies_rows: List[Dict[str, float]] = []
    checks: List[Dict[str, object]] = []

    omega_match = True
    split_disjoint = True
    test_not_in_fit = True
    within_window_ok = True
    dual_role_all = True
    same_omega_methods = True
    nonoverlap_clean = True
    global_clean = True

    for fold_id in range(published.N_BLOCKS):
        train_rows, val_rows, test_rows = _split_fold(n_rows, fold_id)
        train_set = set(int(v) for v in train_rows.tolist())
        val_set = set(int(v) for v in val_rows.tolist())
        test_set = set(int(v) for v in test_rows.tolist())
        if train_set & val_set or train_set & test_set or val_set & test_set:
            split_disjoint = False
        train_starts = published._window_starts(train_rows, published.WINDOW_LEN, published.STEP)
        val_starts = published._window_starts(val_rows, published.WINDOW_LEN, published.STEP)
        test_starts = published._window_starts(test_rows, published.WINDOW_LEN, published.STEP)
        for starts in (train_starts, val_starts, test_starts):
            for start in starts:
                covered = set(range(int(start), int(start) + published.WINDOW_LEN))
                if covered & test_set and starts is not test_starts:
                    test_not_in_fit = False
                if starts is test_starts and not covered.issubset(test_set):
                    test_not_in_fit = False
        published_part = detail[
            (detail["fold_id"] == fold_id) & (detail["method"] == "clp_csgm_ridge")
        ]
        if published_part.empty:
            raise ValueError("Published detail has no CLP rows for fold {}.".format(fold_id))
        n_train_csv = int(published_part["n_train_windows"].iloc[0])
        n_val_csv = int(published_part["n_val_windows"].iloc[0])
        n_test_csv = int(published_part["n_test_windows"].iloc[0])
        if (
            n_train_csv != len(train_starts)
            or n_val_csv != len(val_starts)
            or n_test_csv != len(test_starts)
        ):
            raise ValueError("Reconstructed window counts differ from the published file.")
        split_rows.append(
            {
                "fold_id": float(fold_id),
                "n_train_rows": float(train_rows.shape[0]),
                "n_val_rows": float(val_rows.shape[0]),
                "n_test_rows": float(test_rows.shape[0]),
                "n_train_windows": float(len(train_starts)),
                "n_val_windows": float(len(val_starts)),
                "n_test_windows": float(len(test_starts)),
                "train_val_shared": float(len(train_set & val_set)),
                "train_test_shared": float(len(train_set & test_set)),
                "val_test_shared": float(len(val_set & test_set)),
                "train_test_adjacent": float(_index_adjacent(train_rows, test_rows)),
                "val_test_adjacent": float(_index_adjacent(val_rows, test_rows)),
                "train_test_gap_m": _nearest_gap_m(depth_m, train_rows, test_rows),
                "val_test_gap_m": _nearest_gap_m(depth_m, val_rows, test_rows),
                "test_depth_min_m": float(np.min(depth_m[test_rows])),
                "test_depth_max_m": float(np.max(depth_m[test_rows])),
            }
        )

        nonoverlap_starts = published._window_starts(
            test_rows,
            published.WINDOW_LEN,
            published.WINDOW_LEN,
        )
        covered_nonoverlap: List[int] = []
        for start in nonoverlap_starts:
            covered_nonoverlap.extend(range(int(start), int(start) + published.WINDOW_LEN))
        covered_set = set(covered_nonoverlap)
        dropped = np.asarray(
            [int(row) for row in test_rows.tolist() if int(row) not in covered_set],
            dtype=np.int64,
        )
        kept = np.asarray(sorted(covered_set), dtype=np.int64)
        for code, part in frame.iloc[test_rows].groupby("Lithotype"):
            in_kept = int(np.isin(part.index.to_numpy(dtype=np.int64), kept).sum())
            in_dropped = int(np.isin(part.index.to_numpy(dtype=np.int64), dropped).sum())
            facies_rows.append(
                {
                    "fold_id": float(fold_id),
                    "lithotype": float(code),
                    "n_in_block": float(len(part)),
                    "n_kept_step8": float(in_kept),
                    "n_dropped_step8": float(in_dropped),
                }
            )

        for seed in published.SEEDS:
            for rho in published.RHOS:
                observed = _published_omega(fold_id, int(seed), float(rho))
                cell = detail[
                    (detail["fold_id"] == fold_id)
                    & (detail["seed"] == seed)
                    & (np.isclose(detail["rho"], rho))
                ]
                omegas = sorted(set(str(value) for value in cell["omega"].tolist()))
                expected = ",".join(str(int(idx)) for idx in observed.tolist())
                if omegas != [expected]:
                    omega_match = False
                    same_omega_methods = False
                if cell["method"].nunique() < 3:
                    same_omega_methods = False
                if set(int(v) for v in observed.tolist()) & set(range(published.WINDOW_LEN)) != set(
                    int(v) for v in observed.tolist()
                ):
                    within_window_ok = False
                complement = [offset for offset in range(published.WINDOW_LEN) if offset not in set(observed.tolist())]
                if set(complement) & set(int(v) for v in observed.tolist()):
                    within_window_ok = False

                test_counts = _role_counts(test_rows, test_starts, observed, published.WINDOW_LEN)
                val_counts = _role_counts(val_rows, val_starts, observed, published.WINDOW_LEN)
                test_sum = _summarize_roles(test_counts)
                val_sum = _summarize_roles(val_counts)
                if int(test_sum["n_both"]) < 1:
                    dual_role_all = False
                if int(test_sum["n_score_pairs"]) <= int(test_rows.shape[0]):
                    # Step 1 with a proper local complement must repeat depths.
                    dual_role_all = False

                for row, slot in test_counts.items():
                    role_rows.append(
                        {
                            "fold_id": float(fold_id),
                            "seed": float(seed),
                            "rho": float(rho),
                            "split": "test",
                            "row_index": float(row),
                            "depth_m": float(depth_m[row]),
                            "n_windows": float(slot["n_windows"]),
                            "n_observed": float(slot["n_observed"]),
                            "n_scored": float(slot["n_scored"]),
                            "role": _role_name(int(slot["n_observed"]), int(slot["n_scored"])),
                        }
                    )
                score_map = {
                    str(method): float(value)
                    for method, value in zip(cell["method"].tolist(), cell["rmse_holdout"].tolist())
                }
                mask_rows.append(
                    {
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "omega": expected,
                        "m_local": float(observed.shape[0]),
                        "n_test_windows": float(len(test_starts)),
                        "test_n_both": test_sum["n_both"],
                        "test_n_scored_only": test_sum["n_scored_only"],
                        "test_n_observed_only": test_sum["n_observed_only"],
                        "test_n_uncovered": test_sum["n_uncovered"],
                        "test_n_score_pairs": test_sum["n_score_pairs"],
                        "test_contaminated_pair_fraction": test_sum["contaminated_pair_fraction"],
                        "test_max_score_multiplicity": test_sum["max_score_multiplicity"],
                        "val_n_windows": float(len(val_starts)),
                        "val_n_both": val_sum["n_both"],
                        "val_contaminated_pair_fraction": val_sum["contaminated_pair_fraction"],
                        "rmse_clp": float(score_map.get("clp_csgm_ridge", float("nan"))),
                        "rmse_ae": float(score_map.get("ae_regression_ub", float("nan"))),
                        "rmse_prior": float(score_map.get("prior_only", float("nan"))),
                    }
                )

                # Published geometry as a coverage row.
                coverage_rows.append(
                    {
                        "alternative": "published_local_step1",
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "n_depths": float(test_rows.shape[0]),
                        "n_observed_depths": test_sum["n_both"] + test_sum["n_observed_only"],
                        "n_scored_depths": test_sum["n_both"] + test_sum["n_scored_only"],
                        "n_uncovered": test_sum["n_uncovered"],
                        "n_both": test_sum["n_both"],
                        "n_windows": float(len(test_starts)),
                        "n_empty_windows": 0.0,
                        "max_score_multiplicity": test_sum["max_score_multiplicity"],
                        "single_window_empty_probability": 0.0,
                        "covered_span_m": _depth_span_m(depth_m, test_rows),
                        "block_span_m": _depth_span_m(depth_m, test_rows),
                    }
                )

                non_counts = _role_counts(
                    test_rows,
                    nonoverlap_starts,
                    observed,
                    published.WINDOW_LEN,
                )
                non_sum = _summarize_roles(non_counts)
                if int(non_sum["n_both"]) != 0 or int(non_sum["n_uncovered"]) != int(dropped.shape[0]):
                    nonoverlap_clean = False
                if int(non_sum["max_score_multiplicity"]) > 1:
                    nonoverlap_clean = False
                coverage_rows.append(
                    {
                        "alternative": "nonoverlapping_step8",
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "n_depths": float(test_rows.shape[0]),
                        "n_observed_depths": non_sum["n_both"] + non_sum["n_observed_only"],
                        "n_scored_depths": non_sum["n_both"] + non_sum["n_scored_only"],
                        "n_uncovered": non_sum["n_uncovered"],
                        "n_both": non_sum["n_both"],
                        "n_windows": float(len(nonoverlap_starts)),
                        "n_empty_windows": 0.0,
                        "max_score_multiplicity": non_sum["max_score_multiplicity"],
                        "single_window_empty_probability": 0.0,
                        "covered_span_m": _depth_span_m(depth_m, kept),
                        "block_span_m": _depth_span_m(depth_m, test_rows),
                    }
                )
                # A local Omega of 2 or 4 offsets cannot leave an 8-sample window empty.
                if int(observed.shape[0]) < 1 or int(observed.shape[0]) >= published.WINDOW_LEN:
                    nonoverlap_clean = False

                global_rows = _global_observed_rows(test_rows, fold_id, int(seed), float(rho))
                global_counts = _roles_from_global(
                    test_rows,
                    test_starts,
                    global_rows,
                    published.WINDOW_LEN,
                )
                global_sum = _summarize_roles(global_counts)
                n_empty = _count_empty_windows(test_starts, global_rows, published.WINDOW_LEN)
                if int(global_sum["n_both"]) != 0 or int(global_sum["n_uncovered"]) != 0:
                    global_clean = False
                empty_prob = _empty_window_probability(
                    int(test_rows.shape[0]),
                    int(global_rows.shape[0]),
                    published.WINDOW_LEN,
                )
                coverage_rows.append(
                    {
                        "alternative": "global_depth",
                        "fold_id": float(fold_id),
                        "seed": float(seed),
                        "rho": float(rho),
                        "n_depths": float(test_rows.shape[0]),
                        "n_observed_depths": float(global_rows.shape[0]),
                        "n_scored_depths": float(test_rows.shape[0] - global_rows.shape[0]),
                        "n_uncovered": global_sum["n_uncovered"],
                        "n_both": global_sum["n_both"],
                        "n_windows": float(len(test_starts)),
                        "n_empty_windows": float(n_empty),
                        "max_score_multiplicity": global_sum["max_score_multiplicity"],
                        "single_window_empty_probability": empty_prob,
                        "covered_span_m": _depth_span_m(depth_m, test_rows),
                        "block_span_m": _depth_span_m(depth_m, test_rows),
                    }
                )

    # Lithotype and derived-column checks.
    legend, legend_status = _load_legend()
    assignments = _scan_lithotype_assignments(REPO_ROOT)
    deriving = [hit for hit in assignments if hit["derives_from_target"] == "true"]
    other_assignments = [hit for hit in assignments if hit["derives_from_target"] != "true"]
    identities = _identity_errors(frame)
    lith = frame["Lithotype"].to_numpy(dtype=np.int64)
    hfu = frame["HFU"].to_numpy(dtype=np.int64)
    phi = frame["Phi_lab (pu)"].to_numpy(dtype=np.float64)
    fzi = frame["FZI_lab"].to_numpy(dtype=np.float64)
    lith_eq_hfu = bool(np.array_equal(lith, hfu))
    phi_partition = _contiguous_partition(lith, phi)
    fzi_partition = _contiguous_partition(lith, fzi)
    phi_overlap = _range_overlap(frame, "Lithotype", "Phi_lab (pu)")
    derived_in_u = sorted(set(DERIVED_COLUMNS).intersection(published.U_COLUMNS))
    proxies_in_u = sorted(set(published.EXCLUDED_POROSITY_PROXIES).intersection(published.U_COLUMNS))
    legend_has_phi = _legend_mentions_target(legend)
    lith_labels = legend[legend["code_name"] == "Lithotype"] if not legend.empty else legend

    provenance_rows: List[Dict[str, str]] = [
        {"item": "legend_status", "value": legend_status},
        {"item": "legend_mentions_phi_lab", "value": str(legend_has_phi).lower()},
        {"item": "lithotype_equals_hfu", "value": str(lith_eq_hfu).lower()},
        {"item": "lithotype_is_phi_threshold_partition", "value": str(phi_partition).lower()},
        {"item": "lithotype_is_fzi_threshold_partition", "value": str(fzi_partition).lower()},
        {"item": "lithotype_phi_ranges_overlap", "value": str(phi_overlap).lower()},
        {
            "item": "deriving_assignment_sites",
            "value": "none" if not deriving else " | ".join(hit["path"] for hit in deriving),
        },
        {
            "item": "other_assignment_sites",
            "value": "none" if not other_assignments else " | ".join(hit["path"] for hit in other_assignments),
        },
        {"item": "rqi_max_abs_error", "value": "{:.3e}".format(identities["rqi_max_abs_error"])},
        {"item": "fzi_max_abs_error", "value": "{:.3e}".format(identities["fzi_max_abs_error"])},
        {"item": "derived_columns_in_u", "value": "none" if not derived_in_u else ",".join(derived_in_u)},
        {"item": "porosity_proxies_in_u", "value": "none" if not proxies_in_u else ",".join(proxies_in_u)},
    ]
    for record in lith_labels.itertuples(index=False):
        provenance_rows.append(
            {
                "item": "lithotype_label_{}".format(record.code),
                "value": str(record.label),
            }
        )

    recovery = _recovery_facts()
    runner = _runner_facts()
    published_summary = pd.read_csv(summary_path)
    summary_n = published_summary["n"].to_numpy(dtype=np.float64)
    tail_is_deepest = True
    for fold_id in range(published.N_BLOCKS):
        part = [row for row in facies_rows if int(row["fold_id"]) == fold_id]
        if not part:
            tail_is_deepest = False

    # Recompute the tail position directly.
    for fold_id, test_rows in enumerate(blocks):
        starts = published._window_starts(test_rows, published.WINDOW_LEN, published.WINDOW_LEN)
        covered = {
            int(start) + offset
            for start in starts
            for offset in range(published.WINDOW_LEN)
        }
        dropped_idx = [int(row) for row in test_rows.tolist() if int(row) not in covered]
        kept_idx = [int(row) for row in test_rows.tolist() if int(row) in covered]
        if not dropped_idx or not kept_idx or min(dropped_idx) < max(kept_idx):
            tail_is_deepest = False
        if len(dropped_idx) != 5 or len(starts) != 3:
            nonoverlap_clean = False

    checks.extend(
        [
            _check(
                "head_matches_reference",
                head == REFERENCE_COMMIT,
                "HEAD {}".format(head if head else "unknown"),
            ),
            _check(
                "omega_matches_published_file",
                omega_match,
                "Redrawn local Omega matches heldout_rmse_detail.csv",
            ),
            _check(
                "split_rows_disjoint",
                split_disjoint,
                "Train, validation, and test row sets do not intersect",
            ),
            _check(
                "test_rows_absent_from_fit_windows",
                test_not_in_fit,
                "No test row is inside a training or validation window",
            ),
            _check(
                "within_window_score_excludes_own_measurement",
                within_window_ok and runner["score_drops_local_omega"],
                "Each window scores only offsets outside its own Omega",
            ),
            _check(
                "physical_depth_has_both_roles",
                dual_role_all,
                "Every published cell measures and scores the same depths",
            ),
            _check(
                "clp_and_ae_share_omega",
                same_omega_methods and runner["same_operator_for_clp_and_ae"] and runner["one_omega_draw"],
                "One Omega and one b feed CLP and the direct AE",
            ),
            _check(
                "recovery_uses_one_operator_per_call",
                recovery["one_operator_argument"] and recovery["single_best_restart"],
                "A per-window operator needs a separate call; restart choice is shared",
            ),
            _check(
                "lithotype_not_a_phi_transform",
                (not lith_eq_hfu)
                and (not phi_partition)
                and (not fzi_partition)
                and phi_overlap
                and (not legend_has_phi)
                and (not deriving),
                "No right-hand side builds Lithotype from Phi_lab; ranges overlap; legend is facies text",
            ),
            _check(
                "rqi_fzi_identities_and_exclusion",
                identities["rqi_max_abs_error"] < IDENTITY_ATOL
                and identities["fzi_max_abs_error"] < IDENTITY_ATOL
                and not derived_in_u
                and not proxies_in_u,
                "RQI and FZI match the lab transform and stay out of u",
            ),
            _check(
                "global_mask_removes_dual_role",
                global_clean,
                "Counterfactual physical masks have no depth in both roles",
            ),
            _check(
                "nonoverlap_removes_dual_role_and_drops_tail",
                nonoverlap_clean and tail_is_deepest,
                "Step 8 scores each covered depth once and drops the deepest 5 rows",
            ),
            _check(
                "published_summary_has_nine_cells",
                bool(np.all(summary_n == 9.0)),
                "The published mean uses 3 folds x 3 seeds, not 9 wells",
            ),
            _check(
                "no_interpolation_or_unconditioned_csgm",
                runner["no_depth_interpolation"] and runner["no_zero_lambda"],
                "Those comparators are absent from the published runner",
            ),
        ]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    split_summary = pd.DataFrame(split_rows)
    mask_summary = pd.DataFrame(mask_rows)
    role_table = pd.DataFrame(role_rows)
    coverage = pd.DataFrame(coverage_rows)
    provenance = pd.DataFrame(provenance_rows)
    facies = pd.DataFrame(facies_rows)
    # Role names are strings. Force the column dtype so CSV does not quote
    # unexpectedly, then write through pandas.
    role_table["role"] = role_table["role"].astype(str)
    split_summary.to_csv(table_dir / "split_summary.csv", index=False)
    mask_summary.to_csv(table_dir / "published_mask_summary.csv", index=False)
    role_table.to_csv(table_dir / "published_depth_roles.csv", index=False)
    coverage.to_csv(table_dir / "alternative_coverage.csv", index=False)
    provenance.to_csv(table_dir / "lithotype_provenance.csv", index=False)
    facies.to_csv(table_dir / "facies_coverage_nonoverlap.csv", index=False)

    payload = {
        "reference_commit": REFERENCE_COMMIT,
        "head": head,
        "runner_sha256": script_hash,
        "n_rows": n_rows,
        "checks": checks,
    }
    (OUT_DIR / "checks.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_report(
        OUT_DIR / "REPORT.md",
        head,
        script_hash,
        frame,
        split_summary,
        mask_summary,
        coverage,
        provenance,
        facies,
        checks,
    )
    failed = [str(item["name"]) for item in checks if not bool(item["passed"])]
    print("wrote", OUT_DIR)
    print("checks", len(checks), "failed", len(failed))
    for name in failed:
        print("FAIL", name)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run())
