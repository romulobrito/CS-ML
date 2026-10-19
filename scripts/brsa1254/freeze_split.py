#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze depth blocks, validation plugs, and observation masks.

No model is trained here. The protocol file is written before any test score.
ASCII only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    EXPECTED_CANDIDATES,
    N_BLOCKS,
    OUT_DIR,
    RHOS,
    SEEDS,
    U_COLUMNS,
    VAL_PLUGS,
    WINDOW_LEN,
    WINDOW_STEP,
    mask_seed,
    observation_count,
)


BLOCK_SIZES: Sequence[int] = (53, 54, 54)


def _block_indices(n_plugs: int) -> List[np.ndarray]:
    """Contiguous blocks of 53, 54 and 54 plugs in depth order.

    numpy.array_split puts the remainder in the earlier blocks and would
    yield 54, 54, 53. The study freezes the counted sizes instead.
    """
    if int(n_plugs) != int(sum(BLOCK_SIZES)):
        raise ValueError(
            "Kept plug count {0} does not match blocks {1}.".format(n_plugs, BLOCK_SIZES)
        )
    if int(n_plugs) != EXPECTED_CANDIDATES:
        raise ValueError("Expected {0} kept plugs.".format(EXPECTED_CANDIDATES))
    edges = [0]
    for size in BLOCK_SIZES:
        edges.append(edges[-1] + int(size))
    return [np.arange(edges[i], edges[i + 1], dtype=np.int64) for i in range(N_BLOCKS)]


def _deeper_training_block(blocks: Sequence[np.ndarray], test_fold: int) -> np.ndarray:
    """Training block whose last plug is deepest."""
    pool = [block for fold_id, block in enumerate(blocks) if fold_id != int(test_fold)]
    return max(pool, key=lambda block: int(block[-1]))


def _draw_roles(indices: np.ndarray, n_observed: int, seed: int) -> Dict[int, str]:
    """Mark a frozen subset as observed and the complement as scored."""
    if n_observed < 1 or n_observed >= int(indices.shape[0]):
        raise ValueError("Observation count must leave at least one scored plug.")
    rng = np.random.default_rng(int(seed))
    picked = rng.choice(indices, size=int(n_observed), replace=False)
    observed = set(int(value) for value in picked.tolist())
    roles: Dict[int, str] = {}
    for plug_index in indices.tolist():
        roles[int(plug_index)] = "observed" if int(plug_index) in observed else "scored"
    return roles


def main() -> None:
    """Write the frozen split, masks, protocol, and config."""
    aligned_path = OUT_DIR / "tables" / "plug_log_aligned.csv"
    selection_path = OUT_DIR / "selection.json"
    if not aligned_path.is_file() or not selection_path.is_file():
        raise FileNotFoundError("Run build_plug_log_table.py first.")
    aligned = pd.read_csv(aligned_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if int(selection["n_dropped"]) != 0:
        raise ValueError("Depth match dropped plugs; block sizes would change.")
    if list(aligned["plug_index"]) != list(range(int(aligned.shape[0]))):
        raise ValueError("plug_index is not a depth-sorted range.")
    if not aligned["depth_m"].is_monotonic_increasing:
        raise ValueError("Aligned plugs are not sorted by depth.")

    blocks = _block_indices(int(aligned.shape[0]))
    fold_rows: List[Dict[str, float]] = []
    mask_rows: List[Dict[str, float]] = []
    for fold_id, test_index in enumerate(blocks):
        deeper = _deeper_training_block(blocks, fold_id)
        if int(deeper.shape[0]) <= VAL_PLUGS + WINDOW_LEN:
            raise ValueError("Deeper training block cannot hold validation and a window.")
        val_index = deeper[-VAL_PLUGS:]
        train_pool = np.concatenate(
            [block for i, block in enumerate(blocks) if i != fold_id]
        )
        train_index = np.setdiff1d(train_pool, val_index, assume_unique=False)
        if set(train_index.tolist()) & set(test_index.tolist()):
            raise ValueError("Train and test plugs overlap.")
        if set(val_index.tolist()) & set(test_index.tolist()):
            raise ValueError("Validation and test plugs overlap.")
        if set(train_index.tolist()) & set(val_index.tolist()):
            raise ValueError("Train and validation plugs overlap.")
        for split_name, indices in (
            ("train", train_index),
            ("val", val_index),
            ("test", test_index),
        ):
            for plug_index in indices.tolist():
                depth_m = float(aligned.loc[int(plug_index), "depth_m"])
                fold_rows.append(
                    {
                        "fold_id": float(fold_id),
                        "plug_index": float(plug_index),
                        "split": split_name,
                        "depth_m": depth_m,
                    }
                )
        for seed in SEEDS:
            for rho in RHOS:
                for split_name, indices in (("val", val_index), ("test", test_index)):
                    n_observed = observation_count(int(indices.shape[0]), float(rho))
                    roles = _draw_roles(
                        indices,
                        n_observed,
                        mask_seed(int(seed), fold_id, float(rho)) + (0 if split_name == "test" else 50),
                    )
                    for plug_index, role in roles.items():
                        mask_rows.append(
                            {
                                "fold_id": float(fold_id),
                                "seed": float(seed),
                                "rho": float(rho),
                                "split": split_name,
                                "plug_index": float(plug_index),
                                "role": role,
                            }
                        )

    table_dir = OUT_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(table_dir / "split_by_plug.csv", index=False)
    pd.DataFrame(mask_rows).to_csv(table_dir / "masks.csv", index=False)

    block_text = []
    for fold_id, indices in enumerate(blocks):
        depths = aligned.loc[indices, "depth_m"]
        block_text.append(
            "block {0}: n={1}, depth {2:.2f} to {3:.2f} m".format(
                fold_id,
                int(indices.shape[0]),
                float(depths.min()),
                float(depths.max()),
            )
        )
    protocol_lines = [
        "Well 9-BRSA-1254-RJS plug holdout.",
        "Question: do dense logs help estimate held-out laboratory plugs?",
        "Score: RMSE of log10(k) on reserved plugs, once per plug.",
        "No dense permeability curve is built for training or for the score.",
        "Pass choice used depth coverage only: {0}, frame {1}, resistivity {2}.".format(
            selection["pass_name"],
            selection["frame_name"],
            selection["resistivity"],
        ),
        "Matched plugs: {0}. Dropped for depth or non-finite logs: {1}.".format(
            selection["n_kept"],
            selection["n_dropped"],
        ),
        "Target: log10 of positive longitudinal permeability, direction VH1.",
        "Context u: gr, rhob, nphi, rt. Laboratory porosity and density stay out.",
        "NMR permeability is not used.",
        "Blocks are contiguous in depth with sizes 53, 54, 54.",
        "numpy.array_split would instead yield 54, 54, 53; those sizes are not used.",
        "Each block is the test once.",
        "Validation: the 16 deepest plugs of the deeper training block.",
        "Those validation plugs are excluded from decoder training.",
        "Scalers and the direct regression use the training plugs only.",
        "Inside a test or validation block, rho in {0.25, 0.50} marks observed plugs.",
        "Seeds 7, 23, 41 draw the masks. The mask is not chosen from the test score.",
        "Mask stream: seed + 1009 * fold + 17 * round(100 * rho), plus 50 on validation.",
        "Train windows: 8 successive training plugs, step 1, fully inside the train set.",
        "Interpolation and smoothing use physical depth in meters.",
        "Smoothing alpha and CLP lambda are chosen on validation plugs only.",
        "The same observed plugs are given to every comparator.",
        "This run does not edit the F3 grid or the manuscript.",
        "It does not claim a general win.",
        "",
    ]
    protocol_lines.extend(block_text)
    protocol_lines.append("")
    (OUT_DIR / "PROTOCOL.txt").write_text("\n".join(protocol_lines), encoding="utf-8")
    config = {
        "pass_name": selection["pass_name"],
        "frame_name": selection["frame_name"],
        "resistivity": selection["resistivity"],
        "n_kept": int(selection["n_kept"]),
        "n_dropped": int(selection["n_dropped"]),
        "block_sizes": list(BLOCK_SIZES),
        "val_plugs": VAL_PLUGS,
        "window_len": WINDOW_LEN,
        "window_step": WINDOW_STEP,
        "rhos": list(RHOS),
        "seeds": list(SEEDS),
        "u_columns": list(U_COLUMNS),
        "target": "log10_k",
        "models_trained_by_this_script": False,
    }
    (OUT_DIR / "config_frozen.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\n".join(block_text))
    print("wrote", OUT_DIR / "PROTOCOL.txt")


if __name__ == "__main__":
    main()
