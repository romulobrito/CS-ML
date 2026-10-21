#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two sir_cs_benchmark_direct_ub.py summary.csv files: Gaussian M vs subsample M
(typically both with same residual_basis, e.g. dct).

Long-format output: all columns from summary.csv plus measurement_kind label.

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge Gaussian and subsample direct_ub summary.csv (e.g. DCT for both)."
    )
    p.add_argument(
        "--gaussian-dct-summary",
        type=str,
        required=True,
        help="Path to tables/summary.csv for gaussian M (e.g. frozen DCT run).",
    )
    p.add_argument(
        "--subsample-dct-summary",
        type=str,
        required=True,
        help="Path to tables/summary.csv for subsample M.",
    )
    p.add_argument("--out", type=str, required=True, help="Output CSV path (parent dirs created).")
    args = p.parse_args()

    p_g = args.gaussian_dct_summary.strip()
    p_s = args.subsample_dct_summary.strip()
    p_o = args.out.strip()

    def _load(path: str, label: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing file ({label}): {path}", file=sys.stderr)
            sys.exit(2)
        if "measurement_ratio" not in df.columns or "method" not in df.columns:
            print(f"Unexpected columns in {path}", file=sys.stderr)
            sys.exit(2)
        return df

    df_g = _load(p_g, "gaussian")
    df_g["measurement_kind"] = "gaussian"
    df_s = _load(p_s, "subsample")
    df_s["measurement_kind"] = "subsample"
    out = pd.concat([df_g, df_s], ignore_index=True)
    parent = os.path.dirname(p_o)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out.to_csv(p_o, index=False)
    print(f"Wrote {len(out)} rows to {p_o}", flush=True)


if __name__ == "__main__":
    main()
