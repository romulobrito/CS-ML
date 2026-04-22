#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge two sir_cs_benchmark_direct_ub.py summary.csv files (identity vs dct).

Long-format output: all columns from summary.csv plus residual_basis.

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Merge identity and dct direct_ub summary.csv.")
    p.add_argument("--identity-summary", type=str, required=True, help="Path to identity run tables/summary.csv")
    p.add_argument("--dct-summary", type=str, required=True, help="Path to dct run tables/summary.csv")
    p.add_argument("--out", type=str, required=True, help="Output CSV path (parent dirs created)")
    args = p.parse_args()

    p_i = args.identity_summary.strip()
    p_d = args.dct_summary.strip()
    p_o = args.out.strip()

    def _load(path: str, label: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing {label} file: {path}", file=sys.stderr)
            sys.exit(2)
        if "measurement_ratio" not in df.columns or "method" not in df.columns:
            print(f"Unexpected columns in {path}", file=sys.stderr)
            sys.exit(2)
        return df

    df_i = _load(p_i, "identity")
    df_i["residual_basis"] = "identity"
    df_d = _load(p_d, "dct")
    df_d["residual_basis"] = "dct"
    out = pd.concat([df_i, df_d], ignore_index=True)
    parent = os.path.dirname(p_o)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out.to_csv(p_o, index=False)
    print(f"Wrote {len(out)} rows to {p_o}", flush=True)


if __name__ == "__main__":
    main()
