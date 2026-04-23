#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge several sir_cs_benchmark_direct_ub.py tables/summary.csv files after
varying residual_k only (same Psi, M kind, measurement_noise_std, seeds, rho grid).

Each output row is one (measurement_ratio, method, ...) from the input plus
column residual_k (int).

ASCII-only.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Merge direct_ub summary.csv across residual_k arms.")
    p.add_argument(
        "--arm",
        action="append",
        nargs=2,
        metavar=("K", "CSV"),
        default=[],
        help="Repeatable: residual_k (int string) and path to tables/summary.csv.",
    )
    p.add_argument("--out", type=str, required=True, help="Output long-format CSV path.")
    args = p.parse_args()
    if not args.arm:
        print("Provide at least one --arm K PATH.", file=sys.stderr)
        sys.exit(2)
    frames: list[pd.DataFrame] = []
    for k_str, path in args.arm:
        path = path.strip()
        k_str = k_str.strip()
        try:
            k_val = int(k_str)
        except ValueError:
            print(f"Invalid residual_k: {k_str!r}", file=sys.stderr)
            sys.exit(2)
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            print(f"Missing file: {path}", file=sys.stderr)
            sys.exit(2)
        if "measurement_ratio" not in df.columns or "method" not in df.columns:
            print(f"Unexpected columns in {path}", file=sys.stderr)
            sys.exit(2)
        df = df.copy()
        df["residual_k"] = k_val
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    p_out = args.out.strip()
    parent = os.path.dirname(p_out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    out.to_csv(p_out, index=False)
    print(f"Wrote {len(out)} rows to {p_out}", flush=True)


if __name__ == "__main__":
    main()
