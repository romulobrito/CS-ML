#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator: Etapa 2a POC (10 plugs) + Etapa 2c/2d profile (87 rows).

ASCII-only.
"""

from __future__ import annotations

import argparse

from run_861_dem_sc_poc_plugs import run_poc
from run_861_dem_sc_profile_87 import run_profile_pipeline


def parse_args() -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(description="Well 861 DEM/SC full pipeline")
    parser.add_argument("--smoke", action="store_true", help="Quick test on subset")
    parser.add_argument(
        "--profile-only",
        action="store_true",
        help="Skip POC; require existing hfu_calibration/hfu_ct_stats.csv",
    )
    return parser.parse_args()


def main() -> None:
    """Run POC then profile extrapolation."""
    args = parse_args()
    if not args.profile_only:
        poc = run_poc(smoke=args.smoke)
        print("POC: {}/{} plugs".format(poc["n_ok"], poc["metrics"]["n_plugs"]))
    profile = run_profile_pipeline(smoke=args.smoke)
    print("Profile: {}/{} rows".format(profile["n_ok"], profile["metrics"]["n_rows"]))


if __name__ == "__main__":
    main()
