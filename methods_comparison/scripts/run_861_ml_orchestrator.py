#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate Well 861 ML Etapa 1c: FZI RF + five-regressor comparison.

ASCII-only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
ML_RUNS = ROOT / "methods_comparison" / "data" / "processed" / "ml_runs"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Orchestrate Well 861 ML baseline (Etapa 1c)")
    parser.add_argument("--target", type=str, default="FZI_lab")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    py = sys.executable
    smoke_flag = ["--smoke"] if args.smoke else []

    if not args.skip_rf:
        cmd_rf = [py, str(SCRIPT_DIR / "run_fzi_rf_861.py"), *smoke_flag, "--no-shap"] if args.smoke else [
            py,
            str(SCRIPT_DIR / "run_fzi_rf_861.py"),
            *smoke_flag,
        ]
        print("RUN:", " ".join(cmd_rf))
        subprocess.run(cmd_rf, check=True)

    if not args.skip_compare:
        cmd_cmp = [
            py,
            str(SCRIPT_DIR / "run_861_ml_baseline.py"),
            "--target",
            args.target,
            *smoke_flag,
        ]
        print("RUN:", " ".join(cmd_cmp))
        subprocess.run(cmd_cmp, check=True)

    print("OK orchestration complete out_dir={}".format(ML_RUNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
