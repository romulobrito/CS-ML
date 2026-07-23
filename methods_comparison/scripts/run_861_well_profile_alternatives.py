#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate well_profile model alternatives: Phi Ridge/GAM + HFU classifiers.

ASCII-only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from ml_861_data import (
    WELL_PROFILE_ML_ROOT,
    well_profile_hfu_classifier_dir,
    well_profile_phi_alternatives_dir,
    well_profile_phi_rf_dir,
)

ROOT = SCRIPT_DIR.parents[1]
LOG_PATH = ROOT / "methods_comparison" / "planning" / "agent_poco861.log"


def append_agent_log(section: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = "\n{}\nTimestamp: {}\n{}\n".format("=" * 80, ts, section)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(block)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phi Ridge/GAM + HFU classifiers (well_profile depth-block CV)"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-phi-rf", action="store_true", help="Skip RF refresh before compare")
    parser.add_argument("--skip-phi-alt", action="store_true")
    parser.add_argument("--skip-hfu", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    py = sys.executable
    smoke = ["--smoke"] if args.smoke else []

    if not args.skip_phi_rf:
        cmd_rf = [py, str(SCRIPT_DIR / "run_phi_lab_rf_861.py"), *smoke]
        print("RUN:", " ".join(cmd_rf))
        subprocess.run(cmd_rf, check=True)

    if not args.skip_phi_alt:
        cmd_phi = [
            py,
            str(SCRIPT_DIR / "run_861_phi_alternatives.py"),
            "--out-dir",
            str(well_profile_phi_alternatives_dir()),
            *smoke,
        ]
        print("RUN:", " ".join(cmd_phi))
        subprocess.run(cmd_phi, check=True)

    if not args.skip_hfu:
        cmd_hfu = [
            py,
            str(SCRIPT_DIR / "run_861_hfu_classifier.py"),
            "--out-dir",
            str(well_profile_hfu_classifier_dir()),
            *smoke,
        ]
        print("RUN:", " ".join(cmd_hfu))
        subprocess.run(cmd_hfu, check=True)

    lines = [
        "well_profile alternatives: Phi Ridge/GAM + HFU classifiers",
        "phi alternatives: {}".format(well_profile_phi_alternatives_dir()),
        "hfu classifier: {}".format(well_profile_hfu_classifier_dir()),
        "rf baseline: {}".format(well_profile_phi_rf_dir()),
        "smoke: {}".format(args.smoke),
    ]
    append_agent_log("\n".join(lines))

    print("OK well_profile alternatives complete")
    print("  phi: {}".format(well_profile_phi_alternatives_dir()))
    print("  hfu: {}".format(well_profile_hfu_classifier_dir()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
