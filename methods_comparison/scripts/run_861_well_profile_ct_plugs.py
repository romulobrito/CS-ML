#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 1d orchestrator: well_profile + ct_plugs ML tracks.

Parallel tracks:
  1. well_profile: Phi_lab RF + five-regressor depth-block CV (87 wireline rows)
  2. ct_plugs: Phi_lab + FZI_lab, wireline_only vs wireline_plus_ct (10 plugs LOO)

Does NOT re-run FZI on well_profile (deprecated per diagnostics).

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
    CT_PLUGS_ML_ROOT,
    WELL_PROFILE_ML_ROOT,
    WELL_PROFILE_PRIMARY_TARGET,
    well_profile_phi_compare_dir,
    well_profile_phi_rf_dir,
)

ROOT = SCRIPT_DIR.parents[1]
LOG_PATH = ROOT / "methods_comparison" / "planning" / "agent_poco861.log"


def write_well_profile_manifest() -> None:
    """List well_profile artifacts."""
    out_root = WELL_PROFILE_ML_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    files = sorted(
        str(p.relative_to(out_root))
        for p in out_root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.txt"
    )
    lines = [
        "Well 861 well_profile ML manifest (depth-block CV, 87 rows)",
        "Generated: {}".format(ts),
        "Target: {}".format(WELL_PROFILE_PRIMARY_TARGET),
        "CV: depth-block",
        "",
        "Artifacts:",
    ]
    lines.extend(["  {}".format(f) for f in files])
    (out_root / "MANIFEST.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_agent_log(section: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = "\n{}\nTimestamp: {}\n{}\n".format("=" * 80, ts, section)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(block)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Etapa 1d: well_profile + ct_plugs orchestrator (Well 861)"
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--skip-well-profile",
        action="store_true",
        help="Skip well_profile Phi_lab track (87 rows, depth-block CV)",
    )
    parser.add_argument(
        "--skip-ct-plugs",
        action="store_true",
        help="Skip ct_plugs track (10 samples, leave-one-plug-out)",
    )
    parser.add_argument("--skip-phi-rf", action="store_true")
    parser.add_argument("--skip-phi-compare", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    py = sys.executable
    smoke = ["--smoke"] if args.smoke else []
    shap_flag = ["--no-shap"] if args.smoke else []

    if not args.skip_well_profile:
        rf_dir = well_profile_phi_rf_dir()
        cmp_dir = well_profile_phi_compare_dir()

        if not args.skip_phi_rf:
            cmd_rf = [
                py,
                str(SCRIPT_DIR / "run_phi_lab_rf_861.py"),
                "--out-dir",
                str(rf_dir),
                *smoke,
                *shap_flag,
            ]
            print("RUN well_profile RF:", " ".join(cmd_rf))
            subprocess.run(cmd_rf, check=True)

        if not args.skip_phi_compare:
            cmd_cmp = [
                py,
                str(SCRIPT_DIR / "run_861_ml_baseline.py"),
                "--target",
                WELL_PROFILE_PRIMARY_TARGET,
                "--out-dir",
                str(cmp_dir),
                *smoke,
            ]
            print("RUN well_profile compare:", " ".join(cmd_cmp))
            subprocess.run(cmd_cmp, check=True)

        write_well_profile_manifest()

    if not args.skip_ct_plugs:
        cmd_ct = [
            py,
            str(SCRIPT_DIR / "run_861_ct_plugs_baseline.py"),
            "--out-dir",
            str(CT_PLUGS_ML_ROOT),
            *smoke,
        ]
        print("RUN ct_plugs:", " ".join(cmd_ct))
        subprocess.run(cmd_ct, check=True)

    lines = [
        "ETAPA 1d -- well_profile + ct_plugs",
        "well_profile out: {}".format(WELL_PROFILE_ML_ROOT),
        "ct_plugs out: {}".format(CT_PLUGS_ML_ROOT),
        "Smoke: {}".format(args.smoke),
    ]
    append_agent_log("\n".join(lines))

    print("OK Etapa 1d complete")
    print("  well_profile: {}".format(WELL_PROFILE_ML_ROOT))
    print("  ct_plugs: {}".format(CT_PLUGS_ML_ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
