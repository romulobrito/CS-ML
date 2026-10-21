#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Etapa 3 (roadmap): varre eixos de robustez com perfis robustness_phase3*.

Uso:
  python run_robustness_phase3_sweep.py --explore
  python run_robustness_phase3_sweep.py
  python run_robustness_phase3_sweep.py --axes residual_k measurement_noise_std

Artefactos: outputs/robustness_phase3/<axis>/v_<slug>/runs/<run_id>/
Figuras: paper/figures/robustness_phase3/...
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

# Roadmap Etapa 3A (uma variavel de cada vez; valores como strings para CLI).
ROBUSTNESS_GRID: Dict[str, List[str]] = {
    "residual_k": ["2", "4", "6", "8", "12", "16"],
    "measurement_noise_std": ["0", "0.01", "0.02", "0.05", "0.1"],
    "residual_amplitude": ["0.4", "0.8", "1.2", "1.6", "2.0"],
    "output_noise_std": ["0", "0.005", "0.01", "0.02", "0.05"],
    "measurement_ratio": ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.8"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep robustness_phase3 (Etapa 3 roadmap).")
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Use robustness_phase3_explore (1 seed, 2 rho, LFISTA epochs reduzidos).",
    )
    parser.add_argument(
        "--axes",
        nargs="*",
        default=list(ROBUSTNESS_GRID.keys()),
        metavar="AXIS",
        help="Subconjunto de eixos (default: todos).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Imprime comandos sem executar.",
    )
    args = parser.parse_args()

    profile = "robustness_phase3_explore" if args.explore else "robustness_phase3"
    root = os.path.dirname(os.path.abspath(__file__))
    pipeline = os.path.join(root, "sir_cs_pipeline_optimized.py")
    manifest_path = os.path.join(root, "outputs", "robustness_phase3", "sweep_manifest.csv")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    rows: List[Dict[str, str]] = []
    for axis in args.axes:
        if axis not in ROBUSTNESS_GRID:
            print("Unknown axis: " + axis, file=sys.stderr)
            sys.exit(2)
        for val in ROBUSTNESS_GRID[axis]:
            cmd = [
                sys.executable,
                pipeline,
                "--profile",
                profile,
                "--robustness-axis",
                axis,
                "--robustness-value",
                val,
            ]
            if args.dry_run:
                print(" ".join(cmd))
                continue
            t0 = time.time()
            print("RUN " + " ".join(cmd), flush=True)
            proc = subprocess.run(cmd, cwd=root)
            elapsed = time.time() - t0
            rows.append(
                {
                    "axis": axis,
                    "value": val,
                    "profile": profile,
                    "exit_code": str(proc.returncode),
                    "elapsed_s": f"{elapsed:.1f}",
                }
            )
            if proc.returncode != 0:
                print("FAILED exit=" + str(proc.returncode), flush=True)
                break
        else:
            continue
        break

    if args.dry_run:
        return

    with open(manifest_path, "w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=["axis", "value", "profile", "exit_code", "elapsed_s"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    meta = {
        "profile": profile,
        "axes": args.axes,
        "n_runs": len(rows),
        "manifest_csv": manifest_path,
    }
    with open(
        os.path.join(root, "outputs", "robustness_phase3", "sweep_meta.json"),
        "w",
        encoding="utf-8",
    ) as fp:
        json.dump(meta, fp, indent=2)

    print("Wrote " + manifest_path + " (" + str(len(rows)) + " rows)", flush=True)


if __name__ == "__main__":
    main()
