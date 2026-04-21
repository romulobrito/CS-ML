#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build LaTeX fragments for robustness Phase 3 tables from summary_focus CSVs.
ASCII only. Run from repo root: python paper/build_robustness_tables.py
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PAPER, "robustness_tables_fragment.tex")
OUT_RHO060 = os.path.join(PAPER, "robustness_tables_fragment_rho060.tex")
OUT_MEAS = os.path.join(PAPER, "robustness_tables_fragment_measurement_ratio.tex")


def _axis_val_from_path(path: str) -> Tuple[str, str]:
    rel = path.split("outputs/robustness_phase3/")[1]
    parts = rel.split("/")
    axis = parts[0]
    vdir = parts[1]
    m = re.match(r"^v_(.+)$", vdir)
    val_slug = m.group(1) if m else vdir
    return axis, val_slug


def _slug_to_display(axis: str, slug: str) -> str:
    if slug.replace("p", "").replace("m", "").isalnum() or slug[0].isdigit():
        return slug.replace("p", ".").replace("m", "-")
    return slug


def collect_rows() -> List[Dict[str, object]]:
    pattern = os.path.join(ROOT, "outputs", "robustness_phase3", "**", "summary_focus_ml_hybrid_fista_lfista.csv")
    paths = sorted(glob.glob(pattern, recursive=True))
    skip_sub = ("LATEST", "20260418_142843")
    rows: List[Dict[str, object]] = []
    for p in paths:
        if any(s in p for s in skip_sub):
            continue
        axis, slug = _axis_val_from_path(p)
        df = pd.read_csv(p)
        if axis == "measurement_ratio":
            for rho, g in df.groupby("measurement_ratio"):
                m = g.set_index("method")["rmse_mean"]
                rows.append(
                    {
                        "axis": axis,
                        "param_display": f"{float(rho):.2f}",
                        "param_sort": float(rho),
                        "path": p,
                        "fista": float(m["hybrid_fista"]),
                        "joint": float(m["hybrid_lfista_joint"]),
                        "frozen": float(m["hybrid_lfista_frozen"]),
                        "ml_only": float(m["ml_only"]),
                    }
                )
        else:
            g = df[df["measurement_ratio"] == 0.6]
            if len(g) < 4:
                continue
            m = g.set_index("method")["rmse_mean"]
            disp = _slug_to_display(axis, slug)
            try:
                psort: float | str = float(disp)
            except ValueError:
                psort = disp
            rows.append(
                {
                    "axis": axis,
                    "param_display": disp,
                    "param_sort": psort,
                    "path": p,
                    "fista": float(m["hybrid_fista"]),
                    "joint": float(m["hybrid_lfista_joint"]),
                    "frozen": float(m["hybrid_lfista_frozen"]),
                    "ml_only": float(m["ml_only"]),
                }
            )
    return rows


def fmt(x: float) -> str:
    return f"{x:.3f}"


def emit_table(axis: str, sub: List[Dict[str, object]], f) -> None:
    sub_sorted = sorted(sub, key=lambda r: r["param_sort"])
    cap = {
        "residual_k": "Test RMSE at $\\rho=0.6$ when varying residual sparsity $k$ (other settings Phase~0 baseline).",
        "measurement_noise_std": "Test RMSE at $\\rho=0.6$ when varying measurement noise std.",
        "residual_amplitude": "Test RMSE at $\\rho=0.6$ when varying innovation amplitude.",
        "output_noise_std": "Test RMSE at $\\rho=0.6$ when varying output noise std.",
        "measurement_ratio": (
            "Test RMSE when the sweep fixes a single $\\rho=m/N$ (one column per run). "
            "At $\\rho=0.80$, \\texttt{hybrid\\_fista} is marginally better than \\texttt{hybrid\\_lfista\\_joint}."
        ),
    }.get(axis, axis)
    label = "tab:robust_" + axis.replace("_", "")
    f.write("\\begin{table}[t]\n")
    f.write("  \\centering\n")
    f.write("  \\caption{" + cap + "}\n")
    f.write("  \\label{" + label + "}\n")
    f.write("  \\small\n")
    f.write("  \\begin{tabular}{@{}lcccc@{}}\n")
    f.write("    \\toprule\n")
    if axis == "measurement_ratio":
        f.write("    $\\rho=m/N$ & \\texttt{hybrid\\_fista} & \\texttt{hybrid\\_lfista\\_joint} & \\texttt{hybrid\\_lfista\\_frozen} & \\texttt{ml\\_only} \\\\\n")
    else:
        f.write("    Setting & \\texttt{hybrid\\_fista} & \\texttt{hybrid\\_lfista\\_joint} & \\texttt{hybrid\\_lfista\\_frozen} & \\texttt{ml\\_only} \\\\\n")
    f.write("    \\midrule\n")
    for r in sub_sorted:
        f.write(
            "    "
            + str(r["param_display"])
            + " & "
            + fmt(float(r["fista"]))
            + " & "
            + fmt(float(r["joint"]))
            + " & "
            + fmt(float(r["frozen"]))
            + " & "
            + fmt(float(r["ml_only"]))
            + " \\\\\n"
        )
    f.write("    \\bottomrule\n")
    f.write("  \\end{tabular}\n")
    f.write("\\end{table}\n\n")


def main() -> None:
    rows = collect_rows()
    by_axis: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        by_axis.setdefault(str(r["axis"]), []).append(r)
    order = [
        "residual_k",
        "measurement_noise_std",
        "residual_amplitude",
        "output_noise_std",
        "measurement_ratio",
    ]
    header = "% Auto-generated by paper/build_robustness_tables.py (do not edit by hand).\n\n"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(header)
        for axis in order:
            if axis not in by_axis:
                continue
            emit_table(axis, by_axis[axis], f)
    with open(OUT_RHO060, "w", encoding="utf-8") as f:
        f.write(header)
        for axis in order[:-1]:
            if axis not in by_axis:
                continue
            emit_table(axis, by_axis[axis], f)
    if "measurement_ratio" in by_axis:
        with open(OUT_MEAS, "w", encoding="utf-8") as f:
            f.write(header)
            emit_table("measurement_ratio", by_axis["measurement_ratio"], f)
    print("Wrote " + OUT)
    print("Wrote " + OUT_RHO060)
    if "measurement_ratio" in by_axis:
        print("Wrote " + OUT_MEAS)


if __name__ == "__main__":
    main()
