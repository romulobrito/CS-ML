#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrator: CLP-CSGM Phi_lab profile reconstruction for Well 861 MOGNO.

Modes:
  smoke  -- delegate to scripts/auddys_smoke_direct_ub.py (MOGNO depth filter)
  prod   -- plug_sparse_b + depth-block CV (Phase 2; scaffold until implemented)

Planning: methods_comparison/planning/etapa1f_clp_csgm_phi_lab_poco861.md
ASCII-only.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from clp_861_protocol import (
    Clp861RunPaths,
    compare_rf_baseline_dir,
    export_plug_indices_csv,
    load_plug_measurement_rows,
    mogno_depth_bounds,
    scenario_choices,
    u_channels_csv,
)
from ml_861_data import (
    CLP_861_SCENARIO_PLUG_SPARSE,
    CLP_861_SCENARIO_RHO_SUBSAMPLE,
)

AUDDYS_RUNNER = REPO_ROOT / "scripts" / "auddys_smoke_direct_ub.py"
DEFAULT_EXCEL = REPO_ROOT / "data" / "Auddys_table.xlsx"


def _default_run_id(prefix: str) -> str:
    return "{}_{}".format(prefix, time.strftime("%Y%m%d_%H%M%S"))


def _build_audys_command(
    args: argparse.Namespace,
    run_paths: Clp861RunPaths,
    benchmark_scenario: str,
) -> List[str]:
    """Command line for auddys_smoke_direct_ub on MOGNO interval."""
    depth_min, depth_max = mogno_depth_bounds()
    cmd = [
        sys.executable,
        str(AUDDYS_RUNNER),
        "--excel-path",
        str(Path(args.excel_path)),
        "--sheet",
        "Logs",
        "--u-channels",
        u_channels_csv(),
        "--target",
        "phi_lab",
        "--depth-min-m",
        str(depth_min),
        "--depth-max-m",
        str(depth_max),
        "--window-len",
        str(int(args.window_len)),
        "--step",
        str(int(args.step)),
        "--base-dir",
        str(run_paths.run_root.parent.parent),
        "--run-id",
        run_paths.run_root.name,
        "--benchmark-scenario",
        benchmark_scenario,
        "--well-label",
        "861 MOGNO",
        "--seeds",
        str(args.seeds),
        "--rhos",
        str(args.rhos),
        "--run-csgm-m2",
        "--csgm-prior-type",
        str(args.csgm_prior_type),
        "--measurement-kind",
        "subsample",
        "--csgm-ae-epochs",
        str(int(args.csgm_ae_epochs)),
    ]
    if args.csgm_prior_types.strip():
        cmd.extend(["--csgm-prior-types", str(args.csgm_prior_types)])
    if args.smoke_no_ae:
        cmd.append("--no-ae")
    return cmd


def _run_audys_scenario(
    args: argparse.Namespace,
    scenario: str,
    run_id: str,
    benchmark_scenario: str,
    protocol_lines: List[str],
) -> int:
    """Shared auddys runner with plug index export and PROTOCOL.txt."""
    run_paths = Clp861RunPaths.from_scenario_run(scenario, run_id)
    run_paths.ensure_dirs()

    plugs = load_plug_measurement_rows()
    export_plug_indices_csv(
        plugs,
        run_paths.tables / "plug_measurement_indices.csv",
    )

    cmd = _build_audys_command(args, run_paths, benchmark_scenario)
    proto = run_paths.run_root / "PROTOCOL.txt"
    lines = list(protocol_lines)
    lines.extend(
        [
            "scenario: {}".format(scenario),
            "runner: auddys_smoke_direct_ub.py",
            "command: {}".format(" ".join(cmd)),
            "plug rows exported: {}".format(
                run_paths.tables / "plug_measurement_indices.csv"
            ),
            "",
        ]
    )
    proto.write_text("\n".join(lines), encoding="utf-8")

    print("CLP_861_RUN")
    print("SCENARIO", scenario)
    print("RUN_ROOT", run_paths.run_root)
    print("COMMAND", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), check=False)
    return int(proc.returncode)


def run_smoke(args: argparse.Namespace) -> int:
    """Quick CLP smoke: 1 seed, 1 rho, ridge, optional --smoke-no-ae."""
    scenario = CLP_861_SCENARIO_RHO_SUBSAMPLE
    run_id = args.run_id.strip() or _default_run_id("smoke")
    return _run_audys_scenario(
        args,
        scenario=scenario,
        run_id=run_id,
        benchmark_scenario="861_mogno_phi_lab_clp_smoke",
        protocol_lines=[
            "CLP-861 smoke protocol",
            "mode: smoke",
            "note: quick run; use --mode prod for full grid",
        ],
    )


def run_prod(args: argparse.Namespace) -> int:
    """
    Production CLP: plug-fixed b (default for plug_sparse_b) or auddys rho grid.
    """
    if str(getattr(args, "measurement", "subsample")) == "plug_fixed":
        return run_plug_fixed(args)

    scenario = str(args.scenario)
    run_id = args.run_id.strip() or _default_run_id("prod")
    code = _run_audys_scenario(
        args,
        scenario=scenario,
        run_id=run_id,
        benchmark_scenario="861_mogno_phi_lab_clp_prod_{}".format(scenario),
        protocol_lines=[
            "CLP-861 production protocol",
            "mode: prod",
            "note: full auddys grid; depth-block plug-fixed b pending Phase 2",
        ],
    )

    if code == 0 and args.compare_rf:
        _write_compare_rf_stub(scenario, run_id)

    return code


def run_plug_fixed(args: argparse.Namespace) -> int:
    """Run CLP with b fixed at 10 plug depths (depth-block CV)."""
    from clp_861_plug_fixed_runner import (
        execute_plug_fixed_run,
        save_plug_fixed_figures_from_tables,
    )

    scenario = CLP_861_SCENARIO_PLUG_SPARSE
    run_id = args.run_id.strip() or _default_run_id("plug_fixed")
    run_paths = Clp861RunPaths.from_scenario_run(scenario, run_id)
    run_paths.ensure_dirs()

    if bool(getattr(args, "figures_only", False)):
        paths = save_plug_fixed_figures_from_tables(run_paths, primary_seed=7)
        print("FIGURES_ONLY", [str(p) for p in paths])
        return 0

    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]
    priors_raw = str(args.csgm_prior_types).strip()
    if priors_raw:
        priors = [p.strip() for p in priors_raw.split(",") if p.strip()]
    else:
        priors = [str(args.csgm_prior_type)]

    export_plug_indices_csv(
        load_plug_measurement_rows(),
        run_paths.tables / "plug_measurement_indices.csv",
    )

    print("CLP_861_PLUG_FIXED")
    print("RUN_ROOT", run_paths.run_root)
    primary_seed = int(seeds[0]) if seeds else 7
    device_arg = str(getattr(args, "device", "")).strip() or None
    summary_path = execute_plug_fixed_run(
        run_paths=run_paths,
        excel_path=Path(args.excel_path),
        seeds=seeds,
        prior_types=priors,
        csgm_ae_epochs=int(args.csgm_ae_epochs),
        compare_rf=bool(args.compare_rf),
        primary_seed=primary_seed,
        device=device_arg,
    )
    print("SUMMARY", summary_path)
    return 0


def _write_compare_rf_stub(scenario: str, run_id: str) -> None:
    """Legacy stub when --compare-rf omitted (plug-fixed run writes real tables)."""
    cmp_dir = compare_rf_baseline_dir()
    cmp_dir.mkdir(parents=True, exist_ok=True)
    note = cmp_dir / "notes.md"
    note.write_text(
        "\n".join(
            [
                "# CLP vs RF comparison",
                "",
                "Prod run completed: scenario `{}`, run_id `{}`.".format(
                    scenario, run_id
                ),
                "RF baseline: `well_profile/phi_lab/rf/metrics.json`.",
                "CLP summary: see run tables/summary_focus_clp_csgm_vs_ub.csv.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print("COMPARE_DIR", cmp_dir)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """CLI."""
    parser = argparse.ArgumentParser(
        description="Well 861 CLP-CSGM Phi_lab orchestrator (Etapa 1f)."
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "prod"),
        default="smoke",
        help="smoke=auddys runner; prod=plug_sparse_b depth-block (scaffold)",
    )
    parser.add_argument(
        "--scenario",
        choices=scenario_choices(),
        default=CLP_861_SCENARIO_PLUG_SPARSE,
        help="Prod scenario folder under clp_861/phi_lab/",
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument(
        "--excel-path",
        type=Path,
        default=DEFAULT_EXCEL,
        help="Legacy Logs Excel for smoke runner",
    )
    parser.add_argument("--window-len", type=int, default=16)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--rhos", type=str, default="")
    parser.add_argument(
        "--csgm-prior-type",
        type=str,
        default="ridge",
        choices=("ridge", "mlp", "rf", "rf_residual"),
    )
    parser.add_argument(
        "--csgm-prior-types",
        type=str,
        default="",
        help="Optional comma list ridge,mlp,rf,rf_residual (prod default: ridge,mlp)",
    )
    parser.add_argument("--csgm-ae-epochs", type=int, default=200)
    parser.add_argument(
        "--smoke-no-ae",
        action="store_true",
        help="Skip AE baseline in smoke for speed",
    )
    parser.add_argument(
        "--measurement",
        choices=("subsample", "plug_fixed"),
        default="subsample",
        help="subsample=rho grid (auddys); plug_fixed=b at 10 plug depths",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only regenerate figures (RMSE from tables; full depth profile needs re-run)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        choices=("", "cpu", "cuda"),
        help="Torch device for AE/CSGM (default: auto cuda if available)",
    )
    parser.add_argument(
        "--compare-rf",
        action="store_true",
        help="Write CLP vs RF comparison under compare_rf_baseline/",
    )
    return parser.parse_args(argv)


def _apply_mode_defaults(args: argparse.Namespace) -> None:
    """Fill empty seeds/rhos/priors from mode-specific defaults."""
    if args.mode == "smoke":
        if not str(args.seeds).strip():
            args.seeds = "7"
        if not str(args.rhos).strip():
            args.rhos = "0.3"
        if not str(args.csgm_prior_types).strip():
            args.csgm_prior_types = ""
        args.smoke_no_ae = True
        if int(args.csgm_ae_epochs) == 200:
            args.csgm_ae_epochs = 80
    else:
        if not str(args.seeds).strip():
            args.seeds = "7,23,41"
        if not str(args.rhos).strip():
            args.rhos = "0.2,0.3,0.4,0.5,0.6"
        if not str(args.csgm_prior_types).strip():
            args.csgm_prior_types = "ridge,mlp"
        if str(getattr(args, "measurement", "subsample")) == "plug_fixed":
            args.scenario = CLP_861_SCENARIO_PLUG_SPARSE


def main() -> None:
    """Entry point."""
    args = parse_args()
    _apply_mode_defaults(args)
    needs_audys = not (
        args.mode == "prod"
        and str(getattr(args, "measurement", "subsample")) == "plug_fixed"
    )
    if needs_audys and not AUDDYS_RUNNER.is_file():
        raise FileNotFoundError(str(AUDDYS_RUNNER))

    if args.mode == "smoke":
        code = run_smoke(args)
    elif (
        args.mode == "prod"
        and str(getattr(args, "measurement", "subsample")) == "plug_fixed"
        and bool(getattr(args, "figures_only", False))
    ):
        code = run_plug_fixed(args)
    else:
        code = run_prod(args)

    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
