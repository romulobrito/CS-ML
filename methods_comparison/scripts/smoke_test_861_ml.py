#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smoke tests for Well 861 ML Etapa 1c pipeline.

Validates data loading, XY build, depth-block CV, RF, and regressor comparison.
Exit code 0 = all checks passed.

ASCII-only.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
LOG_PATH = ROOT / "methods_comparison" / "planning" / "agent_poco861.log"
SMOKE_OUT = ROOT / "methods_comparison" / "data" / "processed" / "ml_runs" / "smoke"


def _log(msg: str) -> None:
    print(msg, flush=True)


def check_imports() -> Tuple[bool, str]:
    try:
        if str(SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPT_DIR))
        from ml_861_data import load_logs_enriched, build_xy, depth_block_splits
        from ml_861_metrics import evaluate_depth_blocks
        from sklearn.ensemble import RandomForestRegressor

        df = load_logs_enriched()
        bundle = build_xy(df, target="FZI_lab")
        folds = depth_block_splits(bundle.df, n_blocks=2)
        if len(folds) != 2:
            return False, "expected 2 depth folds in smoke"
        cv = evaluate_depth_blocks(
            lambda: RandomForestRegressor(n_estimators=5, random_state=42),
            bundle,
            n_blocks=2,
        )
        if cv.mean_rmse <= 0:
            return False, "invalid mean_rmse"
        return True, "imports+data rmse={:.4f} r2={:.4f}".format(cv.mean_rmse, cv.mean_r2)
    except Exception as exc:
        return False, str(exc)


def run_script(script: str, extra: List[str]) -> Tuple[bool, str]:
    py = sys.executable
    cmd = [py, str(SCRIPT_DIR / script), "--smoke", *extra]
    _log("SMOKE CMD: " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0:
        return False, out[-2000:]
    return True, out.strip()[-500:]


def append_agent_log(section: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = "\n{}\nTimestamp: {}\n{}\n".format("-" * 80, ts, section)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(block)


def main() -> int:
    SMOKE_OUT.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[str, bool, str]] = []

    ok, msg = check_imports()
    results.append(("ml_861_data+metrics unit", ok, msg))

    ok, msg = run_script(
        "run_fzi_rf_861.py",
        ["--out-dir", str(SMOKE_OUT / "fzi_rf"), "--no-shap"],
    )
    results.append(("run_fzi_rf_861.py --smoke", ok, msg))

    ok, msg = run_script(
        "run_861_ml_baseline.py",
        [
            "--out-dir",
            str(SMOKE_OUT / "compare_861"),
            "--target",
            "FZI_lab",
            "--regressor",
            "rf",
        ],
    )
    results.append(("run_861_ml_baseline.py --smoke --regressor rf", ok, msg))

    ok, msg = run_script(
        "run_861_ml_baseline.py",
        [
            "--out-dir",
            str(SMOKE_OUT / "compare_861_all"),
            "--target",
            "FZI_lab",
        ],
    )
    results.append(("run_861_ml_baseline.py --smoke all", ok, msg))

    ok, msg = run_script(
        "run_861_well_profile_ct_plugs.py",
        ["--smoke"],
    )
    results.append(("run_861_well_profile_ct_plugs.py --smoke", ok, msg))

    ok, msg = run_script(
        "run_861_well_profile_alternatives.py",
        ["--smoke"],
    )
    results.append(("run_861_well_profile_alternatives.py --smoke", ok, msg))

    metrics_path = SMOKE_OUT / "fzi_rf" / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        results.append(("metrics.json exists", True, json.dumps(metrics["depth_block_cv"], indent=0)[:300]))
    else:
        results.append(("metrics.json exists", False, "missing"))

    lines = ["SMOKE TEST Etapa 1c -- Well 861 ML", ""]
    all_ok = True
    for name, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        lines.append("[{}] {} -- {}".format(status, name, detail[:200]))
        if not passed:
            all_ok = False

    lines.append("")
    lines.append("Overall: {}".format("PASS" if all_ok else "FAIL"))
    section = "\n".join(lines)
    append_agent_log(section)

    _log(section)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
