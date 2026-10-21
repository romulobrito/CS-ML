#!/usr/bin/env bash
# Full real-well F03 benchmark: joint-only baselines + hybrid_lfista_joint.
# See docs/real_well_f03_direct_ub.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

RUN_ID="f03_$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/outputs/real_well_f03/direct_ub"

python sir_cs_benchmark_real_well_direct_ub.py \
  --data-path "${ROOT}/data/F03-4_AC+GR+Porosity.txt" \
  --base-dir "${OUT}" \
  --run-id "${RUN_ID}"

echo "Artifacts: ${OUT}/runs/${RUN_ID}"
