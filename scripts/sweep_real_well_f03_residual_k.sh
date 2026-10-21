#!/usr/bin/env bash
# Sensitivity: residual_k only. One full rho grid per k; separate run folders.
# See docs/real_well_f03_direct_ub.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

OUT_BASE="${ROOT}/outputs/real_well_f03/sweeps/residual_k"
DATA="${ROOT}/data/F03-4_AC+GR+Porosity.txt"
KS=(4 6 8 12)

for k in "${KS[@]}"; do
  rid="f03_k${k}_$(date +%Y%m%d_%H%M%S)"
  sub="${OUT_BASE}/k_${k}"
  echo "=== residual_k=${k} run_id=${rid} ==="
  python sir_cs_benchmark_real_well_direct_ub.py \
    --data-path "${DATA}" \
    --base-dir "${sub}" \
    --run-id "${rid}" \
    --residual-k "${k}"
done

echo "Sweep done under ${OUT_BASE}/"
