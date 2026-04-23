#!/usr/bin/env bash
# Sequential robustness sweep: DCT + subsample, joint-only, three seeds (robustness-lite).
# Varies residual_k only; keeps default measurement_noise_std (0.02) unless overridden in CLI.
# Each arm writes all benchmark figures (01--07), parity PNG + parity_pooled.npz, tables, PROTOCOL.
# See docs/direct_ub_robustness_residual_k.txt
#
# Optional: LOG_FILE=path bash ...  appends stdout/stderr (default: no extra log file).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_OUT="outputs/direct_ub_robustness/dct_subsample/residual_k"
FIG_PAPER="${ROOT}/paper/figures/direct_ub_robustness_residual_k"
mkdir -p "${FIG_PAPER}"

if [[ -n "${LOG_FILE:-}" ]]; then
  exec >>"${LOG_FILE}" 2>&1
  echo "Logging to ${LOG_FILE}"
  date -Iseconds
fi

KS=(2 4 6 8 12 16)
declare -a MERGE_ARGS=()

for k in "${KS[@]}"; do
  rid="residual_k_${k}_$(date +%Y%m%d_%H%M%S)"
  subdir="${BASE_OUT}/v_${k}"
  echo "=== residual_k=${k} run_id=${rid} ==="
  echo "Per-arm console (Python tee): ${subdir}/runs/${rid}/logs/run_console.log"
  python sir_cs_benchmark_direct_ub.py \
    --joint-only \
    --robustness-lite \
    --residual-basis dct \
    --measurement-kind subsample \
    --residual-k "${k}" \
    --base-dir "${subdir}" \
    --run-id "${rid}"
  summary="${subdir}/runs/${rid}/tables/summary.csv"
  if [[ ! -f "${summary}" ]]; then
    echo "Missing summary: ${summary}" >&2
    exit 2
  fi
  MERGE_ARGS+=(--arm "${k}" "${summary}")
  parity_src="${subdir}/runs/${rid}/figures/09_parity_ground_truth_vs_prediction.png"
  if [[ -f "${parity_src}" ]]; then
    cp -f "${parity_src}" "${FIG_PAPER}/09_parity_k${k}.png"
    echo "Copied parity -> ${FIG_PAPER}/09_parity_k${k}.png"
  else
    echo "Warning: missing parity figure: ${parity_src}" >&2
  fi
  npz_src="${subdir}/runs/${rid}/tables/parity_pooled.npz"
  if [[ -f "${npz_src}" ]]; then
    cp -f "${npz_src}" "${FIG_PAPER}/parity_pooled_k${k}.npz"
    echo "Copied parity arrays -> ${FIG_PAPER}/parity_pooled_k${k}.npz"
  fi
done

merged="${ROOT}/outputs/direct_ub_robustness/dct_subsample/merged_residual_k_long.csv"
python scripts/merge_direct_ub_residual_k_robustness.py --out "${merged}" "${MERGE_ARGS[@]}"

python scripts/plot_direct_ub_residual_k_robustness.py \
  --merged "${merged}" \
  --out "${FIG_PAPER}/01_rmse_joint_vs_rho_by_k.png" \
  --at-rho 0.6 \
  --out-at-rho "${FIG_PAPER}/02_rmse_vs_k_rho0p6.png"

echo "Merged: ${merged}"
echo "Paper figures under ${FIG_PAPER}/"
date -Iseconds
