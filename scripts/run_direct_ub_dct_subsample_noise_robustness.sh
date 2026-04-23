#!/usr/bin/env bash
# Sequential robustness sweep: DCT + subsample, joint-only, three seeds (robustness-lite).
# See docs/direct_ub_robustness_measurement_noise.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

BASE_OUT="outputs/direct_ub_robustness/dct_subsample/measurement_noise_std"
NOISES=(0.01 0.02 0.04)
declare -a MERGE_ARGS=()

for n in "${NOISES[@]}"; do
  slug="${n//./p}"
  rid="noise_${slug}_$(date +%Y%m%d_%H%M%S)"
  subdir="${BASE_OUT}/v_${slug}"
  echo "=== noise_std=${n} run_id=${rid} ==="
  python sir_cs_benchmark_direct_ub.py \
    --joint-only \
    --robustness-lite \
    --residual-basis dct \
    --measurement-kind subsample \
    --measurement-noise-std "${n}" \
    --base-dir "${subdir}" \
    --run-id "${rid}"
  summary="${subdir}/runs/${rid}/tables/summary.csv"
  if [[ ! -f "${summary}" ]]; then
    echo "Missing summary: ${summary}" >&2
    exit 2
  fi
  MERGE_ARGS+=(--arm "${n}" "${summary}")
done

merged="${ROOT}/outputs/direct_ub_robustness/dct_subsample/merged_measurement_noise_long.csv"
python scripts/merge_direct_ub_noise_robustness.py --out "${merged}" "${MERGE_ARGS[@]}"

fig_dir="${ROOT}/paper/figures/direct_ub_robustness_measurement_noise"
mkdir -p "${fig_dir}"
python scripts/plot_direct_ub_noise_robustness.py \
  --merged "${merged}" \
  --out "${fig_dir}/01_rmse_joint_vs_rho_by_noise.png"

echo "Merged: ${merged}"
echo "Figure: ${fig_dir}/01_rmse_joint_vs_rho_by_noise.png"
