#!/bin/sh
# Pilot: joint-only explore, DCT Psi + subsample M (2 jobs).
# Artifacts: outputs/direct_ub_dct_m_ablation/subsample/runs/<run_id>/
set -e
cd "$(dirname "$0")/.." || exit 1
BASE="outputs/direct_ub_dct_m_ablation/subsample"
RUN_ID="pilot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE/runs"
echo "RUN_ID=$RUN_ID"
python -u sir_cs_benchmark_direct_ub.py --joint-only --explore --no-plots \
  --residual-basis dct \
  --measurement-kind subsample \
  --base-dir "$BASE" \
  --run-id "$RUN_ID"
echo "Done. Compare summary to frozen Gaussian+DCT under outputs/direct_ub_psi_ablation/dct/.../tables/summary.csv"
echo "Merge example:"
echo "python scripts/merge_direct_ub_m_ablation.py \\"
echo "  --gaussian-dct-summary outputs/direct_ub_psi_ablation/dct/runs/20260422_175210/tables/summary.csv \\"
echo "  --subsample-dct-summary $BASE/runs/$RUN_ID/tables/summary.csv \\"
echo "  --out outputs/direct_ub_dct_m_ablation/merged_pilot_${RUN_ID}.csv"
