#!/bin/sh
# Pilot: joint-only explore profile, identity then dct (2 jobs each).
# Artifacts under outputs/direct_ub_psi_ablation/{identity,dct}/runs/<run_id>/
set -e
cd "$(dirname "$0")/.." || exit 1
BASE="outputs/direct_ub_psi_ablation"
RUN_ID="pilot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BASE/identity/runs" "$BASE/dct/runs"
echo "RUN_ID=$RUN_ID"
python sir_cs_benchmark_direct_ub.py --joint-only --explore --no-plots \
  --residual-basis identity \
  --base-dir "$BASE/identity" \
  --run-id "$RUN_ID"
python sir_cs_benchmark_direct_ub.py --joint-only --explore --no-plots \
  --residual-basis dct \
  --base-dir "$BASE/dct" \
  --run-id "$RUN_ID"
echo "Done. Merge with:"
echo "python scripts/merge_direct_ub_psi_ablation.py \\"
echo "  --identity-summary $BASE/identity/runs/$RUN_ID/tables/summary.csv \\"
echo "  --dct-summary $BASE/dct/runs/$RUN_ID/tables/summary.csv \\"
echo "  --out $BASE/merged_${RUN_ID}_summary_long.csv"
