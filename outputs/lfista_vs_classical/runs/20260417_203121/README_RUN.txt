SIR-CS lfista_vs_classical run (hybrid_fista + LFISTA + SPGL1 baselines, same protocol).
run_id: 20260417_203121
started_local: 2026-04-17T20:31:21
cwd: /home/romulo/Documentos/cs-regressor
argv: ["sir_cs_pipeline_optimized.py", "--profile", "lfista_vs_classical"]

Artifacts (this folder):
  detailed_results.csv, summary_by_seed.csv, summary.csv
  summary_focus_*.csv (subset for ml_only, ml_only_torch, hybrid_fista, LFISTA)
  FOCUS_COMPARISON.txt
  config.json, PROTOCOL.txt, run_console.log, README_RUN.txt

Figures (repo): paper/figures/lfista_vs_classical/runs/20260417_203121/

Symlink: outputs/lfista_vs_classical/LATEST -> runs/20260417_203121


finished_local: 2026-04-17T22:45:19
elapsed_seconds: 8037.8

CSV and logs in this directory; figures under paper/figures/lfista_vs_classical/runs/20260417_203121/

Plot files:
  01_rmse_vs_measurement_ratio.png
  02_mae_vs_measurement_ratio.png
  03_relative_l2_vs_measurement_ratio.png
  04_support_f1_vs_measurement_ratio.png
  05_rmse_grouped_bars_by_ratio.png
  06_gain_rmse_over_ml_only.png
  07_gain_mae_over_ml_only.png
  08_example_ground_truth_vs_models.png
  09_parity_ground_truth_vs_prediction.png
  10_residual_distributions_gt_vs_models.png
  11_gain_rmse_over_ml_only_torch.png
  12_gain_mae_over_ml_only_torch.png
