# Figure And Table Plan

This file defines which artifacts belong in the main paper, which should move
to appendix, and what argument each artifact supports.

## Main Tables

### Table: Dataset summary

- File: `tables/eda_dataset_summary.tex`
- Section: Data and exploratory analysis.
- Argument: the paper uses two related but distinct tasks; the held-out F03-4 Vc
  distribution differs from the training wells, and the F03 porosity task has a
  narrower target range.
- Status: main paper.

### Table: Channel-target correlations

- File: `tables/eda_channel_target_correlations.tex`
- Section: Data and exploratory analysis.
- Argument: GR is highly informative for Vc but weak for F03 porosity; the
  GR-only experiment is intentionally challenging.
- Status: main paper.

### Table: Cross-well CLP-CSGM vs AE

- File: `tables/results_crosswell_clp_vs_ae.tex`
- Section: Results.
- Argument: CLP-CSGM Ridge beats the strongest direct AE baseline in all nine
  low-data cells.
- Status: main paper.

### Table: Cross-well overall rank

- File: `tables/results_crosswell_overall_rank.tex`
- Section: Results.
- Argument: CLP-CSGM Ridge has the best average RMSE and relative L2 across the
  low-data grid.
- Status: main paper.

### Table: F03 GR-only summary

- File: `tables/results_f03_gr_only.tex`
- Section: Results.
- Argument: CLP-CSGM Ridge is best on average, strongest at low measurement
  ratios, and slightly loses to AE at higher measurement ratios.
- Status: main paper, but consider compacting if page count grows.

## Main Figures

### EDA target summary

- File: `figures/eda_target_summary.png`
- Argument: shows target scale and shift across datasets.
- Status: main paper.

### Cross-well Vc distribution

- File: `figures/eda_crosswell_vc_distribution.png`
- Argument: held-out F03-4 has lower Vc distribution than training wells.
- Status: main paper or appendix if space is tight.

### Channel-target correlations

- File: `figures/eda_channel_target_correlations.png`
- Argument: GR is strong for Vc but weak for F03 porosity.
- Status: main paper because it directly justifies the GR-only benchmark.

### F03 GR and porosity depth profile

- File: `figures/eda_f03_gr_porosity_depth.png`
- Argument: qualitative view of real-well target and input structure.
- Status: main paper or appendix depending on page budget.

### Cross-well CLP-CSGM vs AE RMSE

- File: `figures/results_crosswell_clp_vs_ae.png`
- Argument: direct visual evidence of main quantitative claim.
- Status: main paper.

### Cross-well all-method RMSE

- File: `figures/results_crosswell_rmse_vs_rho.png`
- Argument: shows the result is not only a pairwise win against AE but also a
  strong comparison against all direct baselines.
- Status: main paper.

### F03 RMSE, MAE, Relative L2

- Files:
  - `01_rmse_vs_measurement_ratio.png`
  - `02_mae_vs_measurement_ratio.png`
  - `03_relative_l2_vs_measurement_ratio.png`
- Argument: real-well result is consistent across metrics.
- Status: RMSE in main paper; MAE and relative L2 can be appendix if page budget
  becomes constrained.

### F03 grouped RMSE bars

- File: `05_rmse_grouped_bars_by_ratio.png`
- Argument: gives a compact grouped view of the same RMSE comparison.
- Status: appendix candidate because it partially repeats the RMSE line plot.

### F03 gain over ML-only

- Files:
  - `06_gain_rmse_over_ml_only.png`
  - `07_gain_mae_over_ml_only.png`
- Argument: shows that CLP-CSGM and AE are the useful sparse-measurement models,
  while MLP `[u,b]` can be worse than ML only.
- Status: main paper if discussing failure of direct concatenation; otherwise
  appendix.

### F03 qualitative diagnostics

- Files:
  - `08_example_ground_truth_vs_models.png`
  - `09_parity_ground_truth_vs_prediction.png`
  - `10_depth_profile_porosity.png`
- Argument: profile-level evidence that CLP-CSGM is smoother than direct MLP and
  tracks observed porosity trends.
- Status: at least parity and depth profile should remain in main paper; example
  windows can move to appendix if page budget is tight.

## Recommended Main-Paper Figure Set

If the paper becomes too long, keep:

1. `eda_channel_target_correlations.png`
2. `results_crosswell_clp_vs_ae.png`
3. `results_crosswell_rmse_vs_rho.png`
4. `01_rmse_vs_measurement_ratio.png`
5. `06_gain_rmse_over_ml_only.png`
6. `09_parity_ground_truth_vs_prediction.png`
7. `10_depth_profile_porosity.png`

Move all other figures to appendix.

## Current Risk

The manuscript currently includes many F03 figures. This is useful for internal
analysis but may be too much for a journal submission. During final editing,
choose a smaller main set and move the rest to appendix.
