# Well 861 global depth-mask pilot

Exploratory reconstruction of Phi_lab. No manuscript text was edited.
The published window-level results were not overwritten.

- Reference audit commit: `92cfea4ca22770b302b7ba2eae7b5e29dca707a3`
- HEAD at run: `92cfea4ca22770b302b7ba2eae7b5e29dca707a3`
- Scope: fold 0, seed 7, fractions 0.25 and 0.50.
- The three depth blocks belong to the same well. This pilot uses
  only the shallowest block and is not a geological validation.
- Aggregation: mean of every window that covers the depth, including
  the documented empty-window fallback.
- Interpolation axis: Depth(m). Ends keep the nearest measured value.
- No artificial measurement noise.
- Lambda is the validation depth RMSE minimum. Ties keep the smaller
  earlier grid value. Test scores are not an input.
- An empty CLP window decodes the conditional prior. That is the
  minimizer of a positive penalty with no data term.
- An empty unconditioned window decodes the zero latent code. A
  restart would be arbitrary because every latent has zero loss.
- An empty AE window uses a zero value-and-mask embedding.
- Per-window recovery selects the restart on that window's own loss.
  The published batch call selected one restart for the whole batch.
  Separate calls are required because each window has its own operator.

## Depth RMSE

The scored set is the holdout only. At fraction 0.25 that is 22
depths. At fraction 0.50 it is 15 depths. All 29 depths stay covered.

Fraction 0.25:
- clp_csgm_ridge: RMSE 0.0274 , MAE 0.0243 , lambda 0.0001
- csgm_unconditioned: RMSE 0.0210 , MAE 0.0178 , lambda none
- ae_regression_ub: RMSE 0.0297 , MAE 0.0260 , lambda none
- prior_only: RMSE 0.0493 , MAE 0.0440 , lambda none
- depth_interpolation: RMSE 0.0187 , MAE 0.0160 , lambda none

Fraction 0.5:
- clp_csgm_ridge: RMSE 0.0280 , MAE 0.0247 , lambda 0.0003
- csgm_unconditioned: RMSE 0.0216 , MAE 0.0184 , lambda none
- ae_regression_ub: RMSE 0.0298 , MAE 0.0261 , lambda none
- prior_only: RMSE 0.0514 , MAE 0.0468 , lambda none
- depth_interpolation: RMSE 0.0177 , MAE 0.0153 , lambda none

Negative `clp_minus_comparator` means the CLP depth RMSE is lower.

- rho 0.25 vs csgm_unconditioned: 0.00648
- rho 0.25 vs ae_regression_ub: -0.00223
- rho 0.25 vs prior_only: -0.02186
- rho 0.25 vs depth_interpolation: 0.00874
- rho 0.5 vs csgm_unconditioned: 0.00646
- rho 0.5 vs ae_regression_ub: -0.00174
- rho 0.5 vs prior_only: -0.02337
- rho 0.5 vs depth_interpolation: 0.01029

## Empty windows

- rho 0.25: 0 of 22 test windows have no observed depth.
- rho 0.5: 0 of 22 test windows have no observed depth.

Train, validation, and test use separate nested masks. The direct
AE is trained at the same fraction on training rows only. Its
input is scaled u, the placed laboratory values, and a binary mask.
Held-out training depths are targets of that fit. Test depths are not.

The 87-row table remains a log-grid derivative, not an assay ledger.

## Checks

- [PASS] split_matches_audit_and_is_disjoint: Row splits match the audit and do not overlap
- [PASS] nested_physical_masks: The 25% observed depths are inside the 50% set
- [PASS] exclusive_depth_role: Each test depth is observed or held out, not both
- [PASS] holdout_depths_covered: Every holdout depth sits in at least one test window
- [PASS] holdout_absent_from_b: No holdout row index is placed in a test measurement
- [PASS] scalers_fit_on_train_only: Feature and AE input scalers see only training windows
- [PASS] same_test_observations: CLP, AE, unconditioned CSGM, and interpolation share the mask
- [PASS] lambda_from_validation_only: Selected lambda is the positive validation minimum
- [PASS] empty_window_fallback: Empty CLP windows match the prior and empty free windows match z=0
- [PASS] finite_depth_scores: Depth RMSE and AE predictions are finite

The full three-block grid was not run.

