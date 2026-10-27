# Well 861 global depth-mask grid

Exploratory reconstruction of Phi_lab on one well. The three blocks
are depth intervals of that well, not three independent wells.
No manuscript text was edited. The saved pilot directory was not
rewritten.

- Audit commit: `92cfea4ca22770b302b7ba2eae7b5e29dca707a3`
- Pilot commit: `d966b0c27fd88656a6cd372425e5cfb71095f054`
- HEAD at run: `d966b0c27fd88656a6cd372425e5cfb71095f054`
- Blocks 0, 1, and 2. Seeds 7, 23, and 41. Fractions 0.25 and 0.50.
- Primary score: RMSE on the holdout of that fraction, after the mean
  of every covering window, including an empty-window fallback.
- Secondary score: the same predictions on the 15 depths that remain
  held out at fraction 0.50. Both fractions are scored on that fixed
  set. The set is not used to choose lambda.
- Interpolation uses Depth(m). Ends keep the nearest measured value.
- No artificial measurement noise.
- Lambda is the validation minimum among positive grid values.

## Saved pilot

The pilot tables were reread and the holdout RMSE was recomputed.
The largest absolute gap from the saved RMSE and MAE is 8.674e-17.
That is the CSV round trip, not a new fit. Fixed-complement RMSE
for the saved pilot, fold 0 seed 7:

- Fraction 0.25: clp_csgm_ridge 0.0282; csgm_unconditioned 0.0205; ae_regression_ub 0.0294; prior_only 0.0514; depth_interpolation 0.0183.
- Fraction 0.5: clp_csgm_ridge 0.0280; csgm_unconditioned 0.0216; ae_regression_ub 0.0298; prior_only 0.0514; depth_interpolation 0.0177.

## Where the conditional prior is lower

A negative difference means the CLP depth RMSE is lower than the
comparator. Counts are cells of one well, 3 blocks times 3 seeds
times 2 fractions.

own holdout:
- CLP lower than csgm_unconditioned in 4 of 18 cells.
- CLP lower than ae_regression_ub in 13 of 18 cells.
- CLP lower than prior_only in 17 of 18 cells.
- CLP lower than depth_interpolation in 2 of 18 cells.

fixed complement:
- CLP lower than csgm_unconditioned in 5 of 18 cells.
- CLP lower than ae_regression_ub in 14 of 18 cells.
- CLP lower than prior_only in 17 of 18 cells.
- CLP lower than depth_interpolation in 2 of 18 cells.

Mean primary RMSE across the three seeds:

- Fold 0 fraction 0.25: clp_csgm_ridge 0.0299; csgm_unconditioned 0.0255; ae_regression_ub 0.0338; prior_only 0.0461; depth_interpolation 0.0202.
- Fold 0 fraction 0.5: clp_csgm_ridge 0.0250; csgm_unconditioned 0.0205; ae_regression_ub 0.0309; prior_only 0.0461; depth_interpolation 0.0195.
- Fold 1 fraction 0.25: clp_csgm_ridge 0.0450; csgm_unconditioned 0.0443; ae_regression_ub 0.0507; prior_only 0.0484; depth_interpolation 0.0452.
- Fold 1 fraction 0.5: clp_csgm_ridge 0.0433; csgm_unconditioned 0.0409; ae_regression_ub 0.0500; prior_only 0.0481; depth_interpolation 0.0348.
- Fold 2 fraction 0.25: clp_csgm_ridge 0.0763; csgm_unconditioned 0.0562; ae_regression_ub 0.0679; prior_only 0.0905; depth_interpolation 0.0412.
- Fold 2 fraction 0.5: clp_csgm_ridge 0.0817; csgm_unconditioned 0.0598; ae_regression_ub 0.0713; prior_only 0.0980; depth_interpolation 0.0351.

Mean fixed-complement RMSE across the three seeds:

- Fold 0 fraction 0.25: clp_csgm_ridge 0.0304; csgm_unconditioned 0.0262; ae_regression_ub 0.0330; prior_only 0.0461; depth_interpolation 0.0208.
- Fold 0 fraction 0.5: clp_csgm_ridge 0.0250; csgm_unconditioned 0.0205; ae_regression_ub 0.0309; prior_only 0.0461; depth_interpolation 0.0195.
- Fold 1 fraction 0.25: clp_csgm_ridge 0.0440; csgm_unconditioned 0.0436; ae_regression_ub 0.0500; prior_only 0.0481; depth_interpolation 0.0448.
- Fold 1 fraction 0.5: clp_csgm_ridge 0.0433; csgm_unconditioned 0.0409; ae_regression_ub 0.0500; prior_only 0.0481; depth_interpolation 0.0348.
- Fold 2 fraction 0.25: clp_csgm_ridge 0.0820; csgm_unconditioned 0.0584; ae_regression_ub 0.0720; prior_only 0.0980; depth_interpolation 0.0399.
- Fold 2 fraction 0.5: clp_csgm_ridge 0.0817; csgm_unconditioned 0.0598; ae_regression_ub 0.0713; prior_only 0.0980; depth_interpolation 0.0351.

## Empty windows

An empty window has no observed depth. The CLP fallback decodes
the conditional prior. The unconditioned fallback decodes the zero
code. The AE fallback uses a zero embedding. The primary mean
keeps those predictions. The comparison below drops empty windows
when a depth still has another window, and keeps the fallback
only when every covering window is empty.

Empty test windows across the grid: 7 of 396.
- Fold 1 seed 23 fraction 0.25: 4 of 22 windows empty.
- Fold 2 seed 7 fraction 0.25: 3 of 22 windows empty.

Primary RMSE minus the RMSE that drops empty windows when possible:

- clp_csgm_ridge: mean delta -0.00025, max abs delta 0.00401, fallback-only depths 7.
- csgm_unconditioned: mean delta -0.00046, max abs delta 0.00592, fallback-only depths 7.
- ae_regression_ub: mean delta 0.00010, max abs delta 0.00191, fallback-only depths 7.
- prior_only: mean delta 0.00000, max abs delta 0.00000, fallback-only depths 0.

## Limits

The blocks share one well and one core table. A win in one block
is not a win on a new well. The 87 rows are a log-grid derivative,
not an assay ledger. Lambda and the AE fit do not see the test
block. The fixed complement is reported after those choices.

Conditioning helps in a cell when the CLP RMSE is lower than the
unconditioned CSGM. It does not, by itself, make the CLP better
than depth interpolation or the direct AE.

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
- [PASS] fixed_complement_is_nested: Both fractions were scored on the 15 depths held out at fraction 0.50
- [PASS] fallback_mean_matches_depth_table: The mean of the window predictions matches the depth table
- [PASS] saved_pilot_scores_reproduced: Saved pilot RMSE gap 8.673617379884035e-17

