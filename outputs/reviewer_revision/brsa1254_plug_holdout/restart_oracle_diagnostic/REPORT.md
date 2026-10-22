# CLP restart and decoder-capacity audit, well 9-BRSA-1254-RJS

Base commit: f6b6159e910c2529a9f750057fcab676b6bcf769.
Remote main matched that commit at the start of this audit. A later fetch is not implied.
Lambda is the validation choice stored in lambda_extension/tables/selection_by_cell.csv.
Collective recovery calls csgm_recover_with_prior. Individual selection is local to this audit.
The repository groups windows that share an observation pattern before recovery.
Collective selection is therefore the best restart inside each group, not one restart for the whole block.
Individual selection chooses inside that same group, one window at a time, with the same objective.
Empty observation windows stay at G(z0) in both rules.
The oracle fits latent codes to the true test windows. It is not a method and it does not choose lambda.
Oracle search: 12 starts, 600 Adam steps, step size 0.03. The first start encodes the true window.
Reproduction tolerance: 0.0001 RMSE against the stored extended-rule test score.
Reproduction status: passed.
Maximum absolute reproduction gap: 2.22045e-16.

## Primary mean of the nine fold-seed RMSE values

| rho | prior | collective | individual | oracle |
|---:|---:|---:|---:|---:|
| 0.25 | 1.432 | 1.448 | 1.447 | 0.715 |
| 0.50 | 1.469 | 1.430 | 1.430 | 0.702 |

## By block, mean over the three seeds

| block | rho | prior | collective | individual | oracle |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.25 | 1.225 | 1.113 | 1.112 | 0.408 |
| 0 | 0.50 | 1.348 | 1.101 | 1.101 | 0.472 |
| 1 | 0.25 | 1.692 | 1.802 | 1.801 | 0.930 |
| 1 | 0.50 | 1.741 | 1.811 | 1.811 | 0.855 |
| 2 | 0.25 | 1.378 | 1.429 | 1.429 | 0.809 |
| 2 | 0.50 | 1.319 | 1.378 | 1.378 | 0.778 |

## Restart selection, recovery cells only

Cells whose validation choice was no_update do not run recovery. Count: 2.
Recovery cells: 16.
Individual objective sum is lower or equal in every recovery call by the script check.
Mean objective sum, individual minus collective: -0.00224091.
Mean observation-loss sum, individual minus collective: -0.0019237.
Test RMSE lower for the individual rule: 8 of 16.
Test RMSE higher for the individual rule: 3 of 16.
Mean test RMSE, individual minus collective: -0.000343664.

## Oracle

Mean fraction of windows whose best start was the encoded truth: 0.251.
Mean fraction of windows updated by a start after the first: 0.749.
This is a multistart approximation, not a certified minimum.

## Previously scored comparators, not refit

Source: corrected_protocol/tables/summary_primary.csv.

| rho | method | mean RMSE |
|---:|---|---:|
| 0.25 | ae_ub | 1.622 |
| 0.25 | clp | 1.524 |
| 0.25 | direct_regression | 1.622 |
| 0.25 | interp | 1.804 |
| 0.25 | mlp_ub | 1.577 |
| 0.25 | prior_only | 1.432 |
| 0.25 | smooth_constant | 1.657 |
| 0.50 | ae_ub | 1.507 |
| 0.50 | clp | 1.546 |
| 0.50 | direct_regression | 1.569 |
| 0.50 | interp | 1.543 |
| 0.50 | mlp_ub | 1.526 |
| 0.50 | prior_only | 1.469 |
| 0.50 | smooth_constant | 1.425 |

## Answers from this run

1. The individual rule lowers the optimized objective. On the 16 cells that run recovery, its objective sum is lower or equal in every cell. The mean difference is -0.00224. The observation term alone is lower on average, by -0.00192, but it is not lower in every cell, because the objective also penalizes distance from z0.

2. That reduction does not produce a material test gain. The mean RMSE difference is -0.00034. The largest absolute cell change is 0.0036. At 25 percent the means are 1.448 collective and 1.447 individual. At 50 percent both are 1.430. Among the 16 recovery cells, 8 are lower, 3 are higher, and 5 are unchanged at 1e-7. Two further cells never run recovery because validation chose no_update.

3. The measurement update beats the prior on the seed-mean of block 0: 1.113 against 1.225 at 25 percent, and 1.101 against 1.348 at 50 percent. In block 1 the prior mean is lower, 1.692 against 1.802 and 1.741 against 1.811. In block 2 the prior mean is also lower, 1.378 against 1.429 and 1.319 against 1.378. The individual rule does not change this block pattern.

4. The approximate oracle reaches 0.715 at 25 percent and 0.702 at 50 percent, after aggregating windows. Block 0 is about 0.41 to 0.47. Blocks 1 and 2 remain about 0.78 to 0.93. The trained decoder can fit the true test windows better than sparse recovery does, and a large error remains. Every decoder still assigned at least one window to the last of the 12 starts, so this is not a certified minimum. A longer search could move the oracle down. It would not raise it above the CLP scores already obtained.

5. The collective restart rule is not the main limitation. The gap from the prior to the oracle is much larger than the gap between the two restart rules. In blocks 1 and 2 the conditional prediction is already better than the measurement update. Representation error and the recovery step both remain. The batch-versus-window choice does not account for them.

6. These numbers do not support a new claim in the main text that CLP improves on G(z0). The negative restart result is usable as a checked limitation. It should not be merged into the F3 grid.

7. Usable now: the extended CLP table was reproduced to 2.2e-16 RMSE; per-window restart selection does not change the test conclusion; the oracle, with test leakage, shows unused decoder fit; block 0 is where the update helps. Premature: calling the oracle an achievable error, treating the 18 cells as independent, or rewriting the noise-benchmark conclusions from this well.

The nine cells are three seeds inside three depth blocks. The block is the dependent unit.
This audit does not edit the manuscript.
