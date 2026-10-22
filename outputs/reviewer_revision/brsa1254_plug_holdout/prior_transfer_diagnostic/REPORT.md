# Investigation 24 - prior transfer diagnostic

Well 9-BRSA-1254-RJS. Base commit 14759285a6ab0462e4f2b215ddd9e32f39e34b5a.
No architecture change, no new hyperparameter search, and no test-target selection.

## What was measured

The stored prior and collective-CLP scores from Investigation 22 match the stored cell table within 1e-4. Rebuilding G(z0) with the frozen training rule reproduces every stored scored-plug prediction. The maximum absolute gap is 4.44e-16. The recomputed direct regression matches the corrected-protocol table within 1.83e-15.

G(z0) does not depend on the observation mask. The two residual rules are fixed: copy the residual of the nearest observed plug, or interpolate the observed residuals in depth and hold the end values. Neither rule has a parameter chosen on the test.

Primary scores below are means of the three seeds. RMSE is on reserved plugs, in log10(k), unless a row says the whole block.

## Prior quality by block

Whole-block prior RMSE, mean of three seeds:

| block | train | validation | test | test bias |
| --- | ---: | ---: | ---: | ---: |
| 0 | 1.482 | 3.169 | 1.281 | +0.129 |
| 1 | 1.049 | 1.555 | 1.785 | +0.121 |
| 2 | 1.158 | 2.131 | 1.437 | +0.021 |

The prior is systematically worse on test block 1. Its in-sample error on that block is the smallest of the three, so the gap is a transfer gap. Validation block 0 is a separate failure: RMSE 3.169 and bias -2.080. Test bias is small in every block, so the test problem is not a constant offset.

Reserved-plug means:

| rho | block | prior | nearest residual | interpolated residual | direct | CLP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0 | 1.225 | 1.435 | 1.141 | 1.109 | 1.113 |
| 0.25 | 1 | 1.692 | 2.629 | 2.432 | 1.863 | 1.802 |
| 0.25 | 2 | 1.378 | 2.051 | 1.826 | 1.895 | 1.429 |
| 0.50 | 0 | 1.348 | 1.093 | 0.996 | 1.170 | 1.101 |
| 0.50 | 1 | 1.741 | 2.232 | 1.983 | 1.779 | 1.811 |
| 0.50 | 2 | 1.319 | 1.995 | 1.725 | 1.760 | 1.378 |

Across the nine cells, the means are:

| rho | prior | nearest | interpolated | direct | CLP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.432 | 2.038 | 1.800 | 1.622 | 1.448 |
| 0.50 | 1.469 | 1.773 | 1.568 | 1.569 | 1.430 |

## Distribution shift

Training-standardized Wasserstein distance on the test block:

| block | GR | RHOB | NPHI | RT | fraction of GR outside training |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.413 | 1.267 | 1.843 | 0.591 | 0.057 |
| 1 | 0.233 | 0.515 | 0.623 | 0.422 | 0.000 |
| 2 | 1.428 | 0.676 | 1.241 | 0.640 | 0.333 |

For GR and NPHI in block 0, and GR in block 2, the Wasserstein distance equals the absolute standardized mean shift. Those margins move mainly by location. Mean absolute change in pairwise channel correlation is 0.641, 0.165, and 0.561 in blocks 0, 1, and 2.

Block 1 has the smallest marginal and joint shift and the worst prior RMSE. Blocks 0 and 2 have the larger shifts, and block 0 has the best prior RMSE. A marginal shift of the logs does not account for the prior-quality ranking. It is also not the conditional distribution of permeability.

## Observed residuals versus reserved residuals

Spearman correlation between the reserved prior residual and the residual at the nearest observed plug, mean of six cells:

| block | correlation | lag-1 residual correlation | semivariance at 0-0.5 m |
| ---: | ---: | ---: | ---: |
| 0 | 0.606 | 0.671 | 0.501 |
| 1 | 0.132 | 0.216 | 2.222 |
| 2 | -0.038 | 0.077 | 1.556 |

Depth interpolation of the observed residual beats the prior in all 6 block-0 cells and in none of the 12 cells from blocks 1 and 2. The nearest-residual rule is worse than interpolation and worse than the prior in block 0 at 25 percent. Mean nearest-observation distance is 0.77, 1.07, and 0.86 m at 25 percent, so the blocks are not separated by how far the reserved plugs sit from an observation.

The same block split appears in direct regression: it beats the prior in all 6 block-0 cells and in 1 of 12 cells elsewhere. The pattern is not specific to the latent representation.

On the 18 test cells, the Spearman association between the nearest-residual correlation and the CLP-minus-prior improvement is 0.815. The association between the interpolated-residual gain and the CLP gain is 0.783. These are descriptive. The 18 cells share blocks, windows, and seeds.

## Validation versus test

Mean gain of residual interpolation, prior RMSE minus interpolated RMSE:

| block | validation gain | test gain |
| ---: | ---: | ---: |
| 0 | +1.116 | +0.218 |
| 1 | +0.057 | -0.491 |
| 2 | -0.338 | -0.427 |

The validation slice of block 0 strongly favors the correction because the prior there is biased by about -2.1. That gain does not have the same size on the test. Block 1 does not warn on validation that the correction will raise the test error. The Spearman association of the 18 validation gains with the 18 test gains is 0.515. It is not a decision rule.

## Answers

1. Yes. On the whole test block, prior RMSE is 1.281, 1.785, and 1.437. The middle block is worse, and its training error was the best of the three.
2. No marginal log shift measured here explains that ranking. The worst prior block is the closest to the training margins.
3. Only in block 0. There the nearest observed residual tracks the reserved residual (correlation 0.606) and interpolation lowers the error in 6 of 6 cells. In blocks 1 and 2 the correlation is 0.132 and -0.038, and interpolation raises the error in 12 of 12 cells.
4. In block 0, yes, and the latent update already uses most of it. At 25 percent CLP (1.113) and interpolation (1.141) are close; at 50 percent interpolation (0.996) is a little lower than CLP (1.101). In blocks 1 and 2 there is no corrective residual signal of this form for the latent update to have missed. Using it makes the reserved error larger.
5. Yes, in the descriptive sense available from three blocks. The block where transfer works has lag-1 residual correlation 0.671 and short-lag semivariance 0.501. The blocks where it fails have lag-1 correlation 0.216 and 0.077 and short-lag semivariance 2.222 and 1.556. Distance to the nearest observation does not separate them.
6. Not as the next change. A residual assimilator helps where CLP already helps and hurts where CLP hurts. Validation does not identify the harmful cases. The result supports retaining G(z0) when the reserved plugs are the target and the residual is not spatially coherent. It does not support adding a residual layer to CLP-CSGM on this evidence.
7. Yes, as a negative mechanistic result. The latent update reduces error where the prior residual is coherent between observed and reserved plugs. It does not fail because a simple residual correction could have transferred information that the latent update ignored. The manuscript was not changed.

## Compatible reading and what remains open

The compatible reading is that block 0 has a permeability residual that varies slowly relative to the observation spacing, while blocks 1 and 2 do not. Both the latent update and a depth interpolation can use the first situation. Neither can invent information in the second.

Not established: a causal path from one log channel to the residual; a variogram model; a claim that another bandwidth, selected only on validation, would reverse blocks 1 and 2; an operational rule that chooses the prior without seeing the test.

## Limits

Three depth blocks are the dependent units. Seeds and overlapping windows are not independent observations. The semivariogram counts pairs inside one block. The two residual rules were declared before scoring and were not tuned. No test target was used to choose a mask, a lambda, or a correction.
