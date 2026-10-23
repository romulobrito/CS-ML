# Investigation 25 - contiguous permeability gaps

Well 9-BRSA-1254-RJS. Base commit 6e98a85d5c66e57410803eada2a88e63bbe8f338.
No architecture change and no test-target selection.

## Protocol

The frozen split and the original random masks were hashed before scoring. One stored random-mask cell, fold 0 seed 7 rho 0.25, was rebuilt first. Prior, extended CLP, and direct regression each matched the stored plug predictions within 4.44e-16. The contiguous-gap run started only after that check.

Each test and validation block then received one missing interval. The offset comes from SeedSequence(seed, fold, round(100*rho), split code). Permeability is not an input. The observed count uses the same observation_count rule as the corrected protocol: 13 or 14 observed plugs and 40 reserved plugs at 25 percent, 26 or 27 observed and 27 reserved at 50 percent. Mean gap length is 17.26 m at 25 percent and 10.88 m at 50 percent.

Lambda, including no update, is chosen on the validation gap. The search starts at no update and replaces it only when a later validation RMSE is strictly smaller. Eight of 18 cells stayed on no update. The validation block has 16 plugs, so that choice is a small-sample score.

Direct regression is the corrected-protocol fit: four logs, training plugs only. Interpolation is linear in depth and holds the nearest measured value outside the observed span. RMSE is one value per reserved plug.

Sixteen of 18 test gaps are interior. Two touch the deep edge of the block, so every reserved plug in those two gaps is an extrapolation. No gap touches the shallow edge.

## What was measured

Primary mean of the nine fold-seed RMSE values, log10(k):

| rho | prior | CLP | interpolation | direct |
| ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.569 | 1.568 | 1.725 | 1.728 |
| 0.50 | 1.559 | 1.558 | 1.893 | 1.717 |

Mean of CLP minus prior across 18 cells: -0.001. CLP is lower in 6 cells and the largest absolute cell difference is 0.089 (fold 1, seed 7, rho 0.50, CLP worse). CLP is lower than interpolation in 12 of 18 cells and lower than direct regression in 13 of 18. The prior has the same 12 and 13.

Block means:

| rho | block | prior | CLP | interpolation | direct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 0 | 1.285 | 1.267 | 1.024 | 1.140 |
| 0.25 | 1 | 1.930 | 1.924 | 2.452 | 2.118 |
| 0.25 | 2 | 1.493 | 1.513 | 1.699 | 1.925 |
| 0.50 | 0 | 1.231 | 1.209 | 1.000 | 1.038 |
| 0.50 | 1 | 2.014 | 2.044 | 2.838 | 2.163 |
| 0.50 | 2 | 1.431 | 1.421 | 1.840 | 1.950 |

On block 0, interpolation is the lowest of the four methods at both fractions, and direct regression is next. CLP improves the prior there by 0.018 and 0.023 and remains above both simple methods. On blocks 1 and 2 the prior is lower than interpolation and direct regression. CLP stays within 0.030 of the prior and is slightly worse on block 1 at 50 percent and on block 2 at 25 percent.

A window has length 8. At 25 percent every test block has 33 windows with no observation. At 50 percent it has 20. A reserved plug covered only by those windows is decoded as G(z0). That is 26.8 of 40 plugs at 25 percent and 13.8 of 27 plugs at 50 percent. On those plugs the largest absolute CLP-prior difference is 6.2e-8, which is float32 batch noise. Their RMSE is therefore the same:

| rho | coverage | prior | CLP | interpolation | direct |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0.25 | rim | 1.242 | 1.203 | 1.485 | 1.663 |
| 0.25 | unreachable | 1.673 | 1.673 | 1.754 | 1.735 |
| 0.50 | rim | 1.463 | 1.455 | 1.766 | 1.580 |
| 0.50 | unreachable | 1.650 | 1.650 | 1.948 | 1.833 |

The rim is the only place an update can act. Its mean change is about -0.04 at 25 percent and -0.01 at 50 percent. On block 0 the rim still does not overturn the full-gap result: interpolation remains lower on the whole reserved set. The two edge gaps have interpolation means of 2.747 against a prior mean of 1.732. CLP differs from that prior by -0.008. Two cells are not a selected regime.

## Random masks, kept separate

The scored plugs are not the same, so these means are not pooled with the table above. Frozen random-mask means were prior 1.432, extended CLP 1.448, direct 1.622, interpolation 1.804 at 25 percent, and prior 1.469, extended CLP 1.430, direct 1.569, interpolation 1.543 at 50 percent.

The contiguous geometry makes 50 percent interpolation worse than it was under random masks (1.893 against 1.543). It does not create a CLP-minus-prior effect. The random-mask study already had CLP within a few hundredths of the prior.

## Relation to Investigation 24

Block 0 was the block whose prior residual was spatially coherent and whose observed residual predicted the reserved residual. The same block is the one where depth interpolation of permeability now beats the prior and CLP. Blocks 1 and 2 were the blocks where that residual did not transfer. Observations outside a long gap still do not correct the gap interior, because most of the interior is not seen by any window that contains a measurement.

## Answers

1. No material improvement. The mean change from the prior is -0.001, and the interior of the gap is G(z0) by the window construction.
2. CLP is lower than interpolation and direct regression on the nine-cell mean, and so is the prior. On block 0 both simple methods are lower than CLP. The mean is not a gain of latent assimilation over the best simple baseline.
3. The comparison with interpolation changes, especially at 50 percent, where one long gap is harder to bridge. The comparison with the prior does not. CLP remains tied with G(z0), as it was under random masks.
4. Yes. The block with coherent permeability favors interpolation. The blocks with weak residual transfer favor the prior, and the latent update cannot reach most of the gap.
5. No. There is no contiguous-gap regime in this design that supports a specific advantage of CLP-CSGM. A rim gain of a few hundredths does not change the reserved-plug ranking, and the two edge gaps are too few to define one.

## Limits

Each seed also moves the gap, so seed and geometry are not separated. Three blocks are the dependent units. Validation uses 16 plugs. The two edge gaps were not a prespecified subset. Random-mask and contiguous-gap RMSE values use different reserved plugs. No test target chose a lambda, a gap, or a method. The manuscript was not changed.
