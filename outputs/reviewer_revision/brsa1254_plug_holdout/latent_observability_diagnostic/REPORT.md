# Latent observability diagnostic, well 9-BRSA-1254-RJS

Base commit: a25d94a544827cd906f188828eac7a0318fe4e21.
Frozen split and mask hashes match the restart-oracle audit.
Lambda is the validation choice already stored by the lambda extension.
The oracle is not used to train, to choose lambda, or to recover.

## Checks

The analytical Jacobian of physical G matches PyTorch autograd.
Maximum absolute difference on eight latent codes, including a ReLU kink: 2.15e-7.
torch.nn.ReLU and the analytical mask both use a zero subgradient when the preactivation is exactly 0.

The recovery objective is (||M G(z) - b||^2 + lambda ||z - z0||^2) / n_group, in physical log10(k) units.
Reported eigenvalues are sigma(J_M)^2 + lambda.
The Hessian carries an extra factor of two on both terms, so condition numbers and effective dimension are unchanged by that factor.
Division by the group size scales the data term and the penalty together.

Test RMSE of the reimplemented collective recovery matches the stored extended-rule score.
Maximum absolute gap: 1.45e-7, inside the 1e-4 tolerance.
The gap is larger than the bit-level match of Investigation 22 because this script reimplemented the recovery instead of calling it.

Structural zeros are the latent directions with no measurement row.
A singular value among the available rows is called numerically small only when it is below 1e-5 times the largest singular value.
No available singular value met that cutoff. Every window with at least four observations had numerical rank 4.
The correlation for numerical deficiency is undefined because that fraction is zero in every cell.

## Test means of the nine fold-seed cells

| rho | prior | CLP | CLP minus prior | windows with fewer than 4 observations | mean rank at z0 | mean effective dimension |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 1.432 | 1.448 | +0.016 | 0.874 | 2.10 | 1.46 |
| 0.50 | 1.469 | 1.430 | -0.039 | 0.351 | 3.50 | 2.10 |

These RMSE means repeat the extended CLP table. The observation fractions repeat the mask count from the previous note.

## Block means on the test plugs

| block | rho | CLP minus prior | fraction with fewer than 4 observations | mean effective dimension | median regularized condition | observation RMSE before | observation RMSE after |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.25 | -0.112 | 0.899 | 2.06 | 1.3e4 | 1.559 | 0.046 |
| 0 | 0.50 | -0.247 | 0.435 | 2.48 | 1.4e3 | 1.438 | 0.368 |
| 1 | 0.25 | +0.109 | 0.844 | 1.08 | 27 | 2.041 | 0.993 |
| 1 | 0.50 | +0.070 | 0.291 | 2.79 | 29 | 1.766 | 1.132 |
| 2 | 0.25 | +0.051 | 0.879 | 1.12 | 4.2 | 1.612 | 1.087 |
| 2 | 0.50 | +0.058 | 0.326 | 1.25 | 2.9 | 1.627 | 1.090 |

On the validation plugs, the same lambda lowers the reserved RMSE in every block and both fractions.
The test gain remains only in block 0.

## Answers

1. Effective observability does not explain the block difference. At fixed rho, the three blocks have similar fractions of windows with fewer than four observations and similar numerical rank. The correlation between that fraction and the RMSE change is +0.025 on the 18 cells and +0.056 on the six block-rho means. The block where the update helps is the one with the larger regularized condition number, not the better-conditioned one.

2. There is scale separation, and there is no rank collapse beyond the missing rows. Among windows with at least four observations, the rank is 4 in all 326 windows. The median ratio of the largest to the smallest of those four singular values is about 7 to 16, and it is not smaller in block 0. Twenty-eight test windows have no observation. Their measurement Jacobian is the zero matrix, and they stay at G(z0).

3. The selected lambda changes the size of the latent step, and it does not by itself account for the sign of the test change. Lambda 30 in block 1 leaves the latent displacement near 0.07 and the RMSE change near zero. Lambda 0.01 in the same block moves the latent code by about 4.3 and raises the reserved RMSE by 0.31. Block 0 uses lambda near 1e-4 to 1e-3, fits the observations tightly, and lowers the reserved RMSE. The harmful blocks are more strongly regularized than block 0.

4. On all 16 cells that run recovery, the mean observation RMSE falls, from about 1.66 to 0.66. In 8 of those 16 cells the reserved RMSE rises. The update can fit the supplied plugs and still worsen the plugs held out inside the same block.

5. The mask count and the numerical rank are not the main difference between blocks. The conditional prior is already better than the measurement update in blocks 1 and 2. Investigation 22 showed that the decoder can fit the true test windows more closely than this sparse recovery does. The remaining gap is the quality of G(z0) outside block 0, together with an update that follows the observations even when that move does not help the reserved plugs.

6. A new adaptive penalty is not justified by these spectra. Increasing lambda can freeze a harmful update, which the existing no-update choice already does when validation prefers it. It does not explain why block 0 improves under a very small lambda. The block sign change is a prior and transfer question.

7. The reviewer reply can use a negative result: sparse windows are common, the analytical Jacobian was checked, and neither the observation count nor the rank separates the block where CLP helps from the blocks where it hurts. This does not support a new claim in the main text, a change of architecture, or a change of the published penalty.

The 18 cells are three seeds in three blocks. Correlations on those cells are descriptive. No p-value on overlapping windows is reported.
Spectra at z0 describe the problem before the update. Spectra and displacements at zhat also reflect the optimization and are not treated as causes.
This audit does not edit the manuscript.
