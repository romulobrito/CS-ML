# Nested masks and conditional-prior ablation

Exploratory study. These numbers are not manuscript results.

Reference commit for the previous factorial: cc57f2bd2babf4b3744bea53e92209bb92911fc9.
Previous outputs were read and were not overwritten.

This run: mode=pilot, seeds=[7], budgets=[24, 88], observed positions=[4, 6, 13].
Full grid (seeds 7, 23, 41 and budgets 24, 56, 88): not run.

Training subsets are the recorded nested indices, not a new draw.
For each seed, one permutation defines Omega_4, Omega_6, and Omega_13.
One noise field per split is sliced onto those prefixes.
The same mask and the same measurements are reused across budgets.

Models inside one seed and budget share one decoder:
- clp_csgm_ridge: start at h(u), positive lambda chosen on F06-1.
- clp_init_only: same start and same restart seed, lambda 0.
- measurement_only_csgm: zero latent start, lambda 0, seed+9200.
- prior_only: G(h(u)), no assimilation.
ae_direct and interpolation see the same b as the latent models.

Lambda selection uses unobserved RMSE on F06-1. F03-4 is not used.
Lambda 0 can win the validation curve and still is not assigned to full CLP.
F06-1 has eight overlapping windows from one well. They are not independent replicates.
The 97 test windows are one pooled score. They are not 97 independent trials.

clp_minus_* columns are RMSE_CLP minus RMSE_comparator.
Negative favors CLP. This sign is the opposite of the previous study's Delta.

A difference among full CLP, init-only, and measurement-only is not, by itself,
an isolated causal estimate. Latent recovery is non-convex, and measurement-only
uses the original seed offset.

G(E(y)) is diagnostic. It does not select lambda, a budget, or a method.

## Pilot record

Executed: seed 7, budgets 24 and 88, observation counts 4, 6, and 13.
Not executed: seeds 23 and 41, and the 56-window budget.
No commit. The manuscript and the previous factorial directory were not changed.

On this single seed, full CLP has lower pooled test RMSE than direct AE in all six cells.
The margin shrinks from 24 to 88 windows at every observation count.
Interpolation has lower RMSE than CLP in all six cells.
The positive penalty improves on init-only in all six cells, by a small amount at 4 and 6 observations and by about 0.004 at 13.
Measurement-only is close to full CLP at 88 windows with 4 or 6 observations.
Two of six selected lambdas sit on the new lower boundary, 1e-6.
These statements are descriptive of seed 7. They are not a general claim.
