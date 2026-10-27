# Three-seed nested-mask and shared-restart grid

Exploratory. Not a manuscript result.
Reference commit: 1d490d2accccea0a08e65cb4302a72c45f887788.
Preserved pilots were not overwritten.

Seeds 7, 23, and 41. Budgets 24, 56, and 88. Observations 4, 6, and 13.
Positive lambda is chosen on F06-1 and shared by the two penalized starts.
The four latent configurations share decoder, mask, measurements, and perturbations.
Direct AE, prior-only, and interpolation use those same measurements.

clp_minus_* and effect columns are RMSE differences.
Negative means the first named method, or the named factor, has lower RMSE.
interaction = init_effect_with_penalty - init_effect_without_penalty.

The retained restart is the one with the lowest sum of the data term and the penalty term over the whole batch, divided by the number of windows. Restarts are not chosen separately for each window. This is the rule in csgm_recover_with_prior and it was not changed.

See LIMITATIONS.md.
