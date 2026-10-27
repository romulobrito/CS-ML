# Limitations

Exploratory full grid. These numbers are not manuscript results.

F06-1 has eight overlapping windows from one well. All 28 pairs overlap.
Lambda selection on that well is not strong statistical evidence.
F03-4 does not choose lambda, a mask, a budget, or a method.

Each reported RMSE pools the test windows and the scored coordinates.
The 97 windows are not 97 independent trials. The paired unit is the seed.
Means across seeds are means of three seed-level scores.

The retained restart is the one with the lowest sum of the data term and the penalty term over the whole batch, divided by the number of windows. Restarts are not chosen separately for each window. This is the rule in csgm_recover_with_prior and it was not changed.
A later sensitivity check could score restarts inside each window.
That check is not part of this run.

Shared perturbations remove the restart-seed confound among the four latent starts.
They do not remove non-convex optimization.
The interaction is the difference of those paired effects, not a causal estimate.

Masks are nested prefixes of one permutation per seed.
Observation count is separated from a new noise draw.
The three seeds still use three different permutations.

The historical measurement-only call with seed+9200 stays in the seed-7 pilot.
This grid uses the shared-noise zero start instead of that offset.

The target is the well-log curve. This run does not use independent lab assays.
G(E(y)) is diagnostic and does not select lambda, a budget, or a method.
