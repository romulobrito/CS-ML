# Shared-restart init and penalty control

Exploratory extension. These numbers are not manuscript results.

Preserved pilot: commit feef3a7, directory scarcity_nested_masks_ablation.
That directory was not overwritten. Its measurement-only call remains the historical reference.

This run: seed 7, budgets 24 and 88, observation counts 4, 6, and 13.
Not run: seeds 23 and 41, and the 56-window budget.

The four shared-noise recoveries use one perturbation list drawn with seed 7.
Restart 0 is the exact start. Later restarts add the same N(0, 0.05^2) tensor.
Zero start with no penalty uses that shared list, not seed+9200.
The historical measurement-only call still uses seed+9200 and the original function.

The penalty weight is the positive lambda already chosen on F06-1 in the preserved pilot.
F03-4 does not choose lambda, the start, or the anchor.
The same lambda and the same h(u) are used for both penalized starts.

A remaining gap is non-convex optimization. Shared noise removes the restart-seed confound;
it does not turn the four losses into an isolated causal estimate.
The 97 test windows remain one pooled score.
