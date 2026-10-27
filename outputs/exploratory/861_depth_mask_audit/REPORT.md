# Well 861 depth-mask audit

Read-only audit of `scripts/run_r1m11_861_phi_lab.py` and
`outputs/reviewer_revision/r1m11_861_phi_lab/`. No model was trained.
The runner and the published tables were not modified.

- Reference commit: `2e657987ac2425aace4191e49e2b9affb6bda509`
- HEAD at audit: `2e657987ac2425aace4191e49e2b9affb6bda509`
- Runner sha256: `95925647bc270bc67c2343be34f707ad46f9563591f24c69f96023bab2cd2f1c`
- Rows: 87
- Depth: 5205.91 m to 5233.72 m
- Depth gap min / median / max: 0.19 / 0.19 / 1.34 m
- Window 8 , step 1 , blocks 3

## Train, validation, and test

Blocks are contiguous groups of 29 rows, split before windowing.
A window is kept only when all 8 rows lie inside one of train,
validation, or test. The three row sets are disjoint in every fold.
No test row enters a training window or a validation window.
The decoder is fit on training windows. Lambda is selected on
validation windows. Test `Phi_lab` is not an input to either fit.

That separation does not remove geological adjacency. Fold 0: train-test gap 0.19 m (adjacent=1), val-test gap 14.48 m (adjacent=0); Fold 1: train-test gap 0.19 m (adjacent=1), val-test gap 5.33 m (adjacent=0); Fold 2: train-test gap 5.91 m (adjacent=0), val-test gap 0.76 m (adjacent=1).
An adjacent flag means consecutive row indices, not a shared
`Phi_lab` value. Neighboring depths can be statistically dependent.

See `tables/split_summary.csv`.

## Published mask

Inside a fold, seed, and observed fraction, one local Omega is drawn
on the 8 window offsets and reused by every window. The published
fractions 0.25 and 0.50 therefore mean 2 and 4 offsets, not 25% and
50% of the 29 physical depths.

The runner scores every window offset outside that local Omega and
pools those offsets. The docstring is correct at window level: a
local offset used as `b` in a window is not scored in that same
window. It is not correct as a statement about physical depth.

Across the 18 published cells (3 folds x 3 seeds x 2 fractions):

- Depths that are both measured and scored: 21 to 27 per block.
- Depths scored and never measured: 0 to 6.
- Depths measured at least once: 23 to 29 of 29 at fraction 0.25, and 25 to 29 of 29 at fraction 0.50.
- Share of test RMSE terms whose depth was also a measurement in another test window: 0.8409 to 1.0000.
- Same share inside the validation windows used to choose lambda: 0.6111 to 1.0000.
- Score pairs per cell: 132 at fraction 0.25 and 88 at fraction 0.50, against 29 depths.
- Maximum times one depth enters the RMSE: 6.

The same dual role appears inside the validation windows used to
choose lambda. See `tables/published_mask_summary.csv` and
`tables/published_depth_roles.csv`. Redrawn Omega strings match the
published detail file.

## What is leakage, and what is not

Demonstrated reuse of laboratory values inside the test evaluation:
for every published cell, the `Phi_lab` value of a physical depth is
placed in `b` of at least one test window and is also a scored target
in another overlapping test window. The share of pooled RMSE terms
with that property is the contaminated fraction above. This is why a
window-level holdout is not a holdout of laboratory depths.

Not demonstrated, and not claimed: the scored window's own `b` does
not contain the offset being scored. The local complement excludes it.
Train, validation, and test do not share rows, so this is not leakage
of test `Phi_lab` into the decoder or into lambda selection.

Dependence that is not that reuse:

- Overlapping windows are not independent samples. An interior depth
  sits in up to 8 windows. In the published scores one depth enters
  the RMSE up to 6 times, including depths that were never measured.
- Nearby depths remain correlated after any mask correction.
- `csgm_recover_with_prior` applies one operator to the whole batch
  and keeps the restart with the lowest summed loss. Window updates
  do not share each other's measurements inside the data term, but
  the chosen restart is common. This audit does not rerun recovery,
  so it does not measure how much that shared choice moves a prediction.

The published files store RMSE only. They do not store per-window
predictions, so residual correlation was not estimated.

## Lithotype

`Lithotype` is one of the four inputs. The Auddys legend labels the
codes Grainstone, Shrubs, Laminite, and Spherulite. The legend does
not define the code from `Phi_lab`. The source scan finds no
assignment whose right-hand side uses `Phi_lab`, `k_lab`, `RQI`,
`FZI_lab`, or `HFU`. Other lines that mention the column are a
per-fold count and an empty placeholder in a different well export;
they are listed in the provenance table and do not build the code.
On the 87 rows the code is not a threshold cut of `Phi_lab` or of
`FZI_lab`, and it is not equal to `HFU`.

`RQI` matches `0.0314 * sqrt(k_lab / Phi_lab)` and `FZI_lab` matches
`RQI / (phi / (1 - phi))` within 1e-6. Those columns, `HFU`, and
`k_lab` are not inputs. Density, neutron porosity, sonic porosity,
and `Phi_ND` are also excluded. Nothing in this table shows that
`Lithotype` was constructed from the target. The remaining limit is
provenance: the integer is copied from the log table, and this file
is not an assay ledger.

See `tables/lithotype_provenance.csv`.

## CLP and the direct AE

The runner draws Omega once and passes the same `b` to CLP recovery
and to `concat_ub` for the direct AE. The published detail file has
one Omega string per fold, seed, and fraction. The two methods
therefore see the same window-level observations, including the same
cross-window reuse.

Prior-only does not receive `b`. The runner has no unconditioned CSGM
and no depth interpolation. A comparison with those baselines is not
in the published result.

## Two corrections, not implemented

Global depth mask. Draw the observed fraction on the 29 physical
rows. A row is a measurement in every window that contains it, or it
is scored, never both. All 29 depths keep a role. Because the current
recovery call accepts one operator for the batch, each window needs
its own call when the local support changes. That uses the existing
function and does not edit `csgm_m2_module.py`. Held-out depths that
sit in several windows must be averaged to one residual before the
RMSE. At 7 observed rows out of 29, a single window of 8 is empty
with probability 0.0745. At 14 observed rows the probability is 0.0015.
In the 9 counterfactual draws at fraction 0.25, empty windows range
from 0 to 4 out of 22. At fraction 0.50 the count is 0 in every draw.
An empty window has no data term. The correction has to state that
case before any training run. The direct AE would need a fixed-width
embedding such as `embed_sparse_observation`, because `b` would no
longer have one width.

Non-overlapping test windows. Step 8 yields 3 windows and covers 24
of 29 depths in every block. The 5 dropped depths are the deepest
tail of each block, 15 of 87 rows. No covered depth has both roles,
and each covered depth is scored once. The lost interval is a fixed
tail, so the evaluated set is shorter than the block. The tail is
not a proportional facies sample: fold 0 (code 2 loses 3 of 19; code 3 loses 1 of 3; code 4 loses 1 of 1); fold 1 (code 1 loses 1 of 19; code 2 loses 4 of 10); fold 2 (code 4 loses 5 of 12).
Counts are in `tables/facies_coverage_nonoverlap.csv`.

The global depth mask is the appropriate primary correction. It
answers whether a laboratory depth was given to the test evaluation.
Non-overlapping windows are a coverage-limited check, not a substitute:
they discard the same tail in every fold. Neither correction is
implemented here. Interpolation, if added later, has to use `Depth(m)`
because the gap is not uniform.

Counterfactual draws use the published seed formula plus salt
9100. They illustrate the geometry. They are not a new experiment.
See `tables/alternative_coverage.csv`.

## Checks

- [PASS] head_matches_reference: HEAD 2e657987ac2425aace4191e49e2b9affb6bda509
- [PASS] omega_matches_published_file: Redrawn local Omega matches heldout_rmse_detail.csv
- [PASS] split_rows_disjoint: Train, validation, and test row sets do not intersect
- [PASS] test_rows_absent_from_fit_windows: No test row is inside a training or validation window
- [PASS] within_window_score_excludes_own_measurement: Each window scores only offsets outside its own Omega
- [PASS] physical_depth_has_both_roles: Every published cell measures and scores the same depths
- [PASS] clp_and_ae_share_omega: One Omega and one b feed CLP and the direct AE
- [PASS] recovery_uses_one_operator_per_call: A per-window operator needs a separate call; restart choice is shared
- [PASS] lithotype_not_a_phi_transform: No right-hand side builds Lithotype from Phi_lab; ranges overlap; legend is facies text
- [PASS] rqi_fzi_identities_and_exclusion: RQI and FZI match the lab transform and stay out of u
- [PASS] global_mask_removes_dual_role: Counterfactual physical masks have no depth in both roles
- [PASS] nonoverlap_removes_dual_role_and_drops_tail: Step 8 scores each covered depth once and drops the deepest 5 rows
- [PASS] published_summary_has_nine_cells: The published mean uses 3 folds x 3 seeds, not 9 wells
- [PASS] no_interpolation_or_unconditioned_csgm: Those comparators are absent from the published runner

## Rerun

```
python3 scripts/audit_r1m11_861_depth_masking.py
```

Outputs stay under `outputs/exploratory/861_depth_mask_audit/`.
