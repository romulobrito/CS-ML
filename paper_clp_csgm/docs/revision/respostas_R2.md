# Reviewer 2 -- proposed replies

Each ID has its own slot. If the same manuscript change fully answers more
than one ID, the English reply is copied verbatim.

Every completed reply names where the manuscript text was changed (Abstract,
section title, or appendix subsection title), not only what was changed.

Source comments: `docs/revision/CLP_CSGM_Pareceres_Consolidados.md`.

---

## R2-01 -- Motivation before the mathematical formulation

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for this structural comment. The Introduction now
states the practical problem before any formulation. Dense wireline logs run
along the target well but measure the target property only indirectly. A few
depths already carry a value of that property, and the task is to reconstruct
the remaining depths while treating those known values as explicit
measurements. The motivation is to investigate that sparse-measurement
assimilation setting. It does not claim that direct predictors are inadequate
in general, nor that CLP-CSGM is universally required or generally superior.

A direct map that joins those sparse values to the dense logs is not required
to reproduce them after training. This paper keeps them as measurements, so
the reconstructed window is penalized for disagreeing with them. That penalty
does not force every observation to be met exactly. Recent log-completion,
geomodel-translation, and direct-regression studies are briefly introduced,
while their methodological differences from CLP-CSGM are developed in Related
Work. The distinction concerns how available observations enter inference,
rather than whether competing methods use known observations.

The objective has been removed from the Introduction. It is stated in Method,
subsection Test-time reconstruction, as equation clp_csgm_objective. The
Introduction describes that step in words and points to the subsection: a
decoder supplies the windows the reconstruction may use, the conditional map
predicts z_0 from the dense logs, and latent refinement balances agreement
with the sparse observations against deviation from z_0. A residual at the
observed depths can remain. Ridge remains the linear low-capacity reference
for the map from dense logs to z_0. It is not presented as best in every
regime.

Where the text was changed: the Introduction (from the opening through the
paragraph that points to Test-time reconstruction). The objective itself is
unchanged in Method, subsection Test-time reconstruction. Related Work retains
the recent-study positioning refined under R1-M10 and R1-m02.

---

## R2-02 -- Processed targets and artificial subsampling

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for emphasizing that the derived origin of the F3
targets is central and must appear early. In the original submission this
limitation was stated in the Data section and in the Discussion, but it was
not explicit in the Abstract or the Conclusions.

We have revised both. The Abstract now states that clay volume is obtained
from gamma ray by the Larionov relation, that porosity is processed and
density-derived, and that sparse observations are coordinate subsamples of
these same curves rather than independently acquired core or laboratory
measurements. It also states that, in a supplementary laboratory-porosity
check, the direct autoencoder has the lower mean error. The Conclusions
repeat this F3 evidence limit before the sentence on remaining validation
needs, and they separate that laboratory supplement from the processed-log
benchmarks. The closing Abstract sentence states that the benefit depends
on the regime and that broader validation is still required. This wording
does not close the request for a broader independent-measurement
validation.

Where the text was changed: the Abstract, and the Conclusion (the F3
evidence-limit paragraph, before the remaining-work sentence). The Data and
exploratory analysis section and the Interpretation, scope, and limitations
section already stated this origin and were left in place.

No new experiments or numerical results were added for this comment.

The supplementary laboratory-porosity experiment addressed under
R1-M11 provides a limited check against an independently measured
target, although sparse observations are still obtained by coordinate
subsampling. The direct autoencoder has the lower pooled mean RMSE.
Broader validation across independent wells and measurement settings
remains open.

## R2-03 -- Realism of excluding GR and sonic

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for asking what field situation these input choices
represent. They are not the log suite we would use when every log is
available. Each log was left out because keeping it would let the model read
a curve that already determines the target.

In the cross-well benchmark, clay volume is computed from gamma ray by the
Larionov relation. Gamma ray is left out of the input so the model does not
read the log from which the target was computed. This is not the usual field
interpretation, in which gamma ray is logged and clay volume is obtained from
that relation directly. If gamma ray is available, that clay volume is
already known. The appendix puts gamma ray back at step 16. The absolute
error falls, and the ordering of CLP-CSGM Ridge, the direct AE, and
measurement-only CSGM does not change. That run checks the effect of the
channel. It is not the suite used for the main comparison.

In the F03-4 porosity benchmark, sonic travel time follows the processed
porosity almost one to one (Pearson r about 0.9996). The density curve behind
that porosity was itself predicted from sonic and gamma ray. Sonic is left
out, and gamma ray is the only dense log. A few values of the porosity curve
are kept as measurements. This is not a porosity interpretation from density,
neutron, and sonic together, and the gamma-ray-only errors do not transfer to
that suite.

The two F3 benchmarks therefore evaluate sparse assimilation on processed
target logs when the log that defines or copies the target is withheld. They
are not a recommendation of which logs to use in a fully logged well, and
they are not a test against core or laboratory measurements.

Interpretation, scope, and limitations now also names a field analogue: a
well in which the log required for a conventional estimate is missing or
unreliable, while a few independently acquired property measurements are
available. The same paragraph states that the F3 benchmarks reproduce only
the restricted-input and sparse-observation structure of that situation.
Their sparse target values remain subsamples of processed curves. The tests
are a controlled comparison of reconstruction methods, not an operational
validation of that field scenario.

Where the text was changed: Data and exploratory analysis (the input-design
paragraph for clay volume, and the F03-4 paragraph after the correlation
table); Experimental setup (the cross-well clay-volume paragraph, and the
F03-4 GR-only porosity paragraph); Interpretation, scope, and limitations
(the paragraph on the empirical scope of the two F3 benchmarks, and the
field-analogue paragraph that follows it).

---

## R2-04 -- Sensitivity to observation placement

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for this comment. The main protocol still uses one
coordinate set for each seed and measurement ratio, held fixed across
windows. A supplement redraws the sparse observations five times on the
cross-well clay-volume grid while the model seeds stay at 7, 23, and 41.
The measurement seed is the model seed plus a fixed offset. Offset 0 is
the published stream. Each later offset draws a new coordinate set and a
new measurement-noise realization from the same generator. The model-seed
identities are unchanged, so the decoder and the ridge anchor keep the
same initialization. The check does not isolate coordinate position from
measurement noise. The direct autoencoder is fit again on each draw,
because the sparse values are part of its input, and the CLP-CSGM penalty
weight is selected again on the validation windows of that draw. The score
is the same full-window RMSE as the main grid.

The cell-level winner matches the published ranking in all nine cells. At
steps 16 and 32, CLP-CSGM Ridge has lower RMSE than the direct
autoencoder in all 15 comparisons (3 seeds times 5 placements) at every
measurement ratio. At step 8 and rho = 0.05, the direct autoencoder is
lower in all 15. At step 8 and rho = 0.10, the direct autoencoder is
lower in 10 of 15. At step 8 and rho = 0.20, CLP-CSGM Ridge is lower in
11 of 15. Individual comparisons in those two step-8 cells can reverse,
so the step-8 ranking is the less stable of the two regimes. The
advantage at the smaller training support does not depend on the one
published observation draw. These 15 comparisons are three model seeds
times five draws on the single held-out well F03-4. They are not 15
independent wells and they are not evidence of cross-well generalization.
The submitted three-seed tables are unchanged.

Where the text was changed: Experimental setup (the paragraph after the
sparse-measurement equation); Results (after the ten-seed and
explicit-mask sentences); Appendix Sparse-coordinate placement.

## R2-05 -- Statistical robustness with more realizations

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for this comment. The main comparison still uses the
three seeds 7, 23, and 41, and those Student-t intervals remain wide. Two
supplements add realizations and leave those tables as the main result.

The ten-seed panel refits CLP-CSGM Ridge and the direct autoencoder. Each
seed also draws its own sparse-measurement realization, so training
randomness, observation coordinates, and measurement noise change together.
None of the nine paired 95 percent intervals include zero. Those intervals
use Student-t with nine degrees of freedom and describe seed variability on
the single held-out well F03-4. They are not ten independent geological
tests, and the nine cells are not a joint multiplicity-corrected test.
CLP-CSGM Ridge remains lower where training support is smallest, at steps
16 and 32. The direct autoencoder remains lower at step 8 for rho = 0.05
and 0.10.

The second supplement holds the three model seeds fixed and redraws the
sparse-measurement realizations, varying observation coordinates and
measurement noise together. It assesses sensitivity to these measurement
realizations without isolating coordinate placement from noise. At steps
16 and 32, CLP-CSGM Ridge has lower RMSE in every draw. At step 8 the
ranking is less stable: the direct autoencoder is lower in all draws at
rho = 0.05 and in 10 of 15 at rho = 0.10, and CLP-CSGM Ridge is lower in
11 of 15 at rho = 0.20. The cell-level winner still matches the published
ranking in all nine cells. These comparisons do not establish general
superiority, including against depth interpolation at the reference noise.

Overlapping windows are not treated as extra independent samples. The
extra realizations are the ten seeds and the five observation draws. The
supplements add no independent wells.

Where the text was changed: Experimental setup (paragraph Uncertainty
summaries); Results (after the placement paragraph); Appendix Ten-seed
CLP versus AE panel; Appendix Sparse-coordinate placement.
