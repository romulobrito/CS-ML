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

We reorganized the Introduction to present the practical problem before the
mathematical formulation. Dense wireline logs provide indirect information
along the well, while the target property is known at only a few depths. The
task is to reconstruct the remaining profile while retaining those known
values as explicit, although not exact, measurement constraints.

The Introduction now explains CLP-CSGM in operational terms: a decoder
represents candidate target windows, dense logs predict the conditional code
\(\boldsymbol{z}_0\), and latent refinement balances agreement with the sparse
measurements against deviation from that code. It does not claim that direct
predictors are generally inadequate or that CLP-CSGM is universally
preferable.

The objective and its notation have been moved to Method, subsection
Test-time reconstruction. Related Work now contains the detailed distinction
from log completion, geomodel translation, and direct regression.

Where the text was changed: Introduction; Method, subsection Test-time
reconstruction; Related work.

---

## R2-02 -- Processed targets and artificial subsampling

**Status:** `closed`

**Proposed reply:**

We agree that the origin of the targets is central to the interpretation of
the study. The Abstract and Conclusion now state that the cross-well clay
volume is derived from gamma ray through the Larionov relation, that the
F03-4 porosity target is processed and density-derived, and that the sparse
observations are coordinate subsamples of those curves rather than
independently acquired core or laboratory measurements.

The revised text therefore presents the F3 experiments as reconstruction of
processed logs under controlled subsampling, not as validation against
independent petrophysical measurements. It also distinguishes the separate
laboratory-porosity supplement, for which the direct autoencoder has the
lower pooled mean RMSE, and retains the need for broader validation.

Where the text was changed: the Abstract and the Conclusion. The corresponding
provenance and scope statements remain in Data and exploratory analysis and
Interpretation, scope, and limitations.

## R2-03 -- Realism of excluding GR and sonic

**Status:** `closed`

**Proposed reply:**

These configurations deliberately withhold a log that would otherwise provide
a direct calculation or a near-direct proxy for the processed target. In the
cross-well benchmark, gamma ray is excluded because \(V_c\) was derived from
it through the Larionov relation. When gamma ray is available, that relation
is the appropriate calculation and CLP-CSGM is not a replacement for it.
Reintroducing gamma ray in the step-16 supplement reduces the absolute error
without changing the ordering of CLP-CSGM Ridge, the direct AE, and
measurement-only CSGM.

In the F03-4 benchmark, sonic is excluded because it is nearly collinear with
the processed density-derived porosity target (Pearson
\(r\approx0.9996\)); gamma ray is the only dense input. The resulting errors
do not transfer to a conventional porosity interpretation using density,
neutron, and sonic logs.

The practical analogue is a setting in which the log required for a
conventional estimate is missing or unreliable but a small number of target
measurements is available. The F3 experiments reproduce only the
restricted-input and sparse-observation structure of that setting, because
their sparse values are subsamples of processed curves. We therefore present
them as controlled method comparisons, not operational validation.

Where the text was changed: Data and exploratory analysis; Experimental
setup; Interpretation, scope, and limitations.

---

## R2-04 -- Sensitivity to observation placement

**Status:** `closed`

**Proposed reply:**

We added a sensitivity analysis comprising the published realization and four
additional draws on the cross-well grid while retaining model seeds 7, 23,
and 41. The additional realizations redraw both the coordinates and the
measurement noise; the direct autoencoder is refitted and the CLP-CSGM
penalty is reselected on validation. The analysis therefore assesses the
combined measurement realization and does not isolate coordinate placement
from noise.

The cell-level winner matches the published result in all nine cells. At
steps 16 and 32, CLP-CSGM Ridge is lower than the direct autoencoder in all
15 seed-draw comparisons at every measurement ratio. At step 8, AE is lower
in 15 of 15 comparisons at rho=0.05 and 10 of 15 at rho=0.10, while
CLP-CSGM is lower in 11 of 15 at rho=0.20. Thus, the low-support ranking is
stable across the tested draws, whereas step 8 remains less stable.

These are repeated draws on the same held-out well, not independent wells or
evidence of geological generalization. The submitted three-seed tables remain
unchanged.

Where the text was changed: Experimental setup; Results; Appendix,
subsection Sparse-coordinate placement.

## R2-05 -- Statistical robustness with more realizations

**Status:** `closed`

**Proposed reply:**

We added two complementary robustness checks while retaining the original
three-seed tables as the main result. First, a ten-seed panel refits both
methods and redraws coordinates and measurement noise. None of the nine
paired 95% Student-t intervals includes zero. The regime-dependent ranking is
unchanged: CLP-CSGM Ridge is lower at steps 16 and 32, whereas the direct
autoencoder is lower at step 8 for rho=0.05 and rho=0.10.

Second, a panel comprising the published realization and four additional
draws holds model seeds 7, 23, and 41 fixed while redrawing coordinates and
noise together. CLP-CSGM is lower in every draw at steps 16 and 32; the
step-8 ordering remains less stable and matches the cell-level pattern
reported in the manuscript.

These checks do not separate coordinate placement from measurement noise,
and neither adds independent wells. The intervals summarize variability on
the single held-out well F03-4; overlapping windows are not treated as
independent samples.

Where the text was changed: Experimental setup, paragraph Uncertainty
summaries; Results; Appendix, subsections Ten-seed CLP versus AE panel and
Sparse-coordinate placement.
