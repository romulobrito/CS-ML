# Reviewer 1 -- proposed replies

Each ID has its own slot. If the same manuscript change fully answers more
than one ID, the English reply is copied verbatim.

Every completed reply names where the manuscript text was changed (Abstract,
section title, or appendix subsection title), not only what was changed.

Source comments: `docs/revision/CLP_CSGM_Pareceres_Consolidados.md`.

---

## R1-M01 -- Origin of targets and scope of validation

**Status:** `closed`

**Reviewer's comment:**

> The cross-well target VcV_c is Larionov-derived from gamma ray, and the F03-4 porosity target is itself processed/density-derived. The authors acknowledge this limitation, but it is central and should be emphasized earlier, including in the Abstract and Conclusions.

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

---

## R1-M02 -- Few wells and operational transfer

**Status:** `closed`

**Reviewer's comment:**

> Only three wells are used for development and one well is held out for testing. That is useful as a controlled test, but it is not enough to support broad statements about cross-well robustness or operational transferability. The language should remain clearly benchmark-specific.

**Proposed reply:**

We thank the reviewer for highlighting the limitations associated with the
number of wells used in the cross-well evaluation. We agree that three
development wells and a single held-out well provide a controlled benchmark
but do not establish general cross-well robustness or operational
transferability.

We have revised the manuscript to make this distinction explicit. The
Abstract now states that the cross-well evaluation uses three wells for
model development and one well held out. It does not name the wells. The
Introduction states that the tests use three development wells and hold out
F03-4. Experimental setup, subsection Cross-well clay-volume, names F02-1,
F03-2, and F06-1 as the development wells and states that reported
cross-well performance refers to this single held-out well. The Conclusion
states that the cross-well comparison is a controlled single-holdout
benchmark on F03-4 and is not evidence of operational cross-well or
multi-field transferability. It also states that broader assessment will
require larger multi-well datasets, independently calibrated targets, and
geological labels.

Where the text was changed: the Abstract; the Introduction (third
contribution); Experimental setup, subsection Cross-well clay-volume (the
single held-out well); and the Conclusion (single-holdout paragraph).

No additional held-out wells were added for this comment. Further held-out
wells remain outside the evidence reported here.

---

## R1-M03 -- Few windows and resampling

**Status:** `closed`

**Reviewer's comment:**

> At the largest window step, the development pool contains only about 27 windows before validation splitting. The performance gains in those low-data settings are interesting, but with such small sample support, repeated resampling or additional held-out wells would make the conclusions much more convincing.

**Proposed reply:**

The reviewer is correct that support is small at the largest step. We also
corrected the count: step 32 produces 30 development windows, of which 27
are used for training and 3 for validation.

We added a supplementary step-32 analysis with ten random train/validation
partitions within the development wells, while retaining F03-4 as the held-out
test well and fixing the sparse-coordinate draw to paper seed 7. CLP-CSGM
Ridge has lower mean RMSE than AE [u,b] at every measurement ratio (0.0342
versus 0.0407 at rho=0.05; 0.0305 versus 0.0388 at rho=0.10; and 0.0188
versus 0.0362 at rho=0.20) and in all ten partitions.

This is a sensitivity analysis of the small development partition, not
independent validation or evidence from additional wells. Because length-64
windows overlap at step 32, every validation window shares depth samples with
a training window from the same well. The main nine-cell results were left
unchanged.

Where the text was changed: Experimental setup, subsection Training,
validation, and test protocol; Appendix, subsection Step-32
development-split resampling.

---

## R1-M04 -- Three seeds and uncertainty intervals

**Status:** `closed`

**Reviewer's comment:**

> The manuscript correctly uses Student-tt intervals and explicitly notes the limitation, but n=3n=3 is still very small. I suggest increasing the seed count if computationally feasible, especially because several intervals include zero and the ranking changes by regime.

**Proposed reply:**

We agree that three seeds provide limited evidence. We therefore added a
supplementary ten-seed panel for the same CLP-CSGM Ridge versus AE [u,b]
comparison, while retaining the submitted three-seed tables as the main
result.

With ten seeds, none of the nine paired Student-t intervals for
\(\Delta\mathrm{RMSE}\) includes zero. These are per-configuration intervals
and were not adjusted for simultaneous multiple comparisons. The
regime-dependent ranking remains: CLP-CSGM Ridge has lower RMSE at every
measurement ratio for steps 16 and 32 and at step 8, \(\rho=0.20\), whereas
AE is lower at step 8 for \(\rho=0.05\) and \(\rho=0.10\). Thus, the expanded
panel strengthens the evidence for the low-training-support regime but does
not support uniform superiority.

Each seed also redraws the sparse coordinates and measurement noise and
refits the models. The intervals therefore summarize seed-level variability
on the single held-out well F03-4; they do not isolate the source of
randomness or establish generalization across wells.

Where the text was changed: Experimental setup, subsection Baselines and
evaluation metrics; Results, subsection Cross-well low-data results; Appendix,
subsection Ten-seed CLP versus AE panel.

---

## R1-M05 -- Conditional advantage, not overall superiority

**Status:** `closed`

**Reviewer's comment:**

> CLP-CSGM outperforms the direct AE baseline in seven of nine cross-well cells, but loses in two cells; in the GR-only benchmark, AE becomes better at higher measurement ratios. This should be framed as a conditional advantage rather than overall superiority. The paper generally does this well, but the Abstract/Highlights could be even more explicit.

**Proposed reply:**

We agree and have revised the Abstract, Conclusion, and submission Highlights
to present the result as regime-dependent. CLP-CSGM has lower mean RMSE than
the direct autoencoder in seven of nine cross-well configurations, primarily
where training support is limited, and is higher in the other two. In the
GR-only benchmark, CLP-CSGM is lower at rho=0.20 and 0.30, the autoencoder
is lower at rho=0.40 and 0.60, and the methods are nearly equal at rho=0.50.

The revised summary also states that depth interpolation has lower RMSE than
CLP-CSGM in all nine cross-well configurations at the reference noise
standard deviation 0.01. The larger perturbations are identified as an
uncalibrated sensitivity analysis. We therefore do not claim superiority
over every comparator or general operational advantage.

Where the text was changed: the Abstract, the Conclusion, and the separate
submission Highlights:

CLP-CSGM beats the direct AE in 7 of 9 cross-well cells.
Gains are concentrated in the smaller training sets.
GR-only: CLP lower at 20%-30%; AE lower at 40% and 60%.
At reference noise, interpolation is lower in all nine cells.
The advantage is conditional, not superiority over every method.

---

## R1-M06 -- Why Ridge is the primary reference

**Status:** `closed`

**Reviewer's comment:**

> The MLP prior is better in all nine cross-well cells, whereas Ridge is better in the single-well benchmark. This is not a minor implementation detail; it shows that the prior model materially affects performance. The authors should discuss more clearly why Ridge remains the primary reference despite MLP being consistently better in the cross-well grid.

**Proposed reply:**

We thank the reviewer for noting that the latent prior is not a minor
implementation detail. An audit of the earlier comparison found that it did
not isolate the class of \(h\). In the cross-well runs, the MLP prior also
received gamma ray, while the Ridge prior did not. In the single-well
GR-only runs, the two priors used different depth partitions, with 224 and
275 test windows. Those runs are not a controlled ablation.

The comparison was repeated with the remaining experimental choices held
fixed. In the controlled cross-well grid, both priors use the channels
sonic, rhob, ai, and vp, without gamma ray, and the same windows and sparse
observations. The MLP prior has lower mean RMSE in seven of the nine
configurations. Ridge has lower mean RMSE at step 32 for measurement ratios
0.10 and 0.20. Across the nine configurations, the unweighted mean RMSE
decreases from 0.0286 for Ridge to 0.0254 for MLP, corresponding to 11.3%
based on the unrounded values. The MLP advantage becomes smaller as the
number of training windows decreases and is nearly absent at step 32. In the
controlled F03-4 GR-only comparison, both priors use the same depth-first
partition and the same 224 test windows. Ridge has lower mean RMSE at all
five measurement ratios, with relative reductions of approximately 15.6% to
19.7% compared with MLP.

Ridge remains the primary reference because it is a linear low-capacity map,
so the main tables use one fixed \(h\) across regimes. It is not presented as
the best prior in every setting. The class of \(h\) should be selected on
validation data for the intended regime, not by adopting whichever variant
wins on the held-out test set. The controlled MLP results were not used to
replace the main CLP-CSGM Ridge versus direct autoencoder comparison.

Where the text was changed: the Introduction and Method, subsection Learned
model components (the MLP comparison now states the same channels and the
same split); Results, subsection Latent-prior model sensitivity
(Tables for the cross-well and F03-4 prior-class comparisons, and the
accompanying figure); Interpretation, scope, and limitations; and the
Conclusion sentence on the opposite Ridge-MLP rankings. The main Ridge
versus autoencoder tables were not changed.

---

## R1-M07 -- Proposition 3 does not verify S-REC on the trained decoder

**Status:** `closed`

**Reviewer's comment:**

> Proposition 3 is explicitly conditional and does not prove S-REC for the trained decoder. That distinction is important and should remain very clear in the Abstract/Introduction so readers do not interpret the theory as a formal recovery guarantee for the actual trained model.

**Proposed reply:**

We thank the reviewer for stressing that the conditional local-recovery
proposition is conditional and does not prove S-REC for the trained
decoder. Method, Interpretation, scope, and limitations, and the appendix
diagnostics already stated that limit. The original Abstract did not.

The conditional local-recovery statement referred to as Proposition 3 in the
reviewed version is now Proposition 1 in the revised manuscript, following
the reorganization of the theoretical section.

The Abstract now states that the local recovery bound remains conditional
and does not show that the trained decoder separates nearby windows. The
Introduction states what the bound does and what it does not: under stated
conditions it separates decoder error, prior mismatch, measurement noise, and
local separability, and it does not prove that the measurement operator
satisfies S-REC on the trained decoder. No new experiments or proofs were
added for this comment.

Where the text was changed: the Abstract (one sentence on the conditional
bound); the Introduction (sentence after the three contributions). Method,
subsection Interpretation and theoretical context, and Interpretation,
scope, and limitations already contained this limit and were left in place.

---

## R1-M08 -- Direct comparator with explicit mask/coordinates

**Status:** `closed`

**Reviewer's comment:**

> The direct baselines use fixed sampled coordinates with implicit positional identity, which is reasonable. Still, I would like to see one stronger direct baseline that explicitly encodes the sampling mask/coordinates, especially because the proposed method uses MΩM_\Omega explicitly. This would better isolate whether the gain comes from the latent inverse formulation or simply from better use of coordinate information.

**Proposed reply:**

We added a supplementary direct autoencoder that explicitly receives the
sampling pattern. Its input is [u, b_Omega, mask], where b_Omega places each
noisy measurement at its window coordinate and the binary mask identifies
the observed coordinates. Numeric features are standardized from the
training data, while the mask remains unscaled. The comparison uses the same
nine cross-well cells, seeds, observation draws, and base training seeds as
the paired protocol.

CLP-CSGM Ridge has lower mean RMSE than this mask-aware autoencoder in seven
of nine configurations. It is lower in all six cells at steps 16 and 32 and
in 17 of the 18 corresponding seed comparisons. The mask-aware autoencoder
remains lower at step 8 for rho=0.05 and rho=0.10.

Because the coordinates are fixed within each cell, the mask is constant
across windows and the conventional AE already receives measurements in a
stable positional order. This supplement therefore tests an explicit spatial
encoding, but it does not isolate the latent inverse formulation as the cause
of the difference.

Where the text was changed: Experimental setup, subsection Training,
validation, and test protocol; Results, subsection Cross-well low-data
results; Appendix, subsection Explicit-mask direct AE.

---

## R1-M09 -- GR-only porosity is a controlled test

**Status:** `closed`

**Reviewer's comment:**

> Using GR alone is useful as a stress test, and the authors are transparent about this. However, this should not be presented as evidence for general porosity reconstruction performance. It is a controlled sparse-assimilation experiment under intentionally incomplete context.

**Proposed reply:**

We thank the reviewer for asking that the GR-only porosity results not be
read as general porosity reconstruction performance. Using gamma ray alone
is a controlled sparse-assimilation experiment under intentionally incomplete
context. Acoustic travel time is excluded because it is nearly collinear
with the processed porosity target and would provide a near-direct proxy.
The reported errors, including 0.016 versus 0.018 at rho=0.20, describe that
stress test. The depth profile at rho=0.20 shows that CLP-CSGM follows parts
of the broad trend of the processed porosity curve and smooths local
variations. The displayed window at that ratio shows that both CLP-CSGM and
the direct autoencoder miss the central depression. These figures are not
evidence of general porosity reconstruction with a full operational log suite
or against independent core measurements. The displayed parity correlations,
0.4632 for CLP-CSGM and 0.3720 for the autoencoder, describe the pooled
points on that plot. They are not a significance test. The Abstract does
not report the GR-only RMSE. The Conclusion keeps the same controlled-test
limitation.

Where the text was changed: Results, subsection Single-well F03 GR-only
results (the F03-4 depth-profile paragraph, and two sentences that now report
the observed RMSE advantage of CLP-CSGM Ridge over the direct autoencoder
rather than an isolated causal effect of latent refinement); Appendix
Illustrative sparse-measurement windows; Appendix F03-4 Qualitative
Reconstruction Views; and the Conclusion (sentences after the F03-4 GR-only
RMSE). Experimental setup, subsection Single-well F03-4 GR-only porosity
benchmark, and Interpretation, scope, and limitations already contained the
controlled-test limit and were left in place.

---

## R1-M10 -- Update literature positioning

**Status:** `closed`

**Reviewer's comment:**

> The Introduction and Discussion could be enriched with more recent studies on well-log reconstruction, transferable subsurface learning, and data-driven petrophysical property prediction, particularly to better position the proposed framework against recent advances. https://doi.org/10.1016/j.engeos.2026.100536, 10.1016/j.geoen.2025.214342. https://doi.org/10.1038/s41598-025-95709-0, https://doi.org/10.1109/ACCESS.2023.3336623

**Proposed reply:**

We incorporated the four suggested studies into the Introduction, Related
work, and Discussion. They broaden the positioning across well-log
completion, geomodel translation, cross-basin time-series learning, and direct
petrophysical regression.

The revised text distinguishes these tasks from CLP-CSGM by the role of the
available target observations. CLP-CSGM predicts
\(z_0=h(\boldsymbol{u})\) from the dense logs and retains the sparse target
values in an explicit measurement-fidelity term during latent refinement.
This is a difference in inference formulation, not evidence of predictive
superiority over studies with different targets and protocols.

We also note that the cited foundation-model study evaluates zero-shot
transfer across basins, whereas our evidence is limited to one held-out well
and does not establish basin-agnostic transferability.

Where the text was changed: Introduction; Related work, subsection
Petrophysical log prediction; Interpretation, scope, and limitations.

---

## R1-M11 -- Test against independent measurements

**Status:** `closed`

**Reviewer's comment:**

> The most valuable addition would be even a small test against independent core-derived porosity/permeability or another target not generated by deterministic processing of the same log suite. The current Lapa permeability case helps, but it is described as limited-sample supplementary evidence.

**Proposed reply:**

We added a separate single-well supplement using 87 porosity values identified
as laboratory measurements in the source table associated with the Lapa field
dataset of Alvarez et al. (2025). These values are not computed from the
gamma-ray, resistivity, or lithotype inputs by our processing pipeline.
Original per-plug assay records were not available for independent
verification, so we describe this as a limited laboratory-porosity check
rather than fully verified core validation.

Training and test intervals are separated by contiguous depth blocks before
windowing. RMSE is evaluated only at coordinates not supplied as measurements
to the corresponding reconstruction. However, the length-8 windows overlap:
a physical depth scored in one window may be supplied as a measurement in
another. The experiment therefore evaluates within-window sparse assimilation
on depth-held-out blocks, not globally withheld physical depths.

The direct AE has the lower pooled mean RMSE at both tested ratios: 0.0537
versus 0.0602 at rho=0.25 and 0.0523 versus 0.0580 at rho=0.50. CLP-CSGM
Ridge is lower in 5 of 9 and 6 of 9 block-seed pairs, respectively, and is
lower than the decoded conditional prediction \(G(\boldsymbol{z}_0)\) in
every pair. The block-level pattern is compatible with sensitivity to prior
mismatch and changes in the auxiliary-log distribution, but the experiment
does not isolate either as a cause.

This supplement is one cored interval and does not establish operational or
cross-well generalization. It also does not extend the mean advantage observed
in the low-support F3 cells to this laboratory target.

Where the text was changed: Experimental setup; Results; Interpretation,
scope, and limitations; Conclusion; Appendix, subsection Laboratory porosity
on a cored interval.

---

## R1-m01 -- Shorten the title

**Status:** `closed`

**Reviewer's comment:**

> The title is accurate but quite long. It could be shortened slightly while retaining “Conditional Latent-Prior” and “Petrophysical Target-Log Reconstruction.”

**Proposed reply:**

We thank the reviewer for the suggestion. We shortened the title slightly by
replacing "with Application to" with "for". The title is now "Conditional
Latent-Prior Generative Compressed Sensing for Petrophysical Target-Log
Reconstruction". It keeps "Conditional Latent-Prior" and "Petrophysical
Target-Log Reconstruction", and it keeps "Generative Compressed Sensing",
which names the method.

Where the text was changed: the title.

---

## R1-m02 -- Reduce overlap between Introduction and Related Work

**Status:** `closed`

**Reviewer's comment:**

> The Introduction and Related Work are strong, but some paragraphs could be tightened to reduce overlap between direct prediction, conditional inverse problems, and measurement-consistency framing.

**Proposed reply:**

We thank the reviewer for asking us to reduce overlap between the
Introduction and Related Work. We made a minimal editorial pass rather than a
structural reorganization.

The Introduction now concentrates on motivation, the measurement role of the
sparse observations, the conceptual definition of CLP-CSGM, and the
contributions. Recent log-completion, geomodel-translation, and
direct-regression studies are named only briefly, with the detailed
comparison deferred to Related Work. Related Work concentrates on positioning:
it retains the four recent references, distinguishes completion, translation,
and direct regression from CLP-CSGM by how observations enter inference, and
shortens repeated descriptions of trajectory initialization and restart
details that remain in the Method section. The mathematical formulation,
equations, and experimental results were left unchanged.

Where the text was changed: Introduction (pointer to Related Work for the
recent-study comparison); Related work, subsection Petrophysical log
prediction (comparative CLP-CSGM paragraph); and Related work, subsection
Conditional generative inverse problems and positioning (closing paragraph).

---

## R1-m03 -- Figure 2 readability

**Status:** `closed`

**Reviewer's comment:**

> Figure 2 is informative and technically useful, but the text density is high. Some labels could be enlarged for print readability.

**Proposed reply:**

We redrew Figure 2 at full line width with larger labels and less text inside
the boxes. Offline training and target-well inference are now separated into
two panels. The inference panel identifies the latent anchor, the coordinate
restriction \(M_{\Omega}\), the original-scale decoder
\(G_{\mathrm{orig}}\), and the measurement residual. The revised figure was
checked at page size; the labels and refinement equation are readable and the
boxes do not overlap.

Where the text was changed: Method, subsection Test-time reconstruction
(Figure 2).

## R1-m04 -- AE as best comparator only in this benchmark

**Status:** `closed`

**Reviewer's comment:**

> The statement that the direct AE is the “most competitive direct baseline” is supported within this benchmark, but should always be read as benchmark-specific rather than generally strongest.

**Proposed reply:**

We thank the reviewer for this clarification. Among the direct baselines on
the published cross-well grid, MLP, PCA, and AE [u,b], the autoencoder has
the lowest unweighted mean RMSE. The seven-of-nine comparison is with that
autoencoder. The Conclusion no longer calls it the most competitive baseline
without naming that set, and it states that the result is not a ranking
against interpolation or the direct linear maps. Results still says that AE
has the lowest unweighted mean RMSE among the direct baselines and is the
most competitive direct comparator in this grid, meaning that direct set.
The Abstract states the seven-of-nine comparison as a conditional
advantage over that autoencoder, not superiority over every comparator,
and does not name the MLP, PCA, and autoencoder set. That named set
remains in the Conclusion.

Where the text was changed: the Abstract (the conditional-advantage
paragraph) and the Conclusion (the sentence that names MLP, PCA, and the
autoencoder).

---

## R1-m05 -- Frozen code version

**Status:** `pending`

**Reviewer's comment:**

> Since the code is public, this is a major strength. I suggest the authors also provide a frozen release/tag corresponding exactly to the manuscript version rather than only a live GitHub repository.

**Proposed reply:**

**Deferred.** Do not create the tag during the revision. The manuscript and
the code are still changing: this round added tests (step-32 development
splits, the seed panel, the explicit-mask AE, and the well 861 laboratory
porosity check). A frozen release must match the resubmitted manuscript, so
the tag is created at the end. Computer Code Availability then cites that tag
instead of only the live repository URL.

**FINAL-01 (2026-11-03):** Not closed. The public repository
`romulobrito/CLP--CSGM` at `bba1423eac98d1f93375905e9d7e974fcd3a8261`
(2026-08-11) has no tag and no release. Its main launcher matches the
manuscript protocol when the documented defaults are used, but it does not
contain the supplementary runners that produced the revision tables. A tag
on that commit would not freeze the code behind the resubmitted manuscript.
No tag was created in this audit, and `main.tex` was not edited. The
proposed identifier is `cageo-d-26-01398`, to be created later on a public
commit that includes those runners and excludes private data.
