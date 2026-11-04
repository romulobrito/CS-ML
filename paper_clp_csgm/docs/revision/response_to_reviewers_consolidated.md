# Response to the Editor and Reviewers

**Manuscript ID:** CAGEO-D-26-01398

**Title:** Conditional Latent-Prior Generative Compressed Sensing for Petrophysical Target-Log Reconstruction

**Journal:** Computers & Geosciences

## Opening statement

Dear Associate Editor and Reviewers,

We thank you for the careful and constructive evaluation of our manuscript. The revision clarifies the scope of the contribution, expands the conventional comparators and robustness checks, improves the presentation of the formulation and principal results, states the limits of the empirical validation more prominently, and documents the use of AI-assisted tools. The scientific claims remain conditional on the evaluated benchmark regimes, and the revised manuscript does not claim general operational superiority.

## Response to the Associate Editor

### AE-01 -- Associate Editor synthesis

**Comment**

> The reviewers agree that the manuscript addresses an important problem in petrophysical log reconstruction and presents a technically interesting latent-space framework supported by open code and thorough ablation studies.
>
> However, an important concern of the reviewers is that the empirical validation remains too limited to support broad claims. The primary targets are derived quantities (e.g., clay volume and porosity), and the sparse observations are obtained through artificial subsampling, meaning that the experiments demonstrate reconstruction under controlled benchmark conditions rather than validation against independent measurements. The reviewers therefore recommend moderating claims regarding practical applicability and transferability. Still regarding applicability, one reviewer underlines that prediction of clay volume and porosity is a simple linear problem and a clear evidence over why the proposed ML workflow is superior to traditional linear methods is therefore needed. Additional concerns include the limited number of wells, the small number of random realizations, and the sensitivity of the results to observation placement. Lastly, maybe because of the use of Ai to edit the text, the manuscript is sometimes hard to understand, with the stacking of terminologies such as, for example, "low-capacity latent-prior reference", "admissible reconstruction range",etc. The paper readability should therefore be checked carefully (and if AI was used to improve the text then it should be acknoweldged according to the Elsevier policy).
>
> Overall, the reviewers find the work interesting but believe that additional validation, stronger benchmarking, and more cautious interpretation of the results are needed to support the conclusions.

**Response**

We thank the Associate Editor for identifying the main issues that required
revision. We have narrowed the empirical claims, strengthened the benchmark
comparisons and robustness checks, clarified the method and theory, and
expanded the disclosure of AI-assisted preparation.

The revised manuscript now states prominently that the main F3 targets are
processed quantities: clay volume is Larionov-derived from gamma ray, F03-4
porosity is density-derived, and the sparse observations are subsamples of
those curves. The main cross-well experiment deliberately withholds gamma ray
and is presented as a controlled incomplete-information test, not as a
replacement for Larionov or validation against independent clay-volume
measurements. A separate one-interval laboratory-porosity supplement is also
reported. The direct autoencoder has the lower pooled mean RMSE in that
supplement, and the provenance and overlapping-window limitations are stated
explicitly.

The comparison now includes direct linear and ridge regression, depth
interpolation, validation-selected second-difference smoothing, and its affine
limit, in addition to the direct autoencoder and structural ablations.
CLP-CSGM Ridge is lower than the direct autoencoder in seven of nine
cross-well cells and higher in two. At the reference noise standard deviation
0.01, interpolation is lower in all nine cells. We therefore describe the
result as regime-dependent and do not claim general superiority over simpler
or conventional methods.

The robustness evidence was also widened. A ten-seed panel and five repeated
measurement draws support the low-training-support ranking within the same
benchmark. They do not add independent wells, and the placement analysis
changes coordinates and measurement noise together. The cross-well evidence
therefore remains limited to three development wells and one held-out well
and does not establish multi-field transferability.

The presentation now explains CLP-CSGM as the combination of a learned
decoder representation, a latent code predicted from dense logs, and
test-time refinement using sparse target observations. Classical MAP and
existence results are distinguished from the conditional local-recovery
statement, which is not presented as a verified S-REC guarantee for the
trained decoder. Figures and terminology were revised accordingly.

Finally, the generative-AI declaration identifies OpenAI's ChatGPT for
translation, language and organization, length reduction, and selected
explanatory schematics identified in their captions. It identifies xAI's
Grok 4.7 as assistance in refactoring existing scientific code, not in
originating the methodology or running the experiments. The authors reviewed
all assisted material and remain responsible for the manuscript, code,
results, and conclusions.

## Response to Reviewer 1

### R1-M01 -- Origin of targets and scope of validation

**Comment**

> The cross-well target VcV_c is Larionov-derived from gamma ray, and the F03-4 porosity target is itself processed/density-derived. The authors acknowledge this limitation, but it is central and should be emphasized earlier, including in the Abstract and Conclusions.

**Response**

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

### R1-M02 -- Few wells and operational transfer

**Comment**

> Only three wells are used for development and one well is held out for testing. That is useful as a controlled test, but it is not enough to support broad statements about cross-well robustness or operational transferability. The language should remain clearly benchmark-specific.

**Response**

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

### R1-M03 -- Few windows and resampling

**Comment**

> At the largest window step, the development pool contains only about 27 windows before validation splitting. The performance gains in those low-data settings are interesting, but with such small sample support, repeated resampling or additional held-out wells would make the conclusions much more convincing.

**Response**

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

### R1-M04 -- Three seeds and uncertainty intervals

**Comment**

> The manuscript correctly uses Student-tt intervals and explicitly notes the limitation, but n=3n=3 is still very small. I suggest increasing the seed count if computationally feasible, especially because several intervals include zero and the ranking changes by regime.

**Response**

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

### R1-M05 -- Conditional advantage, not overall superiority

**Comment**

> CLP-CSGM outperforms the direct AE baseline in seven of nine cross-well cells, but loses in two cells; in the GR-only benchmark, AE becomes better at higher measurement ratios. This should be framed as a conditional advantage rather than overall superiority. The paper generally does this well, but the Abstract/Highlights could be even more explicit.

**Response**

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

### R1-M06 -- Why Ridge is the primary reference

**Comment**

> The MLP prior is better in all nine cross-well cells, whereas Ridge is better in the single-well benchmark. This is not a minor implementation detail; it shows that the prior model materially affects performance. The authors should discuss more clearly why Ridge remains the primary reference despite MLP being consistently better in the cross-well grid.

**Response**

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

### R1-M07 -- Proposition 3 does not verify S-REC on the trained decoder

**Comment**

> Proposition 3 is explicitly conditional and does not prove S-REC for the trained decoder. That distinction is important and should remain very clear in the Abstract/Introduction so readers do not interpret the theory as a formal recovery guarantee for the actual trained model.

**Response**

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

### R1-M08 -- Direct comparator with explicit mask/coordinates

**Comment**

> The direct baselines use fixed sampled coordinates with implicit positional identity, which is reasonable. Still, I would like to see one stronger direct baseline that explicitly encodes the sampling mask/coordinates, especially because the proposed method uses MΩM_\Omega explicitly. This would better isolate whether the gain comes from the latent inverse formulation or simply from better use of coordinate information.

**Response**

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

### R1-M09 -- GR-only porosity is a controlled test

**Comment**

> Using GR alone is useful as a stress test, and the authors are transparent about this. However, this should not be presented as evidence for general porosity reconstruction performance. It is a controlled sparse-assimilation experiment under intentionally incomplete context.

**Response**

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

### R1-M10 -- Update literature positioning

**Comment**

> The Introduction and Discussion could be enriched with more recent studies on well-log reconstruction, transferable subsurface learning, and data-driven petrophysical property prediction, particularly to better position the proposed framework against recent advances. https://doi.org/10.1016/j.engeos.2026.100536, 10.1016/j.geoen.2025.214342. https://doi.org/10.1038/s41598-025-95709-0, https://doi.org/10.1109/ACCESS.2023.3336623

**Response**

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

### R1-M11 -- Test against independent measurements

**Comment**

> The most valuable addition would be even a small test against independent core-derived porosity/permeability or another target not generated by deterministic processing of the same log suite. The current Lapa permeability case helps, but it is described as limited-sample supplementary evidence.

**Response**

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
scope, and limitations; Conclusion; Appendix, section Laboratory porosity
on a cored interval.

### R1-m01 -- Shorten the title

**Comment**

> The title is accurate but quite long. It could be shortened slightly while retaining “Conditional Latent-Prior” and “Petrophysical Target-Log Reconstruction.”

**Response**

We thank the reviewer for the suggestion. We shortened the title slightly by
replacing "with Application to" with "for". The title is now "Conditional
Latent-Prior Generative Compressed Sensing for Petrophysical Target-Log
Reconstruction". It keeps "Conditional Latent-Prior" and "Petrophysical
Target-Log Reconstruction", and it keeps "Generative Compressed Sensing",
which names the method.

Where the text was changed: the title.

### R1-m02 -- Reduce overlap between Introduction and Related Work

**Comment**

> The Introduction and Related Work are strong, but some paragraphs could be tightened to reduce overlap between direct prediction, conditional inverse problems, and measurement-consistency framing.

**Response**

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

### R1-m03 -- Figure 2 readability

**Comment**

> Figure 2 is informative and technically useful, but the text density is high. Some labels could be enlarged for print readability.

**Response**

We redrew Figure 2 at full line width with larger labels and less text inside
the boxes. Offline training and target-well inference are now separated into
two panels. The inference panel identifies the latent anchor, the coordinate
restriction \(M_{\Omega}\), the original-scale decoder
\(G_{\mathrm{orig}}\), and the measurement residual. The revised figure was
checked at page size; the labels and refinement equation are readable and the
boxes do not overlap.

Where the text was changed: Method, subsection Test-time reconstruction
(Figure 2).

### R1-m04 -- AE as best comparator only in this benchmark

**Comment**

> The statement that the direct AE is the “most competitive direct baseline” is supported within this benchmark, but should always be read as benchmark-specific rather than generally strongest.

**Response**

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

### R1-m05 -- Frozen code version

**Comment**

> Since the code is public, this is a major strength. I suggest the authors also provide a frozen release/tag corresponding exactly to the manuscript version rather than only a live GitHub repository.

**Response**

**AUTHOR ACTION PENDING -- Frozen code release to be completed before submission.**

This is an internal editorial marker, not a response to the reviewer. The consolidated letter is not ready for submission while this marker remains.

## Response to Reviewer 2

### R2-01 -- Motivation before the mathematical formulation

**Comment**

> The manuscript structure should be improved. The Introduction moves quickly into the mathematical formulation of CLP-CSGM, making it read more like part of the Method section. The practical motivation, research gap, and necessity of the proposed approach are not sufficiently developed before the formulation is introduced. The mathematical details should be moved to the methodology section, while the Introduction should more clearly establish the problem and contribution.

**Response**

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

### R2-02 -- Processed targets and artificial subsampling

**Comment**

> The practical validity of the benchmark targets requires further discussion. The main targets are processed or derived properties rather than independently measured ground truth: V_c is derived from gamma ray using the Larionov relation, and porosity is density-derived. Moreover, the sparse observations are generated by subsampling these same target curves. Therefore, the experiments primarily demonstrate reconstruction of artificially subsampled processed curves rather than recovery against independent petrophysical measurements. This limitation should be more clearly reflected in the interpretation and claims.

**Response**

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

### R2-03 -- Realism of excluding GR and sonic

**Comment**

> The benchmark design appears somewhat artificial from a practical petrophysical perspective. Gamma ray is excluded from the V_c experiment because V_c is directly derived from GR, while acoustic travel time is excluded from the porosity experiment because of its near-perfect correlation with porosity. Although these choices create controlled stress tests, the authors should better justify what realistic field-data scenarios these configurations represent and clarify the practical applicability of the resulting conclusions.

**Response**

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

### R2-04 -- Sensitivity to observation placement

**Comment**

> The robustness to sparse-observation patterns should be examined more carefully. A single coordinate set is sampled for each seed–measurement-ratio combination and then kept fixed across all windows. The conclusions may therefore depend on the particular locations of the sparse observations. Additional sampling realizations or an analysis of sensitivity to observation placement would strengthen the evaluation.

**Response**

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

### R2-05 -- Statistical robustness with more realizations

**Comment**

> The statistical robustness of the comparison is limited by the use of only three seeds. Given that the main conclusions rely on relatively small performance differences in some experimental settings, three realizations provide limited evidence regarding robustness. Increasing the number of realizations, particularly for random sparse-observation sampling, would provide more convincing support for the reported performance differences.

**Response**

We added two complementary robustness checks while retaining the original
three-seed tables as the main result. First, a ten-seed panel refits both
methods and redraws coordinates and measurement noise. None of the nine
paired 95% Student-t intervals includes zero. These are per-configuration
intervals and were not adjusted for simultaneous multiple comparisons. The
regime-dependent ranking is unchanged: CLP-CSGM Ridge is lower at steps 16
and 32, whereas the direct autoencoder is lower at step 8 for
\(\rho=0.05\) and \(\rho=0.10\).

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

## Response to Reviewer 3

### R3-01 -- Clarity and stacked terminology

**Comment**

> The clarity of writing. The paper is hard to understand with layers of terminology stacking together. Let me take the abstract as an example: “A decoder learned from target-property windows defines the admissible reconstruction range”: what is admissible reconstruction range? Another example is “With Ridge as a low-capacity latent-prior reference”, it is very difficult to understand “low-capacity latent-prior reference”. “Ablations against prior-only and measurement-only CSGM variants indicate that contextual latent anchoring and sparse-measurement consistency make complementary contributions whose relative importance varies across regimes.” If you read it through, it is very difficult to understand things like “contextual latent anchoring”, “sparse- measurement consistency”, etc.

**Response**

We revised the manuscript to replace stacked labels with a direct description
of the three operations. A decoder trained on target windows defines the
curves the model can produce. Dense auxiliary logs predict a latent code
\(\boldsymbol{z}_0\). Test-time refinement then balances disagreement with
the sparse observations against deviation from that code. The first
trajectory starts at \(\boldsymbol{z}_0\); additional restarts are perturbed
around it, while the penalty remains centered at
\(\boldsymbol{z}_0\). The observations are penalized rather than enforced
exactly.

Ridge is now introduced explicitly as ridge regression for the conditional
map from dense logs to \(\boldsymbol{z}_0\). The decoder range, the
conditional code, and the measurement penalty are described as distinct
components. The prior-only and measurement-only variants are likewise named
by the operation they remove.

These revisions improve the Abstract and the corresponding explanations
throughout the manuscript without changing the algorithm, equations, results,
or scope limitations.

Where the text was changed: Abstract; Introduction; Related work; Method;
Experimental setup; Results; Interpretation, scope, and limitations;
Conclusion; Appendix.

### R3-02 -- Practical advantage versus conventional relations

**Comment**

> The method is very interesting; however, what worries me is how practical this method is: The derivation of the target is very simple: V_clay is just linearly derived from GR with endpoints; whereas porosity is almost a linear function of RHOB or acoustic slowness. If you are trying to predict those logs, why use a complicated ML workflow instead of just a simple linear relationship? I think the authors should put more effort on convincing people why we need to choose ML over traditional methods, other than use traditional linear results as benchmark and try to replicate them.

**Response**

We thank the reviewer for this practical question. When the log required by
the conventional formula is available, that formula is the appropriate
calculation, and CLP-CSGM is not a substitute for it.

For clay volume, the gamma-ray index between the clean and shale endpoints
is linear in gamma ray. The Larionov map from that index to clay volume is
not a straight line. If gamma ray has been logged, the clay volume is
obtained from that relation directly. In the cross-well test, gamma ray is
left out because the target curve was computed from it. Without that log the
Larionov relation cannot be evaluated. The task is to reconstruct the
processed curve from sonic, density, acoustic impedance, and compressional
velocity while matching a few known values of the curve.

For the F03-4 porosity, density-derived porosity can be calculated from bulk
density and the hypotheses of that relation. A sonic-porosity estimate uses
another petrophysical relation and is not necessarily identical to the
density-derived porosity. If the bulk-density log required by the
density-porosity relation has been recorded, that relation is the appropriate
calculation. In this dataset, the density behind the processed target was
predicted from sonic and gamma ray. Sonic travel time follows that processed
curve almost one to one. That correlation is not validation against
independent laboratory porosity. Sonic is left out, and gamma ray is the only
dense log. The errors do not transfer to a porosity interpretation that also
uses density, neutron, and sonic.

The two F3 benchmarks therefore compare methods that face this incomplete
input. They are not a contest against Larionov, density porosity, or sonic
porosity on a fully logged well.

The purpose of CLP-CSGM is not to replace conventional petrophysical formulas
when their required logs are available. It is to investigate whether a learned
representation can assist reconstruction when the defining log is unavailable
and sparse target-property values must be assimilated. However, methodological
motivation does not establish practical superiority. In the supplementary
comparison, simple depth interpolation achieves lower RMSE than CLP-CSGM in
all nine cross-well configurations at the reference noise level. CLP-CSGM
becomes more competitive under larger simulated measurement perturbations, but
those perturbations were not calibrated to field uncertainties. Thus, the
present experiments do not establish that CLP-CSGM should be preferred
operationally over conventional calculations or simpler reconstruction
methods.

Where the text was changed: Data and exploratory analysis (the clay-volume
input-design paragraph); Experimental setup (the F03-4 GR-only porosity
paragraph); Interpretation, scope, and limitations (the empirical-scope
paragraphs for the two F3 benchmarks, including the distinction between
density-derived and sonic porosity and the interpolation sentences).

### R3-03 -- Simple comparators and intermediate results

**Comment**

> The intermediate results and baselines The paper listed several baselines, but the comparison is incomplete. Direct linear or ridge regression, with the same input and evaluation should be included. Methods such as smoothing, interpolation based on the sparse target (core data) should also be compared. Moreover, the authors did not report the intermediate results from the multi-step ML workflow, such as how the encoder reconstructed the original well logs? And why do we need an encoder here, considering there are only 64 values included in the window.

**Response**

We added ordinary least squares and direct ridge regression from
\([\boldsymbol{u},\boldsymbol{b}]\), linear interpolation of the sparse
target observations, and a validation-selected second-difference smoother.
The affine least-squares limit of the smoother is also reported. All use the
same cross-well windows, seeds, coordinates, and full-window RMSE as the main
comparison.

At the reference noise standard deviation 0.01, interpolation has lower RMSE
than CLP-CSGM in all nine cells, the affine fit is lower in all nine, and the
selected smoother is lower in eight. CLP-CSGM is lower than both direct
linear maps only at step 16 with rho=0.20 and at step 32 for all three
ratios. Under the larger simulated perturbations, CLP-CSGM becomes lower than
interpolation in 7, 9, and 9 cells at noise 0.05, 0.10, and 0.20, and lower
than the selected smoother in 5, 8, and 9 cells. These larger levels were
used only for sensitivity analysis and were not calibrated to field
uncertainty.

The encoder is used during training to obtain the latent codes on which the
dense-log map is fitted; the unknown target window is not encoded at
inference. The model uses a 16-dimensional code for a length-64 window.
The retrospective reconstruction \(G(E(\boldsymbol{y}))\) is reported only
as a representation diagnostic, with mean test RMSE 0.0073, 0.0099, and
0.0110 at steps 8, 16, and 32. Because it uses the complete target window,
it is not an inference result.

The noise sweep reselects the CLP-CSGM penalty on validation at each noise
level. It does not isolate the decoder restriction, the conditional penalty,
or parameter selection as the cause of the observed robustness, and it does
not establish general superiority.

Where the text was changed: Abstract; Method; Experimental setup; Results;
Conclusion; Appendix, subsection Direct linear maps, depth interpolation,
and measurement noise.

### R3-04 -- Noise choice and prior/data balance

**Comment**

> The noise How does the authors choose the noise, thus balancing the influence from the prior / the data fit?

**Response**

The benchmark adds independent Gaussian noise after coordinate subsampling,
with standard deviation 0.01 for cross-well clay volume and 0.02 for F03-4
porosity, in the respective target units. The value 0.01 is about 0.29 times
the standard deviation of the held-out clay-volume curve. These levels were
adopted for controlled benchmarks and were not calibrated to instrumental
uncertainty. The `30dB` filename designation is separate from this added
noise and is not interpreted as a verified SNR.

We now make the rationale and its limit explicit. Gaussian contamination has
precedent as a synthetic well-log inversion stress test (Abdelrahman and
Szabo, 2024), but this does not calibrate the amplitudes used here. Because
the cross-well target is gamma-ray-derived, an instrument-aware analysis
would need to estimate gamma-ray uncertainty and propagate it through the
nonlinear clay-volume relation. Adjei et al. (2019) illustrate probabilistic
sampling for uncertainty in gamma-ray-based shale-volume estimation. We did
not perform that propagation and therefore retain the synthetic,
uncalibrated interpretation.

The prior-data balance is controlled by \(\lambda\), not by changing the
noise during a reported run. For each seed and measurement ratio, validation
full-window RMSE selects \(\lambda\) from
\(\{10^{-4},3\mathbin{\cdot}10^{-4},10^{-3},3\mathbin{\cdot}10^{-3},
10^{-2},3\mathbin{\cdot}10^{-2},10^{-1}\}\), and that choice is fixed before
test evaluation. The experiments do not estimate \(\lambda\) from a noise
variance. The MAP identity
\(\lambda=\sigma_\eta^2/\sigma_z^2\) applies only under the stated isotropic
Gaussian assumptions and to the residual in the original target units.

We also added a sensitivity sweep at noise standard deviations 0.01, 0.05,
0.10, and 0.20, with \(\lambda\) reselected on validation from a wider grid
at each level. Interpolation is lower in all nine cells at 0.01; CLP-CSGM is
lower in 7, 9, and 9 cells at the three larger levels. Against the
second-difference smoother, the corresponding counts are 5, 8, and 9. These
results describe relative robustness of the complete procedure under
simulated perturbations. They do not isolate a mechanism or establish
performance under calibrated field noise.

Where the text was changed: Method, subsections Test-time reconstruction and
Interpretation and theoretical context; Experimental setup; Results;
Discussion; Conclusion; Appendix, subsections MAP and Tikhonov interpretation
and Direct linear maps, depth interpolation, and measurement noise; the
implementation-summary table.

### R3-05 -- Role of the theory section

**Comment**

> The theoretical section Section 3.4 provides mathematical derivations related to the MAP and Tikhonov regularization. It is unclear how this would impact the application of this method, especially considering that MAP / Tikhonov interpretation and existence of a minimizer are standard arguments.

**Response**

We thank the reviewer for noting that the MAP/Tikhonov reading and the
existence of a minimizer are standard arguments, and for asking how they
bear on the application.

Both results are classical. They are not presented as new theoretical
contributions. Under the stated Gaussian hypotheses, the MAP reading gives
the data term coefficient one and sets
\(\lambda=\sigma_{\eta}^{2}/\sigma_{z}^{2}\) as the multiplier of a
quadratic penalty centered at \(\boldsymbol{z}_0\). The experiments still
select \(\lambda\) by validation and do not estimate that ratio. The
reading explains the form of the penalty. It does not demonstrate
empirical superiority. The existence result says that a continuous decoder
and \(\lambda>0\) make the objective continuous and coercive, so a
minimizer exists. It does not give uniqueness, convexity, or convergence
of Adam to a global minimizer, and it does not extend automatically to
\(\lambda=0\). Lowering the objective does not by itself lower the error
against the unknown target. The comparison between the numerical solution
and \(\boldsymbol{z}_0\) uses the objective values that were obtained. The
formal statements now stand with their proofs in the appendix. The body
states what each result is for and what it does not guarantee. The
conditional local-recovery proposition remains in the body. It organizes
representation error, observed discrepancy, deviation from
\(\boldsymbol{z}_0\), and local separability. It remains conditional: it
does not prove that the coordinate mask satisfies S-REC on the trained
decoder, and the empirical diagnostics do not certify that condition for
every pair of windows. No new theoretical guarantee and no new experiment
were added.

Where the text was changed: Method, subsection Interpretation and
theoretical context; Appendix MAP and Tikhonov interpretation; Appendix
Existence of a minimizer.

### R3-06 -- Main result and figure readability

**Comment**

> Readability of the results The manuscript reports many metrics, experimental settings, ablations, and sensitivity comparisons, but the principal finding is difficult to extract. The results should be organized around a few clear questions: when does refinement help, how much does it improve over simple alternatives, and when does the additional complexity cease to be worthwhile? The figures also need improvement. In Figure 6, the large errors of some baselines compress the differences among the better-performing methods near the bottom of the plots. A separate panel or enlarged view would make the relevant comparison easier to assess. Figure 7 has small labels and differing horizontal scales that complicate visual comparison.

**Response**

We reorganized the opening of the cross-well results around three questions:
when latent refinement helps relative to the direct autoencoder, how it
compares with simple reconstructions, and when the extra optimization stops
being worthwhile.

CLP-CSGM Ridge has lower mean RMSE than direct AE in seven of the nine
cross-well cells. At steps 16 and 32 that ordering holds in the ten-seed
panel and in every sparse-measurement redraw. At the reference noise 0.01,
depth interpolation is lower in all nine cells, the affine fit is lower in
all nine, and the validation-selected smoother is lower in eight of nine.
The noise-sensitivity analysis also shows that the relative ranking changes
under larger simulated perturbations, which were not calibrated to field
uncertainty. The extra cost is not presented as justified in general.

The five-method RMSE figure now has two rows and one vertical scale within
each row. The upper row keeps ML only, MLP, PCA, AE, and CLP-CSGM Ridge,
including the large MLP and PCA errors. The lower row enlarges only ML only,
AE, and CLP-CSGM Ridge. Error bars remain Student-t 95 percent half-widths
with three seeds. The caption states that this figure does not include
depth interpolation or second-difference smoothing, so the enlargement is
not a ranking of the full comparator set. In the lower row, AE is lower
than CLP-CSGM at step 8 for the two sparsest ratios.

The depth-profile figure now compares the processed reference, ML only, AE,
and CLP-CSGM Ridge on one shared clay-volume scale, with depth increasing
downward. The interval is still step 16, seed 7, and rho 0.20. No value of
those four curves was dropped. MLP and PCA remain in a separate appendix
figure on the full range of their stored predictions, including negative
clay volume. The results text points to that figure.

Results and Discussion both cite the runtime table. On the cross-well grid
the mean time per test window is 0.23517 s for CLP-CSGM Ridge and 0.03156 s
for AE. The normalized runtime diagnostic versus AE is 9.79. The text states
that this diagnostic is not the quotient of those two averages.

No new experiment was run. Seeds, splits, hyperparameters, and the published
cell means were not changed.

Where the text was changed: Results, cross-well subsection; Discussion,
operational paragraph; Appendix, new wide-range depth-profile figure.

### R3-A01 -- Define symbols at first use

**Comment**

> Define symbols at first use, including those introduced in the abstract and introductory equations.

**Response**

We reviewed symbols and abbreviations at their first occurrence. The Abstract
now defines root mean square error (RMSE) and identifies \(\rho\) as the
nominal observed fraction before reporting its values. It describes the
gamma-ray-only case in words and states the recovery limitation without
introducing S-REC there.

In the main text, MLP, MAP, S-REC, GAN, ANN, SHAP, MAE, AE, PCA, \(V_c\),
and GR-only are expanded at first use. The Method section defines the target
and input windows, the coordinate set, \(M_{\Omega}\), the noise model, the
conditional map, the decoder, and the regularization parameters before they
are used in the reconstruction equations. The nominal ratio \(\rho\) remains
distinct from the realized fraction \(m/N\).

Where the text was changed: Abstract; Introduction; Related work; Method;
Data and exploratory analysis; Experimental setup. No equation or result was
changed.

### R3-A02 -- Separate sampling operator and output set

**Comment**

> Clearly distinguish the sampling operator MΩ from the decoder range M.

**Response**

We now reserve \(M_{\Omega}\) exclusively for the coordinate-restriction
operator that selects the observed depths. The decoder output set is no
longer written \(\mathcal{M}\) or \(\operatorname{Im}(G)\). We use
\[
\mathcal{R}_{G}=\{G(\boldsymbol{z}):\boldsymbol{z}\in\mathbb{R}^{k}\}
\]
for the standardized decoder range and
\(\mathcal{R}_{\mathrm{orig}}\) for the same outputs in the original target
units. The local set used in the recovery statement is denoted
\(\mathcal{R}_{\mathrm{orig},R^{\dagger}}\).

The reconstruction is explicitly stated to belong to
\(\mathcal{R}_{\mathrm{orig}}\), while \(M_{\Omega}\) acts only on a window
to extract its observed coordinates. Figure 1 and its caption use the same
distinction.

Where the text was changed: Related work, subsection Classical and generative
compressed sensing; Method, subsection Learned model components (the range
definition and Figure 1); Method, subsection Interpretation and theoretical
context (the representation-error remark and the conditional local-recovery
proposition); Interpretation, scope, and limitations; Appendix Conditional
local recovery interpretation.

### R3-A03 -- Redraw Figure 1

**Comment**

> Redraw Figure 1 so that the mappings from z_0 to G(z_0) and from the refined code to its decoded reconstruction are unambiguous. Explain the arrows and the curve representing the decoder range.

**Response**

We thank the reviewer for asking that the two decoder mappings and the range
curve be unambiguous.

Figure 1 draws both mappings. One horizontal arrow applies
\(G_{\mathrm{orig}}\) to \(\boldsymbol{z}_0\) and ends at
\(G_{\mathrm{orig}}(\boldsymbol{z}_0)\). The other applies
\(G_{\mathrm{orig}}\) to the refined code and ends at
\(\widehat{\boldsymbol{y}}=G_{\mathrm{orig}}(\widehat{\boldsymbol{z}})\).
Both outputs lie on the blue curve. That curve is
\(\mathcal{R}_{\mathrm{orig}}\), the set of decoder outputs in the
original target units. The vertical arrow is latent refinement from
\(\boldsymbol{z}_0\) to \(\widehat{\boldsymbol{z}}\). The drawing is
schematic and does not show a measured numerical trajectory.

Where the text was changed: Method, subsection Learned model components
(Figure 1).

### R3-A04 -- Probability density in Figure 3

**Comment**

> Use “probability density” in Figure 3 to avoid confusion with the bulk-density log.

**Response**

We thank the reviewer for pointing out the ambiguity. We have changed the
vertical-axis label of Figure 3 from "density" to "Probability density",
clarifying that the figure represents statistical distributions of clay
volume rather than the petrophysical bulk-density log. The underlying data
and histogram calculations remain unchanged.

Where the figure was changed: Data and exploratory analysis, Figure 3
(vertical-axis label only). The caption was already correct and was not
changed.

### R3-A05 -- State that Ridge is ridge regression

**Comment**

> Introduce Ridge explicitly as ridge regression and state its role in the workflow.

**Response**

We thank the reviewer for requesting clarification of the Ridge component. We
now explicitly identify it as ridge regression and clarify its role as the
conditional latent-prior map. During training, ridge regression maps dense
auxiliary logs to latent codes obtained from the target-window encoder. At
inference, the predicted code provides the initialization and regularization
center for latent refinement. Ridge regression does not directly reconstruct
the target curve in this configuration; the decoder produces the
reconstruction after latent refinement. We also distinguish this component
from the direct ridge-regression baseline, which predicts target windows from
the auxiliary logs and sparse observations.

Where the text was changed: Introduction, the first mention of Ridge;
Method, subsection Learned model components, paragraph Log-conditioned latent
prior. The training equation was not changed.

### R3-P01 -- Unclear objective in the Abstract

**Comment**

> here i think it is hard to understand the goal. consider rephrase it. you have a contiuous well log and a few sparse core data, the goal is to have a continuous prediction over that property, like permeability

**Response**

We thank the reviewer for pointing out that the reconstruction goal was not
sufficiently explicit. We have revised the opening of the Abstract to state
directly that the task is to reconstruct a continuous target-property profile
along a well from dense auxiliary logs and sparse observations of the target
property. We also clarify that the sparse observations in the main benchmarks
are coordinate subsamples of processed target curves, rather than
independently acquired core measurements. The revised wording makes the
reconstruction objective explicit without changing the experimental scope or
implying operational validation.

Where the text was changed: Abstract, opening sentence. The Introduction
already states the same task and was not edited.

### R3-P02 -- Framing as an inverse problem

**Comment**

> the probelm doesn’t sound like an inverse problem, or it i s very weakly linked to inverse problem. the position operator M just mask a few points, from the prediciton to get the actual sparse observation. it is better to call it missing data reconstruciton.

**Response**

We thank the reviewer for raising this important distinction. We agree that
the application is naturally described as missing-data reconstruction.
However, missing-data reconstruction and inverse-problem formulations are not
mutually exclusive.

Classical compressed sensing reconstructs signals from incomplete linear
observations using structural prior information. Compressed Sensing using
Generative Models (CSGM) extends this perspective by using the range of a
learned generator as a prior (Bora et al., 2017). Related generative-prior
approaches also treat missing-data reconstruction, including inpainting, as
an inverse problem (Asim et al., 2020).

Our formulation follows this same mathematical perspective. The forward
observation operator \(M_{\Omega}\) selects \(m<N\) entries of a complete
target window. Because this operator is not injective on the full signal
space, the complete window cannot be uniquely recovered from these
observations alone. CLP-CSGM regularizes the reconstruction through a learned
decoder range and a conditional latent prior estimated from dense auxiliary
logs.

We have clarified this relationship in the Problem setup and in the
compressed-sensing discussion. We do not claim to invert a physical
petrophysical forward model, and we do not transfer recovery guarantees
established for other sensing operators to the coordinate-restriction
operator employed here.

Accordingly, we characterize the application as missing-data reconstruction
formulated as a regularized, underdetermined inverse problem.

Where the text was changed: Method, subsection Problem setup, after the
observation model; Related work, subsection Classical and generative
compressed sensing. Citations reused: `bora2017compressed` and
`asim2020invertible`. No new equation and no change to the Abstract.

### R3-P03 -- Admissible reconstruction set

**Comment**

> very difficult to understand

**Response**

We thank the reviewer for identifying this unclear expression. We have
replaced "admissible reconstruction range" with a direct explanation of the
decoder's role. The Abstract now states that a learned decoder maps latent
codes to candidate target-property windows, limiting which curves it can
reconstruct. The Method section already defines this set formally as the
decoder range, in standardized units and in the original target units. This
restriction concerns model representability and does not imply guaranteed
physical admissibility or exact recovery.

Where the text was changed: Abstract, the decoder sentence. The corresponding
Method definitions were already correct and were not edited.

### R3-P04 -- Meaning of rho in the Abstract

**Comment**

> what is rho here?

**Response**

We thank the reviewer for requesting clarification of rho. The Abstract now
explicitly identifies rho as the nominal fraction of target-window depths
selected for observation. We also clarify the distinction between this
nominal measurement ratio and the realized fraction \(m/N\). In the
experiments, the number of observed coordinates is determined by
\(m=\max\{4,\operatorname{round}(\rho N)\}\), so the realized fraction may
differ slightly from rho because of rounding and the minimum of four
observations. This clarification does not change the sampling protocol or
any reported results.

Where the text was changed: Abstract, the gamma-ray-only sentence; Method,
subsection Problem setup; Experimental setup, the sentence that states
\(m=\max\{4,\operatorname{round}(\rho N)\}\). The sampling rule itself was
not changed.

### R3-P05 -- Weak single-channel context

**Comment**

> what is weak single-channel context?

**Response**

We thank the reviewer for pointing out that the expression "weak
single-channel context" was unclear. We have clarified that the F03-4
benchmark uses gamma ray as its only dense auxiliary input to reconstruct a
processed, density-derived porosity curve. Acoustic travel time is excluded
because it is nearly collinear with this processed target. The experiment
therefore evaluates reconstruction under a deliberately restricted input
configuration, rather than demonstrating that gamma ray is intrinsically
uninformative. We have replaced ambiguous references to "weak context" with
descriptions of the GR-only configuration. This controlled benchmark is not
presented as operational validation of porosity prediction from gamma ray
alone.

Where the text was changed: Experimental setup, the clause that called the
dense context intentionally incomplete; Results, the F03-4 profile paragraph;
Discussion, the depth-profile comparison; Conclusion, the F03-4 stress-test
sentence. The Abstract already names the gamma-ray-only case and was not
edited. The statement that gamma ray is weakly correlated with the processed
porosity, relative to sonic, remains in Data and EDA because it refers to
that correlation, not to the removed label.

### R3-P06 -- Ridge regression and low-capacity prior

**Comment**

> are you using ridge regression? please make it easier to understand

**Response**

We thank the reviewer for requesting clarification. Yes, ridge regression is
used to implement the conditional latent-prior map \(h\). It is trained to
predict latent codes from dense auxiliary logs, using codes obtained from the
target-window encoder during training. At inference, its prediction
\(\boldsymbol{z}_0\) initializes latent refinement and serves as the center
of the quadratic latent penalty. The decoder \(G\) maps the refined code to
the reconstructed target-property window. Ridge regression therefore does
not replace the decoder or perform the latent optimization.

The Introduction and Method now clarify these roles and distinguish the
conditional ridge-regression map from the direct ridge-regression baseline,
which predicts target windows from auxiliary logs and sparse observations.
The ambiguous phrase "low-capacity latent-prior reference" has been removed
from the Abstract.

### R3-P07 -- Symbols in the introductory equations

**Comment**

> you need to explain the symbols in your equations

**Response**

We thank the reviewer for requesting clearer definitions of the mathematical
symbols. In response to the related structural and notation comments, we
reorganized the manuscript so that the reconstruction objective is presented
in the Method section rather than in the Introduction.

The Problem setup now defines the target window, auxiliary-log vector,
observation coordinates, coordinate-restriction operator, sparse measurements,
and noise model before developing the reconstruction formulation. The
subsequent subsections introduce the latent code, decoder, conditional map,
scaling operators, and regularization parameter in their corresponding
mathematical context.

The symbols used in the reconstruction equations are now defined before use,
and the Introduction can be read without requiring the reader to interpret
the full optimization objective. The nominal ratio rho remains distinct from
the realized fraction m/N.

### R3-P08 -- Im(G) may look like the imaginary part

**Comment**

> The Im is like imaginary part of G where actually it indicates the range. please consider use another symbol

**Response**

We thank the reviewer for noting that \(\operatorname{Im}(G)\) can be read
as the imaginary part of \(G\).

That symbol has been removed. The range of the standardized decoder is
written \(\mathcal{R}_{G}\). Decoder outputs in the original target units
are written \(\mathcal{R}_{\mathrm{orig}}\). The local original-scale set
in the recovery condition is written
\(\mathcal{R}_{\mathrm{orig},R^{\dagger}}\). Each symbol is defined before
use, and the sets themselves are unchanged. The related-work sentence that
used \(\operatorname{Im}(G)\) now says "range" and does not introduce the
symbol before its definition. Figure 1 and its caption use
\(\mathcal{R}_{\mathrm{orig}}\) for the reconstructed curve.

Where the text was changed: Related work, subsection Classical and generative
compressed sensing; Method, subsection Learned model components (the range
definition and Figure 1); Method, subsection Interpretation and theoretical
context (the representation-error remark and the conditional local-recovery
proposition); Interpretation, scope, and limitations; Appendix Conditional
local recovery interpretation.

### R3-P09 -- Measurement noise and lambda

**Comment**

> how do you set the measurement noise? and how do you determine the lambda, the regularization coefficient.

**Response**

After coordinate subsampling, each observed coordinate receives an independent
Gaussian perturbation. Its standard deviation is 0.01 for cross-well clay
volume and 0.02 for F03-4 porosity, in the respective target units. These
values define controlled benchmark noise and were not calibrated to
instrumental uncertainty. The `30dB` filename designation is unrelated to
this added perturbation.

The revised Experimental setup cites synthetic Gaussian contamination in
well-log inversion (Abdelrahman and Szabo, 2024) as precedent for the
stress-test form, not for these amplitudes. It also cites probabilistic
sampling of uncertainty in gamma-ray-based shale-volume estimation (Adjei et
al., 2019). We did not estimate tool-specific gamma-ray uncertainty or
propagate it through the Larionov relation, so the added noise is not
presented as instrument-calibrated.

The reported experiments select \(\lambda\) by validation full-window RMSE
for each seed and measurement ratio and fix it before test evaluation. They
do not compute it from a variance ratio. Under the Gaussian hypotheses of the
MAP interpretation, \(\lambda=\sigma_\eta^2/\sigma_z^2\) applies to the
original-scale residual
\[
\|M_{\Omega}G_{\mathrm{orig}}(\boldsymbol{z})-\boldsymbol{b}\|_2^2,
\]
which is equivalent to the implemented weighted standardized residual. It
does not apply to an unweighted standardized residual.

The noise-sensitivity analysis uses a separate wider validation grid for
\(\lambda\); values at its upper boundary are reported as the largest
candidate tested, not as optima beyond the grid.

Where the text was changed: Method, subsection Test-time reconstruction (the latent-refinement objective and the residual identity); Method, subsection Interpretation and theoretical context (the MAP proposition); Experimental setup; Discussion; Appendix MAP and Tikhonov interpretation; Appendix Direct linear maps, depth interpolation, and measurement noise; the implementation-summary table.

### R3-P10 -- Covariance

**Comment**

> covariance?

**Response**

The matrices \(D_y\) and \(D_u\) contain componentwise training standard
deviations; they are scaling matrices, not estimated covariance matrices.
They do not remove correlations already present between profile coordinates.

For fixed observed coordinates, the benchmark assumes
\(\boldsymbol{\eta}\sim\mathcal{N}(\boldsymbol{0},
\sigma_\eta^2I_m)\) in the original target units. After standardization,
\(\widetilde{\boldsymbol{\eta}}=D_{\Omega}^{-1}\boldsymbol{\eta}\), so
\[
\operatorname{Cov}(\widetilde{\boldsymbol{\eta}})
=\sigma_\eta^2D_{\Omega}^{-2}.
\]
The transformed noise remains independent across coordinates, but its
variances differ when the corresponding training standard deviations differ.
This is a modeling assumption for the synthetic benchmark, not an estimated
instrumental covariance.

The factor \(\sigma_z^2I_k\) in the MAP proposition is likewise the
isotropic latent-prior hypothesis that yields the Euclidean penalty. It is
not an estimated covariance of the conditional-map residuals.

Where the text was changed: Method, subsection Problem setup (the paragraph Standardization and modeling assumptions); Method, subsection Interpretation and theoretical context (the local-recovery proposition); Appendix Conditional local recovery interpretation; Appendix Measurement Separability.

### R3-P11 -- Baseline identity

**Comment**

> also not clear what method was used as the baseline in this paper.

**Response**

We thank the reviewer for requesting clarification of the baseline methods.
We have clarified the distinction between the proposed CLP-CSGM configuration
and the methods used for comparison.

In CLP-CSGM Ridge, ridge regression implements the conditional map
\(h\) from dense auxiliary logs to the initial latent code
\(\boldsymbol{z}_0\). It is an internal
component of the proposed method, not the direct prediction baseline. This
distinction is stated in Method, subsection Learned model components.

The principal direct comparator emphasized in the cross-well results is
AE \([\boldsymbol{u},\boldsymbol{b}]\), which predicts codes of a separately
trained target autoencoder
from auxiliary logs and sparse observations. It has the lowest unweighted
mean RMSE among the direct baselines MLP, PCA, and AE on the reported
cross-well grid.

Other direct predictors include ML only,
MLP \([\boldsymbol{u},\boldsymbol{b}]\), and
PCA \([\boldsymbol{u},\boldsymbol{b}]\). Simple
approaches, including direct linear regression, direct ridge regression,
depth interpolation, and second-difference smoothing, are also evaluated.
At the reference noise level, depth interpolation outperforms CLP-CSGM in
all nine cross-well configurations, so AE should not be interpreted as the
strongest overall comparator.

Prior-only and measurement-only CSGM are structural ablations, whereas the
MLP conditional prior is used for architectural sensitivity analysis.

Where the text was changed: Experimental setup, subsection Baselines and
evaluation metrics. The Method definition of the conditional ridge map
remains unchanged.

### R3-P12 -- Why predict z rather than y?

**Comment**

> any reason for mapping u to z instead of y directly? what is the point of compressing the data in here?

**Response**

The purpose of mapping \(\boldsymbol{u}\) to a latent code is to perform
test-time measurement assimilation within the decoder representation. The
conditional map predicts
\(\boldsymbol{z}_0=h(\boldsymbol{u})\); this code initializes the first
trajectory and remains the center of the latent penalty, while the sparse
target observations update the code through the measurement-fidelity term.
The decoder then returns the complete window. In the reported model, this
search adjusts 16 latent variables for a length-64 target window.

This representation does not guarantee better prediction by itself. Direct
models can map \(\boldsymbol{u}\), or
\([\boldsymbol{u},\boldsymbol{b}]\), directly to the target. The cross-well
comparison evaluates the complete conditioning-and-refinement procedure:
CLP-CSGM Ridge is lower than AE [u,b] in seven of nine cells, with the most
consistent gains at steps 16 and 32. Because the step also changes window
sampling and training support, these results do not isolate latent dimension
as the cause.

At the reference noise level, depth interpolation is lower in all nine cells,
showing that latent reconstruction is not always needed. The Method section
now states this motivation and its conditional scope.

### R3-P13 -- Intermediate results

**Comment**

> any of these intermediate results were shown in the paper?

**Response**

The manuscript now points explicitly to three relevant outputs. First,
\(G(E(\boldsymbol{y}))\) is a retrospective representation check using the
complete target window. Its mean test RMSE is 0.0073, 0.0099, and 0.0110 at
steps 8, 16, and 32, respectively. Because it uses the unknown target, it is
not an inference-time prediction.

Second, the prior-only ablation reports
\(G_{\mathrm{orig}}(\boldsymbol{z}_0)\), with
\(\boldsymbol{z}_0=h(\boldsymbol{u})\), before sparse-measurement
refinement. Third, the full CLP-CSGM result is reported after refinement. In
the cross-well benchmark, the full method has lower mean RMSE than prior-only
in all nine configurations, with relative reductions of approximately 20.5%
to 60.6%.

Measurement-only CSGM, which sets \(\lambda=0\), is also reported as a
separate structural ablation rather than an intermediate stage. We do not
show a per-iteration optimization trajectory.

### R3-P14 -- Arrows and symbols in Figure 1

**Comment**

> also, the plot is very hard to understand. why z_tilda decode to G(z0)? the blue line below leads to nothing. what is M = Im(G)? M is the index function, lm(G) is the range of decoded measurements (the target).

**Response**

We thank the reviewer for pointing out the ambiguities in the original
Figure 1.

The figure has been revised to distinguish the conditional latent
prediction, latent refinement, and decoding operations.

The dense auxiliary logs are mapped through \(h\) to the initial latent code
\(\boldsymbol{z}_0\). The tilde in the drawing belongs to the argument of
\(h\). It is not a
second latent code, and that argument is not what is decoded. The upper
horizontal arrow shows the decoding of \(\boldsymbol{z}_0\) into
\(G_{\mathrm{orig}}(\boldsymbol{z}_0)\). The vertical arrow represents
latent refinement from \(\boldsymbol{z}_0\) to the refined code
\(\widehat{\boldsymbol{z}}\), and the lower horizontal arrow shows the
decoding of \(\widehat{\boldsymbol{z}}\) into the final reconstructed window
\(G_{\mathrm{orig}}(\widehat{\boldsymbol{z}})\).

The blue curve represents \(\mathcal{R}_{\mathrm{orig}}\), the range of
decoder outputs in the original target units. This set is distinct from
\(M_{\Omega}\), which is the coordinate-restriction operator selecting the
observed depths. The notation previously written as
\(\operatorname{Im}(G)\) has also been replaced with explicit decoder-range
notation to avoid confusion.

The revised caption explains these mappings and the role of the measurement
residual. It also clarifies that the figure is schematic and does not depict
a recorded numerical optimization trajectory, and that the observations need
not be met exactly.

Where the figure was changed: Method, subsection Learned model components,
Figure 1 and its caption.

### R3-P15 -- Confusing propositions

**Comment**

> it is very confusing the author listed these propositions.

**Response**

We thank the reviewer for pointing out that the original presentation of
the theoretical propositions was difficult to follow.

We have reorganized the theoretical discussion around three questions: how
to interpret the two terms of the reconstruction objective, under which
conditions a minimizer exists, and which factors condition recovery of the
complete target window from sparse observations.

The MAP/Tikhonov interpretation and the existence result are classical
mathematical arguments. They are now summarized in the Method section,
while their formal statements and proofs appear in the Appendix. We do not
present these arguments as new theoretical contributions. Under the stated
Gaussian hypotheses, \(\lambda\) equals the ratio of the noise variance to the
latent-prior variance. The experiments do not use that ratio: validation
selects \(\lambda\). Existence assumes a continuous decoder and
\(\lambda>0\). It gives a minimizer. It does not give uniqueness, convexity,
or convergence of Adam to a global minimizer, and it does not transfer
automatically to \(\lambda=0\). The numerical solution remains distinct from
a certified global minimizer.

The conditional local-recovery proposition remains in the main text because
it directly relates the reconstruction formulation to decoder representation
error, measurement noise and discrepancy, mismatch between the conditioned
latent code and a reference code, and local measurement separability. The
bound is stated for decoder outputs in the original target units. It assumes
a local S-REC condition and does not establish that the trained decoder
satisfies this condition under coordinate subsampling. The statement for a
global minimizer is distinguished from the extension to an inexact numerical
optimizer, which is proved in the Appendix. The result does not claim exact
recovery or superiority over interpolation or direct regression.

The revised organization separates classical interpretations from the
conditional recovery analysis and explains the purpose of each result before
presenting its mathematical statement.

Where the text was changed: Method, subsection Interpretation and theoretical
context; Appendix, subsections MAP and Tikhonov interpretation, Existence of
a minimizer, and Conditional local recovery interpretation.

### R3-P16 -- Meaning of 30 dB

**Comment**

> what do you mean by 30dB?

**Response**

We thank the reviewer for requesting clarification of the "30dB" designation.

The cross-well experiments use processed F3 well-log tables associated with
the dataset of Zhang et al. (2025). In their study, the authors report an
experimental condition with an SNR of 30. However, the publication does not
provide the mathematical definition used to calculate this value or
explicitly indicate whether it is expressed in decibels.

The input files used in our experiments retain the designation "30dB" in
their filenames. We have preserved this nomenclature for consistency with
the dataset and to ensure that the files employed in our experiments remain
clearly identifiable and reproducible.

We have clarified this point in the Data and exploratory analysis section.
The designation is not presented as an independently verified
signal-to-noise ratio.

Importantly, this designation is distinct from the additional Gaussian noise
introduced into the sparse target observations in our reconstruction
experiments, with standard deviations of 0.01 for clay volume and 0.02 for
porosity, expressed in their respective target units.

This clarification does not affect the experimental protocol or the reported
results.

### R3-P17 -- Definition of Vc

**Comment**

> what is Vc?

**Response**

We thank the reviewer for requesting clarification of the notation \(V_c\).

In this manuscript, \(V_c\) denotes the clay-volume fraction, a dimensionless
petrophysical quantity. In the F3 cross-well benchmark, this target is a
processed estimate derived from the gamma-ray log using the Larionov
relation, following the dataset associated with Zhang et al. (2025).

It is therefore not an independently measured laboratory clay fraction. The
cross-well experiments reconstruct this processed target curve from the
available auxiliary logs and sparse observations.

We checked this definition across the manuscript. The Abstract, Discussion,
and Conclusion already identified the Larionov origin and stated that the
sparse observations are subsamples of the processed curve rather than
independent laboratory measurements. At the first use of the notation, in
Data and exploratory analysis, we now state explicitly that \(V_c\) denotes
the dimensionless clay-volume fraction estimated from gamma ray using the
Larionov relation. Tables and figures were left unchanged. The tabulated
values lie between about 0.01 and 0.27, on a fractional scale.

### R3-P18 -- Density is ambiguous

**Comment**

> use probability density, pdf, or other words. the density is confusing as there is the RHOB density log.

**Response**

We thank the reviewer for pointing out the ambiguity of the term "Density" in Figure 3.

We agree that this label could be confused with the bulk-density (RHOB) well log used elsewhere in the manuscript.

The vertical axis of Figure 3 has therefore been relabeled "Probability density", explicitly identifying the statistical quantity represented by the empirical clay-volume distributions. Each histogram is normalized so that its area equals one. The vertical scale is not a count of samples.

This correction distinguishes the probability-density representation from the petrophysical bulk-density log without changing the underlying data or the plotted distributions.

Where the figure was changed: Data and exploratory analysis, Figure 3.

### R3-P19 -- Figure 7 and GR-derived Vc

**Comment**

> very difficult to see the results. also, the Vc is not a very good target, it might be calculated from GR. you could  have predicted it almost perfectly using just GR, see the correlation 0.97

**Response**

We addressed both points. The revised depth profile is Figure 8. Its panels
share one clay-volume scale and one depth axis, increasing downward, for
F03-4 at step 16, seed 7, and rho=0.20. It compares the processed reference,
ML only, the direct autoencoder, and CLP-CSGM Ridge. MLP and PCA are shown
separately in supplementary Figure 24 on their full prediction range.

We also clarify that \(V_c\) is derived from gamma ray through the Larionov
relation, so the approximately 0.97 correlation is expected. When gamma ray
is available, the Larionov relation is the appropriate direct calculation;
CLP-CSGM is not proposed as its replacement. The main benchmark deliberately
withholds gamma ray and reconstructs the processed target from sonic, bulk
density, acoustic impedance, \(V_p\), and sparse target observations. It is
therefore a controlled incomplete-information test.

Reintroducing gamma ray in the step-16 supplement reduces the mean CLP-CSGM
RMSE from 0.02689 to 0.01380 at rho=0.10 without the boundary embargo, while
the ordering relative to the direct autoencoder and measurement-only CSGM is
unchanged. This does not make the main benchmark an operational validation:
its sparse observations are subsamples of the processed Larionov-derived
curve, not independent clay-volume measurements. At the reference noise,
depth interpolation also remains lower than CLP-CSGM in all nine cells.

Where the text was changed: Data and exploratory analysis; Results;
Discussion; Appendix, subsection Cross-well channel ablation and
embargo-style boundary check.
