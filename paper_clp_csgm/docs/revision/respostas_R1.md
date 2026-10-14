# Reviewer 1 -- proposed replies

Each ID has its own slot. If the same manuscript change fully answers more
than one ID, the English reply is copied verbatim.

Every completed reply names where the manuscript text was changed (Abstract,
section title, or appendix subsection title), not only what was changed.

Source comments: `docs/revision/CLP_CSGM_Pareceres_Consolidados.md`.

---

## R1-M01 -- Origin of targets and scope of validation

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for emphasizing that the derived origin of the F3
targets is central and must appear early. In the original submission this
limitation was stated in the Data section and in the Discussion, but it was
not explicit in the Abstract or the Conclusions.

We have revised both. The Abstract now states that the held-out-well
clay-volume target is obtained from gamma ray by the Larionov relation, that
the single-well porosity target is processed and density-derived, and that
sparse observations are coordinate subsamples of these same processed curves.
It also states that the benchmarks test reconstruction of processed target
logs rather than recovery against independently acquired core or laboratory
measurements. The Conclusions now repeat this F3 evidence limit before the
sentence on remaining validation needs. The closing Abstract sentence is
restricted to the tested processed-log benchmarks.

Where the text was changed: the Abstract, and the Conclusion (the F3
evidence-limit paragraph, before the remaining-work sentence). The Data and
exploratory analysis section and the Interpretation, scope, and limitations
section already stated this origin and were left in place.

No new experiments or numerical results were added for this comment. A
separate independent-measurement test remains outside the scope of this
textual revision and is addressed under R1-M11.

---

## R1-M02 -- Few wells and operational transfer

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for noting that three development wells and one
held-out well form a controlled test and cannot support broad claims of
cross-well robustness or operational transfer. The original submission already
named F02-1, F03-2, and F06-1 as the development wells and F03-4 as the
held-out well in the experimental protocol, but the Abstract, Introduction,
and Conclusions did not restrict the language to that single-holdout design.

We have revised the text accordingly. The Abstract now states that the F3
evaluation uses four wells, with three wells for model development and F03-4
held out, and that the results are not presented as operational multi-well
transfer. The third contribution in the Introduction is described as
benchmark-specific. The experimental protocol now states that reported
cross-well performance refers to this single held-out well. The Conclusions
state that the cross-well comparison is a controlled single-holdout benchmark
and is not evidence of operational cross-well or multi-field transferability.

Where the text was changed: the Abstract; the Introduction (third
contribution); Experimental setup, subsection Cross-well clay-volume (the
single held-out well); and the Conclusion (single-holdout paragraph).

No additional wells or resampling experiments were added for this comment.
Requests for further held-out wells or repeated resampling are addressed
under R1-M03.

---

## R1-M03 -- Few windows and resampling

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for pointing out the small development support at the
largest window step. We kept the main nine-cell grid unchanged and added a
supplementary resampling check at step 32 only. The held-out well remains
F03-4. The sparse observation set Omega is frozen to paper seed 7. We redrew
the development train/validation split ten times with the same validation
budget (n_train=27, n_val=3) and compared CLP-CSGM Ridge with AE [u,b] at
rho in {0.05, 0.10, 0.20}.

CLP-CSGM Ridge has lower mean RMSE than AE [u,b] at every ratio (0.0342 vs
0.0407 at rho=0.05; 0.0305 vs 0.0388 at rho=0.10; 0.0188 vs 0.0362 at
rho=0.20) and lower RMSE in all ten splits. This check does not add a
second held-out well.

Where the text was changed: Experimental setup, subsection Training,
validation, and test protocol (pointer at step 32); Appendix, subsection
Step-32 development-split resampling (table and interpretation). The main
nine-cell tables in Results were not changed.

---

## R1-M04 -- Three seeds and uncertainty intervals

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for noting that n=3 Student-t intervals remain small,
that several published intervals include zero, and that the ranking changes
by regime. We agree that a larger seed panel is the appropriate check.

We kept the submitted three-seed nine-cell tables as the main result and
added a supplementary ten-seed panel for the same CLP-CSGM Ridge versus AE
[u,b] comparison (seeds 7, 23, 41 plus 13, 29, 31, 37, 43, 47, 53). The
held-out well remains F03-4. Sparse coordinates Omega are redrawn per
(seed, rho), as in the paper protocol. Direct MLP/PCA, structural
ablations, and the F03-4 grid were not rerun.

With n=10 (Student-t, df=9), none of the nine paired DeltaRMSE intervals
include zero. The expanded panel is favorable to CLP-CSGM Ridge in the
intended low-training-support regime: lower RMSE at every rho for steps 16
and 32 (about 52 and 27 windows; 9/10 or 10/10 seeds) and at step 8,
rho=0.20 (7/10). The two exceptions are the largest training pool (step 8,
about 104 windows) at rho=0.05 (AE better on 10/10 seeds) and rho=0.10
(AE better on 8/10). Additional seeds therefore tighten the evidence that
CLP-CSGM helps when training windows are scarce; they do not show a uniform
gain at every small measurement ratio, nor overall superiority.

Where the text was changed: Experimental setup, subsection Baselines and
evaluation metrics (pointer from the n=3 uncertainty paragraph); Results,
subsection Cross-well low-data results (next to the three-seed paired table);
Appendix, subsection Ten-seed CLP versus AE panel (table and the
low-training-support reading). The published three-seed tables were not
replaced.

---

## R1-M05 -- Conditional advantage, not overall superiority

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for asking that the seven-of-nine cross-well result,
and the GR-only reversal at higher measurement ratios, be framed as a
conditional advantage rather than overall superiority.

The Abstract now states that the comparison is a conditional advantage, not
overall superiority. It places the seven-of-nine gains where training
support is smallest (steps 16 and 32) and states that the direct autoencoder
has lower mean RMSE in the two remaining cells, both at step 8 for rho=0.05
and rho=0.10. For the GR-only benchmark, the Abstract states that CLP-CSGM
has the lowest mean RMSE at the two sparsest ratios and that the direct
autoencoder has lower mean RMSE at rho=0.40 and rho=0.60. The Conclusions
repeat this framing and name the same exceptions.

Where the text was changed: the Abstract (conditional-advantage paragraph)
and the Conclusion (the paired-protocol paragraph). The manuscript has no
separate Highlights section. No new experiments were added for this comment.

---

## R1-M06 -- Why Ridge is the primary reference

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for noting that the latent prior is not a minor
implementation detail. MLP has lower mean RMSE in all nine cross-well cells,
whereas Ridge has lower mean RMSE throughout the single-well GR-only
benchmark. That contrast shows that the class of h materially affects
performance.

We have made this clearer in the text. CLP-CSGM is the reconstruction scheme
(learned decoder, dense-log anchor h(u), and sparse-measurement consistency).
Ridge is the primary reference because it is a linear low-capacity map, so
the main tables use one fixed h across regimes. It is not presented as the
best prior in every setting. The MLP comparison uses the same pipeline: MLP
is better in all nine cross-well cells, and Ridge is better on the F03-4
grid. The prior class should be selected on validation data for the intended
regime, not by adopting whichever variant wins on the held-out test set.

Where the text was changed: the Introduction (Ridge named as the linear
reference, not as uniformly best); Method, subsection Learned model
components (the scheme versus the choice of h); Results, subsection
Latent-prior model sensitivity (MLP wins all nine cells; Ridge stays the
headline comparator); Interpretation, scope, and limitations (why Ridge is
retained, and that h is not chosen from the test ranking). No new experiments
were added, and the main tables were not switched to MLP.

---

## R1-M07 -- Proposition 3 does not verify S-REC on the trained decoder

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for stressing that Proposition 3 is conditional and
does not prove S-REC for the trained decoder. That distinction was already
stated in the Method section and in Interpretation, scope, and limitations,
and the appendix diagnostics already say that they do not prove S-REC. It
was not stated in the Abstract or the Introduction.

We have made it explicit in both places so the local-recovery statement is
not read as a formal recovery guarantee for the trained model. The Abstract
now states what the bound does and what it does not: under stated conditions
it separates decoder error, prior mismatch, measurement noise, and local
separability, and it does not prove that the measurement operator satisfies
S-REC on the trained decoder. The Introduction states the same role and the
same limit after the three contributions. No new experiments or proofs were
added for this comment.

Where the text was changed: the Abstract (sentence after the inference-time
description) and the Introduction (sentence after the three contributions).
Method, subsection Interpretation and theoretical context, and
Interpretation, scope, and limitations already contained this limit and were
left in place.

---

## R1-M08 -- Direct comparator with explicit mask/coordinates

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for asking whether the CLP-CSGM gain could come from
explicit use of the sampling coordinates rather than from the latent inverse
formulation. The main direct baselines still receive [u, b] with positional
identity implicit in the column order. We did not add a new method to that
grid.

We added one supplementary direct AE on the same nine cross-well cells,
seeds {7, 23, 41}, and noise realization. Its input is [u, b_Omega, mask]:
b_Omega places the noisy measurement on the observed window coordinate and
is zero elsewhere, and the mask is the binary indicator of Omega. MLP, PCA,
ablations, and the F03-4 grid were not rerun.

The explicit mask does not remove the CLP-CSGM advantage where training
support is smallest. At steps 16 and 32, CLP-CSGM Ridge has lower mean RMSE
than the masked AE at every rho. At step 8, the masked AE remains better at
rho=0.05 and rho=0.10, as the unmasked AE does. The mask shifts the direct
AE only modestly and not always downward. This check keeps Omega fixed
inside each cell.

Where the text was changed: Experimental setup, subsection Training,
validation, and test protocol (the main protocol is unchanged; one sentence
points to the supplement); Results, subsection Cross-well low-data results
(one sentence next to the three-seed paired comparison); Appendix,
subsection Explicit-mask direct AE (table and interpretation). The published
nine-cell tables were not extended with this method.

---

## R1-M09 -- GR-only porosity is a controlled test

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for asking that the GR-only porosity results not be
read as general porosity reconstruction performance. Using gamma ray alone
is a controlled sparse-assimilation experiment under intentionally incomplete
context. Acoustic travel time is excluded because it is nearly collinear
with the processed porosity target and would provide a near-direct proxy.
The reported errors, including 0.016 versus 0.018 at rho=0.20, describe that
stress test. The depth profile at rho=0.20 shows that CLP-CSGM follows the
broad trend of the processed porosity curve under this weak context. These
figures are not evidence of general porosity reconstruction with a full
operational log suite or against independent core measurements.

The Experimental setup and the Interpretation, scope, and limitations
section already described this benchmark as a controlled weak-context test.
The Abstract and the Conclusions now state the same limit next to the
GR-only RMSE. The depth-profile figure in the Results was retained.

Where the text was changed: the Abstract (sentences after the GR-only RMSE)
and the Conclusion (sentences after the F03-4 GR-only RMSE). Experimental
setup, subsection Single-well F03-4 GR-only porosity benchmark, and
Interpretation, scope, and limitations already contained this limit and were
left in place.

No new experiments were added for this comment. A test against independent
core measurements remains outside this textual revision and is addressed
under R1-M11. The realism of excluding gamma ray and acoustic travel time
from the main inputs is addressed under R2-03.

---

## R1-M10 -- Update literature positioning

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for the four recent studies on well-log reconstruction,
transferable subsurface learning, and data-driven petrophysical prediction. We
read them and positioned CLP-CSGM by the task each method solves.

Log completion reconstructs missing intervals of a wireline curve from that
curve. Sequence-based GANs generate synthetic logs and fill gaps in gamma ray,
sonic, neutron porosity, deep resistivity, and bulk density. A fine-tuned
time-series foundation model forecasts the next interval of the same suite and
flags anomalies, including a zero-shot check in a second basin. Geomodel
translation uses a Pix2Pix conditional GAN to map two-dimensional slices of
facies, porosity, permeability, and water saturation from a masked image or
from another property. Direct regression uses ANN and random-forest models to
predict total porosity, effective porosity, and shale volume from wireline
logs. In those settings the trained map fills a curve, fills a property image,
or returns a property from the logs.

CLP-CSGM recovers a window of a target property when dense logs are available
along the window and only sparse observations of that property are in hand.
The conditional map predicts z_0 = h(u) from the dense logs. This code
initializes the first latent trajectory and remains the center of the
quadratic penalty. The sparse values remain measurements. After training,
latent refinement balances agreement with those measurements against
deviation from z_0. A residual at the observed depths can remain. The
difference is which quantity is unknown. In log completion, geomodel
translation, and direct regression, the missing property values are what the
trained model must produce. In CLP-CSGM those sparse values are already known
measurements, and the unknown is the remainder of the target window. The
inference step penalizes disagreement with those measurements. We do not
compare numerical errors with the four studies, because the unknowns are not
the same.

Where the text was changed: the Introduction (end of the first paragraph);
Related work, subsection Petrophysical log prediction (the comparison of the
three neighboring tasks with CLP-CSGM, including the closing sentences on
which quantity is unknown); and Interpretation, scope, and
limitations (the paragraph that states when the measurement term is the
relevant formulation). Four entries were added to the bibliography.

No new experiments were added for this comment.

---

## R1-M11 -- Test against independent measurements

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for asking for a test against a target that is not
produced by deterministic processing of the same log suite. The F3 benchmarks
remain processed-log reconstructions. We added a separate single-well
supplement on laboratory porosity and did not put it in the main grid.

The interval is well 861, 87 laboratory-porosity samples from 5205.91 m to
5233.72 m. Dense context is gamma ray, deep resistivity, shallow resistivity,
and lithotype. Density and the neutron, sonic, and Phi_ND porosity channels
are excluded because they track the laboratory porosity. Depth is split into
three contiguous blocks before windowing. Windows have length 8. For each
fold, seed, and ratio, the same coordinates are the measurements in every
window, the measurement is the laboratory value with no added noise, and RMSE
is computed only on the other coordinates. Seeds are {7, 23, 41}. Ratios are
0.25 (m=2) and 0.50 (m=4).

The direct AE has the lower mean held-out RMSE at both ratios: 0.0537 versus
0.0602 at rho=0.25, and 0.0523 versus 0.0580 at rho=0.50. CLP-CSGM Ridge has
the lower RMSE in 5 of 9 splits at rho=0.25 and in 6 of 9 at rho=0.50. It has
lower RMSE than the log-only anchor in all 9 splits at both ratios.

The block split is the tradeoff. On 5205.91--5214.48 m, CLP-CSGM Ridge has the
lower block-mean RMSE at both ratios. Raising rho from 0.25 to 0.50 lowers
that RMSE from 0.0338 to 0.0228, while the direct AE stays near 0.035. The
measurement term is what improves: the anchor is still close to the test logs
(mean Phi_lab about 0.092, gamma ray about 15 API, high resistivity), and
extra laboratory coordinates pull the latent code onto the held-out points.
The direct AE already takes those coordinates as features, so the same
increase in rho changes it little. The middle block is close, with CLP-CSGM
Ridge still slightly lower. This is not the scarce-window regime of F3 steps
16 and 32: each test fold still has 28 to 35 overlapping training windows.

The pooled mean goes the other way because of 5224.39--5233.72 m. Held out,
that block has mean Phi_lab about 0.145, gamma ray about 30 API, and median
deep resistivity about 600, against low gamma ray and resistivity near
1600--1900 in the blocks that train the anchor. The lithotype mix changes as
well. The ridge anchor is a linear map fitted on the other blocks, so z0
starts far from the laboratory windows. With m=2 or m=4 in a length-8 window,
the residual still beats the log-only anchor (0.0931 versus 0.1142 at
rho=0.25, and 0.1016 versus 0.1234 at rho=0.50) but does not catch the direct
AE (0.0691 and 0.0676). The AE uses the same b as features and does not have
to pass through the shifted anchor.

CLP-CSGM Ridge therefore gains when the logs seen in training still describe
the test interval and the laboratory coordinates can finish the correction. It
loses the pooled mean when the test logs move outside that support and only a
few measurements are available to repair z0. The measurement term does not
fail. The F3 low-support advantage is not a mean win on this laboratory
target. The supplement is one cored interval. The Lapa permeability case
remains a limited-sample supplement whose sparse observations are subsamples
of the same permeability series.

Where the text was changed: Experimental setup (the sentence after the Lapa
pointer); Results (the paragraph after the Lapa permeability summary, which
states the mean ranking, the anchor correction, and where that correction is
enough); Interpretation, scope, and limitations (the laboratory-porosity
paragraph, same conclusion); the Conclusion (the sentence after the F3
evidence limit); and the Appendix, subsection Laboratory porosity on a cored
interval (pooled table, block table, log summaries, and the gain/loss
discussion).

---

## R1-m01 -- Shorten the title

**Status:** `text_done`

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

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for this suggestion. We tightened the repeated framing
and left each distinction in the section where it does the work.

The Introduction now states once that a direct map from dense logs and sparse
values need not match those values at inference, and that this paper treats
the sparse values as measurements. It names log completion, geomodel
translation, and direct regression only to point to the comparison in Related
work. The two sentences after the objective name the measurement term and the
anchor term, and they no longer restate that role.

Related work, subsection Petrophysical log prediction, no longer opens by
repeating that direct regression treats sparse values as predictors. The
comparison of the three neighboring tasks with CLP-CSGM stays there, including
the statement of which quantity is unknown.

Related work, subsection Conditional generative inverse problems and
positioning, no longer restates the direct-fusion contrast before the
experiments. The closing paragraph states where the dense-log anchor and the
measurement residual enter, then names the three comparators used in the
experiments.

The objective in the Introduction is unchanged.

Where the text was changed: the Introduction (end of the first paragraph, and
the two sentences after the objective); Related work, subsection Petrophysical
log prediction (opening paragraph); and Related work, subsection Conditional
generative inverse problems and positioning (closing paragraph).

---

## R1-m03 -- Figure 2 readability

**Status:** `pending`

**Deferred.** No change to the figure in this pass. The workflow stays as drawn.
If the figure is revised later, enlarge only the smaller labels: axis names,
the note under the sparse observations, and the lines for \(M\) and
\(\lambda\). Those notes can instead move into the caption, which already
states them. Do not redraw the flow.

## R1-m04 -- AE as best comparator only in this benchmark

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for this clarification. The statement that the direct AE
is the most competitive direct baseline is now explicitly limited to this
cross-well benchmark. In the Conclusion, the phrase is "the most competitive
direct baseline in this cross-well benchmark". Results already restricts the
same ranking to this grid: "the direct baseline with the lowest unweighted
mean RMSE across the grid" and "the most competitive direct comparator in this
grid". The Abstract names the direct autoencoder as a baseline and does not
call it the strongest method in general.

Where the text was changed: the Conclusion (the sentence that reports the
seven-of-nine cross-well comparison).

---

## R1-m05 -- Frozen code version

**Status:** `pending`

**Deferred.** Do not create the tag during the revision. The manuscript and
the code are still changing: this round added tests (step-32 development
splits, the seed panel, the explicit-mask AE, and the well 861 laboratory
porosity check). A frozen release must match the resubmitted manuscript, so
the tag is created at the end. Computer Code Availability then cites that tag
instead of only the live repository URL.
