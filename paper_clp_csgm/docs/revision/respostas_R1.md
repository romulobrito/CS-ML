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

---

## R1-M02 -- Few wells and operational transfer

**Status:** `text_done`

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

No additional held-out wells were added for this comment. Requests for
further held-out wells or repeated resampling are addressed under R1-M03.

---

## R1-M03 -- Few windows and resampling

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for pointing out the small development support at the
largest window step. The earlier statement of about 27 windows before the
split referred to the training count. At step 32 the development pool has 30
windows. Reserving 10 percent of each development well leaves 27 training
windows and 3 validation windows.

We kept the main nine-cell grid unchanged and added a supplementary
resampling check at step 32 only. The held-out well remains F03-4. The sparse
observation set Omega is frozen to paper seed 7. We redrew the development
train/validation assignment ten times by random sampling within each
development well, with the same validation budget, and compared CLP-CSGM
Ridge with AE [u,b] at rho in {0.05, 0.10, 0.20}.

CLP-CSGM Ridge has lower mean RMSE than AE [u,b] at every ratio (0.0342 vs
0.0407 at rho=0.05; 0.0305 vs 0.0388 at rho=0.10; 0.0188 vs 0.0362 at
rho=0.20) and lower RMSE in all ten splits. Adjacent windows have length 64
and step 32, so they share 32 depth samples. This redraw does not apply a
boundary embargo. In each of the ten splits, all three validation windows
share at least one depth sample with a training window of the same well
(30 of 30 validation windows). One validation window comes from F02-1 and
two from F03-2; F06-1 has only two windows and stays in the training set.
The ten splits therefore assess sensitivity to the development partition.
They do not represent independent validation without shared samples. They
are not ten independent wells, and this check does not add a second held-out
well.

Where the text was changed: Experimental setup, subsection Training,
validation, and test protocol (window counts and the step-32 pointer);
Appendix, subsection Step-32 development-split resampling (table,
interpretation, and the overlap limitation). The main nine-cell tables in
Results were not changed.

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
include zero. Each interval describes variability across seeds on the single
held-out well F03-4. It is not evidence of generalization across wells. The
nine intervals are separate cell summaries, not a simultaneous test, and
overlapping windows are not counted as independent samples. The expanded
panel is favorable to CLP-CSGM Ridge in the intended low-training-support
regime: lower RMSE at every rho for steps 16 and 32 (52 and 27 training
windows; 9/10 or 10/10 seeds) and at step 8, rho=0.20 (7/10). The two
exceptions are step 8, where the training count is 104, at rho=0.05 (AE
better on 10/10 seeds) and rho=0.10 (AE better on 8/10). Additional seeds
therefore tighten the evidence that CLP-CSGM helps when training windows are
scarce; they do not show a uniform gain at every small measurement ratio,
nor overall superiority.

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

The Abstract states that CLP-CSGM has lower mean RMSE than the direct
autoencoder in seven of nine cross-well configurations, chiefly when
training support is limited, and that it loses in two. This comparison
remains a conditional advantage over that autoencoder, not superiority
over every comparator. The Abstract names the GR-only regimes: CLP-CSGM
has lower mean RMSE at rho in {0.20, 0.30}, the autoencoder is lower at
rho in {0.40, 0.60}, and the errors are nearly equal at rho=0.50. Those
exceptions remain in the Results and in the Conclusion. The Abstract also
states that, at the reference noise standard deviation 0.01, depth
interpolation has lower error than CLP-CSGM in all nine configurations, and
that under larger perturbations that comparison changes. Those noise levels
were used for sensitivity analysis and were not calibrated to field
measurement uncertainties. The Conclusion names the step-8 losses at
rho=0.05 and rho=0.10, the GR-only reversal at rho=0.40 and rho=0.60, and
the noise-level exceptions. This wording does not close requests for
further experiments or for broader generalization.

Where the text was changed: the Abstract (conditional-advantage paragraph)
and the Conclusion (the paired-protocol paragraph and the noise paragraph).
The manuscript body has no Highlights section. The separate submission
Highlights, each within 85 characters, are in highlights.txt:

CLP-CSGM beats the direct AE in 7 of 9 cross-well cells.
Gains are concentrated in the smaller training sets.
GR-only: CLP lower at 20%-30%; AE lower at 40% and 60%.
At reference noise, interpolation is lower in all nine cells.
The advantage is conditional, not superiority over every method.

No new experiments were added for this comment, and the body was not
edited again for the Highlights.

---

## R1-M06 -- Why Ridge is the primary reference

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for noting that the latent prior is not a minor
implementation detail. An audit of the earlier comparison found that it did
not isolate the class of h. In the cross-well runs, the MLP prior also
received gamma ray, while the Ridge prior did not. In the single-well
GR-only runs, the two priors used different depth partitions, with 224 and
275 test windows. Those runs are not a controlled ablation.

The comparison was repeated with the remaining experimental choices held
fixed. In the controlled cross-well grid, both priors use the channels
sonic, rhob, ai, and vp, without gamma ray, and the same windows and sparse
observations. The MLP prior has lower mean RMSE in seven of the nine
configurations. Ridge has lower mean RMSE at step 32 for measurement ratios
0.10 and 0.20. The unweighted mean RMSE across the nine configurations
decreases from 0.028623668 for Ridge to 0.025383384 for MLP, a relative
reduction of 11.3%. The MLP advantage becomes smaller as the number of
training windows decreases and is nearly absent at step 32. In the
controlled F03-4 GR-only comparison, both priors use the same depth-first
partition and the same 224 test windows. Ridge has lower mean RMSE at all
five measurement ratios, with relative reductions of approximately 15.6% to
19.7% compared with MLP.

Ridge remains the primary reference because it is a linear low-capacity map,
so the main tables use one fixed h across regimes. It is not presented as
the best prior in every setting. The class of h should be selected on
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

**Proposed reply:**

We thank the reviewer for stressing that the conditional local-recovery
proposition is conditional and does not prove S-REC for the trained
decoder. Method, Interpretation, scope, and limitations, and the appendix
diagnostics already stated that limit. The original Abstract did not.

The Abstract now states that the local recovery bound remains conditional
and does not certify S-REC. The Introduction states what the bound does
and what it does not: under stated conditions it separates decoder error,
prior mismatch, measurement noise, and local separability, and it does not
prove that the measurement operator satisfies S-REC on the trained
decoder. No new experiments or proofs were added for this comment.

Where the text was changed: the Abstract (one sentence on the conditional
bound); the Introduction (sentence after the three contributions). Method,
subsection Interpretation and theoretical context, and Interpretation,
scope, and limitations already contained this limit and were left in place.

---

## R1-M08 -- Direct comparator with explicit mask/coordinates

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for asking for a direct comparator that encodes the
sampling positions explicitly, so that the CLP-CSGM advantage is not
attributed only to a more informative observation representation. The main
direct baselines still receive [u, b] with positional identity implicit in
the column order. We did not add a new method to that grid.

We added one supplementary direct AE on the same nine cross-well cells,
seeds {7, 23, 41}, and paired noise realizations. Its input is
[u, b_Omega, mask]: b_Omega places the noisy measurement on the observed
window coordinate and is zero elsewhere, and the mask is the binary
indicator of Omega. An audit of the first implementation found that
standardizing the full concatenated vector zeroed the constant mask
columns, so the network did not receive the original binary indicators.
That representation was corrected: numeric channels are standardized from
training data only, and the trailing binary mask is left unscaled. The
mask-aware AE also uses the same base training seed as the conventional
direct AE. MLP, PCA, ablations, and the F03-4 grid were not rerun.

Under that controlled protocol, CLP-CSGM Ridge has lower mean RMSE than the
mask-aware AE in seven of the nine configurations. At steps 16 and 32, the
smaller training regimes, CLP-CSGM Ridge has lower mean RMSE in all six
configurations and lower RMSE in 17 of the 18 paired seed comparisons. At
step 8, the mask-aware AE remains better at rho=0.05 and rho=0.10. Because
Omega is fixed within each configuration, the binary mask is constant across
windows, and the conventional AE already receives the measurements in a
stable positional order. The experiment therefore assesses an alternative
explicit spatial encoding. It does not isolate the causal contribution of
latent inverse optimization, and the difference relative to the historical
masked-AE run cannot be attributed to the mask preservation alone.

Where the text was changed: Experimental setup, subsection Training,
validation, and test protocol; Results, subsection Cross-well low-data
results; Appendix, subsection Explicit-mask direct AE
(Table~\ref{tab:crosswell_explicit_mask} and the corrected protocol). The
published nine-cell Ridge-versus-AE tables were not extended with this
method.

---

## R1-M09 -- GR-only porosity is a controlled test

**Status:** `closed`

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
limit next to that RMSE.

Where the text was changed: Results, subsection Single-well F03 GR-only
results (the F03-4 depth-profile paragraph, and two sentences that now report
the observed RMSE advantage of CLP-CSGM Ridge over the direct autoencoder
rather than an isolated causal effect of latent refinement); Appendix
Illustrative sparse-measurement windows; Appendix F03-4 Qualitative
Reconstruction Views; and the Conclusion (sentences after the F03-4 GR-only
RMSE). Experimental setup, subsection Single-well F03-4 GR-only porosity
benchmark, and Interpretation, scope, and limitations already contained the
controlled-test limit and were left in place.

No new experiments were added for this comment. A test against independent
core measurements remains outside this textual revision and is addressed
under R1-M11. The realism of excluding gamma ray and acoustic travel time
from the main inputs is addressed under R2-03.

---

## R1-M10 -- Update literature positioning

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for the four recent studies on well-log reconstruction,
transferable subsurface learning, and data-driven petrophysical prediction. We
read them and incorporated them into the bibliographic positioning of
CLP-CSGM.

Log completion reconstructs missing intervals of a wireline curve from
available observations and contextual information, including neighboring
depths and other log channels. Sequence-based GANs generate synthetic logs
and fill gaps in gamma ray, sonic, neutron porosity, deep resistivity, and
bulk density. A fine-tuned time-series foundation model forecasts the next
interval of the same suite and flags anomalies, including a zero-shot check
in a second basin. Geomodel translation uses a Pix2Pix conditional GAN to map
two-dimensional slices of facies, porosity, permeability, and water
saturation from a masked image or from another property. Direct regression
uses ANN and random-forest models to predict total porosity, effective
porosity, and shale volume from wireline logs. In those settings the trained
map fills a curve, fills a property image, or returns a property from the
logs. Imputation methods can therefore exploit known observations and
cross-channel context.

CLP-CSGM recovers a window of a target property when dense logs are available
along the window and only sparse observations of that property are in hand.
The conditional map predicts z_0 = h(u) from the dense logs. This code
initializes the first latent trajectory and remains the center of the
quadratic penalty. The sparse values remain measurements. After training,
latent refinement balances agreement with those measurements against
deviation from z_0. A residual at the observed depths can remain. The central
distinction is that CLP-CSGM combines the conditional latent prior with an
explicit measurement-fidelity penalty during inference. The cited completion
and translation methods primarily learn mappings for gap filling or property
prediction. This distinction concerns the inference formulation and does not
establish superior predictive performance over the cited approaches, whose
targets, observation patterns, and evaluation protocols differ. We do not
compare numerical errors with the four studies. The TimeGPT study also
examines zero-shot transfer across geological basins. Our cross-well
benchmark retains a single held-out well and does not demonstrate
basin-agnostic transferability.

Where the text was changed: the Introduction (end of the first paragraph);
Related work, subsection Petrophysical log prediction (imputation
characterization refined; the distinction remains the conditional latent
prior plus an explicit measurement penalty at inference); and
Interpretation, scope, and limitations (transferability limit relative to
basin-level foundation-model evaluation). The four bibliography entries were
already present. No new experiments were added for this comment.

---

## R1-M11 -- Test against independent measurements

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for asking for a test against a target that is not
produced by deterministic processing of the same log suite. The F3 benchmarks
remain processed-log reconstructions. We added a separate single-well
supplement on laboratory porosity and did not put it in the main grid.

The supplementary target consists of porosity values identified as laboratory
measurements in the Auddys source dataset, rather than values
deterministically computed from the auxiliary gamma-ray, resistivity, or
lithotype channels by our processing pipeline. This provides a limited
response to the request for a target outside that deterministic log-suite
processing. Original per-plug laboratory assay records were not available for
independent verification. The interval is well 861, 87 Phi_lab samples from
5205.91 m to 5233.72 m. Dense context is gamma ray, deep resistivity, shallow
resistivity, and lithotype. Density and the neutron, sonic, and Phi_ND
porosity channels are excluded because they track the laboratory porosity.

Training and test intervals are separated by contiguous depth blocks before
windowing. Windows have length 8 and step 1. For each fold, seed, and ratio,
the same relative coordinates are the measurements in every window, the
measurement is the laboratory value with no added noise, and RMSE is computed
only at coordinates not supplied to that reconstruction as measurements.
Because the test windows overlap, a physical depth scored in one window may
have been measured in another. Therefore, the experiment is a within-window
sparse-assimilation evaluation on held-out depth blocks, not a globally
withheld-depth validation. Seeds are {7, 23, 41}. Ratios are 0.25 (m=2) and
0.50 (m=4). The nine comparisons per ratio are three depth blocks repeated
under three seeds, not nine independent geological intervals.

The direct AE has the lower mean within-window held-out RMSE at both ratios:
0.0537 versus 0.0602 at rho=0.25, and 0.0523 versus 0.0580 at rho=0.50.
CLP-CSGM Ridge has the lower RMSE in 5 of 9 block-seed pairs at rho=0.25 and
in 6 of 9 at rho=0.50. It has lower RMSE than the log-only anchor G(z0) in
all 9 pairs at both ratios.

On 5205.91--5214.48 m, CLP-CSGM Ridge has the lower block-mean RMSE at both
ratios. In that shallowest block, mean held-out RMSE decreases from 0.0338 at
rho=0.25 to 0.0228 at rho=0.50, while the direct AE stays near 0.035. Because
the coordinate sets are sampled separately at each measurement ratio, this
comparison changes both the number and the placement of observations and does
not isolate the effect of measurement count. The middle block is close, with
CLP-CSGM Ridge still slightly lower. This is not the scarce-window regime of
F3 steps 16 and 32: each test fold still has 28 to 35 overlapping training
windows.

The pooled mean is lower for the direct AE, and the larger share of that
error sits in 5224.39--5233.72 m. When that block is held out for testing,
it has mean Phi_lab about 0.145, gamma ray about 30 API, and median deep
resistivity about 600, against low gamma ray and resistivity near 1600--1900
in the blocks that train the anchor. The lithotype mix changes as well. The
decoded prediction G(z0) differs substantially from the target windows in
that block. With m=2 or m=4 in a length-8 window, CLP-CSGM Ridge still beats
the log-only anchor (0.0931 versus 0.1142 at rho=0.25, and 0.1016 versus
0.1234 at rho=0.50) but does not catch the direct AE (0.0691 and 0.0676).
These block-level differences are compatible with sensitivity to prior
mismatch and changes in the auxiliary-log distribution. They do not establish
prior mismatch as a demonstrated causal explanation of the relative ranking.

CLP-CSGM Ridge improves upon G(z0) in every block-seed pair and has the lower
block-mean RMSE on the two shallower blocks. On the pooled mean the direct AE
is lower. The F3 low-support advantage is not a mean win on this laboratory
target. The supplement is one cored interval and does not demonstrate
operational or cross-well generalization. The Lapa permeability case remains
a limited-sample supplement whose sparse observations are subsamples of the
same permeability series. This wording does not close a request for broader
operational validation.

Where the text was changed: Experimental setup (the laboratory-porosity
sentence); Results (the laboratory-porosity paragraph); Interpretation,
scope, and limitations (the laboratory-porosity paragraph); the Conclusion
(the laboratory-supplement sentence); and the Appendix, subsection Laboratory
porosity on a cored interval (provenance, within-window versus physical-depth
holdout, residual-entry audit counts, and non-causal block interpretation).

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

**Status:** `closed`

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
equations, experimental results, and earlier revisions were left unchanged.

Where the text was changed: Introduction (pointer to Related Work for the
recent-study comparison); Related work, subsection Petrophysical log
prediction (comparative CLP-CSGM paragraph); and Related work, subsection
Conditional generative inverse problems and positioning (closing paragraph).

---

## R1-m03 -- Figure 2 readability

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for noting that Figure 2 was useful but dense in print.
The workflow figure has been redrawn and is included at the full line width.
The compact notes of the previous schematic are now larger panel and box
labels. Offline training and target-well inference are separate panels. The
inference panel names the latent anchor, the coordinate selection
\(M_{\Omega}\), and the original-scale decoder \(G_{\mathrm{orig}}\). The
refinement box states the original-scale residual. The caption still
summarizes the same workflow and was not rewritten. At page size the panel
labels and the refinement equation remain readable, and the boxes do not
overlap.

Where the text was changed: Method, subsection Test-time reconstruction
(Figure 2).

## R1-m04 -- AE as best comparator only in this benchmark

**Status:** `closed`

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

**Deferred.** Do not create the tag during the revision. The manuscript and
the code are still changing: this round added tests (step-32 development
splits, the seed panel, the explicit-mask AE, and the well 861 laboratory
porosity check). A frozen release must match the resubmitted manuscript, so
the tag is created at the end. Computer Code Availability then cites that tag
instead of only the live repository URL.
