# Revision status -- CAGEO-D-26-01398

Working rule: one reviewer ID at a time. If the same manuscript change fully
answers more than one ID, copy the same English reply into each of those IDs
and mark them `text_done`. Do not copy a reply when the other ID still asks
for a different experiment, figure, or comparison. Every completed English
reply must name where the manuscript text changed (Abstract, section title,
or appendix subsection title). When a check is mentioned in the manuscript,
state its conclusion there, not only that the check was run. Original reviewer
text stays in `CLP_CSGM_Pareceres_Consolidados.md`.

States: `pending` | `in_progress` | `text_done` | `needs_experiment` | `needs_info` | `closed`

## Log

### R1-M01

- State: `text_done`
- Reviewer request: state early that cross-well `Vc` is Larionov-derived from
  GR and F03-4 porosity is processed/density-derived, including Abstract and
  Conclusions.
- Checked: `sections/04_data_and_eda.tex` (already explicit);
  `sections/07_discussion.tex` (already explicit);
  `sections/05_experiments.tex` (processed-curve, not independent measurement);
  `main.tex` Abstract (was missing);
  `sections/08_conclusion.tex` (mentioned future independent targets only).
- Change made: Abstract names Larionov clay volume, density-derived
  porosity, coordinate-subsampled observations, and processed curves rather
  than independent core or laboratory measurements. It also states that the
  laboratory-porosity supplement has the lower mean error for the direct
  autoencoder. Conclusion states the same F3 evidence limit and separates
  that supplement from the processed-log benchmarks. Closing Abstract
  sentence states that the benefit depends on the regime and that broader
  validation is still required.
- Files: `main.tex`; `sections/08_conclusion.tex`
- Evidence: submitted Data/Discussion already documented origin; no new
  numbers. This ID does not require a new experiment.
- Pending for this ID: none for the named Abstract/Conclusions emphasis.
- Not closed by this change: independent-measurement test (`R1-M11`);
  conventional linear comparison (`R3-02`, `R3-03`); GR/AC exclusion realism
  (`R2-03`); GR-only as operational porosity (`R1-M09`).
- Shared reply: copied verbatim to `R2-02` (same claim limit: derived
  targets, subsampled observations, processed-log reconstruction rather
  than independent measurements).

### R1-M02

- State: `text_done`
- Reviewer request: three development wells and one held-out well are a
  controlled test; language must stay benchmark-specific and must not claim
  operational cross-well transfer.
- Checked: `sections/05_experiments.tex` already named the 3+1 split;
  `sections/07_discussion.tex` asked for larger multi-well data but did not
  forbid transfer language; Abstract/Introduction/Conclusion did not state
  the 3+1 design as a claim limit.
- Change made: Abstract states three development wells and one held-out
  well. It does not name the wells and does not repeat the
  operational-transfer sentence. That limit remains in the Introduction
  and in the Conclusion. Experiments state that reported cross-well
  performance refers to this single held-out well. No new well and no new
  experiment.
- Files: `main.tex`; `sections/01_introduction.tex`;
  `sections/05_experiments.tex`; `sections/08_conclusion.tex`
- Shared reply: none. Nearby IDs (`R2-03`, `R3-02`, `R1-M03`, `R1-M11`,
  `AE-01`) ask for different justification, experiments, or a bundled AE
  synthesis.
- Pending for this ID: none for the requested language limit.

### R1-M03

- State: `text_done`
- Reviewer request: repeated resampling or extra held-out well for the
  ~27-window step-32 cell.
- Change made: 10 random development train/val splits, Omega frozen at seed
  7, F03-4 fixed, CLP Ridge vs AE. CLP lower RMSE in 10/10 splits at all
  three rho. Appendix table
  `robustness_crosswell_step32_dev_splits.tex`. Pointer in
  `sections/05_experiments.tex`. Main 9-cell CSV unchanged.
- Files: `outputs/reviewer_revision/r1m03_step32_dev_splits/`;
  `paper_clp_csgm/tables/robustness_crosswell_step32_dev_splits.csv|.tex`;
  `sections/09_appendix.tex`; `sections/05_experiments.tex`.
- Shared reply: none (no other ID asked for this split check).
- Not done: additional held-out well (the reviewer allowed resampling or
  extra wells).

### R1-M04

- State: `text_done`
- Reviewer request: increase the seed count if feasible; several n=3
  intervals include zero and the ranking changes by regime.
- Change made: 10-seed CLP Ridge vs AE on the 9-cell grid (CPU). None of
  the nine paired DeltaRMSE intervals include zero. AE still wins step 8
  at rho=0.05 and 0.10; CLP still wins the other seven cells. Appendix
  table `robustness_crosswell_seed_panel.tex`. Pointer in
  `sections/05_experiments.tex`. Main 3-seed CSVs unchanged.
- Files: `outputs/reviewer_revision/r1m04_seed_panel/`;
  `paper_clp_csgm/tables/robustness_crosswell_seed_panel.csv|.tex`;
  `sections/09_appendix.tex`; `sections/05_experiments.tex`.
- Shared reply: none (R2-04 isolates placement; R2-05 is a separate
  request for more Omega draws).
- Not done: F03-4 seed expansion; rewriting the published n=3 tables.

### R1-M05

- State: `text_done`
- Reviewer request: frame 7/9 and the GR-only reversal as a conditional
  advantage, not overall superiority; make Abstract/Highlights explicit.
- Change made: the Abstract states the seven-of-nine result as a
  conditional advantage over the direct autoencoder, not superiority over
  every comparator, with gains in the smallest training set. It states
  that interpolation is lower in all nine configurations at the reference
  noise 0.01, and that under larger perturbations that comparison changes.
  Those noise levels were used for sensitivity analysis and were not
  calibrated to field measurement uncertainties. The Abstract does not
  list steps, fractions, or the GR-only ratios. Those exceptions remain
  in the Results and in the Conclusion.
- Files: `main.tex`; `sections/08_conclusion.tex`.
- Shared reply: none. Nearby IDs ask for different experiments or claim
  limits.
- Not done: a separate Highlights file (the manuscript has none).

### R1-M06

- State: `text_done`
- Reviewer request: explain why Ridge stays the primary reference when MLP
  is better in all nine cross-well cells.
- Change made: Introduction, Method, Results, and Discussion state that
  Ridge is a fixed linear low-capacity reference, that MLP wins the
  cross-well grid and Ridge wins F03-4, and that h is chosen on validation,
  not after the test ranking. Main tables not switched to MLP.
- Files: `sections/01_introduction.tex`; `sections/03_method.tex`;
  `sections/06_results.tex`; `sections/07_discussion.tex`.
- Shared reply: none (R3 items are not the same request).

### R1-M07

- State: `text_done`
- Reviewer request: keep Proposition 3 clearly conditional in the Abstract
  and Introduction; it does not prove S-REC for the trained decoder.
- Change made: the Abstract no longer repeats the bound, because the
  journal limit is 300 words. The Introduction still states that the bound
  separates decoder error, prior mismatch, measurement noise, and local
  separability, and that it does not prove S-REC for the trained decoder.
  Method and Discussion already stated the limit and were left in place.
- Files: `sections/01_introduction.tex`.
- Shared reply: none (R3-05, R3-P02, R3-P15 are not the same request).

### R1-M08

- State: `text_done`
- Reviewer request: one stronger direct baseline that encodes the sampling
  mask or coordinates explicitly.
- Change made: supplement only. AE [u, b_placed, mask] on the 9-cell
  3-seed grid. Mask does not remove the CLP advantage at steps 16 and 32.
  Main tables not extended.
- Files: `outputs/reviewer_revision/r1m08_explicit_mask/`;
  `paper_clp_csgm/tables/robustness_crosswell_explicit_mask.csv|.tex`;
  `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/09_appendix.tex`.
- Shared reply: none. R2-04 varies Omega placement; R3-02 and R3-03 ask
  for linear regression and interpolation, not this mask AE.

### R1-M09

- State: `text_done`
- Reviewer request: GR-only porosity must not be presented as general
  porosity reconstruction performance. It is a controlled sparse-assimilation
  test under intentionally incomplete context.
- Checked: Experimental setup and Interpretation, scope, and limitations
  already call it a controlled weak-context test. The depth profile in
  Results (fig:f03_depth) already shows that CLP-CSGM follows the broad
  trend of the processed curve at rho=0.20. Abstract and Conclusion reported
  the GR-only RMSE without that claim limit.
- Change made: the Abstract no longer reports the GR-only RMSE. The
  Results depth profile now says that CLP-CSGM follows parts of the broad
  trend and smooths local variations, and that the displayed window misses
  the central depression for both CLP-CSGM and the direct autoencoder. The
  displayed parity correlations 0.4632 and 0.3720 are descriptive. The
  Conclusion keeps the limit next to the GR-only RMSE.
- Files: `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/08_conclusion.tex`; `sections/09_appendix.tex`.
- Evidence: no new numbers. The depth-profile figure was retained.
- Pending for this ID: none.
- Not closed by this change: independent core measurements (`R1-M11`);
  realism of excluding GR and acoustic travel time (`R2-03`).
- Shared reply: none.

### R1-M10

- State: `text_done`
- Reviewer request: enrich the Introduction and Discussion with recent
  well-log reconstruction, transferable subsurface learning, and data-driven
  petrophysical prediction, using the four suggested papers, and position
  CLP-CSGM against them.
- Checked: the four papers. Log completion fills the wireline curve itself
  (sequence GAN; TimeGPT, including a zero-shot basin check). Pix2Geomodel
  translates 2D geomodel property images. The IEEE Access study regresses
  porosity and shale volume from logs. None keeps a sparse target-property
  observation as an inference-time measurement.
- Change made: comparison paragraphs in Related work, including the reason
  the formulations differ (the missing values are the unknown in the four
  studies; in CLP-CSGM the sparse values are known measurements and the
  unknown is the rest of the window). One positioning sentence group in the
  Introduction; one in Interpretation, scope, and limitations. Four
  bibliography entries. Their reported errors were not copied into the
  manuscript.
- Files: `sections/01_introduction.tex`; `sections/02_related_work.tex`;
  `sections/07_discussion.tex`; `references.bib`.
- Evidence: publisher records and the PDFs in
  `docs/refs-adicionais/` (Pix2Geomodel read from the arXiv preprint that
  matches the journal article). No new experiment.
- Pending for this ID: none.
- Shared reply: none.

### R1-M11

- State: `text_done`
- Reviewer request: even a small test against independent core-derived
  porosity or permeability, or another target not generated by deterministic
  processing of the same log suite. Lapa remains limited-sample.
- Change made: well 861 laboratory porosity, 87 samples, three depth blocks,
  seeds {7, 23, 41}, rho {0.25, 0.50}. u is GR, deep and shallow resistivity,
  and lithotype. Score is held-out laboratory coordinates only. Direct AE has
  the lower mean RMSE at both ratios. CLP wins 5/9 and 6/9 splits and beats
  prior-only in 9/9. Deepest block drives the mean: higher GR, lower
  resistivity, higher Phi_lab, so the ridge anchor starts far from the test
  windows. Shallower blocks favor CLP, and rho 0.25 to 0.50 helps CLP there
  through the measurement term. Not added to the F3 grid. Main text states
  that conclusion; block RMSE, log summaries, and the gain/loss discussion
  stay in the appendix.
- Files: `scripts/run_r1m11_861_phi_lab.py`;
  `outputs/reviewer_revision/r1m11_861_phi_lab/`;
  `paper_clp_csgm/tables/robustness_phi_lab_861_heldout.tex`;
  `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/07_discussion.tex`; `sections/08_conclusion.tex`;
  `sections/09_appendix.tex`.
- Shared reply: none. Lapa is still a subsample of the same permeability
  series. R1-M01 remains the F3 target-origin statement.

### R1-m01

- State: `text_done`
- Reviewer request: shorten the title slightly, keeping "Conditional
  Latent-Prior" and "Petrophysical Target-Log Reconstruction".
- Change made: "with Application to" replaced by "for". "Generative
  Compressed Sensing" stays, because it names the method.
- Files: `main.tex` (title only).
- Shared reply: none. `R2-01` asks to move the formulation out of the
  Introduction.

### R1-m02

- State: `text_done`
- Reviewer request: tighten overlap among direct prediction, conditional
  inverse problems, and measurement consistency in the Introduction and
  Related work.
- Change made: Introduction states the measurement role once and points to
  Related work for the three-task comparison. The sentences after the
  objective no longer restate that role. Petrophysical log prediction no
  longer reopens the direct-regression contrast. The conditional-inverse
  closing states where the anchor and the residual enter, then names the
  comparators. The objective stays in the Introduction.
- Files: `sections/01_introduction.tex`; `sections/02_related_work.tex`.
- Shared reply: none. `R2-01` still asks to move the formulation and develop
  the motivation before it.

### R1-m03

- State: `pending` (deferred)
- Reviewer request: Figure 2 is useful, but the text is dense; enlarge some
  labels for print.
- Decision: leave the workflow figure unchanged for now. If revised later,
  enlarge only the smaller labels (axis names, the note under the sparse
  observations, and the \(M\) and \(\lambda\) lines), or move those notes
  into the caption. Do not redraw the flow.
- Files: none changed. Figure is `figures/pipeline_CLP-CSGM_v3.pdf`,
  included from `sections/03_method.tex`.

### R1-m04

- State: `text_done`
- Reviewer request: read "most competitive direct baseline" as specific to
  this benchmark, not as a general ranking.
- Change made: the seven-of-nine comparison is with AE [u,b]. On the
  published grid that autoencoder has the lowest unweighted mean RMSE among
  MLP, PCA, and AE. The Conclusion states that this is not a ranking against
  interpolation or the direct linear maps. The Abstract states the
  comparison as a conditional advantage and does not name that set.
  The noise sweep and the laboratory supplement are not closed by this
  wording.
- Files: `main.tex`; `sections/08_conclusion.tex`.
- Shared reply: none. Nearby IDs ask for other comparators or for limits on
  the F3 targets, not for this ranking phrase.

### R1-m05

- State: `pending` (deferred)
- Reviewer request: add a frozen release or tag that matches the manuscript,
  not only the live GitHub repository.
- Decision: leave it until the end of the revision. The code is still
  changing because this round added tests (step-32 development splits, the
  seed panel, the explicit-mask AE, and the well 861 laboratory porosity
  check). The tag must match the resubmitted manuscript. Then Computer Code
  Availability cites that tag.
- Files: none changed.

### R2-01

- State: `text_done`
- Reviewer request: develop the practical motivation before the formulation,
  and move the mathematical details from the Introduction to Method.
- Change made: Introduction now states the well setting, why a direct map is
  not enough, and what CLP-CSGM does, in prose. The objective equation was
  removed. Method, subsection Test-time reconstruction, already contains it;
  the Introduction points there. Ridge stays the linear reference for the
  starting map.
- Files: `sections/01_introduction.tex`.
- Shared reply: none. `R1-m02` only reduced overlap and left the equation in
  the Introduction.

### R2-03

- State: `text_done`
- Reviewer request: the exclusion of gamma ray from clay volume and of sonic
  from F03-4 porosity looks artificial; say what field situation each
  represents and how far the conclusions apply.
- Change made: both exclusions are explained as leaving out the log that
  already determines the target, not as the suite for a fully logged well.
  Gamma ray back in the appendix lowers the error and does not change the
  method order at step 16. F03-4 gamma-ray-only errors do not transfer to a
  density-neutron-sonic porosity suite. No new experiment.
- Files: `sections/04_data_and_eda.tex`; `sections/05_experiments.tex`;
  `sections/07_discussion.tex`.
- Shared reply: none. `R1-M09` limits the gamma-ray-only porosity claim.
  `R1-M11` is the separate laboratory-porosity check.

### R2-04

- State: `text_done`
- Reviewer request: the conclusions may depend on one coordinate set per
  seed and measurement ratio. Additional sampling realizations or a
  placement-sensitivity analysis would strengthen the evaluation.
- Change made: five Omega draws on the cross-well Vc grid, model seeds
  fixed at 7, 23, 41. Draw 0 is the published stream. Full-window RMSE.
  Cell winner matches the published ranking in 9/9 cells. CLP lower in
  15/15 comparisons at steps 16 and 32. At step 8, AE is lower in 15/15
  at rho=0.05 and in 10/15 at rho=0.10; CLP is lower in 11/15 at
  rho=0.20. Main 3-seed tables unchanged.
- Files: `outputs/reviewer_revision/r2m04_omega_placement/`;
  `paper_clp_csgm/tables/robustness_crosswell_omega_placement.csv|.tex`;
  `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/09_appendix.tex`.
- Shared reply: none. `R1-M04` changes seeds as well as Omega. `R1-M08`
  adds a mask and keeps Omega fixed. `R2-05` still asks for more
  realizations.
- Not done: F03-4 placement sweep; merging the draws into the main tables.

### R2-05

- State: `text_done`
- Reviewer request: three seeds give limited evidence when some gaps are
  small. Add realizations, particularly of the random sparse-observation
  sampling.
- Change made: no new run. The uncertainty paragraph now reads the ten-seed
  panel and the five-placement panel together. Ten seeds refit the models
  and redraw Omega; none of the nine intervals include zero. Five placements
  hold seeds 7, 23, 41 fixed; the steps 16 and 32 advantage holds in every
  draw, and step 8 is less stable. Overlapping windows are not counted as
  extra independent samples. Main 3-seed tables unchanged.
- Files: `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/09_appendix.tex`.
- Shared reply: none. `R1-M04` is the ten-seed panel alone. `R2-04` is the
  placement panel alone. This reply uses both and does not copy either.
- Not done: a new factorial of ten seeds by five placements.

### R3-01

- State: `text_done`
- Reviewer request: the writing stacks terminology. The abstract examples
  are admissible reconstruction range, low-capacity latent-prior reference,
  contextual latent anchoring, and sparse-measurement consistency.
- Change made: those labels are replaced by the operations, and a later
  wording pass states both roles of z_0. The decoder supplies the windows
  the reconstruction may use. The conditional map predicts z_0 from the
  dense logs. The first trajectory starts at z_0; the other two start at
  z_0 plus N(0, 0.05^2 I). The penalty center remains z_0 on every restart
  and every iteration. The measurement term is a penalty, so a residual can
  remain. The decoder range and that penalty are separate. Prior-only is
  the decoded prediction G(z_0), with no latent search. Ridge is the linear
  reference for z_0. This pass clarifies the existing formulation. The
  equations, the algorithm, and the reported numbers are unchanged.
  R3-P01, R3-P03, and R3-P06 stay pending.
- Files: `main.tex`; `sections/01_introduction.tex`;
  `sections/02_related_work.tex`; `sections/03_method.tex`;
  `sections/04_data_and_eda.tex`; `sections/05_experiments.tex`;
  `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/08_conclusion.tex`; `sections/09_appendix.tex`.
- Shared reply: none. `R3-P01`, `R3-P03`, and `R3-P06` annotate nearby
  abstract sentences and stay in the queue.
- Not done: a rewrite of the proofs, or closing the PDF annotations.

### R3-02

- State: `text_done`
- Reviewer request: clay volume looks linear in gamma ray and porosity looks
  linear in density or slowness, so why use this workflow instead of the
  conventional formula.
- Change made: text only. When the log the formula needs is logged, that
  formula is the calculation and CLP-CSGM is not a substitute. The gamma-ray
  index is linear between endpoints; the Larionov map is not a straight line.
  Density porosity is linear in bulk density and a standard sonic porosity is
  linear in slowness. The F3 tests remove the log the formula needs, so the
  formula cannot be evaluated. No new experiment.
- Files: `sections/04_data_and_eda.tex`; `sections/05_experiments.tex`;
  `sections/07_discussion.tex`.
- Shared reply: none. `R2-03` explains the withheld logs. `R3-03` asks for
  direct linear regression and interpolation.
- Not done: a comparison against Larionov or density porosity on a full log
  suite.

### R3-03

- State: `text_done`
- Reviewer request: add direct linear or ridge regression and interpolation
  of the sparse target, and explain the encoder and the intermediate
  reconstruction for a window of 64 values.
- Change made: the encoder is training-only; inference does not encode the
  unknown window; k=16 is the code dimension. Retrospective G(E(y)) test
  RMSE is 0.0073, 0.0099, and 0.0110 at steps 8, 16, and 32, and is not
  ranked as a predictor. At published noise 0.01, depth interpolation has
  the lowest RMSE in 9/9 cells and 27/27 seed comparisons. CLP-CSGM Ridge
  is lower than both direct linear maps at step 16, rho 0.20, and at step
  32 for every rho. The held-out Vc standard deviation is 0.035. At noise
  0.05, CLP-CSGM Ridge is lower in every seed at steps 16 and 32 and at
  step 8 for rho 0.20, with the selected penalty inside the published grid.
  At 0.10 and 0.20 it is lower in all nine cells. These noise levels were
  used for sensitivity analysis and were not calibrated to field
  measurement uncertainties. At 0.20 some selected weights lie above the
  published grid. Results includes an RMSE-versus-noise figure with one
  panel per step and one curve per measurement fraction. Relative to the
  held-out standard deviation 0.035, noise 0.05, 0.10, and 0.20 are about
  1.43, 2.86, and 5.71 times larger. The figure axis is logarithmic. The decoder and the dense-log map
  are fit once per step and seed; the penalty is selected again for each
  measurement fraction and noise level. Noise 0.01 in the figure is that
  refit: at step 16, rho 0.20, the published CLP mean is 0.01771 and the
  refit mean is 0.0159, with selected weight 0.0001. The ordering at 0.01
  is unchanged. The step-8 pattern is compatible with a weak dense-log
  window and does not isolate that window as the only cause. The depth
  figure uses noise 0.01 and 0.05 for one interval (seed 7, step 32, rho
  0.20). The main nine-cell grid is unchanged. At step 8, noise 0.05, rho
  0.05 and 0.10, the dense-log window has RMSE 0.0564, above interpolation
  at 0.0456 and 0.0442, while G(E(y)) is 0.0073. CLP stays between those
  values at 0.0508 and 0.0456. At rho 0.20, with thirteen measurements, the
  CLP mean is 0.0372 and interpolation is 0.0423.
- Files: `sections/03_method.tex`; `sections/05_experiments.tex`;
  `sections/06_results.tex`; `sections/09_appendix.tex`;
  `tables/robustness_crosswell_simple_baselines.tex`;
  `tables/robustness_crosswell_noise_sweep.tex`;
  `figures/crosswell_vc_noise_panels_step32_rho020_seed7.png`;
  `figures/crosswell_vc_rmse_vs_noise.png`.
- Shared reply: none. `R3-04` asks how the published noise was chosen and
  stays in the queue.

## Queue (not yet processed)

R3-04 .. R3-06,
R3-A01 .. R3-A05, R3-P01 .. R3-P19, AE-01.
R1-m03 and R1-m05 remain deferred.
