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
  well and does not name them. Introduction names the three-plus-one design
  and F03-4. Experiments state that reported cross-well performance refers
  to this single held-out well. The explicit limit, that the comparison is
  not evidence of operational cross-well or multi-field transferability, is
  in the Conclusion. No new well and no new experiment.
- Files: `main.tex`; `sections/01_introduction.tex`;
  `sections/05_experiments.tex`; `sections/08_conclusion.tex`
- Shared reply: none. Nearby IDs (`R2-03`, `R3-02`, `R1-M03`, `R1-M11`,
  `AE-01`) ask for different justification, experiments, or a bundled AE
  synthesis.
- Pending for this ID: none for the requested language limit.

### R1-M03

- State: `text_done`
- Reviewer request: repeated resampling or extra held-out well for the
  step-32 cell. The development pool has 30 windows; 27 training and 3
  validation remain after the 10 percent reservation. The earlier "about 27
  before the split" was the training count.
- Change made: 10 random development train/val splits, Omega coordinates
  drawn from seed 7, F03-4 fixed, CLP Ridge vs AE. Per-split summaries
  reproduce the appendix means (CLP 0.0342428731, 0.0305056174, 0.0187939244
  vs AE 0.0406650491, 0.0388075101, 0.0361629459) and CLP is lower in 10/10
  splits at all three rho. The saved runs do not contain `omega_sets.csv`;
  the coordinate draw is fixed by seed 7 before the noise stream, while the
  same noise rows are reassociated when a development window changes role.
  In each split, all 3 validation windows share a depth sample with a
  training window of the same well (30 of 30). F06-1 has 2 windows and stays
  in training. The splits test the development partition, not independent
  validation and not ten wells. Appendix table
  `robustness_crosswell_step32_dev_splits.tex`. Counts corrected in
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
- Change made: 10-seed CLP Ridge vs AE on the 9-cell grid (CPU). The 90
  paired comparisons were recomputed from the three `summary_by_seed.csv`
  files and match the appendix table. None of the nine Student-t intervals
  (df=9) include zero. AE still wins step 8 at rho=0.05 (10/10) and
  rho=0.10 (8/10); CLP wins the other seven cells, including step 8 at
  rho=0.20 (7/10) and step 16 at rho=0.05 (9/10). Saved runs have no
  `omega_sets.csv`; each seed redraws Omega and also refits the models, so
  the panel does not isolate placement. The intervals are seed variability
  on held-out F03-4, not cross-well generalization and not a simultaneous
  test. Appendix table `robustness_crosswell_seed_panel.tex`. Pointer in
  `sections/05_experiments.tex`. Main 3-seed CSVs unchanged. Wording only:
  the n=3 interval sentence now names the main nine-cell benchmark, and the
  step-8 appendix sentence calls 104 the training count.
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
  every comparator, with the gains chiefly where training support is
  limited and with losses in two cells. It now names the GR-only
  observation fractions: CLP lower at rho 0.20 and 0.30, nearly equal at
  rho 0.50, and AE lower at rho 0.40 and 0.60. It states that
  interpolation is lower in all nine configurations at the reference
  noise 0.01, and that under larger perturbations that comparison changes.
  Those noise levels were used for sensitivity analysis and were not
  calibrated to field measurement uncertainties.
- Files: `main.tex`; `sections/08_conclusion.tex`; `highlights.txt`.
- Shared reply: none. Nearby IDs ask for different experiments or claim
  limits.
- Not done: none in the manuscript body. Submission Highlights are the
  five lines in `highlights.txt` (85 characters or fewer). They were not
  inserted into the PDF.

### R1-M06

- State: `closed`
- Reviewer request: explain why Ridge stays the primary reference when the
  earlier comparison gave MLP a lower mean RMSE in every cross-well cell.
- Inconsistency: the published cross-well MLP run added gamma ray, which
  Ridge did not use, and the published F03-4 MLP run split windows before
  depth rows (275 test windows versus 224). Those contrasts do not isolate
  the class of h.
- Controlled protocol: cross-well development F02-1, F03-2, and F06-1, test
  F03-4, channels sonic, rhob, ai, vp, steps 8, 16, and 32, seeds 7, 23, and
  41, same tail split and the same sparse observations. F03-4 GR-only uses
  `data/F03-4_AC+GR+Porosity.txt`, GR only, depth-first split, and
  798/224/224 windows for both priors.
- Controlled result: unweighted cross-well RMSE 0.028623668 (Ridge) and
  0.025383384 (MLP), a 11.3% reduction. MLP is lower in 7 of 9 cells and in
  23 of 27 seed comparisons. Ridge is lower at step 32 for rho 0.10 and
  0.20. On GR-only, Ridge is lower at all five ratios and in all 15 seed
  comparisons, with relative reductions of about 15.6% to 19.7%.
- Valid conclusion: the prior class changes accuracy, the preferred class
  depends on the regime, and Ridge stays the fixed linear low-capacity
  reference. The class is chosen on validation, not from the test ranking.
  The main Ridge versus direct autoencoder comparison was not replaced.
- Artifacts: `outputs/reviewer_revision/r1m06_prior_class_controlled/`;
  `scripts/clp_csgm_r1m06_controlled_assets.py`;
  `paper_clp_csgm/tables/ablation_prior_class_crosswell_summary.csv`,
  `ablation_prior_class_crosswell_wide.csv`,
  `ablation_prior_class_crosswell_wide.tex`,
  `ablation_prior_class_f03_summary.csv`,
  `ablation_prior_class_f03_wide.csv`,
  `ablation_prior_class_f03_wide.tex`;
  `paper_clp_csgm/figures/ablation_prior_class_ridge_vs_mlp.png`.
- Files: `sections/01_introduction.tex`; `sections/03_method.tex`;
  `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/08_conclusion.tex`; `docs/revision/respostas_R1.md`.
- Shared reply: none (R3 items are not the same request).

### R1-M07

- State: `closed`
- Reviewer request: keep Proposition 3 clearly conditional in the Abstract
  and Introduction; it does not prove S-REC for the trained decoder.
- Audit: Method, appendix proof, S-REC diagnostics, Discussion, and the
  Introduction already separate the assumed local separability from a
  guarantee for the trained decoder. The Abstract omitted that limit.
- Change made: Abstract adds one sentence, "The local recovery bound
  remains conditional and does not certify S-REC." Introduction kept.
  Method and Discussion left in place. No new experiment or proof.
- Word count: Abstract rendered words about 285 before and 296 after,
  under the 300-word journal limit.
- Files: `main.tex`; `sections/01_introduction.tex`;
  `docs/revision/respostas_R1.md`.
- Shared reply: none (R3-05, R3-P02, R3-P15 are not the same request).

### R1-M08

- State: `closed`
- Reviewer request: one stronger direct baseline that encodes the sampling
  mask or coordinates explicitly.
- Problem found: the first masked AE scaled the full [u, b_placed, mask]
  vector, so the constant Omega mask became zero, and it used seed+100.
- Correction: controlled rerun preserves the binary mask without scaling,
  uses the same base seed as the direct AE, and keeps paired Omega and
  noise. Historical runs under r1m08_explicit_mask/ are retained but are
  no longer the published source.
- Controlled result: CLP Ridge has lower mean RMSE than the mask-aware AE
  in 7 of 9 cells. At steps 16 and 32 it wins all six cell means and 17
  of 18 seed comparisons. Exceptions remain step 8 at rho 0.05 and 0.10.
  Causal isolation of the latent inverse is not claimed because Omega is
  fixed and the mask is constant across windows.
- Files: `outputs/reviewer_revision/r1m08_mask_controlled/`;
  `scripts/clp_csgm_r1m08_explicit_mask_assets.py`;
  `paper_clp_csgm/tables/robustness_crosswell_explicit_mask.csv|.tex`;
  `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/09_appendix.tex`; `docs/revision/respostas_R1.md`.
- Shared reply: none. R2-04 varies Omega placement; R3-02 and R3-03 ask
  for linear regression and interpolation, not this mask AE.

### R1-M09

- State: `closed`
- Reviewer request: GR-only porosity must not be presented as general
  porosity reconstruction performance. It is a controlled sparse-assimilation
  test under intentionally incomplete context.
- Checked: Experimental setup keeps GR as the only dense log and reports
  sonic-target Pearson r about 0.9996. Discussion keeps processed-target
  limits. Conclusion keeps the non-general porosity claim. Abstract has no
  new generalization or significance wording. Figures and RMSE numbers were
  not changed.
- Change made: two Results sentences now report the observed RMSE advantage
  of CLP-CSGM Ridge over the direct AE at the sparse end of the controlled
  GR-only grid, instead of attributing that gap to isolated latent
  refinement. Earlier Abstract/Conclusion/depth-profile limits remain.
- Files: `sections/06_results.tex`; `docs/revision/respostas_R1.md`.
- Evidence: no new numbers, tables, or figures.
- Pending for this ID: none.
- Not closed by this change: independent core measurements (`R1-M11`);
  realism of excluding GR and acoustic travel time (`R2-03`).
- Shared reply: none.

### R1-M10

- State: `closed`
- Reviewer request: enrich the Introduction and Discussion with recent
  well-log reconstruction, transferable subsurface learning, and data-driven
  petrophysical prediction, using the four suggested papers, and position
  CLP-CSGM against them.
- Checked: DOIs match the four bibliography entries
  (engeos.2026.100536; geoen.2025.214342; s41598-025-95709-0;
  ACCESS.2023.3336623). Introduction, Related work, and Discussion already
  cite all four. No general superiority claim and no numerical comparison
  with those studies.
- Change made: Related work now states that log completion uses available
  observations and cross-channel context, and contrasts that mapping task
  with conditional latent optimization under an explicit measurement penalty.
  Discussion adds that TimeGPT-style basin transfer is broader than the
  single held-out F03-4 protocol and that basin-agnostic transferability is
  not established here.
- Files: `sections/02_related_work.tex`; `sections/07_discussion.tex`;
  `docs/revision/respostas_R1.md`.
- Evidence: no new references, experiments, tables, or figures.
- Pending for this ID: none.
- Shared reply: none.

### R1-M11

- State: `closed`
- Reviewer request: even a small test against independent core-derived
  porosity or permeability, or another target not generated by deterministic
  processing of the same log suite. Lapa remains limited-sample.
- Audit completed: Phi_lab is identified as laboratory porosity in the Auddys
  Logs source and is not computed from GR/resistivity/lithotype by the
  pipeline. Primary per-plug assay ledgers were not found. Train/test
  depth-block separation and within-window relative holdout are confirmed.
  Global physical-depth holdout is not satisfied. Quantitative results are
  unchanged. Prior-mismatch explanations are treated as hypotheses, not
  demonstrated causes; rho 0.25 versus 0.50 comparisons do not isolate
  measurement count because Omega is redrawn.
- Change made: final Option 1 wording. Appendix/main text/reply state
  provenance limits, within-window scoring on depth-held-out blocks, dual-role
  residual-entry counts (21--27 depths; 84--100%), and non-causal block
  interpretation. Direct AE remains lower on the mean; CLP remains better
  than prior-only in all block-seed pairs.
- Files: `outputs/reviewer_revision/r1m11_861_phi_lab/AUDITORIA_INDEPENDENCIA_R1M11.md`;
  `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/07_discussion.tex`; `sections/08_conclusion.tex`;
  `sections/09_appendix.tex`; `docs/revision/respostas_R1.md`.
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

- State: `closed`
- Reviewer request: tighten overlap among direct prediction, conditional
  inverse problems, and measurement consistency in the Introduction and
  Related work.
- Change made: minimal editorial pass. Introduction points to Related work
  for the recent-study comparison without restating the trained-map contrast.
  Petrophysical log prediction keeps the four R1-M10 references and shortens
  the CLP-CSGM positioning paragraph. Conditional-inverse closing no longer
  repeats restart/initialization details kept in Method. Equations, results,
  and R1-M07--M11 wording were preserved. R1-m05 left untouched.
- Files: `sections/01_introduction.tex`; `sections/02_related_work.tex`;
  `docs/revision/respostas_R1.md`.
- Shared reply: none. `R2-01` still asks to move the formulation and develop
  the motivation before it.

### R1-m03

- State: `text_done`
- Reviewer request: Figure 2 is useful, but the text is dense; enlarge some
  labels for print.
- Change made: the workflow drawing was replaced and included at full line
  width. Compact notes are now larger panel and box labels. Offline training
  and target-well inference are separate panels. The inference panel names
  the latent anchor, \(M_{\Omega}\), and \(G_{\mathrm{orig}}\), and the
  refinement box states the original-scale residual. The Figure 2 caption
  wording is unchanged.
- Checked: compiled page 12. Panel labels and the refinement equation are
  readable. The boxes do not overlap.
- Files: `sections/03_method.tex`; `figures/pipeline_CLP-CSGM_v4.png`.
- Not done: none from this entry. `R3-06` is closed. This entry
  does not close `R3-P08`.

### R1-m04

- State: `closed`
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

- State: `closed`
- Reviewer request: develop the practical motivation before the formulation,
  and move the mathematical details from the Introduction to Method.
- Verified after R1-m02: Introduction keeps practical motivation and a
  conceptual CLP-CSGM description without the objective equation. Related Work
  holds the R1-M10/R1-m02 positioning. Method, subsection Test-time
  reconstruction, still contains `eq:clp_csgm_objective`. Motivation explains
  why sparse observations are treated as measurements and does not claim
  general inadequacy of direct methods or universal superiority of CLP-CSGM.
- Change made: documentation alignment only for this closing check. Manuscript
  structure was already in place.
- Files: `docs/revision/respostas_R2.md` (reply wording); manuscript files
  unchanged in this step.
- Shared reply: none. `R1-m02` reduced Introduction/Related Work overlap and
  left the objective in Method.

### R2-02

- State: `text_done`
- Reviewer request: state that the main targets are processed or derived,
  not independently measured ground truth, and that sparse observations are
  subsamples of those same curves.
- Checked: the Abstract (`main.tex`); Data and exploratory analysis
  (`sections/04_data_and_eda.tex`); Interpretation, scope, and limitations,
  the Discussion (`sections/07_discussion.tex`); and the Conclusion
  (`sections/08_conclusion.tex`). All four already state the processed
  origin of the F3 targets. Clay volume is Larionov-derived from gamma ray,
  F03-4 porosity is density-derived, and sparse observations are coordinate
  subsamples of those curves.
- Change made: none in the manuscript. This task only replaces the closing
  sentence of the shared reply. The laboratory-porosity supplement under
  `R1-M11` is a limited check and does not demonstrate operational transfer.
- Files: `docs/revision/respostas_R1.md`; `docs/revision/respostas_R2.md`;
  `docs/revision/status.md`. No `.tex`, table, or result file was changed.
- Shared reply: identical to `R1-M01`.

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
- Audit: Data and exploratory analysis and Experimental setup already justify
  the gamma-ray and sonic exclusions. Discussion already said the benchmarks
  are not a fully logged suite and not a test against independent core or
  laboratory measurements. That text did not name the field situation the
  restrictions resemble.
- Added: one paragraph in `sections/07_discussion.tex`, after the two-benchmark
  scope paragraph. The analogue is a missing or unreliable conventional log
  plus a few independently acquired measurements. F3 reproduces only the
  restricted-input and sparse-observation structure, because its sparse values
  are subsamples of processed curves. No new experiment. Results and tables
  unchanged. Exploratory permeability runs were not used.
- Files: `sections/04_data_and_eda.tex`; `sections/05_experiments.tex`;
  `sections/07_discussion.tex`; `docs/revision/respostas_R2.md`;
  `docs/revision/status.md`.
- Shared reply: none. `R1-M09` limits the gamma-ray-only porosity claim.
  `R1-M11` is the separate laboratory-porosity check. `R2-02` was not edited.

### R2-04

- State: `closed`
- Reviewer request: the conclusions may depend on one coordinate set per
  seed and measurement ratio. Additional sampling realizations or a
  placement-sensitivity analysis would strengthen the evaluation.
- Change made: five Omega draws on the cross-well Vc grid, model seeds
  fixed at 7, 23, 41. Draw 0 is the published stream. Full-window RMSE.
  Cell winner matches the published ranking in 9/9 cells. CLP lower in
  15/15 comparisons at steps 16 and 32. At step 8, AE is lower in 15/15
  at rho=0.05 and in 10/15 at rho=0.10; CLP is lower in 11/15 at
  rho=0.20. Main 3-seed tables unchanged.
- Audit: `sir_cs_benchmark_direct_ub.py` draws Omega and the noise on b
  from one generator seeded by `model_seed + measurement_seed_offset`.
  `scripts/run_r2m04_omega_placement.sh` passes offsets 0, 10007, 20011,
  30013, and 40009 while `--seeds` stays 7,23,41. The decoder, ridge
  anchor, and direct AE keep those model seeds. A new draw therefore
  changes coordinates and noise together and does not isolate position
  from noise. The 15 comparisons are 3 seeds times 5 draws on held-out
  F03-4, not 15 independent wells and not cross-well generalization.
- Files: `outputs/reviewer_revision/r2m04_omega_placement/`;
  `paper_clp_csgm/tables/robustness_crosswell_omega_placement.csv|.tex`;
  `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/09_appendix.tex`; `docs/revision/respostas_R2.md`;
  `docs/revision/status.md`.
- Shared reply: none. `R1-M04` changes seeds as well as Omega. `R1-M08`
  adds a mask and keeps Omega fixed. `R2-05` still asks for more
  realizations.
- Not done: F03-4 placement sweep; merging the draws into the main tables.
  No new experiment in this audit. Tables and figures unchanged.

### R2-05

- State: `closed`
- Reviewer request: three seeds give limited evidence when some gaps are
  small. Add realizations, particularly of the random sparse-observation
  sampling.
- Method note: the ten-seed panel refits the models and redraws coordinates
  and measurement noise together. The five-draw panel holds seeds 7, 23, and
  41 fixed and also varies coordinates and noise together. It does not isolate
  placement from noise. Neither panel adds independent wells or a joint test
  of the nine cells. Nine paired 95 percent intervals with df=9 exclude zero.
  CLP remains lower at steps 16 and 32. Step 8 still favors AE at rho 0.05
  and 0.10. The five-draw counts stay those of R2-04. No claim of general
  superiority, including against reference-noise interpolation.
- Change made: wording only. Numbers, tables, and intervals unchanged.
- Files: `sections/05_experiments.tex`; `sections/06_results.tex`;
  `sections/09_appendix.tex`; `docs/revision/respostas_R2.md`.
- Shared reply: none. `R1-M04` is the ten-seed panel alone. `R2-04` is the
  placement panel alone. This reply uses both and does not copy either.
- Not done: a new factorial of ten seeds by five placements.

### R3-01

- State: `closed`
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
  `R3-P01` is closed. `R3-P03` and `R3-P06` stay pending.
- Files: `main.tex`; `sections/01_introduction.tex`;
  `sections/02_related_work.tex`; `sections/03_method.tex`;
  `sections/04_data_and_eda.tex`; `sections/05_experiments.tex`;
  `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/08_conclusion.tex`; `sections/09_appendix.tex`.
- Checked: Abstract no longer uses the four stacked labels and describes the
  decoder, z_0, and the measurement balance. z_0 initializes the first
  trajectory, the other restarts add N(0, 0.05^2 I), and the penalty center
  stays z_0. Prior-only is G(z_0) with no search. Measurement-only sets
  lambda to 0. Scope sentences in the Abstract, Introduction, Method,
  Experimental setup, and Discussion match the reply. No manuscript edit in
  this closure.
- Shared reply: none. `R3-P01` is closed. `R3-P03` and `R3-P06` annotate
  nearby abstract sentences and stay pending.
- Not done: a rewrite of the proofs, or closing the PDF annotations.

### R3-02

- State: `closed`
- Reviewer request: clay volume looks linear in gamma ray and porosity looks
  linear in density or slowness, so why use this workflow instead of the
  conventional formula.
- Change made: text only. When the log the formula needs is logged, that
  formula is the calculation and CLP-CSGM is not a substitute. The gamma-ray
  index is linear between endpoints; the Larionov map is not a straight line.
  Density-derived porosity is calculated from bulk density and its hypotheses.
  Sonic porosity is a different relation and is not the same quantity. In
  F03-4 the density behind the processed target was predicted from sonic and
  gamma ray, so the sonic correlation is not laboratory validation. The reply
  uses the published R3-03 interpolation result: lower RMSE than CLP-CSGM in
  all nine cells at the reference noise, with larger perturbations uncalibrated
  to field uncertainty. No new experiment. Wells 1254 and 861 permeability
  explorations were not used. `R2-02`, `R2-03`, and `R3-03` were not reopened.
- Files: `sections/05_experiments.tex`; `sections/07_discussion.tex`;
  `docs/revision/respostas_R3.md`; `docs/revision/status.md`.
- Shared reply: none. `R2-03` explains the withheld logs. `R3-03` reports the
  interpolation comparison used here.
- Not done: a comparison against Larionov or density porosity on a full log
  suite.

### R3-03

- State: `closed`
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
  CLP mean is 0.0372 and interpolation is 0.0423. A later editorial
  pass replaces the Results explanation that attributed the smaller RMSE
  increase to the dense-log window alone. The revised text cites Hansen
  (2010) for the role of regularization and Bora et al. (2017), already in
  the bibliography, as context rather than a recovery guarantee for these
  coordinate masks. The sweep is not said to isolate the decoder
  restriction, the conditional penalty, or validation-based weight
  selection. A later pass adds the validation-selected second-difference
  smoother and the affine limit. The lambda-grid notes stay: the
  sweep grid differs from the published grid; one seed selects 1 at noise
  0.10; some cells select 100 at noise 0.20, the largest candidate tried.
  The added noise remains distinct from the 30dB label. Previous numerical
  results were not changed.
- Files: `sections/03_method.tex`; `sections/05_experiments.tex`;
  `sections/06_results.tex`; `sections/08_conclusion.tex`;
  `sections/09_appendix.tex`; `references.bib`;
  `tables/robustness_crosswell_simple_baselines.tex`;
  `tables/robustness_crosswell_noise_sweep.tex`;
  `tables/robustness_crosswell_smooth.tex`;
  `figures/crosswell_vc_noise_panels_step32_rho020_seed7.png`;
  `figures/crosswell_vc_rmse_vs_noise.png`;
  `figures/crosswell_vc_smooth_vs_clp.png`;
  `figures/crosswell_vc_smooth_affine_gap.png`.
- Shared reply: none. `R3-04` is answered in its own entry. This item
  does not close it.
- Affine audit: alpha=1e6 matches the selected ceiling scores to 1e-16.
  On the 62 ceiling rows the median profile RMSE versus the affine fit is
  7.6e-5 and the largest scored-RMSE gap is 1.9e-5. The original smoother
  table was not overwritten. Affine cell means favor that fit over CLP in
  9/9 cells at noise 0.01, 4/9 at 0.05, 1/9 at 0.10, and 0/9 at 0.20.
  Selected smoothing favors the smoother in 8/9, 4/9, 1/9, and 0/9.
- Not done: a field-calibrated noise model; a definition of the 30dB label
  (`R3-P16`).

### R3-04

- State: `closed`
- Reviewer request: how the noise level is chosen, and how that choice
  balances the prior against the data fit.
- Change made: text only. After coordinate subsampling, independent
  Gaussian noise is added with standard deviation 0.01 on cross-well Vc
  and 0.02 on F03-4 porosity, in the units of the target curve. Relative
  to the held-out Vc standard deviation of about 0.035, 0.01 is about
  0.29 times that deviation. These levels were adopted for controlled
  benchmarks. No calibration to instrumental uncertainty is documented.
  The 30dB filename label is not this added standard deviation, and the
  procedure behind the label is not defined here. Every method in a run
  uses the same noisy observations. Direct [u,b] models standardize after
  the noise is added. The implemented data term is the squared residual in
  the original target units. It equals the standardized residual weighted
  by D_Omega, the diagonal of the training standard deviations at the
  observed coordinates. On the cross-well Vc training windows those
  standard deviations are not equal: the largest-to-smallest ratio is about
  1.13 at step 8, 1.23 at step 16, and 1.40 at step 32. The displayed
  objective, the measurement-only ablation, and the existence objective now
  use that weighted residual. The MAP identity lambda = sigma_eta^2 /
  sigma_z^2 applies to the isotropic likelihood in original units. It does
  not apply to the unweighted standardized residual. The reported runs
  select lambda by validation full-window RMSE on the published grid and
  freeze it before the test score. Dividing both terms by the number of
  windows multiplies the objective by one positive factor and leaves the
  minimizers unchanged. The sensitivity sweep uses a different, wider grid:
  1e-4, 1e-3, 1e-2, 1e-1, 1, 10, and 100. It omits the factor-of-three
  values from the published grid. At 0.01 and 0.05 every selected weight
  also belongs to the published grid. That membership does not mean the two
  searches were the same. At noise 0.10, step 16 and rho 0.10, one seed
  selects the weight 1. At noise 0.20 some cells select 100, the largest
  candidate that was tried. That choice is not an optimum outside the
  grid. The algorithm and the reported numbers are unchanged. The Results
  noise discussion now separates the observed RMSE ordering from a single
  cause. It cites Hansen (2010) and the existing Bora et al. (2017) entry
  as context, and it states that the sweep does not isolate the decoder
  restriction, the conditional penalty, or the validation choice of the
  weight. The noise sweep is now read as relative robustness of full-window
  RMSE for the complete procedure, with lambda reselected at each noise
  level. Interpolation is lower in 9/9 cells at noise 0.01. CLP is lower
  than interpolation in 7/9, 9/9, and 9/9 cells at 0.05, 0.10, and 0.20,
  and lower than the R3-03 smoother in 5/9, 8/9, and 9/9. The reference
  noise still favors interpolation and, in 8/9 cells, smoothing. No new
  experiment. R3-P09 and R3-P10 were not edited. R3-P16 stays pending.
- Files: `sections/03_method.tex`; `sections/05_experiments.tex`;
  `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/09_appendix.tex`; `references.bib`;
  `tables/implementation_summary.tex`.
- Shared reply: the same evidence is restated under `R3-P09`. `R3-P10`
  is answered in its own entry. `R3-P16` stays pending.
- Not done: a field-calibrated noise model and a definition of the 30dB
  label.

### R3-P09

- State: `text_done`
- Reviewer request: how the measurement noise is set, and how lambda is
  determined.
- Change made: the same text as `R3-04`. The objective now states the
  identity between the original-scale residual and the standardized residual
  weighted by D_Omega. The MAP ratio is limited to the original-scale
  likelihood. The sweep is described as a different, wider grid. A common
  positive factor is said to leave the minimizers unchanged.
- Files: `sections/03_method.tex`; `sections/05_experiments.tex`;
  `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/09_appendix.tex`; `tables/implementation_summary.tex`.
- Shared reply: same evidence as `R3-04`, written as its own entry.
- Not done: `R3-P16` (origin of 30dB).

### R3-P10

- State: `text_done`
- Reviewer request: the annotation "covariance?" on the standardization
  paragraph, at the standard deviations and the sparse-measurement scaling.
- Change made: text only. D_y and D_u are per-coordinate scales, not
  estimated profile covariances. The added benchmark noise is isotropic in
  the original target units. After the diagonal standardization its
  covariance is sigma_eta^2 D_Omega^{-2}: still independent, with unequal
  variances when the training standard deviations differ. The likelihood
  quadratic form is the implemented data term divided by sigma_eta^2.
  sigma_z^2 I remains the MAP hypothesis for the Euclidean penalty, not an
  estimated Ridge-residual covariance. G_orig denotes the decoder output in
  original units. The local-recovery constants stay on that image, and the
  separability ratios, already computed there, are not those constants.
  No new experiment.
- Files: `sections/03_method.tex`; `sections/07_discussion.tex`;
  `sections/09_appendix.tex`.
- Shared reply: none. `R3-04` and `R3-P09` now point to this entry.
  `R3-P16` stays pending.
- Not done: a definition of the 30dB label, and an estimated latent
  covariance.

### R3-A03

- State: `closed`
- Reviewer request: redraw Figure 1 so the mappings from \(z_0\) to
  \(G(z_0)\) and from the refined code to its reconstruction are
  unambiguous, and explain the arrows and the decoder-range curve.
- Change made: the figure shows two horizontal decoding arrows, from
  \(z_0\) to \(G_{\mathrm{orig}}(z_0)\) and from \(\widehat{z}\) to
  \(G_{\mathrm{orig}}(\widehat{z})\), both on the blue curve
  \(\mathcal{R}_{\mathrm{orig}}\). The vertical arrow is latent
  refinement. The caption states that the drawing is schematic and is not
  a measured numerical trajectory.
- Checked: compiled page 11. Both codes meet the curve. Labels, including
  \(M_{\Omega}\widehat{y}\) and the footer, are readable and do not overlap.
- Checked on close: the local PDF shows \(z_0\) decoded to
  \(G_{\mathrm{orig}}(z_0)\) and \(\widehat{z}\) decoded to
  \(\widehat{y}=G_{\mathrm{orig}}(\widehat{z})\). Both outputs lie on the
  blue curve \(\mathcal{R}_{\mathrm{orig}}\). The vertical arrow is latent
  refinement. The caption separates \(M_{\Omega}\) from that curve and
  names the residual between \(M_{\Omega}\widehat{y}\) and \(\boldsymbol{b}\).
  No figure or tex edit in this closure. `R3-P14` stays `text_done`.
- Files: `sections/03_method.tex`;
  `figures/view_conditional_latent_prior_final.pdf`.
- Shared reply: none. `R3-P14` uses the same figure and answers a different
  question.
- Not done: none from this entry. `R3-06` is closed.

### R3-P14

- State: `text_done`
- Reviewer request: the old Figure 1 was hard to read; a tilded code seemed
  to decode to \(G(z_0)\); the lower blue line led nowhere; \(M=\operatorname{Im}(G)\)
  confused the index operator with the decoder range.
- Change made: the tilde is on the dense-log argument of \(h\). The lower
  arrow ends at \(G_{\mathrm{orig}}(\widehat{z})\) on the blue curve.
  \(M_{\Omega}\) selects coordinates. \(\mathcal{R}_{\mathrm{orig}}\)
  is that curve. The caption states that they are not the same object.
- Checked: same page-11 inspection as `R3-A03`.
- Files: `sections/03_method.tex`;
  `figures/view_conditional_latent_prior_final.pdf`.
- Shared reply: none. `R3-A03` explains the two mappings and the arrows.
- Not done: none from this entry. `R3-06` is closed.

### R3-05

- State: `closed`
- Reviewer request: MAP/Tikhonov and existence are standard arguments; say
  how they help interpret the application.
- Change made: the body states three questions, then explains the MAP
  reading and the existence result, including what each does not guarantee.
  The formal statements moved to the appendix, each before its proof. The
  conditional local-recovery proposition stays in the body. Its S-REC limit
  is unchanged. No new theorem and no new experiment.
- Checked: compiled pages 12--15 and appendix pages 45 and 47. No undefined
  or duplicate labels. The existence statement on page 47 first set
  \(\mathcal{J}(z)\) in the text line, and that line crossed the right
  margin. The definition is now displayed on two lines and stays inside
  the margin. Proposition
  numbers follow source order: conditional local recovery is first, then
  MAP, then existence.
- Files: `sections/03_method.tex`; `sections/09_appendix.tex`.
- Shared reply: none. `R3-P15` uses the same reorganization and answers the
  order of the propositions. `R1-M07` stays text_done; its caveat is
  preserved, and its reply now names the proposition rather than the old
  number.
- Checked on close: MAP/Tikhonov and existence are called classical, not
  new contributions. MAP explains the latent penalty; lambda is still
  chosen by validation. Existence needs a continuous decoder and lambda
  greater than zero, and it does not give uniqueness or global convergence
  of Adam. The local-recovery proposition depends on S-REC, which is not
  proved for the trained decoder. A lower objective is not a lower error
  against the unknown target. Proofs stay in the appendix. The body states
  the implications. No manuscript edit in this closure. `R3-P15` stays
  `text_done`.
- Not done: none from this entry. `R3-06` is closed.

### R3-06

- State: `closed`
- Reviewer request: make the main result easier to find, and stop the large
  MLP and PCA errors from compressing Figures 6 and 7.
- Change made: the cross-well results open with when refinement helps
  against direct AE, how it compares with interpolation, the affine fit,
  and smoothing, and when the extra cost is not repaid. The RMSE figure has
  a full five-method row and a lower-RMSE row, each with one vertical scale.
  The depth profile uses one clay-volume scale for the reference, ML only,
  AE, and CLP-CSGM Ridge. MLP and PCA are in a new appendix figure on their
  full range. Results and Discussion cite the runtime table and separate
  mean time per window from the normalized diagnostic. The diagnostic is not
  described as the quotient of the mean times, and the extra cost is not
  called justified in general.
- Checked: PDF, 88 pages, no undefined references. Figure 7 is page 29,
  Figure 8 is page 31, and Figure 24 is page 79. Shared scales, error bars,
  and the high-error methods are visible. The plotted CLP means match
  Table 4. Previous PNGs are in `figures/r3m06_originals/`.
- Files: `sections/06_results.tex`; `sections/07_discussion.tex`;
  `sections/09_appendix.tex`; `scripts/plot_r3m06_manuscript_figures.py`;
  `figures/results_crosswell_rmse_vs_rho.png`;
  `figures/10_depth_profile_crosswell_vc.png`;
  `figures/10_depth_profile_crosswell_vc_wide.png`.
- Shared reply: none. `R3-03` and `R3-04` stay closed. Their interpolation,
  smoothing, affine, and noise counts were not changed.
- Not done: none for this comment.

### R3-P15

- State: `text_done`
- Reviewer request: the list of propositions was confusing.
- Change made: section 3.4 opens with the purpose of each result. Classical
  statements are in the appendix. Four factors are named before the
  conditional local-recovery proposition, and the limits follow it.
- Checked: same pages as `R3-05`.
- Files: `sections/03_method.tex`; `sections/09_appendix.tex`.
- Shared reply: none. `R3-05` answers the role of the classical arguments.
- Not done: none from this entry. `R3-06` is closed.

### R3-A02

- State: `closed`
- Reviewer request: distinguish the sampling operator \(M_{\Omega}\) from the
  decoder output set.
- Change made: text only. The output set is no longer \(\mathcal{M}\) or
  \(\operatorname{Im}(G)\). \(\mathcal{R}_{G}\) is the range of the
  standardized decoder. \(\mathcal{R}_{\mathrm{orig}}\) is that class in
  the original target units. \(\mathcal{R}_{\mathrm{orig},R^{\dagger}}\)
  is the local original-scale image \(G_{\mathrm{orig}}(B(z_0,R^{\dagger}))\).
  Each symbol is defined before use. The reconstruction is stated to lie in
  \(\mathcal{R}_{\mathrm{orig}}\). The Figure 1 caption names the blue curve
  \(\mathcal{R}_{\mathrm{orig}}\) and states that \(M_{\Omega}\) is not that
  set. The figure file was not regenerated. No new experiment.
- Checked: compiled PDF, 80 pages. Page 9, equation (8), defines both ranges
  on one line inside the margin, before Figure 1. Page 10 caption uses
  \(\mathcal{R}_{\mathrm{orig}}\) for the curve the drawing already labels.
  Page 14 defines \(\mathcal{R}_{\mathrm{orig},R^{\dagger}}\) before
  equation (15). Appendix page 51 uses that symbol. Page 48 still shows the
  existence objective on two lines inside the margin. No undefined or
  duplicate references.
- Files: `sections/02_related_work.tex`; `sections/03_method.tex`;
  `sections/07_discussion.tex`; `sections/09_appendix.tex`.
- Shared reply: none. `R3-P08` uses the same symbols and answers the
  imaginary-part reading. `R3-A03` is closed. `R3-P14` stays text_done.
  Both replies name the curve \(\mathcal{R}_{\mathrm{orig}}\).
- Checked on close: \(M_{\Omega}\) is only the coordinate-restriction
  operator. \(\mathcal{R}_{G}\) is the standardized decoder range.
  \(\mathcal{R}_{\mathrm{orig}}\) is that range in the original target
  units. \(\mathcal{R}_{\mathrm{orig},R^{\dagger}}\) is the local set in
  the recovery condition. The Figure 1 caption separates the blue curve
  from \(M_{\Omega}\). No \(\mathcal{M}\) or \(\operatorname{Im}(G)\)
  remains. The notation change did not redraw the figure. The later redraw
  belongs to `R3-A03`, which is closed. `R3-P08` and
  `R3-P14` stay `text_done`. No manuscript edit in this closure.
- Not done: none from this entry. `R3-06` is closed.

### R3-P08

- State: `text_done`
- Reviewer request: \(\operatorname{Im}(G)\) can be read as the imaginary
  part; use another symbol for the range.
- Change made: text only. \(\operatorname{Im}(G)\) and
  \(\operatorname{Im}(G_{\mathrm{orig}})\) were removed. The related-work
  sentence now says "range" and does not use the symbol before the
  definition in Method. The three sets are \(\mathcal{R}_{G}\),
  \(\mathcal{R}_{\mathrm{orig}}\), and
  \(\mathcal{R}_{\mathrm{orig},R^{\dagger}}\), matched to the standardized
  range, the original-scale range, and the local recovery set. The identity
  \(I_m\) in the Gaussian noise model was left unchanged. No new experiment.
- Checked: same pages as `R3-A02`. A search of the manuscript sources found
  no remaining \(\operatorname{Im}\) and no remaining \(\mathcal{M}\).
- Files: `sections/02_related_work.tex`; `sections/03_method.tex`;
  `sections/07_discussion.tex`; `sections/09_appendix.tex`.
- Shared reply: none. `R3-A02` answers the distinction between \(M_{\Omega}\)
  and the output set.
- Not done: none from this entry. `R3-06` is closed.

### R3-A01

- State: `closed`
- Reviewer request: define symbols and abbreviations at first use, including
  the abstract and the opening equations.
- Change made: the abstract defines RMSE and rho, replaces GR-only with the
  gamma-ray description, and states the recovery caveat without naming
  S-REC. The body expands MLP, MAP, RMSE, S-REC, GAN, ANN, SHAP, MAE, AE,
  PCA, V_c, and GR-only at first use. The method names h, I_m, n_tr,
  gamma_AE, and lambda where they enter. The sampling length L is now N,
  the window length already used for the length-64 protocol. No equation,
  table, or reported value changed. Abstract length is 284 words.
- Checked: compiled PDF, 88 pages, no undefined references. The abstract
  remains within 300 words.
- Files: `main.tex`; `sections/01_introduction.tex`;
  `sections/02_related_work.tex`; `sections/03_method.tex`;
  `sections/04_data_and_eda.tex`; `sections/05_experiments.tex`.
- Shared reply: none. `R3-A02` is closed. `R3-06` stays closed.
- Not done: none for this comment.

### R3-A04

- State: `closed`
- Reviewer request: use "probability density" in Figure 3 so the axis is
  not read as the bulk-density log.
- Change made: the vertical label is now "Probability density". Bins,
  normalization, colors, transparency, scales, legend, and axis limits are
  unchanged. The caption already described the clay-volume distributions
  and was not edited.
- Checked: compiled page 17. The label is fully readable. The plotting
  panel matches the previous figure pixel for pixel outside the label
  strip. 88 pages, no undefined references.
- Files: `figures/eda_crosswell_vc_distribution.png`;
  `scripts/clp_csgm_paper_assets.py`.
- Shared reply: none.
- Not done: none from this entry.

### R3-A05

- State: `closed`
- Reviewer request: introduce Ridge explicitly as ridge regression and state
  its role in the workflow.
- Change made: the Introduction names ridge regression as the reference
  implementation of the conditional map from dense logs to \(z_0\), and
  keeps it as a linear low-capacity reference rather than the best map in
  every regime. The Method names the same fit in the log-conditioned prior:
  it learns \(h\) from standardized auxiliary logs and encoder codes;
  \(z_0\) initializes the first trajectory and remains the penalty center;
  \(G\) produces the target windows; ridge regression does not replace the
  decoder or the latent optimization. Direct ridge regression stays the
  separate \([u,b]\) comparator. The training equation was not changed, and
  no result was regenerated.
- Checked: compiled PDF, 89 pages, no undefined references, no new overfull
  lines. The MLP comparison and the statement that the linear map is not
  best in every test remain.
- Files: `sections/01_introduction.tex`; `sections/03_method.tex`.
- Shared reply: none.
- Not done: none from this entry.

### R3-P01

- State: `closed`
- Reviewer request: the Abstract goal is hard to see; state that a continuous
  log and a few sparse observations should yield a continuous prediction of
  the property.
- Change made: the Abstract opens by stating that goal, using dense auxiliary
  wireline logs and sparse observations at selected depths. The penalty still
  allows a residual. The sparse benchmark observations remain subsamples of
  processed curves, not independent core measurements. Numbers and
  conclusions are unchanged. The Introduction already stated the same task
  and was not edited. `R3-P02` is closed.
- Checked: compiled PDF, 89 pages, no undefined references and no overfull
  lines. Abstract length is 284 prose words, 300 tokens including numbers.
- Files: `main.tex`.
- Shared reply: none.
- Not done: none from this entry.

### R3-P02

- State: `closed`
- Reviewer request: the coordinate mask looks weakly linked to an inverse
  problem and might better be called missing-data reconstruction.
- Change made: Problem setup states that the application is missing-data
  reconstruction and also an underdetermined inverse problem. \(M_{\Omega}\)
  selects \(m<N\) entries and is not injective on \(\mathbb{R}^{N}\).
  Classical compressed sensing uses a structural prior; CSGM uses a learned
  generator range \citep{bora2017compressed}. CLP-CSGM adds the conditional
  latent prior. The text says this is not inversion of a physical
  petrophysical model and that guarantees for other operators, including
  certain random matrices, do not transfer to \(M_{\Omega}\). Related work
  adds that missing-data reconstruction, including inpainting, is studied as
  an inverse problem \citep{asim2020invertible}. No new equation. The
  Abstract, S-REC statement, and reported results were not changed.
- Checked: compiled PDF, 89 pages, no undefined references and no overfull
  lines. Both citation keys were already in `references.bib`.
- Files: `sections/03_method.tex`; `sections/02_related_work.tex`.
- Shared reply: none. `R3-05` and `R3-P15` stay as previously recorded.
- Not done: none from this entry.

## Queue (not yet processed)

R3-P03 .. R3-P07, R3-P11 .. R3-P13, R3-P16 .. R3-P19, AE-01.
R1-m05 remains deferred. `R1-m03` and `R3-P14` are text_done. `R3-A03` is closed.
`R3-05` and `R3-P15` are text_done. `R3-A02` is closed. `R3-P08` stays text_done.
`R3-06` is closed.
`R3-P09` and `R3-P10` are text_done. `R3-P16` remains pending.
