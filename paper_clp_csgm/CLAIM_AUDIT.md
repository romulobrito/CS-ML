# Claim Audit

This audit checks whether the manuscript claims are aligned with the current
evidence, references, and stated scope.

## Overall Assessment

The manuscript is scientifically coherent in its current direction. The central
claim is appropriately narrow: CLP-CSGM Ridge is a conditional generative
compressed sensing method that is useful for low-data and sparse-measurement
petrophysical estimation. The text does not need to claim universal superiority
over direct regressors.

## Claims That Are Well Supported

### Sparse measurements as constraints

- Current claim: sparse target-well measurements are more useful when enforced
  through a measurement-consistency term than when used only as supervised
  features.
- Evidence: cross-well CLP-CSGM wins over AE `[u,b]` in all low-data cells; F03
  shows strongest gains at low measurement ratios.
- Status: supported, but should be phrased as "in the tested settings" when
  used outside the abstract/results.

### CLP-CSGM Ridge wins the cross-well low-data grid

- Current claim: CLP-CSGM Ridge outperforms AE `[u,b]` in all nine step-rho
  cells.
- Evidence: `results_crosswell_clp_vs_ae.tex` and
  `results_crosswell_clp_vs_ae.png`.
- Status: supported.

### F03 result is sparse-regime advantage, not universal dominance

- Current claim: CLP-CSGM Ridge is best on average and strongest at low rho; AE
  `[u,b]` slightly wins at higher rho.
- Evidence: F03 RMSE table and figures.
- Status: well balanced and should be preserved.

### CSGM theory is motivational, not directly proven for this field setting

- Current claim: the method is motivated by CSGM and Tikhonov/MAP
  regularization; direct guarantees are not asserted.
- Evidence: Method remarks and Discussion limitations.
- Status: scientifically safe.

## Claims That Need Careful Wording

### "Physical evidence"

- Risk: sparse measurements are direct observations, but the measurement model
  here is coordinate subsampling, not a full physical forward simulator.
- Recommended wording: "measurement evidence" or "known measurement-operator
  evidence" when precision is important.

### "Generative manifold"

- Risk: an autoencoder decoder range is often called a manifold, but the paper
  does not prove manifold properties.
- Recommended wording: "learned generative range" in theoretical passages and
  "latent manifold" only as intuitive language.

### "Hard evidence" or "hard constraint"

- Risk: the objective uses a squared residual with finite lambda, so sparse
  measurements are softly enforced.
- Recommended wording: "measurement-consistency term" or "softly enforced
  measurement consistency".

### "Distribution shift"

- Risk: the EDA shows a target-distribution shift, but not a full statistical
  domain-shift test.
- Recommended wording: "visible target-distribution mismatch" or "shift in the
  observed target distribution".

## References Coverage

### Adequate

- Classical CS is covered by Donoho and Candes/Romberg/Tao.
- CSGM is covered by Bora et al. and later generative-prior works.
- Ridge/Tikhonov interpretations are covered by Hoerl/Kennard and Tikhonov.
- Petrophysical log context is covered by Ellis/Singer and Asquith/Krygowski.

### Could Be Strengthened Before Submission

- Add two or three recent petrophysical deep-learning papers focused on
  well-log property prediction.
- Add one or two references on core-log calibration or sparse core assimilation,
  if available.
- Add a reference on inverse problems with learned priors in geoscience if the
  target journal expects geoscience-specific inverse-problem positioning.

## Manuscript Structure Audit

### Introduction

- Status: aligned with the paper brief.
- Main improvement: add one more sentence delimiting the claim to sparse and
  low-data regimes.

### Related Work

- Status: adequate but compact.
- Main improvement: expand petrophysical ML and learned-prior inverse problems
  before submission.

### Method

- Status: strong and appropriately formal.
- Main improvement: explicitly state that measurement consistency is soft,
  controlled by lambda.

### Data And EDA

- Status: useful and connected to the experiment design.
- Main improvement: avoid implying that the EDA alone proves direct regressors
  must fail.

### Results

- Status: persuasive and consistent with the figures.
- Main improvement: reduce figure count in the main text if targeting a journal
  with strict page limits.

### Discussion

- Status: balanced and scientifically cautious.
- Main improvement: add a short paragraph on what result would falsify the
  method claim: for example, if AE `[u,b]` dominates in low-rho cross-well
  settings across additional wells.

## Required Final Checks Before Submission

- Verify all numerical values against the CSV artifacts after final asset
  generation.
- Compile after any table or figure move to appendix.
- Search for wording such as "always", "guarantees", "hard constraint", and
  "universal" and qualify if found.
- Ensure that `hybrid_lfista_joint` remains out of the main narrative unless
  explicitly discussed as prior negative evidence or appendix context.
