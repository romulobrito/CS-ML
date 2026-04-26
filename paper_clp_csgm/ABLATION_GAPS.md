# Ablation And Gap List

This file lists the minimum additional experiments or checks that would make the
paper stronger before submission. It separates necessary ablations from optional
extensions.

## Minimum Ablations

### 1. Prior-only decoder ablation

- Purpose: isolate the value of sparse measurement refinement.
- Compare:
  - `G(h(u))`, with no test-time refinement;
  - CLP-CSGM Ridge, with refinement.
- Expected interpretation: if CLP-CSGM improves over `G(h(u))`, the sparse
  measurements are doing useful inverse-problem work rather than the result
  being only a latent regression effect.
- Priority: high.

### 2. Measurement-only CSGM ablation

- Purpose: isolate the value of the conditional latent prior.
- Compare:
  - standard CSGM, `min_z ||M G(z) - b||_2^2`;
  - CLP-CSGM Ridge, `min_z ||M G(z) - b||_2^2 + lambda ||z - h(u)||_2^2`.
- Expected interpretation: if CLP-CSGM improves, the log-conditioned prior is
  useful in low-rho regimes.
- Priority: high.

### 3. Lambda sensitivity

- Purpose: show that the result is not a fragile single hyperparameter choice.
- Compare a small grid of lambda values for one representative cross-well cell
  and one F03 sparse-rho cell.
- Expected interpretation: a broad valley supports robustness; a sharp optimum
  suggests the method needs careful validation.
- Priority: medium-high.

### 4. Ridge prior versus MLP prior

- Purpose: justify the paper-facing Ridge variant.
- Compare CLP-CSGM Ridge with CLP-CSGM MLP when available.
- Expected interpretation: Ridge is preferred if it is more stable in low-data
  cells, even if MLP can win in larger-data cases.
- Priority: medium.

## Useful Robustness Checks

### Seed uncertainty

- Report mean and standard deviation or confidence intervals for the key
  cross-well and F03 results.
- Priority: high if the journal expects statistical reporting.

### Main-figure simplification

- Reduce repeated F03 figures in the main text and move secondary metrics to
  appendix.
- Priority: medium.

### Additional real-well split

- Test a second contiguous split or another well if data are available.
- Priority: high for field-generalization claims, optional for a methods-first
  paper.

### More petrophysical references

- Add recent references on deep learning and sparse core/log integration in
  petrophysics.
- Priority: medium-high before submission.

## Optional Extensions

### Uncertainty quantification

- Add latent posterior sampling or bootstrap intervals.
- Benefit: aligns with reservoir decision-making.
- Cost: substantial method and experiment expansion.

### Measurement design

- Compare random coordinate sampling with stratified or geologically informed
  sampling.
- Benefit: connects the method to core acquisition strategy.
- Cost: additional experimental design.

### Richer decoder architecture

- Compare the current small autoencoder with convolutional or variational
  alternatives.
- Benefit: tests representation-error limits.
- Cost: may shift the paper toward architecture search.

## Recommended Next Experimental Step

Run the prior-only and measurement-only ablations first. They are the most
scientifically important because they test the two essential components of the
method:

1. the conditional prior from dense logs;
2. the sparse measurement-consistency refinement.

If both ablations support the main result, the paper's methodological claim will
be much stronger.
