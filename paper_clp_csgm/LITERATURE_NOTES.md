# Literature Notes

This file maps each reference to its function in the paper. It is an editorial
control document, not a replacement for the bibliography.

## Petrophysics And Well Logs

### Ellis and Singer (2007), Well Logging for Earth Scientists

- Use in: Introduction, Related Work.
- Supports: well logs are dense indirect measurements and core-calibrated
  interpretation is central to petrophysical analysis.
- Claim supported: petrophysical properties are inferred from logs but usually
  require calibration from sparse direct measurements.

### Asquith and Krygowski (2004), Basic Well Log Analysis

- Use in: Introduction, Related Work.
- Supports: standard well-log interpretation context and practical role of core
  calibration.
- Claim supported: porosity, clay/shale volume, and related properties are
  routinely interpreted from logs but depend on calibration.

### Ahmadi and Chen (2019), Petroleum

- Use in: Related Work.
- Supports: machine learning methods are used to estimate porosity and
  permeability from petrophysical logs.
- Claim supported: direct log-to-property regression is an established
  petrophysical prediction strategy.

### Wood (2020), Journal of Petroleum Science and Engineering

- Use in: Related Work.
- Supports: data-mining and machine-learning approaches for predicting porosity,
  permeability, and water saturation from well logs.
- Claim supported: well-log machine learning is a strong baseline family, not a
  straw-man comparison.

## Classical Compressed Sensing

### Donoho (2006), IEEE Transactions on Information Theory

- Use in: Introduction, Related Work.
- Supports: compressed sensing reconstructs structured signals from incomplete
  measurements.
- Claim supported: sparse recovery is the classical way to encode structure in
  underdetermined inverse problems.

### Candes, Romberg, and Tao (2006), IEEE Transactions on Information Theory

- Use in: Introduction, Related Work.
- Supports: robust uncertainty principles and exact recovery from incomplete
  measurements.
- Claim supported: CS provides a forward-model view where measurements constrain
  reconstruction.

### Candes, Romberg, and Tao (2006), Communications on Pure and Applied Mathematics

- Use in: Introduction, Related Work.
- Supports: stable recovery from inaccurate measurements.
- Claim supported: CS can handle noisy measurements under appropriate
  conditions.

## Generative Priors And Inverse Problems

### Bora, Jalal, Price, and Dimakis (2017), ICML

- Use in: Introduction, Related Work, Method.
- Supports: CSGM replaces fixed-basis sparsity with the range of a generator and
  optimizes over latent codes.
- Claim supported: generative priors can outperform sparsity priors with fewer
  measurements in some inverse-problem settings.
- Caution: their guarantees require assumptions such as suitable measurement
  matrices/S-REC; the present paper uses the theory as motivation, not as a
  direct proof for real-well coordinate subsampling.

### Kamath, Price, and Scarlett (2020), ICML

- Use in: Related Work.
- Supports: theoretical analysis of the power and limits of compressed sensing
  with generative models.
- Claim supported: CSGM is an established research line with known theoretical
  boundaries.

### Asim, Daniels, Leong, Ahmed, and Hand (2020), ICML

- Use in: Related Work.
- Supports: invertible generative models as priors for inverse problems and
  discussion of representation error/dataset bias.
- Claim supported: generative priors for inverse problems are broader than the
  original GAN/VAE CSGM setting.

### Ulyanov, Vedaldi, and Lempitsky (2018), CVPR

- Use in: Related Work.
- Supports: neural parameterizations can serve as priors in inverse problems.
- Claim supported: inverse-problem reconstruction can be improved by searching a
  structured neural representation instead of using hand-crafted priors only.

### Kingma and Welling (2014), ICLR

- Use in: optional background if discussing VAEs.
- Supports: learned latent generative models.
- Current status: included in references but not central to the current text.

### Goodfellow et al. (2014), NeurIPS

- Use in: optional background if discussing GANs.
- Supports: learned generative modeling as a broad family.
- Current status: included in references but not central to the current text.

## Regularization And Conditional Prior

### Hoerl and Kennard (1970), Technometrics

- Use in: Method.
- Supports: ridge regression as a stable regularized linear estimator.
- Claim supported: Ridge is a reasonable low-data prior map from logs to latent
  codes.

### Tikhonov (1963), Soviet Mathematics Doklady

- Use in: Method.
- Supports: Tikhonov regularization interpretation.
- Claim supported: the latent prior term can be read as regularization around
  the log-conditioned latent estimate.

### Bishop (2006), Pattern Recognition and Machine Learning

- Use in: Introduction.
- Supports: general supervised learning and nonlinear regression context.
- Claim supported: machine-learning regressors can model nonlinear relations
  between logs and target properties.

## Novelty Positioning

The closest methodological ancestor is Bora et al. (2017). The present paper
differs by adding a conditional latent prior from dense well logs and by using
sparse core measurements as target-well consistency constraints. The closest
application context is petrophysical machine learning from well logs; the paper
differs by treating sparse target observations as a measurement operator in a
latent inverse problem rather than as raw supervised features.
