# CLP-CSGM Paper Brief

## Working Title

Conditional Latent-Prior Generative Compressed Sensing for Petrophysical
Property Estimation from Sparse Core Measurements

## Research Question

How should sparse target-well core measurements be used when dense well logs are
available but direct labels are scarce: as supervised input features, or as
measurement-consistency constraints over a learned conditional generative prior?

## Central Hypothesis

Sparse core measurements are most useful when enforced at test time as physical
constraints over a conditional latent generative prior. Dense logs should anchor
the reconstruction through a predicted latent code, while sparse measurements
should refine that code through the known measurement operator.

## Proposed Method

The proposed method is CLP-CSGM Ridge:

- train an autoencoder decoder `G(z)` on target-property windows;
- train a Ridge prior `h(u)` from dense log windows to encoded latents;
- solve, at test time, `min_z ||M G(z) - b||_2^2 + lambda ||z - h(u)||_2^2`;
- output `G(z_hat)`.

## Main Contribution

The contribution is not generic CSGM theory. The contribution is the
petrophysical specialization of CSGM in which target-well logs define a
conditional latent prior and sparse core measurements define the data-fidelity
term of the inverse problem.

## Evidence Used In The Paper

- Cross-well low-data Vc benchmark: CLP-CSGM Ridge wins against AE `[u,b]` in
  all nine step-rho cells.
- Real-well F03-4 GR-only porosity benchmark: CLP-CSGM Ridge has the best
  average RMSE and the clearest advantage at low measurement ratios.
- Qualitative diagnostics: parity, window examples, and depth-profile figures
  show that CLP-CSGM improves ordering and profile stability compared with
  direct baselines.

## In Scope

- Dense-log plus sparse-core petrophysical estimation.
- Low-data cross-well transfer.
- Sparse measurement assimilation through a known subsampling operator.
- Comparison against direct `[u,b]` baselines: ML only, MLP, PCA, and AE.
- Methodological interpretation through MAP/Tikhonov and CSGM theory.

## Out Of Scope

- Claiming a new general CSGM theorem.
- Claiming universal field-scale generalization.
- Treating SIR-CS/LFISTA as the proposed method in this paper.
- Exhaustive architecture search over autoencoders and priors.
- Full uncertainty quantification.

## Allowed Claims

- CLP-CSGM Ridge outperforms the direct AE baseline in the tested low-data
  cross-well Vc grid.
- CLP-CSGM Ridge provides the best average RMSE in the tested F03 GR-only
  porosity benchmark.
- The method is most advantageous in sparse-measurement regimes.
- Direct MLP `[u,b]` can degrade when sparse measurement features are used
  naively.
- The method is theoretically motivated by CSGM and MAP/Tikhonov
  regularization, but empirical validation is the main evidence in this paper.

## Claims To Avoid Or Qualify

- Do not claim that CLP-CSGM is universally superior to AE regression.
- Do not claim that the CSGM recovery guarantees directly apply to the real-well
  coordinate-subsampling setting.
- Do not claim that Ridge is the optimal prior class; it is the reported stable
  prior in the current low-data setting.
- Do not overstate the real-well result: AE `[u,b]` is slightly better at higher
  measurement ratios.

## One-Sentence Paper Message

CLP-CSGM Ridge shows that sparse core measurements can be more effectively
assimilated as test-time constraints over a log-conditioned generative latent
manifold than as raw input features to a direct supervised regressor.
