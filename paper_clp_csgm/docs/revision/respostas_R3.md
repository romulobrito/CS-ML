# Reviewer 3 -- proposed replies

Each ID keeps its own reply, including PDF annotations R3-P01..P19. If the
same manuscript change fully answers more than one ID, the English reply is
copied verbatim.

Every completed reply names where the manuscript text was changed (Abstract,
section title, or appendix subsection title), not only what was changed.

Source comments: `docs/revision/CLP_CSGM_Pareceres_Consolidados.md`.

---

## R3-01 -- Clarity and stacked terminology

**Status:** `closed`

**Proposed reply:**

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

## R3-02 -- Practical advantage versus conventional relations

**Status:** `closed`

**Proposed reply:**

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

## R3-03 -- Simple comparators and intermediate results

**Status:** `closed`

**Proposed reply:**

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

## R3-04 -- Noise choice and prior/data balance

**Status:** `closed`

**Proposed reply:**

The benchmark adds independent Gaussian noise after coordinate subsampling,
with standard deviation 0.01 for cross-well clay volume and 0.02 for F03-4
porosity, in the respective target units. The value 0.01 is about 0.29 times
the standard deviation of the held-out clay-volume curve. These levels were
adopted for controlled benchmarks and were not calibrated to instrumental
uncertainty. The `30dB` filename designation is separate from this added
noise and is not interpreted as a verified SNR.

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

## R3-05 -- Role of the theory section

**Status:** `closed`

**Proposed reply:**

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
were added. This comment overlaps R3-P15, which concerns the order of the
propositions. That request is answered in its own reply.

Where the text was changed: Method, subsection Interpretation and
theoretical context; Appendix MAP and Tikhonov interpretation; Appendix
Existence of a minimizer.

## R3-06 -- Main result and figure readability

**Status:** `closed`

**Proposed reply:**

We reorganized the opening of the cross-well results around three questions:
when latent refinement helps relative to the direct autoencoder, how it
compares with simple reconstructions, and when the extra optimization stops
being worthwhile.

CLP-CSGM Ridge has lower mean RMSE than direct AE in seven of the nine
cross-well cells. At steps 16 and 32 that ordering holds in the ten-seed
panel and in every sparse-measurement redraw. At the reference noise 0.01,
depth interpolation is lower in all nine cells, the affine fit is lower in
all nine, and the validation-selected smoother is lower in eight of nine.
The relative robustness at higher simulated noise, already reported for
R3-04, is unchanged. The extra cost is not presented as justified in
general.

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

## R3-A01 -- Define symbols at first use

**Status:** `closed`

**Proposed reply:**

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

## R3-A02 -- Separate sampling operator and output set

**Status:** `closed`

**Proposed reply:**

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

## R3-A03 -- Redraw Figure 1

**Status:** `closed`

**Proposed reply:**

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

## R3-A04 -- Probability density in Figure 3

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for pointing out the ambiguity. We have changed the
vertical-axis label of Figure 3 from "density" to "Probability density",
clarifying that the figure represents statistical distributions of clay
volume rather than the petrophysical bulk-density log. The underlying data
and histogram calculations remain unchanged.

Where the figure was changed: `figures/eda_crosswell_vc_distribution.png`,
and its caption in Data and exploratory analysis. The caption was already
correct and was not edited.

## R3-A05 -- State that Ridge is ridge regression

**Status:** `closed`

**Proposed reply:**

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

## R3-P01 -- Unclear objective in the Abstract

**Status:** `closed`

**Proposed reply:**

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
already states the same task and was not edited. `R3-P02` is closed.

## R3-P02 -- Framing as an inverse problem

**Status:** `closed`

**Proposed reply:**

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

## R3-P03 -- Admissible reconstruction set

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for identifying this unclear expression. We have
replaced "admissible reconstruction range" with a direct explanation of the
decoder's role. The Abstract now states that a learned decoder maps latent
codes to candidate target-property windows, limiting which curves it can
reconstruct. The Method section already defines this set formally as the
decoder range, in standardized units and in the original target units. This
restriction concerns model representability and does not imply guaranteed
physical admissibility or exact recovery.

Where the text was changed: Abstract, the decoder sentence. The Method
definitions from R3-A02 were already correct and were not edited.

## R3-P04 -- Meaning of rho in the Abstract

**Status:** `closed`

**Proposed reply:**

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

## R3-P05 -- Weak single-channel context

**Status:** `closed`

**Proposed reply:**

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

## R3-P06 -- Ridge regression and low-capacity prior

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for requesting clarification. Yes, ridge regression is
used to implement the conditional latent-prior map h. It is trained to
predict latent codes from dense auxiliary logs, using codes obtained from the
target-window encoder during training. At inference, its prediction z_0
initializes latent refinement and serves as the center of the quadratic
latent penalty. The decoder G maps the refined code to the reconstructed
target-property window. Ridge regression therefore does not replace the
decoder or perform the latent optimization.

The Introduction and Method now clarify these roles and distinguish the
conditional ridge-regression map from the direct ridge-regression baseline,
which predicts target windows from auxiliary logs and sparse observations.
The ambiguous phrase "low-capacity latent-prior reference" has been removed
from the Abstract.

## R3-P07 -- Symbols in the introductory equations

**Status:** `closed`

**Proposed reply:**

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

## R3-P08 -- Im(G) may look like the imaginary part

**Status:** `closed`

**Proposed reply:**

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

## R3-P09 -- Measurement noise and lambda

**Status:** `closed`

**Proposed reply:**

After coordinate subsampling, each observed coordinate receives an independent
Gaussian perturbation. Its standard deviation is 0.01 for cross-well clay
volume and 0.02 for F03-4 porosity, in the respective target units. These
values define controlled benchmark noise and were not calibrated to
instrumental uncertainty. The `30dB` filename designation is unrelated to
this added perturbation.

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

Where the text was changed: Method, subsection Test-time reconstruction (the latent-refinement objective and the residual identity); Method, subsection Interpretation and theoretical context (the MAP proposition); Appendix MAP and Tikhonov interpretation; Appendix Direct linear maps, depth interpolation, and measurement noise; the implementation-summary table.

## R3-P10 -- Covariance

**Status:** `closed`

**Proposed reply:**

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

## R3-P11 -- Baseline identity

**Status:** `closed`

**Proposed reply:**

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

## R3-P12 -- Why predict z rather than y?

**Status:** `closed`

**Proposed reply:**

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

## R3-P13 -- Intermediate results

**Status:** `closed`

**Proposed reply:**

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

## R3-P14 -- Arrows and symbols in Figure 1

**Status:** `closed`

**Proposed reply:**

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

## R3-P15 -- Confusing propositions

**Status:** `closed`

**Proposed reply:**

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
Gaussian hypotheses, lambda equals the ratio of the noise variance to the
latent-prior variance. The experiments do not use that ratio: validation
selects lambda. Existence assumes a continuous decoder and lambda greater
than zero. It gives a minimizer. It does not give uniqueness, convexity, or
convergence of Adam to a global minimizer, and it does not transfer
automatically to lambda equal to zero. The numerical solution remains
distinct from a certified global minimizer.

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

## R3-P16 -- Meaning of 30 dB

**Status:** `closed`

**Proposed reply:**

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

## R3-P17 -- Definition of Vc

**Status:** `closed`

**Proposed reply:**

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

## R3-P18 -- Density is ambiguous

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for pointing out the ambiguity of the term "Density" in Figure 3.

We agree that this label could be confused with the bulk-density (RHOB) well log used elsewhere in the manuscript.

The vertical axis of Figure 3 has therefore been relabeled "Probability density", explicitly identifying the statistical quantity represented by the empirical clay-volume distributions. Each histogram is normalized so that its area equals one. The vertical scale is not a count of samples.

This correction distinguishes the probability-density representation from the petrophysical bulk-density log without changing the underlying data or the plotted distributions.

Where the figure was changed: Data and exploratory analysis, Figure 3.

## R3-P19 -- Figure 7 and GR-derived Vc

**Status:** `closed`

**Proposed reply:**

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

## AE-01 -- Associate Editor synthesis

**Status:** `closed`

The compliance matrix is `docs/revision/ae01_matriz_conformidade.md`.
The conditional-geometry schematic was prepared with ChatGPT assistance, and
xAI's Grok 4.7 was used to refactor existing scientific code. The research
implementation was not originally generated by AI. The geometry caption, the
general declaration, and Implementation details now state those uses. No
result was changed.

**Proposed reply:**

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
