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

We thank the reviewer for this comment. The stacked labels in the abstract
have been replaced by a description of the operations. The decoder, trained
on windows of the target curve, supplies the windows the reconstruction may
use. The conditional map predicts a latent code z_0 from the dense auxiliary
logs. This code provides an informed initialization and serves as the
reference for the quadratic penalty during latent refinement. The first
trajectory starts at z_0. Each additional restart starts at z_0 plus an
independent draw from N(0, 0.05^2 I). The penalty center remains z_0 on every
restart and every iteration. Latent refinement balances agreement with the
sparse observations against deviation from z_0. A residual at the observed
depths can remain. The decoder range and that penalty are separate. Ridge is
the linear map used as the reference for z_0. Using only the decoded
prediction of the conditional prior, or only the measurement penalty, raises
the error in most cells. This wording clarifies the formulation already
implemented. It does not change the algorithm, the equations, or the reported
results.

The same plain description is used in the Introduction, Related work, Method,
Data and exploratory analysis, Experimental setup, Results, Interpretation,
scope, and limitations, and the Conclusion. The numerical results and the
manuscript's scope limitations remain unchanged. The Abstract reports
regime-dependent performance and the need for broader validation. The
Introduction and theoretical discussion clarify that the conditional bound
does not establish S-REC for the trained decoder, while the Experimental
setup and Discussion limit the GR-only porosity benchmark to a controlled
test.

Where the text was changed: Abstract; Introduction; Related work; Method
(the opening, the paragraph after the measurement model, the conditional-code
figure caption, Test-time reconstruction, and the ablation paragraph);
Experimental setup (the protocol caption, the restart paragraph, and the
implementation table); Results (Ablation analysis); Interpretation, scope,
and limitations; Conclusion; Appendix Supplementary Theoretical Details
(the opening paragraph); Appendix Laboratory porosity on a cored interval
(the definition of the decoded prediction).

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

We thank the reviewer for asking for these comparators and for the role of the encoder.

The encoder is used only in training. It builds the code of the known windows, and the dense-log map is fit on those codes. At inference the unknown window is not encoded. The code has dimension 16. Passing the full window through the encoder and the decoder only checks that the decoder can represent the curve. That check is not part of the comparison.

The experiment uses the same cross-well grid as the paper. We added linear regression and direct ridge regression from the dense logs and the sparse measurements, and interpolation of those measurements along depth. We then kept the windows and the measured positions fixed and raised only the noise.

At the published noise, interpolation has the lowest error in every cell. The measurements are a slightly perturbed copy of the same curve, so connecting them is enough. CLP-CSGM is ahead of both linear maps only in the few-window cells: at step 16 with the largest measurement fraction, and at step 32.

When the noise rises to the same order as the variation of the curve, the interpolation advantage falls in the few-window cells, and also at step 8 with the largest measurement fraction. The penalty weight selected for this comparison stays inside the published grid.

At step 8, with the two smaller measurement fractions, interpolation still has the lower mean at that noise. The decoder can represent the curve. The window predicted from the other logs alone has a larger error than the interpolation, and the CLP-CSGM error stays between those two error values. With the larger measurement fraction, the CLP-CSGM mean falls below the interpolation. At steps 16 and 32 the dense-log window has an error of similar magnitude to the interpolation. Similar magnitudes do not show that the reconstructions are close to each other. That pattern is compatible with a weak dense-log window. It does not isolate that window as the only cause.

At the two larger noise levels, CLP-CSGM has the lower error in every cell. These noise levels were used for sensitivity analysis and were not calibrated to field measurement uncertainties. At the largest of them, the noise is several times the variation of the curve.

Results also shows this ordering in one figure, with one panel per step and one curve per measurement fraction, so the step-8 cells are not averaged away. The horizontal axis is logarithmic. Relative to the curve standard deviation 0.035, on the same scale as the noise, 0.05 is about 1.43 times larger, and 0.10 and 0.20 are about 2.86 and 5.71 times larger. The decoder and the dense-log map are fit once per step and seed. The penalty is selected again on validation for each measurement fraction and each noise level. The noise 0.01 points in that figure are this refit. They keep the same ordering as the published grid and can differ slightly from its cell means.

The detail is in the appendix Direct linear maps, depth interpolation, and measurement noise, with the two tables and one depth example. The Results paragraph states the cell-level ordering next to that figure. The Abstract states that interpolation has lower error in all nine configurations at the reference noise 0.01, and that under larger perturbations that comparison changes. Those noise levels were used for sensitivity analysis and were not calibrated to field measurement uncertainties. The Conclusion keeps the step-8 exceptions at noise 0.05 and states that AE, ordinary least squares, and direct ridge were not rerun across the sweep. A second-difference smoother was added later in this same reply. The Experimental setup announces the experiment, and the Method explains the encoder. These runs are not merged into the submitted tables. This wording does not close the question of how the published noise level was chosen.

We have revised the discussion of the noise-sensitivity results to distinguish the observed RMSE ordering from its possible explanation. The revised Results text relates the smaller RMSE increase of CLP-CSGM to the general role of regularization \cite{hansen2010discrete} and discusses representation, noise, and optimization errors in the context of generative compressed sensing \cite{bora2017compressed}.

We explicitly state that the sweep does not isolate the effects of the decoder restriction, the conditional penalty, and validation-based parameter selection, and that these references do not provide a recovery guarantee for our coordinate-sampling experiments. The revised text retains the advantage of interpolation at the reference noise level and identifies the larger perturbations as a controlled sensitivity analysis. No experiments or numerical results were changed.

Changes were made in the Results discussion accompanying the RMSE-versus-noise figure, in the appendix Direct linear maps, depth interpolation, and measurement noise, with the Hansen reference added to the bibliography.

A smoothing comparator was added on the same cross-well stream. It minimizes the squared residual at the observed coordinates plus alpha times the squared second difference along the window. Alpha is chosen only on validation, from the frozen grid 1e-6, 1e-4, 1e-2, 1, 1e2, 1e4, and 1e6. The grid was not changed after the test scores. The largest value is selected in 62 of 108 seed rows. Those ceiling curves are already near an affine least-squares fit to the same noisy observations, which is the limit as alpha tends to infinity. The median profile RMSE between alpha=1e6 and that affine fit is 5.2e-5 on all 108 rows and 7.6e-5 on the 62 ceiling rows. The largest scored-RMSE gap on the ceiling rows is 1.9e-5. The affine fit is an additional comparator. It does not replace the validation-selected smoother.

At the reference noise 0.01, the selected smoother has lower mean RMSE than CLP-CSGM in eight of nine cells, and interpolation remains lower than CLP-CSGM in all nine. At noise 0.05 the ordering is mixed: CLP-CSGM is lower in five cells and smoothing in four. At 0.10, CLP-CSGM is lower in eight cells, with smoothing lower at step 8 and measurement fraction 0.20. At 0.20, CLP-CSGM is lower in all nine cells. At measurement fraction 0.05, the smoother mean is higher than interpolation at every noise level. The affine cell mean is lower than CLP-CSGM in all nine cells at noise 0.01. In the one selected-smoother exception, step 32 and measurement fraction 0.05, only one seed selects the ceiling: the affine mean is 0.0240, the selected smoother is 0.0429, and CLP-CSGM is 0.0317. These noise levels remain a sensitivity analysis. They were not calibrated to field or laboratory uncertainty. The ordering does not establish general superiority and does not isolate a regularization mechanism.

Where the text was changed: Abstract (the sentence on interpolation at noise 0.01); Method (the generative-decoder paragraph); Experimental setup (the paragraph after the sparse-coordinate placement check, and the simple-baseline sentence); Results (the discussion accompanying the RMSE-versus-noise figure); Conclusion (the noise paragraph); Appendix Direct linear maps, depth interpolation, and measurement noise (the smoothing table, the affine-limit paragraph, and the two new figures).

**Shared reply:** none. `R3-04` is answered in its own entry. This reply does not close it.

## R3-04 -- Noise choice and prior/data balance

**Status:** `closed`

We thank the reviewer for asking how the noise level was chosen and how that choice balances the prior against the data fit.

The published cross-well clay-volume benchmark adds independent Gaussian noise after coordinate subsampling. The standard deviation is 0.01 in the units of the clay-volume curve, not one percent of the curve. The held-out curve has standard deviation about 0.035, so 0.01 is about 0.29 times that deviation. The F03-4 porosity benchmark uses the same construction with standard deviation 0.02. These levels were adopted for controlled benchmarks. No calibration to instrumental uncertainty is documented. The input files are labeled 30dB. That label is not the added standard deviation, and this manuscript does not define the procedure behind the label. R3-P16 records that designation as a filename label, not a verified signal-to-noise ratio. Within one run, every compared method receives the same noisy observations. Direct models from the concatenated dense logs and sparse observations standardize those features after the noise is added.

The MAP statement identifies lambda with the ratio of the noise variance to the latent prior variance when the likelihood is an isotropic Gaussian in the original target units. On that scale the implemented residual equals a standardized residual weighted by the diagonal matrix of the training standard deviations at the observed coordinates. Those standard deviations are not equal. On the cross-well clay-volume training windows, the ratio of the largest to the smallest is about 1.13 at step 8, 1.23 at step 16, and 1.40 at step 32, so one scalar conversion does not turn the unweighted standardized residual into the implemented objective. The displayed objective now uses that weighted residual. The experiments do not compute lambda from the variance ratio. For each seed and measurement ratio, validation full-window RMSE selects one value from the grid 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, and 1e-1, and that value is fixed before the test evaluation. Dividing both objective terms by the number of windows multiplies the objective by one positive factor. That factor leaves the minimizers unchanged. The covariance of this added noise, after standardization, is stated in the standardization paragraph and answered under R3-P10.

The later noise sweep is a sensitivity check of the adopted noise level, not the reason that level was chosen. The sensitivity sweep uses a different, wider grid: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, and 100. It omits the factor-of-three candidates from the published grid. At 0.01 and 0.05 every selected weight also belongs to the published grid. That membership does not mean the two searches were the same. At noise 0.10, step 16 and measurement fraction 0.10, one seed selects the weight 1, which is above the published grid and still inside the sweep. At noise 0.20, some cells select 100, the largest candidate that was tried. That selection is not an optimum outside the searched grid. The interpolation and CLP-CSGM ordering depends on the noise level and the configuration, as already reported next to the RMSE-versus-noise figure and the noise-sweep table. No new experiments were added, and the algorithm and the reported numbers are unchanged.

We have revised the discussion of the noise-sensitivity results to distinguish the observed RMSE ordering from its possible explanation. The revised Results text relates the smaller RMSE increase of CLP-CSGM to the general role of regularization \cite{hansen2010discrete} and discusses representation, noise, and optimization errors in the context of generative compressed sensing \cite{bora2017compressed}.

We explicitly state that the sweep does not isolate the effects of the decoder restriction, the conditional penalty, and validation-based parameter selection, and that these references do not provide a recovery guarantee for our coordinate-sampling experiments. The revised text retains the advantage of interpolation at the reference noise level and identifies the larger perturbations as a controlled sensitivity analysis. No experiments or numerical results were changed.

Changes were made in the Results discussion accompanying the RMSE-versus-noise figure, in the appendix Direct linear maps, depth interpolation, and measurement noise, with the Hansen reference added to the bibliography. This addition does not define the 30dB label.

The additional comparisons also clarify the effect of increasing measurement noise on reconstruction accuracy. At the reference noise level, interpolation outperforms CLP-CSGM in all nine cross-well configurations. As the simulated perturbations increase, CLP-CSGM exhibits a smaller increase in reconstruction error and outperforms interpolation in seven of nine configurations at sigma_eta = 0.05 and in all nine at 0.10 and 0.20. Against second-difference regularized smoothing, CLP-CSGM achieves lower RMSE in five, eight, and nine configurations at those respective noise levels. These results support a regime-dependent advantage in robustness to simulated measurement noise, rather than overall superiority. The higher perturbation levels were not calibrated to field measurement uncertainties, and lambda was reselected on validation at each noise level. The score is full-window RMSE. It is not a guarantee that local peaks are recovered. The runs do not isolate the decoder, the conditional prior, or the reselected weight. Smoothing and the affine limit remain documented under R3-03. R3-P09 and R3-P10 stay separate, and R3-P16 remains open for the 30dB label.

Where the text was changed: Method, subsection Test-time reconstruction (the latent-refinement objective and the residual identity); Method, subsection Interpretation and theoretical context (the MAP proposition and the existence objective); Experimental setup, subsection Training, validation, and test protocol (the noise-sweep sentence); Results (the relative-robustness sentences next to the RMSE-versus-noise figure); Discussion (the cross-well noise paragraph); Conclusion (the noise paragraph); Appendix MAP and Tikhonov interpretation; Appendix Existence of a minimizer; Appendix Direct linear maps, depth interpolation, and measurement noise; the implementation-summary table.

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
cell means were not changed. The previous figure files are kept under
`figures/r3m06_originals/`.

Where the text was changed: Results, cross-well subsection; Discussion,
operational paragraph; Appendix, new wide-range depth-profile figure.
Script: `scripts/plot_r3m06_manuscript_figures.py`.

## R3-A01 -- Define symbols at first use

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for asking that symbols and abbreviations be defined
at first use, including in the abstract and the opening equations.

The abstract now writes root mean square error (RMSE) in full, names rho as
the fraction of window depths observed before the numerical fractions, and
replaces GR-only with gamma ray as the only dense log. The conditional latent
code z_0 is unchanged. The set-restricted eigenvalue condition is not named
in the abstract. The sentence states that the local recovery bound remains
conditional and does not show that the trained decoder separates nearby
windows. The abstract is 284 words, within the 300-word limit.

The introduction expands multilayer perceptron (MLP), maximum a posteriori
(MAP), RMSE, and the set-restricted eigenvalue condition (S-REC) at their
first use in the body. The prose account of the contributions is unchanged.
Related work expands generative adversarial networks, artificial neural
network, and Shapley additive explanations at first use. The method names
the conditional map h, the identity I_m, the training count n_tr, the
autoencoder weight gamma_AE, and the penalty weight lambda where those
symbols enter. The sampling formula uses the window length N already defined
for the length-64 windows; the count m = max{4, round(rho N)} is the same
rule as the code, which multiplies the measurement ratio by n_output. Mean
absolute error (MAE), the direct autoencoder (AE), and principal component
analysis (PCA) are expanded at first use. Clay volume is named V_c when that
symbol appears, and gamma-ray-only (GR-only) is expanded in the first caption
that uses it. M_Omega remains the coordinate-restriction operator and is not
the decoder range. No equation, table, or reported value was changed.

Where the text was changed: Abstract; Introduction; Related work; Method;
Data and exploratory analysis; Experimental setup.

## R3-A02 -- Separate sampling operator and output set

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for asking that the sampling operator be distinguished
from the decoder output set.

\(M_{\Omega}\) remains the coordinate-restriction operator: each row selects
one observed depth. The output set is no longer written \(\mathcal{M}\) or
\(\operatorname{Im}(G)\). The range of the standardized decoder is
\(\mathcal{R}_{G}=\{G(\boldsymbol{z}):\boldsymbol{z}\in\mathbb{R}^{k}\}\).
The same windows in the original target units are
\(\mathcal{R}_{\mathrm{orig}}=\{G_{\mathrm{orig}}(\boldsymbol{z}):\boldsymbol{z}\in\mathbb{R}^{k}\}\).
The local set in the recovery condition is
\(\mathcal{R}_{\mathrm{orig},R^{\dagger}}=G_{\mathrm{orig}}(B(\boldsymbol{z}_0,R^{\dagger}))\).
Each symbol is defined before it is used. The reconstruction is stated to lie
in \(\mathcal{R}_{\mathrm{orig}}\), not in the sampling operator. The notation
change did not require redrawing Figure 1. The caption names the blue curve
\(\mathcal{R}_{\mathrm{orig}}\) and states that \(M_{\Omega}\) is not that
set. Figure 1 was redrawn later under R3-A03. That redraw is not part of
this comment. This comment overlaps R3-P08, which asks that \(\operatorname{Im}\) not
be read as an imaginary part. That request is answered in its own reply.

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
regenerated by `scripts/clp_csgm_paper_assets.py`
(`plot_crosswell_target_hist`). The caption in Data and EDA was already
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

We have clarified these roles in the Introduction and Method. We also
distinguish the conditional ridge-regression map from the direct
ridge-regression baseline, which predicts target windows from auxiliary logs
and sparse observations. The ambiguous phrase in the original Abstract has
been removed.

No manuscript file was edited for this comment. The wording was already
introduced by R3-A05, and the Abstract no longer says "low-capacity
latent-prior reference".

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

These changes were introduced under the earlier methodological and notation
revisions, in particular R2-01 and R3-A01. We have verified that the symbols
used in the reconstruction equations are explained and that the Introduction
can now be read without requiring the reader to interpret the full
optimization objective. The nominal ratio rho remains distinct from the
realized fraction m/N. No manuscript file was edited for this comment.

## R3-P08 -- Im(G) may look like the imaginary part

**Status:** `text_done`

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
symbol before its definition. Figure 1 was not regenerated. The caption
uses \(\mathcal{R}_{\mathrm{orig}}\) for the curve the drawing already
labels. This comment overlaps R3-A02, which asks that the sampling operator
be kept distinct from the output set. That request is answered in its own
reply.

Where the text was changed: Related work, subsection Classical and generative
compressed sensing; Method, subsection Learned model components (the range
definition and Figure 1); Method, subsection Interpretation and theoretical
context (the representation-error remark and the conditional local-recovery
proposition); Interpretation, scope, and limitations; Appendix Conditional
local recovery interpretation.

## R3-P09 -- Measurement noise and lambda

**Status:** `text_done`

We thank the reviewer for asking how the measurement noise is set and how the regularization coefficient lambda is determined.

After coordinate subsampling, each observed coordinate receives an independent Gaussian perturbation. Its standard deviation is 0.01 for the cross-well clay-volume benchmark and 0.02 for the F03-4 porosity benchmark, in the units of the target curve. These levels were adopted for controlled benchmarks. No calibration to instrumental uncertainty is documented. The 30dB label of the input files is not this added standard deviation. R3-P16 records that label as a filename designation, not a verified signal-to-noise ratio. Within one run, every method uses the same noisy observations.

The MAP statement gives lambda as the ratio of the noise variance to the latent prior variance when the likelihood is isotropic in the original target units. The implemented residual is that original-scale residual. It equals the standardized residual weighted by the diagonal matrix of the training standard deviations at the observed coordinates. Those standard deviations vary by coordinate, so the ratio does not apply to the unweighted standardized residual. The reported runs do not use the ratio. Validation full-window RMSE selects one value from the published grid, and that value is fixed before the test score. Multiplying the whole objective by the same positive factor leaves its minimizers unchanged. The sensitivity sweep uses a different, wider grid: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, and 100. It omits the factor-of-three candidates from the published grid. A selected weight that also belongs to the published grid does not mean the two searches were the same. Selections that equal 100 are the largest candidate that was tried, not an optimum outside that grid. The covariance of this added noise, after standardization, is stated in the standardization paragraph and answered under R3-P10.

Where the text was changed: Method, subsection Test-time reconstruction (the latent-refinement objective and the residual identity); Method, subsection Interpretation and theoretical context (the MAP proposition); Appendix MAP and Tikhonov interpretation; Appendix Direct linear maps, depth interpolation, and measurement noise; the implementation-summary table.

## R3-P10 -- Covariance

**Status:** `text_done`

We thank the reviewer for asking what covariance is implied by the standardization paragraph.

D_y and D_u rescale each coordinate by its training standard deviation. They are not estimated covariance matrices of the profiles, and this rescaling leaves any correlation already present between coordinates unchanged.

For a fixed set of observed coordinates, the benchmark writes b = M y + eta with eta an isotropic Gaussian in the original target units. Independence across coordinates and a common variance are assumptions of this synthetic noise. They are not a demonstrated description of instrumental noise, and they are not identified with the 30dB filename label. R3-P16 records that label as a filename designation, not a verified signal-to-noise ratio.

The standardized noise is D_Omega inverse times eta. Because D_Omega is diagonal, its covariance is sigma_eta squared times D_Omega to the minus two. The transformation preserves independence. When the training standard deviations differ across coordinates, the standardized variances differ as well. The quadratic likelihood term built from that covariance is the implemented data term divided by sigma_eta squared, which is the term used in the MAP derivation on the original measurement scale.

The annotation is on that measurement standardization. The factor sigma_z squared times the identity, in the MAP proposition, is the isotropic hypothesis behind the Euclidean latent penalty. It is not an estimated covariance of the residuals of the conditional map. Ridge standardizes the latent codes while fitting that map and returns z_0 to the encoder scale.

G denotes the standardized decoder. G_orig(z) = mu_y + D_y G(z) denotes the same window in the original target units. The local-recovery statement and its proof use G_orig. The constants gamma_R and delta_R refer to windows in that original-scale image. The separability ratios are computed after the decoded windows are returned to the original units, and they are not those constants.

Where the text was changed: Method, subsection Problem setup (the paragraph Standardization and modeling assumptions); Method, subsection Interpretation and theoretical context (the local-recovery proposition); Appendix Conditional local recovery interpretation; Appendix Measurement Separability.

## R3-P11 -- Baseline identity

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for requesting clarification of the baseline methods.
We have clarified the distinction between the proposed CLP-CSGM configuration
and the methods used for comparison.

In CLP-CSGM Ridge, ridge regression implements the conditional map h from
dense auxiliary logs to the initial latent code z_0. It is an internal
component of the proposed method, not the direct prediction baseline. This
distinction was already stated in the Method section under R3-A05 and was
not rewritten here.

The principal direct comparator emphasized in the cross-well results is
AE [u,b], which predicts codes of a separately trained target autoencoder
from auxiliary logs and sparse observations. It has the lowest unweighted
mean RMSE among the direct baselines MLP, PCA, and AE on the reported
cross-well grid.

Other direct predictors include ML only, MLP [u,b], and PCA [u,b]. Simple
approaches, including direct linear regression, direct ridge regression,
depth interpolation, and second-difference smoothing, are also evaluated.
At the reference noise level, depth interpolation outperforms CLP-CSGM in
all nine cross-well configurations, so AE should not be interpreted as the
strongest overall comparator.

Prior-only and measurement-only CSGM are structural ablations, whereas the
MLP conditional prior is used for architectural sensitivity analysis.

The experimental description now states the role of AE and names the
additional comparators. The Method subsection on the conditional map was
left unchanged. Results, figures, and reported values were not changed.

## R3-P12 -- Why predict z rather than y?

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for asking why the auxiliary logs are mapped to a
latent code rather than directly to the target curve, and what role the
compression plays.

CLP-CSGM is built for limited-information reconstruction. The target
property is observed only at sparse depths, and the cross-well protocol
also includes configurations with limited training support. The decoder
defines the candidate target windows. The conditional map h estimates z_0
from the dense auxiliary logs. That code initializes the first trajectory
and remains the center of the latent penalty, while the sparse observations
update it through the measurement-fidelity term. The latent coordinates are
therefore the common place for learned window structure, dense-log context,
and sparse-measurement refinement. In the reported model the search adjusts
16 variables for each window of length 64.

This is the procedure intended for those regimes. On the cross-well grid,
CLP-CSGM Ridge has lower mean RMSE than AE [u,b] in seven of nine cells,
with the more consistent gains at steps 16 and 32 (52 and 27 training
windows). AE [u,b] also predicts a latent code, so that comparison
evaluates how the code is conditioned and refined. It is evidence for the
complete procedure. The step size also changes the window sampling, so the
result is not a separated effect of window count or of dimension alone.

Direct predictors can map the logs, or the logs together with the sparse
values, onto the target window. They are not assumed to ignore the sparse
values. At the reference noise 0.01, depth interpolation has lower mean
RMSE in all nine cells: when those sparse measurements already support a
simple reconstruction, that reconstruction is the stronger result. Under
the larger simulated perturbations, with lambda reselected on validation,
the complete CLP-CSGM procedure shows the smaller error growth already
reported. Those levels were not calibrated to field measurement
uncertainty.

A length-64 window can contain fewer than 16 observations, so the latent
problem can remain underdetermined. The decoder range and the penalty
around z_0 are the structural information used in that case. The
retrospective reconstruction G(E(y)), with mean test RMSE 0.0073, 0.0099,
and 0.0110 at steps 8, 16, and 32, checks whether the decoder can represent
a fully observed window. It is not an inference result.

The Method section now states this motivation. The Discussion already
records the conditional scope of the recovery statement, and the reported
numbers were not changed.

## R3-P13 -- Intermediate results

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for asking whether intermediate outputs of the
proposed methodology are reported.

The manuscript reports three architectural outputs, rather than a trace of
the optimizer iterations. No per-iteration trajectory is shown.

The retrospective autoencoder reconstruction G(E(y)) evaluates how well the
trained decoder represents a fully observed target window. The Appendix
reports mean test RMSE values of 0.0073, 0.0099, and 0.0110 for steps 8,
16, and 32. This diagnostic uses the complete target window, so it is not
an inference-time prediction.

The prior-only ablation evaluates the decoded conditional prediction
G_orig(z_0), with z_0 = h(u), before sparse-measurement refinement. The
ablation tables place that prediction next to the complete CLP-CSGM
reconstruction. In the cross-well benchmark, the full method has lower mean
RMSE than prior-only in all nine configurations, with relative reductions
from approximately 20.5% to 60.6%.

The full CLP-CSGM results evaluate the reconstruction after latent
refinement, with the decoder held fixed and with both the measurement
fidelity term and the penalty centered at z_0. Those results are in the
main comparison and in the ablation tables.

The measurement-only CSGM variant is also reported. It sets lambda to 0 and
searches from the sparse observations alone. It is a separate structural
ablation, not an intermediate stage of the complete method. The comparison
across these outputs does not isolate compression by itself.

A short pointer in the Method section now states where the retrospective
check, the prior-only prediction, and the refined reconstruction are
reported. The numerical results were not changed.

## R3-P14 -- Arrows and symbols in Figure 1

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for pointing out the ambiguities in the original
Figure 1.

The figure has been revised to distinguish the conditional latent
prediction, latent refinement, and decoding operations.

The dense auxiliary logs are mapped through h to the initial latent code
z_0. The tilde in the drawing belongs to the argument of h. It is not a
second latent code, and that argument is not what is decoded. The upper
horizontal arrow shows the decoding of z_0 into G_orig(z_0). The vertical
arrow represents latent refinement from z_0 to the refined code z_hat, and
the lower horizontal arrow shows the decoding of z_hat into the final
reconstructed window G_orig(z_hat).

The blue curve represents R_orig, the range of decoder outputs in the
original target units. This set is distinct from M_Omega, which is the
coordinate-restriction operator selecting the observed depths. The notation
previously written as Im(G) has also been replaced with explicit
decoder-range notation to avoid confusion.

The revised caption explains these mappings and the role of the measurement
residual. It also clarifies that the figure is schematic and does not depict
a recorded numerical optimization trajectory, and that the observations need
not be met exactly.

These corrections were already implemented under R3-A03 and the related
notation revisions, R3-A02 and R3-P08. We inspected Figure 1 on page 12 of
the current 90-page PDF. The labels remain readable, the arrows end at the
stated outputs, and the blue curve is R_orig. No further change was made to
the figure or to the manuscript text.

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

These changes were previously implemented under the theoretical revisions,
particularly R3-05. We have checked their consistency in the current
manuscript and made no additional changes to the propositions or proofs.

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

The change was previously implemented under R3-A04. We have verified that the corrected label is preserved in the current manuscript, and no additional figure modification was necessary.

## R3-P19 -- Figure 7 and GR-derived Vc

**Status:** `closed`

**Proposed reply:**

We thank the reviewer for raising two important concerns regarding the readability of the cross-well depth profile and the use of gamma-ray-derived clay volume as the reconstruction target.

The readability revision was implemented earlier. In the revised manuscript, the depth profile is Figure 8. Its four panels share a common clay-volume scale and depth axis, increasing downward, for the specific configuration F03-4, step 16, seed 7, and rho = 0.20. The panels show the processed reference, ML only, the direct autoencoder, and CLP-CSGM Ridge. MLP and PCA, whose substantially larger prediction ranges compressed the earlier visualization, are presented separately in supplementary Figure 24. The revised figure was inspected, and no further changes were necessary.

Regarding the target variable, we agree that clay volume in this benchmark is derived from gamma ray using the Larionov relation, following the dataset associated with Zhang et al. (2025). A correlation of approximately 0.97 between gamma ray and the processed target is therefore expected. When gamma ray is available, the Larionov relation provides the appropriate direct calculation, and we do not propose CLP-CSGM as a replacement for it.

The purpose of this experiment is different: to investigate whether a generative inverse-reconstruction framework can recover a processed petrophysical target when the input information required for its direct calculation is deliberately withheld. To evaluate this setting, we exclude gamma ray from the dense auxiliary inputs while retaining sonic, bulk density, acoustic impedance, and P-wave velocity. CLP-CSGM then combines information from these remaining logs with sparse observations of the target property to reconstruct its continuous profile.

This controlled restriction is intended to investigate the role of conditional latent information and sparse-measurement assimilation in an underdetermined reconstruction problem. It provides a methodological test of reconstruction under incomplete input information, rather than a comparison with the conventional Larionov calculation when gamma ray is available.

We also report a supplementary channel-ablation analysis in which gamma ray is reintroduced. As expected, including this channel reduces reconstruction error. At step 16, rho = 0.10, and without the boundary embargo, CLP-CSGM Ridge achieves a mean RMSE of 0.02689 without gamma ray and 0.01380 with gamma ray. The relative ordering against the direct autoencoder and measurement-only CSGM remains unchanged in this comparison.

The results also establish important limits. Although CLP-CSGM Ridge achieves lower mean RMSE than the direct autoencoder in seven of nine cross-well configurations, depth interpolation outperforms CLP-CSGM in all nine configurations at the reference measurement-noise level. We therefore do not claim general superiority over simpler reconstruction methods.

Finally, the experiment represents a controlled analogue of incomplete petrophysical information, not an operational validation in wells where gamma ray is genuinely unavailable. The sparse target observations are obtained by subsampling a processed Larionov-derived curve rather than from independent laboratory or core measurements. Accordingly, the results characterize the capabilities and limitations of CLP-CSGM under deliberately restricted inputs, while further validation with independently measured targets remains necessary.

These points are addressed in the dataset description, the cross-well results, the supplementary channel-ablation analysis, and the Discussion. We have added one sentence in the Discussion stating this purpose. The present revision does not change the underlying data, figures, or numerical results.

## AE-01 -- Associate Editor synthesis

**Status:** `closed`

The compliance matrix is `docs/revision/ae01_matriz_conformidade.md`.
The conditional-geometry schematic was prepared with ChatGPT assistance, and
xAI's Grok 4.7 was used to refactor existing scientific code. The research
implementation was not originally generated by AI. The geometry caption, the
general declaration, and Implementation details now state those uses. No
result was changed.

**Proposed reply:**

We thank the Associate Editor for the careful assessment of our manuscript and for identifying the principal limitations that needed to be addressed.

We have revised the manuscript to clarify the scope of the proposed framework and to qualify its empirical evaluation.

First, the main F3 benchmarks reconstruct processed petrophysical targets rather than independently measured laboratory properties. Clay volume is derived from gamma ray using the Larionov relation, and the F03-4 porosity target is density-derived. Sparse observations in these benchmarks are coordinate subsamples of the processed target curves. Gamma ray was available and was deliberately withheld from the main cross-well inputs, so the experiment is a controlled test of reconstruction under incomplete information. It is not a replacement for the Larionov relation when gamma ray is available, and it is not evidence of operational transfer. A separate laboratory-porosity supplement on one cored interval identifies the target as laboratory porosity in its source, but the primary assay records were not independently verified. On the pooled mean, the direct autoencoder has the lower RMSE. CLP-CSGM improves on the decoded conditional prediction in every block-seed pair of that supplement. That improvement does not establish general operational superiority.

Second, the revised comparisons include direct linear regression, direct ridge regression, depth interpolation, second-difference smoothing, and the affine limit of that smoother, together with the direct autoencoder and the prior-only and measurement-only ablations. These comparisons qualify the results. CLP-CSGM Ridge has lower mean RMSE than the direct autoencoder in seven of nine cross-well configurations, chiefly where training support is limited, and is higher in two. At the reference noise standard deviation 0.01, depth interpolation has lower full-window RMSE than CLP-CSGM in all nine configurations. We do not claim general superiority over simpler or linear methods. Excluding gamma ray also means that this benchmark is not a contest against the Larionov calculation.

Third, the cross-well evidence uses three development wells and one held-out well, F03-4. Windows from that well, model seeds, and repeated draws of the sparse coordinates are not additional wells. The ten-seed panel and the five-draw placement panel are robustness checks inside this benchmark. Each draw changes observation coordinates and measurement noise together, so it does not isolate placement from noise. These checks do not demonstrate geological generalization across fields.

Fourth, the presentation now separates the architectural motivation from the empirical findings. CLP-CSGM combines a learned generative representation, a latent code predicted from the auxiliary logs, and refinement that assimilates sparse target observations. Classical MAP and existence statements are distinguished from the conditional recovery bound, and that bound is not presented as a verified guarantee for the trained decoder. The abstract and conclusion state the same regime-dependent limits.

Finally, the manuscript contains a declaration of generative-AI assistance in manuscript preparation. It identifies OpenAI's ChatGPT as assisting with translation into English, improvement of language and organization, reduction of manuscript length to satisfy journal requirements, and preparation of selected explanatory schematic figures. The corresponding figure captions identify this assistance. The manuscript also now states that xAI's Grok 4.7 was used to assist with refactoring existing scientific code; the implementation was not originally generated by AI. The authors reviewed all AI-assisted modifications and remain fully responsible for the scientific methodology, computational implementation, reported results, and conclusions. The AI tools did not generate the research data, run the reported experiments, or determine the scientific conclusions.

Taken together, these revisions present CLP-CSGM as a methodological investigation of conditional generative reconstruction with explicit sparse-measurement assimilation. The results identify settings in which the formulation has lower error than the direct autoencoder and settings in which interpolation remains preferable. Broader operational validation still requires additional wells and independently measured petrophysical targets.
