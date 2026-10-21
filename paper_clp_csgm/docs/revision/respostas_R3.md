# Reviewer 3 -- proposed replies

Each ID keeps its own reply, including PDF annotations R3-P01..P19. If the
same manuscript change fully answers more than one ID, the English reply is
copied verbatim.

Every completed reply names where the manuscript text was changed (Abstract,
section title, or appendix subsection title), not only what was changed.

Source comments: `docs/revision/CLP_CSGM_Pareceres_Consolidados.md`.

---

## R3-01 -- Clarity and stacked terminology

**Status:** `text_done`

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
scope, and limitations, and the Conclusion. The reported numbers and the
limits already stated in the abstract are unchanged: the advantage is
conditional on small training support, the local bound is not an S-REC proof,
and the gamma-ray-only porosity test is not general porosity performance.

Where the text was changed: Abstract; Introduction; Related work; Method
(the opening, the paragraph after the measurement model, the conditional-code
figure caption, Test-time reconstruction, and the ablation paragraph);
Experimental setup (the protocol caption, the restart paragraph, and the
implementation table); Results (Ablation analysis); Interpretation, scope,
and limitations; Conclusion; Appendix Supplementary Theoretical Details
(the opening paragraph); Appendix Laboratory porosity on a cored interval
(the definition of the decoded prediction).

## R3-02 -- Practical advantage versus conventional relations

**Status:** `text_done`

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

For the F03-4 porosity, density porosity is linear in bulk density, and a
standard sonic porosity is linear in slowness. If those logs have been
recorded, the processed porosity is already determined. In this dataset,
sonic travel time follows the processed porosity almost one to one, and the
density curve behind it was predicted from sonic and gamma ray. Sonic is left
out, and gamma ray is the only dense log. The errors do not transfer to a
porosity interpretation that also uses density, neutron, and sonic.

The two F3 benchmarks therefore compare methods that face this incomplete
input. They are not a contest against Larionov, density porosity, or sonic
porosity on a fully logged well.

Where the text was changed: Data and exploratory analysis (the clay-volume
input-design paragraph); Experimental setup (the F03-4 GR-only porosity
paragraph); Interpretation, scope, and limitations (the empirical-scope
paragraphs for the two F3 benchmarks).

## R3-03 -- Simple comparators and intermediate results

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for asking for these comparators and for the role of the encoder.

The encoder is used only in training. It builds the code of the known windows, and the dense-log map is fit on those codes. At inference the unknown window is not encoded. The code has dimension 16. Passing the full window through the encoder and the decoder only checks that the decoder can represent the curve. That check is not part of the comparison.

The experiment uses the same cross-well grid as the paper. We added linear regression and direct ridge regression from the dense logs and the sparse measurements, and interpolation of those measurements along depth. We then kept the windows and the measured positions fixed and raised only the noise.

At the published noise, interpolation has the lowest error in every cell. The measurements are a slightly perturbed copy of the same curve, so connecting them is enough. CLP-CSGM is ahead of both linear maps only in the few-window cells: at step 16 with the largest measurement fraction, and at step 32.

When the noise rises to the same order as the variation of the curve, the interpolation advantage falls in the few-window cells, and also at step 8 with the largest measurement fraction. The penalty weight selected for this comparison stays inside the published grid.

At step 8, with the two smaller measurement fractions, interpolation still has the lower mean at that noise. The decoder can represent the curve. The window predicted from the other logs alone has a larger error than the interpolation, and the CLP-CSGM error stays between those two error values. With the larger measurement fraction, the CLP-CSGM mean falls below the interpolation. At steps 16 and 32 the dense-log window has an error of similar magnitude to the interpolation. Similar magnitudes do not show that the reconstructions are close to each other. That pattern is compatible with a weak dense-log window. It does not isolate that window as the only cause.

At the two larger noise levels, CLP-CSGM has the lower error in every cell. These noise levels were used for sensitivity analysis and were not calibrated to field measurement uncertainties. At the largest of them, the noise is several times the variation of the curve.

Results also shows this ordering in one figure, with one panel per step and one curve per measurement fraction, so the step-8 cells are not averaged away. The horizontal axis is logarithmic. Relative to the curve standard deviation 0.035, on the same scale as the noise, 0.05 is about 1.43 times larger, and 0.10 and 0.20 are about 2.86 and 5.71 times larger. The decoder and the dense-log map are fit once per step and seed. The penalty is selected again on validation for each measurement fraction and each noise level. The noise 0.01 points in that figure are this refit. They keep the same ordering as the published grid and can differ slightly from its cell means.

The detail is in the appendix Direct linear maps, depth interpolation, and measurement noise, with the two tables and one depth example. The Results paragraph states the cell-level ordering next to that figure. The Abstract states that interpolation has lower error in all nine configurations at the reference noise 0.01, and that under larger perturbations that comparison changes. Those noise levels were used for sensitivity analysis and were not calibrated to field measurement uncertainties. The Conclusion keeps the step-8 exceptions at noise 0.05 and states that AE, ordinary least squares, and direct ridge were not rerun across the sweep and that no smoother was added. The Experimental setup announces the experiment, and the Method explains the encoder. These runs are not merged into the submitted tables. This wording does not close the question of how the published noise level was chosen.

We have revised the discussion of the noise-sensitivity results to distinguish the observed RMSE ordering from its possible explanation. The revised Results text relates the smaller RMSE increase of CLP-CSGM to the general role of regularization \cite{hansen2010discrete} and discusses representation, noise, and optimization errors in the context of generative compressed sensing \cite{bora2017compressed}.

We explicitly state that the sweep does not isolate the effects of the decoder restriction, the conditional penalty, and validation-based parameter selection, and that these references do not provide a recovery guarantee for our coordinate-sampling experiments. The revised text retains the advantage of interpolation at the reference noise level and identifies the larger perturbations as a controlled sensitivity analysis. No experiments or numerical results were changed.

Changes were made in the Results discussion accompanying the RMSE-versus-noise figure, in the appendix Direct linear maps, depth interpolation, and measurement noise, with the Hansen reference added to the bibliography.

Where the text was changed: Abstract (the sentence on interpolation at noise 0.01); Method (the generative-decoder paragraph); Experimental setup (the paragraph after the sparse-coordinate placement check); Results (the discussion accompanying the RMSE-versus-noise figure); Conclusion (the noise paragraph); Appendix Direct linear maps, depth interpolation, and measurement noise.

**Shared reply:** none. `R3-04` is answered in its own entry. This reply does not close it.

## R3-04 -- Noise choice and prior/data balance

**Status:** `text_done`

We thank the reviewer for asking how the noise level was chosen and how that choice balances the prior against the data fit.

The published cross-well clay-volume benchmark adds independent Gaussian noise after coordinate subsampling. The standard deviation is 0.01 in the units of the clay-volume curve, not one percent of the curve. The held-out curve has standard deviation about 0.035, so 0.01 is about 0.29 times that deviation. The F03-4 porosity benchmark uses the same construction with standard deviation 0.02. These levels were adopted for controlled benchmarks. No calibration to instrumental uncertainty is documented. The input files are labeled 30dB. That label is not the added standard deviation, and this manuscript does not define the procedure behind the label. That definition remains open under R3-P16. Within one run, every compared method receives the same noisy observations. Direct models from the concatenated dense logs and sparse observations standardize those features after the noise is added.

The MAP statement identifies lambda with the ratio of the noise variance to the latent prior variance when the likelihood is an isotropic Gaussian in the original target units. On that scale the implemented residual equals a standardized residual weighted by the diagonal matrix of the training standard deviations at the observed coordinates. Those standard deviations are not equal. On the cross-well clay-volume training windows, the ratio of the largest to the smallest is about 1.13 at step 8, 1.23 at step 16, and 1.40 at step 32, so one scalar conversion does not turn the unweighted standardized residual into the implemented objective. The displayed objective now uses that weighted residual. The experiments do not compute lambda from the variance ratio. For each seed and measurement ratio, validation full-window RMSE selects one value from the grid 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, and 1e-1, and that value is fixed before the test evaluation. Dividing both objective terms by the number of windows multiplies the objective by one positive factor. That factor leaves the minimizers unchanged. The covariance of this added noise, after standardization, is stated in the standardization paragraph and answered under R3-P10.

The later noise sweep is a sensitivity check of the adopted noise level, not the reason that level was chosen. The sensitivity sweep uses a different, wider grid: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, and 100. It omits the factor-of-three candidates from the published grid. At 0.01 and 0.05 every selected weight also belongs to the published grid. That membership does not mean the two searches were the same. At noise 0.10, step 16 and measurement fraction 0.10, one seed selects the weight 1, which is above the published grid and still inside the sweep. At noise 0.20, some cells select 100, the largest candidate that was tried. That selection is not an optimum outside the searched grid. The interpolation and CLP-CSGM ordering depends on the noise level and the configuration, as already reported next to the RMSE-versus-noise figure and the noise-sweep table. No new experiments were added, and the algorithm and the reported numbers are unchanged.

We have revised the discussion of the noise-sensitivity results to distinguish the observed RMSE ordering from its possible explanation. The revised Results text relates the smaller RMSE increase of CLP-CSGM to the general role of regularization \cite{hansen2010discrete} and discusses representation, noise, and optimization errors in the context of generative compressed sensing \cite{bora2017compressed}.

We explicitly state that the sweep does not isolate the effects of the decoder restriction, the conditional penalty, and validation-based parameter selection, and that these references do not provide a recovery guarantee for our coordinate-sampling experiments. The revised text retains the advantage of interpolation at the reference noise level and identifies the larger perturbations as a controlled sensitivity analysis. No experiments or numerical results were changed.

Changes were made in the Results discussion accompanying the RMSE-versus-noise figure, in the appendix Direct linear maps, depth interpolation, and measurement noise, with the Hansen reference added to the bibliography. This addition does not define the 30dB label and does not add a smoothing comparator.

Where the text was changed: Method, subsection Test-time reconstruction (the latent-refinement objective and the residual identity); Method, subsection Interpretation and theoretical context (the MAP proposition and the existence objective); Experimental setup, subsection Training, validation, and test protocol; Results (the discussion accompanying the RMSE-versus-noise figure); Discussion (the measurement-term sentence); Appendix MAP and Tikhonov interpretation; Appendix Existence of a minimizer; Appendix Direct linear maps, depth interpolation, and measurement noise; the implementation-summary table.

## R3-05 -- Role of the theory section

**Status:** `text_done`

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

**Status:** `pending`

## R3-A01 -- Define symbols at first use

**Status:** `pending`

## R3-A02 -- Separate sampling operator and output set

**Status:** `text_done`

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
in \(\mathcal{R}_{\mathrm{orig}}\), not in the sampling operator. Figure 1
was not redrawn. Its caption now names the blue curve
\(\mathcal{R}_{\mathrm{orig}}\) and states that \(M_{\Omega}\) is not that
set. This comment overlaps R3-P08, which asks that \(\operatorname{Im}\) not
be read as an imaginary part. That request is answered in its own reply.

Where the text was changed: Related work, subsection Classical and generative
compressed sensing; Method, subsection Learned model components (the range
definition and Figure 1); Method, subsection Interpretation and theoretical
context (the representation-error remark and the conditional local-recovery
proposition); Interpretation, scope, and limitations; Appendix Conditional
local recovery interpretation.

## R3-A03 -- Redraw Figure 1

**Status:** `text_done`

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

**Status:** `pending`

## R3-A05 -- State that Ridge is ridge regression

**Status:** `pending`

## R3-P01 -- Unclear objective in the Abstract

**Status:** `pending` (R1-M01 adds target origin in the Abstract; the
objective/clarity request is not closed)

## R3-P02 -- Framing as an inverse problem

**Status:** `pending`

## R3-P03 -- Admissible reconstruction set

**Status:** `pending`

## R3-P04 -- Meaning of rho in the Abstract

**Status:** `pending`

## R3-P05 -- Weak single-channel context

**Status:** `pending`

## R3-P06 -- Ridge regression and low-capacity prior

**Status:** `pending`

## R3-P07 -- Symbols in the introductory equations

**Status:** `pending`

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

After coordinate subsampling, each observed coordinate receives an independent Gaussian perturbation. Its standard deviation is 0.01 for the cross-well clay-volume benchmark and 0.02 for the F03-4 porosity benchmark, in the units of the target curve. These levels were adopted for controlled benchmarks. No calibration to instrumental uncertainty is documented. The 30dB label of the input files is not this added standard deviation. The procedure behind that label remains open under R3-P16. Within one run, every method uses the same noisy observations.

The MAP statement gives lambda as the ratio of the noise variance to the latent prior variance when the likelihood is isotropic in the original target units. The implemented residual is that original-scale residual. It equals the standardized residual weighted by the diagonal matrix of the training standard deviations at the observed coordinates. Those standard deviations vary by coordinate, so the ratio does not apply to the unweighted standardized residual. The reported runs do not use the ratio. Validation full-window RMSE selects one value from the published grid, and that value is fixed before the test score. Multiplying the whole objective by the same positive factor leaves its minimizers unchanged. The sensitivity sweep uses a different, wider grid: 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, and 100. It omits the factor-of-three candidates from the published grid. A selected weight that also belongs to the published grid does not mean the two searches were the same. Selections that equal 100 are the largest candidate that was tried, not an optimum outside that grid. The covariance of this added noise, after standardization, is stated in the standardization paragraph and answered under R3-P10.

Where the text was changed: Method, subsection Test-time reconstruction (the latent-refinement objective and the residual identity); Method, subsection Interpretation and theoretical context (the MAP proposition); Appendix MAP and Tikhonov interpretation; Appendix Direct linear maps, depth interpolation, and measurement noise; the implementation-summary table.

## R3-P10 -- Covariance

**Status:** `text_done`

We thank the reviewer for asking what covariance is implied by the standardization paragraph.

D_y and D_u rescale each coordinate by its training standard deviation. They are not estimated covariance matrices of the profiles, and this rescaling leaves any correlation already present between coordinates unchanged.

For a fixed set of observed coordinates, the benchmark writes b = M y + eta with eta an isotropic Gaussian in the original target units. Independence across coordinates and a common variance are assumptions of this synthetic noise. They are not a demonstrated description of instrumental noise, and they are not identified with the 30dB filename label. The origin of that label remains open under R3-P16.

The standardized noise is D_Omega inverse times eta. Because D_Omega is diagonal, its covariance is sigma_eta squared times D_Omega to the minus two. The transformation preserves independence. When the training standard deviations differ across coordinates, the standardized variances differ as well. The quadratic likelihood term built from that covariance is the implemented data term divided by sigma_eta squared, which is the term used in the MAP derivation on the original measurement scale.

The annotation is on that measurement standardization. The factor sigma_z squared times the identity, in the MAP proposition, is the isotropic hypothesis behind the Euclidean latent penalty. It is not an estimated covariance of the residuals of the conditional map. Ridge standardizes the latent codes while fitting that map and returns z_0 to the encoder scale.

G denotes the standardized decoder. G_orig(z) = mu_y + D_y G(z) denotes the same window in the original target units. The local-recovery statement and its proof use G_orig. The constants gamma_R and delta_R refer to windows in that original-scale image. The separability ratios are computed after the decoded windows are returned to the original units, and they are not those constants.

Where the text was changed: Method, subsection Problem setup (the paragraph Standardization and modeling assumptions); Method, subsection Interpretation and theoretical context (the local-recovery proposition); Appendix Conditional local recovery interpretation; Appendix Measurement Separability.

## R3-P11 -- Baseline identity

**Status:** `pending`

## R3-P12 -- Why predict z rather than y?

**Status:** `pending`

## R3-P13 -- Intermediate results

**Status:** `pending`

## R3-P14 -- Arrows and symbols in Figure 1

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for pointing out that the previous drawing was hard
to follow.

The tilde in the drawing belongs to the dense-log argument of \(h\). It is
not a second latent code, and that argument is not what is decoded. The
upper horizontal arrow decodes \(\boldsymbol{z}_0\) to
\(G_{\mathrm{orig}}(\boldsymbol{z}_0)\). The lower horizontal arrow decodes
\(\widehat{\boldsymbol{z}}\) and ends at
\(\widehat{\boldsymbol{y}}=G_{\mathrm{orig}}(\widehat{\boldsymbol{z}})\) on
the blue curve. \(M_{\Omega}\) and \(\mathcal{R}_{\mathrm{orig}}\)
are different objects. \(M_{\Omega}\) selects the observed coordinates.
\(\mathcal{R}_{\mathrm{orig}}\) is the blue curve, the set of
decoder outputs in the original target units. The drawing no longer
identifies them with one symbol.

Where the text was changed: Method, subsection Learned model components
(Figure 1).

## R3-P15 -- Confusing propositions

**Status:** `text_done`

**Proposed reply:**

We thank the reviewer for saying that the list of propositions was
confusing.

The subsection now opens with the three questions the theory is meant to
answer: how to read the two terms of the objective, when a minimizer
exists, and which factors condition recovery of the full window. The
MAP/Tikhonov and existence arguments are identified there as classical
interpretations, not as new theorems. Their formal statements were moved
to the appendix, each immediately before its proof. In the body, one
paragraph states the MAP reading and its limits, and another states what
existence does and does not give, including the distinction between a
global minimizer and the numerical solution. Before the conditional
local-recovery proposition, the text names the four factors in the bound:
representation error, noise and observed discrepancy, deviation from the
conditioned code, and local separability. That proposition remains in the
body, with its hypotheses. The paragraph after it states that the result
is conditional, that it does not prove S-REC on the trained decoder, that
the diagnostics do not certify the condition for every pair of windows,
and that the global-minimizer statement is separate from the
inexact-optimizer extension. No new guarantee and no new experiment were
added. This comment overlaps R3-05, which asks how the classical arguments
help the application. That request is answered in its own reply.

Where the text was changed: Method, subsection Interpretation and
theoretical context; Appendix MAP and Tikhonov interpretation; Appendix
Existence of a minimizer.

## R3-P16 -- Meaning of 30 dB

**Status:** `pending`

## R3-P17 -- Definition of Vc

**Status:** `pending` (Abstract/Conclusions now name Larionov `Vc`; Data
already defined it. Item stays open until this ID is processed)

## R3-P18 -- Density is ambiguous

**Status:** `pending`

## R3-P19 -- Figure 7 and GR-derived Vc

**Status:** `pending`
