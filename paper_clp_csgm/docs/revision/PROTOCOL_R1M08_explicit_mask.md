# R1-M08 supplement -- explicit-mask direct AE

This note describes the R1-M08 check. It does not add a method to the
main comparison grid.

Reviewer wording: the direct baselines use fixed coordinates with implicit
positional identity. A stronger direct baseline that encodes the sampling
mask or coordinates would show whether the CLP-CSGM gain comes from the
latent inverse formulation or from better use of coordinate information.

## Do not touch

- Published `results_crosswell_clp_vs_ae.*` and `paired_delta_crosswell_*`
- R1-M03 split runs and R1-M04 seed-panel runs
- F03, Lapa, ablations, MLP latent prior

## What is compared

Same nine cross-well cells as the paper: steps {8, 16, 32},
rho in {0.05, 0.10, 0.20}, seeds {7, 23, 41}, tail validation split,
channels sonic,rhob,ai,vp, F03-4 held out.

Three methods share M and the measurement noise:

- ridge_prior_csgm
- ae_regression_ub, input [u, b] with length p+m
- ae_regression_ub_mask, input [u, b_placed, mask]

b_placed is length N: the noisy measurement at each observed coordinate and
zero elsewhere. The mask is the binary indicator of Omega, repeated on every
window. Omega stays fixed inside each (seed, rho) cell.

The flag `--run-ae-explicit-mask` is off unless this supplement asks for it.

## Commands

```bash
bash scripts/run_r1m08_explicit_mask.sh
```

## Appendix table

One row per (step, rho). Report mean RMSE for CLP, AE [u,b], and
AE [u,b,mask], plus paired gaps. Do not merge the new method into the
main Results tables.
