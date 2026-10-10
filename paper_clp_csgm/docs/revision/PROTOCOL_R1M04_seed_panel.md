# R1-M04 optional supplement -- 10-seed CLP vs AE panel

This note describes the optional R1-M04 check. It does not replace the
submitted 3-seed 9-cell grid.

Reviewer wording: n=3 Student-t intervals are still small; increase the
seed count if computationally feasible, especially because several
intervals include zero and the ranking changes by regime.

This protocol expands the seed panel on the same cross-well CLP-CSGM Ridge
versus AE [u,b] comparison. It does not rerun MLP, PCA, ablations, F03-4,
or Lapa.

## Do not touch

- `outputs/cross_well_vc/direct_ub/runs/crosswell_step{08,16,32}_*`
- `scripts/clp_csgm_paper_assets.py` main tables
  (`results_crosswell_clp_vs_ae.*`, `paired_delta_crosswell_clp_vs_ae.*`)
- F03, Lapa, ablation, runtime, S-REC, R1-M03 split assets

Those remain the submitted numbers.

## Artifact chain

```
1) launcher run
   sir_cs_benchmark_multi_well_vc.py
        |
        v
2) raw run folder per step (PROTOCOL, DATASET_MANIFEST, RUN_MANIFEST,
   tables/summary_by_seed.csv)
        |
        v
3) aggregate under outputs/reviewer_revision/r1m04_seed_panel/
        |
        v
4) paper copy robustness_crosswell_seed_panel.csv/.tex
        |
        v
5) LaTeX \input in sections/09_appendix.tex (new subsection)
```

## Layout

```
outputs/reviewer_revision/r1m04_seed_panel/
  PROTOCOL.txt
  config_frozen.json
  runs/
    step08/
    step16/
    step32/
  aggregate/tables/
    per_seed_focus.csv
    paired_delta_clp_vs_ae.csv
    summary_by_cell.csv
paper_clp_csgm/tables/
  robustness_crosswell_seed_panel.csv
  robustness_crosswell_seed_panel.tex
```

## Frozen knobs

| Knob | Value | Why |
|------|-------|-----|
| train/test files | F3 30 dB tables | same wells |
| channels | sonic,rhob,ai,vp | main no-GR protocol |
| target | vc | same |
| steps | 8, 16, 32 | full 9-cell grid |
| rhos | 0.05, 0.10, 0.20 | same three ratios |
| seeds | 7,23,41,13,29,31,37,43,47,53 | paper three plus seven extras |
| val_split_mode | tail | submitted protocol |
| methods reported | ridge_prior_csgm, ae_regression_ub | R1-M04 pair |
| test well | F03-4 | never in train/val |
| device | CPU | CUDA unavailable on this host |

Omega is sampled per (seed, rho), as in the paper. Extra seeds therefore
also redraw the sparse coordinates. That is not an isolated placement
sweep (R2-04).

## Commands

```bash
bash scripts/run_r1m04_seed_panel.sh
```

The shell script calls `scripts/clp_csgm_r1m04_seed_panel_assets.py` at
the end.

## Appendix table

One row per (step, rho). Columns: n_seeds, mean paired DeltaRMSE (AE-CLP),
seed-level SE, Student-t 95% half-width with df=9, CLP wins, AE wins, and
whether the interval includes zero.

Do not merge this table into the main Results 3-seed numbers.
