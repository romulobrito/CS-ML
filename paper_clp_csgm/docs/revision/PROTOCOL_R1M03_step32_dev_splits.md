# R1-M03 optional supplement -- step-32 development-split resampling

This note describes how to run the optional R1-M03 check so that artifacts
follow the same chain already used for the paper and for Appendix B.7
(embargo). It does not replace the main 9-cell grid.

Reviewer wording: repeated resampling of the small development pool
(~27 windows at step 32) would make the low-data gains more convincing.
This protocol is that resampling branch. It is not a new held-out well.

## Do not touch

- `outputs/cross_well_vc/direct_ub/runs/crosswell_step{08,16,32}_clp_csgm_ablation_ridge/`
- `outputs/cross_well_vc/csgm/m2_grid/`
- `scripts/clp_csgm_paper_assets.py` main tables
  (`results_crosswell_clp_vs_ae.*`)
- F03, Lapa, ablation, runtime, S-REC assets

Those remain the submitted numbers. The supplement is a sibling of
`outputs/reviewer_blindagem/`, not a overwrite of the production runs.

## Artifact chain (same as today)

```
1) launcher run
   sir_cs_benchmark_multi_well_vc.py
        |
        v
2) raw run folder (PROTOCOL.txt, DATASET_MANIFEST.txt, RUN_MANIFEST.txt,
   tables/summary_focus_clp_csgm_vs_ub.csv, tables/detailed_results.csv)
        |
        v
3) aggregate (like outputs/reviewer_blindagem/comparisons_embargo/)
        |
        v
4) paper copy (like robustness_crosswell_step16_gr_embargo.csv/.tex)
        |
        v
5) LaTeX \input in sections/09_appendix.tex  (new subsection, not B.7)
```

## Layout

```
outputs/reviewer_revision/r1m03_step32_dev_splits/
  PROTOCOL.txt                         # this protocol, copied at freeze time
  config_frozen.json                   # step, rhos, Omega seed, n_splits
  runs/
    split00/                           # --run-id r1m03_step32_split00
      PROTOCOL.txt
      DATASET_MANIFEST.txt
      RUN_MANIFEST.txt
      config.json
      tables/
        detailed_results.csv
        summary.csv
        summary_focus_clp_csgm_vs_ub.csv
    split01/
    ...
    split09/                           # 10 splits; add split10..19 if n=20
  aggregate/
    tables/
      per_split_focus.csv              # long table, one row per split x rho x method
      summary_by_rho.csv               # mean/std of RMSE across splits
      paired_delta_clp_vs_ae.csv       # DeltaRMSE = AE - CLP per split x rho
    figures/
      rmse_by_split_rho.png            # optional; not a main-paper figure
paper_clp_csgm/tables/
  robustness_crosswell_step32_dev_splits.csv
  robustness_crosswell_step32_dev_splits.tex
paper_clp_csgm/docs/revision/
  PROTOCOL_R1M03_step32_dev_splits.md  # this file
```

Keep `--no-plots` on each split run. The 01--10 PNG set is for the main
benchmark, not for this supplement.

## Frozen knobs (must match the submitted step-32 cell)

| Knob | Value | Why |
|------|-------|-----|
| train/test files | `data/F02-1,F03-2,F06-1_6logs_30dB.txt`, `data/F03-4_6logs_30dB.txt` | same wells |
| channels | `sonic,rhob,ai,vp` | main no-GR protocol |
| target | `vc` | same |
| window / step | L=64, step=32 | the ~27-window cell |
| rhos | 0.05, 0.10, 0.20 | same three ratios |
| Omega | **fixed** from paper seed 7 | isolate the train/val factor |
| measurement noise | 0.01 | paper cross-well |
| methods | `ridge_prior_csgm` and `ae_regression_ub` only | R1-M03 pair |
| val_frac | 0.10 | same budget (~3 val windows) |
| test well | F03-4 entire hold-out | never reshuffled into train/val |

Do not vary Omega, GR inclusion, embargo, prior class, or F03-4 membership
in this folder. Those are other reviewer IDs.

## Commands (after the CLI flag exists)

```bash
# example, split k = 0..9
python sir_cs_benchmark_multi_well_vc.py \
  --train-path data/F02-1,F03-2,F06-1_6logs_30dB.txt \
  --test-path data/F03-4_6logs_30dB.txt \
  --channels sonic,rhob,ai,vp \
  --target vc \
  --step 32 \
  --rhos 0.05,0.10,0.20 \
  --seeds 7 \
  --val-frac 0.1 \
  --val-split-mode random_per_well \
  --val-split-seed 0 \
  --run-csgm-m2 \
  --csgm-prior-type ridge \
  --no-lfista \
  --no-plots \
  --no-parity \
  --base-dir outputs/reviewer_revision/r1m03_step32_dev_splits \
  --run-id split00
```

Repeat `--val-split-seed` 1..9 and `--run-id split01` .. `split09`, or:

```bash
bash scripts/run_r1m03_step32_dev_splits.sh
# optional: N_SPLITS=20 bash scripts/run_r1m03_step32_dev_splits.sh
```

Then:

```bash
python scripts/clp_csgm_r1m03_step32_split_assets.py
```

The shell script already calls the aggregator at the end.

The compact mean/std table is copied to `paper_clp_csgm/tables/` as
`robustness_crosswell_step32_dev_splits.csv` and `.tex`, analogous to
`robustness_crosswell_step16_gr_embargo.tex`.

`--val-split-mode` and `--val-split-seed` are implemented in
`multi_well_vc.py` and `sir_cs_benchmark_multi_well_vc.py`. Default remains
`tail` (submitted paper). Do not pass `--val-embargo-windows` with
`random_per_well`.

## What the appendix table should show

One row per `rho`. Columns:

- n_splits
- n_train / n_val (from DATASET_MANIFEST; should stay ~24 / ~3)
- CLP RMSE mean +/- std across splits
- AE RMSE mean +/- std across splits
- mean paired DeltaRMSE (AE - CLP)
- fraction of splits with CLP < AE

That is enough for the reviewer letter. Do not add this table to the main
Results section and do not merge it into `results_crosswell_clp_vs_ae.tex`.

## LaTeX hook

New appendix subsection, e.g. `app:crosswell_step32_dev_splits`, with
`\input{tables/robustness_crosswell_step32_dev_splits.tex}`.
Point from the step-32 paragraph in `sections/05_experiments.tex` the same
way B.7 is pointed from the embargo sentence. Keep B.7 unchanged.

## Freeze checklist

- [ ] PROTOCOL.txt copied into the aggregate folder
- [ ] each split has DATASET_MANIFEST + RUN_MANIFEST
- [ ] Omega seed recorded as 7 in config_frozen.json
- [ ] paper csv/tex checksum or row count matches aggregate
- [ ] main `results_crosswell_clp_vs_ae` files unchanged (`git diff`)
