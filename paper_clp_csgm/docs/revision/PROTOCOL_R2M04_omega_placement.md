# R2-04 supplement -- sparse-coordinate placement

This note describes the R2-04 check. It does not replace the submitted
3-seed 9-cell grid.

Reviewer wording: one coordinate set is sampled for each seed and
measurement ratio and then kept fixed across windows. The conclusions may
depend on where those sparse observations fall.

This protocol redraws that coordinate set while the model seed stays fixed.
It does not rerun MLP as a latent prior, F03-4, Lapa, or the main tables.

## What stays fixed

- Train wells F02-1, F03-2, F06-1 and held-out test well F03-4
- Channels sonic, rhob, ai, vp
- Target Vc
- Steps 8, 16, 32 and ratios 0.05, 0.10, 0.20
- Model seeds 7, 23, 41 (decoder, ridge anchor, direct AE initialization)
- Tail validation split
- Full-window RMSE, the same metric as the published grid

## What changes

Five draws of Omega and of the measurement noise. Draw 0 uses
measurement seed = model seed, which is the published stream. Draws 1
to 4 add a fixed offset to the model seed and use that sum only for
Omega and the noise on b. The direct AE is fit again on each draw
because b is part of its input. Lambda for CLP-CSGM is selected again
on the validation windows of that same draw.

| Draw | Offset |
|------|--------|
| 0 | 0 |
| 1 | 10007 |
| 2 | 20011 |
| 3 | 30013 |
| 4 | 40009 |

## Layout

```
outputs/reviewer_revision/r2m04_omega_placement/
  PROTOCOL.txt
  config_frozen.json
  runs/
    offset00000_step08/
    offset00000_step16/
    offset00000_step32/
    ...
  aggregate/tables/
    placement_detail.csv
    summary_by_cell.csv
    omega_sets.csv
```

Each run folder keeps the benchmark PROTOCOL, DATASET_MANIFEST,
RUN_MANIFEST, detailed_results.csv, and tables/omega_sets.csv
(the coordinate order seen by both methods).

## Command

```bash
bash scripts/run_r2m04_omega_placement.sh
```

Do not merge these numbers into the main Results tables.
