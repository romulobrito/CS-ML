# DEM lab inverse calibration -- Well 861

Fits per-HFU DEM parameters (aspect ratio alpha and matrix scale) by minimizing
RMSE of Vp_DEM vs Vp_lab from ROCKPHYS_Database (22.1 MPa, Z-axis, dry).

Planning: `methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md`

## Regenerate

```bash
python methods_comparison/scripts/run_861_dem_sc_lab_calibration.py
python methods_comparison/scripts/run_861_dem_sc_lab_calibration.py --robust
python methods_comparison/scripts/run_861_dem_sc_profile_87.py --lab-calib
```

## Layout

```
lab_calibration/
  MANIFEST.txt
  metrics.json              # last run
  metrics_standard.json     # all 10 plugs
  metrics_robust.json       # excludes F2911V outlier
  tables/
    hfu_calibrated_params.csv
    plug_validation_calibrated.csv
    loo_validation.csv          # leave-one-plug-out (honest error)
  figures/
    vp_before_after_calib.png
    alpha_calibrated_by_hfu.png
    vp_error_before_after.png
    mape_insample_vs_loo.png
    loo_error_by_sample.png

hfu_calibration/
  hfu_lab_calibrated.csv         # standard (10 plugs)
  hfu_lab_calibrated_robust.csv  # robust (9 plugs)

profile_87_lab_calib/            # 87-row profile with lab-calibrated HFU params
```

## Results (production)

| Mode | Plugs | MAPE before | MAPE in-sample | MAPE LOO | RMSE LOO |
|------|-------|-------------|----------------|----------|----------|
| Standard | 10 | 26.0% | 9.1% | **14.8%** | 0.85 km/s |
| Robust (--robust) | 9 | 22.7% | 7.5% | (re-run) | ~0.50 km/s |

**LOO (leave-one-plug-out)** is the honest generalization metric: each plug is
predicted with HFU parameters fit on the other plugs only. Expect LOO MAPE
between in-sample (~9%) and uncalibrated (~26%).

Chosen scenario per HFU: `alpha_matrix_scale` (joint fit alpha + uniform K/G scale).

| HFU | alpha CT | alpha calib | K/G scale | RMSE after |
|-----|----------|-------------|-----------|------------|
| 1 | 0.554 | 0.950 | 0.698 | 0.65 km/s |
| 2 | 0.560 | 0.191 | 0.706 | 0.33 km/s |
| 3 | 0.615 | 0.950 | 0.721 | 0.55 km/s |
| 4 | fallback | 0.570 | 1.000 | n/a |

Profile mean Vp after calibration: ~5.2 km/s (vs ~6.3 km/s uncalibrated).

## Notes

- F2911V (lab F2911H) is a strong outlier; use `--robust` to exclude from HFU2 fit.
- HFU4 has no lab plugs; uses average of calibrated HFU2+HFU3 parameters.
- Next: Gassmann (PVT) for saturated rock comparison.
