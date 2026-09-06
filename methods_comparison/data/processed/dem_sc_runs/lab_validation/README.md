# DEM/SC vs ROCKPHYS lab validation -- Well 861

Compares **dry-rock DEM/SC** predictions (Etapa 2a POC) against **laboratory
ultrasonic velocities** from ROCKPHYS_Database at confining pressure.

Planning: `methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md`

## Prerequisites

1. DEM POC: `dem_sc_runs/poc_10plugs/` (run `run_861_dem_sc_poc_plugs.py`)
2. ROCKPHYS ingest: `processed/rockphys_861/` (run `run_861_rockphys_ingest.py`)

## Regenerate

```bash
python methods_comparison/scripts/run_861_dem_sc_lab_validation.py
```

Optional reference pressure (default 22.1 MPa):

```bash
python methods_comparison/scripts/run_861_dem_sc_lab_validation.py --ref-pressure-mpa 22.1
```

## Layout

```
lab_validation/
  MANIFEST.txt
  metrics.json
  tables/
    dem_vs_lab_validation.csv   # per-plug DEM vs lab
    summary_by_hfu.csv
  figures/
    vp_dem_vs_lab_z.png
    vpvs_dem_vs_lab_z.png
    vp_rel_error_by_sample.png
```

## Interpretation (production run)

| Metric | Value | Note |
|--------|-------|------|
| n samples | 10 | All CT plugs matched (F2911V -> F2911H) |
| MAPE Vp | ~26% | DEM systematically higher than lab |
| Bias Vp | ~+1.2 km/s | Overestimation |
| Pearson r (Vp) | ~0.21 | Weak linear fit (F2911H outlier) |
| MAE Vp/Vs diff | ~0.15 | DEM Vp/Vs also high |
| Best plug | F2870H | ~1.5% Vp error |
| Worst plug | F2911V/H | ~56% (lab Vp Z very low: 3.82 km/s) |

Lab measurement: dry transmission, Z-axis, 22.1 MPa confining pressure.
DEM model: dry rock, aspect ratio and matrix moduli from microCT by HFU.

## Next steps

- Inverse calibration of alpha or Km/Gm per HFU to minimize RMSE vs lab.
- Gassmann saturation (PVT) for comparison when saturated lab data available.
- Mineralogy tab from ROCKPHYS to refine VRH matrix fractions.
