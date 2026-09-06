# ROCKPHYS lab data -- Well 861

Ingested from `methods_comparison/data/ROCKPHYS_Database_04_12_2024 (7).xlsx`.

Planning: `methods_comparison/planning/etapa2_dem_sc_vpvs_poco861.md`

## Regenerate

```bash
python methods_comparison/scripts/run_861_rockphys_ingest.py
python methods_comparison/scripts/run_861_dem_sc_lab_validation.py
```

## Layout

```
rockphys_861/
  MANIFEST.txt
  metrics.json
  tables/
    861_rockphys_velocity.csv           # 196 rows, 28 samples, 7 pressures
    861_rockphys_velocity_ct_plugs.csv  # 10 CT plugs at 22.1 MPa
    861_rockphys_porosity.csv
    861_rockphys_mineralogy.csv
    861_rockphys_rock_info.csv
```

## Key fields (Velocity)

| Column | Unit | Description |
|--------|------|-------------|
| sample_id | -- | Lab plug id (e.g. F2852H) |
| depth_m | m | Sample depth |
| load_pressure_mpa | MPa | Confining pressure step |
| vp_z_km_s, vs_z_km_s | km/s | Dry transmission, Z-axis |
| vp_x_km_s, vs_x_km_s | km/s | X-axis (when measured) |
| vp_mean_axes_km_s | km/s | Mean of available axes |
| vpvs_z | -- | Vp/Vs on Z-axis |

## CT plug alias

CT integrated table uses **F2911V**; ROCKPHYS lab book lists **F2911H** at the
same depth (5225.75 m). Validation maps V -> H automatically.

## Validation output

Lab vs DEM comparison: `dem_sc_runs/lab_validation/` (see that README).
