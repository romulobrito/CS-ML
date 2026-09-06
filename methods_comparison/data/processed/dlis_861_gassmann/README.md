# Well 861 -- DLIS validation (Gassmann profile)

Compares Gassmann-saturated DEM Vp/Vs against DSI sonic log (87/87 matched).

## Regenerate

```bash
.venv/bin/python methods_comparison/scripts/run_861_dem_sc_gassmann.py
```

## Interpretacao (production run, 2026-06-15)

| Metrica | Valor |
|---------|-------|
| MAPE Vp (%) | 15.4 |
| RMSE Vp (km/s) | 0.81 |
| Bias Vp (km/s) | +0.59 |
| Mean Vp sonic (km/s) | 4.58 |
| Mean Vp DEM sat (km/s) | 5.17 |

Dry vs saturated comparison: `metrics_dry_vs_saturated.json`.

## Proximos passos

- Etapa 3: ML on Vp residuals (physical baseline = Gassmann profile).
- Optional: refine Kf with measured PVT if available.
