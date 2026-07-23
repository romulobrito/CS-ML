# Etapa 3b -- CLP-CSGM residual Vp (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO) |
| Intervalo | 5205,91 -- 5233,72 m (87 linhas) |
| Etapa anterior | 3 (RF residual Vp OOF MAPE 8,1%) |
| Esta etapa | CLP-CSGM sobre resíduo elastico apos Gassmann |
| Fora de escopo | Substituir DEM/SC; cross-well |

**Dependencias:** `residual_vp/` (Etapa 3 RF), `csgm_m2_module.py`, protocolo depth-block CV.

---

## 1. Objetivo

Comparar o hibrido fisica + ML da Etapa 3 com uma variante **CLP-CSGM** que
modela o resíduo em janelas deslizantes:

```
Vp_hybrid = Vp_gassmann + G(z_hat)
```

onde `G` e o decoder do AE treinado em janelas de
`delta_vp = vp_sonic - vp_gassmann`, e `z_hat` vem do prior condicional
`h(u)` (Ridge sobre logs + Vp Gassmann na janela) ou de refinamento com
medidas esparsas nos plugs.

Analogia com Etapa 1f (variante B residual para phi_lab), mas a baseline
fisica e **Gassmann**, nao RF.

---

## 2. Dataset e features u

Mesmo merge 87/87 da Etapa 3 (`residual_dataset.csv`).

| Simbolo | Papel | Colunas |
|---------|-------|---------|
| **y** | Residuo na janela | `vp_residual_km_s` |
| **u** | Logs densos + Vp Gassmann | 8 wireline + `vp_gassmann_km_s` |
| **b** | (opcional) delta nos plugs | indices dos 10 plugs CT |

Proibido em u: `vp_sonic`, `vp_residual`, DTCO/DTSM.

---

## 3. Metodos

| ID | Inferencia teste | Comparavel a RF OOF? |
|----|------------------|----------------------|
| `clp_ridge_prior_m0` | z0 = h(u), m=0 | Sim (principal) |
| `clp_zero_residual_m0` | z0 = encode(0), m=0 | Baseline CLP (confia em Gassmann) |
| `clp_plug_sparse_b` | z0 = encode(0), b nos plugs | Complementar (estilo Etapa 1f) |

Protocolo: depth-block CV (3 folds), janela L=16, passo 1, stitch uniform_mean.

---

## 4. Artefatos

```
methods_comparison/data/processed/ml_runs/residual_vp/clp_csgm/
  tables/comparison_gassmann_rf_clp.csv
  tables/oof_predictions_clp.csv
  tables/summary_by_hfu_clp.csv
  figures/comparison_mape_bar.png
  figures/vp_depth_rf_vs_clp.png
  metrics.json
  MANIFEST.txt
```

Script: `methods_comparison/scripts/run_861_clp_vp_residual.py`
Core: `methods_comparison/scripts/clp_861_vp_residual.py`

LaTeX: secao em `poco861_etapa3_residual.tex`, figuras `fig3_clp_*`.

---

## 5. Criterios de aceite

- [x] 87/87 linhas, depth-block OOF para os tres metodos
- [x] Tabela Gassmann vs RF vs CLP (MAPE, RMSE, bias)
- [x] Figuras copiadas para `latex/figures/fig3_clp_*`
- [x] LaTeX atualizado (`poco861_etapa3_residual.tex`, secao 3b)

### Resultados producao (2026-06-24)

| Modelo | MAPE Vp (%) | Bias (km/s) | RMSE (km/s) |
|--------|-------------|-------------|-------------|
| Gassmann | 15,4 | +0,59 | 0,81 |
| RF hibrido OOF | 8,1 | +0,03 | 0,46 |
| CLP Ridge prior m=0 | 14,0 | +0,25 | 0,75 |
| CLP zero residual m=0 | 12,4 | +0,34 | 0,66 |
| CLP plug sparse b | 11,4 | +0,31 | 0,63 |

**Recomendacao:** manter RF (ou Ridge 7,7% MAPE) como corretor principal;
CLP nao supera RF no perfil completo OOF neste poco.

---

## 6. Execucao

```bash
cd methods_comparison/scripts
python run_861_clp_vp_residual.py --smoke   # rapido
python run_861_clp_vp_residual.py           # producao
```

Requer Etapa 3 RF pre-calculada para comparacao lado a lado
(`residual_vp/tables/oof_predictions.csv`).
