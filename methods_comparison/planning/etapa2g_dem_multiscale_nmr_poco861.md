# Etapa 2g -- DEM multiescala com fracoes NMR/CMR (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO, carbonatos) |
| Dependencia | Etapa 2e (lab calibration) + Etapa 2f (multiscale A/B) concluidas |
| Objetivo | Substituir fracoes CT por HFU (frageis, N=2-4) por fracoes NMR ponto a ponto |
| Escopo | Perfil 87 linhas + validacao cruzada 10 plugs CT |
| Codigo | `extract_861_cmr.py`, `run_861_dem_sc_multiscale_nmr.py` |
| Saidas | `dem_sc_runs/multiscale_nmr/` |

**Relacao com Etapa 2f:** a Etapa 2f (multiscale A/B) mostrou que o DEM multiescala
com fracoes por HFU *nao* melhorou sobre o monoescala (delta MAPE in-sample = 0.006
p.p.). A hipotese aqui e que fracoes NMR ponto a ponto resolvem a fragilidade das
medianas CT por HFU.

---

## 1. Motivacao

Na Etapa 2f, o modelo M2b forward usou medianas de `f_meso` e `f_micro` por HFU (2-4
plugs) como proxy para o perfil. Isso produziu MAPE ~= M1 monoescala porque:

1. AR_micro varia muito dentro do mesmo HFU (0.01 a 1.00)
2. Fracoes medianas de 2-4 plugs nao representam a variabilidade local
3. Sem fracoes ponto a ponto, nao ha informacao nova vs monoescala

O poco 861 tem dados **CMR** (NMR Schlumberger) nos DLIS:
- `3-brsa-861-sps_8_cmr.dlis` (58 MB)
- `3-brsa-861-sps_8_cmr_ecs.dlis` (194 MB)

Curvas disponiveis: `CMRP_3MS`, `CMFF`, `BFV`, `T2_DIST`, `TCMR`.

O CMR fornece fracoes de porosidade macro vs micro **continuamente**, resolvendo o
gargalo principal da Etapa 2f.

---

## 2. Dados de entrada

### 2.1 CMR DLIS

| Arquivo | Logical File | Frame | Curvas-chave |
|---------|-------------|-------|-------------|
| `_8_cmr.dlis` | 0 | 75B (idx 5) | BFV, CMRP_3MS, CMFF, CBP1-CBP8, 82 canais |
| `_8_cmr_ecs.dlis` | 2 | 75B (idx 1) | T2_DIST, TCMR, CMRP_3MS, CMFF |

### 2.2 Perfil existente

- `sonic_log.csv` (190 linhas, 5205.9--5233.7 m)
- `861_integrated_logs_enriched.xlsx` (87 linhas com HFU, Phi_ND, etc.)
- `861_integrated_ct_samples.xlsx` (10 plugs com fracoes CT)
- `hfu_lab_calibrated.csv` (K_scale, G_scale, alpha por HFU)
- `hfu_calibrated_params.csv` (AR_micro oracle medianas por HFU)
- `pvt_861_defaults.json` (Kf=2.2 GPa, rho_f=1.03 g/cc, Sw=1.0)

---

## 3. Parametros herdados (calibrados na Etapa 2e, fixos)

| Parametro | HFU 1 | HFU 2 | HFU 3 | HFU 4 |
|-----------|-------|-------|-------|-------|
| alpha | 0.950 | 0.191 | 0.950 | 0.570 |
| matrix_k_gpa | 58.63 | 58.67 | 58.98 | 82.46 |
| matrix_g_gpa | 25.78 | 25.63 | 25.48 | 35.83 |
| matrix_rho_gcc | 2.774 | 2.766 | 2.754 | 2.760 |
| matrix_k_scale | 0.698 | 0.706 | 0.721 | 1.000 |
| matrix_g_scale | 0.698 | 0.706 | 0.721 | 1.000 |

AR_micro e AR_meso por HFU: das medianas oracle da Etapa 2f
(`multiscale_ab/tables/plug_comparison.csv`).

PVT: Kf=2.2 GPa, rho_f=1.03 g/cc, Sw=1.0

**Nenhum parametro e re-ajustado ao sonico neste experimento.**

---

## 4. Fases de execucao

### Fase A -- Extracao e QC do CMR (ETL)

**Script:** `extract_861_cmr.py`

1. Abrir `3-brsa-861-sps_8_cmr.dlis` com `dlisio`
2. Extrair de logical_file[0], frame 75B (idx 5):
   - TDEP, CMRP_3MS, CMFF, BFV
3. Calibrar profundidade: mesma corrida 8, testar `depth_m = TDEP / tdep_scale`
4. Validar: cross-check GR do CMR frame 60B vs GR do sonico
5. Salvar `cmr_log_861.csv`
6. QC: cobertura no intervalo 5205.9--5233.7 m, NaNs, soma CMFF+BFV vs CMRP_3MS

**Saida:** `dlis_861/tables/cmr_log_861.csv`

### Fase B -- Merge CMR + perfil e fracoes NMR

Integrado no script principal (Fase D).

1. merge_asof `cmr_log_861.csv` com `dem_vs_sonic_validation.csv` (87 linhas)
2. Calcular fracoes:
   - `f_macro_nmr = CMFF / CMRP_3MS`
   - `f_micro_nmr = BFV / CMRP_3MS`
3. QC: `abs(f_macro + f_micro - 1) < tol`

### Fase C -- Validacao cruzada NMR vs CT (10 plugs)

Integrado no script principal.

1. Nos 10 plugs com CT, interpolar fracoes NMR na profundidade do plug
2. Comparar `f_macro_CT` vs `f_macro_NMR` e `f_micro_CT` vs `f_micro_NMR`
3. Gerar tabela + crossplots

### Fase D -- DEM multiescala + Gassmann no perfil (87 linhas)

**Script:** `run_861_dem_sc_multiscale_nmr.py`

Para cada linha com CMR valido:

```
phi_total  = Phi_ND (pu)
f_meso     = f_macro_nmr(z)     <-- do CMR
f_micro    = f_micro_nmr(z)     <-- do CMR
AR_meso    = mediana CT do HFU  <-- fixo (Etapa 2f)
AR_micro   = mediana oracle HFU <-- fixo (Etapa 2f)
K_scale, G_scale = do lab (Etapa 2e, fixos)

Forward:
  Matriz VRH * scale --> DEM_seq(phi_meso, AR_meso; phi_micro, AR_micro)
  --> Gassmann(K_dry, G_dry, Kf, phi) --> Vp saturado
```

### Fase E -- Comparacao M1 vs M3 vs sonico

| Modelo | Descricao |
|--------|-----------|
| M1 | Monoescala Gassmann (baseline perfil) |
| M3 | Multiescala NMR + Gassmann (este experimento) |

Metricas: MAPE, RMSE, Pearson r, bias (vs sonico DSI).

---

## 5. Criterio de decisao (pre-registrado)

```
GAIN_THRESHOLD_MAPE_PP = 2.0

Manter monoescala se delta_MAPE(M3 vs M1) < 2.0 p.p. nas linhas com CMR.
Investigar multiescala NMR se delta >= 2.0 p.p.
```

---

## 6. Artefatos

```
methods_comparison/
  scripts/
    extract_861_cmr.py
    run_861_dem_sc_multiscale_nmr.py
  data/processed/
    dlis_861/tables/
      cmr_log_861.csv
    dem_sc_runs/multiscale_nmr/
      tables/
        cmr_vs_ct_validation.csv
        profile_m1_vs_m3.csv
        summary_metrics.csv
      figures/
        crossplot_vp_m1_vs_m3.png
        profile_vp_depth.png
        cmr_fractions_depth.png
        crossplot_ct_vs_nmr_fractions.png
      metrics.json
      MANIFEST.txt
  planning/
    etapa2g_dem_multiscale_nmr_poco861.md
```

---

## 7. Riscos

1. Cobertura CMR pode nao cobrir 5205--5234 m
2. CMFF + BFV pode nao somar CMRP_3MS (processamento do campo)
3. Calibracao de profundidade CMR vs sonico (mesma corrida 8, mas offset possivel)
4. AR_micro fixo por HFU continua sendo fraqueza (NMR resolve fracoes, nao AR)

---

## 8. Fora de escopo

- Regressao de AR_micro ou K_scale/G_scale
- Correlacao AR_micro vs T2LM
- Extracao de T2_DIST (distribuicao completa)
- Poco 1045
