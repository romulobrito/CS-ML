# Etapa 3 -- ML de residuo Vp (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO) |
| Intervalo | 5205,91 -- 5233,72 m (87 linhas) |
| Etapa anterior | 2b concluida (DEM/SC + HFU + Gassmann + DLIS) |
| Esta etapa | ML sobre residuo elastico apos fisica Gassmann |
| Fora de escopo | Substituir DEM/SC por ML direto perfil -> Vp |

**Dependencias:** `profile_87_gassmann/`, `dlis_861_gassmann/`, Etapa 1d (protocolo depth-block CV).

---

## 1. Objetivo

Corrigir o desvio residual entre \Vp{} teorico (Gassmann) e \Vp{} sônico DSI,
sem abandonar a cadeia fisica da Etapa 2.

Modelo hibrido:

```
Vp_hybrid = Vp_gassmann + ML_residual
```

onde `ML_residual` prediz `Vp_sonic - Vp_gassmann` a partir de curvas de perfil
e da propria \Vp{} fisica.

---

## 2. Dataset (87 linhas)

| Coluna | Fonte | Papel |
|--------|-------|-------|
| Depth(m) | Perfil integrado | Chave |
| HFU | Laboratorio | Segmentacao / analise |
| Curvas wireline (8) | Perfil integrado | Features ML |
| vp_gassmann_km_s | profile_87_gassmann | Feature + baseline fisica |
| vp_sonic_km_s | dlis_861_gassmann | Alvo observado (nao entra em X) |
| vp_residual_km_s | derivada | Alvo ML: sonic - gassmann |

Casamento: 87/87 (mesmo merge do pipeline Gassmann).

---

## 3. Protocolo ML

### Features (X)

- Oito curvas wireline (GR, densidade, resistividades, porosidades, litotipo)
- `vp_gassmann_km_s` (baseline fisico; permitido)

### Proibido em X (leakage)

- `vp_sonic_km_s`, `vs_sonic_km_s`, DTCO, DTSM, `vp_residual_km_s`
- Nao usar HFU predito; HFU lab apenas para analise pos-hoc

### Alvo (y)

- `vp_residual_km_s` = `vp_sonic_km_s` - `vp_gassmann_km_s`

### Validacao

- **Primaria:** depth-block CV (3 blocos), igual Etapa 1
- Predicoes OOF por bloco para montar `Vp_hybrid_oof`
- Holdout 80/20 apenas ilustrativo (legado)

### Modelos

1. Random Forest (`n_estimators=200`, `random_state=42`) -- principal
2. Ridge (`alpha=1.0`) -- alternativa linear

### Metricas de sucesso

Comparar **OOF** contra DSI:

| Metrica | Baseline (Gassmann) | Meta hibrido |
|---------|---------------------|--------------|
| MAPE Vp (%) | ~15,4 | reduzir |
| Bias Vp (km/s) | +0,59 | aproximar 0 |
| RMSE Vp (km/s) | ~0,81 | reduzir |

Criterio de aceite parcial: bias |< 0,35| km/s ou MAPE < 12% global;
melhoria clara em HFU1 (bias alto).

---

## 4. Artefatos

```
methods_comparison/data/processed/ml_runs/residual_vp/
  tables/residual_dataset.csv
  tables/oof_predictions.csv
  tables/comparison_physics_vs_hybrid.csv
  tables/summary_by_hfu.csv
  figures/residual_oof_scatter.png
  figures/vp_physics_vs_hybrid_vs_sonic_depth.png
  metrics.json
  MANIFEST.txt
```

Script: `methods_comparison/scripts/run_861_ml_residual.py`

---

## 5. Criterios de aceite

- [x] 87/87 linhas no dataset residual
- [x] Depth-block CV com OOF completo
- [x] Tabela comparando Gassmann vs hibrido (MAPE, RMSE, bias)
- [x] README + metrics.json alinhados
- [x] LaTeX `poco861_etapa3_residual.tex` compilado
- [x] Etapa 3b CLP-CSGM residual Vp (`etapa3b_clp_csgm_vp_residual_poco861.md`)

### Resultados producao (2026-06-15)

| Modelo | MAPE Vp (%) | Bias (km/s) |
|--------|-------------|-------------|
| Gassmann | 15,4 | +0,59 |
| Hibrido RF OOF | 8,1 | +0,03 |

---

## 6. Ordem de execucao

```
1. Este plano
2. run_861_ml_residual.py
3. Revisar metrics.json (physics vs hybrid)
4. make etapa3 (LaTeX)
5. (Futuro) integrar melhor modelo em produto de perfil
```

---

## 7. Honestidade metodologica

- ML nao substitui fisica: corrige resíduo local
- 87 amostras limitam complexidade (RF ok; MLP descartado)
- Melhoria em OOF nao garante extrapolacao fora do intervalo MOGNO
- HFU4 (n=6) permanece instavel

**Proximo passo apos Etapa 3:** consolidar pipeline unico ou exportar perfil Vp_hybrid para inversao.
