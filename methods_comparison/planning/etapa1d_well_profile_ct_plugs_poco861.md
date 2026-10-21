# Etapa 1d -- well_profile + ct_plugs + alternativas (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 apenas (MOGNO) |
| Intervalo | 5205,91 -- 5233,72 m (87 amostras perfil+lab) |
| Plugs CT | 10 amostras (`861_integrated_ct_samples.xlsx`) |
| Etapa anterior | 1c (diagnostico ML; FZI+perfis descartado como linha principal) |
| Esta etapa | Baselines **well_profile** + **ct_plugs** + alternativas Phi/HFU |
| Fora de escopo | Tuning pesado, poco 1045, DEM/SC/Vp/Vs (Etapa 2) |

**Ultima execucao producao:** `run_861_well_profile_ct_plugs.py` e `run_861_well_profile_alternatives.py` (smoke=False).

---

## 1. Nomenclatura das abordagens

| Nome | Descricao | Dados | Validacao |
|------|-----------|-------|-----------|
| **well_profile** | ML ao longo do poco com curvas de perfil | 87 linhas enriched | depth-block CV (3 blocos) |
| **ct_plugs** | ML multiescala em plugs com microCT | 10 linhas CT integradas | leave-one-plug-out (10 folds) |

Feature sets (ct_plugs):

| Codigo interno (`--feature-mode`) | Pasta de saida | Conteudo |
|-----------------------------------|----------------|----------|
| `log_only` | `wireline_only/` | 8 curvas de perfil |
| `log_plus_ct` | `wireline_plus_ct/` | perfis + 7 features `ct_*` |

Targets testados:

| Target | well_profile | ct_plugs | Papel para Etapa 2 (Vp/Vs) |
|--------|--------------|----------|----------------------------|
| Phi_lab (pu) | **Sim (principal)** | Sim | Entrada DEM/SC (porosidade) |
| FZI_lab | Diagnostico apenas | Sim | Textura; nao usar perfil sozinho |
| HFU | Classificacao (1e) | -- | Segmentacao para calibracao por unidade |
| k_lab (mD) | Diagnostico 1c | -- | Descartado (cauda longa) |

---

## 2. Protocolo ML comum

### Features (X) -- wireline

```
GR (API), Density (g/cc), Res_Deep, Res_Shallow,
Phi_Neutron (pu), Phi_Sonic (pu), Phi_ND (pu), Lithotype
```

### Leakage (obrigatorio)

| Target | Excluir do X |
|--------|--------------|
| Phi_lab | Phi_lab, RQI, FZI_lab |
| FZI_lab | FZI_lab, Phi_lab, k_lab, RQI, HFU |
| HFU | HFU, FZI_lab, RQI |

### Validacao

- **well_profile:** 3 blocos contiguos de profundidade (~29 linhas/fold). Nao usar holdout 80/20 como metrica principal.
- **ct_plugs:** leave-one-plug-out; reportar `global_oof_r2` (regressao) ou `global_oof_accuracy` (classificacao).

### Hiperparametros (producao)

- RF regressao: `n_estimators=200`, `random_state=42`
- Sem RandomizedSearchCV nesta etapa

---

## 3. Estrutura de artefatos

```
methods_comparison/data/processed/ml_runs/
  diagnostics_861/              # Etapa 1c: oracle, HFU, correlacoes
  compare_861/                  # Etapa 1c: 3 targets x 5 regressores
  fzi_rf/                       # RF legado FZI (referencia historica)
  well_profile/
    MANIFEST.txt
    phi_lab/
      rf/                       # RF Phi_lab (baseline principal)
      compare/                  # 5 regressores Phi_lab
      alternatives/             # Ridge + GAM vs RF (Etapa 1e)
    hfu/
      classifier/               # RF + Logistic HFU (Etapa 1e)
  ct_plugs/
    MANIFEST.txt
    ct_plugs_scenarios_summary.csv
    ct_plugs_rf_r2_wireline_vs_ct.csv
    by_target/Phi_lab_pu|FZI_lab/wireline_only|wireline_plus_ct/
```

---

## 4. Resultados -- well_profile Phi_lab (producao)

**RF depth-block CV** (`well_profile/phi_lab/rf/metrics.json`):

| Metrica | Valor |
|---------|-------|
| mean RMSE | 0,033 pu |
| mean R2 | **+0,146** |
| Fold 0 R2 (5206--5214 m) | +0,367 |
| Fold 1 R2 (5215--5223 m) | +0,227 |
| Fold 2 R2 (5225--5234 m) | -0,157 |

**Cinco regressores** (`well_profile/phi_lab/compare/`):

| Regressor | mean R2 |
|-----------|---------|
| **RF** | **+0,146** |
| XGB | -0,099 |
| LR | -0,261 |
| GB | -0,313 |
| MLP | -6,871 |

**Alternativas Phi (Etapa 1e)** (`phi_lab/alternatives/861_phi_model_comparison.csv`):

| Modelo | mean R2 | Nota |
|--------|---------|------|
| **RF baseline** | **+0,146** | Manter para Etapa 2 |
| GAM (`spline_ridge_sklearn`) | -0,052 | pygam nao instalado; fallback sklearn |
| Ridge (alpha=1) | -0,261 | Pior que RF |

**Decisao Phi:** manter **RF** (ou Phi_ND do perfil como proxy direto no DEM/SC).

---

## 5. Resultados -- HFU classificacao (producao)

Distribuicao: HFU1=42, HFU2=29, HFU3=10, HFU4=6.

**OOF global depth-block CV** (`well_profile/hfu/classifier/861_hfu_model_comparison.csv`):

| Modelo | OOF accuracy | OOF balanced acc | OOF F1 macro |
|--------|--------------|------------------|--------------|
| **Logistic (balanced)** | **35,6%** | **22,0%** | **0,217** |
| RF balanced | 29,9% | 17,9% | 0,161 |

Baseline aleatoria (4 classes): ~25% accuracy.

**Decisao HFU:** usar **Logistic** como segmentador preliminar para calibracao DEM/SC por HFU, com **alta incerteza** documentada. Nao substitui litologia/HFU de laboratorio onde disponivel.

---

## 6. Resultados -- ct_plugs (producao)

**RF global OOF R2** (`ct_plugs/ct_plugs_rf_r2_wireline_vs_ct.csv`):

| Target | wireline_only | wireline_plus_ct |
|--------|---------------|----------------|
| **FZI_lab** | **+0,151** | +0,116 |
| Phi_lab (pu) | -0,334 | -0,466 |

**Interpretacao:**

- FZI nos 10 plugs: sinal positivo fraco com perfis; CT **nao** melhorou RF neste run.
- Phi nos plugs: negativo (10 amostras nao representam bem o intervalo de 87 linhas).
- Usar ct_plugs para calibrar **geometria de poros** (aspect ratio, solidos) na Etapa 2, nao como substituto do well_profile.

---

## 7. Resultados -- diagnostico Etapa 1c (referencia)

Arquivo: `diagnostics_861/decision/TARGET_DECISION_SUMMARY.md`

| Target | RF well_profile R2 | Oracle / nota |
|--------|-------------------|---------------|
| Phi_lab | +0,146 | Melhor target perfil |
| FZI_lab | -0,819 | Oracle phi+k: R2 ~+0,69 |
| k_lab | -15,4 | Cauda longa; descartado |

**Conclusao central:** FZI e previsivel com lab (phi+k), nao com perfis sozinhos.

---

## 8. Conexao com Etapa 2 (Vp/Vs)

Cadeia planejada (nao implementada nesta etapa):

```
Perfis (well_profile) --> Phi estimada (RF ou Phi_ND)
                       --> HFU estimada (Logistic, fraca)
CT plugs (ct_plugs)   --> aspect ratio, phi CT, solidos
         + HFU         --> DEM/SC (Pyrockphys) por unidade
                       --> Vp/Vs teoricos
                       --> ML residual (Etapa 3)
```

O que **nao** fazer com base nestes resultados:

- Prever FZI ou Vp/Vs diretamente so com perfis RF.
- Assumir que ct_plugs substitui well_profile para Phi.
- Tuning pesado antes de fixar insumos da fisica de rochas.

O que **levar para Etapa 2:**

| Insumo | Fonte recomendada | Confianca |
|--------|-------------------|-----------|
| Phi | RF well_profile ou Phi_ND | Moderada |
| HFU | Logistic OOF ou HFU lab | Baixa (OOF) / Alta (lab) |
| AR / porosidade CT | `861_integrated_ct_samples.xlsx` | Alta nos 10 plugs |
| Segmentacao DEM/SC | Por HFU (lab ou predito) | -- |

---

## 9. Scripts e comandos

```bash
cd /home/romulo/Documentos/cs-regressor
source .venv/bin/activate

# Etapa 1d completa (Phi RF + compare + ct_plugs)
python methods_comparison/scripts/run_861_well_profile_ct_plugs.py

# Alternativas Phi (Ridge/GAM) + HFU classificadores
python methods_comparison/scripts/run_861_well_profile_alternatives.py

# Diagnostico Etapa 1c (oracle, correlacoes, HFU stratificado)
python methods_comparison/scripts/diagnose_861_ml.py --skip-baselines

# Smoke (rapido)
python methods_comparison/scripts/run_861_well_profile_ct_plugs.py --smoke
python methods_comparison/scripts/run_861_well_profile_alternatives.py --smoke
```

| Script | Funcao |
|--------|--------|
| `run_phi_lab_rf_861.py` | RF Phi_lab well_profile |
| `run_861_ml_baseline.py` | 5 regressores (`--dataset enriched\|ct`) |
| `run_861_ct_plugs_baseline.py` | Grid ct_plugs |
| `run_861_well_profile_ct_plugs.py` | Orquestrador 1d |
| `run_861_phi_alternatives.py` | Ridge + GAM vs RF |
| `run_861_hfu_classifier.py` | RF + Logistic HFU |
| `run_861_well_profile_alternatives.py` | Orquestrador 1e |
| `diagnose_861_ml.py` | Diagnostico decisao target |

---

## 10. Criterios de aceite

- [x] well_profile Phi RF com depth-block CV e artefatos em `ml_runs/well_profile/`
- [x] ct_plugs: 4 cenarios (2 targets x 2 feature modes)
- [x] Diagnostico 1c com oracle FZI e metricas por HFU
- [x] Alternativas Phi (Ridge/GAM) comparadas com RF
- [x] HFU classificacao com confusion matrix OOF
- [x] Documentacao de decisoes para Etapa 2
- [ ] Etapa 2: DEM/SC + Vp/Vs (proximo)

---

## 11. Lacunas e dependencias

| Item | Status | Acao |
|------|--------|------|
| `pygam` | Nao instalado | GAM usa `SplineTransformer+Ridge`; opcional `pip install pygam` |
| `Vp_lab`, `Vs_lab` | Ausentes no dataset | Etapa 2+: sonic lab ou calibracao |
| HFU OOF ~36% | Fraco | Usar HFU lab onde existir; predito so como fallback |
| CT nao melhora RF | Documentado | CT entra na fisica (DEM/SC), nao no ML de perfil |

---

## 12. Referencias

- Integracao dados: `etapa1_dataset_ml_poco861.md`
- Baseline ML 1c: `etapa1c_ml_baseline_poco861.md`
- Log de execucao: `agent_poco861.log` (iteracoes 10--12)
- README colunas: `data/processed/861_ML_DATASET_README.md`

---

## 13. Resumo executivo

A Etapa 1d/1e validou o que e previsivel no poco 861 **antes** da fisica de rochas:

1. **Phi_lab + perfis** e o unico baseline regressivo modestamente positivo (R2 ~0,15; RF melhor que Ridge/GAM).
2. **FZI + perfis** falha no well_profile; nos ct_plugs ha sinal fraco (R2 ~0,15, n=10).
3. **HFU** e classificavel fracamente (~36% OOF); Logistic ligeiramente melhor que RF.
4. **CT features** nao melhoraram RF; valor do CT esta na **geometria de poros** para DEM/SC (Etapa 2).
5. Proximo passo alinhado a **Vp/Vs**: DEM/SC + Pyrockphys calibrado por HFU, usando Phi + aspect ratio CT.
