# Etapa 1c -- Baseline ML no dataset integrado (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 apenas (MOGNO) |
| Etapa anterior | 1 concluida (`integrate_861_mogno_ct.py`, produtos em `data/processed/`) |
| Esta etapa | Conectar e executar ML nos datasets 861 integrados |
| Fora de escopo | Poco 1045, DEM/SC, Pyrockphys, Vp/Vs, novo ETL microCT |

---

## 1. Objetivo imediato

Treinar e comparar modelos de Machine Learning usando os arquivos gerados na Etapa 1, **sem alterar a fisica de rochas**.

Metas concretas:

1. Eliminar dependencia de caminhos Windows e da planilha legada nos scripts de `methods_comparison/scripts/`.
2. Estabelecer um **protocolo unico** de features, targets e validacao para o poco 861.
3. Reproduzir o baseline **Random Forest + SHAP** para `FZI_lab` com o dataset integrado.
4. Adaptar `Testing_regressor.py` para rodar os **5 regressores** em dados reais do 861 (nao sinteticos).
5. Salvar metricas, modelos e figuras em pasta versionada para reproducao.

Esta etapa responde: *"Os regressores alinham-se aos dados de perfil e laboratorio do 861?"* -- ainda **nao** responde sobre Vp/Vs.

---

## 2. Pre-requisitos (ja atendidos)

Executar antes de qualquer script ML desta etapa:

```bash
python methods_comparison/scripts/integrate_861_mogno_ct.py
```

Arquivos necessarios:

| Arquivo | Linhas | Uso nesta etapa |
|---------|--------|-----------------|
| `methods_comparison/data/processed/861_integrated_logs_enriched.xlsx` | 87 | ML ao longo do poco (principal) |
| `methods_comparison/data/processed/861_integrated_ct_samples.xlsx` | 10 | ML multiescala / leave-one-plug-out |
| `methods_comparison/data/processed/861_integration_qc.json` | -- | Auditoria de join |
| `methods_comparison/data/processed/861_ML_DATASET_README.md` | -- | Dicionario de colunas |

---

## 3. Escopo de dados (dois modos)

### well_profile -- Perfil completo (default)

- **Arquivo:** `861_integrated_logs_enriched.xlsx`
- **Linhas:** 87
- **CT:** esparsa (`has_ct_sample=True` em 8 profundidades; ver colisoes no QC)
- **Uso:** prever `FZI_lab`, `Phi_lab`, `k_lab` ou `HFU` a partir de curvas de perfil
- **CV:** blocos de profundidade
- **Scripts:** RF legado, `run_861_ml_baseline.py` (`--dataset enriched`)

### ct_plugs -- Plugs com microCT

- **Arquivo:** `861_integrated_ct_samples.xlsx`
- **Linhas:** 10
- **CT:** completa em todas as linhas
- **Uso:** ML com features `ct_*` + perfis; validacao leave-one-plug-out
- **Scripts:** `run_861_ct_plugs_baseline.py`, `run_861_ml_baseline.py` (`--dataset ct`)

**Recomendacao para o primeiro entregavel:** comecar pelo **well_profile** (87 linhas).

---

## 4. Protocolo ML (Poco 861)

### 4.1 Features (X) -- well_profile

Incluir (colunas de perfil):

```
GR (API), Density (g/cc), Res_Deep, Res_Shallow,
Phi_Neutron (pu), Phi_Sonic (pu), Phi_ND (pu), Lithotype
```

Excluir sempre do X:

```
Depth(m), depth_index, well_id, hfu_label,
FZI_lab, Phi_lab (pu), k_lab (mD), RQI,
has_ct_sample, sample_id, ct_depth_m, depth_delta_m, match_quality
```

Excluir colunas `ct_*` no well_profile inicial (null na maioria das linhas). Ver Etapa 1d para ct_plugs.

### 4.2 Targets (y) -- prioridade

| Prioridade | Target | Justificativa |
|------------|--------|---------------|
| 1 | `FZI_lab` | Alinhado ao RF existente e a textura/HFU |
| 2 | `Phi_lab (pu)` | Validacao petrofisica |
| 3 | `k_lab (mD)` | Validacao permeabilidade |
| 4 | `HFU` | Classificacao ordinal (tratar como regressao ou classificacao separada) |

**Leakage:** nao usar `RQI` nem `FZI_lab` como input quando o target for `FZI_lab`.

### 4.3 Validacao cruzada (obrigatoria)

Nao usar `train_test_split` aleatorio simples como unica metrica (vizinhos em profundidade correlacionam).

**Protocolo minimo (well_profile):**

1. **Split por blocos de profundidade:** 3 blocos contiguos (~29 linhas cada); treino em 2, teste em 1; rotacionar.
2. Reportar RMSE e R2 por bloco e media.
3. Manter `random_state=42` apenas dentro de subsplits ou para RandomizedSearchCV.

**Protocolo complementar (ct_plugs, 10 linhas):**

- Leave-one-plug-out (10 folds): cada fold reserva 1 `sample_id`.

### 4.4 Scalers

| Regressor | Scaler recomendado |
|-----------|-------------------|
| Random Forest | nenhum (baseline) |
| Gradient Boosting | nenhum ou MinMax (testar) |
| XGBoost | nenhum ou MinMax |
| MLP | StandardScaler |
| Linear Regression | StandardScaler |

Controlar via flag existente em `Testing_regressor.py` (`flag_scaler`).

---

## 5. Trabalho a implementar

### 5.1 Modulo compartilhado de dados (novo)

**Arquivo proposto:** `methods_comparison/scripts/ml_861_data.py`

Responsabilidades:

- `load_logs_enriched(path) -> pd.DataFrame`
- `load_ct_samples(path) -> pd.DataFrame`
- `build_xy(df, target, feature_mode="log_only") -> (X, y, feature_names)`
- `depth_block_splits(df, n_blocks=3) -> List[Tuple[ndarray, ndarray]]`
- `leave_one_plug_out_splits(df) -> ...` (ct_plugs)

Constantes: listas de colunas X/y documentadas na Secao 4.

### 5.2 Atualizar `FZI_RandonForest_w861.py`

| Antes | Depois |
|-------|--------|
| `os.chdir('D:\\ROCKPHYS\\...')` | caminhos relativos ao repo via `Path` |
| `Auddys_table.xlsx` | `861_integrated_logs_enriched.xlsx` |
| `train_test_split` 80/20 | manter para comparacao + adicionar blocos de profundidade |
| saida na pasta cwd | `methods_comparison/data/processed/ml_runs/fzi_rf/` |

Manter: SHAP (bar, waterfall, beeswarm), `joblib` do modelo, figura predito vs observado.

### 5.3 Atualizar `Testing_regressor.py`

| Antes | Depois |
|-------|--------|
| `make_regression(...)` | `ml_861_data.load_logs_enriched` + `build_xy` |
| `flag_MLsim` global | CLI `--regressor rf|gb|xgb|mlp|lr|all` |
| sem target configuravel | CLI `--target FZI_lab` |
| sem caminho de dados | CLI `--data-path` default enriched xlsx |
| saida apenas stdout | CSV de metricas em `ml_runs/compare_861/` |

Reutilizar funcoes `train_optimize_model_*` e `evaluate_model` ja existentes.

### 5.4 Script orquestrador (novo, opcional mas recomendado)

**Arquivo proposto:** `methods_comparison/scripts/run_861_ml_baseline.py`

Executa em sequencia:

1. RF FZI (rapido, reproduz legado)
2. Cinco regressores com hyperparameter search (subconjunto de `n_iter` para teste rapido)
3. Gera `861_ml_baseline_summary.csv`

---

## 6. Produtos de saida

```
methods_comparison/data/processed/ml_runs/
  fzi_rf/
    model_fzi_rf_861.joblib
    metrics.json
    fzi_pred_vs_obs.png
    shap_bar.png
  compare_861/
    861_ml_baseline_summary.csv
    per_target/
      FZI_lab_metrics.csv
      Phi_lab_metrics.csv   (opcional)
```

Atualizar `methods_comparison/planning/agent_poco861.log` a cada iteracao de implementacao.

---

## 7. Criterios de aceite

- [ ] Nenhum script referencia `D:\` ou `F:\` ou `Auddys_table` nos caminhos de execucao padrao
- [ ] RF roda de ponta a ponta com `861_integrated_logs_enriched.xlsx`
- [ ] Metricas reportadas com split por blocos de profundidade (nao apenas 80/20 aleatorio)
- [ ] `Testing_regressor.py` consome dados reais do 861 com `--target FZI_lab`
- [ ] Pelo menos um CSV de comparacao dos 5 regressores salvo em `ml_runs/compare_861/`
- [ ] Documentacao de leakage respeitada (sem RQI no X para target FZI)
- [ ] `agent_poco861.log` atualizado com resultados numericos (R2, RMSE)

---

## 8. Ordem de execucao sugerida

| Passo | Acao | Estimativa |
|-------|------|------------|
| 1 | Criar `ml_861_data.py` | 1 sessao |
| 2 | Refatorar `FZI_RandonForest_w861.py` | 1 sessao |
| 3 | Rodar RF + validar metricas vs run legado | 0,5 sessao |
| 4 | Refatorar `Testing_regressor.py` (CLI + dados reais) | 1 sessao |
| 5 | Rodar comparacao 5 regressores (`n_iter` reduzido primeiro) | 1 sessao |
| 6 | Documentar resultados em `agent_poco861.log` + summary CSV | 0,5 sessao |

Comando alvo apos implementacao:

```bash
python methods_comparison/scripts/run_861_ml_baseline.py --target FZI_lab
```

---

## 9. Riscos e mitigacoes

| Risco | Mitigacao |
|-------|-----------|
| Poucas linhas (87) para tuning pesado | Reduzir `n_iter` no primeiro run; priorizar RF e GB |
| Overfitting em MLP | Early stopping; StandardScaler; poucas camadas |
| Colisoes CT na enriched (8 vs 10) | well_profile ignora `ct_*`; ct_plugs usa `861_integrated_ct_samples.xlsx` |
| HFU desbalanceado | Reportar metricas por HFU; nao usar HFU como X se target for HFU |
| SHAP lento | KernelExplainer apenas em amostra (`max_samples=50`) |

---

## 10. O que vem depois (nao e esta etapa)

| Etapa | Conteudo |
|-------|----------|
| 1d | ML ct_plugs (10 amostras, wireline +/- CT, leave-one-plug-out) |
| 1e | Phi Ridge/GAM + HFU classifier (well_profile); ver `etapa1d_well_profile_ct_plugs_poco861.md` |
| 2 | DEM/SC + Pyrockphys + Vp/Vs calibrado por HFU |
| 3 | ML hibrido (features = perfis + Vp/Vs teoricos + CT) |

Nenhuma atividade do poco **1045** entra neste planejamento.

---

## 11. Referencias internas

- Etapa 1 concluida: `etapa1_dataset_ml_poco861.md`
- Log de agente: `agent_poco861.log`
- Integracao: `methods_comparison/scripts/integrate_861_mogno_ct.py`
- Scripts legados: `FZI_RandonForest_w861.py`, `Testing_regressor.py`
- Contexto teorico: `methods_comparison/docs/Aplicacao_Python_RockPhys_IA_tests.docx`

---

## 12. Resumo executivo

O **proximo passo imediato** para o poco 861 e a **Etapa 1c**: criar `ml_861_data.py`, refatorar RF e `Testing_regressor.py` para ler `861_integrated_logs_enriched.xlsx`, aplicar validacao por blocos de profundidade e gerar o primeiro **baseline comparativo dos 5 regressores** contra `FZI_lab` (e opcionalmente `Phi_lab`, `k_lab`), sem DEM/SC e sem envolver o poco 1045.
