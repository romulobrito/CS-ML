# Etapa 1 -- Construcao do dataset para Machine Learning (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO, carbonato pre-sal) |
| Etapa | 1 de N (preparacao de dados; sem treino de modelos) |
| Escopo | Integrar perfis de poco, laboratorio e microCT em tabelas prontas para ML |
| Fora de escopo nesta etapa | DEM/SC, Pyrockphys, previsao de Vp/Vs, hyperparameter tuning |

---

## 1. Objetivo da Etapa 1

Produzir um **dataset tabular reproduzivel** que una tres escalas de informacao na mesma referencia de profundidade:

1. **Escala de poco** -- curvas de perfilagem e propriedades derivadas (`861_logs_table.xlsx`, derivado de `data/Auddys_table.xlsx`).
2. **Escala de laboratorio** -- porosidade, permeabilidade, RQI, FZI e HFU medidos em plugs (`861_logs_table.xlsx`, mesma tabela).
3. **Escala micro** -- geometria e ocorrencia de poros por microCT (`General_Output_Results_PoreInfo_samples_w861_MOGNOTomo.xlsx` e abas por amostra).

A Etapa 1 **nao treina** Random Forest, XGBoost ou qualquer outro regressor. Ela apenas:

- padroniza nomes de colunas e unidades;
- associa cada amostra CT (F2829V, F2830H, ...) a uma profundidade no poco 861;
- gera tabelas com flags de qualidade (QC);
- documenta quais colunas podem ser **features (X)** e quais podem ser **targets (y)** em etapas futuras de ML.

---

## 2. Motivacao (por que esta etapa e necessaria)

Os scripts atuais em `methods_comparison/scripts/` leem uma planilha Excel com caminhos Windows fixos e colunas no formato legado. Os dados de microCT estao em arquivos separados, com layout irregular (abas `Plots`, `Plan1`, chave-valor por amostra).

Sem integracao:

- o ML so enxerga perfis + lab (87 linhas), **sem** aspect ratio nem porosidade CT;
- as 10 amostras CT nao se conectam automaticamente a HFU/FZI na mesma profundidade;
- nao ha auditoria de unidades (porosidade em % vs pu vs fracao v/v).

A Etapa 1 fecha essa lacuna e prepara o insumo para:

- ML de **FZI** ou **HFU** (ja iniciado com Random Forest);
- ML futuro de **Vp/Vs** (quando houver alvo lab ou sonic calibrado);
- calibracao posterior por **HFU** em modelos de fisica de rochas (Etapa 2).

---

## 3. Inventario de arquivos de entrada

### 3.1 Tabela principal do poco 861

| Campo | Valor |
|-------|-------|
| Arquivo fonte | `data/Auddys_table.xlsx` |
| Nome logico | **861_logs_table** (renomear no pipeline; nao expor "Auddys" nos produtos finais) |
| Aba | `Logs` |
| Linhas | 87 |
| Profundidade | 5205,91 m a 5233,72 m |
| Passo medio | ~0,32 m (irregular: 0,19 a 1,34 m) |

**Colunas existentes (manter nomes canonicos na exportacao):**

| Coluna canonica | Tipo | Papel no ML |
|-----------------|------|-------------|
| `Depth(m)` | float | Chave de join; eixo de perfil |
| `GR (API)` | float | Feature de perfil |
| `Density (g/cc)` | float | Feature de perfil |
| `Res_Deep` | float | Feature de perfil |
| `Res_Shallow` | float | Feature de perfil (muitos nulls possiveis) |
| `Phi_Neutron (pu)` | float | Feature; proxy NMR/neutrão |
| `Phi_Sonic (pu)` | float | Feature; proxy acustico |
| `Phi_ND (pu)` | float | Feature; porosidade densidade-neutrão |
| `Phi_lab (pu)` | float | Target candidato; validacao CT |
| `k_lab (mD)` | float | Target candidato; validacao Kozeny CT |
| `RQI` | float | Feature derivada; **risco de leakage** se target for FZI |
| `FZI_lab` | float | **Target principal** nos scripts RF atuais |
| `Lithotype` | int | Feature categorica (MOGNO = 1 em parte do intervalo) |
| `HFU` | int | Feature / estratificacao (1=Poor ... 4=Excellent) |

**Aba `Legend`:** mapa HFU -> Poor / Medium / Good / Excellent (incluir em metadados do dataset).

### 3.2 MicroCT MOGNO (pasta `methods_comparison/data/`)

| Arquivo | Papel |
|---------|-------|
| `General_Output_Results_PoreInfo_samples_w861_MOGNOTomo.xlsx` | **Fonte primaria** de integracao |
| `Samples_2.5mm-*/Samples_2.5mm/Output_Results_PoreInfo_*.xlsx` (9 arquivos) | Redundante com abas `F*` do General; usar so se General incompleto |
| `Samples_6mm-*/Samples_6mm/Output_Results_PoreInfo_F2870H.xlsx` | Idem para amostra 6 mm |
| Pastas `Samples_2.5mm-*` duplicadas | **Ignorar uma copia** (mesmo conteudo) |

**Amostras CT (10 plugs derivados de testemunho 1,5"):**

| sample_id | ct_depth_m | orientacao | diametro_mm |
|-----------|------------|------------|-------------|
| F2829V | 5206,00 | V | 2,5 |
| F2830H | 5206,05 | H | 2,5 |
| F2852H | 5211,40 | H | 2,5 |
| F2854H | 5212,05 | H | 2,5 |
| F2859H | 5212,90 | H | 2,5 |
| F2880H | 5218,25 | H | 2,5 |
| F2910H | 5225,70 | H | 2,5 |
| F2911V | 5225,75 | V | 2,5 |
| F2935H | 5232,25 | H | 2,5 |
| F2870H | 5215,55 | H | 6,0 |

Fonte da profundidade e do aspect ratio medio: aba **`Plots`**, colunas 13-15 (`Depth`, `Sample`, `Pore Aspect Ratio`).

### 3.3 Documentacao de referencia

| Arquivo | Uso |
|---------|-----|
| `methods_comparison/docs/README - MOGNO results.docx` | Metodologia CT, limite de 2 voxels, classificacao Anselmetti |
| `methods_comparison/docs/Aplicacao_Python_RockPhys_IA_tests.docx` | Contexto ML + fisica de rochas (Zhang, Li) |

---

## 4. Produtos de saida (nomenclatura Poco 861)

Todos os artefatos devem ficar em:

```
methods_comparison/data/processed/
```

| Arquivo de saida | Linhas | Descricao |
|------------------|--------|-----------|
| `861_logs_table.xlsx` | 87 | Copia normalizada da tabela de perfis+lab (aba `Logs`) |
| `861_ct_samples.parquet` ou `.csv` | 10 | Uma linha por amostra microCT com parametros extraidos |
| `861_integrated_ct_samples.xlsx` | 10 | CT + linha de log mais proxima (join por profundidade) |
| `861_integrated_logs_enriched.xlsx` | 87 | Log completo + colunas CT onde houver amostra (esparsas) |
| `861_integration_qc.json` | -- | Metricas de match, deltas, alertas de unidade |
| `861_INTEGRATION_MANIFEST.txt` | -- | Resumo legivel para reproducao cientifica |
| `861_ML_DATASET_README.md` | -- | Dicionario de dados (features, targets, leakage) |

**Regra de nomenclatura:** prefixo `861_` em todos os produtos; nunca `Auddys_` nos nomes finais.

---

## 5. Pipeline de processamento (detalhado)

### Passo 5.1 -- Ingestao da tabela do poco 861

**Entrada:** `data/Auddys_table.xlsx`, aba `Logs`.

**Acoes:**

1. Validar presenca das 14 colunas canonicas.
2. Ordenar por `Depth(m)` ascendente.
3. Remover duplicatas de profundidade (se existirem).
4. Adicionar colunas derivadas:
   - `well_id` = `"861"` (string constante);
   - `depth_index` = indice 0..N-1;
   - `hfu_label` = mapeamento 1->Poor, 2->Medium, 3->Good, 4->Excellent.
5. Exportar como `861_logs_table.xlsx`.

**Validacoes:**

- `87 <= n_rows <= 87` (registrar se mudar);
- `Phi_lab`, `k_lab`, `FZI_lab`, `HFU` sem null;
- `Depth(m)` estritamente nao decrescente apos sort.

### Passo 5.2 -- Extracao do indice de amostras CT

**Entrada:** `General_Output_Results_PoreInfo_samples_w861_MOGNOTomo.xlsx`, aba `Plots`.

**Acoes:**

1. Ler linhas 3-12 (indices 0-based: 3 a 12 inclusive no parser).
2. Extrair `ct_depth_m`, `sample_id_raw`, `ar_mean_plots`.
3. Normalizar `sample_id`:
   - remover sufixo `(6mm)` -> `F2870H`;
   - extrair `orientation` = ultimo caractere (`V` ou `H`);
   - definir `diameter_mm` = 6,0 se `F2870H`, senao 2,5.
4. Extrair da mesma aba `Plots` (colunas 29-33), por linha de amostra:
   - `phi_ultrapore_ambient_pct` (lab UltraPore, %);
   - `phi_ct_image_pct` (porosidade imagem CT, %);
   - `phi_macro_meso_frac`, `phi_micro_frac` (fracoes Anselmetti).

### Passo 5.3 -- Extracao de resumo por amostra (Plan1 e Mineral)

**Entrada:** mesma planilha General.

**Aba `Plan1` (linha de cabecalho de amostras = linha index 3):**

| Metrica | Linha Plan1 |
|---------|-------------|
| `phi_meso_macropores_vv` | 4 |
| `ar_meso_macropores` | 5 |
| `phi_micropores_vv` | 6 |

Valores sao fracao volumetrica (0-1), nao percentual.

**Aba `Mineral dual thresh`:**

- Bloco **Original** (colunas 3-12): `phi_ct_original_pct`, `solid1_original_pct`, `solid2_original_pct`.
- Bloco **Volume Corrected** (colunas 16-25): `phi_ct_corrected_pct`, `solid1_corrected_pct`, `solid2_corrected_pct`.

Pivotar de formato wide (colunas por amostra) para long (uma linha por `sample_id`).

### Passo 5.4 -- Extracao de parametros detalhados (abas F*)

**Entrada:** abas `F2829V`, `F2830H`, ..., `F2870H` no General.

**Formato:** pares chave-valor (coluna 0 = label, coluna 1 = valor).

**Campos obrigatorios a extrair:**

| Chave no Excel | Coluna de saida | Unidade |
|----------------|-----------------|---------|
| Mean Aspect Ratio | `ct_ar_mean` | adimensional (0-1) |
| Median Aspect Ratio | `ct_ar_median` | adimensional |
| Mean Gamma | `ct_mean_gamma` | adimensional |
| Porosity (%) | `ct_porosity_pct` | % (subamostra CT) |
| Microcroporosity (%) ... | `ct_phi_micro_pct` | % |
| Macro-mesoporosity (%) ... | `ct_phi_macro_meso_pct` | % |
| Permeability (Kozeny) (mD) | `ct_k_kozeny_md` | mD |
| Permeability (Kozeny-Carman) (mD) | `ct_k_kozeny_carman_md` | mD |
| Mean pore diameter (microns) | `ct_mean_pore_diameter_um` | um |
| Specific pore surface (mean) | `ct_specific_pore_surface` | um^-1 |
| Pixel resolution (microns) | `ct_pixel_resolution_um` | um |

**Campos opcionais (fase 1b):** histograma de aspect ratio (bins 0,1-1,0), `Global-DomSize`, ocorrencia Anselmetti por faixa de area.

**Convencao de prefixo:** todas as colunas vindas de microCT recebem prefixo `ct_` para evitar colisao com perfis.

### Passo 5.5 -- Montagem da tabela `861_ct_samples`

Unir por `sample_id`:

- indice Plots (passo 5.2);
- Plan1 + Mineral (passo 5.3);
- parametros F* (passo 5.4).

Resultado: **10 linhas**, chave primaria `sample_id`.

### Passo 5.6 -- Join CT <-> logs do poco 861

**Metodo:** `merge_asof` com vizinho mais proximo em profundidade.

```text
left:  861_ct_samples.ct_depth_m
right: 861_logs_table.Depth(m)
direction: nearest
tolerance: 0.5 m
```

**Colunas adicionadas no join:**

- `log_depth_m` -- profundidade da linha de log associada;
- `depth_delta_m` = abs(ct_depth_m - log_depth_m);
- todas as colunas de perfil/lab da linha correspondente.

**Qualidade esperada do match (referencia):**

| sample_id | depth_delta_m | Nota |
|-----------|---------------|------|
| F2880H | ~0,04 m | Excelente |
| F2935H | ~0,05 m | Excelente |
| F2829V | ~0,09 m | Bom |
| F2854H | ~0,43 m | Limite; flag `match_quality=warn` |

**Saida:** `861_integrated_ct_samples.xlsx`.

### Passo 5.7 -- Enriquecimento da tabela completa de logs

**Entrada:** `861_logs_table` (87 linhas) + `861_ct_samples` (10 linhas).

**Metodo:**

1. Para cada linha de log, verificar se existe amostra CT com `depth_delta_m <= 0.5` m.
2. Se sim, preencher colunas `ct_*` e `sample_id`; senao, null e `has_ct_sample=False`.
3. Coluna booleana `has_ct_sample` em todas as 87 linhas.

**Saida:** `861_integrated_logs_enriched.xlsx`.

**Nota para ML:** nas 87 linhas, apenas ~10 terao features `ct_*` preenchidas. Para treino que exija CT em todas as amostras, usar somente `861_integrated_ct_samples.xlsx`. Para perfil ao longo do poco, usar enriched com CT esparsa ou imputar por HFU (Etapa 2).

### Passo 5.8 -- Controle de qualidade e manifest

**Arquivo `861_integration_qc.json` deve conter:**

```json
{
  "well_id": "861",
  "n_log_rows": 87,
  "n_ct_samples": 10,
  "depth_tolerance_m": 0.5,
  "per_sample": [
    {
      "sample_id": "F2829V",
      "ct_depth_m": 5206.0,
      "log_depth_m": 5205.91,
      "depth_delta_m": 0.09,
      "phi_lab_pu": 0.11,
      "phi_ultrapore_pct": 6.7234,
      "phi_ct_image_pct": 0.01,
      "k_lab_md": 11.1,
      "k_kozeny_md": 1.11,
      "match_quality": "ok"
    }
  ],
  "unit_warnings": [
    "phi_ct_image_pct is local 2.5mm subsample; not comparable directly to phi_lab_pu"
  ]
}
```

**Regras de alerta:**

| Condicao | Severidade |
|----------|------------|
| `depth_delta_m > 0.5` | error (excluir do join ou revisar manualmente) |
| `depth_delta_m > 0.25` e `<= 0.5` | warn |
| `phi_ct_image_pct << phi_lab_pu * 100` | info (esperado: subamostra CT) |
| `abs(k_lab - k_kozeny) / k_lab > 10` | warn (Kozeny e indicativo, nao medida) |

---

## 6. Convencoes de unidades

| Sufixo | Significado | Exemplo |
|--------|-------------|---------|
| `_pu` | porosidade volume/volume (0-1) | `Phi_lab (pu)` = 0,11 |
| `_pct` | percentual (0-100) | `phi_ultrapore_ambient_pct` = 6,72 |
| `_vv` | fracao volumetrica (0-1) | `phi_meso_macropores_vv` = 0,355 |
| `_md` | permeabilidade em miliDarcy | `k_lab (mD)` |
| `_um` | micrometros | `ct_mean_pore_diameter_um` |

**Nunca** somar ou comparar colunas sem converter para a mesma base. O script da Etapa 1 deve preservar valores originais e adicionar colunas convertidas apenas quando explicitamente documentado (ex.: `phi_ultrapore_pu = phi_ultrapore_ambient_pct / 100`).

---

## 7. Dicionario ML -- features e targets (para etapas futuras)

### 7.1 Dataset A -- perfil completo (87 linhas)

**Arquivo:** `861_integrated_logs_enriched.xlsx`

| Papel | Colunas sugeridas |
|-------|-------------------|
| ID | `well_id`, `Depth(m)`, `depth_index` |
| Features (X) perfil | `GR`, `Density`, `Res_Deep`, `Res_Shallow`, `Phi_Neutron`, `Phi_Sonic`, `Phi_ND`, `Lithotype` |
| Features (X) CT esparsas | `ct_ar_mean`, `ct_phi_macro_meso_pct`, ... (null em 77 linhas) |
| Features (X) estratificacao | `HFU` (usar com cuidado em split) |
| Targets (y) candidatos | `FZI_lab`, `Phi_lab`, `k_lab`, `HFU` |
| Evitar como X se y=FZI | `RQI`, `FZI_lab`, `Phi_lab`, `k_lab` (leakage) |

**Uso:** mesmo protocolo dos scripts `FZI_RandonForest_w861.py` e `Testing_regressor.py`, com caminhos atualizados para `861_*`.

### 7.2 Dataset B -- amostras com microCT (10 linhas)

**Arquivo:** `861_integrated_ct_samples.xlsx`

| Papel | Colunas sugeridas |
|-------|-------------------|
| Features (X) | perfis na profundidade do plug + todas `ct_*` |
| Targets (y) | `FZI_lab`, `Phi_lab`, `k_lab`; futuro: `Vp_lab`, `Vs_lab` |
| Chave para fisica de rochas | `ct_ar_mean`, `phi_meso_macropores_vv`, `solid1_corrected_pct` |

**Uso:** validacao multiescala, calibracao por HFU, prototipo de modelo elastico (Etapa 2).

### 7.3 Regras de split para ML (documentar; aplicar na Etapa 3)

- **Nao** usar split aleatorio simples em dados de poco vertical: vizinhos de profundidade sao correlacionados.
- Preferir:
  - **Leave-one-plug-out** (10 folds) no Dataset B;
  - split por **blocos de profundidade** ou por **HFU** no Dataset A;
  - `random_state=42` apenas dentro de cada bloco.
- Reportar metricas por HFU e por profundidade.

---

## 8. Script a implementar (Etapa 1b)

| Item | Valor |
|------|-------|
| Caminho proposto | `methods_comparison/scripts/integrate_861_mogno_ct.py` |
| CLI | `--logs-path`, `--general-xlsx`, `--out-dir` |
| Defaults | `data/Auddys_table.xlsx`, `methods_comparison/data/General_...xlsx`, `methods_comparison/data/processed/` |
| Dependencias | pandas, openpyxl |

**Ordem de execucao:**

```bash
python methods_comparison/scripts/integrate_861_mogno_ct.py
```

**Criterio de aceite:**

- [ ] 10 linhas em `861_integrated_ct_samples.xlsx`, 10 `sample_id` unicos
- [ ] 87 linhas em `861_integrated_logs_enriched.xlsx`, 10 com `has_ct_sample=True`
- [ ] Nenhum `depth_delta_m > 0.5` sem flag de erro
- [ ] `861_integration_qc.json` gerado
- [ ] `861_ML_DATASET_README.md` gerado automaticamente ou mantido em sync

---

## 9. Atualizacao dos scripts ML existentes (Etapa 1c, apos dataset pronto)

| Script legado | Mudanca necessaria |
|---------------|-------------------|
| `FZI_RandonForest_w861.py` | `file_path = 'methods_comparison/data/processed/861_integrated_logs_enriched.xlsx'`; remover `os.chdir` Windows |
| `Supporting_HFU_w861.py` | Idem; HFU ja vem na tabela |
| `Testing_regressor.py` | Substituir `make_regression` por leitura de `861_integrated_logs_enriched.xlsx` com `--target FZI_lab` |

---

## 10. Lacunas conhecidas (nao bloqueiam Etapa 1)

| Lacuna | Impacto | Etapa futura |
|--------|---------|--------------|
| Sem `Vp_lab`, `Vs_lab` | ML nao pode treinar alvo elastico ainda | Etapa 2+ (medicao lab) |
| Porosidade CT << porosidade lab | QC documentado; nao e erro de join | Interpretacao multiescala |
| Apenas 10 plugs CT | CT esparsa no Dataset A | Imputacao por HFU ou ML sem CT |
| Pyrockphys / DEM nao instalado | Sem Vp/Vs teoricos ainda | Etapa 2 fisica de rochas |
| Poco 1045 | Dataset separado (`data/1045/processed/`) | Outro planejamento |

---

## 11. Cronograma sugerido

| Ordem | Entrega | Dependencia |
|-------|---------|-------------|
| 1.1 | Este documento (`etapa1_dataset_ml_poco861.md`) | -- |
| 1.2 | Script `integrate_861_mogno_ct.py` | 1.1 |
| 1.3 | Produtos em `data/processed/` + QC | 1.2 |
| 1.4 | Revisao manual das 10 linhas integradas | 1.3 |
| 1.5 | Atualizar scripts RF / Testing_regressor | 1.4 |
| 2.0 | Planejamento Etapa 2 (DEM/SC + Vp/Vs) | 1.5 |

---

## 12. Referencias internas

- Metodologia microCT: `methods_comparison/docs/README - MOGNO results.docx`
- Contexto ML: `methods_comparison/docs/Aplicacao_Python_RockPhys_IA_tests.docx`
- Schema de colunas de perfil (legado): `scripts/dlis_prepare_well.py` (`AUDDYS_COLUMNS`)
- EDA e leakage HFU/FZI: `scripts/auddys_clp_csgm_eda.py`

---

## 13. Resumo executivo

A **Etapa 1** transforma dados dispersos (tabela do **poco 861**, Excel MOGNO microCT) em **dois datasets ML-ready**:

1. **87 linhas** -- perfil + lab + flags CT esparsas (`861_integrated_logs_enriched.xlsx`).
2. **10 linhas** -- plugs CT totalmente caracterizados + perfil/lab na mesma profundidade (`861_integrated_ct_samples.xlsx`).

Isso habilita o ML a trabalhar com **textura (FZI/HFU)** e **geometria de poros (aspect ratio)** no mesmo registro, sem ainda implementar DEM/SC ou previsao de Vp/Vs. A nomenclatura **861** substitui referencias legadas a "Auddys" em todos os produtos finais desta etapa.
