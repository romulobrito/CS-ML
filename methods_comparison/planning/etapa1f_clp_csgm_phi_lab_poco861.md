# Etapa 1f -- CLP-CSGM para Phi_lab no poco 861 (MOGNO)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO, carbonatos) |
| Intervalo | 5205,91 -- 5233,72 m (87 amostras) |
| Plugs CT | 10 amostras; 8 profundidades com CT na tabela enriched |
| Etapa anterior | 1d (RF well_profile; ct_plugs; Phi_lab R2 ~ +0,15 depth-block) |
| Esta etapa | CLP-CSGM: reconstruir curva Phi_lab com u denso + b esparso nos plugs |
| Fora de escopo inicial | FZI/k como y; CT em b; cross-well; tuning pesado |

**Planejamento:** este documento. **Implementacao:** ver Secao 8.

---

## 1. Pergunta cientifica

No RF da Etapa 1d, cada profundidade e prevista **sem** usar lab em X:

> Wireline sozinho preve Phi_lab ponto a ponto?

No CLP-CSGM (Etapa 1f):

> Com wireline denso (u) **e** porosidade de lab **apenas nos 10 plugs** (b),
> reconstruimos a **curva** Phi_lab(z) ao longo do intervalo?

Isso e complementar ao RF: o CLP pode usar informacao de calha esparsa de forma
estruturalmente correta (termo de consistencia ||M G(z) - b||), nao como feature
com vazamento.

---

## 2. Notacao CLP (alinhada ao repositorio)

| Simbolo | Papel no 861 | Colunas / origem |
|---------|--------------|------------------|
| **y** | Curva-alvo a reconstruir | `Phi_lab (pu)` em janelas deslizantes |
| **u** | Logs densos na janela | 8 curvas wireline (Secao 3) |
| **b** | Observacoes esparsas de lab | `Phi_lab (pu)` **somente** nos indices dos 10 plugs |
| **M** | Subamostragem por coordenada | Mascara fixa nos indices-plug dentro da janela |
| **G, h** | AE + prior condicional | Treino offline em janelas dos blocos de treino |

**Regra:** Phi_lab, RQI, FZI_lab **nunca** entram em u (mesma regra anti-vazamento
da Etapa 1d). CT **nao** entra em b; fase opcional 1f-b enriquece u com ct_* onde
`has_ct_sample=True`.

---

## 3. Features u (8 curvas wireline)

Identicas ao well_profile (Etapa 1d):

```
GR (API), Density (g/cc), Res_Deep, Res_Shallow,
Phi_Neutron (pu), Phi_Sonic (pu), Phi_ND (pu), Lithotype
```

Mapeamento para runner legado Auddys (smoke):

```
density, gr, res_deep, res_shallow, phi_neutron, phi_sonic, phi_nd, lithotype
```

HFU de lab **omitida** de u na fase principal (correlaciona com alvo).

---

## 4. Cenarios experimentais

| ID | Pasta de saida | b (medidas esparsas) | u | Validacao |
|----|----------------|----------------------|---|-----------|
| **plug_sparse_b** | `clp_861/phi_lab/plug_sparse_b/` | Fixo: 10 profundidades plug | wireline 8 | depth-block 3 folds |
| **rho_subsample** | `clp_861/phi_lab/rho_subsample/` | Subsample aleatorio rho in-window | wireline 8 | depth-block + rho grid |
| **wireline_plus_ct_u** | `clp_861/phi_lab/wireline_plus_ct_u/` | Igual plug_sparse_b | wireline + ct_* (8 linhas) | depth-block 3 folds |

Prioridade: **plug_sparse_b** (pergunta principal). rho_subsample e comparavel ao
protocolo 1045. wireline_plus_ct_u e fase 1f-b.

---

## 5. Validacao por blocos de profundidade

Reutilizar `depth_block_splits()` de `ml_861_data.py` (3 blocos ~29 linhas):

| Bloco | Profundidade (m) | Papel rotativo |
|-------|------------------|----------------|
| 0 | 5206 -- 5214 | treino / val / teste |
| 1 | 5215 -- 5223 | treino / val / teste |
| 2 | 5225 -- 5234 | treino / val / teste |

Por fold:

1. **Treino:** janelas dos 2 blocos restantes; AE em y; prior h(u) em (u, E(y)).
2. **Validacao:** escolha de lambda e hiperparametros CLP (dentro do treino ou
   bloco reservado internamente).
3. **Teste:** bloco held-out; em cada janela, b so nos indices-plug que caem
   na janela; metricas vs Phi_lab verdadeiro em **todas** as profundidades do bloco.

Reportar:

- RMSE / R2 global OOF no perfil reconstruido (stitching de janelas)
- RMSE por bloco
- Fracao de janelas de teste com >= 1 plug em b

---

## 6. Indices dos 10 plugs (b fixo)

Fonte: `861_integrated_ct_samples.xlsx` (colunas `sample_id`, `log_depth_m`) ou
`861_integration_qc.json` (`per_sample[].log_depth_m`).

Casamento com enriched 87 linhas: indice de linha onde `Depth(m)` e o mais proximo
(tolerancia 0,5 m), mesma logica de `integrate_861_mogno_ct.py`.

Helper: `methods_comparison/scripts/clp_861_protocol.py` -> `load_plug_row_indices()`.

Colisoes conhecidas (2 plugs -> mesma linha enriched): F2830H perde para F2829V;
F2911V perde para F2910H. **8 linhas** com plug unico; protocolo b usa **10**
valores de Phi_lab nos plugs (LOPO nao e o foco aqui; b pode incluir ambos plugs
na mesma profundidade apenas se protocolo rho alternativo).

---

## 7. Comparacao com RF (Etapa 1d)

| Metodo | Informacao na inferencia | Saida | Referencia |
|--------|--------------------------|-------|------------|
| RF wireline | 8 curvas na profundidade | 1 valor / linha | `well_profile/phi_lab/rf/` |
| CLP plug_sparse_b | janela u + Phi_lab nos plugs em b | curva / janela | `clp_861/phi_lab/plug_sparse_b/` |

Comparacao justa:

- RF: sem lab em X (baseline existente).
- CLP: **mais** informacao (10 amarras); titulo do experimento deve deixar isso
  explicito.

Tabela alvo: `clp_861/compare_rf_baseline/clp_vs_rf_phi_lab_depth_block.csv`

Colunas sugeridas: `fold_id`, `depth_min_m`, `depth_max_m`, `rmse_rf`, `r2_rf`,
`rmse_clp`, `r2_clp`, `n_plugs_in_test_block`.

---

## 8. Scripts e dependencias de codigo

### 8.1 Arquivos desta etapa

| Caminho | Funcao |
|---------|--------|
| `methods_comparison/planning/etapa1f_clp_csgm_phi_lab_poco861.md` | Este plano |
| `methods_comparison/scripts/clp_861_protocol.py` | Indices plug, canais u, paths |
| `methods_comparison/scripts/run_861_clp_csgm_phi_lab.py` | Orquestrador CLI |
| `methods_comparison/scripts/ml_861_data.py` | Constantes `CLP_861_*`, `clp_861_scenario_dir()` |

### 8.2 Codigo CLP legado (raiz do repo)

| Caminho | Funcao |
|---------|--------|
| `csgm_m2_module.py` | Nucleo CLP-CSGM |
| `sir_cs_benchmark_direct_ub.py` | Benchmark direct-UB + CSGM |
| `direct_ub_baselines.py` | Baselines AE / ML |
| `scripts/auddys_smoke_direct_ub.py` | Smoke Auddys/861 (split contiguo, rho) |
| `scripts/auddys_clp_csgm_eda.py` | EDA pre-CLP |

### 8.3 Fases de implementacao

| Fase | Entrega | Script |
|------|---------|--------|
| **0 -- EDA** | Recomendacoes window_len, rhos | `auddys_clp_csgm_eda.py` |
| **1 -- Smoke** | CLP Ridge/MLP no intervalo MOGNO, rho grid | `run_861_clp_csgm_phi_lab.py --mode smoke` |
| **2 -- plug_sparse_b** | b fixo nos plugs + depth-block CV | `run_861_clp_csgm_phi_lab.py --mode prod` |
| **3 -- compare RF** | Tabela CLP vs RF por bloco | mesmo orquestrador + `compare_rf_baseline/` |
| **4 -- wireline_plus_ct_u** | u enriquecido nas 8 linhas CT | cenario opcional |

Fase 2 exige estender o pipeline direct-UB para:

- split por `DepthBlockFold` (nao apenas `contiguous_split` por indice de janela);
- mascara `b` fixa nos indices-plug globais mapeados para cada janela.

Ate Fase 2 estar pronta, `--mode smoke` delega para `auddys_smoke_direct_ub.py`
com filtros de profundidade MOGNO.

---

## 9. Estrutura de artefatos

```
methods_comparison/data/processed/ml_runs/clp_861/
  README.md
  MANIFEST.txt
  phi_lab/
    plug_sparse_b/
      runs/<run_id>/
        tables/
          summary_by_fold.csv
          summary_clp_vs_baselines.csv
          plug_measurement_indices.csv
          oof_profile_predictions.csv
          split_depth_block_summary.csv
        figures/
          depth_profile_phi_lab_clp.png
          depth_profile_phi_lab_rf_overlay.png
          parity_clp_oof.png
        logs/
          run_console.log
        PROTOCOL.txt
        metrics.json
    rho_subsample/
      runs/<run_id>/ ...
    wireline_plus_ct_u/
      runs/<run_id>/ ...
  compare_rf_baseline/
    clp_vs_rf_phi_lab_depth_block.csv
    notes.md
  eda/
    -> symlink ou copia de outputs/auddys_clp_csgm_eda/runs/<run_id>/
```

Entrada de dados padrao:

- Enriched: `methods_comparison/data/processed/861_integrated_logs_enriched.xlsx`
- Plugs: `methods_comparison/data/processed/861_integrated_ct_samples.xlsx`
- Excel legado (smoke Auddys): `data/Auddys_table.xlsx` (aba Logs)

---

## 10. Hiperparametros iniciais (smoke / prod)

| Parametro | Smoke | Producao (proposta) |
|-----------|-------|---------------------|
| window_len | 16 | 16 ou 24 (EDA) |
| step | 1 | 1 |
| seeds | 7 | 7, 23, 41 |
| rhos (rho_subsample) | 0.3, 0.5 | 0.2 -- 0.6 |
| csgm_prior_type | ridge | ridge, mlp |
| csgm_latent_dim | 16 | 16 |
| csgm_lambda_grid | ver EDA json | 0.0001 -- 0.1 |
| measurement_kind | subsample | subsample (plug_sparse: fixo) |

---

## 11. Comandos

### EDA (opcional, uma vez)

```bash
source .venv/bin/activate
python scripts/auddys_clp_csgm_eda.py \
  --excel-path data/Auddys_table.xlsx \
  --target phi_lab \
  --base-dir methods_comparison/data/processed/ml_runs/clp_861/eda
```

### Smoke CLP no intervalo MOGNO

```bash
python methods_comparison/scripts/run_861_clp_csgm_phi_lab.py --mode smoke
```

### Producao plug_sparse_b + comparacao RF (quando Fase 2 pronta)

```bash
python methods_comparison/scripts/run_861_clp_csgm_phi_lab.py \
  --mode prod \
  --scenario plug_sparse_b \
  --compare-rf
```

---

## 12. Criterios de aceite

- [ ] `clp_861_protocol.py` retorna 10 indices-plug consistentes com QC
- [ ] Smoke CLP compila e grava em `clp_861/phi_lab/rho_subsample/runs/`
- [ ] Prod plug_sparse_b: 3 folds depth-block, sem Phi_lab em u
- [ ] `oof_profile_predictions.csv` cobre 87 profundidades (stitch OOF)
- [ ] `clp_vs_rf_phi_lab_depth_block.csv` gerado
- [ ] PROTOCOL.txt documenta mascara b e blocos
- [ ] LaTeX / Beamer: slide opcional "CLP vs RF" (futuro)

---

## 13. Riscos e mitigacao

| Risco | Mitigacao |
|-------|-----------|
| Poucas janelas (n=87) | window_len 16; reportar n_windows; smoke primeiro |
| Janelas sem plug em b | Reportar cobertura; prior h(u) only nessas janelas |
| AE instavel | Prior Ridge primeiro; seeds multiplos |
| Split contiguo vs bloco | Smoke != prod; prod usa depth_block_splits |
| Lab denso "escondido" | Phi_lab fora de b so entra em metrica, nunca em u/treino do bloco teste |

---

## 14. Relacao com outras etapas

| Etapa | Relacao |
|-------|---------|
| 1d RF | Baseline comparativo (wireline only) |
| 1f CLP | Reconstrucao perfil + calha esparsa |
| 2 DEM/SC | CT/lab calibram HFU; nao substitui CLP |
| 3 residual Vp | Outro alvo (Vp); pipeline depth-block separado |

---

## 15. Referencias internas

- Etapa 1d: `methods_comparison/planning/etapa1d_well_profile_ct_plugs_poco861.md`
- Paper brief CLP: `paper_clp_csgm/PAPER_BRIEF.md`
- Beamer 1045: `reports/1045_uenf_clp_csgm/beamer.tex`
- Integracao plugs: `methods_comparison/data/processed/861_integration_qc.json`
