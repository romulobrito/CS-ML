# Roadmap operacional SIR-CS (unificado)

Este ficheiro e a **fonte de verdade** para a ordem das experiencias, pastas de saida e criterios de avanco. Atualize os checkboxes quando concluir cada item.

---

## Legenda de estado

| Simbolo | Significado |
| ------- | ----------- |
| `[x]`   | Concluido no estado atual do repositorio |
| `[~]`   | Parcial / em progresso |
| `[ ]`   | Pendente |

---

## Visao macro (uma frase)

**Nucleo do paper:** `hybrid` (FISTA ou solver escolhido na Etapa 1) vs `ml_only` vs `cs_only`, com ganho crescente em $\rho$ no sintetico controlado. **Weighted** so como contribuicao secundaria se a Etapa 4 validar.

**Proxima execucao recomendada:** Etapa 1 — comparar **FISTA vs SPGL1** no mesmo protocolo da Etapa 0, antes de baselines externos.

---

## Etapa 0 — Consolidar o baseline sintetico atual

**Objetivo:** ponto de referencia **reprodutivel** antes de mexer no solver sparse ou no gerador.

**Manter:** gerador atual; `ml_only`, `hybrid`, `weighted_hybrid`, `cs_only`; 10 seeds; $\rho \in \{0.2,0.3,0.4,0.5,0.6\}$; agregacao por seed e depois entre seeds; grelha de $\lambda$ refinada.

### Checklist operacional

- [x] Perfil `phase0_baseline` no pipeline (`sir_cs_pipeline_optimized.py --profile phase0_baseline`).
- [x] Seeds fixas (10): `7,13,23,29,31,37,41,43,47,53`.
- [x] `measurement_kind="gaussian"`, `residual_basis="identity"`, `residual_k=6`, `residual_mode="support_from_u"`, ruidos 0.02 / 0.01.
- [x] Artefactos em `outputs/phase0_baseline/`: `config.json`, `summary.csv`, `summary_by_seed.csv`, `detailed_results.csv`, `PROTOCOL.txt`.
- [x] Figuras numeradas em `paper/figures/phase0_baseline/` (01--10) e texto/tabela no LaTeX (`paper/experiments_synthetic.tex`, `main.tex`).
- [x] Repositorio publico: [CS-ML](https://github.com/romulobrito/CS-ML) (branch `main`).
- [ ] **Opcional (imutabilidade):** copiar `outputs/phase0_baseline/` para `outputs/phase0_reference/` **uma vez** e nunca sobrescrever (baseline arquivado); hoje o canónico continua a ser `phase0_baseline` gerado pelo script.

### Criterio de conclusao

- [x] Historia coerente: `hybrid` melhor que `ml_only` em todos os $\rho$; `cs_only` pior; `weighted` entre `ml_only` e `hybrid` neste regime.

---

## Etapa 1 — Comparacao de solver no nucleo sparse (FISTA vs SPGL1)

**Objetivo:** saber se o desempenho esta limitado pelo **solver** ou pela **formulacao**. **Antes** de baselines externos.

### Metodos a comparar (mesmo protocolo da Etapa 0)

- [ ] `hybrid_fista` / `hybrid_spgl1`
- [ ] `weighted_hybrid_fista` / `weighted_hybrid_spgl1`
- [ ] Opcional: `cs_only_fista` / `cs_only_spgl1`

### Implementacao (codigo)

- [ ] `Config.solver_name`: `"fista"` | `"spgl1"`.
- [ ] Dispatcher `solve_sparse_problem(..., solver_name=...)`.
- [ ] Nomes de metodo padronizados (ex.: `hybrid_fista`, `hybrid_spgl1`).
- [ ] Dependencia SPGL1 (ex. `spgl1` ou implementacao interna) documentada em `README_SIR_CS.md`.

### Saidas

- [ ] Pasta: `outputs/solver_comparison/`
- [ ] Ficheiros minimos: `summary.csv`, `summary_by_seed.csv`, `solver_times.csv` (tempo por job / por amostra), `config.json`, `README_run.txt`
- [ ] Figuras sugeridas: `fig_solver_rmse.png`, `fig_solver_mae.png`, `fig_solver_time.png`

### Medir

- [ ] RMSE, MAE, relative $\ell_2$, support F1
- [ ] Tempo total por bloco (seed, $\rho$); tempo medio por amostra no teste
- [ ] $\lambda$ (ou parametro equivalente) escolhido por metodo

### Criterio de decisao

- SPGL1 **claramente melhor ou mais robusto** $\Rightarrow$ tornar solver principal no resto do roadmap.
- **Empate** $\Rightarrow$ manter FISTA por simplicidade.
- SPGL1 **pior** $\Rightarrow$ manter FISTA e registar evidencia no paper/apendice.

### Entregavel

- [ ] Quadro resumo: metodo | solver | RMSE medio | MAE medio | tempo | observacao

---

## Etapa 2 — Robustez do `hybrid` com o solver escolhido

**Objetivo:** o ganho nao depende de uma condicao estreita. **Uma variavel de cada vez**, resto fixo ao baseline.

| ID | Variavel | Valores sugeridos |
| -- | -------- | ----------------- |
| E2.1 | Esparsidade $k$ | $\{2,4,6,8,12,16\}$ |
| E2.2 | Ruido de medicao $\sigma_\eta$ | $\{0,0.01,0.02,0.05,0.1\}$ |
| E2.3 | Ruido de saida $\sigma_{\mathrm{out}}$ | $\{0,0.005,0.01,0.02,0.05\}$ |
| E2.4 | Amplitude inovacao | `residual_amplitude` $\in \{0.4,0.8,1.2,1.6,2.0\}$ |
| E2.5 | Razao $\rho$ | $\{0.1,0.2,\ldots,0.6,0.8\}$ |

### Comparar sempre

- [ ] `ml_only`, `hybrid_best_solver`, `weighted_hybrid_best_solver`, `cs_only_best_solver`

### Pastas de saida (exemplo)

- [ ] `outputs/ablation_k/`
- [ ] `outputs/ablation_measurement_noise/`
- [ ] `outputs/ablation_output_noise/`
- [ ] `outputs/ablation_amplitude/`
- [ ] `outputs/ablation_ratio/`

### Figuras minimas por ablacao

- [ ] RMSE vs variavel; MAE vs variavel; ganho sobre `ml_only`; relative $\ell_2$; support F1

### Criterio de avanco para Etapa 5

- [ ] `hybrid` vence `ml_only` na **maior parte** da grade razoavel; ganho nao desaparece ao primeiro desvio do baseline.

---

## Etapa 3 — Robustez estrutural

**Objetivo:** sair do caso favoravel sem ainda usar dado real.

| ID | Experimento | Notas |
| -- | ----------- | ----- |
| E3.1 | `residual_basis` | `identity`, `dct` |
| E3.2 | `residual_mode` | `support_from_u`, `random` |
| E3.3 | `measurement_kind` | `gaussian`, `subsample` |
| E3.4 | Mismatch de base | gerar em `dct` / reconstruir em `identity` e vice-versa — exige separar `generation_basis` e `recovery_basis` no `Config` |
| E3.5 | Residual compressivel | modo `compressible` (nao estritamente $k$-esparso) |

### Pastas (exemplo)

- [ ] `outputs/structural_basis/`, `outputs/structural_support_mode/`, `outputs/structural_measurement_kind/`, `outputs/structural_basis_mismatch/`, `outputs/structural_compressible/`

### Criterio

- [ ] Metodo **nao** precisa ganhar sempre; precisa **mapa de validade** (onde funciona / onde degrada).

---

## Etapa 4 — Diagnostico da variante `weighted_hybrid`

**Objetivo:** decidir se weighted merece contribuicao secundaria real.

### Metricas do prior (implementar no pipeline)

- [ ] Correlacao $\mathrm{corr}(\hat{\alpha}_{\mathrm{pred}}, \alpha^\star)$
- [ ] Precision / recall de suporte; overlap top-$k$
- [ ] MSE em $\alpha$ (opcional); calibracao simples do prior

### Experimentos

- [ ] **E4.1** Qualidade do prior sozinho
- [ ] **E4.2** Esquemas de peso: `inverse_magnitude`, top-$k$ binario, clipping, `weight_power`, etc.
- [ ] **E4.3** Oracle weighted (pesos a partir do suporte verdadeiro)

### Pastas (exemplo)

- [ ] `outputs/weighted_prior_quality/`, `outputs/weighted_weight_schemes/`, `outputs/weighted_oracle/`

### Leitura

- Oracle melhora e aprendido nao $\Rightarrow$ gargalo no preditor.
- Oracle nao melhora $\Rightarrow$ weighted perde prioridade.

---

## Etapa 5 — Baselines externos estabelecidos

**Objetivo:** "o hibrido compete com alternativas conhecidas?" — **depois** da Etapa 1.

### Baselines minimos sugeridos

- [ ] Ridge; Elastic Net / Lasso multi-saida
- [ ] PCA + regressao + reconstrucao
- [ ] Autoencoder + regressao no latente
- [ ] Random Forest ou boosting, se viavel em $N=128$

### Comparar

- [ ] `hybrid_best_solver`, `ml_only`, `cs_only_best_solver`, `ridge`, `elastic_net`, `pca_regression`, `ae_regression`, ...

### Saida

- [ ] `outputs/external_baselines/` + figuras agrupadas, ganho sobre melhor baseline classico, parity dos 3 melhores

---

## Etapa 6 — Cenario semi-realista

**Objetivo:** aproximar do problema final sem dataset real completo.

- [ ] $\Psi$ mais fisica (DCT/wavelet); saida 1D/2D estruturada; mascara de sensores; eventos localizados; ruido heterogeneo
- [ ] TV / restricoes simples **so** se a versao basica estiver consolidada

### Pastas (exemplo)

- [ ] `outputs/semi_realistic_1d/`, `outputs/semi_realistic_2d/`, `outputs/semi_realistic_sensor_mask/`

---

## Etapa 7 — Caso real

**Objetivo:** validade externa.

- [ ] Pipeline aceita `X_train, X_val, X_test`, `Y_*`, $M$ externo, $\Psi$ externa, solver da Etapa 1
- [ ] Saida `outputs/real_case/`

---

## Ordem operacional resumida

1. [x] Etapa 0 — baseline sintetico (10 seeds, $\rho$ 0.2--0.6, paper + figuras)
2. [ ] **Etapa 1 — FISTA vs SPGL1**  **<= proxima execucao**
3. [ ] Etapa 2 — robustez do `hybrid` (E2.1--E2.5)
4. [ ] Etapa 3 — robustez estrutural (E3.1--E3.5)
5. [ ] Etapa 4 — diagnostico weighted (E4.1--E4.3)
6. [ ] Etapa 5 — baselines externos
7. [ ] Etapa 6 — semi-realista
8. [ ] Etapa 7 — caso real

**Nota:** Etapas 2, 3 e 4 podem ser reordenadas em detalhe apos Etapa 1; o utilizador sugeriu rodar cedo **Etapa 4** em paralelo conceptual com robustez se o weighted ainda for prioridade de escrita.

---

## O que entra no paper em cada fase

| Conteudo | Estado |
| -------- | ------ |
| Formulacao `hybrid`; comparacao `ml_only`, `cs_only`; ganho com $\rho$ | [x] Nucleo atual no draft (Phase 0) |
| Comparacao FISTA vs SPGL1 | [ ] Secao / apendice apos Etapa 1 |
| Weighted como secundario + diagnostico | [ ] Apos Etapa 4 |
| Baselines externos | [ ] Apos Etapa 5 |
| Semi-real / real | [ ] Apos Etapas 6--7 |

---

## Criterios de avanco entre etapas

| Transicao | Condicao |
| --------- | -------- |
| Etapa 0 $\to$ 1 | [x] Baseline estavel e documentado |
| Etapa 1 $\to$ 2 | Solver principal escolhido com confianca |
| Etapas 2/3 $\to$ 5 | `hybrid` vence `ml_only` em parte substancial das condicoes |
| Etapa 4 $\to$ 5 | Weighted validado ou conscientemente rebaixado |
| Etapa 5 $\to$ 6/7 | Competitivo com baselines; regiao de validade mapeada |

---

## Figuras: recomendacoes (valor cientifico)

Manter no **conjunto principal** do paper:

- [x] RMSE (e MAE) vs $\rho=m/N$
- [x] Ganho sobre `ml_only`
- [x] **Parity plots** (GT vs $\hat{y}$) — mostram vies, dispersao, colapso (ex.: `cs_only`)
- [x] **Example traces** (indice vs valor) — estrutura local qualitativa

Complementos fortes (implementar quando possivel):

- [ ] **Residual vs GT:** eixo $x=y_{\mathrm{true}}$, $y=\hat{y}-y$ (heterocedasticidade / falhas por amplitude)
- [ ] Subfiguras por metodo com **mesma escala**, identidade, correlacao e RMSE no painel
- [ ] Versao enxuta das traces: so GT + `ml_only` + `hybrid`; weighted/cs em suplemento se poluir

**Mensagem:** parity + traces **nao substituem** RMSE/MAE; mostram que o ganho e **estrutural**, nao so numerico.

---

## Checklist tecnico de implementacao (futuro)

Campos sugeridos em `Config` (ainda nao obrigatorios):

```python
solver_name: str = "fista"  # ou "spgl1" apos Etapa 1
generation_basis: Optional[str] = None
recovery_basis: Optional[str] = None
run_oracle_weighted: bool = False
save_alpha_prior_metrics: bool = False
experiment_tag: str = "default"
```

Cada pasta de experiencia deve conter, quando possivel: `config.json`, `summary.csv`, `summary_by_seed.csv`, `detailed_results.csv`, `README_run.txt`.

---

## Referencia rapida: equacao estruturada (Etapa 6+)

So quando dominio justificar (ver discussao anterior no repositorio):

$$
\hat{\alpha} \in \arg\min_{\alpha} \frac{1}{2}\| M\Psi\alpha - (b - M f_\theta(u)) \|_2^2
+ \lambda_1 \| W\alpha \|_1
+ \lambda_2 \, \mathrm{TV}(\Psi\alpha)
+ \lambda_3 \, \mathcal{P}(\Psi\alpha)
$$

---

## Historico (notas)

- Corrida **piloto** (3 seeds, quatro $\rho$, grelha antiga de $\lambda$) foi **substituida** no texto principal do paper pelo **Phase 0** (10 seeds, cinco $\rho$, grelha refinada) — ver discussao no chat: substituicao metodologicamente coerente no nucleo do PDF; material antigo pode viver em pastas separadas se quiser arquivo.
