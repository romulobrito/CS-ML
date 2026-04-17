# Roadmap operacional SIR-CS (unificado, aderente ao codigo)

Este ficheiro e a **fonte de verdade** para a ordem das experiencias, pastas de saida, criterios de avanco e relacao entre o estado atual do codigo e as proximas etapas. Atualize os checkboxes ao concluir cada item.

---

## Legenda de estado

| Simbolo | Significado |
| ------- | ----------- |
| `[x]`   | Concluido no estado atual do repositorio |
| `[~]`   | Parcial / em progresso |
| `[ ]`   | Pendente |

---

## Visao macro

**Nucleo do paper:** `hybrid` vs `ml_only` vs `cs_only`, com ganho crescente em $\rho=m/N$ no sintetico controlado.

**Mensagem metodologica central:** aprender a componente global com ML e recuperar apenas a inovacao esparsa por CS.

**Posicao do weighted:** contribuicao secundaria apenas se a etapa de diagnostico do prior (Etapa 5) confirmar utilidade robusta.

**Ramificacao importante:** alem da comparacao de solvers classicos (FISTA vs SPGL1), existe uma trilha futura de **solver treinavel e acoplado ao treino**, via **LFISTA unrolled** (Etapa 2 abaixo).

**Proxima execucao recomendada:** fechar **Etapa 1** (correr `python sir_cs_pipeline_optimized.py --profile solver_comparison` com `pip install -r requirements.txt` para `pylops` e `spgl1`), arquivar resultados por `run_id`, preencher o quadro resumo; em seguida avaliar **Etapa 2 (LFISTA)** ou robustez conforme prioridade.

---

## Estado atual do repositorio (aderente ao codigo)

### Pipeline sintetico principal

- [x] Existe `sir_cs_pipeline_optimized.py` com perfis CLI: `paper`, `explore`, `phase0_baseline`, `solver_comparison`.

### Etapa 0 implementada

- [x] `phase0_baseline` com 10 seeds e $\rho \in \{0.2,0.3,0.4,0.5,0.6\}$ no codigo.
- [x] Agregacao: primeiro por seed, depois entre seeds (`summary_by_seed.csv`, `summary.csv`).
- [x] Figuras comparativas 01--10 geradas pelo pipeline.

### Etapa 1 parcialmente / majoritariamente implementada

- [x] Perfil `solver_comparison` no codigo.
- [x] Comparacao FISTA vs SPGL1 para `hybrid` e `cs_only` via `dual_cs_solver=True`.
- [x] Dispatcher `solve_sparse_alpha(..., cs_engine=...)`.
- [x] Artefactos por run: `runs/<run_id>/` com `README_RUN.txt`, `PROTOCOL.txt`, `run_console.log`, symlink `LATEST`.
- [~] `weighted_hybrid` com SPGL1 **nao** suportado; o codigo bloqueia weighted no ramo SPGL1.

### LFISTA

- [ ] **Nao implementado** no pipeline sintetico.
- [x] Faz sentido metodologicamente; o draft pode mencionar treino task-aware com metodos proximais desenrolados quando se quiser diferenciabilidade ponta a ponta.

---

## Etapa 0 — Consolidar o baseline sintetico atual

**Objetivo:** ponto de referencia **reprodutivel** antes de mexer no nucleo sparse, no gerador ou no acoplamento treino-solver.

### Configuracao canonica

- [x] Perfil `phase0_baseline`
- [x] Seeds fixas (10): `7,13,23,29,31,37,41,43,47,53`
- [x] `measurement_kind="gaussian"`, `residual_basis="identity"`, `residual_k=6`, `residual_mode="support_from_u"`
- [x] `measurement_noise_std=0.02`, `output_noise_std=0.01`
- [x] Grade refinada de $\lambda$ no FISTA (`l1_lambda_grid`)

### Artefactos esperados

- [x] `outputs/phase0_baseline/`: `config.json`, `summary.csv`, `summary_by_seed.csv`, `detailed_results.csv`, `PROTOCOL.txt`
- [x] Figuras em `paper/figures/phase0_baseline/`
- [x] Texto/tabela no LaTeX (`paper/experiments_synthetic.tex`, `main.tex`) alinhado ao Phase 0
- [x] Repositorio publico: [CS-ML](https://github.com/romulobrito/CS-ML) (branch `main`)

### Criterio de conclusao

- [x] Historia coerente: `hybrid` melhor que `ml_only` em todos os $\rho$; `cs_only` pior; `weighted` entre `ml_only` e `hybrid` neste regime.

### Acao opcional

- [ ] Snapshot imutavel `outputs/phase0_reference/` (copiar uma vez, nunca sobrescrever).

---

## Etapa 1 — Comparacao de solver classico no nucleo sparse (FISTA vs SPGL1)

**Objetivo:** separar qualidade da formulacao de qualidade do solver. **Antes** de baselines externos e antes da trilha LFISTA.

**Esta etapa e a proxima execucao recomendada se ainda nao foi corrida e analisada por completo (incl. quadro resumo).**

### Metodos a comparar (mesmo protocolo da Etapa 0)

- [x] `hybrid_fista`, `hybrid_spgl1`
- [x] `cs_only_fista`, `cs_only_spgl1`
- [ ] `weighted_hybrid_fista` / `weighted_hybrid_spgl1` **fora desta etapa** enquanto weighted-SPGL1 nao estiver suportado
- [x] `ml_only` (referencia)

### Implementacao aderente ao codigo

- [x] `Config.config_profile = "solver_comparison"`
- [x] `dual_cs_solver=True`
- [x] Grades: `l1_lambda_grid` (FISTA), `spgl1_tau_grid` (SPGL1)
- [x] `spgl1_lasso_alpha(...)` via `pylops.MatrixMult(A)` + `pylops.optimization.sparsity.spgl1` (requer pacotes `pylops` e `spgl1`)

### Saidas

- [x] Pasta base: `outputs/solver_comparison/`
- [x] Por run: `outputs/solver_comparison/runs/<run_id>/` com CSVs, `config.json`, `PROTOCOL.txt`, `README_RUN.txt`, `run_console.log`
- [x] Symlink `outputs/solver_comparison/LATEST`
- [x] Figuras: `paper/figures/solver_comparison/runs/<run_id>/` (01--10, ordem de metodos alargada)

### Medir

- [x] RMSE, MAE, relative $\ell_2$, support F1
- [~] Tempo por bloco nos logs (`block_time`); **sem** CSV dedicado de tempos por solver
- [x] `lambda` (FISTA) e `tau` (SPGL1) nos CSVs

### Criterio de decisao

- SPGL1 **claramente melhor ou mais robusto** $\Rightarrow$ solver classico principal no resto do roadmap
- **Empate** $\Rightarrow$ manter FISTA por simplicidade e custo
- SPGL1 **pior** $\Rightarrow$ manter FISTA; SPGL1 como benchmark / controle numerico

### Entregavel

- [ ] Quadro resumo: metodo | solver | RMSE medio | MAE medio | tempo | observacao

---

## Etapa 2 — LFISTA unrolled (solver treinavel e acoplado ao treino)

**Objetivo:** sair do regime desacoplado (treinar background, depois CS no residual) e testar ganho com treino **end-to-end**.

### Justificativa

No pipeline atual o MLP de background e o solver sparse estao **desacoplados**. O draft pode prever treino task-aware com metodos proximais desenrolados quando a diferenciabilidade for desejavel.

### Posicao no roadmap

**Depois** da Etapa 1 e **antes** dos baselines externos. Ordem logica:

1. Escolher solver classico de referencia (Etapa 1)
2. Implementar LFISTA unrolled
3. Comparar solver aprendido vs classicos
4. So depois ampliar baselines e cenarios mais dificeis

### Escopo

#### Variantes

- [ ] `hybrid_lfista` = MLP de background + bloco LFISTA unrolled no residual
- [ ] Opcional futuro: `weighted_hybrid_lfista`

#### Formulacao-alvo (esboco)

- [ ] Residuo observado $z = b - M f_\theta(u)$
- [ ] Bloco unrolled com $K$ camadas: $\alpha^{k+1} = \mathcal{S}_{\tau_k}( v^k - \eta_k A^\top(A v^k - z) )$ com $A=M\Psi$ (aceleracao tipo FISTA opcional)
- [ ] Saida $\hat{y} = f_\theta(u) + \Psi \alpha^K$

#### Losses

- [ ] Loss principal: $\|\hat{y}-y\|_2^2$
- [ ] No sintetico, opcional: supervisao em $\alpha$, ex. $\lambda_\alpha \|\alpha^K-\alpha^\star\|_1$

### Implementacao minima sugerida

- [ ] Modulo PyTorch `LFISTAUnrolled` ou `UnrolledSparseBlock` com parametros $\eta_k$, $\tau_k$, $K$ fixo
- [ ] Novo `config_profile` (ex. `lfista_baseline`), metodo `hybrid_lfista`, rotina de treino end-to-end
- [ ] Artefactos: `outputs/lfista_baseline/`, `paper/figures/lfista_baseline/`

### Comparacoes minimas

- [ ] `ml_only`
- [ ] `hybrid_best_classical_solver` (FISTA ou SPGL1 conforme Etapa 1)
- [ ] `hybrid_lfista`
- [ ] Opcional: `cs_only_best_classical_solver`

### Medir

- [ ] RMSE, MAE, relative $\ell_2$, support F1; tempo de treino e inferencia; estabilidade vs $K$

### Criterio de decisao

- LFISTA **melhora de forma robusta** $\Rightarrow$ trilha principal ou segunda contribuicao forte
- **Empate** $\Rightarrow$ extensao metodologica; nucleo do paper no solver classico
- **Piora / instabiliza** $\Rightarrow$ linha exploratoria documentada, nao nucleo

---

## Etapa 3 — Robustez do `hybrid` com o solver principal escolhido

**Objetivo:** o ganho nao depende de uma condicao estreita. **Uma variavel de cada vez**, resto fixo ao baseline.

> "Solver principal" = melhor solver **classico** da Etapa 1, ou LFISTA se a Etapa 2 for promovido com evidencia robusta.

| ID | Variavel | Valores sugeridos |
| -- | -------- | ----------------- |
| E3.1 | Esparsidade $k$ | $\{2,4,6,8,12,16\}$ |
| E3.2 | Ruido de medicao $\sigma_\eta$ | $\{0,0.01,0.02,0.05,0.1\}$ |
| E3.3 | Ruido de saida $\sigma_{\mathrm{out}}$ | $\{0,0.005,0.01,0.02,0.05\}$ |
| E3.4 | Amplitude da inovacao | `residual_amplitude` $\in \{0.4,0.8,1.2,1.6,2.0\}$ |
| E3.5 | Razao $\rho$ | $\{0.1,0.2,\ldots,0.6,0.8\}$ |

### Comparar sempre

- [ ] `ml_only`, `hybrid_best_solver`, `weighted_hybrid_best_solver` (se fizer sentido), `cs_only_best_solver`, `hybrid_lfista` (se Etapa 2 ativa)

### Pastas de saida (exemplo)

- [ ] `outputs/ablation_k/`, `ablation_measurement_noise/`, `ablation_output_noise/`, `ablation_amplitude/`, `ablation_ratio/`

### Figuras minimas por ablacao

- [ ] RMSE vs variavel; MAE; ganho sobre `ml_only`; relative $\ell_2$; support F1

### Criterio de avanco

- [ ] `hybrid` vence `ml_only` na maior parte da grade razoavel; ganho nao desaparece ao primeiro desvio do caso base.

---

## Etapa 4 — Robustez estrutural

**Objetivo:** sair do caso favoravel sem ainda usar dado real.

| ID | Experimento | Notas |
| -- | ----------- | ----- |
| E4.1 | `residual_basis` | `identity`, `dct` |
| E4.2 | `residual_mode` | `support_from_u`, `random` |
| E4.3 | `measurement_kind` | `gaussian`, `subsample` |
| E4.4 | Mismatch de base | separar `generation_basis` e `recovery_basis` no `Config` |
| E4.5 | Residual compressivel | modo `compressible` |

### Pastas (exemplo)

- [ ] `outputs/structural_basis/`, `structural_support_mode/`, `structural_measurement_kind/`, `structural_basis_mismatch/`, `structural_compressible/`

### Criterio

- [ ] O metodo nao precisa ganhar sempre; obter **mapa de validade** (onde funciona / onde degrada / onde a hipotese deixa de valer).

---

## Etapa 5 — Diagnostico da variante `weighted_hybrid`

**Objetivo:** decidir se weighted merece contribuicao secundaria real.

### Metricas do prior (implementar no pipeline)

- [ ] Correlacao $\mathrm{corr}(\hat{\alpha}_{\mathrm{pred}}, \alpha^\star)$
- [ ] Precision / recall de suporte; overlap top-$k$
- [ ] MSE em $\alpha$ (opcional)

### Experimentos

- [ ] E5.1 Qualidade do prior sozinho
- [ ] E5.2 Esquemas de peso: `inverse_magnitude`, top-$k$ binario, clipping, `weight_power`
- [ ] E5.3 Oracle weighted

### Pastas (exemplo)

- [ ] `outputs/weighted_prior_quality/`, `weighted_weight_schemes/`, `weighted_oracle/`

### Leitura

- Oracle melhora e prior aprendido nao $\Rightarrow$ gargalo no preditor
- Oracle nao melhora $\Rightarrow$ weighted perde prioridade

---

## Etapa 6 — Baselines externos estabelecidos

**Objetivo:** o hibrido compete com alternativas conhecidas?

### Baselines minimos sugeridos

- [ ] Ridge; Elastic Net / Lasso multi-saida
- [ ] PCA + regressao + reconstrucao
- [ ] Autoencoder + regressao no latente
- [ ] Random Forest ou boosting, se viavel em $N=128$

### Comparar

- [ ] `hybrid_best_solver`, `hybrid_lfista` (se validado), `ml_only`, `cs_only_best_solver`, baselines externos

### Saida

- [ ] `outputs/external_baselines/` + figuras agrupadas, ganho sobre melhor baseline classico, parity dos melhores

---

## Etapa 7 — Cenario semi-realista

**Objetivo:** aproximar do problema final sem dataset real completo.

- [ ] Saida 1D/2D mais estruturada; mascara de sensores; base mais fisica (DCT, wavelet, etc.); eventos localizados; ruido heterogeneo
- [ ] TV / restricoes simples **so** se a versao basica estiver consolidada

### Pastas (exemplo)

- [ ] `outputs/semi_realistic_1d/`, `semi_realistic_2d/`, `semi_realistic_sensor_mask/`

---

## Etapa 8 — Caso real

**Objetivo:** validade externa.

- [ ] Pipeline aceita `X_train, X_val, X_test`, `Y_*`, $M$ externo, $\Psi$ externa
- [ ] Comparar solver principal + baselines externos
- [ ] Saida `outputs/real_case/`

---

## Ordem operacional resumida

1. [x] Etapa 0 — baseline sintetico
2. [ ] Etapa 1 — FISTA vs SPGL1 (analise + quadro resumo)
3. [ ] Etapa 2 — LFISTA unrolled (end-to-end)
4. [ ] Etapa 3 — robustez do `hybrid` (E3.1--E3.5)
5. [ ] Etapa 4 — robustez estrutural (E4.1--E4.5)
6. [ ] Etapa 5 — diagnostico weighted
7. [ ] Etapa 6 — baselines externos
8. [ ] Etapa 7 — semi-realista
9. [ ] Etapa 8 — caso real

**Nota:** Etapas 3--5 podem ser refinadas em detalhe apos Etapa 1; weighted pode ser antecipado em paralelo conceptual se for prioridade de escrita.

---

## O que entra no paper em cada fase

| Conteudo | Estado |
| -------- | ------ |
| Formulacao `hybrid`; comparacao `ml_only`, `cs_only`; ganho com $\rho$ | [x] Nucleo atual do draft (Phase 0) |
| Comparacao FISTA vs SPGL1 | [ ] Apos Etapa 1 |
| LFISTA unrolled | [ ] Apos Etapa 2 |
| Weighted como secundario + diagnostico | [ ] Apos Etapa 5 |
| Baselines externos | [ ] Apos Etapa 6 |
| Semi-real / real | [ ] Apos Etapas 7--8 |

---

## Criterios de avanco entre etapas

| Transicao | Condicao |
| --------- | -------- |
| 0 $\to$ 1 | Baseline estavel e documentado |
| 1 $\to$ 2 | Solver classico principal escolhido com confianca |
| 2 $\to$ 3 | LFISTA validado ou conscientemente rebaixado |
| 3 / 4 $\to$ 6 | `hybrid` vence `ml_only` em parte substancial das condicoes |
| 5 $\to$ 6 | Weighted validado ou conscientemente rebaixado |
| 6 $\to$ 7 / 8 | Competitivo com baselines; regiao de validade mapeada |

---

## Figuras principais recomendadas

Manter no conjunto principal do paper:

- [x] RMSE e MAE vs $\rho=m/N$
- [x] Ganho sobre `ml_only`
- [x] Parity plots (GT vs $\hat{y}$)
- [x] Example traces (indice vs valor)

Complementos fortes:

- [ ] Residual vs GT: eixo $x=y_{\mathrm{true}}$, $y=\hat{y}-y$
- [ ] Subfiguras por metodo com mesma escala, identidade, correlacao e RMSE no painel
- [ ] Versao enxuta das traces: so GT + `ml_only` + `hybrid`

**Mensagem:** parity + traces complementam RMSE/MAE; mostram ganho estrutural, nao so escalar.

---

## Checklist tecnico e aderencia ao codigo

### O codigo ja cobre

- [x] Perfil `solver_comparison`, dispatcher FISTA/SPGL1, `dual_cs_solver`
- [x] Por run: `README_RUN.txt`, `run_console.log`, `runs/<run_id>/`, `LATEST`

### Ainda nao implementado

- [ ] LFISTA unrolled
- [ ] Weighted SPGL1
- [ ] Metricas explicitas de qualidade do prior (`alpha_pred`)
- [ ] `generation_basis` vs `recovery_basis`
- [ ] Modo `compressible`
- [ ] CSV dedicado de tempos por solver

### Campos sugeridos em `Config` (futuro, nao obrigatorios)

```python
# exemplo — nomes ilustrativos
generation_basis: Optional[str] = None
recovery_basis: Optional[str] = None
run_oracle_weighted: bool = False
save_alpha_prior_metrics: bool = False
experiment_tag: str = "default"
```

Cada pasta de experiencia deve conter, quando possivel: `config.json`, `summary.csv`, `summary_by_seed.csv`, `detailed_results.csv`, `README_RUN.txt` ou equivalente por run.

---

## Referencia rapida: equacao estruturada (Etapas 6+)

So quando o dominio justificar:

$$
\hat{\alpha} \in \arg\min_{\alpha} \frac{1}{2}\| M\Psi\alpha - (b - M f_\theta(u)) \|_2^2
+ \lambda_1 \| W\alpha \|_1
+ \lambda_2 \, \mathrm{TV}(\Psi\alpha)
+ \lambda_3 \, \mathcal{P}(\Psi\alpha)
$$

---

## Historico (notas)

- Corrida piloto (3 seeds, quatro $\rho$) foi substituida no texto principal pelo **Phase 0** (10 seeds, cinco $\rho$, grelha refinada).
- O draft em alto nivel pode mencionar treino task-aware com metodos proximais desenrolados; a **Etapa 2 (LFISTA)** formaliza essa trilha no roadmap.
