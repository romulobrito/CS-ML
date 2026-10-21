# Roadmap operacional SIR-CS (unificado, aderente ao codigo e atualizado apos LFISTA)

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

**Atualizacao importante:** a trilha **LFISTA unrolled** deixou de ser apenas conceptual. O codigo ja integra LFISTA ao pipeline principal, e os testes sinteticos actuais indicam a hierarquia:

$$
\texttt{hybrid\_lfista\_joint} > \texttt{hybrid\_lfista\_frozen} > \texttt{hybrid\_fista} \gg \texttt{ml\_only},\texttt{ml\_only\_torch},\texttt{hybrid\_spgl1}.
$$

**Consequencia pratica:** o objectivo imediato deixa de ser "provar viabilidade do LFISTA" e passa a ser **testar sua robustez fora do caso base**, sem perder a comparacao com o hibrido classico.

**Posicao do weighted:** contribuicao secundaria apenas se a etapa de diagnostico do prior (Etapa 5) confirmar utilidade robusta.

**Proxima execucao recomendada:** iniciar **Etapa 3 — robustez do `hybrid_lfista_joint`**, comecando por ablacoes em `residual_k`, `measurement_noise_std`, `residual_amplitude` e `output_noise_std`, sempre comparando contra `hybrid_fista` e `ml_only`.

---

## Estado atual do repositorio (aderente ao codigo)

### Pipeline sintetico principal

- [x] Existe `sir_cs_pipeline_optimized.py` com perfis CLI: `paper`, `explore`, `phase0_baseline`, `solver_comparison`, `lfista_integrated`, `lfista_integrated_explore`, `lfista_vs_classical`, `lfista_vs_classical_explore`.
- [x] `run_single_setting(...)` ja concatena resultados classicos e LFISTA no mesmo `detailed_results.csv` quando `run_lfista=True`.
- [x] `METHOD_COLORS`, `method_order_for_cfg(...)`, figuras 01--12 e parity/residual plots ja suportam metodos LFISTA.

### Etapa 0 implementada

- [x] `phase0_baseline` com 10 seeds e $\rho \in \{0.2,0.3,0.4,0.5,0.6\}$ no codigo.
- [x] Agregacao: primeiro por seed, depois entre seeds (`summary_by_seed.csv`, `summary.csv`).
- [x] Figuras comparativas 01--10 geradas pelo pipeline.

### Etapa 1 implementada e analisada no essencial

- [x] Perfil `solver_comparison` no codigo.
- [x] Comparacao FISTA vs SPGL1 para `hybrid` e `cs_only` via `dual_cs_solver=True`.
- [x] Dispatcher `solve_sparse_alpha(..., cs_engine=...)`.
- [x] Artefactos por run: `runs/<run_id>/` com `README_RUN.txt`, `PROTOCOL.txt`, `run_console.log`, symlink `LATEST`.
- [~] `weighted_hybrid` com SPGL1 **nao** suportado; o codigo bloqueia weighted no ramo SPGL1.
- [x] Conclusao practica atual: **FISTA permanece como solver classico de referencia**; SPGL1, do jeito que esta integrado hoje, nao e competitivo.

### Etapa 2 implementada e validada no caso base

- [x] Integracao aditiva do LFISTA no pipeline sintetico (`lfista_module.py`, `run_lfista`, `run_lfista_branch`, perfis `lfista_integrated*`, `lfista_vs_classical*`).
- [x] Laboratorio standalone `sir_cs_lfista.py` + mesmo nucleo importavel.
- [x] Comparacoes directas no protocolo sintetico base entre `hybrid_fista`, `hybrid_lfista_frozen` e `hybrid_lfista_joint`.
- [x] Evidencia atual: `hybrid_lfista_frozen` e `hybrid_lfista_joint` superam `hybrid_fista`; `joint` supera `frozen` de forma consistente, embora moderada.
- [ ] Comparacao opcional futura **"mesmo z"** (FISTA no residual do MLP PyTorch) — nao bloqueia o roadmap.

---

## Etapa 0 — Consolidar o baseline sintetico atual

**Objectivo:** ponto de referencia **reprodutivel** antes de mexer no nucleo sparse, no gerador ou no acoplamento treino-solver.

### Configuracao canonica

- [x] Perfil `phase0_baseline`
- [x] Seeds fixas (10): `7,13,23,29,31,37,41,43,47,53`
- [x] `measurement_kind="gaussian"`, `residual_basis="identity"`, `residual_k=6`, `residual_mode="support_from_u"`
- [x] `measurement_noise_std=0.02`, `output_noise_std=0.01`
- [x] Grade refinada de $\lambda$ no FISTA (`l1_lambda_grid`)

### Artefactos esperados

- [x] `outputs/phase0_baseline/`: `config.json`, `summary.csv`, `summary_by_seed.csv`, `detailed_results.csv`, `PROTOCOL.txt`
- [x] Figuras em `paper/figures/phase0_baseline/`
- [x] Texto/tabela no LaTeX alinhado ao Phase 0

### Criterio de conclusao

- [x] Historia coerente: `hybrid` melhor que `ml_only` em todos os $\rho$; `cs_only` pior; `weighted` entre `ml_only` e `hybrid` neste regime.

### Accao opcional

- [ ] Snapshot imutavel `outputs/phase0_reference/` (copiar uma vez, nunca sobrescrever).

---

## Etapa 1 — Comparacao de solver classico no nucleo sparse (FISTA vs SPGL1)

**Objectivo:** separar qualidade da formulacao de qualidade do solver. **Antes** dos baselines externos e **antes** da robustez ampla.

### Metodos comparados (mesmo protocolo da Etapa 0)

- [x] `hybrid_fista`, `hybrid_spgl1`
- [x] `cs_only_fista`, `cs_only_spgl1`
- [x] `ml_only` (referencia)
- [ ] `weighted_hybrid_fista` / `weighted_hybrid_spgl1` ficam fora desta etapa enquanto weighted-SPGL1 nao estiver suportado

### Saidas

- [x] Pasta base: `outputs/solver_comparison/`
- [x] Por run: `outputs/solver_comparison/runs/<run_id>/` com CSVs, `config.json`, `PROTOCOL.txt`, `README_RUN.txt`, `run_console.log`
- [x] Symlink `outputs/solver_comparison/LATEST`
- [x] Figuras: `paper/figures/solver_comparison/runs/<run_id>/`

### Decisao metodologica atual

- [x] FISTA mantido como **solver classico principal**.
- [~] SPGL1 preservado como benchmark/controle numerico; nao e prioridade optimizar essa linha agora.
- [ ] Quadro resumo final para documentacao: metodo | solver | RMSE medio | MAE medio | tempo | observacao.

---

## Etapa 2 — LFISTA unrolled (solver treinavel e acoplado ao treino)

**Objectivo:** sair do regime desacoplado (treinar background, depois CS no residual) e testar ganho com treino **end-to-end**.

### Justificativa

No pipeline classico, o MLP de background e o solver sparse estao desacoplados. LFISTA permite aprender o bloco proximal e, no modo `joint`, adaptar tambem o background a essa dinamica.

### Estado atual

#### Variantes implementadas e testadas

- [x] `hybrid_lfista_frozen` = background PyTorch treinado, congelado, bloco LFISTA treinado no residual
- [x] `hybrid_lfista_joint` = treino conjunto background + bloco LFISTA
- [ ] Opcional futuro: `weighted_hybrid_lfista`

#### Formulacao-alvo

- [x] Residuo observado $z = b - M f_\theta(u)$
- [x] Bloco unrolled com $K$ camadas e shrinkage aprendivel
- [x] Saida $\hat{y} = f_\theta(u) + \Psi \alpha^K$

#### Losses

- [x] Loss principal em $\hat{y}$
- [ ] Supervisao opcional em $\alpha$ permanece futura

### Conclusao da etapa

- [x] LFISTA mostrou ganho consistente sobre `hybrid_fista` no caso base.
- [x] `joint` supera `frozen` de forma consistente.
- [x] LFISTA passa a ser **candidato principal da trilha aprendida**.
- [~] Etapa concluida para o caso base; falta apenas validar robustez para promove-lo a nucleo principal do paper.

---

## Etapa 3 — Robustez do modelo principal (prioridade atual)

**Objectivo:** verificar se o ganho do LFISTA nao depende de uma condicao estreita. **Uma variavel de cada vez**, com o resto fixo ao baseline.

> **Modelo principal atual para robustez:** `hybrid_lfista_joint`.
>
> **Referencias obrigatorias:** `hybrid_fista` e `ml_only`.
>
> **Controles secundarios:** `hybrid_lfista_frozen`; opcionais `ml_only_torch`, `cs_only_fista`.

### Comparacao minima obrigatoria em toda ablacao

- [ ] `ml_only`
- [ ] `hybrid_fista`
- [ ] `hybrid_lfista_joint`
- [ ] `hybrid_lfista_frozen` (controle secundario recomendado)

### Etapa 3A — robustez continua (rodar primeiro)

Estas ablacoes sao as mais importantes **agora**.

| ID | Variavel | Valores sugeridos | Status |
| -- | -------- | ----------------- | ------ |
| E3A.1 | Esparsidade `residual_k` | $\{2,4,6,8,12,16\}$ | [ ] |
| E3A.2 | Ruido de medicao `measurement_noise_std` | $\{0,0.01,0.02,0.05,0.1\}$ | [ ] |
| E3A.3 | Amplitude da inovacao `residual_amplitude` | $\{0.4,0.8,1.2,1.6,2.0\}$ | [ ] |
| E3A.4 | Ruido de saida `output_noise_std` | $\{0,0.005,0.01,0.02,0.05\}$ | [ ] |
| E3A.5 | Razao $\rho=m/N$ ampliada | $\{0.1,0.2,0.3,0.4,0.5,0.6,0.8\}$ | [ ] |

### Ordem recomendada dentro da Etapa 3A

1. [ ] `residual_k`
2. [ ] `measurement_noise_std`
3. [ ] `residual_amplitude`
4. [ ] `output_noise_std`
5. [ ] grade ampliada de $\rho$

### Pastas de saida sugeridas

- [ ] `outputs/robustness_k/`
- [ ] `outputs/robustness_measurement_noise/`
- [ ] `outputs/robustness_amplitude/`
- [ ] `outputs/robustness_output_noise/`
- [ ] `outputs/robustness_ratio/`

### Figuras minimas por ablacao

- [ ] RMSE vs variavel
- [ ] MAE vs variavel
- [ ] relative $\ell_2$ vs variavel
- [ ] ganho sobre `ml_only`
- [ ] parity plots dos melhores metodos
- [ ] traces enxutas: `ground_truth`, `hybrid_fista`, `hybrid_lfista_joint`

### Criterio de sucesso da Etapa 3A

- [ ] `hybrid_lfista_joint` mantem vantagem sobre `hybrid_fista` em parte substancial da grade
- [ ] o ganho nao colapsa imediatamente fora do caso base
- [ ] a diferenca `joint` vs `frozen` permanece pelo menos em parte dos regimes

### Etapa 3B — robustez estrutural (rodar depois)

| ID | Experimento | Valores sugeridos | Status |
| -- | ----------- | ----------------- | ------ |
| E3B.1 | `residual_basis` | `identity`, `dct` | [ ] |
| E3B.2 | `residual_mode` | `support_from_u`, `random` | [ ] |
| E3B.3 | `measurement_kind` | `gaussian`, `subsample` | [ ] |

### Pastas de saida sugeridas

- [ ] `outputs/structural_basis/`
- [ ] `outputs/structural_support_mode/`
- [ ] `outputs/structural_measurement_kind/`

### Criterio de sucesso da Etapa 3B

- [ ] produzir um **mapa de validade** do LFISTA e do hibrido classico
- [ ] identificar em que regimes a vantagem do LFISTA se mantem, diminui ou desaparece

---

## Etapa 4 — Robustez estrutural avancada (apos Etapa 3)

**Objectivo:** sair do caso favoravel sem ainda usar dado real, em cenarios mais duros que exigem pequenas extensoes do gerador.

| ID | Experimento | Notas |
| -- | ----------- | ----- |
| E4.1 | Mismatch de base | separar `generation_basis` e `recovery_basis` no `Config` |
| E4.2 | Residual compressivel | modo `compressible` |
| E4.3 | Sensores nao ideais / mascaras mais estruturadas | extensao de `measurement_kind` |

### Pastas (exemplo)

- [ ] `outputs/structural_basis_mismatch/`
- [ ] `outputs/structural_compressible/`
- [ ] `outputs/structural_sensor_masks/`

### Criterio

- [ ] o metodo nao precisa ganhar sempre; obter **mapa de validade** mais realista.

---

## Etapa 5 — Diagnostico da variante `weighted_hybrid`

**Objectivo:** decidir se weighted merece contribuicao secundaria real.

### Metricas do prior (implementar no pipeline)

- [ ] Correlacao $\mathrm{corr}(\hat{\alpha}_{\mathrm{pred}}, \alpha^\star)$
- [ ] Precision / recall de suporte; overlap top-$k$
- [ ] MSE em $\alpha$ (opcional)

### Experimentos

- [ ] E5.1 Qualidade do prior sozinho
- [ ] E5.2 Esquemas de peso: `inverse_magnitude`, top-$k$ binario, clipping, `weight_power`
- [ ] E5.3 Oracle weighted

### Pastas (exemplo)

- [ ] `outputs/weighted_prior_quality/`
- [ ] `outputs/weighted_weight_schemes/`
- [ ] `outputs/weighted_oracle/`

### Leitura

- Oracle melhora e prior aprendido nao $\Rightarrow$ gargalo no preditor
- Oracle nao melhora $\Rightarrow$ weighted perde prioridade

---

## Etapa 6 — Baselines externos estabelecidos

**Objectivo:** o hibrido LFISTA compete com alternativas conhecidas?

### Baselines minimos sugeridos

- [ ] Ridge; Elastic Net / Lasso multi-saida
- [ ] PCA + regressao + reconstrucao
- [ ] Autoencoder + regressao no latente
- [ ] Random Forest ou boosting, se viavel em $N=128$

### Comparar

- [ ] `hybrid_lfista_joint`, `hybrid_fista`, `ml_only`, `ml_only_torch`, baselines externos
- [ ] manter `cs_only_*` so como controlo, nao como competidor principal

### Saida

- [ ] `outputs/external_baselines/`
- [ ] figuras agrupadas, ganho sobre melhor baseline classico, parity dos melhores

---

## Etapa 7 — Cenario semi-realista

**Objectivo:** aproximar do problema final sem dataset real completo.

- [ ] Saida 1D/2D mais estruturada
- [ ] mascara de sensores
- [ ] base mais fisica (DCT, wavelet, etc.)
- [ ] eventos localizados
- [ ] ruido heterogeneo
- [ ] TV / restricoes simples **so** se a versao basica estiver consolidada

### Pastas (exemplo)

- [ ] `outputs/semi_realistic_1d/`
- [ ] `outputs/semi_realistic_2d/`
- [ ] `outputs/semi_realistic_sensor_mask/`

---

## Etapa 8 — Caso real

**Objectivo:** validade externa.

- [ ] Pipeline aceita `X_train, X_val, X_test`, `Y_*`, $M$ externo, $\Psi$ externa
- [ ] Comparar solver principal + baselines externos
- [ ] Saida `outputs/real_case/`

---

## Ordem operacional resumida (atualizada)

1. [x] Etapa 0 — baseline sintetico
2. [x] Etapa 1 — FISTA vs SPGL1 (decisao practica: FISTA classico como referencia)
3. [x] Etapa 2 — LFISTA unrolled integrado e validado no caso base
4. [ ] Etapa 3A — robustez continua do `hybrid_lfista_joint`
5. [ ] Etapa 3B — robustez estrutural basica
6. [ ] Etapa 4 — robustez estrutural avancada
7. [ ] Etapa 5 — diagnostico weighted
8. [ ] Etapa 6 — baselines externos
9. [ ] Etapa 7 — semi-realista
10. [ ] Etapa 8 — caso real

**Nota:** a prioridade imediata e **robustez do LFISTA**, nao mais integracao ou prova de viabilidade.

---

## O que entra no paper em cada fase

| Conteudo | Estado |
| -------- | ------ |
| Formulacao `hybrid`; comparacao `ml_only`, `cs_only`; ganho com $\rho$ | [x] Nucleo atual do draft (Phase 0) |
| Comparacao FISTA vs SPGL1 | [x] Resultado metodologico ja claro; falta consolidar quadro resumo |
| LFISTA unrolled | [x] Ja justificado no sintetico base; falta seccao final robusta |
| Robustez do LFISTA | [ ] Proxima seccao experimental prioritaria |
| Weighted como secundario + diagnostico | [ ] Apos Etapa 5 |
| Baselines externos | [ ] Apos Etapa 6 |
| Semi-real / real | [ ] Apos Etapas 7--8 |

---

## Criterios de avanco entre etapas

| Transicao | Condicao |
| --------- | -------- |
| 0 $\to$ 1 | Baseline estavel e documentado |
| 1 $\to$ 2 | Solver classico principal escolhido com confianca |
| 2 $\to$ 3 | LFISTA validado no caso base |
| 3 $\to$ 4 | Ganho do `hybrid_lfista_joint` robusto em parte substancial das ablacoes basicas |
| 4 / 5 $\to$ 6 | Regiao de validade mapeada; weighted validado ou conscientemente rebaixado |
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
- [ ] Versao enxuta das traces: so GT + `hybrid_fista` + `hybrid_lfista_joint`
- [ ] Figura de robustez principal: ganho do LFISTA joint sobre `hybrid_fista` nas ablacoes E3A

**Mensagem:** parity + traces complementam RMSE/MAE; mostram ganho estrutural, nao so escalar.

---

## Checklist tecnico e aderencia ao codigo

### O codigo ja cobre

- [x] Perfil `solver_comparison`, dispatcher FISTA/SPGL1, `dual_cs_solver`
- [x] Por run: `README_RUN.txt`, `run_console.log`, `runs/<run_id>/`, `LATEST`
- [x] Integracao LFISTA com `run_lfista=True`
- [x] Perfis `lfista_integrated*` e `lfista_vs_classical*`
- [x] Plots combinados com LFISTA no pipeline principal

### Ainda nao implementado

- [ ] Weighted SPGL1
- [ ] Metricas explicitas de qualidade do prior (`alpha_pred`)
- [ ] `generation_basis` vs `recovery_basis`
- [ ] Modo `compressible`
- [ ] CSV dedicado de tempos por solver / por etapa LFISTA
- [ ] Perfis de robustez automatizados (Etapa 3A / 3B)

### Campos sugeridos em `Config` (futuro)

```python
# exemplo — nomes ilustrativos
generation_basis: Optional[str] = None
recovery_basis: Optional[str] = None
run_oracle_weighted: bool = False
save_alpha_prior_metrics: bool = False
experiment_tag: str = "default"
robustness_axis: Optional[str] = None
robustness_values: Optional[List[float]] = None
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
- A integracao aditiva do LFISTA no pipeline principal foi concluida antes da robustez ampla.
- Os testes actuais favorecem `hybrid_lfista_joint` como candidato principal da trilha aprendida; a prioridade passa a ser robustez e nao mais integracao.
