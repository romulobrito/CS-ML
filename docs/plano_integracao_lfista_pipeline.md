# Plano de implementacao: LFISTA no `sir_cs_pipeline_optimized.py` (integracao aditiva)

Este documento fixa a ordem de trabalho, convencoes e criterios de sucesso para integrar a trilha LFISTA como **opcao** do pipeline principal, **sem** substituir o fluxo classico.

---

## 1. Objetivo

- Expor `hybrid_lfista_frozen` e `hybrid_lfista_joint` (e `ml_only_torch` como referencia no mesmo ramo PyTorch) **no mesmo formato de artefactos** que o pipeline principal (`detailed_results.csv`, `summary_by_seed.csv`, `summary.csv`, figuras comparaveis).
- Manter **inalterados** o comportamento e os outputs dos perfis existentes quando a opcao LFISTA estiver **desligada**.

### Criterio de sucesso (formal)

> O perfil que activa LFISTA deve produzir artefactos no **mesmo formato** que o pipeline principal. Com `run_lfista=False` (default), os outputs dos perfis classicos (`paper`, `explore`, `phase0_baseline`, `solver_comparison`) devem permanecer **semanticamente equivalentes** aos do estado actual do repositorio (mesmas linhas de metodo, mesma agregacao; diferencas byte-a-byte so por ordem de linhas ou timestamps sao aceitaveis se documentadas).

---

## 2. Convencao de nomes (fixa desde a primeira integracao)

Usar **exactamente** estes identificadores de `method` em CSVs, plots e texto do paper:

| method | Descricao curta |
|--------|-----------------|
| `ml_only_torch` | MLP PyTorch so em `u` (baseline dentro do ramo LFISTA) |
| `hybrid_lfista_frozen` | MLP PyTorch fixo + bloco LFISTA treinado |
| `hybrid_lfista_joint` | MLP PyTorch + LFISTA end-to-end |

Onde replicar **antes** de fechar a integracao:

- `METHOD_COLORS` em `sir_cs_pipeline_optimized.py`
- `method_order_for_cfg` (ou equivalente)
- legendas de figuras e tabelas LaTeX quando existirem

**Nao** renomear `ml_only` (sklearn) para `ml_only_torch`. Ambos podem coexistir no mesmo `detailed_results` quando o perfil LFISTA estiver activo; sao **metodos distintos**.

---

## 3. O que nao fazer na primeira integracao

- Nao substituir `hybrid` (FISTA classico) nem `ml_only` pelo ramo PyTorch.
- Nao misturar treino PyTorch e inferencia FISTA no **mesmo** bloco de codigo de forma ilegivel.
- Nao antecipar a comparacao **"mesmo z"** (FISTA numerico no residual identico ao forward PyTorch): fica para **fase opcional** posterior.
- Nao forcar equivalencia numerica entre `ml_only` e `ml_only_torch` (frameworks e treinos diferentes).

---

## 4. Diretorios e perfil

- **Saida de dados:** `outputs/lfista_integrated/` (ou nome final unico acordado), com subpastas `runs/<run_id>/` alinhadas ao padraio `solver_comparison` / LFISTA lab.
- **Figuras:** `paper/figures/lfista_integrated/runs/<run_id>/` (nao misturar na primeira integracao com `phase0_baseline/` ou `synthetic/`).
- **CLI:** novo perfil, por exemplo `lfista_integrated`, com `run_lfista=True`; **default** do `Config`: `run_lfista=False`.

---

## 5. Fases de implementacao

### Fase A — Modulo importavel

1. Extrair de `sir_cs_lfista.py` para `lfista_module.py` (nome final a confirmar):
   - `BackgroundMLP`, `LFISTAUnrolled`, `HybridLFISTA`
   - treino/avaliacao: funcoes autocontidas com assinaturas estaveis
2. Refactorizar `sir_cs_lfista.py` para ser **cliente** desse modulo (comportamento e artefactos iguais ao commit actual: **teste de regressao** com `--profile explore`).

**Entregavel:** `import lfista_module` funcional; script standalone inalterado para utilizadores do lab isolado.

### Fase B — Config e perfil no orquestrador

1. Em `Config`: `run_lfista: bool = False` e hiperparametros LFISTA (K, epocas, lrs, etc.).
2. `apply_config_profile`: perfil `lfista_integrated` com `run_lfista=True`, `save_dir` / `plots_subdir` dedicados, seeds e `measurement_ratios` alinhados ao Phase 0 quando for corrida "completa".
3. `parse_cli_args` / `Literal` actualizados; **default** permanece sem LFISTA.

**Entregavel:** binario principal aceita o novo perfil sem executar LFISTA nos perfis antigos.

### Fase C — Wrapper e ramo condicional

Implementar **uma** funcao de orquestracao, por exemplo:

```text
run_lfista_branch(cfg, ...) -> (detailed_df, extras opcionais)
```

- Entrada: arrays tensores ou numpy ja coerentes com o job `(seed, measurement_ratio)` (mesmo `M`, `Psi`, splits que o pipeline ja construiu).
- Saida: `DataFrame` de linhas no **mesmo schema** que `run_single_setting` usa para outros metodos (colunas `method`, `rmse`, `sample_id`, etc.).
- Em `run_single_setting`: **se** `cfg.run_lfista`, chamar o wrapper e **concatenar** ao resultado; **senao**, caminho actual sem alteracoes.

Opcional na mesma funcao: `examples` / `gt_pred_bundle` se o pipeline principal precisar para figuras 08-10; pode ficar fase C2.

**Checkpoint obrigatorio (smoke):** antes de qualquer alteracao global a plots, o novo perfil deve correr **fim-a-fim** e gerar `detailed_results.csv`, `summary_by_seed.csv`, `summary.csv` **validos** (sem erros, dimensoes coerentes).

### Fase D — Plots

**So apos** o checkpoint da Fase C:

1. Registar `ml_only_torch`, `hybrid_lfista_frozen`, `hybrid_lfista_joint` em `METHOD_COLORS` e ordem de metodos.
2. Estender `save_all_comparison_plots` / `method_order_for_cfg` para o perfil integrado.
3. Smoke visual: gerar PNGs e verificar legendas.

### Fase E — Documentacao

1. `README_SIR_CS.md`: perfil `lfista_integrated`, dependencia `torch`, tempo esperado.
2. `requirements.txt`: ja contem `torch`; referencia cruzada "opcional para perfis sem LFISTA" se fizer sentido.
3. `paper/roadmap_proximos_passos.md`: item "integracao aditiva LFISTA no pipeline principal" com checkbox.

### Fase F — Regressao

1. Correr perfis **sem** `run_lfista`: comparar `summary.csv` / linhas por metodo com versao anterior (ou teste automatizado leve).
2. Correr perfil LFISTA integrado em modo **explore** (1 seed, poucos rho, epocas baixas): fim-a-fim em minutos.

---

## 6. Fase opcional (posterior): comparacao "mesmo z"

- Avaliacao FISTA numerico no `z` produzido pelo MLP PyTorch **no teste**, grelha de `lambda` em validacao.
- Metodo sugerido: `hybrid_fista_same_z` ou nome fixado no mesmo espirito da Sec. 2.
- **Nao** bloqueia as Fases A-F.

---

## 7. Resumo da ordem (com ajustes)

| Ordem | Tarefa | Checkpoint |
|-------|--------|------------|
| A | `lfista_module.py` + refactor `sir_cs_lfista.py` | Lab standalone OK |
| B | `Config` + perfil + CLI | Default sem LFISTA |
| C | `run_lfista_branch` + concat em `run_single_setting` | CSVs gerados **antes** de plots |
| D | Cores e figuras | Legendas legiveis |
| E | README + roadmap | — |
| F | Regressao perfis classicos + smoke LFISTA | Criterio Sec. 1 |

---

## 8. Uma frase

Integrar o LFISTA como **biblioteca importavel** e **trilha opcional** no orquestrador principal, com **wrapper unico**, **nomes fixos**, **pastas proprias**, **smoke de dados antes de plots**, e **criterio de sucesso** que preserva os perfis classicos quando `run_lfista=False`.

---

## 9. Checklist de conclusao (estado no repositorio)

| ID | Entrega | Concluido |
|----|---------|-----------|
| A | `lfista_module.py` (modelos, treino, `run_lfista_experiment_dataframe`) | sim |
| A | `sir_cs_lfista.py` refactorizado para importar o modulo | sim |
| B | `Config.run_lfista`, hiperparametros, perfis `lfista_integrated` / `lfista_integrated_explore`, CLI | sim |
| C | `run_lfista_branch`, concat em `run_single_setting`, CSVs no mesmo schema | sim |
| D | `METHOD_COLORS`, `method_order_for_cfg`, `save_all_comparison_plots` (+ 11/12 vs `ml_only_torch` quando aplicavel) | sim |
| E | `README_SIR_CS.md`, `paper/roadmap_proximos_passos.md`, nota em `requirements.txt` | sim |
| F | Smoke: `explore` classico + `lfista_integrated_explore`; perfis classicos sem `run_lfista` inalterados por defeito | sim (manual) |

Fase opcional Sec. 6 (**mesmo z**): pendente por desenho (nao faz parte de A-F).
