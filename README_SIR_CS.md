
# Plano metodológico mínimo → publicável para SIR-CS

## Objetivo
Verificar a viabilidade da hipótese:

\[
y = f_\theta(u) + \Psi \alpha^\star + \xi, \qquad
b = M y + \eta
\]

onde:
- `f_theta(u)` aprende a componente global/previsível;
- `Psi alpha*` representa a inovação esparsa/compressível;
- a inferência observa apenas medidas comprimidas `b`.

---

## Etapa 0 — Critério de viabilidade mínima

A proposta é **viável** se, em dados sintéticos controlados:

1. `hybrid` superar `ml_only` em RMSE/MAE;
2. o ganho persistir em múltiplos `measurement_ratio`;
3. `cs_only` for pior que `hybrid` quando a componente global não for esparsa;
4. opcionalmente, `weighted_hybrid` melhorar `hybrid` quando o prior de suporte fizer sentido.

Se esses quatro pontos ocorrerem, você já tem sinal de que vale avançar.

---

## Etapa 1 — Experimento mínimo

Referencia recomendada: `sir_cs_pipeline_optimized.py` (mesma logica experimental, com cache de Lipschitz e protocolo de validacao alinhado ao teste). O script `sir_cs_pipeline.py` permanece como variante mais simples.

O pipeline implementa:

- geração sintética;
- baseline multi-saída (`ml_only`);
- `cs_only`;
- `hybrid`;
- `weighted_hybrid`;
- seleção de lambda em validação;
- varredura em `measurement_ratio`;
- salvamento de CSVs e figuras.

### Detalhe de protocolo (importante)

- Na versao otimizada: **um** subsample de validacao e **um** vetor de ruido `eta` em `b` sao compartilhados entre `hybrid`, `weighted_hybrid` e `cs_only` na escolha de `lambda` (comparacao justa).
- A selecao de `lambda` usa o **mesmo** `measurement_noise_std` nas medicoes `b` que o teste, quando `measurement_noise_std > 0`.

### Resumo estatistico (entre seeds)

- `detailed_results.csv`: uma linha por amostra de teste e por metodo.
- `summary_by_seed.csv`: media sobre amostras de teste **dentro de cada** `(seed, measurement_ratio, method)`.
- `summary.csv`: media **entre seeds** das medias acima, com `rmse_std_across_seeds`, `rmse_sem`, `rmse_ci95_half` e `n_seeds`. Barras no grafico usam desvio **entre seeds**, nao entre amostras de teste misturadas.

### Perfis (`--profile` na CLI)

- `paper` (default em `python sir_cs_pipeline_optimized.py`): hiperparametros do `Config` (3 seeds, grade de `lambda` refinada no codigo).
- `explore`: menos dados e grade curta via `apply_config_profile` (iteracao rapida).
- `phase0_baseline`: **Fase 0 do roadmap** (`paper/roadmap_proximos_passos.md`) — 10 seeds, `measurement_ratio` em `{0.2,0.3,0.4,0.5,0.6}`, mesma grade de `lambda` que o paper, saidas em `outputs/phase0_baseline/`, figuras em `paper/figures/phase0_baseline/`, mais `PROTOCOL.txt` no diretorio de saida.
- `solver_comparison`: **Etapa 1 do roadmap** — mesmo protocolo numerico que a Fase 0, mas com `dual_cs_solver=True`: metodos `hybrid_fista`, `hybrid_spgl1`, `cs_only_fista`, `cs_only_spgl1` mais `ml_only`. Requer **PyLops** e o pacote **spgl1** (`pip install pylops spgl1` ou `pip install -r requirements.txt`) para o ramo SPGL1. Cada execucao cria `outputs/solver_comparison/runs/<YYYYMMDD_HHMMSS>/` (CSVs, `config.json`, `PROTOCOL.txt`, `README_RUN.txt`, `run_console.log`) e `paper/figures/solver_comparison/runs/<mesmo_id>/` (PNG). Symlink `outputs/solver_comparison/LATEST` aponta para a ultima corrida.
- `lfista_integrated`: **integracao Etapa 2 no pipeline principal** (`docs/plano_integracao_lfista_pipeline.md`): mesmo protocolo que `phase0_baseline` (10 seeds, mesmos `measurement_ratio`, mesma grade de `lambda` para os metodos classicos), mais ramo PyTorch com `ml_only_torch`, `hybrid_lfista_frozen`, `hybrid_lfista_joint`. Activado por `run_lfista=True` no perfil; artefactos em `outputs/lfista_integrated/runs/<run_id>/` e figuras em `paper/figures/lfista_integrated/runs/<run_id>/`. Requer **PyTorch** (`torch` em `requirements.txt`). Tempo: ordem de **horas** em GPU para a corrida completa (10 seeds x 5 razoes x treino LFISTA); use `lfista_integrated_explore` para smoke rapido (minutos).
- `lfista_integrated_explore`: como `explore` (1 seed, poucos `measurement_ratio`, menos epocas LFISTA no codigo), mas com o mesmo ramo LFISTA e pastas `outputs/lfista_integrated/`.
- `lfista_vs_classical`: **comparacao Etapa 2 vs hibrido FISTA classico** no mesmo protocolo Phase 0 (10 seeds, mesmos `measurement_ratio`, mesmo gerador). Ativa `dual_cs_solver=True` (metodos `hybrid_fista`, `hybrid_spgl1`, `cs_only_*`) e `run_lfista=True` (`ml_only_torch`, `hybrid_lfista_*`). Saida: `outputs/lfista_vs_classical/runs/<run_id>/`, figuras em `paper/figures/lfista_vs_classical/runs/<run_id>/`. Artefactos extra: `summary_focus_ml_hybrid_fista_lfista.csv`, `summary_by_seed_focus_ml_hybrid_fista_lfista.csv`, `FOCUS_COMPARISON.txt` (subset para `ml_only`, `ml_only_torch`, `hybrid_fista`, `hybrid_lfista_frozen`, `hybrid_lfista_joint`). Requer **torch**, **pylops**, **spgl1**. Corrida completa: muito pesada (FISTA+SPGL1 por job + treino LFISTA); use `lfista_vs_classical_explore` para validar o pipeline em poucos minutos.
- `lfista_vs_classical_explore`: mesmo desenho que `lfista_vs_classical`, com hiperparametros tipo `explore` e pastas `outputs/lfista_vs_classical/`.

Opcional: `reset_warm_start_each_lambda=True` zera warm-start a cada novo `lambda` na selecao (mais limpo, mais lento).

**Nota:** os perfis sem LFISTA (`paper`, `explore`, `phase0_baseline`, `solver_comparison`) nao precisam de `torch` instalado para correr apenas o pipeline classico; `pip install -r requirements.txt` instala tudo, inclusive `torch`, para suportar LFISTA e o script `sir_cs_lfista.py`.

### Cuidado experimental

- Com `residual_basis="identity"` e `measurement_kind="subsample"`, o operador em coeficientes pode ficar **mal condicionado** (coeficientes nao observados). Para viabilidade inicial prefira `measurement_kind="gaussian"` com `identity` ou `dct`.

### Logs de andamento

No `sir_cs_pipeline_optimized.py`, `Config.log_progress` (default `True`) imprime no stdout:

- resumo do bloco (`data`, `setup`, treino);
- cada valor da grade de `lambda` por metodo (metrica e tempo);
- progresso no teste a cada `test_log_interval` amostras (use `0` para so mostrar inicio/fim do teste).

### Ambiente virtual (recomendado)

Na raiz do repositorio (a pasta `.venv/` ja esta no `.gitignore`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Rodar
```bash
python sir_cs_pipeline_optimized.py
python sir_cs_pipeline_optimized.py --profile phase0_baseline
python sir_cs_pipeline_optimized.py --profile explore
python sir_cs_pipeline_optimized.py --profile solver_comparison
python sir_cs_pipeline_optimized.py --profile lfista_integrated
python sir_cs_pipeline_optimized.py --profile lfista_integrated_explore
python sir_cs_pipeline_optimized.py --profile lfista_vs_classical
python sir_cs_pipeline_optimized.py --profile lfista_vs_classical_explore
```

Dependencias extra: perfil `solver_comparison` precisa de `pylops` e `spgl1`; perfis `lfista_integrated*` e o laboratorio `sir_cs_lfista.py` precisam de `torch`; perfis `lfista_vs_classical*` precisam de `torch` + `pylops` + `spgl1` (ver `requirements.txt`).

Saidas esperadas (perfil `paper`):
- `outputs/detailed_results.csv`
- `outputs/summary_by_seed.csv`
- `outputs/summary.csv`
- `outputs/config.json`
- `paper/figures/synthetic/` — figuras comparativas numeradas (destino padrao do pipeline):
  - `01_rmse_vs_measurement_ratio.png`, `02_mae_...`, `03_relative_l2_...`
  - `04_support_f1_...` (se houver metrica valida)
  - `05_rmse_grouped_bars_by_ratio.png`
  - `06_gain_rmse_over_ml_only.png`, `07_gain_mae_over_ml_only.png`
  - `08_example_ground_truth_vs_models.png` — curvas por indice de saida
  - `09_parity_ground_truth_vs_prediction.png` — dispersao GT vs pred (linha identidade)
  - `10_residual_distributions_gt_vs_models.png` — histogramas de `y_hat - y`

Com `lfista_integrated` / `lfista_integrated_explore`, o mesmo conjunto base mais `11_gain_rmse_over_ml_only_torch.png` e `12_gain_mae_over_ml_only_torch.png` (ganhos face a `ml_only_torch`), quando esses metodos estao no `summary`.

Laboratorio standalone Etapa 2 (sem orquestrador): `python sir_cs_lfista.py` (perfis `phase2_lfista`, `explore`); nucleo reutilizavel em `lfista_module.py`.

---

## Etapa 2 — Se passar no mínimo viável

Rodar ablações:

1. variar `residual_basis`: `"identity"` e `"dct"`;
2. variar `residual_k`;
3. variar `measurement_noise_std`;
4. variar `measurement_kind`: `"gaussian"` e `"subsample"`;
5. variar `residual_mode`: `"support_from_u"` e `"random"`.

### Perguntas a responder
- O ganho depende de a inovação ser previsível?
- O ganho cai quando o residual deixa de ser esparso?
- O ganho aumenta quando `m/N` diminui?
- O weighted-CS só ajuda quando o prior de suporte é realmente informativo?

---

## Etapa 3 — Tornar publicável

Se a etapa 2 for favorável, avançar para:

1. **estudo de ablação completo**
   - retirar prior de suporte;
   - trocar base;
   - retirar predictor de alpha;
   - comparar com um baseline low-rank/AE residual.

2. **análise estatística**
   - múltiplas seeds;
   - IC95%;
   - teste pareado entre métodos.

3. **caso aplicado**
   - campo espaço-temporal;
   - sensor network;
   - anomalia localizada;
   - monitoring/time-lapse.

4. **teoria honesta**
   - enunciar cota de erro condicional:
     - RIP/boa geometria de `M Psi`;
     - residual compressível;
     - baseline removendo a parte densa.

---

## Etapa 4 — Linha de paper

Estrutura sugerida:

1. Introdução
2. Método SIR-CS
3. Garantia teórica condicional
4. Estudos sintéticos controlados
5. Ablações
6. Caso aplicado
7. Discussão: quando funciona / quando falha

**Rascunho LaTeX (metodo + experimentos sinteticos):** compile `paper/main.tex`. O draft atual usa **Phase 0**: figuras em `paper/figures/phase0_baseline/` (com fallback em `../outputs/paper/figures/phase0_baseline/` no `\graphicspath`). Correr `python sir_cs_pipeline_optimized.py --profile phase0_baseline` repoe os PNGs no destino do pipeline; copie para `paper/figures/phase0_baseline/` se necessario. Texto e tabela em `paper/experiments_synthetic.tex`.

---

## Dica prática

Comece com:
- `residual_basis="identity"`
- `measurement_kind="gaussian"`
- `residual_mode="support_from_u"`

Esse é o cenário mais favorável para detectar a existência do efeito.
Depois você endurece o problema gradualmente.
