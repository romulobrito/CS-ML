# Etapa 2f -- Teste A/B DEM multiescala vs monoescala calibrado (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO, carbonatos) |
| Dependencia | Etapa 2e concluida (`run_861_dem_sc_lab_calibration.py`) |
| Objetivo | Evidencia empirica para **nao** adotar DEM multiescala no pipeline atual |
| Escopo | 10 plugs microCT; **nao** altera perfil 87 linhas |
| Referencia externa | `scripts/refs_scripts_dem_multiscale/` (Equinor / Berryman sequencial) |
| Codigo | `dem_sc_861_multiscale.py`, `run_861_dem_sc_multiscale_ab.py` |
| Saidas | `dem_sc_runs/multiscale_ab/` |

**Relacao com Etapa 2 principal:** ver `etapa2_dem_sc_vpvs_poco861.md` (Fase 2e = calibracao
monoescala; esta sub-etapa 2f = experimento aditivo de validacao).

---

## 1. Motivacao

A referencia Equinor (`aspect_ratio_Image_dem_multiscale.py`) modela porosidade
macro-meso e micro como **inclusoes DEM sequenciais**, invertendo o aspect ratio (AR)
dos microporos por busca em grade.

No pipeline 861 atual:

- O perfil usa **um alpha efetivo por HFU** (calibrado vs lab).
- Fracoes `phi_meso_macropores_vv` e `phi_micropores_vv` existem no CT mas **nao**
  entram no forward model do perfil.

Antes de estender o perfil, este experimento responde:

> Multiescala sequencial (sem nova regressao global) melhora Vp nos 10 plugs
> em relacao ao monoescala calibrado (M1)?

Se **nao**, documentamos a limitacao e mantemos o perfil monoescala.

---

## 2. Modelos comparados

| ID | Nome | Parametros livres | Regressao | Papel |
|----|------|-------------------|-----------|-------|
| **M1** | Monoescala calibrado | alpha + escala K/G por HFU | Sim (ja rodado) | Baseline perfil |
| **M2a oracle** | Multiescala por plug | AR_micro (grid 1D **ao Vp_lab**) | Nao | Teto diagnostico |
| **M2b oracle** | HFU medianas + grid plug | AR_micro por plug ao Vp_lab | Nao | Diagnostico |
| **M2b forward** | HFU medianas + AR_micro HFU | Nenhum no plug avaliado | Nao | **Metrica principal** |
| M2c | Calibrado multiescala HFU | Varios | Sim | Fora de escopo |

### 2.1 Distincao oracle vs forward (critica)

**Oracle:** `invert_ar_micro_grid` minimiza `|Vp_pred - Vp_lab|` no **mesmo plug**.
Isso nao e previsao; e ajuste pos-dado. MAPE oracle ~3% e esperado e **nao**
justifica trocar o perfil.

**Forward (M2b forward):** fracoes e AR_meso = medianas CT por HFU; AR_micro =
mediana dos fits oracle nos plugs de treino do HFU; Vp do plug avaliado e
**somente forward**, sem inverter AR_micro no plug alvo.

**LOO forward:** recomputar medianas HFU e AR_micro sem o plug held-out; prever
held-out em forward. Comparar MAPE LOO com M1 LOO (~15%).

### 2.1 M1 (baseline, somente leitura)

- Fonte: `dem_sc_runs/lab_calibration/tables/plug_validation_calibrated.csv`
- LOO: `dem_sc_runs/lab_calibration/tables/loo_validation.csv`

### 2.2 M2a (teto de ganho local)

Por plug, usando **somente** dados daquele plug:

```
phi_total     = Phi_lab (pu)
f_meso        = phi_meso_macropores_vv   (fracao dentro do espaco poroso)
f_micro       = phi_micropores_vv
AR_meso       = ar_meso_macropores       (fixo, CT)
AR_micro      = grid [0.01, 1.00], passo 0.001
K_scale, G_scale = herdados do M1 (HFU), fixos
```

Forward: VRH matriz -> DEM sequencial (ordem decrescente de f * phi) -> Vp seco.

**Nota LOO:** M2a e **independente por plug**; in-sample = LOO (nao ha treino).

### 2.3 M2b (proxy de extrapolacao por HFU)

Por plug, fracoes e AR_meso = **medianas CT dos plugs do mesmo HFU**:

```
f_meso, f_micro, AR_meso  <- mediana(HFU)
AR_micro                  <- grid 1D (mesmo criterio M2a)
K_scale, G_scale          <- M1 (fixos)
```

**LOO M2b:** recomputar medianas HFU excluindo o plug held-out; prever held-out.

### 2.4 M2c (nao implementado)

Calibracao conjunta por HFU (espelharia `dem_sc_861_calibrate.py`).
Nao entra neste experimento para separar **fisica multiescala** de **overfitting
por regressao** com n = 2--4 plugs por HFU.

---

## 3. Convencoes de dados

### 3.1 Fracoes CT

Colunas em `861_ct_samples.csv` / `861_integrated_ct_samples.xlsx`:

| Coluna | Significado |
|--------|-------------|
| `phi_meso_macropores_vv` | Fracao do **espaco poroso** atribuida a meso-macroporos |
| `phi_micropores_vv` | Fracao do espaco poroso atribuida a microporos |
| `ar_meso_macropores` | AR medio meso-macro (fixo em M2) |

Validacao: `abs(f_meso + f_micro - 1) < 1e-3` em cada plug.

Incrementos DEM:

```
phi_inc_meso  = f_meso  * phi_lab
phi_inc_micro = f_micro * phi_lab
```

### 3.2 Ordem DEM sequencial

Igual referencia Equinor: inclusoes ordenadas por `phi_inc` **decrescente**.

### 3.3 Matriz e escalas

- VRH calcite/dolomita: `matrix_from_solids(corrected_solid1_pct, corrected_solid2_pct)`
- `matrix_k_scale`, `matrix_g_scale`: lidos de `hfu_lab_calibrated.csv` (M1), **nao refitados**

---

## 4. Criterio de decisao (pre-registrado)

```text
GAIN_THRESHOLD_MAPE_PP = 2.0

Manter monoescala se **qualquer** condicao abaixo for verdadeira:
  (a) delta_MAPE_in_sample(M2b_forward vs M1) < 2.0 p.p.
  (b) delta_MAPE_LOO < 2.0 p.p.
  (c) LOO melhora mas in-sample nao (ganho marginal, nao sustenta perfil)

Investigar multiescala somente se in-sample **e** LOO ganharem >= 2.0 p.p.
```

---

## 5. Artefatos de codigo

| Arquivo | Papel |
|---------|-------|
| `scripts/dem_sc_861_multiscale.py` | Forward sequencial, grid AR_micro, records |
| `scripts/run_861_dem_sc_multiscale_ab.py` | CLI, tabelas, figuras, metrics.json |
| `scripts/ml_861_data.py` | `DEM_SC_MULTISCALE_AB_ROOT` |

### 5.1 Execucao

```bash
cd methods_comparison/scripts
python run_861_dem_sc_multiscale_ab.py
python run_861_dem_sc_multiscale_ab.py --robust-exclude   # exclui F2911V
```

### 5.2 Saidas

```
dem_sc_runs/multiscale_ab/
  MANIFEST.txt
  metrics.json
  tables/
    plug_comparison.csv
    summary_metrics.csv
    loo_m2b_forward_comparison.csv
    loo_m2b_oracle_diagnostic.csv
  figures/
    vp_crossplot_m1_vs_m2b_forward.png
    abs_error_m1_m2a_oracle_m2b_forward.png
```

---

## 6. Validacao obrigatoria

| Teste | Criterio |
|-------|----------|
| Soma fracoes | 10/10 plugs com f_meso + f_micro ~= 1 |
| Caso degenerado | f_micro < 0.05: Vp pouco sensivel a AR_micro |
| Reprodutibilidade | `metrics.json` com `generated_utc` |
| Cross-check manual (opcional) | 1 plug vs `EffectiveMedium.dem1` da referencia |

---

## 7. Uso no paper / Beamer

Apos `metrics.json`:

- Apendice CLP-CSGM ou Etapa 2 LaTeX: paragrafo de limitacao + tabela resumo.
- Slide: MAPE M1 vs M2a vs M2b; nota "sem nova regressao".

Texto modelo:

> O teste multiescala foi aditivo ao pipeline monoescala calibrado: fracoes e AR
> meso-macro fixos pelo micro-CT, escala de matriz herdada da calibracao HFU
> existente, e AR de microporos por busca em grade plug a plug, sem regressao
> global. Nao houve ganho estavel acima de 2 pontos percentuais de MAPE em Vp
> (Tabela X); o perfil de 87 linhas permanece no modelo monoescala com alpha
> efetivo por HFU.

---

## 8. Fora de escopo

- Alterar `run_861_dem_sc_profile_87.py`
- Gassmann / saturacao no multiescala
- SC no multiescala
- M2c (calibracao multiescala por HFU)
- Poco 1045

---

## 9. Ordem de execucao (atualizada)

```
...
6. Calibracao inversa vs lab + perfil recalibrado   (Fase 2e) [done]
7. Teste A/B multiescala vs monoescala             (Fase 2f) [este plano]
8. Gassmann/PVT + LaTeX Etapa 2
9. Etapa 3: ML de residuo
```

---

## 10. Referencias

- Plano Etapa 2: `etapa2_dem_sc_vpvs_poco861.md`
- Referencia multiescala: `scripts/refs_scripts_dem_multiscale/`
- Calibracao M1: `scripts/dem_sc_861_calibrate.py`
- Nucleo DEM: `scripts/dem_sc_861_core.py`
