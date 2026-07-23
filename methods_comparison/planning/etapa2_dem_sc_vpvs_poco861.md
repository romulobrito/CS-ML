# Etapa 2 -- DEM/SC + Vp/Vs calibrado por HFU (Poco 861)

## Status

| Item | Valor |
|------|-------|
| Poco | 861 (MOGNO, carbonatos) |
| Intervalo | 5205,91 -- 5233,72 m |
| Etapa anterior | 1d concluida (ML baseline; Phi viavel; FZI descartado como alvo de perfil) |
| Esta etapa | Modelagem fisica de meio efetivo (DEM + SC) -> Vp, Vs, Vp/Vs por HFU |
| Fora de escopo | Etapa 3 (ML hibrido sobre residuos), poco 1045, tuning ML pesado |
| Codigo existente | Nenhum modulo DEM/SC no repositorio |
| Biblioteca alvo | `rockphypy` (RockPhysicsPy; referida nos docs anteriores como Pyrockphys) |

**Dependencia critica:** resultados e decisoes da Etapa 1d (`etapa1d_well_profile_ct_plugs_poco861.md`, relatorio LaTeX).

---

## 1. Objetivo

Estimar **velocidades sismicas teoricas** ($V_p$, $V_s$ e razao $V_p/V_s$) ao longo do
intervalo perfurado do poco 861, usando:

1. **Porosidade** ao longo do poco (perfil ou laboratorio).
2. **Geometria de poros** (aspect ratio, fracao de solidos) calibrada nos **10 plugs microCT**.
3. **Segmentacao por HFU** (prioridade: HFU de laboratorio).
4. Modelos **DEM** (Differential Effective Medium, Berryman) e **SC** (Self-Consistent)
   para modulos elasticos a seco, com possivel saturacao por Gassmann na fase 2b.

Esta etapa **nao** repete ML direto perfil -> FZI ou perfil -> Vp/Vs.
O ML da Etapa 1 serve apenas para **insumos** (principalmente porosidade e, com ressalvas, HFU).

---

## 2. O que a Etapa 1 entregou (insumos fixos)

### 2.1 Decisoes que amarram a Etapa 2

| Decisao Etapa 1 | Implicacao para Etapa 2 |
|-----------------|-------------------------|
| Phi_lab previsivel com RF (R2 ~ +0,15) ou Phi_ND direto | Usar **Phi_ND** como default conservador; RF de phi como alternativa documentada |
| FZI **nao** previsivel so com perfis | **Nao** usar FZI predito como driver de Vp/Vs |
| HFU logistic OOF ~36% | Usar **HFU de laboratorio** nas 87 linhas; HFU predito so onde faltar lab |
| microCT nao melhorou RF de perfil | CT entra na **fisica** (AR, solidos), nao como feature extra de ML |
| Validacao por blocos de profundidade | Manter logica espacial ao extrapolar parametros por HFU |

### 2.2 Conjuntos de dados

| Conjunto | Amostras | Papel na Etapa 2 |
|----------|----------|------------------|
| Perfil enriquecido (87 linhas) | 87 | Phi de perfil, HFU lab, profundidade, litotipo |
| Plugs integrados com microCT | 10 | Calibrar AR e modulos de matriz por HFU |
| Sobreposicao CT no perfil | 8 profundidades | Validacao cruzada local (plug vs linha de perfil) |

**Lacuna elastica (atualizada 2026-06-15):** o dataset integrado Etapa 1 ainda nao
tinha $V_{p,\mathrm{lab}}$ nem $V_{s,\mathrm{lab}}$. A planilha
`ROCKPHYS_Database_04_12_2024 (7).xlsx` (aba **Velocity**, poco **3-BRSA-861-SPS**)
fornece **28 amostras** e **196 medicoes** (7 pressoes de confinamento, 5,5--22,1 MPa).
Ingestao: `processed/rockphys_861/`. Validacao DEM vs lab: `dem_sc_runs/lab_validation/`.

Ainda **nao** ha DTCO/DTSM de perfil digital nem velocidades saturadas preenchidas
na ROCKPHYS para os plugs CT.

---

## 3. Variaveis de entrada (por escala)

### 3.1 Ao longo do poco (87 linhas)

| Variavel | Fonte recomendada | Unidade | Confianca |
|----------|-------------------|---------|-----------|
| Profundidade | Perfil integrado | m | Alta |
| $\phi$ (porosidade total) | **Phi_ND (pu)** ou Phi_lab onde existir | v/v | Moderada (perfil) / Alta (lab) |
| HFU | Coluna HFU de laboratorio | classe 1--4 | Alta |
| Phi_Sonic | Perfil | v/v | Alta como **referencia de porosidade acustica**, nao como alvo Vp |

Distribuicao HFU no perfil: HFU1=42, HFU2=29, HFU3=10, HFU4=6.

### 3.2 Nos 10 plugs microCT (calibracao)

| Variavel | Coluna integrada | Unidade | Faixa observada (10 plugs) |
|----------|------------------|---------|----------------------------|
| Aspect ratio medio | `ct_ar_mean` | adimensional (0--1) | 0,54 -- 0,68 |
| Porosidade imagem CT | `ct_porosity_pct` | % | 0,006 -- 13,7 (subamostra 2,5 mm; **nao** usar como phi de entrada DEM) |
| Fracao meso-macroporos | `phi_meso_macropores_vv` | v/v | 0,47 -- 0,97 |
| Solido 1 (corrigido) | `corrected_solid1_pct` | % | 13,6 -- 95,3 |
| Solido 2 (corrigido) | `corrected_solid2_pct` | % | 2,3 -- 50,5 |
| Phi de laboratorio (plug) | `Phi_lab (pu)` | v/v | 0,075 -- 0,181 |
| HFU (plug) | `HFU` | classe | 1, 2, 3 (HFU4 **ausente** nos plugs) |

**Regra importante:** a porosidade de entrada do DEM/SC em cada plug e **Phi_lab (pu)**,
nao `ct_porosity_pct` (escala local da imagem, muito menor que lab na maioria dos plugs).

### 3.3 Parametros de matriz e fluido (a fixar / calibrar)

| Parametro | Simbolo | Unidade | Origem proposta |
|-----------|---------|---------|-----------------|
| Modulo de bulk da matriz | $K_m$ | GPa | Mistura VRH calcite/dolomita a partir de `corrected_solid*_pct` ou literatura MOGNO |
| Modulo de cisalhamento da matriz | $G_m$ | GPa | Idem |
| Modulo de bulk do fluido | $K_f$ | GPa | Agua formacao (default ~2,2) ou salmoura; fase 2b |
| Densidade da matriz | $\rho_m$ | g/cm3 | Mistura mineralogica |
| Densidade do fluido | $\rho_f$ | g/cm3 | ~1,0--1,1 |
| Aspect ratio dos poros | $\alpha$ | adimensional | Mediana de `ct_ar_mean` **por HFU** (fallback: mediana global) |

**Valores iniciais de referencia (calcite, ate confirmar com litologia MOGNO):**

- $K_{\mathrm{calcite}} \approx 77$ GPa, $G_{\mathrm{calcite}} \approx 32$ GPa
- $K_{\mathrm{dolomite}} \approx 95$ GPa, $G_{\mathrm{dolomite}} \approx 45$ GPa

---

## 4. Equacoes e cadeia de calculo

### 4.1 Visao geral do fluxo

```
[87 linhas]  phi(z), HFU(z)  ──────────────────────────────┐
                                                            │
[10 plugs]   phi_lab, ct_ar_mean, solidos, HFU  ──>  Calibracao por HFU
                              │                              │
                              v                              v
                    K_m, G_m (por HFU)              alpha (por HFU)
                              │                              │
                              └──────────┬───────────────────┘
                                         v
                              DEM(phi, alpha, K_m, G_m) -> K_dry, G_dry
                              SC(phi, alpha, K_m, G_m)  -> K_dry, G_dry  (comparacao)
                                         v
                              Gassmann(K_dry, G_dry, K_f, phi) -> K_sat, G_sat  [fase 2b]
                                         v
                              rho_eff(phi, rho_m, rho_f)
                                         v
                              Vp = sqrt((K_sat + 4/3 G_sat) / rho_eff)
                              Vs = sqrt(G_sat / rho_eff)
                              VpVs = Vp / Vs
```

### 4.2 DEM (Berryman, via `rockphypy.EM.Berryman_DEM`)

O modelo DEM trata a rocha como matriz mineral com inclusoes de poros adicionadas
incrementalmente ate a porosidade alvo $\phi$.

**Entradas:**

- $K_m$, $G_m$: modulos da fase matriz (mineral)
- $K_i = 0$, $G_i = 0$: inclusoes de poro **a seco** (dry)
- $\alpha$: aspect ratio dos poros (do microCT)
- $\phi$: porosidade total

**Saida:** $K_{\mathrm{dry}}^{\mathrm{DEM}}$, $G_{\mathrm{dry}}^{\mathrm{DEM}}$

Equacoes diferenciais acopladas (Berryman, 1992):

$$
(1-y)\frac{dK_{\mathrm{eff}}^{\mathrm{DEM}}}{dy}
  = (K_2 - K_{\mathrm{eff}}^{\mathrm{DEM}})\, P^{*2}(y)
$$

$$
(1-y)\frac{dG_{\mathrm{eff}}^{\mathrm{DEM}}}{dy}
  = (G_2 - G_{\mathrm{eff}}^{\mathrm{DEM}})\, Q^{*2}(y)
$$

com condicoes iniciais $K_{\mathrm{eff}}^{\mathrm{DEM}}(0)=K_m$,
$G_{\mathrm{eff}}^{\mathrm{DEM}}(0)=G_m$, e $y=\phi$ ao final da integracao.
As funcoes $P^{*2}$ e $Q^{*2}$ dependem de $\alpha$ e dos modulos da inclusao e da matriz
(detalhes em Berryman, 1992; implementacao em `rockphypy`).

### 4.3 Self-Consistent (SC, via `rockphypy.EM`)

O modelo SC impoe consistencia entre fase matriz e inclusoes em todas as concentracoes.
Usar **em paralelo ao DEM** nos 10 plugs para:

- verificar sensibilidade ao esquema de homogeneizacao;
- escolher DEM como modelo principal se SC divergir muito sem justificativa geologica.

**Entradas:** mesmas de 4.2 ($K_m$, $G_m$, $\alpha$, $\phi$; poros a seco com $K_i=G_i=0$).

**Saida:** $K_{\mathrm{dry}}^{\mathrm{SC}}$, $G_{\mathrm{dry}}^{\mathrm{SC}}$.

### 4.4 Saturacao (Gassmann, fase 2b)

Para rocha saturada com fluido:

$$
\frac{K_{\mathrm{sat}} - K_{\mathrm{dry}}}
     {K_{\mathrm{dry}} - K_0}
  = \frac{K_f \,(K_0 - K_{\mathrm{dry}})}
         {K_0\,(K_f + \frac{4}{3}G_{\mathrm{dry}})\,\phi}
$$

onde $K_0$ e o modulo de bulk do grao mineral (aproximacao: $K_m$).
$G_{\mathrm{sat}} \approx G_{\mathrm{dry}}$ (Gassmann nao altera modulo de cisalhamento).

Na **prova de conceito (2a)** trabalhar com rocha **a seco** ($K_{\mathrm{sat}}=K_{\mathrm{dry}}$)
para reduzir graus de liberdade; introduzir fluido na fase 2b.

### 4.5 Densidade efetiva e velocidades

$$
\rho_{\mathrm{eff}} = (1-\phi)\,\rho_m + \phi\,\rho_f
$$

(com $\rho_f=0$ ou omitido na fase a seco).

$$
V_p = \sqrt{\frac{K_{\mathrm{sat}} + \frac{4}{3}G_{\mathrm{sat}}}{\rho_{\mathrm{eff}}}}
\qquad
V_s = \sqrt{\frac{G_{\mathrm{sat}}}{\rho_{\mathrm{eff}}}}
\qquad
\frac{V_p}{V_s} = \frac{V_p}{V_s}
$$

Unidades SI internas: modulos em Pa, densidade em kg/m3, velocidades em m/s.
Converter para us/ft ou km/s apenas na saida, se necessario para comparacao com perfil futuro.

---

## 5. Calibracao por HFU (10 plugs -> parametros por unidade)

### 5.1 Estatisticas CT por HFU (dados atuais)

| HFU | n plugs | $\overline{\phi_{\mathrm{lab}}}$ | $\overline{\alpha}$ (`ct_ar_mean`) | Nota |
|-----|---------|----------------------------------|-------------------------------------|------|
| 1 | 4 | 0,116 | 0,56 | AR relativamente alto |
| 2 | 4 | 0,130 | 0,55 | Maior variancia de phi |
| 3 | 2 | 0,110 | 0,61 | Apenas 2 plugs (F2829V, F2830H) |
| 4 | 0 | -- | -- | **Sem plug CT**; usar fallback |

Medias de `corrected_solid1_pct` / `corrected_solid2_pct` por HFU alimentam a mistura
mineralogica ($K_m$, $G_m$) via media de Reuss-Voigt ou VRH (`rockphypy.EM.VRH`).

### 5.2 Regra de extrapolacao para as 87 linhas

Para cada profundidade $z$:

1. Ler HFU($z$) de laboratorio.
2. Atribuir $\alpha_{\mathrm{HFU}}$ = mediana de `ct_ar_mean` nos plugs da mesma HFU.
3. Atribuir $K_m$, $G_m$, $\rho_m$ da tabela calibrada por HFU.
4. Atribuir $\phi(z)$ = Phi_ND($z$) [default] ou Phi_lab($z$) se disponivel.
5. Calcular DEM/SC -> $V_p(z)$, $V_s(z)$, $V_p/V_s(z)$.

**Fallback HFU4 (sem plug):**

- $\alpha$: mediana global dos 10 plugs, ou mediana de HFU3 (mais proxima em phi medio).
- $K_m$, $G_m$: media ponderada de HFU2 e HFU3, ou valor de HFU2.
- Documentar incerteza elevada para HFU4 ($n=6$ no perfil, 0 plugs CT).

**Fallback HFU predito (apenas se HFU lab ausente em alguma linha):**

- Nao esperado nas 87 linhas atuais (HFU completo no dataset).
- Se surgir em extrapolacao futura: usar classe logistic com flag de baixa confianca.

### 5.3 Porosidade ao longo do poco: qual usar?

| Opcao | Vantagem | Desvantagem |
|-------|----------|-------------|
| **Phi_ND (pu)** | Sem erro de ML; correlacao r~0,70 com Phi_lab | Ignora ajuste local RF |
| Phi_lab (pu) | Medida direta onde ha testemunho | Esparsa; nao cobre todo o intervalo de predicao |
| RF phi (Etapa 1d) | Estimativa em toda a coluna | R2~0,15; bloco 2 negativo |

**Recomendacao Etapa 2a:** rodar cenarios **Phi_ND** (principal) e **Phi_lab nos 10 plugs**
(apenas POC). Para as 87 linhas: **Phi_ND** como linha de base; RF como cenario sensibilidade.

---

## 6. Fases de implementacao

### Fase 2a -- Prova de conceito nos 10 plugs (prioridade imediata apos este plano)

**Objetivo:** demonstrar que DEM/SC + `rockphypy` roda com dados reais do 861.

**Entradas por plug:** Phi_lab, ct_ar_mean, corrected_solid1/2_pct, HFU.

**Passos:**

1. Instalar `rockphypy` no ambiente do projeto.
2. Calcular $K_m$, $G_m$ por mistura mineralogica (solidos corrigidos).
3. Rodar `Berryman_DEM` e SC para cada plug (poro seco).
4. Calcular $V_p$, $V_s$, $V_p/V_s$ a seco.
5. Salvar tabela de saida + graficos predito vs observado (nao ha Vp observado; graficos
   de consistencia: DEM vs SC, sensibilidade a $\alpha$).

**Criterio de aceite 2a:**

- 10/10 plugs processados sem erro numerico.
- $V_p$ e $V_p/V_s$ monotonicos com $\phi$ dentro de cada HFU (tendencia fisica esperada).
- Diferenca relativa DEM vs SC documentada (espera-se concordancia moderada, nao identidade).
- Artefatos em `methods_comparison/data/processed/dem_sc_runs/poc_10plugs/`.

### Fase 2b -- Saturacao e fluido

- Adicionar Gassmann com $K_f$ de agua/salmoura.
- Comparar Vp a seco vs saturado nos plugs.
- Decidir fluido de formacao com petrofisica do campo (se disponivel).

### Fase 2c -- Extrapolacao para 87 linhas

- Aplicar parametros por HFU a todo o perfil enriquecido.
- Produzir colunas: `Vp_dem`, `Vs_dem`, `VpVs_dem`, `Vp_sc`, `Vs_sc`, `VpVs_sc`,
  `phi_input`, `alpha_hfu`, `hfu_source`.
- Exportar `861_dem_sc_profile.xlsx` (87 linhas).

### Fase 2d -- Validacao e diagnostico

Ver Secao 7.

### Fase 2e -- Documentacao e LaTeX

- Atualizar relatorio cientifico com resultados Etapa 2.
- Registrar incertezas e fallbacks HFU4.

---

## 7. Validacao (incluindo ``vs sonic'')

### 7.1 O que **nao** temos (ou ainda nao)

- Curva de perfil de slowness (DTCO, DTSM) ou $V_p/V_s$ log digital.
- Velocidades **saturadas** preenchidas na aba Velocity da ROCKPHYS (colunas sat vazias).
- Portanto: validacao de perfil continua **indireta** via Phi_Sonic; validacao **direta**
  plug-a-plug usa ROCKPHYS lab (Secao 7.5).

### 7.2 O que **temos** -- validacao indireta com Phi_Sonic

`Phi_Sonic (pu)` e porosidade estimada a partir do perfil acustico (correlacao com
Phi_lab: r ~ 0,67). Serve para:

| Teste | Metodo | Criterio de sucesso (preliminar) |
|-------|--------|----------------------------------|
| Consistencia de porosidade | Comparar $\phi_{\mathrm{input}}$ (Phi_ND) vs Phi_Sonic ao longo do poco | Correlacao mantida (r > 0,6); sem divergencia sistematica por HFU |
| Tendencia fisica | $V_p$ teorico vs $\phi$ e vs HFU | $V_p$ decresce com $\phi$ dentro de cada HFU; HFU1 (mais tight) tende a Vp maior que HFU3 |
| Coerencia local (8 profundidades com CT) | Comparar Vp/Vs do plug com vizinho no perfil | Ordem de magnitude consistente; flag se delta > limiar |
| DEM vs SC | Diferenca relativa em Vp nos 10 plugs | < 15--20% se alpha similar; senao investigar |

### 7.3 Validacao futura (quando houver dado)

- Ingerir DTCO/DTSM do poco 861 (pasta Perfis_Digitais).
- Calibracao inversa de $\alpha$ ou $K_m$, $G_m$ por HFU para minimizar RMSE vs lab.
- Gassmann com PVT quando velocidades saturadas estiverem disponiveis.

### 7.4 Metricas de saida da validacao 2d (Phi_Sonic)

- Correlacao Pearson: $\phi_{\mathrm{input}}$ vs Phi_Sonic (global e por HFU).
- RMSE de porosidade: Phi_ND vs Phi_lab (onde lab existe).
- Faixa de $V_p$, $V_s$, $V_p/V_s$ por HFU (tabela resumo).
- Painel profundidade: HFU, $\phi$, $V_p/Vs$ teorico, Phi_Sonic.

### 7.5 Validacao direta DEM vs ROCKPHYS lab (2026-06-15)

**Fonte:** `ROCKPHYS_Database_04_12_2024 (7).xlsx`, aba Velocity, filtro 861.

| Item | Valor |
|------|-------|
| Amostras 861 na planilha | 28 |
| Plugs CT com lab | 10/10 (F2911V mapeado para F2911H) |
| Pressao de referencia | 22,1 MPa (maior passo de confinamento) |
| Eixo comparado | Z (transmissao a seco) |
| Scripts | `run_861_rockphys_ingest.py`, `run_861_dem_sc_lab_validation.py` |
| Artefatos | `rockphys_861/`, `dem_sc_runs/lab_validation/` |

**Resultados preliminares (DEM a seco vs lab a 22,1 MPa):**

| Metrica | Valor | Interpretacao |
|---------|-------|---------------|
| MAPE $V_p$ | ~26% | DEM superestima $V_p$ na maioria dos plugs |
| Bias $V_p$ | ~+1,2 km/s | Sistematico |
| $r$ Pearson ($V_p$) | ~0,21 | Fraco (F2911H e outlier; excluir para analise HFU) |
| MAE $V_p/V_s$ | ~0,15 | DEM $V_p/V_s$ ~1,80--1,88 vs lab ~1,64--1,72 |
| Melhor plug | F2870H | Erro ~1,5% em $V_p$ |
| Pior plug | F2911H (CT: F2911V) | Erro ~56% ($V_{p,\mathrm{lab},Z}$=3,82 km/s) |

**Leitura fisica:** o modelo a seco com $\alpha$ e matriz calibrados por microCT captura
a ordem de grandeza e a hierarquia entre plugs, mas **nao** reproduz $V_p$ lab sem
recalibracao (pressao efetiva, anisotropia, mineralogia fina). Proximo passo: ajuste
inverso de $\alpha$ ou $K_m$, $G_m$ por HFU; depois Gassmann (PVT).

**Alias de amostra:** CT usa F2911V; ROCKPHYS lista F2911H na mesma profundidade (5225,75 m).

### 7.6 Calibracao inversa por HFU (2026-06-15)

**Metodo:** para cada HFU, minimizar RMSE($V_{p,\mathrm{DEM}}$, $V_{p,\mathrm{lab}}$) ajustando
$\alpha$ e escala uniforme de $K_m$, $G_m$. Cenarios testados: `alpha_only` vs
`alpha_matrix_scale`; escolhe o de menor RMSE in-sample.

| Modo | Plugs | MAPE antes | MAPE depois | RMSE depois |
|------|-------|------------|-------------|-------------|
| Standard (10 plugs) | 10 | 26,0% | **9,1%** | 0,52 km/s |
| Robust (--robust, sem F2911V) | 9 | 22,7% | **7,5%** | 0,50 km/s |

**Parametros calibrados (standard):**

| HFU | $\alpha_{\mathrm{CT}}$ | $\alpha_{\mathrm{calib}}$ | escala $K,G$ | cenario |
|-----|------------------------|---------------------------|--------------|---------|
| 1 | 0,554 | 0,950 | 0,698 | alpha_matrix_scale |
| 2 | 0,560 | 0,191 | 0,706 | alpha_matrix_scale |
| 3 | 0,615 | 0,950 | 0,721 | alpha_matrix_scale |
| 4 | fallback | 0,570 | 1,000 | media HFU2+3 calibrados |

**Perfil 87 recalibrado:** `profile_87_lab_calib/` -- $V_p$ medio ~5,2 km/s (vs ~6,3 km/s
antes da calibracao). Scripts: `run_861_dem_sc_lab_calibration.py`,
`run_861_dem_sc_profile_87.py --lab-calib`.

### 7.7 Validacao LOO (leave-one-plug-out, 2026-06-15)

Cada plug e previsto com parametros HFU ajustados nos **outros** plugs (sem leakage).

| Modo | MAPE $V_p$ | RMSE | $r$ |
|------|------------|------|-----|
| Nao calibrado | 26,0% | 1,36 km/s | 0,21 |
| In-sample calibrado | 9,1% | 0,52 km/s | 0,57 |
| **LOO calibrado** | **14,8%** | **0,85 km/s** | 0,08 |

**Leitura:** LOO confirma que a calibracao generaliza melhor que o modelo CT puro
(14,8% vs 26%), mas o erro honesto e ~6 pontos percentuais acima do in-sample (9%).
Pior fold LOO: F2870H (~26% erro). Melhor LOO: F2910H (~3,7%).

Artefatos: `lab_calibration/tables/loo_validation.csv`, `metrics_loo.json`,
figuras `loo_error_by_sample.png`, `mape_insample_vs_loo.png`.

---

## 8. Produtos de saida (artefatos)

```
methods_comparison/data/processed/dem_sc_runs/
  poc_10plugs/
    MANIFEST.txt
    plug_dem_sc_summary.csv          # 10 linhas: plug, HFU, phi, alpha, K_dry, G_dry, Vp, Vs, VpVs
    dem_vs_sc_comparison.png
    sensitivity_alpha.png
  profile_87/
    861_dem_sc_profile.xlsx          # 87 linhas com Vp, Vs, VpVs por HFU
    validation_phi_sonic.csv
    figures/
      vpvs_vs_depth.png
      phi_nd_vs_phi_sonic.png
      vpvs_by_hfu.png
  hfu_calibration/
    hfu_matrix_moduli.csv            # K_m, G_m, rho_m, alpha por HFU
    hfu_ct_stats.csv                 # estatisticas dos 10 plugs por HFU
```

---

## 9. Script minimo (prova de conceito) -- escopo

Um unico modulo Python (ASCII, sem dependencia de paths Windows legados):

| Funcao | Descricao |
|--------|-----------|
| Carregar 10 plugs integrados | Reutilizar loader da Etapa 1 (`ml_861_data`) |
| `matrix_moduli_from_solids(s1, s2)` | VRH calcite/dolomita -> $K_m$, $G_m$ |
| `run_dem_sc(phi, alpha, Km, Gm)` | Wrapper `rockphypy.EM.Berryman_DEM` + SC |
| `velocities(K, G, rho)` | $V_p$, $V_s$, $V_p/V_s$ |
| `run_poc_10_plugs()` | Loop 10 plugs, CSV + figuras |
| `extrapolate_profile_87()` | Fase 2c (apos POC aprovado) |

**Dependencia nova:** `rockphypy` em `requirements.txt` (ou arquivo opcional `requirements-rockphys.txt`).

---

## 10. Riscos e mitigacoes

| Risco | Impacto | Mitigacao |
|-------|---------|-----------|
| HFU4 sem plug CT | Parametros AR e matriz incertos | Fallback documentado; intervalo de sensibilidade |
| `ct_porosity_pct` << Phi_lab | Confusao de escala | Usar **sempre** Phi_lab / Phi_ND no DEM; CT so para AR e solidos |
| Apenas 2 plugs em HFU3 | Mediana AR instavel | Reportar desvio; ponderar com HFU2 na discussao |
| Sem Vp observado | Validacao elastica fraca | **Resolvido parcialmente:** ROCKPHYS lab; falta DTCO perfil |
| Solidos CT sem mapa mineralogico confirmado | $K_m$, $G_m$ errados | Validar com litologia MOGNO; sensibilidade +/- 10% em $K_m$ |
| Phi_ND com erro em bloco permeavel (5225--5234 m) | Vp/Vs distorcido no pe | Flag de qualidade por bloco de profundidade (herdar Etapa 1) |

---

## 11. Criterios de aceite (Etapa 2 completa)

- [x] POC 10 plugs: DEM + SC + Vp/Vs a seco (`dem_sc_runs/poc_10plugs/`, 2026-06-15)
- [x] Tabela de calibracao por HFU ($\alpha$, $K_m$, $G_m$) em `hfu_calibration/hfu_ct_stats.csv`
- [ ] `rockphypy` instalado (opcional; POC usa `dem_sc_861_core.py` alinhado a Berryman/SC)
- [x] Perfil 87 linhas com $V_p$, $V_s$, $V_p/V_s$ teoricos (`profile_87/`, 2026-06-15)
- [x] Validacao indireta vs Phi_Sonic (`validation_phi_sonic.csv`, r~0,94 global)
- [x] ROCKPHYS lab ingerido (`processed/rockphys_861/`, 28 amostras, 2026-06-15)
- [x] Validacao direta DEM vs lab (`lab_validation/`, MAPE Vp~23%, 2026-06-15)
- [x] Calibracao inversa alpha/Km por HFU vs lab (`lab_calibration/`, MAPE 26%->9%, 2026-06-15)
- [x] Perfil 87 com params calibrados (`profile_87_lab_calib/`, Vp medio ~5.2 km/s)
- [ ] Incerteza HFU4 e gap sistematico DEM-lab explicitados em relatorio LaTeX
- [ ] Relatorio / LaTeX atualizado (opcional nesta entrega)

### Resultados POC 2a (producao)

| Metrica | Valor |
|---------|-------|
| Plugs processados | 10/10 |
| Vp DEM medio | 6,26 km/s |
| Vp/Vs DEM medio | 1,84 |
| Diferenca relativa media Vp (DEM vs SC) | 0,37% |
| Artefatos | `dem_sc_runs/poc_10plugs/tables/`, `figures/` |

---

## 12. Ordem de execucao recomendada

```
1. Este plano (etapa2_dem_sc_vpvs_poco861.md)
2. POC 10 plugs DEM/SC                              (Fase 2a) [done]
3. Extrapolar 87 linhas por HFU                     (Fase 2c) [done]
4. Validacao Phi_Sonic                              (Fase 2d) [done]
5. Ingerir ROCKPHYS + validacao DEM vs lab          (Fase 2d+) [done 2026-06-15]
6. Calibracao inversa vs lab + perfil recalibrado   (Fase 2e) [done 2026-06-15]
7. Teste A/B multiescala vs monoescala (10 plugs)   (Fase 2f) [etapa2f_dem_multiscale_ab_poco861.md]
8. Gassmann/PVT + LaTeX Etapa 2
9. Etapa 3: ML de residuo (somente apos calibrar fisica)
```

---

## 13. Referencias

- Etapa 1d (decisoes ML): `etapa1d_well_profile_ct_plugs_poco861.md`
- Integracao dados / colunas CT: `etapa1_dataset_ml_poco861.md`
- Dicionario ML: `data/processed/861_ML_DATASET_README.md`
- Relatorio LaTeX Etapa 1: `methods_comparison/latex/poco861_etapa1_ml.tex`
- Berryman (1992) -- DEM theory; implementacao: [rockphypy documentation](https://rockphypy.readthedocs.io/)
- Contexto MOGNO / CT: documentacao de microtomografia do intervalo (quando disponivel no projeto)

---

## 14. Resumo executivo (uma pagina)

A Etapa 2 troca a pergunta ``o perfil prevê FZI?'' (respondida: **nao**) pela pergunta
``dado phi, HFU e geometria de poros do microCT, qual Vp/Vs teorico a fisica de rochas
preve?''.

**Entrada principal ao longo do poco:** Phi_ND + HFU de laboratorio.

**Calibracao multiescala:** 10 plugs microCT definem aspect ratio e modulos de matriz **por HFU**.

**Modelo:** DEM (Berryman) como linha principal; SC para comparacao.

**Validacao atual:** indireta via Phi_Sonic; **direta** via ROCKPHYS lab (10 plugs,
MAPE $V_p$~23% com DEM a seco nao calibrado); comparacao DEM/SC interna.

**Proximo entregavel:** calibracao inversa por HFU; Gassmann com PVT; LaTeX Etapa 2.
