# Próximos passos (visão macro): SIR-CS

**Em uma frase:** o próximo passo é **robustez e ablação do `hybrid` básico no sintético**, não a versão final estruturada com TV e $\mathcal{P}$ ainda.

---

## Por que não ir direto para TV e $\mathcal{P}$

O próximo passo **agora** não é a versão mais forte com TV e $\mathcal{P}$.

O passo certo é fazer uma **rodada de robustez e ablação no sintético**.

**Motivo:** o primeiro marco já mostrou o que precisava: o `hybrid` vence o `ml_only` em todas as razões de medição, o ganho cresce com $\rho$, e `cs_only` é um referencial mal especificado e pior, alinhado à metodologia. Ao mesmo tempo, o experimento atual está num **regime favorável**: $\Psi = I$, $M$ gaussiana e inovação esparsa no mesmo domínio da recuperação.

---

## Passo 1 — Consolidar a história do `hybrid` básico

Estudo curto de sensibilidade variando:

- esparsidade ($k$);
- nível de ruído;
- amplitude da inovação;
- base $\Psi$ (`identity` vs `dct`);
- tipo de suporte (`support_from_u` vs `random`);
- mais seeds.

**Objetivo central:** responder se *"o método funciona só no cenário desenhado para ele, ou numa faixa razoável de regimes?"*

Este passo está alinhado ao papel da seção sintética como *sanity check* e referência de ablação no draft.

---

## Passo 2 — Diagnosticar o `weighted_hybrid`

Não promover o método ainda; primeiro medir se o preditor auxiliar de $\alpha$ tem qualidade.

Métricas sugeridas para o prior:

- correlação entre $\hat{\alpha}_{\mathrm{pred}}$ e $\alpha^\star$;
- precision/recall de suporte;
- overlap top-$k$.

Sem isso, não dá para separar se o `weighted_hybrid` fica atrás porque:

- a ideia é ruim,
- os pesos estão ruins,
- ou o preditor de $\alpha$ ainda não carrega informação suficiente.

Isso é importante porque, nos resultados atuais, o `weighted_hybrid` ainda fica atrás do `hybrid` básico.

---

## Passo 3 — Semi-realista antes do real

Após a robustez, um cenário mais próximo do uso real:

- $\Psi$ = DCT ou wavelet;
- saída com estrutura espacial mais clara;
- suporte menos "limpo";
- talvez $M$ por máscara/subsampling e mais ruído.

Assim se reduz o regime "favorável por construção", coerente com as limitações já apontadas no draft.

---

## Passo 4 — Versão forte (TV + $\mathcal{P}$) por último

A formulação mais publicável

$$
\hat{\alpha} = \arg\min_{\alpha} \frac{1}{2}\| M\Psi\alpha - (b - M f_\theta(u)) \|_2^2
+ \lambda_1 \| W\alpha \|_1
+ \lambda_2 \, \mathrm{TV}(\Psi\alpha)
+ \lambda_3 \, \mathcal{P}(\Psi\alpha)
$$

faz sentido **depois** de existir um caso em que:

- a saída tenha estrutura espacial real;
- TV seja fisicamente defensável;
- e $\mathcal{P}$ tenha interpretação clara no domínio.

No setup atual ($\Psi = I$, sinal sintético simples), TV pode até melhorar numericamente sem ser a melhor mensagem científica. O draft já trata essa extensão como variante estruturada; ela não precisa ser o passo imediato.

---

## Ordem recomendada (resumo)

1. Robustez do `hybrid`.
2. Diagnóstico do `weighted_hybrid`.
3. Cenário semi-realista.
4. Caso real.
5. Versão forte com TV / $\mathcal{P}$ se o domínio justificar.

---

# Plano de ablação em camadas (detalhado)

O melhor próximo passo é um **plano de ablação em camadas**, começando no regime sintético atual (construído para validar a hipótese central) e avançando gradualmente para cenários menos favoráveis.

No estudo atual, o gerador usa $p=12$, $N=128$, $\Psi=I$, $k=6$, $M$ gaussiana e três seeds; nesse regime o `hybrid` vence `ml_only` em todas as razões de medição, enquanto `cs_only` é um referencial *misspecified* e pior. O draft já posiciona essa seção sintética como *sanity check* e referência de ablação antes de dados reais.

**Mensagem estratégica:** o paper, por enquanto, deve ser construído em torno do **`hybrid` básico**. A variante `weighted_hybrid` entra como **extensão exploratória**, porque hoje ainda não superou o `hybrid` simples de forma consistente.

---

## Fase 0 — Baseline de referência

Antes das ablações, **congelar** uma configuração de referência:

- seeds: **10**
- $N=128$
- $p=12$
- $\rho \in \{0.2, 0.3, 0.4, 0.5, 0.6\}$
- $\Psi = I$
- $k=6$
- `residual_mode = support_from_u`
- `measurement_kind = gaussian`
- `measurement_noise_std = 0.02`
- `output_noise_std = 0.01`

**Refinar a grade de $\lambda$:** nos logs, o melhor valor caiu repetidamente na **borda inferior** da grade atual para `hybrid` e `cs_only`, o que sugere *tuning* ainda grosseiro. Grade sugerida (mesma ordem de magnitude, mais fina no extremo baixo):

$$
\{10^{-4},\; 3\cdot 10^{-4},\; 10^{-3},\; 3\cdot 10^{-3},\; 10^{-2},\; 3\cdot 10^{-2}\}.
$$

Isso importa porque o melhor $\lambda$ apareceu muitas vezes em $10^{-3}$, e em alguns casos do `weighted_hybrid` foi para $10^{-1}$ — sinal de **comportamento sensível** ao regularizador.

---

## Fase 1 — Robustez no regime favorável

Permanece na família do problema atual, mas mede **robustez** ao variar um parâmetro de cada vez (ou desenho fatorial enxuto, conforme orçamento).

| ID | O que variar | Grade sugerida | Hipótese |
| --- | --- | --- | --- |
| E1 | esparsidade $k$ | $k \in \{2,4,6,8,12,16\}$ | `hybrid` ganha mais quando o residual é mais esparso |
| E2 | ruído de medição | $\sigma_\eta \in \{0, 0.01, 0.02, 0.05, 0.1\}$ | ganho cai com ruído, mas `hybrid` segue acima de `ml_only` numa faixa útil |
| E3 | ruído de saída | $\sigma_{\mathrm{out}} \in \{0, 0.005, 0.01, 0.02, 0.05\}$ | baseline sofre menos que `cs_only`, mas `hybrid` mantém vantagem enquanto o residual continuar recuperável |
| E4 | amplitude da inovação | $a \in \{0.4, 0.8, 1.2, 1.6, 2.0\}$ | se a inovação cresce, o benefício do estágio CS deve crescer |
| E5 | razão de medição | $\rho \in \{0.1, 0.2, \ldots, 0.6, 0.8\}$ | ganho do `hybrid` cresce de forma monótona ou quase monótona com $\rho$ |

**Métricas principais:** RMSE, MAE, erro $\ell_2$ relativo, ganho sobre `ml_only`, support F1.

**Critério de sucesso (Fase 1):**

- `hybrid` melhor que `ml_only` na **maior parte** da grade;
- `cs_only` pior que `hybrid` de forma **estável**;
- tendência de ganho com $\rho$ **mantida**.

**Gráficos sugeridos:** RMSE vs parâmetro variado; ganho sobre `ml_only`; support F1 vs parâmetro; **heatmap** de “quem vence” por condição.

---

## Fase 2 — Quebrar as hipóteses aos poucos

Testa se o método só funciona no cenário “perfeito” ou também em cenários mais ásperos — alinhado às **limitações** do draft ($M$ não gaussiana, *mismatch*, $\Psi$ inadequada).

| ID | Mudança | Grade sugerida | Hipótese |
| --- | --- | --- | --- |
| E6 | base de esparsidade | `identity`, `dct` | `hybrid` continua útil; ganho pode diminuir se $\Psi$ deixar de coincidir com a estrutura verdadeira |
| E7 | modo do suporte | `support_from_u`, `random` | sem relação entre suporte e $u$, o weighted tende a piorar ainda mais |
| E8 | operador de sensores | `gaussian`, `subsample` | `gaussian` mais favorável; `subsample` + identidade pode ficar mal condicionado |
| E9 | *mismatch* de base | gerar em `dct` e reconstruir em `identity`, e vice-versa | ganho do `hybrid` cai com base “errada” |
| E10 | residual menos esparso | coeficientes compressíveis, não exatamente $k$-esparso | `hybrid` ainda pode ganhar, mas menos |

**Critério de sucesso:** o método **não** precisa vencer em tudo; precisa mostrar **região de validade bem delimitada** (“não só funciona, mas **quando** funciona”).

---

## Fase 3 — Diagnosticar a variante `weighted_hybrid`

O `weighted_hybrid` ainda não está validado: o passo certo é **diagnosticar o prior**, não insistir só no RMSE final. O pipeline já usa preditor auxiliar de $\alpha$ e pesos a partir da magnitude prevista.

| ID | O que medir | Hipótese |
| --- | --- | --- |
| E11 | correlação entre $\hat{\alpha}_{\mathrm{pred}}$ e $\alpha^\star$ | baixa correlação $\Rightarrow$ weighted falha por prior ruim |
| E12 | precision/recall de suporte e overlap top-$k$ do prior | weighted só ajuda quando o prior tem qualidade real |
| E13 | esquemas de peso | `inverse_magnitude`, máscara top-$k$ *hard*, pesos binários, **oracle** — isola prior vs parametrização |
| E14 | weighted com prior oracle | suporte verdadeiro para construir $W$ | teto de desempenho da ideia *weighted* |

**Leitura:** se nem o **oracle** melhorar muito, a formulação *weighted* pode não valer o esforço; se o oracle melhorar bastante mas o aprendido não, o gargalo é o **preditor auxiliar**.

**Gráficos:** ganho *weighted* vs qualidade do prior; precision/recall do prior; tabela `weighted_learned` vs `weighted_oracle`.

---

## Fase 4 — Semi-realista

Após robustez, aproximar o gerador do uso real:

- $\Psi$ = DCT ou wavelet;
- saída com estrutura 1D/2D mais suave;
- componente global mais estruturada (menos “aleatória”);
- medição por máscara/sensores;
- residual em blocos, frentes ou eventos localizados.

| ID | Cenário | Hipótese |
| --- | --- | --- |
| E15 | sinal 1D suave + eventos localizados | `hybrid` continua na frente |
| E16 | campo 2D achatado em vetor | ganho se inovação for localizada |
| E17 | medições por máscara de sensores | queda geral, mas vantagem híbrida pode persistir |
| E18 | ruído heterocedástico ou *outliers* leves | robustez além do gaussiano homogêneo |

**Critério de sucesso:** repetir a narrativa do paper num cenário **menos construído para ganhar**.

---

## Fase 5 — Pré-real

Preparar a transição para caso real (próximo marco experimental do draft). Pipeline deve aceitar, no mínimo:

- `X_train`, `X_val`, `X_test`
- `Y_train`, `Y_val`, `Y_test`
- operador $M$ definido externamente
- base $\Psi$ definida externamente

Rodar no mínimo: `ml_only`, `hybrid`, `cs_only`. `weighted_hybrid` só se a Fase 3 justificar.

---

## Ordem de execução (custo computacional)

### Etapa A — Triagem rápida

- 3 seeds; grade curta
- E1, E2, E6, E8, E11, E14

**Objetivo:** decidir o que merece aprofundamento.

### Etapa B — Rodada “paper” sintético

- 10 seeds; grade refinada de $\lambda$
- E1–E10; E11–E14 se o *weighted* ainda estiver “vivo”

### Etapa C — Semi-realista

- E15–E18

### Etapa D — Caso real

- protocolo externo (dados e $M$, $\Psi$ reais)

---

## Quais gráficos gerar em cada fase

### Obrigatórios

- RMSE vs condição; MAE vs condição
- ganho sobre `ml_only`
- support F1 vs condição
- *parity plots*; histogramas de resíduos
- 3 a 5 *example traces*

### Muito úteis

- heatmap “hybrid vence ml_only”
- sensibilidade a $\lambda$
- qualidade do prior vs ganho *weighted*
- tabela de ranking médio dos métodos

---

## O que vira texto de paper

Se as Fases 1 e 2 forem sólidas, a narrativa pode ser:

1. o método é correto no regime favorável;
2. o ganho cresce com $\rho$;
3. a vantagem persiste numa faixa não trivial de ruído/esparsidade/*mismatch*;
4. `cs_only` falha por não modelar o *background*;
5. `weighted` é extensão promissora **ou** ablação negativa informativa.

---

## Critério objetivo de “pronto para avançar”

Avançar para semi-realista/real se, após as ablações:

- `hybrid` superar `ml_only` em **70–80%** das condições razoáveis;
- ganho médio em RMSE **claramente positivo** e reprodutível;
- **região de falha** bem caracterizada;
- `weighted` **validado** ou **rebaixado** oficialmente a experimento secundário.
