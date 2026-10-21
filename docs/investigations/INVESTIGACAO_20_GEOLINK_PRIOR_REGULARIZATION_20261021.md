# Investigação 20 — Viés do prior condicionado e intensidade da regularização latente (Geolink)

**Data:** 21 October 2026. **Status:** execução completa, três sementes. **Tipo de evidência:** diagnóstico exploratório, não teste confirmatório.

**Repositório:** \`romulobrito/CS-ML\`; **branch:** \`audit/geolink-prior-lambda-diagnosis-20261021\`.
**Código:** \`scripts/geolink_fullscale/investigation20.py\`.
**Execução:** https://github.com/romulobrito/CS-ML/actions/runs/39681104543
**Artefato:** https://github.com/romulobrito/CS-ML/actions/runs/39681104543/artifacts/12384988342
**SHA256 do arquivo Geolink:** \`67a659043770fab643d60d4d8fd674da50d178797f49e664d8c9abdc532a7aa6\`

## Pergunta científica

O CLP perde para métodos simples porque (i) a penalização \`lambda*||z-z0||²\` prende a solução a um prior pouco informativo; (ii) o uso de \`z0=h(GR,RHOB,NPHI)\` como inicialização induz uma bacia desfavorável; ou (iii) ambos, associados à escassez de amostras do DTC?

Esta investigação isola **condicionamento como inicialização** e **condicionamento como penalização**, comparando o método com um CSGM sem prior. Não modifica o gerador: autoencoder \`TinyAE\`, 16 latentes, treinamento 200 épocas, e \`RidgePrior(alpha=1)\` em dados auxiliares.

## Desenho experimental

- Fold 0 do benchmark oficial Geolink, janelas de 256 amostras; teste externo invariável: **1.019 janelas, 36 poços**.
- Desenvolvimento original: 3.126 janelas para treino / 778 para validação, sem poços em comum. Desenvolvimento estratificado por extremos de DTC: 3.221/683 janelas, novamente com 113/28 poços disjuntos; estratificação baseada apenas nos poços de desenvolvimento, sem informação do teste.
- Máscaras predefinidas: **32 ou 64 observações consecutivas de DTC**, posições determinadas pela semente; entrada condicionante completa GR/RHOB/NPHI.
- Sementes: **7, 23, 41**. Lâmbdas: 0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1. Seleção pelo RMSE global de validação (análise suplementar: média de RMSE por poço).
- Três inicializações/reinicializações por janela; 400 iterações de Adam; mesma máscara, objetivo de observações e gerador entre braços.
- Braços: **CLP com λ escolhido na validação**; **inicialização condicionada e λ=0**; **CSGM não condicionado e λ=0, inicializado com códigos latentes de treinamento**. Baselines GP, interpolação/extrapolação, média observada e prior isolado.
- O codificador que acessa DTC externo completo foi utilizado **somente para o diagnóstico retrospectivo de viés latente**. Não houve uso de DTC externo para escolher λ, arquitetura ou inicializações operacionais.
- Importante: todos os resultados externos são **exploratórios** porque o fold 0 já havia sido examinado em investigações anteriores. Os números não devem ser convertidos em evidência confirmatória de superioridade.

## Resultados principais

RMSE global no alvo **ocultado**, médias simples das três sementes, arredondadas:

| Divisão de desenvolvimento | Obs. | CLP λ selecionado | Condicionado, λ=0 | CSGM não condicionado, λ=0 | GP | Interpolação |
|---|---:|---:|---:|---:|---:|---:|
| Original | 32 | **0.4186** | 0.4160 | **0.4052** | **0.1559** | 0.1873 |
| Original | 64 | **0.3751** | 0.3745 | **0.3659** | **0.1591** | 0.1533 |
| Estratificada | 32 | **0.5260** | 0.5295 | **0.5336** | **0.1563** | 0.1873 |
| Estratificada | 64 | **0.4204** | 0.4219 | **0.4042** | **0.1591** | 0.1533 |

**Destaques:** com divisão original, retirar a penalização melhora pouco a média global: 0.4186→0.4160 (32) e 0.3751→0.3745 (64). Eliminar **também a inicialização condicionada** diminui para 0.4052 e 0.3659, mas permanece longe dos métodos simples. Na divisão estratificada de 32 observações, o CLP vence ligeiramente o CSGM na métrica global (0.5260 vs 0.5336), porém **não há vantagem consistente** entre regimes e tamanhos de suporte.

### Comparação por poço e bootstrap

Cada poço externo é a unidade de reamostragem. Valores representam a diferença média por poço **RMSE(CLP) − RMSE(comparador)**; sinal positivo favorece o comparador.

| Divisão | Observações | Comparador | Diferença por poço | IC95% (bootstrap) |
|---|---:|---|---:|---|
| Original | 32 | CSGM sem prior | **+0.0413** | [+0.0327, +0.0502] |
| Original | 64 | CSGM sem prior | **+0.0282** | [+0.0231, +0.0336] |
| Estratificada | 32 | CSGM sem prior | **+0.0266** | [+0.0173, +0.0354] |
| Estratificada | 64 | CSGM sem prior | **+0.0192** | [+0.0128, +0.0283] |
| Original | 32 | Condicionado, λ=0 | **−0.0044** | [−0.0065, −0.0015] |
| Original | 64 | Condicionado, λ=0 | **−0.0037** | [−0.0052, −0.0022] |
| Estratificada | 32 | Condicionado, λ=0 | **−0.0024** | [−0.0059, +0.0004] |
| Estratificada | 64 | Condicionado, λ=0 | **−0.0120** | [−0.0143, −0.0096] |

**Interpretação:** a pequena penalização do CLP pode funcionar como regularizador de estabilidade **por poço**, embora não melhore consistentemente o RMSE global ponderado pelo número de janelas. O condicionamento como **inicialização** parece particularmente inadequado: mesmo sem penalização, o braço condicionado perde do CSGM sem prior na análise por poço.

**Crítica de rigor:** as bandas de bootstrap descrevem incerteza amostral no **mesmo** conjunto de 36 poços já analisados; não corrigem escolha exploratória de regimes e não estabelecem generalização a novos campos.

### Viés latente

O artefato \`latent_bias_dims.csv\` contém, por dimensão (0–15), média prevista pelo prior, média do código estimado pelo encoder com DTC completo, viés padronizado por dispersão latente de treinamento, RMSE padronizado e correlação. Inclui diagnóstico separado para janelas com DTC extremo vs ordinárias.

A **qualidade do prior isolado** já era muito baixa: RMSE externo ~0.94 nas investigações 18–20, enquanto o gerador com acesso ao DTC completo alcança RMSE externo da ordem de ~0.09 pelo oracle otimizado (Investigação 18). Portanto, o erro de \`h(u)\` é um candidato forte a explicar a recuperação deficiente. Os resultados desta investigação mostram que a penalização pequena não é, isoladamente, a causa principal.

### Mudança de distribuição

- Validação original: desvio padrão de DTC ~0.029 e zero valores extremos (DTC normalizado > 3).
- Validação estratificada: inclui uma população extrema, porém apenas **um dos seis poços extremos de desenvolvimento** foi destinado à validação. O CLP piorou no teste externo. O objetivo da mudança não era maximizar teste, e sim diagnosticar o efeito da representatividade.
- Teste externo: valores extremos presentes, inclusive nos poços \`34_8-3\` e \`35_11-6\` destacados na Investigação 18.

A escolha por poço evita vazamento entre treino/validação, mas ainda é fundamental auditar qualquer **pré-processamento global publicado** na origem.

## Conclusão em relação ao manuscrito

1. O gerador tem capacidade representacional não explorada adequadamente pela inferência condicionada (Investigação 18).
2. A remoção isolada da penalização λ **não recupera** a vantagem frente aos métodos simples.
3. Usar um ponto inicial não condicionado reduz o erro em parte dos cenários, sugerindo que o prior Ridge \`h(u)\` é pouco informativo e/ou induz bacias de atração inadequadas.
4. **Ainda não demonstramos** superioridade robusta do CLP num regime operacional relevante. Não apresentar os ganhos do piloto como confirmatórios.
5. Futuros testes devem demonstrar **ganho incremental condicionado** frente ao CSGM e a um preditor multivariado com as mesmas informações; a comparação direta com interpolação continua necessária para tornar a aplicação convincente.

## Próximo teste realmente discriminante (não executado nesta etapa)

Com o gerador fixo, treinar um condicionador latente **fora de distribuição por poço**, comparar Ridge, predição não linear e condicionamento com incerteza/calibração. Usar **lambda calibrado por qualidade prevista do prior** obtida no desenvolvimento, nunca por erro do teste. Controlar adequação dos códigos aos poços externos, sem utilizar informações destes para seleção. Em paralelo, verificar distribuição dos perfis por poço e onde valores extremos concentram o RMSE.

Uma alternativa mais rigorosa é avaliar folds inteiramente não utilizados para ajustar a hipótese; não tratar o fold 0 revisitado como teste confirmatório.

## Artefatos auditáveis

O arquivo do GitHub Actions contém \`metric_cells.csv\`, \`metric_wells.csv\`, \`validation_regularization.csv\`, \`latent_bias_dims.csv\`, \`paired_bootstrap_wells.csv\`, \`manifest.json\`, \`summary.json\`; possui também a rodada técnica \`smoke\`, que **não** integra as tabelas acima.

**Uso dos resultados:** investigação exploratória e revisão crítica; nenhuma incorporação ao manuscrito ou à branch principal nesta etapa.
