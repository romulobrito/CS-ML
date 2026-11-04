# Auditoria final da redacao cientifica

Manuscrito: CAGEO-D-26-01398

Checkpoint pre-FINAL-04:
`38b1e5ca33a78593257a711b8503e4f02a0498ef`

Data: 2026-11-04

## A. Diagnostico inicial

| Secao | Problema identificado | Exemplo do texto original | Consequencia para a leitura | Prioridade |
|---|---|---|---|---|
| Abstract | A formulacao apresenta componentes e ressalvas antes de desenvolver os resultados centrais. | `A residual at the observed depths can remain. The local recovery bound remains conditional...` | A contribuicao empirica perde destaque e o resumo adquire tom defensivo antes dos resultados. | Alta |
| Introduction | O mecanismo do decoder, de `z_0` e do refinamento e descrito novamente na lista de contribuicoes. | `The conditional map predicts a latent code...` e, depois, `First, the dense logs predict z_0...` | A repeticao reduz a progressao entre motivacao, contribuicao e evidencia. | Media |
| Related work | O posicionamento e correto, mas a ultima frase repete uma delimitacao ja presente na Introduction. | `these comparisons establish methodological positioning rather than predictive superiority` | A secao termina em ressalva, em vez de sintetizar a diferenca metodologica. | Baixa |
| Method | A abertura repete quase literalmente Abstract e Introduction. | `CLP-CSGM uses three pieces. A decoder... The conditional map predicts...` | A secao formal demora a avancar para a definicao operacional. | Media |
| Data and EDA | A justificativa para excluir gamma ray combina a origem do alvo, a pratica de campo e a limitacao do benchmark em frases sucessivas. | `When gamma ray has been logged, that relation is the calculation...` | A informacao correta fica dispersa e reaparece em Experimental setup e Discussion. | Media |
| Experimental setup | As delimitacoes dos dois benchmarks sao repetidas em varias subsecoes. | `This is not the log suite...`, `It is not a recommendation...`, `not recovery against...` | O protocolo parece mais defensivo do que descritivo. | Media |
| Results | A abertura antecipa comparadores, custo e ressalvas antes de desenvolver a evidencia principal frente ao AE. | `Against simple reconstructions... it does not... The extra iterative cost is not generally repaid.` | A pergunta cientifica e respondida antes de a evidencia ser apresentada, enfraquecendo a narrativa. | Alta |
| Results | Os resultados de dez sementes e novas realizacoes repetem varias vezes a mesma ordenacao dos steps 16 e 32. | `The CLP-CSGM advantage remains...`, `has lower RMSE... in every draw`, `therefore holds for every tested placement` | A repeticao fragmenta a verificacao de robustez. | Alta |
| Results | O sweep de ruido alterna resultado, mecanismo plausivel e varias negacoes consecutivas. | `It is not a guarantee...`, `does not isolate...`, `does not establish...` | A interpretacao positiva fica diluida e a ressalva aparece antes de uma sintese coesa. | Alta |
| Discussion | A abertura redefine detalhadamente decoder, prior e penalidade. | `CLP-CSGM uses three pieces that act together...` | A Discussion repete a Method em vez de interpretar o que foi aprendido. | Alta |
| Discussion | As limitacoes de Larionov, porosidade processada e validacao operacional aparecem em varios paragrafos parcialmente sobrepostos. | `CLP-CSGM is not a substitute...`, `not validation...`, `not a contest...` | O escopo correto domina a argumentacao e reduz o protagonismo dos achados. | Alta |
| Discussion | Resultados favoraveis das ablacoes e da robustez sao seguidos imediatamente por varias qualificacoes. | `They do not show...`, `do not isolate...`, `do not establish...` | O texto comunica cautela, mas pouco da implicacao cientifica dos resultados. | Alta |
| Conclusion | A primeira parte reexplica o metodo em detalhe e os paragrafos seguintes reproduzem muitos numeros de Results. | `A decoder trained... The conditional map predicts...` | A conclusao funciona como resumo extenso, nao como sintese do aprendizado cientifico. | Alta |
| Conclusion | O fechamento acumula limitacoes em frases consecutivas. | `does not establish...`, `is not evidence...`, `will require...` | O artigo termina pelas ausencias, sem equilibrar contribuicao e proxima etapa. | Alta |
| Appendix | As provas e protocolos estao adequadamente detalhados; algumas ressalvas se repetem, mas sustentam a reproducibilidade. | Varias notas sobre sementes, janelas e limites de cada suplemento. | Uma revisao ampla teria alto risco e baixo beneficio. | Baixa |

## B. Alteracoes realizadas

| Local | Trecho anterior | Trecho revisado | Justificativa editorial | Impacto esperado |
|---|---|---|---|---|
| Abstract | `A learned decoder maps... Dense auxiliary logs predict... Latent refinement balances...` | `CLP-CSGM instead combines a learned target-window representation, a latent anchor... and test-time refinement...` | Articular os componentes pelo papel que desempenham, sem uma sequencia de definicoes curtas. | A proposta aparece antes das ressalvas e conduz diretamente aos resultados. |
| Introduction | A descricao do mecanismo era repetida na motivacao e na lista de contribuicoes. | A motivacao agora distingue contextual prediction de measurement assimilation; as contribuicoes passam a formular o problema inverso, a interpretacao teorica e a avaliacao empirica. | Separar motivacao, contribuicao e evidencia. | Progressao mais clara do problema para a solucao e os achados. |
| Method | `CLP-CSGM uses three pieces. A decoder... The conditional map...` e antecipacoes dos resultados cross-well. | A abertura separa representacao, predicao contextual e assimilacao; os resultados empiricos foram direcionados a Results. | Manter a definicao formal sem repetir Abstract, Introduction ou achados posteriores. | Entrada mais direta na formulacao matematica e melhor separacao entre metodo e evidencia. |
| Data and EDA | A exclusao de gamma ray era explicada por varias frases sobre derivacao, pratica de campo e substituicao da relacao. | A origem Larionov, o risco de expor o log-fonte e a finalidade do teste foram reunidos em um unico paragrafo. | Concentrar a justificativa factual sem remover a proveniencia. | Menos repeticao com Experimental setup e Discussion. |
| Experimental setup | A descricao GR-only acumulava explicacoes separadas sobre densidade, sonic, gamma ray e ausencia de validacao laboratorial; controles suplementares antecipavam seus vencedores. | A redacao liga o processamento da curva, a quase colinearidade de sonic e o carater restricted-input, mantendo no setup apenas os protocolos dos controles. | Priorizar o desenho experimental e concentrar a evidencia em Results. | Maior legibilidade metodologica e progressao mais clara entre protocolo e achados. |
| Results | A abertura antecipava a derrota para metodos simples e o custo antes de apresentar a comparacao principal com AE. | Tres perguntas organizam a secao: comparacao externa, contribuicao das ablacoes e condicoes em que metodos simples bastam. | Dar sequencia argumentativa aos resultados. | A evidencia favoravel e o contraponto aparecem na ordem em que sao demonstrados. |
| Results | Os paineis de dez sementes, mascara e novas realizacoes repetiam a vantagem nos steps 16 e 32. | Os tres controles foram consolidados em dois paragrafos, preservando ordenacoes, multiplicidade, dependencia e limite de um unico poco. | Remover repeticao sem ampliar a inferencia estatistica. | Robustez mais facil de acompanhar. |
| Results | O sweep alternava contagens, mecanismos, negacoes e detalhes da grade. | As ordenacoes frente a interpolacao e smoothing aparecem juntas; depois vem a interpretacao, o escopo sintetico, a selecao de lambda e o custo. | Distinguir evidencia, mecanismo compativel e limitacao. | Narrativa progressiva sem ocultar o resultado 9/9 da interpolacao no ruido de referencia. |
| Discussion | A abertura redefinia decoder, prior e refinamento em detalhe. | A secao passa a iniciar pela diferenca metodologica e pela evidencia direta das ablacoes. | Fazer a Discussion interpretar, em vez de repetir a Method. | O papel da assimilacao esparsa ganha protagonismo. |
| Discussion | As limitacoes dos alvos processados e da aplicacao operacional estavam distribuidas em varios paragrafos. | A proveniencia de `V_c`, porosidade e observacoes foi reunida em uma discussao de escopo experimental, seguida pelos suplementos e necessidades de validacao. | Preservar todas as limitacoes com menor tom defensivo. | Equilibrio entre utilidade demonstrada e alcance empirico. |
| Conclusion | A conclusao reexplicava todo o pipeline e reproduzia uma lista longa de metricas. | Quatro paragrafos sintetizam contribuicao, comparacao com AE e ablacoes, contraste com metodos simples e limites de transferencia. | Responder ao aprendizado cientifico, nao repetir Results. | Fechamento mais curto e argumentativamente equilibrado. |

## C. Narrativa cientifica

A revisao reposiciona a contribuicao como a combinacao de representacao
aprendida, prior condicionado e assimilacao explicita das observacoes. O
espaco latente nao e apresentado, isoladamente, como fonte de superioridade.

Os resultados frente ao AE aparecem primeiro: sete de nove configuracoes,
ganhos mais consistentes nos steps 16 e 32 e reducao media nao ponderada de
18.1%. A redacao preserva que o step tambem altera amostragem e sobreposicao,
evitando inferencia causal exclusiva sobre quantidade de treinamento.

As ablacoes fornecem a evidencia interna mais direta. O modelo completo e
menor que prior-only nas nove celulas cross-well e menor que
measurement-only em sete, enquanto no benchmark GR-only e menor que ambas as
ablacoes em todas as razoes. Essa evidencia e distinguida das comparacoes
externas.

Interpolacao e smoothing formam o contraponto: no ruido 0.01, as solucoes
simples apresentam menor RMSE na maior parte ou em toda a grade. O sweep
sintetico e apresentado como robustez relativa do procedimento completo, com
reselecao de lambda, sem limiar universal ou calibracao instrumental.

As limitacoes foram concentradas por tema: proveniencia dos alvos,
single-holdout, ausencia de rotulos geologicos, suplemento laboratorial e
condicionalidade teorica. Elas permanecem explicitas sem interromper cada
resultado favoravel com varias negacoes consecutivas.

## D. Verificacao cientifica

As alteracoes foram conferidas contra as tabelas, a formulacao matematica e a
carta consolidada. Permaneceram inalterados:

- resultados numericos e contagens por configuracao;
- comparacoes CLP-CSGM versus AE, interpolacao, smoothing e ablacoes;
- reselecao de lambda no sweep de ruido;
- equacoes, labels, proposicoes, provas e algoritmos;
- origem processada dos alvos e subamostragem artificial;
- limite S-REC e distincao entre solucao numerica e minimizador global;
- declaracoes de assistencia de ChatGPT e Grok 4.7.

Exigiram julgamento cientifico, e nao apenas gramatical:

- caracterizar os steps 16 e 32 como configuracoes de menor suporte sem
  atribuir causalidade exclusiva ao tamanho do treinamento;
- apresentar o sweep como robustez relativa do procedimento completo;
- afirmar que a assimilacao melhora o prior-only sem converter esse resultado
  em superioridade frente a todos os comparadores;
- concentrar as limitacoes operacionais sem remover o single-holdout, a
  proveniencia dos alvos ou a falta de calibracao instrumental.

Uma segunda leitura separada identificou tres pontos. A definicao de `rho` e
seus valores foram mantidos no Abstract para preservar a carta; a Conclusion
voltou a declarar explicitamente o unico poco retido e a ausencia de
transferibilidade multi-field; e `accurate reconstruction` foi substituido
por `lower-RMSE reconstruction in this benchmark`. R1-m05 permanece pendente
e fora do escopo desta revisao.

Na verificacao final, antecipacoes de resultados ainda presentes na Method e
no Experimental setup foram removidas. As descricoes de seeds, mascaras,
realizacoes de observacao, multiplicidade e demais detalhes reprodutiveis
permaneceram nesses locais, enquanto vencedores e ordenacoes ficaram
concentrados em Results.

O parecer independente posterior recomendou tres correcoes pontuais, todas
incorporadas sem alterar resultados: a Conclusion agora identifica
explicitamente o comparador como AE
`[\boldsymbol{u},\boldsymbol{b}]`; a Introduction qualifica a leitura
MAP/Tikhonov como classica; e Results descreve a melhora da ordenacao relativa
sob ruido sintetico, sem inferir uma taxa geral de crescimento do RMSE.

## E. Resultado tecnico

### E.1 Separacao das alteracoes

Antes da FINAL-04, a working tree ja continha modificacoes e artefatos de
outras frentes. Dentro de `paper_clp_csgm`, esse conjunto incluia arquivos de
acompanhamento editorial, documentos de apoio e referencias locais; fora dele,
havia codigo, resultados e dados nao relacionados ao manuscrito. Esses itens
foram preservados e nao foram incorporados indiscriminadamente a esta revisao.

As alteracoes textuais da FINAL-04 estao restritas a `main.tex` e as sete
secoes listadas abaixo. `main.pdf` e `main.blg` foram atualizados pela
compilacao, e este relatorio foi criado como artefato de auditoria. O diff
integral das secoes contra o checkpoint foi salvo em
`docs/revision/final04_sections_vs_checkpoint.patch`; o diff do Abstract em
`main.tex` foi salvo separadamente em
`docs/revision/final04_main_vs_checkpoint.patch`.

### E.2 Arquivos e diff

Arquivos textuais modificados:

- `main.tex`;
- `sections/01_introduction.tex`;
- `sections/03_method.tex`;
- `sections/04_data_and_eda.tex`;
- `sections/05_experiments.tex`;
- `sections/06_results.tex`;
- `sections/07_discussion.tex`;
- `sections/08_conclusion.tex`;
- `docs/revision/auditoria_redacao_cientifica_final.md`.

`sections/02_related_work.tex` e `sections/09_appendix.tex` foram relidos e
permaneceram inalterados porque uma reescrita produziria baixo ganho e maior
risco de alterar posicionamento ou reproducibilidade.

Diff textual contra o baseline:

- oito arquivos LaTeX: 352 linhas adicionadas e 516 removidas;
- somente `sections/`: 321 linhas adicionadas e 486 removidas;
- este relatorio foi criado nesta tarefa;
- `main.pdf` recompilado.

### E.3 Paginacao e verificacao estrutural

Resultado tecnico:

- `latexmk` concluido com codigo de saida zero;
- PDF final com 87 paginas, tres a menos que o baseline de 90 paginas;
- reducao de paginacao explicada pela consolidacao de prosa repetitiva, sem
  alteracao de figuras, tabelas ou equacoes;
- nenhuma citacao ou referencia cruzada indefinida;
- nenhum `Overfull \hbox`;
- dois avisos `Underfull \hbox`, os mesmos ja presentes no baseline;
- 60 labels preservados nos arquivos revisados;
- 17 ambientes `equation`, uma `proposition` e dois `remark`, sem mudanca de
  contagem;
- dez ambientes `figure`, onze ambientes `table`, dez comandos
  `includegraphics` e onze comandos `input`, todos preservados;
- nenhum arquivo em `tables/`, `figures/` ou `references.bib` foi alterado
  pela FINAL-04;
- carta consolidada e respostas R1--R3 sem alteracoes;
- `git diff --check` e linter sem erros.

Nenhuma referencia bibliografica precisou ser substituida ou removida. As
citacoes acrescentadas antes desta tarefa para a justificativa do ruido
sintetico permanecem resolvidas.

A comparacao com o checkpoint confirma que equacoes, resultados numericos,
figuras, tabelas, parametros, procedimentos de validacao e conclusoes
estatisticas foram preservados. As mudancas recaem sobre organizacao,
transicoes, concisao e proporcionalidade entre contribuicoes e limitacoes.
