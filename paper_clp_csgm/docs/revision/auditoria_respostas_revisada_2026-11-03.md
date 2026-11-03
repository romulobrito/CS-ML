# Auditoria independente do manuscrito e das respostas aos pareceristas

**Artigo:** *Conditional Latent-Prior Generative Compressed Sensing for Petrophysical Target-Log Reconstruction*
**Submissão:** CAGEO-D-26-01398 — *Computers & Geosciences*
**Data da auditoria:** 3 de novembro de 2026
**Fontes verificadas:** `main.pdf` (90 paginas), `main.tex`, secoes LaTeX
`01` a `09`, Figura 2, comentarios consolidados, `respostas_R1.md`,
`respostas_R2.md`, `respostas_R3.md`, `respostas_AE.md`, `status.md` e
`ae01_matriz_conformidade.md` no repositorio.

> **Delimitação solicitada:** a questão da disponibilização de código em versão congelada (*R1-m05*, tag/release) permanece aberta, **mas está fora do escopo de correção desta etapa**. Este relatório não recomenda criar uma tag, migrar scripts ou alterar a seção *Computer Code Availability* agora. A pendência é registrada para acompanhamento posterior.

## 1. Objetivo e método

Verificar a **aderência das respostas aos comentários**, sua clareza e naturalidade em inglês acadêmico e sua **coerência com o manuscrito efetivamente anexado**. A intenção é selecionar ajustes editoriais de alto valor, sem iniciar nova reescrita geral nem mudar resultados científicos.

A verificacao abrangeu as 52 entradas organizadas nos arquivos de resposta
(51 comentarios dos tres revisores e a sintese AE-01), com atencao especial
as contestacoes metodologicas e aos resultados quantitativos. Foram conferidos
o comentario original, o LaTeX atual, o PDF de 90 paginas, os elementos dos
fluxogramas e as declaracoes de IA. **Nao foram reexecutados experimentos,
auditados dados brutos ou certificados os scripts de reproducao.** Assim, a
conferencia de numeros neste documento significa correspondencia com os
valores publicados no PDF, e nao reproducao computacional independente.

## 2. Parecer executivo

**Parecer: favorável, com poucas correções editoriais localizadas.** As respostas examinadas atendem, em termos gerais, aos comentários e apresentam uma narrativa científica mais clara do que um histórico de investigação. Os limites são reconhecidos: apenas um poço cross-well retido, curvas F3 processadas, observações esparsas simuladas por subamostragem e ausência de validação operacional generalizada.

A principal comparação também não é ocultada: **CLP-CSGM Ridge tem RMSE médio inferior ao AE direto em 7/9 configurações cross-well, enquanto a interpolação em profundidade tem RMSE inferior ao CLP-CSGM nas 9/9 configurações com ruído de referência 0,01**. A vantagem condicional em um comparador não é indevidamente apresentada como superioridade global.

**Estado apos a correcao:** `respostas_AE.md` agora marca AE-01 como `closed`
e aponta para a unica resposta canonica, mantida em `respostas_R3.md`. A
divergencia documental foi eliminada sem duplicar o texto da carta.

## 3. Achados e estado das acoes

| Estado | Fonte / localizacao | Evidencia observada | Tratamento |
|---|---|---|---|
| **Resolvida — documental** | `respostas_AE.md`; `respostas_R3.md`, AE-01 | O arquivo separado marcava AE-01 como `pending`, enquanto a resposta consolidada estava `closed`. | `respostas_AE.md` agora marca `closed` e aponta para a resposta canonica em `respostas_R3.md`, sem duplicacao. |
| **Resolvida — clareza** | Figura 2, PDF p. 14 | O fluxograma ilustra GR, AC e RHOB, embora GR seja excluido do benchmark cross-well principal. | A legenda agora informa que os canais sao ilustrativos, variam por benchmark e que GR e excluido do benchmark principal de volume de argila. A figura nao foi redesenhada. |
| **Resolvida — notacao** | R3-P11 e R3-P14 em `respostas_R3.md` | Expressoes como `z_0`, `h`, `G_orig(z_hat)` e `M_Omega` apareciam parcialmente como texto literal. | A notacao foi uniformizada para \(\boldsymbol{z}_0\), \(h\), \(G_{\mathrm{orig}}(\widehat{\boldsymbol{z}})\) e \(M_\Omega\). |
| **Resolvida — foco editorial** | R3-A04, R3-P08, R3-P15, R3-P18 e respostas selecionadas do R1 | Restavam referencias a scripts, IDs de outros comentarios e cronologia interna. | As referencias que nao sustentavam a resposta foram removidas; a rastreabilidade permanece em `status.md` e nos relatorios de auditoria. |
| **Conferida — concisao seletiva** | R3-02, R3-06, R3-P15, R1-M09 | Algumas respostas permanecem extensas porque tratam de criticas substantivas. | Foram removidas apenas redundancias e metalinguagem evidentes. Resultados, hipoteses e ressalvas foram preservados. |
| **Resolvida — precisao da conclusao** | Abstract, PDF p. 2 | A frase *“These findings show regime-dependent, not general, superiority”* podia parecer mais ampla que a evidencia. | O texto agora explicita ganhos condicionais em relacao ao AE direto e nega superioridade geral. |
| **Pendente futura — não atuar agora** | R1-m05 / *Computer Code Availability*, PDF p. 2 | O texto de disponibilidade menciona scripts e diagnósticos suplementares; a auditoria anterior registrou que o pacote público ainda não inclui todos os lançadores usados na revisão. | **Não criar tag nem modificar o código ou a seção nesta etapa.** Retomar separadamente, depois de concluir a carta-resposta, sem marcar R1-m05 como `closed`. |

## 4. Conferência cruzada: evidência no PDF e resposta correspondente

| Tema | Páginas do PDF | Respostas relevantes | Resultado da conferência |
|---|---|---|---|
| Objetivo e limitação empírica | 1–4, 39–43 | R1-M01, R1-M02, R1-M05, R2-02, AE-01 | Coerente: o objetivo é reconstrução condicionada com observações esparsas; não se afirma transferência multiárea ou superioridade geral. |
| Origem do alvo \(V_c\), Larionov e GR | 17–19, 40 | R2-03, R3-02, R3-P17, R3-P19 | Coerente: \(V_c\) é fração adimensional derivada de GR, e a exclusão do canal é deliberada para um teste controlado. |
| Proveniência `30dB` e ruído acrescentado | 17, 21–24 | R3-04, R3-P09, R3-P16 | Coerente: `30dB` é mantido como identificador do arquivo, separado de \(\sigma_\eta\) aplicado às observações esparsas. |
| Operador, escalonamento e covariância | 7–13 | R3-P08, R3-P09, R3-P10, R3-P11 | Coerente: \(M_\Omega\) seleciona coordenadas; \(D_\Omega\) é escala diagonal; a covariância do ruído padronizado é \(\sigma_\eta^2D_\Omega^{-2}\), sob a hipótese gaussiana adotada. |
| Priorização latente e objetivo | 10–15 | R3-P06, R3-P11, R3-P12, R3-P13 | Coerente: \(z_0=h(u)\) inicializa o refinamento e é o centro da penalidade; não é reconstrução independente e não garante maior acurácia. |
| Interpretação MAP e recuperação condicional | 14–17, 52–55 | R1-M07, R3-05, R3-P15 | Coerente: MAP e existência são resultados clássicos; S-REC permanece hipótese, não garantia demonstrada no decoder treinado. |
| Figuras e legibilidade | 12 (Fig. 1), 14 (Fig. 2), 31 (Fig. 7), 33 (Fig. 8), 81 (Fig. 24) | R1-m03, R3-A03, R3-06, R3-P14, R3-P19 | Figuras principais e legendas estao coerentes com as respostas; a legenda da Figura 2 agora explicita a natureza ilustrativa dos canais. |
| Comparação com métodos simples | 26–36 e apêndices correspondentes | R1-M05, R3-03, R3-04, AE-01 | Coerente quanto às vitórias 7/9 frente ao AE e 9/9 de interpolação à frente no ruído de referência. A comparação não isola causalmente os componentes. |
| Suplemento de porosidade laboratorial | 35, 87–90 | R1-M11, AE-01 | Coerente: limitado a um intervalo, com restrições de proveniência e janelas sobrepostas. O AE direto tem melhor média agregada. |
| Transparência sobre IA | 12, 14, 20, 24 e 45 | AE-01 | Declaração geral e legendas identificam ChatGPT em tradução/organização/esquemas e Grok 4.7 na refatoração do código preexistente. |

As localizações acima são as **páginas impressas do PDF anexado**, não páginas de uma versão anterior do projeto.

## 5. Avaliação da redação da carta

### O que já está bom

A redação tende a seguir a sequência mais eficaz para uma carta-resposta: reconhecer a crítica, explicar a solução ou o resultado relevante e indicar a seção alterada. As versões recentes reduziram detalhes de debugging, histórico de commits e cronologias de investigação. Há cuidado em diferenciar **resultado observado** de **interpretação causal**.

Destacam-se positivamente:

- **R3-P09/R3-P10:** respondem diretamente à origem do ruído, seleção de \(\lambda\) e covariância transformada, sem adicionar discussões matemáticas não solicitadas.
- **R3-P12/R3-P13:** justificam a reconstrução latente e distinguem representação retrospectiva \(G(E(y))\), prior-only e resultado após assimilação de medições.
- **R3-P19:** defende a finalidade do experimento com GR deliberadamente omitido, sem alegar substituir Larionov.
- **AE-01:** organiza a síntese em escopo, comparadores, robustez, interpretação teórica e transparência do uso de IA.

### O que não vale fazer

Não reabrir as conclusões sobre alvos derivados, ruído não calibrado, poucos poços e interpolação mais forte no ruído de referência só para tornar a carta mais persuasiva. Também não convém trocar todas as aberturas por uma fórmula fixa (*“We thank the reviewer...”*), porque a variação natural e a resposta direta são preferíveis à padronização mecânica.

### Critério de concisão

Comentários simples podem ser respondidos em poucas frases. Para objeções metodológicas sérias — especialmente R1-M06, R1-M11, R3-03, R3-04, R3-05 e R3-P19 — dois ou mais parágrafos podem ser necessários. **Concisão significa remover material que não ajuda a esclarecer a crítica, não omitir controles, números decisivos ou limitações.**

## 6. Acoes executadas nesta passada

1. `respostas_AE.md` foi sincronizado com AE-01 `closed` e passou a apontar
   para a resposta canonica em `respostas_R3.md`.
2. A notacao matematica de R3-P11 e R3-P14 foi corrigida sem alterar o
   conteudo cientifico.
3. Referencias internas desnecessarias foram removidas de R3-A04, R3-P08,
   R3-P15, R3-P18 e de respostas selecionadas do Revisor 1.
4. A legenda da Figura 2 passou a declarar que os canais desenhados sao
   ilustrativos e variam por benchmark; tambem registra que GR e excluido do
   benchmark cross-well principal.
5. A ultima frase interpretativa do Abstract agora limita explicitamente os
   ganhos ao comparador AE direto e rejeita superioridade geral.
6. R1-m05 permanece `pending`. Nenhuma tag, release, migracao de scripts ou
   alteracao de codigo de reproducao foi realizada.

## 7. Verificacao final

- O manuscrito foi recompilado com `latexmk` apos as duas alteracoes textuais.
- A compilacao terminou com sucesso e o PDF permanece com 90 paginas.
- Nao ha referencias ou citacoes indefinidas no log.
- A Figura 1 permanece na pagina 12 e a Figura 2 na pagina 14.
- Permanecem apenas avisos nao fatais de caixas subpreenchidas e de versao de
  alguns PDFs incorporados; nenhum foi introduzido pelas correcoes desta
  passada.
- Nenhum numero, tabela, resultado experimental ou arquivo de dados foi
  alterado.

## 8. Estado recomendado

- **Ciência e resultados:** preservar o manuscrito atual; nenhuma evidência nesta auditoria exige novos experimentos.
- **Respostas dos revisores:** conteudo aprovado como **base** da carta final,
  com os ajustes localizados desta auditoria incorporados.
- **Editor Associado:** AE-01 permanece `closed`; a divergencia entre os
  arquivos de resposta foi corrigida.
- **R1-m05:** `pending`, deliberadamente adiada, **não bloqueando a revisão editorial ora solicitada**.
- **Controle de versao:** registrar apenas os arquivos desta passada na branch
  de revisao, sem incluir mudancas preexistentes de outras frentes.

**Conclusao:** o conjunto esta consistente como base da carta de resposta. A
preparacao da tag e do pacote de reproducao permanece uma frente separada.
