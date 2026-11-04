# Auditoria de montagem da carta consolidada

Manuscrito: CAGEO-D-26-01398

Data: 2026-11-04

## Etapa A - versionamento documental

- Branch: `revision/cageo-d-26-01398-checkpoint`.
- Commit-base verificado: `929dc570cb0a7d298a02960408b2838f819e4fbe`.
- Commit documental criado:
  `b580c556a1b3db84b8b9e48ebe8c42d1f16d0af0`.
- Mensagem: `revision: finalize reviewer-response editorial corrections`.
- Push confirmado em
  `origin/revision/cageo-d-26-01398-checkpoint`.
- Diff do commit: 248 insercoes e 32 remocoes em sete arquivos.

Arquivos incluidos:

- `docs/revision/auditoria_editorial_respostas.md`
- `docs/revision/respostas_AE.md`
- `docs/revision/respostas_R1.md`
- `docs/revision/respostas_R2.md`
- `docs/revision/respostas_R3.md`
- `sections/09_appendix.tex`
- `main.pdf`

O commit inclui a nomenclatura publica `Lapa field` no lugar do nome
interno da planilha. `Auddys Alvarez` foi preservado em `references.bib`
porque e o nome correto do autor.

O inventario confirma que esse commit nao e exclusivamente Markdown: ele
inclui `sections/09_appendix.tex` e o PDF compilado. O texto extraido do PDF
confirma a mesma descricao publica `Lapa field dataset of Alvarez et al.
(2025)` presente no apendice.

Nao foram incluidos `status.md`, `ae01_matriz_conformidade.md`,
`AUDIT_LAPA_FEASIBILITY_20261019.md` ou
`conferencia_independente_anexos_2026-11-03.md`.

## Etapa B - montagem

- Carta criada em `docs/revision/response_to_reviewers_consolidated.md`.
- Comentarios incorporados: 52.
- Respostas finalizadas incorporadas: 51.
- Marcadores internos de pendencia: 1.
- Identificadores faltantes: nenhum.
- Identificadores duplicados: nenhum.
- AE-01 aparece uma unica vez.
- R1-m05 contem somente o marcador interno de acao pendente; a resposta
  provisoria dos arquivos de trabalho nao foi apresentada como definitiva.
- Os 52 comentarios foram comparados literalmente com os blocos atuais das
  respostas.
- As 51 respostas finalizadas foram comparadas com as respostas canonicas.

## Correcoes editoriais durante a montagem

- R3-05: removida da resposta canonica a referencia administrativa a
  R3-P15.
- R3-06: substituida a referencia administrativa a R3-04 por uma descricao
  direta do sweep de ruido e de sua falta de calibracao de campo.
- R1-M11: na carta, `Appendix, subsection Laboratory porosity on a cored
  interval` foi corrigido para `Appendix, section Laboratory porosity on a
  cored interval`, em conformidade com o LaTeX.
- O ponteiro de `respostas_AE.md` nao foi usado como resposta. O texto
  canonico de AE-01 foi inserido a partir de `respostas_R3.md`.

## Conferencias cientificas

Foram preservados:

- tres pocos de desenvolvimento e um poco F03-4 retido;
- alvo cross-well derivado por Larionov, com gamma ray deliberadamente
  excluido;
- observacoes esparsas como subamostras das curvas processadas;
- CLP-CSGM Ridge com menor RMSE medio que o AE direto em sete de nove
  configuracoes;
- interpolacao com menor RMSE que CLP-CSGM nas nove configuracoes no ruido
  de referencia 0.01;
- ausencia de calibracao instrumental dos niveis maiores do sweep;
- porosidade F03-4 processada e derivada de densidade;
- suplemento laboratorial limitado, sem verificacao dos registros primarios,
  com melhor media agregada do AE direto e janelas sobrepostas;
- distincao entre prior condicionado, decoder, fidelidade as observacoes,
  penalidade em torno de `z_0`, MAP/Tikhonov, existencia, solucao numerica e
  recuperacao local condicional;
- ausencia de verificacao de S-REC no decoder treinado;
- distincao entre o identificador `30dB`, a SNR mencionada pela fonte e o
  ruido gaussiano acrescentado;
- ChatGPT para traducao, edicao, reducao de extensao e esquemas, e Grok 4.7
  para refatoracao de codigo preexistente, sob responsabilidade dos autores.

As referencias citadas as Figuras 1, 2, 3, 8 e 24 foram verificadas. Nao foi
encontrada divergencia quantitativa material com o manuscrito.

Na segunda leitura realizada durante a montagem, nao foi encontrado problema
material; a correcao de `subsection` para `section` em R1-M11 foi aplicada na
carta.

## Ajustes apos o parecer independente

As recomendacoes fundamentadas do parecer independente de 2026-11-04 foram
aplicadas aos arquivos canonicos de respostas e a carta:

- R1-M07 passou a explicitar que a Proposition 3 da versao avaliada
  corresponde a Proposition 1 do manuscrito reorganizado;
- R1-M04 e R2-05 passaram a declarar que os intervalos sao individuais por
  configuracao e nao foram ajustados para comparacoes multiplas simultaneas;
- R1-M06 passou a apresentar as medias como 0.0286 e 0.0254, preservando a
  reducao de 11.3% com a ressalva de que ela foi calculada a partir dos
  valores nao arredondados;
- a notacao matematica foi normalizada em R1-M04, R1-M06, R3-P06 e R3-P15;
- os artefatos de extracao presentes nos comentarios originais foram
  preservados, pois os arquivos-fonte dos pareceres nao estao disponiveis
  para confirmar correcoes literais.

Esses ajustes nao alteram resultados, metodologia ou interpretacao
cientifica.

## Fundamentacao do ruido sintetico

A justificativa documental do ruido foi ampliada sem atribuir calibracao de
campo aos niveis testados:

- o manuscrito passou a citar Abdelrahman e Szabo (2024) como precedente para
  contaminacao Gaussiana em um teste sintetico de inversao de perfis;
- Adjei et al. (2019) foi citado para documentar a propagacao probabilistica
  de incerteza em estimativas de volume de argila baseadas em gamma ray;
- Experimental setup e Discussion agora distinguem explicitamente o
  precedente para a forma do teste da ausencia de justificativa instrumental
  para suas amplitudes;
- R3-04 e R3-P09, bem como suas copias na carta, registram que o estudo nao
  estimou incerteza especifica da ferramenta nem a propagou pela relacao de
  Larionov.

Nenhum nivel de ruido foi reinterpretado como representativo de uma
incerteza instrumental real.

O manuscrito foi recompilado com `latexmk`. A compilacao terminou sem
referencias indefinidas e o PDF permanece com 90 paginas.

## Metadados e naturalidade

As respostas finalizadas nao contem `closed`, `text_done`, nomes de branch,
commits, caminhos de scripts, referencias ao Cursor ou cronologia de
auditoria. Os IDs permanecem apenas nos titulos de trabalho da carta.

Os comentarios originais foram preservados sem correcao silenciosa. Antes da
submissao, os autores devem conferir no DOCX/PDF original artefatos de
extracao como `VcV_c` em R1-M01 e grafias presentes no comentario AE-01.

## Pendencia

R1-m05 permanece pendente. A carta nao esta pronta para submissao enquanto
contiver:

`AUTHOR ACTION PENDING -- Frozen code release to be completed before
submission.`

Nenhuma tag, release, migracao de scripts ou merge foi realizado.

## Estado de versionamento da carta

A carta consolidada e este relatorio foram aprovados para versionamento local
apos a segunda avaliacao independente. Esse versionamento nao resolve
R1-m05 nem torna a carta pronta para submissao. O envio ao remoto deve ser
registrado separadamente quando realizado.
