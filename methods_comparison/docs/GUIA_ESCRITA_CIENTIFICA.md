# Guia de escrita cientifica -- methods_comparison

Este documento define o padrao de escrita para relatorios LaTeX, notas de
planejamento (`.md`), READMEs de artefatos e metadados (`metrics.json`,
`MANIFEST.txt`) do projeto **methods_comparison**, em especial o poco 861
(MOGNO).

Referencias canonicas:

- LaTeX Etapa 1: `methods_comparison/latex/poco861_etapa1_ml.tex`
- LaTeX Etapa 2: `methods_comparison/latex/poco861_etapa2_dem_sc.tex`
- Planejamento: `methods_comparison/planning/etapa*.md`

---

## 1. Principios gerais

### 1.1 Publico e tom

- Escreva para um petrofisico ou geofisico que **nao** leu o codigo.
- Use prosa completa (frases com sujeito, verbo e conclusao), nao telegrafe.
- Explique **o que** foi feito, **por que** importa e **como interpretar** cada
  numero ou figura.
- Evite jargao de implementacao no corpo do texto (nomes de funcoes, flags de
  CLI, caminhos de pasta) salvo na secao de reprodutibilidade ou em notas
  tecnicas breves.

### 1.2 Idioma e acentuacao

| Tipo de arquivo | Idioma | Acentos |
|-----------------|--------|---------|
| Relatorios `.tex` em `methods_comparison/latex/` | Portugues BR | **Obrigatorios** (corpo, apendices, legendas) |
| Planejamento e README `.md` | Portugues BR | Sim |
| Codigo Python, JSON, MANIFEST | Ingles ou ASCII | Apenas ASCII |
| Comentarios em codigo | Ingles ou ASCII | Apenas ASCII |

**LaTeX em portugues:** o preambulo padrao usa `inputenc` UTF-8 e `babel` com
opcao `brazil`. Escreva **diretamente em UTF-8** no corpo do `.tex`: poço,
arcabouço, não, transparência, calibração, equação, validação, etc.

**Padrao preferido:** caracteres acentuados normais no arquivo fonte (o PDF
herda automaticamente). Nao use `po\c{c}o`, `n\~ao`, `num\'erico` em texto novo.

**Proibido:** omitir acentos (`numerico`, `ate`, `definicao`).
Labels `\label{...}`, comentarios `%` e caminhos de figura podem ficar em ASCII.
No modo matematico, evite acentos nas variaveis; explique simbolos em portugues
no paragrafo adjacente (com UTF-8).

### 1.3 Honestidade metodologica

- Distinga sempre **validacao in-sample** de **validacao honesta** (LOO, blocos
  de profundidade, perfil independente).
- Reporte viés (bias) junto com MAPE/RMSE: um MAPE moderado com viés alto tem
  interpretacao diferente de erro simetrico.
- Quando um resultado for fraco, diga explicitamente (ex.: HFU4 sem plugs, FZI
  com R2 negativo).
- Nao use divisao aleatoria 80/20 como metrica oficial em dados de poco; use-a
  apenas como ilustracao, com ressalva.

---

## 2. Estrutura de um relatorio LaTeX (poco 861)

### 2.1 Esqueleto obrigatorio

1. **Titulo** -- intervalo, metodo e tipo de validacao.
2. **Resumo (abstract)** -- contexto, metodo em uma frase, resultados numericos
   principais, limitacao e proximo passo (um paragrafo denso, sem bullets).
3. **Introducao**
   - Contexto do poco e do projeto.
   - Grandezas ou propriedades avaliadas (lista com significado, nao so sigla).
   - Organizacao do relatorio (mapa das secoes).
4. **Dados** -- fontes, amostras, profundidade, lacunas; tabela seguida de
   prosa que explica o conteudo.
5. **Protocolo / Metodologia** -- como o experimento foi conduzido; regras de
   validacao; como ler metricas (MAPE, RMSE, R2, vies).
6. **Resultados** -- uma subsecao por fase ou pergunta; tabela e/ou figura
   introduzida no texto e discutida em prosa continua (sem cabecalho
   ``Interpretacao da...'').
7. **Discussao** -- subsecoes tematicas em prosa (nao lista de bullets).
8. **Conclusoes** -- lista numerada com rotulo em negrito e frase completa.
9. **Reprodutibilidade** -- onde estao artefatos; comandos ou referencia ao
   repositorio (prosa, nao dump de terminal).

### 2.2 Interpretacao integrada ao texto

Nao use cabecalhos do tipo ``\textbf{Interpretacao da Tabela...}'' ou
``\textbf{Interpretacao da Figura...}''. A leitura petrofisica deve fluir no
proprio paragrafo, citando a tabela ou figura no meio da frase.

Padrao recomendado:

```latex
A Tabela~\ref{tab:exemplo} resume ... (contexto antes ou depois do bloco).

\begin{table}[H]
  ...
\end{table}

Na Tabela~\ref{tab:exemplo}, o vi\'es de $+0{,}6$~km/s indica que ...
Esse patamar orienta a Se\c{c}\~ao~\ref{sec:proxima}.
```

Para figuras:

```latex
A Figura~\ref{fig:exemplo} exibe ... (introducao opcional antes do bloco).

\begin{figure}[H]
  ...
\end{figure}

No painel (a), a trilha de profundidade mostra ...; no painel (b), os pontos
pr\'oximos \`a diagonal 1:1 confirmam ...
```

O paragrafo deve responder:

1. O que o leitor deve ver de imediato (tendencia, ordem de grandeza).
2. O que isso significa geologica ou petrofisicamente.
3. Limitacao ou cautela (onde nao generalizar).

Para figuras com dois paineis, use `\emph{Painel (a)---...}` e
`\emph{Painel (b)---...}` dentro da prosa, nao como secao separada rotulada.

### 2.3 Notacao matematica

Defina comandos no preambulo e use-os de forma consistente:

```latex
\newcommand{\philab}{\ensuremath{\phi_{\mathrm{lab}}}}
\newcommand{\phiND}{\ensuremath{\phi_{\mathrm{ND}}}}
\newcommand{\phiS}{\ensuremath{\phi_{\mathrm{S}}}}
\newcommand{\Vp}{\ensuremath{V_{\mathrm{p}}}}
\newcommand{\Vs}{\ensuremath{V_{\mathrm{s}}}}
```

Regras:

- Primeira mencao de sigla: nome por extenso + sigla (ex.: *unidade
  hidrodinamica de fluxo (HFU)*).
- DEM e SC: escreva *Differential Effective Medium (DEM)* na primeira ocorrencia;
  depois use DEM/SC em texto corrido (nao precisa `\textsc` excessivo).
- HFU, FZI: texto corrido ou *HFU de laboratorio*; evite HFU como codigo.
- Numeros decimais no texto: vírgula (`0,15`, `5205,91`); no modo matematico
  use `{,}` (`$r \approx 0{,}94$`).
- Unidades: km/s, MPa, m, pu (porosidade) explicitas na tabela ou na primeira
  mencao.

### 2.4 Tabelas

- Use `booktabs` (`\toprule`, `\midrule`, `\bottomrule`).
- Legenda acima da tabela: o que esta sendo comparado, condicoes (pressao,
  eixo, numero de amostras).
- Colunas com significado claro (evite abreviacoes obscuras sem nota de rodape).
- Valores numericos alinhados com `metrics.json` do pipeline (mesma rodada);
  nao arredonde de forma diferente entre LaTeX, JSON e README.

Exemplo de legenda boa:

> *Validacao DEM versus laboratorio (dez plugs CT, 22,1 MPa, eixo Z, sem
> calibracao).*

Exemplo de legenda fraca:

> *Resultados.*

### 2.5 Figuras

- Copie PNGs para `methods_comparison/latex/figures/` com prefixo por etapa
  (`fig2_` para Etapa 2).
- Largura tipica: `0.72`--`0.85\linewidth` para uma figura; `0.48\linewidth` em
  `subfigure` lado a lado.
- Legenda descreve **conteudo e leitura** (o que esta no eixo, o que significa
  afastar-se da diagonal 1:1), nao so o nome do script.
- Referencie todas as figuras no texto e explique o conteudo em prosa
  continua (antes, depois ou nos dois lados do bloco).

### 2.6 Pacotes e compilacao

Preambulo alinhado aos relatorios existentes:

- `babel` com opcao `brazil`
- `booktabs`, `graphicx`, `subcaption`, `siunitx`, `float` (`[H]` para tabelas
  ancoradas)
- Compilar duas vezes: `make etapa1` / `make etapa2` em `methods_comparison/latex/`

### 2.7 Labels

- Prefixos: `sec:`, `tab:`, `fig:`, `eq:`.
- Labels unicos e estaveis (nao renomear sem atualizar referencias).
- Referencias cruzadas com `Se\c{c}\~ao~\ref{...}`, `Tabela~\ref{...}`,
  `Figura~\ref{...}`, `Equacao~\eqref{...}`.

### 2.8 Equacoes

- Toda equacao exibida deve ter `\label{eq:...}`.
- Cite a equacao no texto antes ou depois de usa-la (ex.: *conforme a
  Equacao~\eqref{eq:velocidades}*).
- Opcional no preambulo: `\numberwithin{equation}{section}` para numeracao
  por secao (Etapa 2).
- Explique em prosa cada simbolo na primeira equacao da secao, nao apenas
  liste a formula.

### 2.9 Caminhos e nomes de arquivo no corpo do texto

**Proibido no corpo narrativo** (introducao, dados, resultados, discussao):

- `\texttt{processed/...}`, `\texttt{run_861\_...}`, `\texttt{.py}`, `.json`.
- Flags de linha de comando e nomes de frames DLIS crus sem contexto.

**Permitido:**

- Nome de ferramenta ou produto comercial (ex.: planilha ROCKPHYS, perfil DSI).
- Secao de reprodutibilidade em **prosa** (como Etapa 1): *pipelines
  automatizados*, *artefatos no repositorio*, sem dump de caminhos.
- Nota de rodape breve com data da rodada (`generated_utc`) se necessario para
  rastreio.

Substitua caminho por funcao:

| Evitar | Preferir |
|--------|----------|
| `dem_sc_861_core.py` | modulo de meio efetivo DEM/SC |
| `ait_pex_dsi.dlis` | arquivo sonic DSI do poço |
| `metrics.json` | metricas exportadas pelo pipeline |
| `ct_ar_mean` | razao de aspecto mediana do microCT |

### 2.10 Citacao obrigatoria no fluxo do texto

Padrao de paragrafo de resultado:

1. Frase de contexto (*A Tabela X resume...* / *A Figura Y exibe...*).
2. Tabela ou figura.
3. Prosa narrativa que interpreta o conteudo (sem cabecalho **Interpretacao**).
4. Ligacao para a proxima secao (*Esse patamar orienta a Secao...*).

Nenhuma tabela ou figura deve aparecer sem mencao explicita no texto com
`\ref`.

### 2.11 Tabela-sintese de validacao

Quando houver varias etapas de validacao (linha de base, calibracao, LOO,
perfil), inclua uma **tabela-sintese** (ex.: `tab:validation_summary`) que
reuna MAPE, RMSE e viés por escala. Facilita a leitura e evita que o leitor
tenha de montar a comparacao mentalmente.

---

## 3. Escrita em Markdown (planning e README)

Os arquivos `.md` de planejamento e de pastas `processed/` devem seguir a **mesma
logica narrativa** do LaTeX, com adaptacoes de formato.

### 3.1 Planejamento (`planning/etapa*.md`)

Estrutura recomendada:

1. Status (tabela curta: poco, intervalo, dependencias).
2. Objetivo (prosa + lista numerada de entregas).
3. Dados e decisoes herdadas da etapa anterior.
4. Protocolo / fases do pipeline.
5. Metricas esperadas e criterios de aceite.
6. Lacunas e proximos passos.

Use tabelas para decisoes e metricas; apos tabelas criticas, paragrafo
**Interpretacao:** em prosa (como no LaTeX).

### 3.2 README de artefatos (`processed/**/README.md`)

Cada pasta de run deve conter:

```markdown
# Titulo descritivo -- Well 861

Uma frase sobre o proposito deste run.

Planning: `caminho/para/planning.md`

## Pre-requisitos

1. Run anterior X
2. Dados Y

## Regenerar

(comandos bash completos)

## Layout

(arvore de arquivos)

## Interpretacao (rodada de producao)

| Metrica | Valor | Nota |
|---------|-------|------|

Paragrafo em prosa explicando viés, limitacoes e proximo passo.

## Proximos passos

(lista curta, acionavel)
```

Regras:

- Secao **Interpretacao** obrigatoria quando houver `metrics.json`.
- Valores copiados de `metrics.json` da mesma rodada documentada em
  `MANIFEST.txt` (`generated_utc`).
- Evite README apenas em ingles se o relatorio LaTeX correspondente for em
  portugues; preferir PT-BR para consistencia do poco 861.

### 3.3 O que nao fazer em `.md`

- README so com arvore de diretorios e comandos, sem interpretacao.
- Tabela de numeros sem coluna *Nota* ou paragrafo explicativo.
- Misturar resultados de rodadas diferentes sem indicar `generated_utc`.

---

## 4. JSON e MANIFEST (metadados de pipeline)

### 4.1 `metrics.json`

- Chaves em `snake_case`, ASCII.
- Incluir sempre: `generated_utc`, metricas principais, `n_samples` ou
  equivalente, flags (`smoke`, `robust`) quando aplicavel.
- Valores numericos em JSON com ponto decimal (padrao JSON); no LaTeX/README
  converter para virgula ao exibir.
- Nao duplicar logica: o JSON e a fonte de verdade; LaTeX e README citam os
  mesmos numeros.

Exemplo de campos minimos (validacao):

```json
{
  "generated_utc": "2026-06-15T20:39:53Z",
  "n_matched_vp": 87,
  "mape_vp_pct": 16.21,
  "bias_vp_km_s": 0.616,
  "rmse_vp_km_s": 0.853
}
```

### 4.2 `MANIFEST.txt`

- ASCII.
- Cabecalho: titulo do run, `generated_utc`, dependencias.
- Lista de arquivos gerados.
- Proximo comando sugerido (uma linha).
- Pode incluir resumo de metricas-chave para leitura rapida sem abrir JSON.

### 4.3 Rastreabilidade

Ao atualizar o relatorio LaTeX, atualize em conjunto (mesma sessao de trabalho):

1. `metrics.json` / `metrics_validation.json`
2. `MANIFEST.txt`
3. README da pasta do run
4. Tabelas e paragrafos correspondentes no `.tex`
5. Figuras em `latex/figures/` (copia dos PNGs do pipeline)

Indique no texto ou no MANIFEST o `generated_utc` da rodada citada.

---

## 5. Metricas: como descrever em prosa

| Metrica | Como escrever para o leitor |
|---------|----------------------------|
| MAPE | Erro percentual medio absoluto; 15% = erro tipico de 15% em relacao ao observado. |
| RMSE | Erro tipico na unidade da grandeza (km/s); penaliza outliers. |
| Viés | Diferenca media predito - observado; positivo = superestima. |
| R2 | Fracao da variancia explicada; negativo = pior que prever a media. |
| Pearson r | Forca de associacao linear; nao implica causalidade. |
| LOO / LOPO | Validacao honesta excluindo uma amostra por vez. |

Sempre que citar MAPE, prefira acrescentar viés ou uma frase sobre condicoes
(lab seco vs sonic in-situ, in-sample vs LOO).

---

## 6. Checklist antes de fechar uma etapa

### LaTeX

- [ ] Abstract com numeros principais e ressalva metodologica.
- [ ] Cada tabela/figura importante tem prosa interpretativa no texto (sem
  cabecalho **Interpretacao**).
- [ ] Siglas definidas na primeira ocorrencia.
- [ ] Numeros consistentes com `metrics.json` da rodada citada.
- [ ] Figuras copiadas para `latex/figures/` e compilacao `make etapaN` sem erro.
- [ ] Discussao em prosa por eixos tematicos, nao so bullets.
- [ ] Equacoes numeradas com `\label` e citadas no texto.
- [ ] Nenhum caminho de arquivo no corpo (so prosa na reprodutibilidade).
- [ ] Tabela-sintese quando houver multiplas validacoes.
- [ ] Conclusoes com recomendacao acionavel (proxima fase).

- [ ] README com secao **Interpretacao** e `generated_utc`.
- [ ] Planning atualizado (lacunas fechadas, checklist).
- [ ] JSON com chaves ASCII e timestamp.
- [ ] MANIFEST lista outputs e proximo passo.

---

## 7. Anti-padroes (evitar)

| Anti-padrao | Alternativa |
|-------------|-------------|
| "MAPE 16%, ver metrics.json" | Paragrafo: MAPE, viés, condicoes, comparacao com LOO |
| Lista de bullets na Discussao | Subsecoes com paragrafos |
| `\texttt{run_861_...}` no meio do texto | "pipeline de calibracao por HFU" |
| Figura sem mencao no texto | Referencia `Figura~\ref{...}` + interpretacao |
| Tabela sem legenda contextual | Legenda com N, pressao, eixo, filtro |
| Resultado in-sample como validacao final | LOO ou validacao independente (perfil, blocos) |
| README so em ingles para entrega PT | README em PT-BR alinhado ao LaTeX |

---

## 8. Referencia rapida de arquivos

| Papel | Caminho tipico |
|-------|----------------|
| Relatorio PDF | `methods_comparison/latex/poco861_etapa*.pdf` |
| Figuras para LaTeX | `methods_comparison/latex/figures/` |
| Planejamento | `methods_comparison/planning/` |
| Runs processados | `methods_comparison/data/processed/` |
| Scripts | `methods_comparison/scripts/` |
| Este guia | `methods_comparison/docs/GUIA_ESCRITA_CIENTIFICA.md` |

---

## 9. Relacao com outros documentos do repositorio

- `cursor/regras.md` -- regras gerais do Cursor; a secao **Producao de texto
  LaTeX** remete a este guia para **todo** `.tex` em `methods_comparison/latex/`
  (corpo, apendices e relatorios futuros), e separa o escopo do artigo em
  `paper/` (ingles).
- `methods_comparison/latex/README.md` -- como compilar os relatorios.

Ao escrever qualquer conteudo LaTeX em `methods_comparison/latex/`, **priorize
este guia** e os relatorios das Etapas 1--3 como modelo de estilo.

### 2.12 Apendices e notas tecnicas

Apendices **nao** sao excecao ao padrao de escrita: o leitor continua sendo o
petrofisico que nao leu o codigo.

- Explicar motivacao, equacoes e encadeamento fisico em prosa didatica.
- Citacao obrigatoria de tabelas e equacoes no texto (`Tabela~\ref`,
  `Equacao~\eqref`).
- Evitar mapas funcao-a-funcao com nomes Python; preferir tabelas por *etapa
  fisica* (entrada, transformacao, saida).
- Cross-checks e limitacoes numericas entram como discussao metodologica, nao
  como dump de biblioteca ou script.
- Mesmas regras de notacao, virgula decimal e honestidade (in-sample vs LOO).

### 2.13 Acentuacao em portugues (lembrete)

Escreva com **UTF-8 direto** no `.tex`: poço, arcabouço, não, transparência,
definição, calibração, equação. O preambulo ja carrega `inputenc` + `babel`.

Nao use comandos legados (`po\c{c}o`, `n\~ao`) em texto novo. Titulos de
`\section`, legendas e apendices seguem a mesma regra.

---
