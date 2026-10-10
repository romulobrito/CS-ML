# CLP-CSGM — Consolidação dos pareceres e trechos do manuscrito

**Manuscrito:** CAGEO-D-26-01398  
**Título:** Conditional Latent-Prior Generative Compressed Sensing with Application to Petrophysical Target-Log Reconstruction  
**Decisão:** revisão maior, 6 de outubro de 2026  
**Prazo informado:** 5 de novembro de 2026  
**Levantamento:** 6 de outubro de 2026

## Escopo e forma de leitura

Este arquivo reúne os textos dos três revisores, as 19 anotações textuais do revisor 3 e os assuntos comuns. É um mapa de revisão, não uma carta de respostas nem uma declaração de alterações já executadas. A base é a versão submetida incluída no ZIP; o projeto completo recebido posteriormente foi usado para localizar os trechos no LaTeX e consultar o material suplementar pertinente. Não foi feita uma comparação exaustiva entre versões nem uma auditoria dos resultados.

**Fontes conferidas:**

- `CAGEO-D-26-01398-reviews.zip` → `CAGEO-D-26-01398-reviews.pdf`: 3 páginas, com os pareceres 2 e 3 e referência ao anexo do revisor 1.
- `Initial submission/Reviewer 1/CAGEO-D-26-01398_Comments.docx`: parecer do revisor 1.
- `Initial submission/Reviewer 3/CAGEO-D-26-01398-with-comments.pdf`: manuscrito de 37 páginas, com 19 anotações textuais e uma anotação sem texto.
- Carta editorial de 6 de outubro enviada na conversa: síntese do editor associado, registrada separadamente ao fim.

**Identificadores:** `R1-M01…M11` = comentários maiores do revisor 1; `R1-m01…m05` = menores. A numeração foi criada aqui, pois o DOCX não os enumera. `R2-01…05` e `R3-01…06` seguem os números dos pareceres. `R3-A01…A05` identifica os cinco comentários adicionais; `R3-P01…P19` identifica as anotações do PDF na ordem de extração. Esses dois últimos conjuntos foram numerados aqui.

**Fidelidade:** os blocos “Original do revisor” preservam o inglês recebido, inclusive erros de digitação. As quebras de linha foram normalizadas nos pareceres em PDF. A extração do DOCX conserva algumas duplicações de notação (por exemplo, `VcV_c` e `n=3n=3`). Traduções e sugestões são apresentadas fora das citações. Os excertos do manuscrito removem números de linha isolados e normalizam espaços; símbolos extraídos de PDF podem ter aparência diferente da composição original. `[…]` indica corte do excerto.

**Paginação:** todos os números de página do manuscrito referem-se ao arquivo PDF, contando a capa editorial, e não à numeração impressa do artigo. O resumo também aparece na capa, onde se concentram as primeiras anotações. O PDF de 37 páginas termina com um link para o suplemento. Nesta atualização, o conteúdo dos apêndices está disponível no arquivo `sections/09_appendix.tex` do ZIP e foi consultado nos pontos relacionados aos pareceres.

## Localização no projeto LaTeX completo

**Fonte desta atualização:** `paper_clp_csgm.zip`, recebido em 6 de outubro de 2026. Foram conferidos `main.tex`, as nove seções, os trechos dos apêndices relacionados aos pedidos e a tabela de implementação. O `main.tex` do ZIP é idêntico ao arquivo enviado isoladamente.

Cada um dos 51 IDs abaixo agora possui ao menos um trecho confirmado no fonte, com arquivo e linhas. Isso identifica onde o assunto aparece ou deve ser revisto; não significa que os 51 pedidos já estejam atendidos. Pedidos de novos testes apontam para o protocolo e os resultados relacionados, pois o experimento solicitado pode ainda não existir.

**Convenções de localização:**

- Os caminhos são relativos à raiz `paper_clp_csgm/` do ZIP. As linhas contam todas as linhas do arquivo, inclusive comentários e linhas em branco; não são os números impressos pelo pacote `lineno`.
- Os excertos LaTeX são literais. O nome da seção/subseção e os labels ajudam a encontrar o trecho mesmo depois de alterações que mudem a numeração das linhas.
- Para anotações no PDF, a página/destaque é a indicação original do revisor; a correspondência com o fonte foi identificada nesta análise. Para pedidos gerais, as passagens são pontos de revisão pertinentes, não indicações de linha atribuídas ao revisor.
- As páginas citadas continuam sendo as do PDF submetido de 37 páginas. Não devem ser transferidas automaticamente ao `main.pdf` do ZIP, que inclui material adicional.
- As chamadas de figuras identificam os arquivos gráficos; esta atualização não constitui uma nova avaliação visual de todos os gráficos nem uma execução dos experimentos.

**Conteúdo dos apêndices relevante para a resposta:** já existem o diagnóstico de separabilidade (`app:separability`), o teste com/sem GR e embargo (`app:crosswell_gr_embargo`), custos computacionais (`app:runtime`) e o caso Lapa (`app:lapa_case`). Suas localizações aparecem nos itens correspondentes. A existência desses resultados não demonstra, sozinha, atendimento integral aos pedidos.

## Mapa dos pontos comuns

Os temas T01–T14 agrupam questões relacionadas. Uma mesma alteração pode atender a vários IDs, mas cada comentário deve receber sua própria resposta com referência à mudança. “Tema comum” não significa que os pedidos sejam idênticos.

| Tema | Revisores | IDs relacionados | Ação consolidada sugerida |
|---|---|---|---|
| **T01 — Origem derivada dos alvos e validação independente** | R1, R2, R3 | `R1-M01`, `R1-M11`, `R2-02`, `R3-02`, `R3-P01`, `R3-P17`, `R3-P19` | Explicitar a origem e separar teste controlado de evidência independente; avaliar o caso Lapa e uma extensão viável. |
| **T02 — Aplicabilidade, contexto incompleto e limites das conclusões** | R1, R2, R3 | `R1-M01`, `R1-M02`, `R1-M05`, `R1-M09`, `R1-m04`, `R2-02`, `R2-03`, `R3-02`, `R3-P05`, `R3-P19` | Delimitar os cenários e declarar ganhos condicionais; justificar exclusão de GR/AC e alcance entre poços. |
| **T03 — Quantidade de poços e suporte amostral** | R1 | `R1-M02`, `R1-M03`, `R1-M11` | Avaliar reamostragem e outros poços; reconhecer que mais sementes não ampliam diversidade geológica. |
| **T04 — Número de sementes e estabilidade estatística** | R1, R2 | `R1-M03`, `R1-M04`, `R2-04`, `R2-05` | Ampliar realizações pareadas e atualizar incerteza; distinguir sementes de treinamento e máscaras. |
| **T05 — Posição das observações e informação de coordenadas** | R1, R2 | `R1-M08`, `R2-04` | Executar sensibilidade às posições e comparador com máscara explícita: são duas verificações complementares. |
| **T06 — Comparadores simples e ganho frente a alternativas** | R1, R2, R3 | `R1-M08`, `R1-m04`, `R2-03`, `R3-02`, `R3-03`, `R3-06`, `R3-P11`, `R3-P12`, `R3-P19` | Comparar modelos com informação disponível equivalente; incluir Ridge/linear direto e interpolação/suavização. |
| **T07 — Representação latente, prior Ridge/MLP e etapas intermediárias** | R1, R3 | `R1-M06`, `R3-03`, `R3-A05`, `R3-P06`, `R3-P11`, `R3-P12`, `R3-P13` | Explicar papéis, escolha do prior e benefício do espaço latente; mostrar diagnósticos das etapas. |
| **T08 — Clareza, estrutura, símbolos e terminologia** | R1, R2, R3 | `R1-m01`, `R1-m02`, `R2-01`, `R3-01`, `R3-A01`, `R3-A02`, `R3-A04`, `R3-A05`, `R3-P01`, `R3-P02`, `R3-P03`, `R3-P04`, `R3-P05`, `R3-P06`, `R3-P07`, `R3-P08`, `R3-P14`, `R3-P17`, `R3-P18` | Reorganizar motivação antes da matemática e padronizar termos e definições. |
| **T09 — Ruído, padronização, covariância e regularização** | R3 | `R3-04`, `R3-P09`, `R3-P10`, `R3-P16` | Documentar escalas, procedimentos e seleção de λ; conferir coerência da interpretação probabilística. |
| **T10 — Problema inverso, função da teoria e garantias condicionais** | R1, R3 | `R1-M07`, `R3-05`, `R3-P02`, `R3-P15` | Explicar o enquadramento, o que as proposições demonstram e quais hipóteses não foram verificadas. |
| **T11 — Figuras e distinção visual dos objetos** | R1, R3 | `R1-m03`, `R3-06`, `R3-A02`, `R3-A03`, `R3-A04`, `R3-P08`, `R3-P14`, `R3-P18`, `R3-P19` | Redesenhar diagramas, ampliar rótulos e tornar comparáveis escalas e resultados. |
| **T12 — Organização dos resultados e custo-benefício** | R1, R3 | `R1-M05`, `R3-06` | Apresentar quando ajuda, quanto ajuda, quando perde e quando o custo adicional é justificável. |
| **T13 — Posicionamento na literatura recente** | R1 | `R1-M10` | Ler e avaliar os estudos indicados; incorporar apenas referências pertinentes. |
| **T14 — Versão reproduzível do código** | R1 | `R1-m05` | Identificar release/tag correspondente à versão avaliada. |

**Sobreposições que merecem atenção:**

- **R1-M01 e R2-02:** pedem explicitação da origem derivada e das limitações. **R1-M11** vai além e sugere teste independente.
- **R1-M04 e R2-05:** convergem em mais realizações. **R2-04** focaliza a localização dos pontos; **R1-M08** pede um novo comparador que receba coordenadas/máscara explicitamente.
- **R3-03, R3-P12 e R3-P13:** cobrem benefício do espaço latente e diagnóstico das etapas; o parecer geral acrescenta comparadores lineares e interpolação.
- **R1-M07 e R3-05:** ambos tratam da teoria, mas R1 pede preservar o caráter condicional e R3 pede demonstrar sua utilidade e melhorar a exposição.
- **R1-M06** questiona por que manter Ridge como referência; **R3-A05/R3-P06** pedem primeiro explicar o que Ridge faz.


## Revisor 1 — parecer no DOCX

**Avaliação geral original:**

> The paper has a good core idea, solid experimental discipline, open code, meaningful ablations, and unusually careful limitation statements. My main concern is not the method itself; it is that the empirical evidence is still too narrow to support broader claims beyond the specific benchmark regimes.

**Síntese em português:** O revisor reconhece boa ideia central, disciplina experimental, código aberto, ablações úteis e cuidado com limitações; sua preocupação principal é a evidência empírica restrita.


### R1-M01 — Origem dos alvos e alcance da validação

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Texto e delimitação  
**Temas comuns:** T01, T02

**Original do revisor:**

> The cross-well target VcV_c is Larionov-derived from gamma ray, and the F03-4 porosity target is itself processed/density-derived. The authors acknowledge this limitation, but it is central and should be emphasized earlier, including in the Abstract and Conclusions.

**Tradução para português:**

O alvo Vc entre poços é derivado de GR por Larionov, e o alvo de porosidade de F03-4 também é processado/derivado de densidade. Os autores reconhecem a limitação, mas ela é central e deve ser enfatizada mais cedo, inclusive no resumo e nas conclusões.

**Trecho/local do manuscrito relacionado:**

**Página 14 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> The experiments use the processed F3 well-log curves of Zhang et al. [2025]; no new field data were acquired. We use the FS2–FS6 intervals of F02-1, F03-2, F03-4, and F06-1. Sonic and gamma ray are available in all wells; density is measured in F02-1/F03-2 and predicted in F03-4/F06-1. Porosity is density-derived and Vc is Larionov-derived from gamma ray; further geological and processing details are given in the […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 90–93.**

```latex
We evaluate CLP-CSGM on public Dutch North Sea well-log data reorganized into a
held-out-well clay-volume benchmark and a single-well porosity stress test with
weak dense context. A limited-sample permeability case provides supplementary
evidence. With Ridge as a low-capacity latent-prior reference, CLP-CSGM achieves
```

**`sections/04_data_and_eda.tex`, linhas 10–18.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
The experiments use processed well-log curves described by
\citet{zhang2025rockphysicslogs}; no new field data were acquired. We use
the FS2--FS6 intervals of wells F02-1, F03-2, F03-4, and F06-1 from the F3 block
in the Dutch North Sea. Sonic and gamma-ray logs are available in all four
wells; density is measured in F02-1 and F03-2 and predicted from sonic and gamma
ray in F03-4 and F06-1. The processed porosity target is density-derived, while
clay-volume fraction $V_c$ is obtained from gamma ray through the Larionov
relation. Geological, acquisition, and rock-physics details are provided in the
source study.
```

**`sections/08_conclusion.tex`, linhas 22–26.**

Contexto no fonte: `\section{Conclusion}`.

```latex
CLP-CSGM is a targeted sparse-assimilation method and is not proposed as a
general replacement for direct petrophysical regression. Validation on larger
multi-well datasets, independently calibrated targets, geological labels,
uncertainty estimates, and faster latent-refinement strategies remains
necessary.
```

**Observação para a revisão:** A origem derivada já está explícita na seção de dados. O pedido inclui tornar essa limitação mais visível no resumo e na conclusão; a existência do parágrafo em dados não encerra o comentário.

**Ação sugerida nesta análise:** Explicitar a procedência dos alvos no resumo, no protocolo e na conclusão; distinguir reconstrução de curvas processadas de validação com medições independentes.

**Cruzamento com outros revisores:** `R2-02`, `R2-03`, `R3-02`, `R3-P01`, `R3-P05`, `R3-P17`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-M02 — Poucos poços e transferência operacional

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Texto e delimitação  
**Temas comuns:** T02, T03

**Original do revisor:**

> Only three wells are used for development and one well is held out for testing. That is useful as a controlled test, but it is not enough to support broad statements about cross-well robustness or operational transferability. The language should remain clearly benchmark-specific.

**Tradução para português:**

Apenas três poços são usados no desenvolvimento e um é reservado para teste. Isso é útil como teste controlado, mas insuficiente para sustentar afirmações amplas sobre robustez entre poços ou transferência operacional. A linguagem deve permanecer específica ao benchmark.

**Trecho/local do manuscrito relacionado:**

**Página 16 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> F02-1, F03-2, and F06-1 provide the model-development data, and the dense input u concatenates sonic, bulk density, acoustic impedance, and Vp. As justified in Section 4, gamma ray is excluded because the target is Larionov-derived from that log [Zhang et al., 2025]. Sparse observations follow b = MΩy + η at ρ ∈{0.05, 0.10, 0.20}. Sliding-window steps {8, 16, 32} control training support, and each step–ρ cell uses […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 26–43.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Cross-well clay-volume`.

```latex
The cross-well benchmark reconstructs length-$N=64$ processed $V_c$ windows in
the held-out F03-4 well. F02-1, F03-2, and F06-1 provide the model-development
data, and the dense input $\boldsymbol{u}$ concatenates sonic, bulk density,
acoustic impedance, and $V_p$. As justified in
Section~\ref{sec:data_eda}, gamma ray is excluded because the target is
Larionov-derived from that log
\citep{zhang2025rockphysicslogs}.

Sparse observations follow
$\boldsymbol{b}=M_{\Omega}\boldsymbol{y}+\boldsymbol{\eta}$ at
$\rho\in\{0.05,0.10,0.20\}$. Sliding-window steps
$\{8,16,32\}$ control training support, and each step--$\rho$ cell uses three
paired seed realizations. This is a controlled sparse-assimilation benchmark on
a processed target curve, not recovery against an independently acquired
measurement. Split and sampling details are given in
Section~\ref{subsec:train_val_test_protocol}; GR inclusion and the
embargo-style boundary check are reported in
Appendix~\ref{app:crosswell_gr_embargo}.
```

**`sections/07_discussion.tex`, linhas 69–73.**

Contexto no fonte: `\section{Interpretation, scope, and limitations}`.

```latex
The Lapa permeability case
\citep{alvarez2025nonlinear} extends the evaluation to another lithology and
target property, but its small sample size limits generalization. Larger
multi-well datasets, additional log suites, and facies-conditioned settings
remain necessary for broader validation.
```

**Ação sugerida nesta análise:** Delimitar a conclusão ao poço retido e aos cenários avaliados. Uma ampliação da validação entre poços seria uma melhoria experimental, mas este comentário pede explicitamente cautela na linguagem.

**Cruzamento com outros revisores:** `R2-02`, `R2-03`, `R3-02`, `R3-P05`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-M03 — Poucas janelas e reamostragem

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Experimento sugerido  
**Temas comuns:** T03, T04

**Original do revisor:**

> At the largest window step, the development pool contains only about 27 windows before validation splitting. The performance gains in those low-data settings are interesting, but with such small sample support, repeated resampling or additional held-out wells would make the conclusions much more convincing.

**Tradução para português:**

No maior passo das janelas, o conjunto de desenvolvimento contém apenas cerca de 27 janelas antes da divisão de validação. Os ganhos com poucos dados são interessantes, mas reamostragem repetida ou poços adicionais retidos tornariam as conclusões mais convincentes.

**Trecho/local do manuscrito relacionado:**

**Página 18 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Steps {8, 16, 32} produce approximately {104, 52, 27} windows before this split. Because the validation tail is selected after window construction, boundary windows may overlap when step < N; Appendix B.7 reports an embargo-style check that removes these overlap-prone windows. For F03-4 porosity, depth samples are divided into contiguous training, validation, and test blocks using fractions 60%, 20%, and 20% before […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 76–82.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Training, validation, and test protocol}`.

```latex
In the cross-well benchmark, F02-1, F03-2, and F06-1 form the development pool,
F03-4 is held out entirely for testing, and $10\%$ of the development windows
are reserved for validation. Steps $\{8,16,32\}$ produce approximately
$\{104,52,27\}$ windows before this split. Because the validation tail is
selected after window construction, boundary windows may overlap when
$\mathrm{step}<N$; Appendix~\ref{app:crosswell_gr_embargo} reports an
embargo-style check that removes these overlap-prone windows.
```

**`sections/09_appendix.tex`, linhas 554–571.**

Contexto no fonte: `\section{Supplementary Diagnostics}`; `\subsection{Cross-well Channel-Ablation and Embargo-Style Boundary Check}`.

```latex
This subsection reports a paired robustness check for the cross-well \(V_c\)
step-16 setting. It tests two possible sensitivities: reintroducing GR into
\(u\), relative to the main no-GR protocol in Section~\ref{sec:experiments}, and
removing overlap-prone windows near the train/validation boundary. The latter is
an embargo-style boundary check: a small band of windows adjacent to the boundary
is removed before validation to create a gap between training and validation
windows. We compare four scenarios under the same seeds and measurement-ratio
grid, \(\rho\in\{0.05,0.10,0.20\}\):
\begin{itemize}
  \item No boundary check + GR in \(u\) (stronger dense context reference).
  \item Embargo-style removal (4 boundary windows) + GR in \(u\).
  \item No boundary check + GR excluded from \(u\) (matches the main cross-well \(u\)).
  \item Embargo-style removal (4 boundary windows) + GR excluded from \(u\).
\end{itemize}

Operationally, the embargo-style removal drops four overlap-prone boundary
windows before the validation tail in each training well. In this setup,
\(n_{\mathrm{train}}\) decreases from 52 to 41, while \(n_{\mathrm{val}}\) and
```

**Observação para a revisão:** O apêndice documenta um embargo de fronteira, não uma validação em novos poços nem a reamostragem repetida sugerida. Evitar apresentar esses procedimentos como equivalentes.

**Ação sugerida nesta análise:** Avaliar reamostragem repetida e/ou outros poços retidos, preservando as divisões adequadas. Mais sementes no mesmo poço não equivalem a mais poços independentes.

**Cruzamento com outros revisores:** `R2-04`, `R2-05`. A associação é temática, não equivalência literal dos pedidos.


### R1-M04 — Três sementes e intervalos de incerteza

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Experimento sugerido  
**Temas comuns:** T04

**Original do revisor:**

> The manuscript correctly uses Student-tt intervals and explicitly notes the limitation, but n=3n=3 is still very small. I suggest increasing the seed count if computationally feasible, especially because several intervals include zero and the ranking changes by regime.

**Tradução para português:**

O manuscrito usa corretamente intervalos de Student-t e reconhece a limitação, mas n=3 ainda é muito pequeno. Sugiro aumentar o número de sementes, se computacionalmente viável, especialmente porque vários intervalos incluem zero e a ordenação muda entre regimes.

**Trecho/local do manuscrito relacionado:**

**Página 20 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> All seed-level intervals use the three fixed seeds {7, 23, 41}. Reported 95% half-widths are Student-t intervals with df = nseeds −1 = 2 (critical value t2,0.975 ≈4.303), not normal z = 1.96 intervals. Where CLP-CSGM Ridge is compared with AE [u, b], we also report the paired seed-wise difference ∆RMSE = RMSEAE −RMSECLP, together with its seed-level standard error and Student-t half-width. Positive ∆RMSE favors CLP- […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 174–191.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
\paragraph{Uncertainty summaries.}
All seed-level intervals use the three fixed seeds $\{7,23,41\}$. Reported
$95\%$ half-widths are Student-$t$ intervals with
$\mathrm{df}=n_{\mathrm{seeds}}-1=2$ (critical value
$t_{2,0.975}\approx 4.303$), not normal $z=1.96$ intervals. Where CLP-CSGM
Ridge is compared with AE $[\boldsymbol{u},\boldsymbol{b}]$, we also report the
paired seed-wise difference
\[
\Delta\mathrm{RMSE}
=
\mathrm{RMSE}_{\mathrm{AE}}
-
\mathrm{RMSE}_{\mathrm{CLP}},
\]
together with its seed-level standard error and Student-$t$ half-width.
Positive $\Delta\mathrm{RMSE}$ favors CLP-CSGM Ridge. With only three seeds,
these intervals quantify reproducibility of the paired protocol rather than a
large-sample confidence claim.
```

**Ação sugerida nesta análise:** Ampliar realizações pareadas e atualizar médias, dispersões e diferenças entre métodos. Preservar a ressalva de viabilidade presente no pedido original.

**Cruzamento com outros revisores:** `R2-04`, `R2-05`. A associação é temática, não equivalência literal dos pedidos.


### R1-M05 — Vantagem condicional, sem superioridade geral

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Texto e delimitação  
**Temas comuns:** T02, T12

**Original do revisor:**

> CLP-CSGM outperforms the direct AE baseline in seven of nine cross-well cells, but loses in two cells; in the GR-only benchmark, AE becomes better at higher measurement ratios. This should be framed as a conditional advantage rather than overall superiority. The paper generally does this well, but the Abstract/Highlights could be even more explicit.

**Tradução para português:**

CLP-CSGM supera o AE direto em sete das nove células entre poços, mas perde em duas; no benchmark apenas com GR, AE se torna melhor em frações maiores de medição. Isso deve ser apresentado como vantagem condicional, não superioridade geral. O artigo geralmente já faz isso bem, mas resumo e Highlights podem ser mais explícitos.

**Trecho/local do manuscrito relacionado:**

**Página 21 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Table 4: Cross-well Vc comparison between CLP-CSGM Ridge and AE [u, b]. Values are RMSE mean and standard deviation across seeds; a negative gap favors CLP-CSGM. step n_train_approx rho clp_csgm_rmse_mean clp_csgm_rmse_std ae_ub_rmse_mean ae_ub_rmse_std gap_vs_ae_pct 0.05000 0.04483 0.00524 0.03567 0.00100 25.69014 0.10000 0.03646 0.00194 0.03195 0.00208 14.10258 0.20000 0.02267 0.00381 0.02600 0.00164 -12.83374 […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 94–101.**

```latex
lower mean RMSE than the direct autoencoder baseline in seven of nine cross-well
training-size--measurement-ratio cells. The unweighted mean of the nine
cell-level RMSE values decreases from 0.0350 to 0.0286, an overall reduction of
about 18\%; the largest gains occur in the most data-limited cells. In the
GR-only porosity benchmark, CLP-CSGM has the lowest mean RMSE at the sparsest
ratios, including 0.016 versus 0.018 at
\(\rho=0.20\). The direct autoencoder becomes competitive as more target entries
are observed.
```

**`sections/08_conclusion.tex`, linhas 10–26.**

Contexto no fonte: `\section{Conclusion}`.

```latex
Under the paired F3 protocols, CLP-CSGM Ridge has lower mean RMSE than AE
$[\boldsymbol{u},\boldsymbol{b}]$, the most competitive direct baseline, in
seven of nine cross-well cells. The unweighted mean of the nine cell-level mean
RMSE values decreases from $0.03496$ to $0.02862$, a reduction of approximately
$18.1\%$. In the F03-4 GR-only benchmark, CLP-CSGM Ridge has the lowest mean
RMSE at the two sparsest measurement ratios, while direct AE fusion becomes
competitive as $\rho$ increases. The full Ridge model has lower mean RMSE than
prior-only in every cross-well cell and than measurement-only CSGM in seven of
nine cells. It also has lower mean RMSE than both ablations throughout the
F03-4 grid. The opposite Ridge--MLP rankings show that the preferred
latent-prior architecture is regime-dependent.

CLP-CSGM is a targeted sparse-assimilation method and is not proposed as a
general replacement for direct petrophysical regression. Validation on larger
multi-well datasets, independently calibrated targets, geological labels,
uncertainty estimates, and faster latent-refinement strategies remains
necessary.
```

**Ação sugerida nesta análise:** Manter no resumo e nos destaques tanto os regimes favoráveis quanto os desfavoráveis; não tratar sete de nove células como superioridade universal.

**Cruzamento com outros revisores:** `R2-02`, `R2-03`, `R3-02`, `R3-06`, `R3-P05`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-M06 — Por que Ridge é a referência principal?

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Justificativa técnica  
**Temas comuns:** T07

**Original do revisor:**

> The MLP prior is better in all nine cross-well cells, whereas Ridge is better in the single-well benchmark. This is not a minor implementation detail; it shows that the prior model materially affects performance. The authors should discuss more clearly why Ridge remains the primary reference despite MLP being consistently better in the cross-well grid.

**Tradução para português:**

O prior MLP é melhor nas nove células entre poços, enquanto Ridge é melhor no benchmark de um poço. Isso não é um detalhe menor: a escolha do prior afeta materialmente o desempenho. Explicar melhor por que Ridge permanece a referência principal apesar de MLP ser consistentemente melhor no teste entre poços.

**Trecho/local do manuscrito relacionado:**

**Página 28 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> The sensitivity results in Section 6.4 show that no single architecture for h is uniformly preferable. MLP has lower mean RMSE in the multi-channel cross-well benchmark, whereas Ridge has lower mean RMSE under the single-channel F03-4 setting. In both cases, the difference decreases as ρ increases, consistent with reduced dependence on the contextual prior as more target observations become available. These […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/06_results.tex`, linhas 271–284.**

Contexto no fonte: `\section{Results}`; `\subsection{Latent-prior model sensitivity}`.

```latex
Tables~\ref{tab:prior_class_crosswell} and~\ref{tab:prior_class_f03} compare
Ridge and MLP implementations of $h$ while keeping the remaining pipeline
fixed. The comparison is a sensitivity analysis; it does not constitute an
exhaustive architecture search. Signed Ridge gain is defined as
$100(\mathrm{RMSE}_{\mathrm{MLP}}-\mathrm{RMSE}_{\mathrm{Ridge}})
 /\mathrm{RMSE}_{\mathrm{MLP}}$; positive values favor Ridge.

In the cross-well benchmark, MLP has lower mean RMSE in all nine cells. The
relative difference is largest at sparse measurement ratios and decreases as
$\rho$ increases; at step $16$, for example, signed Ridge gain changes from
approximately $-82\%$ at
$\rho=0.05$ to $-26\%$ at $\rho=0.20$.

\begin{table}[t]
```

**`sections/07_discussion.tex`, linhas 38–49.**

Contexto no fonte: `\section{Interpretation, scope, and limitations}`.

```latex
The sensitivity results in
Section~\ref{subsec:latent_prior_sensitivity} show that no single architecture
for $h$ is uniformly preferable. MLP has lower mean RMSE in the multi-channel
cross-well benchmark, whereas Ridge has lower mean RMSE under the single-channel
F03-4 setting. In both cases, the difference decreases as $\rho$ increases,
consistent with reduced dependence on the contextual prior as more target
observations become available. These comparisons distinguish the CLP-CSGM
framework from its reference implementation: Ridge is retained as the primary
low-capacity configuration, while the prior class, latent dimension, and
regularization should be selected using validation data representative of the
intended measurement regime. The experiments establish this sensitivity but do
not identify which architectural property causes the opposite rankings.
```

**Ação sugerida nesta análise:** Distinguir o método da configuração de referência; explicar a escolha de Ridge e dar destaque comparável à variante MLP, sem escolher a configuração a posteriori usando o teste.

**Cruzamento com outros revisores:** `R3-03`, `R3-A05`, `R3-P06`, `R3-P11`, `R3-P12`, `R3-P13`. A associação é temática, não equivalência literal dos pedidos.


### R1-M07 — Proposição 3 não verifica S-REC no modelo treinado

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Texto e precisão teórica  
**Temas comuns:** T10

**Original do revisor:**

> Proposition 3 is explicitly conditional and does not prove S-REC for the trained decoder. That distinction is important and should remain very clear in the Abstract/Introduction so readers do not interpret the theory as a formal recovery guarantee for the actual trained model.

**Tradução para português:**

A Proposição 3 é explicitamente condicional e não prova S-REC para o decoder treinado. A distinção deve permanecer clara no resumo e na introdução para evitar que a teoria seja interpretada como garantia formal de recuperação do modelo efetivamente treinado.

**Trecho/local do manuscrito relacionado:**

**Página 13 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Proposition 3 is a sufficient conditional interpretation, not a verified guarantee for the trained decoder. The latent penalty localizes the S-REC condition around z0, while the bound separates representation error, measurement residual/noise, prior mismatch, and local separability, following the decoder-range view of generative compressed sensing [Candes et al., 2006b, Bora et al., 2017]. Coordinate restriction is […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 595–610.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

```latex
Proposition~\ref{prop:conditional_local_recovery} provides a sufficient
conditional interpretation, not a verified recovery guarantee for the trained
decoder. The latent penalty localizes the relevant S-REC condition around
$\boldsymbol{z}_0$. The bound separates decoder representation error,
measurement residual and noise, latent-prior mismatch, and local measurement
separability. This formulation differs from classical RIP analysis for
fixed-basis sparse vectors and follows the decoder-range perspective of
generative compressed sensing
\citep{candes2006stable,bora2017compressed}.

The proposition does not establish that coordinate restriction satisfies
S-REC on the trained decoder range; this depends on local decoder secants not
being concentrated on a few coordinates. The complete proof, the extension to
an inexact numerical optimizer, and the corresponding empirical separability
diagnostics are provided in
Appendix~\ref{app:conditional_local_recovery} and
```

**`sections/09_appendix.tex`, linhas 335–353.**

Contexto no fonte: `\section{Supplementary Diagnostics}`; `\subsection{Measurement Separability}`.

```latex
Table~\ref{tab:srec_percentiles} and Fig.~\ref{fig:srec_percentiles} provide an
empirical check of the separation property that is relevant for generative
compressed sensing. For pairs of decoder-generated training windows, we compute
\[
  r_{ij}
  =
  \frac{\sqrt{N/m}\,\|M_\Omega(y_i-y_j)\|_2}
       {\|y_i-y_j\|_2}.
\]
Values close to one indicate that coordinate subsampling preserves pairwise
distances after energy normalization. This diagnostic does not prove S-REC over
the learned decoder range. It instead examines the corresponding practical
question of whether sparse observations distinguish plausible generated
windows. The median ratio
remains close to one in all tested cells. Lower percentiles are generally
stronger in denser settings, despite some non-monotonic variation at intermediate
\(\rho\) values. The weakest case is the sparsest cross-well setting, where only
four entries are observed per window, consistent with the larger ill-posedness
expected at very low measurement ratios.
```

**Observação para a revisão:** O corpo e o apêndice já afirmam que os diagnósticos não provam S-REC. São trechos úteis para a resposta e devem continuar explícitos.

**Ação sugerida nesta análise:** Explicitar as hipóteses e o caráter condicional; separar existência de minimizador, solução numérica e garantia de recuperação.

**Cruzamento com outros revisores:** `R3-05`, `R3-P02`, `R3-P15`. A associação é temática, não equivalência literal dos pedidos.


### R1-M08 — Comparador direto com máscara ou coordenadas explícitas

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Experimento solicitado  
**Temas comuns:** T05, T06

**Original do revisor:**

> The direct baselines use fixed sampled coordinates with implicit positional identity, which is reasonable. Still, I would like to see one stronger direct baseline that explicitly encodes the sampling mask/coordinates, especially because the proposed method uses MΩM_\Omega explicitly. This would better isolate whether the gain comes from the latent inverse formulation or simply from better use of coordinate information.

**Tradução para português:**

Os comparadores diretos usam coordenadas fixas com identidade posicional implícita, o que é razoável. Ainda assim, gostaria de ver um comparador direto mais forte que codifique explicitamente máscara/coordenadas, já que o método proposto usa MΩ explicitamente. Isso isolaria se o ganho vem da formulação inversa latente ou de melhor uso da informação de coordenadas.

**Trecho/local do manuscrito relacionado:**

**Página 18 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Direct baselines receive [u, b] ∈Rp+m, with only the measured target values in the ordering of MΩ. Because Ωis fixed within the cell, the j-th entry of b consistently corresponds to coordinate ωj across windows, making coordinate identity implicit in the stable input ordering. No zero-padded target, binary mask, or measurement matrix is added. CLP-CSGM uses MΩexplicitly in ∥MΩG(z)−b∥2 2. The protocol therefore […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 111–122.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Training, validation, and test protocol}`.

```latex
Within each $(\mathrm{seed},\rho)$ cell, all methods share the partitions, the
fixed sampled coordinates, their ordering, and the noise realization. Direct
baselines receive
$[\boldsymbol{u},\boldsymbol{b}]\in\mathbb{R}^{p+m}$, with only the measured
target values in the ordering of $M_{\Omega}$. Because $\Omega$ is fixed within
the cell, the $j$-th entry of $\boldsymbol{b}$ consistently corresponds to
coordinate $\omega_j$ across windows, making coordinate identity implicit in
the stable input ordering. No zero-padded target, binary mask, or measurement
matrix is added. CLP-CSGM uses $M_{\Omega}$ explicitly in
$\|M_{\Omega}G(\boldsymbol{z})-\boldsymbol{b}\|_2^2$. The protocol therefore
compares direct fusion with operator-based assimilation under a fixed sampling
pattern; it does not require the direct models to infer a window-varying mask.
```

**`sections/05_experiments.tex`, linhas 159–166.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
The direct baselines are ML only, which uses only $\boldsymbol{u}$; MLP
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts the target directly; PCA
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts target principal-component
scores; and AE $[\boldsymbol{u},\boldsymbol{b}]$, which predicts codes of a
separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference
configuration.
Prior-only Ridge and measurement-only CSGM are structural ablations, while the
MLP latent prior is used only for sensitivity analysis.
```

**Ação sugerida nesta análise:** Adicionar comparador direto que receba observações e informação posicional explícita, com protocolo pareado. Esse pedido é relacionado, mas não idêntico, ao teste de novas posições de R2-04.

**Cruzamento com outros revisores:** `R2-03`, `R2-04`, `R3-02`, `R3-03`, `R3-06`, `R3-P11`, `R3-P12`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-M09 — Porosidade com GR apenas é um teste controlado

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Texto e delimitação  
**Temas comuns:** T02

**Original do revisor:**

> Using GR alone is useful as a stress test, and the authors are transparent about this. However, this should not be presented as evidence for general porosity reconstruction performance. It is a controlled sparse-assimilation experiment under intentionally incomplete context.

**Tradução para português:**

Usar apenas GR é útil como teste de estresse, e os autores são transparentes. Porém, isso não deve ser apresentado como evidência de desempenho geral para reconstrução de porosidade. É um experimento controlado de assimilação esparsa sob contexto intencionalmente incompleto.

**Trecho/local do manuscrito relacionado:**

**Página 17 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> The bench- mark is a controlled weak-context sparse-assimilation test. It is not presented as a recommendation of GR-only input for operational porosity prediction. The smaller Lapa pre-salt permeability case [Alvarez et al., 2025] is retained as supplementary evidence in Appendix D. 5.3 Training, validation, and test protocol All preprocessing statistics and model choices are determined without test-target leakage. […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 47–60.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Single-well F03-4 GR-only porosity benchmark}`.

```latex
The second benchmark reconstructs length-$N=64$ windows of the processed
density-derived F03-4 porosity curve using gamma ray as the only dense input.
Acoustic travel time is excluded because it is nearly collinear with the target
(Pearson $r\approx0.9996$) and would provide a near-direct proxy. The resulting
GR-only setting is a weak-context test. The target follows the processing of
\citet{zhang2025rockphysicslogs}, in which the density curve underlying F03-4
porosity is itself predicted from sonic and gamma ray.

Sparse observations are sampled at
$\rho\in\{0.20,0.30,0.40,0.50,0.60\}$. Depth is partitioned before window
construction, preventing overlap among training, validation, and test windows.
The benchmark is a controlled weak-context sparse-assimilation test. It is not
presented as a recommendation of GR-only input for operational porosity
prediction.
```

**`sections/07_discussion.tex`, linhas 60–67.**

Contexto no fonte: `\section{Interpretation, scope, and limitations}`.

```latex
Similarly, the F03-4 target is a processed density-derived porosity curve from
\citet{zhang2025rockphysicslogs}, and the entries of $\boldsymbol{b}$ are
coordinate subsamples of that curve, not independent core measurements. Using
gamma ray alone while excluding the nearly collinear acoustic-travel-time
channel creates a controlled weak-context test. This design is not presented as
a recommendation of GR-only input for operational porosity prediction. The two
F3 benchmarks evaluate the sparse-assimilation mechanism on processed targets
and do not provide validation against independent petrophysical ground truth.
```

**Ação sugerida nesta análise:** Tornar a limitação visível no resumo e nas conclusões; justificar o cenário sem apresentá-lo como prática operacional recomendada.

**Cruzamento com outros revisores:** `R2-02`, `R2-03`, `R3-02`, `R3-P05`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-M10 — Atualizar o posicionamento na literatura

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Literatura e texto  
**Temas comuns:** T13

**Original do revisor:**

> The Introduction and Discussion could be enriched with more recent studies on well-log reconstruction, transferable subsurface learning, and data-driven petrophysical property prediction, particularly to better position the proposed framework against recent advances. https://doi.org/10.1016/j.engeos.2026.100536, 10.1016/j.geoen.2025.214342. https://doi.org/10.1038/s41598-025-95709-0, https://doi.org/10.1109/ACCESS.2023.3336623

**Tradução para português:**

A introdução e a discussão podem ser enriquecidas com estudos recentes de reconstrução de perfis, aprendizado transferível de subsuperfície e previsão petrofísica baseada em dados, para posicionar melhor o método. São sugeridos quatro trabalhos, cujos DOI constam no original abaixo.

**Trecho/local do manuscrito relacionado:**

**Página 7 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> CLP-CSGM belongs to this explicit conditional inverse-reconstruction setting but separates contextual conditioning from measurement assimilation. Dense logs determine z0 = h(u), while sparse targets enter through ∥MΩG(z) −b∥2 2; the anchor remains a regularizer throughout opti- mization. This differs from direct fusion [u, b] 7→y, measurement-only CSGM [Bora et al., 2017], and measurement-conditional CSGM. […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/01_introduction.tex`, linhas 19–32.**

Contexto no fonte: `\section{Introduction}`.

```latex
Compressed sensing represents observations through an explicit forward operator
\citep{donoho2006compressed,candes2006robust,candes2006stable}. Classical
formulations impose sparsity in a fixed basis, which may be restrictive when
cross-well geological variability and distribution shift alter the signal
representation
\citep{bora2017compressed,hegde2018algorithmic,anirudh2018unsupervised,wang2025heterocarbonate,zare2026crosswell,simoes2025subsurface}.
Compressed sensing with generative models (CSGM) replaces this fixed-basis prior
with the range of a learned generator or decoder
\citep{bora2017compressed}. Reconstruction then searches for a latent code
$\boldsymbol{z}$ whose decoded signal $G(\boldsymbol{z})$ matches the sparse
measurements. This learned-range formulation is suitable for target-log
windows. Direct supervised models nevertheless remain essential comparators in
realistic petrophysical settings
\citep{pan2023multiwell,wang2024loggingreview,sun2026corecalibrated}.
```

**`sections/02_related_work.tex`, linhas 4–26.**

Contexto no fonte: `\section{Related work}`; `\subsection{Petrophysical log prediction}`.

```latex
\subsection{Petrophysical log prediction}

Well logs provide dense but indirect measurements of rock and fluid properties,
and petrophysical interpretation commonly combines them with processed or
calibrated target-property curves
\citep{ellis2007well,asquith2004basic}. Data-driven methods are widely used to
predict porosity, permeability, saturation, and shale-related properties from
well logs \citep{ahmadi2019comparison,wood2020predicting}. Most such
formulations use direct supervised regression: dense logs, with sparse
target-well information when available, predict the target property without an
explicit measurement-consistency requirement. Our focus is the role assigned to
the sparse observations. They may serve as auxiliary predictors or as
measurements that the reconstruction must honor.

The experiments use the processed F3 block data of
\citet{zhang2025rockphysicslogs}, who combine direct supervised prediction,
Gardner and Larionov relations, and SHAP-based interpretation for porosity and
clay-volume estimation. We use the same processed logs to study a
different computational problem: reconstructing target-property windows from
dense logs while enforcing explicit test-time consistency with sparse
target-well observations.


```

**Ação sugerida nesta análise:** Ler os quatro trabalhos indicados antes de decidir sua pertinência. O pedido é de posicionamento: não basta acrescentar citações sem discutir a relação com o método.

**Cruzamento:** sem pedido equivalente explícito de outro revisor neste tema; consultar o mapa para comentários relacionados do próprio revisor.


### R1-M11 — Teste com medidas independentes

**Fonte:** Anexo DOCX do revisor 1, bloco Major comments.  
**Natureza:** Validação adicional sugerida  
**Temas comuns:** T01, T03

**Original do revisor:**

> The most valuable addition would be even a small test against independent core-derived porosity/permeability or another target not generated by deterministic processing of the same log suite. The current Lapa permeability case helps, but it is described as limited-sample supplementary evidence.

**Tradução para português:**

A adição mais valiosa seria mesmo um pequeno teste contra porosidade/permeabilidade de testemunhos independentes, ou outro alvo não gerado pelo processamento determinístico da mesma suíte de perfis. O caso de permeabilidade de Lapa ajuda, mas é apresentado como evidência suplementar com poucas amostras.

**Trecho/local do manuscrito relacionado:**

**Página 25 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> The limited-sample Lapa permeability case in Appendix D provides supplementary evidence. The prior-guided CLP-CSGM variants have lower mean RMSE than the direct baselines that use [u, b] across the tested grid. Their comparison with measurement-only CSGM is ratio-dependent. 6.3 Ablation analysis The structural ablations isolate the two terms of the CLP-CSGM objective. Prior-only returns G(h(u)) without refinement. […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/09_appendix.tex`, linhas 684–705.**

Contexto no fonte: `\section{Supplementary Lapa Pre-salt Permeability Case Study}`.

```latex
This appendix reports a complementary case study built from the well-log and
permeability data context described by \citet{alvarez2025nonlinear} for the
Lapa pre-salt carbonate reservoir in the Santos Basin. Geologically, the Lapa
oilfield targets the Aptian Barra Velha Formation, a heterogeneous pre-salt
carbonate succession in which laminated microbialites, spherulite,
stromatolite, rudstone, and grainstone facies coexist. In such rocks,
permeability is controlled by pore structure and diagenesis as much as by
porosity. Permeability is therefore the sparse reconstruction target in this
supplementary check. Full core--log integration,
hydraulic flow units, and nonlinear permeability prediction details are
reported in \citet{alvarez2025nonlinear}. The benchmark adaptation uses
the same CLP-CSGM protocol as the main paper: sliding
windows, contiguous train/validation/test split, coordinate subsampling for
sparse target observations, paired seeds and measurement ratios, and matched
direct \([u,b]\) baselines.

This case documents performance under a limited-sample setting and does not
constitute a new primary benchmark. The configuration uses \(L=16\), step
\(=4\), seven dense channels in \(u\), target \(k_{\mathrm{lab}}\) in
\(\log_{10}\) scale,
\(\rho\in\{0.20,0.30,0.40,0.50,0.60\}\), and the same three fixed random-seed
realizations used only for reproducible paired averaging.
```

**`sections/07_discussion.tex`, linhas 69–73.**

Contexto no fonte: `\section{Interpretation, scope, and limitations}`.

```latex
The Lapa permeability case
\citep{alvarez2025nonlinear} extends the evaluation to another lithology and
target property, but its small sample size limits generalization. Larger
multi-well datasets, additional log suites, and facies-conditioned settings
remain necessary for broader validation.
```

**Observação para a revisão:** O apêndice identifica o alvo como k_lab em escala log10 e cita a integração core–log. Contudo, as observações esparsas continuam obtidas por subamostragem. Este caso não elimina automaticamente o pedido de ampliar a evidência independente; sua procedência e o alcance precisam ser explicados.

**Ação sugerida nesta análise:** Verificar o que o caso Lapa já demonstra no suplemento e a viabilidade de ampliar validação independente. Não transformar a formulação sugestiva do revisor em exigência literal inexistente.

**Cruzamento com outros revisores:** `R2-02`, `R3-02`, `R3-P01`, `R3-P17`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-m01 — Encurtar o título

**Fonte:** Anexo DOCX do revisor 1, bloco Medium/minor comments.  
**Natureza:** Texto  
**Temas comuns:** T08

**Original do revisor:**

> The title is accurate but quite long. It could be shortened slightly while retaining “Conditional Latent-Prior” and “Petrophysical Target-Log Reconstruction.”

**Tradução para português:**

O título é preciso, mas longo. Ele pode ser ligeiramente encurtado mantendo “Conditional Latent-Prior” e “Petrophysical Target-Log Reconstruction”.

**Trecho/local do manuscrito relacionado:**

**Página 3 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Conditional Latent-Prior Generative Compressed Sensing with Application to Petrophysical Target-Log Reconstruction Romulo Brito da Silva1,3,*, Irineu de Azevedo Lima Neto3 Diego Passos4, Herson Oliveira da Rocha1,2,3 Roseane Marchezi Missagia1, Marco Antonio Rodrigues de Ceia1 1LENEP, North Fluminense State University (UENF), Rodovia Amaral Peixoto, km 163, Imboassica, Macaé, RJ 27925-310, Brazil 2Polytechnic […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linha 47.**

```latex
\title{Conditional Latent-Prior Generative Compressed Sensing with Application to Petrophysical Target-Log Reconstruction}
```

**Ação sugerida nesta análise:** Revisar o título preservando os termos que identificam a contribuição e a aplicação.

**Cruzamento com outros revisores:** `R2-01`, `R3-01`, `R3-A01`, `R3-A02`, `R3-A04`, `R3-A05`, `R3-P01`, `R3-P02`, `R3-P03`, `R3-P04`, `R3-P05`, `R3-P06`, `R3-P07`, `R3-P08`, `R3-P14`, `R3-P17`, `R3-P18`. A associação é temática, não equivalência literal dos pedidos.


### R1-m02 — Reduzir sobreposição entre introdução e trabalhos relacionados

**Fonte:** Anexo DOCX do revisor 1, bloco Medium/minor comments.  
**Natureza:** Texto e estrutura  
**Temas comuns:** T08

**Original do revisor:**

> The Introduction and Related Work are strong, but some paragraphs could be tightened to reduce overlap between direct prediction, conditional inverse problems, and measurement-consistency framing.

**Tradução para português:**

A introdução e os trabalhos relacionados são fortes, mas alguns parágrafos podem ser condensados para reduzir a sobreposição entre previsão direta, problemas inversos condicionais e consistência com medições.

**Trecho/local do manuscrito relacionado:**

**Página 5 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Such a direct map [u, b] 7→y need not enforce MΩby ≈b at inference time. Here, sparse target values are instead treated as explicit reconstruction measurements. Compressed sensing represents observations through an explicit forward operator [Donoho, 2006, Candes et al., 2006a,b]. Fixed-basis sparsity may be restrictive when cross-well variability and distribution shift alter the signal representation [Bora et al., […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/01_introduction.tex`, linhas 19–41.**

Contexto no fonte: `\section{Introduction}`.

```latex
Compressed sensing represents observations through an explicit forward operator
\citep{donoho2006compressed,candes2006robust,candes2006stable}. Classical
formulations impose sparsity in a fixed basis, which may be restrictive when
cross-well geological variability and distribution shift alter the signal
representation
\citep{bora2017compressed,hegde2018algorithmic,anirudh2018unsupervised,wang2025heterocarbonate,zare2026crosswell,simoes2025subsurface}.
Compressed sensing with generative models (CSGM) replaces this fixed-basis prior
with the range of a learned generator or decoder
\citep{bora2017compressed}. Reconstruction then searches for a latent code
$\boldsymbol{z}$ whose decoded signal $G(\boldsymbol{z})$ matches the sparse
measurements. This learned-range formulation is suitable for target-log
windows. Direct supervised models nevertheless remain essential comparators in
realistic petrophysical settings
\citep{pan2023multiwell,wang2024loggingreview,sun2026corecalibrated}.

We propose Conditional Latent-Prior CSGM (CLP-CSGM), a latent inverse-problem
framework for reconstructing target-property windows from dense logs and sparse
target observations. An autoencoder decoder $G$ defines the admissible
target-window range, while a predictor $h$ maps the dense log window
$\boldsymbol{u}$ to the context-dependent latent anchor
$\boldsymbol{z}_0=h(\boldsymbol{u})$. Ridge is used as the primary
low-capacity reference implementation of $h$, and an MLP variant is examined in
the sensitivity analysis. At test time, CLP-CSGM solves
```

**`sections/02_related_work.tex`, linhas 63–80.**

Contexto no fonte: `\section{Related work}`; `\subsection{Conditional generative inverse problems and positioning}`.

```latex
\subsection{Conditional generative inverse problems and positioning}

Conditional generative approaches differ in where side information and
measurements enter the reconstruction. Direct conditional predictors map side
information, alone or combined with observations, to an output signal or latent
code \citep{mirza2014conditional,sohn2015learning}. Measurement-conditional CSGM
instead conditions the generator directly on compressed measurements and often
on their sampling pattern \citep{kim2021measurementconditional}. Other methods
retain an explicit inverse-problem formulation: side information may regularize
a measurement-consistent reconstruction
\citep{ongie2020deep,lucas2018using}; conditional generative models may
approximate or sample from inverse posteriors
\citep{adler2018deepbayes,altekruger2024robustcgm,bendel2024rcgan}; and
score- or diffusion-based models may support inverse reconstruction or posterior
sampling \citep{dasgupta2024mechanicsdiffusion,jacobsen2025cocogen}.

CLP-CSGM belongs to the explicit conditional inverse-reconstruction setting but
separates contextual conditioning from measurement assimilation. Dense logs
```

**Ação sugerida nesta análise:** Concentrar motivação e contribuição na introdução; manter diferenças técnicas e equações na metodologia.

**Cruzamento com outros revisores:** `R2-01`, `R3-01`, `R3-A01`, `R3-A02`, `R3-A04`, `R3-A05`, `R3-P01`, `R3-P02`, `R3-P03`, `R3-P04`, `R3-P05`, `R3-P06`, `R3-P07`, `R3-P08`, `R3-P14`, `R3-P17`, `R3-P18`. A associação é temática, não equivalência literal dos pedidos.


### R1-m03 — Legibilidade da Figura 2

**Fonte:** Anexo DOCX do revisor 1, bloco Medium/minor comments.  
**Natureza:** Figura  
**Temas comuns:** T11

**Original do revisor:**

> Figure 2 is informative and technically useful, but the text density is high. Some labels could be enlarged for print readability.

**Tradução para português:**

A Figura 2 é informativa e tecnicamente útil, mas tem muito texto. Alguns rótulos podem ser ampliados para facilitar a leitura impressa.

**Trecho/local do manuscrito relacionado:**

**Página 11 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Figure 2: CLP-CSGM workflow. Offline, standardized target windows train the autoencoder (E, G), and paired dense-log windows fit the latent-prior predictor h. At inference, u∗determines z0 = h(u∗), sparse observations b∗constrain latent refinement through MΩ, and the decoded result is returned to the original target scale. OpenAI’s ChatGPT, using the GPT-5.5 model, was used to assist in generating the initial […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 291–305.**

Contexto no fonte: `\section{Method}`; `\subsection{Test-time reconstruction}`.

Labels neste trecho: `fig:clp_csgm_workflow`.

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.86\linewidth]
    {figures/pipeline_CLP-CSGM_v3.pdf}
    \caption{CLP-CSGM workflow. Offline, standardized target windows train the
    autoencoder $(E,G)$, and paired dense-log windows fit the latent-prior
    predictor $h$. At inference, $\boldsymbol{u}_{\ast}$ determines
    $\boldsymbol{z}_0=h(\boldsymbol{u}_{\ast})$, sparse observations
    $\boldsymbol{b}_{\ast}$ constrain latent refinement through $M_{\Omega}$,
    and the decoded result is returned to the original target scale.
    OpenAI's ChatGPT, using the GPT-5.5 model, was used to assist in generating
    the initial explanatory schematic and visual layout. The final figure was
    reviewed, edited, and validated by the authors.}
    \label{fig:clp_csgm_workflow}
\end{figure}
```

**Observação para a revisão:** A Figura 2 do manuscrito revisado corresponde ao fluxo CLP-CSGM. O arquivo gráfico chamado nesta versão é figures/pipeline_CLP-CSGM_v3.pdf; alterar apenas a legenda não muda os rótulos internos da imagem.

**Ação sugerida nesta análise:** Ampliar rótulos e reduzir texto dentro do fluxograma, transferindo detalhes para a legenda.

**Cruzamento com outros revisores:** `R3-06`, `R3-A02`, `R3-A03`, `R3-A04`, `R3-P08`, `R3-P14`, `R3-P18`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-m04 — AE como melhor comparador apenas neste benchmark

**Fonte:** Anexo DOCX do revisor 1, bloco Medium/minor comments.  
**Natureza:** Texto e delimitação  
**Temas comuns:** T02, T06

**Original do revisor:**

> The statement that the direct AE is the “most competitive direct baseline” is supported within this benchmark, but should always be read as benchmark-specific rather than generally strongest.

**Tradução para português:**

A afirmação de que o AE direto é o comparador direto mais competitivo é sustentada neste benchmark, mas deve sempre ser entendida como específica dele, não como superioridade geral.

**Trecho/local do manuscrito relacionado:**

**Página 21 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Figure 6 places the pairwise result within the full benchmark. The mean RMSE of direct MLP and PCA increases sharply as training support decreases, whereas AE has the lowest unweighted mean RMSE among the direct baselines. Table 4 therefore focuses on the most competitive direct comparator in this grid. Figure 7 shows a representative held-out F03-4 profile for step 16, seed 7, and ρ = 0.20. With sonic, bulk […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/08_conclusion.tex`, linhas 10–20.**

Contexto no fonte: `\section{Conclusion}`.

```latex
Under the paired F3 protocols, CLP-CSGM Ridge has lower mean RMSE than AE
$[\boldsymbol{u},\boldsymbol{b}]$, the most competitive direct baseline, in
seven of nine cross-well cells. The unweighted mean of the nine cell-level mean
RMSE values decreases from $0.03496$ to $0.02862$, a reduction of approximately
$18.1\%$. In the F03-4 GR-only benchmark, CLP-CSGM Ridge has the lowest mean
RMSE at the two sparsest measurement ratios, while direct AE fusion becomes
competitive as $\rho$ increases. The full Ridge model has lower mean RMSE than
prior-only in every cross-well cell and than measurement-only CSGM in seven of
nine cells. It also has lower mean RMSE than both ablations throughout the
F03-4 grid. The opposite Ridge--MLP rankings show that the preferred
latent-prior architecture is regime-dependent.
```

**`sections/05_experiments.tex`, linhas 159–166.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
The direct baselines are ML only, which uses only $\boldsymbol{u}$; MLP
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts the target directly; PCA
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts target principal-component
scores; and AE $[\boldsymbol{u},\boldsymbol{b}]$, which predicts codes of a
separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference
configuration.
Prior-only Ridge and measurement-only CSGM are structural ablations, while the
MLP latent prior is used only for sensitivity analysis.
```

**Ação sugerida nesta análise:** Qualificar a afirmação em todas as ocorrências: entre os comparadores e cenários avaliados.

**Cruzamento com outros revisores:** `R2-02`, `R2-03`, `R3-02`, `R3-03`, `R3-06`, `R3-P05`, `R3-P11`, `R3-P12`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R1-m05 — Versão congelada do código

**Fonte:** Anexo DOCX do revisor 1, bloco Medium/minor comments.  
**Natureza:** Reprodutibilidade  
**Temas comuns:** T14

**Original do revisor:**

> Since the code is public, this is a major strength. I suggest the authors also provide a frozen release/tag corresponding exactly to the manuscript version rather than only a live GitHub repository.

**Tradução para português:**

O código público é um ponto forte. Sugiro fornecer também uma release/tag congelada que corresponda exatamente à versão do manuscrito, em vez de apenas um repositório GitHub em atualização.

**Trecho/local do manuscrito relacionado:**

O parecer não aponta uma passagem específica. Aplica-se à disponibilidade/versionamento do código; a declaração correspondente deverá ser localizada na versão-fonte e no suplemento.

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 129–135.**

Contexto no fonte: `\section*{Computer Code Availability}`.

```latex
Source code, installation instructions, dependencies, benchmark scripts,
and documentation are available at
\url{https://github.com/romulobrito/CLP--CSGM} under the MIT License.
The repository covers the cross-well clay-volume and F03-4 porosity
benchmarks, structural ablations, latent-prior sensitivity, and
supplementary diagnostics. Processed well-log tables are not redistributed;
the README specifies the required inputs and reproduction steps.
```

**Observação para a revisão:** O trecho fornece a URL, mas não uma release, tag ou commit congelado.

**Ação sugerida nesta análise:** Documentar uma versão identificável dos experimentos e instruções de reprodução. Publicar tag/release é uma ação futura, não realizada neste levantamento.

**Cruzamento:** sem pedido equivalente explícito de outro revisor neste tema; consultar o mapa para comentários relacionados do próprio revisor.


## Revisor 2 — parecer geral

**Avaliação geral original:**

> The manuscript presents an interesting framework for combining dense well-log information with sparse target observations through latent-space reconstruction. However, several important issues regarding the manuscript organization, experimental design, and practical relevance should be addressed.

**Síntese em português:** O revisor considera a abordagem interessante e pede melhorias de organização, desenho experimental e relevância prática.


### R2-01 — Motivação antes da formulação matemática

**Fonte:** PDF de pareceres, página 1, item 1.  
**Natureza:** Texto e estrutura  
**Temas comuns:** T08

**Original do revisor:**

> The manuscript structure should be improved. The Introduction moves quickly into the mathematical formulation of CLP-CSGM, making it read more like part of the Method section. The practical motivation, research gap, and necessity of the proposed approach are not sufficiently developed before the formulation is introduced. The mathematical details should be moved to the methodology section, while the Introduction should more clearly establish the problem and contribution.

**Tradução para português:**

A estrutura precisa melhorar. A introdução entra rapidamente na matemática de CLP-CSGM e parece parte do método. A motivação prática, a lacuna de pesquisa e a necessidade da abordagem não são suficientemente desenvolvidas antes da formulação. Mover detalhes matemáticos para o método e estabelecer melhor o problema e a contribuição na introdução.

**Trecho/local do manuscrito relacionado:**

**Página 5 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Such a direct map [u, b] 7→y need not enforce MΩby ≈b at inference time. Here, sparse target values are instead treated as explicit reconstruction measurements. Compressed sensing represents observations through an explicit forward operator [Donoho, 2006, Candes et al., 2006a,b]. Fixed-basis sparsity may be restrictive when cross-well variability and distribution shift alter the signal representation [Bora et al., […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/01_introduction.tex`, linhas 12–17.**

Contexto no fonte: `\section{Introduction}`.

```latex
A direct map
$[\boldsymbol{u},\boldsymbol{b}]\mapsto\boldsymbol{y}$ need not satisfy
$M_{\Omega}\widehat{\boldsymbol{y}}\approx\boldsymbol{b}$ at the measured
coordinates at inference time. We examine this computational role of sparse
target-well information; the study does not question the general validity of
supervised petrophysical prediction.
```

**`sections/01_introduction.tex`, linhas 34–62.**

Contexto no fonte: `\section{Introduction}`.

Labels neste trecho: `eq:intro_clp_csgm`.

```latex
We propose Conditional Latent-Prior CSGM (CLP-CSGM), a latent inverse-problem
framework for reconstructing target-property windows from dense logs and sparse
target observations. An autoencoder decoder $G$ defines the admissible
target-window range, while a predictor $h$ maps the dense log window
$\boldsymbol{u}$ to the context-dependent latent anchor
$\boldsymbol{z}_0=h(\boldsymbol{u})$. Ridge is used as the primary
low-capacity reference implementation of $h$, and an MLP variant is examined in
the sensitivity analysis. At test time, CLP-CSGM solves

\begin{equation}
\widehat{\boldsymbol{z}}
\in
\operatorname*{arg\,min}_{\boldsymbol{z}\in\mathbb{R}^{k}}
\left\{
\left\|
M_{\Omega}G(\boldsymbol{z})-\boldsymbol{b}
\right\|_2^2
+
\lambda
\left\|
\boldsymbol{z}-\boldsymbol{z}_0
\right\|_2^2
\right\},
\qquad
\widehat{\boldsymbol{y}}
=
G(\widehat{\boldsymbol{z}}).
\label{eq:intro_clp_csgm}
\end{equation}
```

**Ação sugerida nesta análise:** Reorganizar a introdução em problema, limitações das alternativas, contribuição e escopo; deslocar a notação detalhada para o método.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R3-01`, `R3-A01`, `R3-A02`, `R3-A04`, `R3-A05`, `R3-P01`, `R3-P02`, `R3-P03`, `R3-P04`, `R3-P05`, `R3-P06`, `R3-P07`, `R3-P08`, `R3-P14`, `R3-P17`, `R3-P18`. A associação é temática, não equivalência literal dos pedidos.


### R2-02 — Alvos processados e subamostragem artificial

**Fonte:** PDF de pareceres, página 1, item 2.  
**Natureza:** Texto e delimitação  
**Temas comuns:** T01, T02

**Original do revisor:**

> The practical validity of the benchmark targets requires further discussion. The main targets are processed or derived properties rather than independently measured ground truth: V_c is derived from gamma ray using the Larionov relation, and porosity is density-derived. Moreover, the sparse observations are generated by subsampling these same target curves. Therefore, the experiments primarily demonstrate reconstruction of artificially subsampled processed curves rather than recovery against independent petrophysical measurements. This limitation should be more clearly reflected in the interpretation and claims.

**Tradução para português:**

A validade prática dos alvos exige discussão. Vc é derivado de GR por Larionov e a porosidade é derivada da densidade; não são referências medidas independentemente. Além disso, as observações esparsas são obtidas por subamostragem dessas mesmas curvas. Os experimentos demonstram principalmente a reconstrução de curvas processadas artificialmente subamostradas. Essa limitação deve refletir-se melhor na interpretação e nas afirmações.

**Trecho/local do manuscrito relacionado:**

**Página 16 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> This is a controlled sparse-assimilation benchmark on a processed target curve, not recovery against an independently acquired measurement. Split and sampling details are given in Section 5.3; GR inclusion and the embargo-style boundary check are reported in Appendix B.7. 5.2 Single-well F03-4 GR-only porosity benchmark The second benchmark reconstructs length-N = 64 windows of the processed density-derived F03-4 […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 10–30.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
The experiments use processed well-log curves described by
\citet{zhang2025rockphysicslogs}; no new field data were acquired. We use
the FS2--FS6 intervals of wells F02-1, F03-2, F03-4, and F06-1 from the F3 block
in the Dutch North Sea. Sonic and gamma-ray logs are available in all four
wells; density is measured in F02-1 and F03-2 and predicted from sonic and gamma
ray in F03-4 and F06-1. The processed porosity target is density-derived, while
clay-volume fraction $V_c$ is obtained from gamma ray through the Larionov
relation. Geological, acquisition, and rock-physics details are provided in the
source study.

Appendix~\ref{app:lapa_case} reports a complementary limited-sample Lapa
permeability case based on \citet{alvarez2025nonlinear}. This case extends the
sparse-assimilation evaluation to the Aptian Barra Velha pre-salt carbonates and
serves as supplementary evidence.

The cross-well benchmark reconstructs $V_c$ from $30~\mathrm{dB}$ noisy logs,
using F02-1, F03-2, and F06-1 for model development and holding out F03-4 for
testing. The single-well benchmark reconstructs F03-4 porosity using gamma ray
as the only dense input. In both benchmarks, the sparse target vector
$\boldsymbol{b}$ is generated by coordinate subsampling of the corresponding
processed target curve.
```

**`sections/07_discussion.tex`, linhas 51–67.**

Contexto no fonte: `\section{Interpretation, scope, and limitations}`.

```latex
The empirical scope is limited by the construction of the benchmark targets.
In the cross-well experiment, $V_c$ is a processed Larionov-derived curve. The
task evaluates sparse reconstruction under cross-well shift; it does not assess
recovery of an independently acquired clay-volume measurement. Gamma ray is
excluded from the main dense input to avoid a near-direct GR-to-$V_c$ shortcut,
and the GR-inclusive runs in
Appendix~\ref{app:crosswell_gr_embargo} are interpreted only as channel
sensitivity.

Similarly, the F03-4 target is a processed density-derived porosity curve from
\citet{zhang2025rockphysicslogs}, and the entries of $\boldsymbol{b}$ are
coordinate subsamples of that curve, not independent core measurements. Using
gamma ray alone while excluding the nearly collinear acoustic-travel-time
channel creates a controlled weak-context test. This design is not presented as
a recommendation of GR-only input for operational porosity prediction. The two
F3 benchmarks evaluate the sparse-assimilation mechanism on processed targets
and do not provide validation against independent petrophysical ground truth.
```

**Ação sugerida nesta análise:** Identificar claramente a origem dos alvos e das observações e limitar a interpretação; articular com R1-M11 se houver teste independente.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M02`, `R1-M05`, `R1-M09`, `R1-M11`, `R1-m04`, `R3-02`, `R3-P01`, `R3-P05`, `R3-P17`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R2-03 — Realismo da exclusão de GR e sônico

**Fonte:** PDF de pareceres, página 1, item 3.  
**Natureza:** Justificativa e delimitação  
**Temas comuns:** T02, T06

**Original do revisor:**

> The benchmark design appears somewhat artificial from a practical petrophysical perspective. Gamma ray is excluded from the V_c experiment because V_c is directly derived from GR, while acoustic travel time is excluded from the porosity experiment because of its near-perfect correlation with porosity. Although these choices create controlled stress tests, the authors should better justify what realistic field-data scenarios these configurations represent and clarify the practical applicability of the resulting conclusions.

**Tradução para português:**

O benchmark parece artificial do ponto de vista petrofísico. GR é excluído do teste de Vc por ser sua fonte, e tempo de trânsito acústico é excluído do teste de porosidade por correlação quase perfeita. Embora criem testes controlados, os autores devem justificar quais cenários reais essas configurações representam e esclarecer o alcance prático das conclusões.

**Trecho/local do manuscrito relacionado:**

**Página 14 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Channel-target correlations and input design. Table 2 reports descriptive Pearson correla- tions used only to guide input design. Because Vc is Larionov-derived from gamma ray [Zhang et al., 2025], the main cross-well input excludes gamma ray and uses sonic, density, acoustic impedance, and Vp. Gamma ray is restored only in Appendix B.7. In F03-4, acoustic travel time is nearly collinear with porosity, whereas gamma […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 67–75.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
\paragraph{Channel-target correlations and input design.}
Table~\ref{tab:eda_correlations} reports descriptive Pearson correlations used
to justify the input channels; these values are not supplied to the models.
In the cross-well data, gamma ray is strongly correlated with $V_c$ because
the target is Larionov-derived from that log
\citep{zhang2025rockphysicslogs}. To avoid a near-direct shortcut, the main
dense input $\boldsymbol{u}$ excludes gamma ray and uses sonic, density,
acoustic impedance, and $V_p$. Gamma ray is reintroduced only in the
sensitivity analysis of Appendix~\ref{app:crosswell_gr_embargo}.
```

**`sections/05_experiments.tex`, linhas 47–60.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Single-well F03-4 GR-only porosity benchmark}`.

```latex
The second benchmark reconstructs length-$N=64$ windows of the processed
density-derived F03-4 porosity curve using gamma ray as the only dense input.
Acoustic travel time is excluded because it is nearly collinear with the target
(Pearson $r\approx0.9996$) and would provide a near-direct proxy. The resulting
GR-only setting is a weak-context test. The target follows the processing of
\citet{zhang2025rockphysicslogs}, in which the density curve underlying F03-4
porosity is itself predicted from sonic and gamma ray.

Sparse observations are sampled at
$\rho\in\{0.20,0.30,0.40,0.50,0.60\}$. Depth is partitioned before window
construction, preventing overlap among training, validation, and test windows.
The benchmark is a controlled weak-context sparse-assimilation test. It is not
presented as a recommendation of GR-only input for operational porosity
prediction.
```

**`sections/09_appendix.tex`, linhas 554–566.**

Contexto no fonte: `\section{Supplementary Diagnostics}`; `\subsection{Cross-well Channel-Ablation and Embargo-Style Boundary Check}`.

```latex
This subsection reports a paired robustness check for the cross-well \(V_c\)
step-16 setting. It tests two possible sensitivities: reintroducing GR into
\(u\), relative to the main no-GR protocol in Section~\ref{sec:experiments}, and
removing overlap-prone windows near the train/validation boundary. The latter is
an embargo-style boundary check: a small band of windows adjacent to the boundary
is removed before validation to create a gap between training and validation
windows. We compare four scenarios under the same seeds and measurement-ratio
grid, \(\rho\in\{0.05,0.10,0.20\}\):
\begin{itemize}
  \item No boundary check + GR in \(u\) (stronger dense context reference).
  \item Embargo-style removal (4 boundary windows) + GR in \(u\).
  \item No boundary check + GR excluded from \(u\) (matches the main cross-well \(u\)).
  \item Embargo-style removal (4 boundary windows) + GR excluded from \(u\).
```

**Observação para a revisão:** O apêndice já contém uma comparação com e sem GR no cenário step-16. Esse resultado pode apoiar a resposta; não substitui automaticamente o comparador linear direto solicitado pelo revisor 3.

**Ação sugerida nesta análise:** Justificar a informação disponível em cada cenário; discutir explicitamente a situação com perfis completos e o teste com GR já mencionado no Apêndice B.7.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M02`, `R1-M05`, `R1-M08`, `R1-M09`, `R1-m04`, `R3-02`, `R3-03`, `R3-06`, `R3-P05`, `R3-P11`, `R3-P12`, `R3-P19`. A associação é temática, não equivalência literal dos pedidos.


### R2-04 — Sensibilidade à posição das observações

**Fonte:** PDF de pareceres, página 1, item 4.  
**Natureza:** Experimento solicitado  
**Temas comuns:** T04, T05

**Original do revisor:**

> The robustness to sparse-observation patterns should be examined more carefully. A single coordinate set is sampled for each seed–measurement-ratio combination and then kept fixed across all windows. The conclusions may therefore depend on the particular locations of the sparse observations. Additional sampling realizations or an analysis of sensitivity to observation placement would strengthen the evaluation.

**Tradução para português:**

É necessário examinar melhor a robustez aos padrões de observações esparsas. Uma única seleção de coordenadas por combinação semente–fração é mantida em todas as janelas. As conclusões podem depender dessas posições. Mais realizações de amostragem ou análise de sensibilidade à localização fortaleceriam a avaliação.

**Trecho/local do manuscrito relacionado:**

**Página 18 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> One coordinate set Ω= {ω1, . . . , ωm} is sampled for each (seed, ρ) cell and held fixed across all training, validation, and test windows in that cell. The set defines b =  yω1 · · · yωm ⊤ + η. (16) Within each (seed, ρ) cell, all methods share the partitions, the fixed sampled coordinates, their ordering, and the noise realization. Direct baselines receive [u, b] ∈Rp+m, with only the measured target values in […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 89–122.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Training, validation, and test protocol}`.

Labels neste trecho: `eq:experimental_sparse_measurements`.

```latex
For each seed and measurement ratio $\rho$, the number of observed coordinates
is
\[
m=\max\{4,\operatorname{round}(\rho L)\}.
\]
We use nearest-integer rounding, with half-integer ties resolved to the nearest
even integer. One coordinate set
$\Omega=\{\omega_1,\ldots,\omega_m\}$ is sampled for each
$(\mathrm{seed},\rho)$ cell and held fixed across all training, validation, and
test windows in that cell. The set defines

\begin{equation}
\boldsymbol{b}
=
\begin{bmatrix}
y_{\omega_1} & \cdots & y_{\omega_m}
\end{bmatrix}^{\top}
+
\boldsymbol{\eta}.
\label{eq:experimental_sparse_measurements}
\end{equation}

Within each $(\mathrm{seed},\rho)$ cell, all methods share the partitions, the
fixed sampled coordinates, their ordering, and the noise realization. Direct
baselines receive
$[\boldsymbol{u},\boldsymbol{b}]\in\mathbb{R}^{p+m}$, with only the measured
target values in the ordering of $M_{\Omega}$. Because $\Omega$ is fixed within
the cell, the $j$-th entry of $\boldsymbol{b}$ consistently corresponds to
coordinate $\omega_j$ across windows, making coordinate identity implicit in
the stable input ordering. No zero-padded target, binary mask, or measurement
matrix is added. CLP-CSGM uses $M_{\Omega}$ explicitly in
$\|M_{\Omega}G(\boldsymbol{z})-\boldsymbol{b}\|_2^2$. The protocol therefore
compares direct fusion with operator-based assimilation under a fixed sampling
pattern; it does not require the direct models to infer a window-varying mask.
```

**Ação sugerida nesta análise:** Variar a realização das posições preservando orçamento e comparações pareadas. Separar, quando possível, aleatoriedade de treinamento e de amostragem.

**Cruzamento com outros revisores:** `R1-M03`, `R1-M04`, `R1-M08`. A associação é temática, não equivalência literal dos pedidos.


### R2-05 — Robustez estatística com mais realizações

**Fonte:** PDF de pareceres, página 1, item 5.  
**Natureza:** Experimento solicitado  
**Temas comuns:** T04

**Original do revisor:**

> The statistical robustness of the comparison is limited by the use of only three seeds. Given that the main conclusions rely on relatively small performance differences in some experimental settings, three realizations provide limited evidence regarding robustness. Increasing the number of realizations, particularly for random sparse-observation sampling, would provide more convincing support for the reported performance differences.

**Tradução para português:**

A comparação usa apenas três sementes. Como algumas diferenças são pequenas, três realizações dão evidência limitada de robustez. Aumentar as realizações, sobretudo de amostragem aleatória das observações, daria suporte mais convincente às diferenças de desempenho.

**Trecho/local do manuscrito relacionado:**

**Página 20 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> All seed-level intervals use the three fixed seeds {7, 23, 41}. Reported 95% half-widths are Student-t intervals with df = nseeds −1 = 2 (critical value t2,0.975 ≈4.303), not normal z = 1.96 intervals. Where CLP-CSGM Ridge is compared with AE [u, b], we also report the paired seed-wise difference ∆RMSE = RMSEAE −RMSECLP, together with its seed-level standard error and Student-t half-width. Positive ∆RMSE favors CLP- […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 174–191.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
\paragraph{Uncertainty summaries.}
All seed-level intervals use the three fixed seeds $\{7,23,41\}$. Reported
$95\%$ half-widths are Student-$t$ intervals with
$\mathrm{df}=n_{\mathrm{seeds}}-1=2$ (critical value
$t_{2,0.975}\approx 4.303$), not normal $z=1.96$ intervals. Where CLP-CSGM
Ridge is compared with AE $[\boldsymbol{u},\boldsymbol{b}]$, we also report the
paired seed-wise difference
\[
\Delta\mathrm{RMSE}
=
\mathrm{RMSE}_{\mathrm{AE}}
-
\mathrm{RMSE}_{\mathrm{CLP}},
\]
together with its seed-level standard error and Student-$t$ half-width.
Positive $\Delta\mathrm{RMSE}$ favors CLP-CSGM Ridge. With only three seeds,
these intervals quantify reproducibility of the paired protocol rather than a
large-sample confidence claim.
```

**Ação sugerida nesta análise:** Ampliar o painel de realizações e relatar diferenças pareadas e incerteza, sem tratar janelas sobrepostas como repetições independentes.

**Cruzamento com outros revisores:** `R1-M03`, `R1-M04`. A associação é temática, não equivalência literal dos pedidos.


## Revisor 3 — parecer geral e comentários adicionais

**Avaliação geral original:**

> The paper proposes an ML-based method to predict continuous petrophysical targets such as porosity, permeability, clay concentration etc. The method uses other logs as conditions, solving the prediction problem as minimizing the difference between a decoder result, and sparse observations such as core samples. The problem the authors try to solve is interesting, relevant, and meaningful. However, there are several issues with the current manuscript. I would suggest this manuscript to have a major revision before publication.

**Síntese em português:** O revisor considera o problema interessante e relevante, mas recomenda revisão maior por clareza, comparações, justificativa prática e apresentação.


### R3-01 — Clareza e excesso de terminologia

**Fonte:** PDF de pareceres, página 2, item 1.  
**Natureza:** Texto  
**Temas comuns:** T08

**Original do revisor:**

> The clarity of writing. The paper is hard to understand with layers of terminology stacking together. Let me take the abstract as an example: “A decoder learned from target-property windows defines the admissible reconstruction range”: what is admissible reconstruction range? Another example is “With Ridge as a low-capacity latent-prior reference”, it is very difficult to understand “low-capacity latent-prior reference”. “Ablations against prior-only and measurement-only CSGM variants indicate that contextual latent anchoring and sparse-measurement consistency make complementary contributions whose relative importance varies across regimes.” If you read it through, it is very difficult to understand things like “contextual latent anchoring”, “sparse- measurement consistency”, etc.

**Tradução para português:**

O texto é difícil de entender devido a camadas de terminologia. Exemplos no resumo são “admissible reconstruction range”, “low-capacity latent-prior reference”, “contextual latent anchoring” e “sparse-measurement consistency”. O revisor pede explicações compreensíveis para essas expressões.

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> A decoder learned from target-property windows defines the admissible reconstruction range. Dense wireline logs predict a context-dependent latent anchor, and sparse target-log observations refine the estimate through an explicit data-fidelity term at inference time. We evaluate CLP-CSGM on public Dutch North Sea well-log data reorganized into a held-out-well clay-volume benchmark and a single-well porosity stress […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 84–93.**

```latex
Conditional Latent-Prior Compressed Sensing with Generative Models (CLP-CSGM), a
latent inverse-problem framework that separates these roles. A decoder learned
from target-property windows defines the admissible reconstruction range. Dense
wireline logs predict a context-dependent latent anchor, and sparse target-log
observations refine the estimate through an explicit data-fidelity term at
inference time.
We evaluate CLP-CSGM on public Dutch North Sea well-log data reorganized into a
held-out-well clay-volume benchmark and a single-well porosity stress test with
weak dense context. A limited-sample permeability case provides supplementary
evidence. With Ridge as a low-capacity latent-prior reference, CLP-CSGM achieves
```

**`sections/01_introduction.tex`, linhas 34–41.**

Contexto no fonte: `\section{Introduction}`.

```latex
We propose Conditional Latent-Prior CSGM (CLP-CSGM), a latent inverse-problem
framework for reconstructing target-property windows from dense logs and sparse
target observations. An autoencoder decoder $G$ defines the admissible
target-window range, while a predictor $h$ maps the dense log window
$\boldsymbol{u}$ to the context-dependent latent anchor
$\boldsymbol{z}_0=h(\boldsymbol{u})$. Ridge is used as the primary
low-capacity reference implementation of $h$, and an MLP variant is examined in
the sensitivity analysis. At test time, CLP-CSGM solves
```

**Ação sugerida nesta análise:** Trocar rótulos abstratos por descrições das operações e dos dados; definir apenas os termos necessários.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-02 — Vantagem prática frente a relações convencionais

**Fonte:** PDF de pareceres, página 2, item 2.  
**Natureza:** Justificativa e comparação  
**Temas comuns:** T01, T02, T06

**Original do revisor:**

> The method is very interesting; however, what worries me is how practical this method is: The derivation of the target is very simple: V_clay is just linearly derived from GR with endpoints; whereas porosity is almost a linear function of RHOB or acoustic slowness. If you are trying to predict those logs, why use a complicated ML workflow instead of just a simple linear relationship? I think the authors should put more effort on convincing people why we need to choose ML over traditional methods, other than use traditional linear results as benchmark and try to replicate them.

**Tradução para português:**

O método é interessante, mas preocupa sua utilidade prática. O revisor descreve Vclay como derivado linearmente de GR com extremos e porosidade como quase linear em RHOB ou vagarosidade acústica. Pergunta por que usar um fluxo complexo de ML em vez dessas relações e pede evidência de sua necessidade, além de reproduzir resultados convencionais.

**Trecho/local do manuscrito relacionado:**

**Página 14 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Channel-target correlations and input design. Table 2 reports descriptive Pearson correla- tions used only to guide input design. Because Vc is Larionov-derived from gamma ray [Zhang et al., 2025], the main cross-well input excludes gamma ray and uses sonic, density, acoustic impedance, and Vp. Gamma ray is restored only in Appendix B.7. In F03-4, acoustic travel time is nearly collinear with porosity, whereas gamma […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 67–75.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
\paragraph{Channel-target correlations and input design.}
Table~\ref{tab:eda_correlations} reports descriptive Pearson correlations used
to justify the input channels; these values are not supplied to the models.
In the cross-well data, gamma ray is strongly correlated with $V_c$ because
the target is Larionov-derived from that log
\citep{zhang2025rockphysicslogs}. To avoid a near-direct shortcut, the main
dense input $\boldsymbol{u}$ excludes gamma ray and uses sonic, density,
acoustic impedance, and $V_p$. Gamma ray is reintroduced only in the
sensitivity analysis of Appendix~\ref{app:crosswell_gr_embargo}.
```

**`sections/05_experiments.tex`, linhas 47–60.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Single-well F03-4 GR-only porosity benchmark}`.

```latex
The second benchmark reconstructs length-$N=64$ windows of the processed
density-derived F03-4 porosity curve using gamma ray as the only dense input.
Acoustic travel time is excluded because it is nearly collinear with the target
(Pearson $r\approx0.9996$) and would provide a near-direct proxy. The resulting
GR-only setting is a weak-context test. The target follows the processing of
\citet{zhang2025rockphysicslogs}, in which the density curve underlying F03-4
porosity is itself predicted from sonic and gamma ray.

Sparse observations are sampled at
$\rho\in\{0.20,0.30,0.40,0.50,0.60\}$. Depth is partitioned before window
construction, preventing overlap among training, validation, and test windows.
The benchmark is a controlled weak-context sparse-assimilation test. It is not
presented as a recommendation of GR-only input for operational porosity
prediction.
```

**`sections/05_experiments.tex`, linhas 159–166.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
The direct baselines are ML only, which uses only $\boldsymbol{u}$; MLP
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts the target directly; PCA
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts target principal-component
scores; and AE $[\boldsymbol{u},\boldsymbol{b}]$, which predicts codes of a
separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference
configuration.
Prior-only Ridge and measurement-only CSGM are structural ablations, while the
MLP latent prior is used only for sensitivity analysis.
```

**Ação sugerida nesta análise:** Separar cálculo convencional com perfis completos e reconstrução sob entradas limitadas; verificar a fórmula exata de Larionov antes de responder à afirmação de linearidade; avaliar comparadores adequados.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M02`, `R1-M05`, `R1-M08`, `R1-M09`, `R1-M11`, `R1-m04`, `R2-02`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


### R3-03 — Comparadores simples e resultados intermediários

**Fonte:** PDF de pareceres, página 2, item 3.  
**Natureza:** Experimentos solicitados e justificativa  
**Temas comuns:** T06, T07

**Original do revisor:**

> The intermediate results and baselines The paper listed several baselines, but the comparison is incomplete. Direct linear or ridge regression, with the same input and evaluation should be included. Methods such as smoothing, interpolation based on the sparse target (core data) should also be compared. Moreover, the authors did not report the intermediate results from the multi-step ML workflow, such as how the encoder reconstructed the original well logs? And why do we need an encoder here, considering there are only 64 values included in the window.

**Tradução para português:**

A comparação está incompleta. Incluir regressão linear ou Ridge direta com as mesmas entradas e avaliação, além de suavização/interpolação a partir do alvo esparso. Apresentar resultados intermediários, por exemplo a reconstrução pelo autoencoder, e explicar a necessidade de um encoder para janelas de apenas 64 valores.

**Trecho/local do manuscrito relacionado:**

**Página 19 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> 5.5 Baselines and evaluation metrics The direct baselines are ML only, which uses only u; MLP [u, b], which predicts the target directly; PCA [u, b], which predicts target principal-component scores; and AE [u, b], which predicts codes of a separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference configuration. Prior-only Ridge and measurement-only CSGM are structural ablations, while the MLP […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 127–133.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

```latex
\paragraph{Generative decoder.}
Let
$E_{\theta_E}:\mathbb{R}^{N}\rightarrow\mathbb{R}^{k}$ and
$G_{\theta_G}:\mathbb{R}^{k}\rightarrow\mathbb{R}^{N}$, with $k<N$, denote
the encoder and decoder. Given standardized training targets
$\{\widetilde{\boldsymbol{y}}_i\}_{i=1}^{n_{\mathrm{tr}}}$, their parameters
are obtained from
```

**`sections/05_experiments.tex`, linhas 159–166.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
The direct baselines are ML only, which uses only $\boldsymbol{u}$; MLP
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts the target directly; PCA
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts target principal-component
scores; and AE $[\boldsymbol{u},\boldsymbol{b}]$, which predicts codes of a
separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference
configuration.
Prior-only Ridge and measurement-only CSGM are structural ablations, while the
MLP latent prior is used only for sensitivity analysis.
```

**`tables/implementation_summary.tex`, linhas 9–24.**

```latex
Autoencoder \((E,G)\) &
MLP with one hidden layer (width 128) in encoder and decoder, latent dimension \(k=16\), ReLU activations, linear output; 200 epochs, Adam \citep{kingma2015adam} (\(10^{-3}\)), weight decay \(10^{-5}\), batch size 64. \\
&
Latent prior \(h(u)\) &
Ridge regressor with \(\alpha=1.0\) on standardized \((u,z)\) pairs (main experiments). Sensitivity variant: MLP with two hidden layers (width 128), \(L_2\) penalty \(10^{-4}\), learning rate \(10^{-3}\), max.\ 500 iterations, early stopping. \\
\midrule
Target-well inference &
Latent refinement &
Adam \citep{kingma2015adam} for 400 iterations, learning rate \(0.05\), three restarts (anchor \(z_0=h(u)\) plus perturbed starts); \(\lambda\) selected on validation from \(\{10^{-4},3{\cdot}10^{-4},10^{-3},3{\cdot}10^{-3},10^{-2},3{\cdot}10^{-2},10^{-1}\}\). \\
&
Reconstruction &
\(\hat{y}=G(\hat{z})\) after inverse standardization to the original target scale. \\
\midrule
Direct baselines &
\([u,b]\) regressors &
ML only and MLP \([u,b]\): sklearn MLP with two hidden layers (width 128), max.\ 500 iterations, learning rate \(10^{-3}\), \(L_2\) penalty \(10^{-4}\), early stopping. PCA \([u,b]\): PCA score grid \(\{8,16,24,32,48,64\}\) with the same MLP head. AE \([u,b]\): separate MLP autoencoder (\(k=32\), hidden width 128, 120 epochs) plus MLP from \([u,b]\) to latent codes. \\
```

**Observação para a revisão:** A tabela distingue o autoencoder do CLP-CSGM (k=16) daquele do baseline AE (k=32). A comparação do papel do espaço latente deve respeitar essa diferença. Ridge como prior não é o mesmo teste que regressão Ridge direta do alvo.

**Ação sugerida nesta análise:** Incluir comparadores simples e diagnóstico de G(E(y)), G(h(u)) e G(z refinado). O diagnóstico com alvo completo deve ser identificado como retrospectivo, não como previsão operacional.

**Cruzamento com outros revisores:** `R1-M06`, `R1-M08`, `R1-m04`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


### R3-04 — Escolha do ruído e balanço prior/dados

**Fonte:** PDF de pareceres, páginas 2–3, item 4.  
**Natureza:** Esclarecimento técnico  
**Temas comuns:** T09

**Original do revisor:**

> The noise How does the authors choose the noise, thus balancing the influence from the prior / the data fit?

**Tradução para português:**

Como os autores escolhem o ruído e, consequentemente, equilibram a influência do prior e do ajuste aos dados?

**Trecho/local do manuscrito relacionado:**

**Página 18 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Noise is applied after subsampling, with ση = 0.01 for cross-well Vc and ση = 0.02 for F03-4 porosity, independently across coordinates and windows.

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/05_experiments.tex`, linhas 124–127.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Training, validation, and test protocol}`.

```latex
Noise is applied after subsampling, with
$\sigma_{\eta}=0.01$ for cross-well $V_c$ and
$\sigma_{\eta}=0.02$ for F03-4 porosity, independently across coordinates and
windows.
```

**`sections/03_method.tex`, linhas 278–289.**

Contexto no fonte: `\section{Method}`; `\subsection{Test-time reconstruction}`.

```latex
The data-fidelity term enforces sparse-measurement consistency, and the latent
penalty regularizes the search toward the log-conditioned anchor. The parameter
$\lambda\geq 0$ is selected on validation windows for each measurement ratio
and seed.

Because $G$ is nonlinear, the objective is generally nonconvex and is solved
with Adam \citep{kingma2015adam}. The first run starts from
$\boldsymbol{z}_0$, additional restarts use
$\boldsymbol{z}_0+\boldsymbol{\epsilon}_r$ with small Gaussian perturbations,
and the candidate with the lowest objective is retained. The resulting
$\widehat{\boldsymbol{z}}_{\mathrm{num}}$ is the numerical optimizer output,
not a certified global minimizer.
```

**`tables/implementation_summary.tex`, linhas 15–17.**

```latex
Target-well inference &
Latent refinement &
Adam \citep{kingma2015adam} for 400 iterations, learning rate \(0.05\), three restarts (anchor \(z_0=h(u)\) plus perturbed starts); \(\lambda\) selected on validation from \(\{10^{-4},3{\cdot}10^{-4},10^{-3},3{\cdot}10^{-3},10^{-2},3{\cdot}10^{-2},10^{-1}\}\). \\
```

**Ação sugerida nesta análise:** Conectar níveis, escala e origem do ruído à seleção de λ por validação e à interpretação probabilística.

**Cruzamento:** sem pedido equivalente explícito de outro revisor neste tema; consultar o mapa para comentários relacionados do próprio revisor.


### R3-05 — Papel da seção teórica

**Fonte:** PDF de pareceres, página 3, item 5.  
**Natureza:** Justificativa e estrutura  
**Temas comuns:** T10

**Original do revisor:**

> The theoretical section Section 3.4 provides mathematical derivations related to the MAP and Tikhonov regularization. It is unclear how this would impact the application of this method, especially considering that MAP / Tikhonov interpretation and existence of a minimizer are standard arguments.

**Tradução para português:**

A seção 3.4 apresenta derivações de MAP e regularização de Tikhonov. Não está claro como isso afeta a aplicação, sobretudo porque essa interpretação e a existência de minimizador são argumentos conhecidos.

**Trecho/local do manuscrito relacionado:**

**Página 12 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Proposition 1 (MAP and Tikhonov interpretation). Suppressing tildes, assume p(b | z, u) = p(b | z) and b | z ∼N MΩG(z), σ2 ηIm  , while z | u ∼N z0, σ2 zIk  , where z0 = h(u), ση > 0, and σz > 0. Then every MAP estimator satisfies bzMAP ∈arg min z∈Rk n ∥MΩG(z) −b∥2 2 + λ ∥z −z0∥2 o , λ = σ2 η σ2z . (10) Thus, CLP-CSGM is a Tikhonov-regularized latent inverse problem. Its center is the log- conditioned anchor, not […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 334–337.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

Labels neste trecho: `subsec:interpretation_scope`.

```latex
\subsection{Interpretation and theoretical context}
\label{subsec:interpretation_scope}

The theoretical results required to interpret CLP-CSGM are stated below.
```

**`sections/03_method.tex`, linhas 391–403.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

Labels neste trecho: `rem:csgm_relation`.

```latex
Thus, CLP-CSGM is a Tikhonov-regularized latent inverse problem. Its center is
the log-conditioned anchor, not the origin
\citep{tikhonov1963solution,burger2014map}. The likelihood induces the
data-fidelity term, while the conditional prior induces the latent penalty.
\end{proposition}

\begin{remark}[Relation to classical CSGM]
\label{rem:csgm_relation}
Setting $\lambda=0$ recovers the measurement-only CSGM problem in
Eq.~\eqref{eq:measurement_only_ablation}
\citep{bora2017compressed}. Increasing $\lambda$ shifts the reconstruction
toward the latent-prior estimate $G(\boldsymbol{z}_0)$ by penalizing departures
from $\boldsymbol{z}_0=h(\boldsymbol{u})$ more strongly.
```

**`sections/03_method.tex`, linhas 595–610.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

```latex
Proposition~\ref{prop:conditional_local_recovery} provides a sufficient
conditional interpretation, not a verified recovery guarantee for the trained
decoder. The latent penalty localizes the relevant S-REC condition around
$\boldsymbol{z}_0$. The bound separates decoder representation error,
measurement residual and noise, latent-prior mismatch, and local measurement
separability. This formulation differs from classical RIP analysis for
fixed-basis sparse vectors and follows the decoder-range perspective of
generative compressed sensing
\citep{candes2006stable,bora2017compressed}.

The proposition does not establish that coordinate restriction satisfies
S-REC on the trained decoder range; this depends on local decoder secants not
being concentrated on a few coordinates. The complete proof, the extension to
an inexact numerical optimizer, and the corresponding empirical separability
diagnostics are provided in
Appendix~\ref{app:conditional_local_recovery} and
```

**Ação sugerida nesta análise:** Explicar o que cada resultado informa para o método, evitar apresentar argumentos clássicos como novidade e considerar deslocar detalhes para o suplemento.

**Cruzamento com outros revisores:** `R1-M07`. A associação é temática, não equivalência literal dos pedidos.


### R3-06 — Resultado principal e legibilidade das figuras

**Fonte:** PDF de pareceres, página 3, item 6.  
**Natureza:** Texto, figuras e interpretação  
**Temas comuns:** T06, T11, T12

**Original do revisor:**

> Readability of the results The manuscript reports many metrics, experimental settings, ablations, and sensitivity comparisons, but the principal finding is difficult to extract. The results should be organized around a few clear questions: when does refinement help, how much does it improve over simple alternatives, and when does the additional complexity cease to be worthwhile? The figures also need improvement. In Figure 6, the large errors of some baselines compress the differences among the better-performing methods near the bottom of the plots. A separate panel or enlarged view would make the relevant comparison easier to assess. Figure 7 has small labels and differing horizontal scales that complicate visual comparison.

**Tradução para português:**

Há muitas métricas, configurações, ablações e sensibilidades, mas o resultado principal é difícil de identificar. Organizar os resultados em torno de quando o refinamento ajuda, quanto melhora sobre alternativas simples e quando a complexidade deixa de valer a pena. Na Figura 6, erros grandes comprimem diferenças entre bons métodos: usar painel separado ou ampliação. A Figura 7 tem rótulos pequenos e escalas horizontais diferentes.

**Trecho/local do manuscrito relacionado:**

**Página 22 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Figure 6: Cross-well Vc RMSE for the main methods. CLP-CSGM Ridge has the lowest mean RMSE in the medium-to-severe low-data cells, while the mean RMSE of direct MLP and PCA increases as training windows become scarce. Error bars are Student-t 95% half-widths across seeds (df = 2). Figure 7: Held-out F03-4 Vc reconstruction at step 16, seed 7, and ρ = 0.20. The target is the processed Larionov-derived curve, and the […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/06_results.tex`, linhas 69–77.**

Contexto no fonte: `\section{Results}`; `\subsection{Cross-well low-data results}`.

Labels neste trecho: `fig:crosswell_all_methods`.

```latex
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\linewidth]{results_crosswell_rmse_vs_rho.png}
    \caption{Cross-well $V_c$ RMSE for the main methods. CLP-CSGM Ridge has the
    lowest mean RMSE in the medium-to-severe low-data cells, while the mean RMSE
    of direct MLP and PCA increases as training windows become scarce. Error
    bars are Student-$t$ $95\%$ half-widths across seeds ($\mathrm{df}=2$).}
    \label{fig:crosswell_all_methods}
\end{figure}
```

**`sections/06_results.tex`, linhas 88–95.**

Contexto no fonte: `\section{Results}`; `\subsection{Cross-well low-data results}`.

Labels neste trecho: `fig:crosswell_depth_profile`.

```latex
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\linewidth]{10_depth_profile_crosswell_vc.png}
    \caption{Held-out F03-4 $V_c$ reconstruction at step $16$, seed $7$, and
    $\rho=0.20$. The target is the processed Larionov-derived curve, and the
    sparse observations are coordinate subsamples of that curve.}
    \label{fig:crosswell_depth_profile}
\end{figure}
```

**`sections/09_appendix.tex`, linhas 609–622.**

Contexto no fonte: `\section{Supplementary Diagnostics}`; `\subsection{Computational Trade-Offs}`.

```latex
Table~\ref{tab:runtime_cost} and Fig.~\ref{fig:runtime_cost} quantify the
practical computational trade-off. CLP-CSGM
Ridge is more expensive than AE \([u,b]\) because it solves a latent inverse
problem at validation and test time instead of producing a single direct forward
prediction. Table~\ref{tab:runtime_cost} reports both mean wall-clock time per
test window and a separate normalized runtime diagnostic relative to AE; the
latter is the diagnostic shown in Fig.~\ref{fig:runtime_cost}; it is not the exact
quotient of the averaged seconds in the previous column. In the representative
cross-well case, the additional cost is substantial relative to AE, but it comes
with the lowest mean RMSE among the compared methods. In the F03-4 GR-only case,
CLP-CSGM Ridge and AE \([u,b]\) have closer average RMSE, so the runtime gap is
less clearly justified. The extra optimization is most readily justified when
explicit sparse-measurement consistency produces a meaningful reconstruction
gain, especially under stronger ill-posedness or weaker direct supervision.
```

**Observação para a revisão:** Há custo computacional discutido no apêndice. O pedido de organizar a mensagem principal e melhorar a legibilidade pode ser atendido parcialmente trazendo essa informação para o corpo; a suficiência deve ser avaliada na revisão.

**Ação sugerida nesta análise:** Reorganizar por perguntas científicas e apresentar ganho, variabilidade e custo. Melhorar escala e legibilidade sem ocultar os métodos com pior resultado.

**Cruzamento com outros revisores:** `R1-M05`, `R1-M08`, `R1-m03`, `R1-m04`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


### R3-A01 — Definir símbolos na primeira ocorrência

**Fonte:** PDF de pareceres, página 3, Additional comments.  
**Natureza:** Texto e notação  
**Temas comuns:** T08

**Original do revisor:**

> Define symbols at first use, including those introduced in the abstract and introductory equations.

**Tradução para português:**

Definir os símbolos na primeira utilização, inclusive no resumo e nas equações introdutórias.

**Trecho/local do manuscrito relacionado:**

**Página 5 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Such a direct map [u, b] 7→y need not enforce MΩby ≈b at inference time. Here, sparse target values are instead treated as explicit reconstruction measurements. Compressed sensing represents observations through an explicit forward operator [Donoho, 2006, Candes et al., 2006a,b]. Fixed-basis sparsity may be restrictive when cross-well variability and distribution shift alter the signal representation [Bora et al., […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/01_introduction.tex`, linhas 12–17.**

Contexto no fonte: `\section{Introduction}`.

```latex
A direct map
$[\boldsymbol{u},\boldsymbol{b}]\mapsto\boldsymbol{y}$ need not satisfy
$M_{\Omega}\widehat{\boldsymbol{y}}\approx\boldsymbol{b}$ at the measured
coordinates at inference time. We examine this computational role of sparse
target-well information; the study does not question the general validity of
supervised petrophysical prediction.
```

**`sections/03_method.tex`, linhas 13–21.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

```latex
Let $\boldsymbol{y}\in\mathbb{R}^{N}$ denote a target-property window of $N$
depth samples and let $\boldsymbol{u}\in\mathbb{R}^{p}$ denote the vectorized
dense-log channels over the same interval. Target values are observed at the
ordered coordinate set
$\Omega=\{\omega_1,\ldots,\omega_m\}\subset\{1,\ldots,N\}$, where
$1\leq m<N$ and the measurement ratio is $\rho=m/N$.

The coordinate-restriction operator
$M_{\Omega}:\mathbb{R}^{N}\rightarrow\mathbb{R}^{m}$ is represented by
```

**Ação sugerida nesta análise:** Adicionar definições locais e tabela de símbolos; remover notação desnecessária do resumo.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-A02 — Separar operador de amostragem e conjunto de saídas

**Fonte:** PDF de pareceres, página 3, Additional comments.  
**Natureza:** Texto e notação  
**Temas comuns:** T08, T11

**Original do revisor:**

> Clearly distinguish the sampling operator MΩ from the decoder range M.

**Tradução para português:**

Distinguir claramente o operador de amostragem MΩ do conjunto de saídas do decoder M.

**Trecho/local do manuscrito relacionado:**

**Página 9 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> The frozen decoder defines the learned target-window class M = Im(G) = n G(z) : z ∈Rko ⊆RN. (6) Target-well reconstruction therefore optimizes only z ∈Rk and produces by = G(bz) ∈M; it does not update the network parameters or search over arbitrary vectors in RN. Log-conditioned latent prior. Each training target is encoded as zi = E(eyi), and h : Rp →Rk is fitted from {(eui, zi)}ntr i=1. For the Ridge variant, […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 20–41.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

Labels neste trecho: `eq:measurement_operator`.

```latex
The coordinate-restriction operator
$M_{\Omega}:\mathbb{R}^{N}\rightarrow\mathbb{R}^{m}$ is represented by

\begin{equation}
M_{\Omega}
=
\begin{bmatrix}
\boldsymbol{e}_{\omega_1}^{\top}\\
\vdots\\
\boldsymbol{e}_{\omega_m}^{\top}
\end{bmatrix}
\in\mathbb{R}^{m\times N},
\qquad
M_{\Omega}\boldsymbol{y}
=
\begin{bmatrix}
y_{\omega_1}\\
\vdots\\
y_{\omega_m}
\end{bmatrix},
\label{eq:measurement_operator}
\end{equation}
```

**`sections/03_method.tex`, linhas 169–181.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `eq:decoder_range`.

```latex
\begin{equation}
\mathcal{M}
=
\operatorname{Im}(G)
=
\left\{
G(\boldsymbol{z}):
\boldsymbol{z}\in\mathbb{R}^{k}
\right\}
\subseteq
\mathbb{R}^{N}.
\label{eq:decoder_range}
\end{equation}
```

**Observação para a revisão:** As definições de M_Omega e do conjunto mathcal M já estão presentes; a questão é distingui-los com clareza na leitura e nos diagramas.

**Ação sugerida nesta análise:** Usar, por exemplo, R_G para o conjunto de saídas e MΩ para a seleção de coordenadas; uniformizar texto e figuras.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R1-m03`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-A03 — Redesenhar Figura 1

**Fonte:** PDF de pareceres, página 3, Additional comments.  
**Natureza:** Figura  
**Temas comuns:** T11

**Original do revisor:**

> Redraw Figure 1 so that the mappings from z_0 to G(z_0) and from the refined code to its decoded reconstruction are unambiguous. Explain the arrows and the curve representing the decoder range.

**Tradução para português:**

Redesenhar a Figura 1 para tornar inequívocas as aplicações de z0 em G(z0) e do código refinado em sua reconstrução. Explicar as setas e a curva que representa o conjunto de saídas do decoder.

**Trecho/local do manuscrito relacionado:**

**Página 10 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Figure 1: Geometry of the log-conditioned latent prior. Dense logs determine z0 = h(u), and test- time refinement moves the code to bz so that G(bz) ∈M better matches the sparse observations b through MΩ. target windows can be represented, and h uses dense-log context to center the latent search. Ridge is the primary low-capacity implementation; an MLP variant is evaluated in the sensitivity analysis. 3.3 Test-time […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 228–238.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `fig:conditional_prior_geometry`.

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.66\linewidth]
    {figures/view_conditional_latent_prior.png}
    \caption{Geometry of the log-conditioned latent prior. Dense logs determine
    $\boldsymbol{z}_0=h(\boldsymbol{u})$, and test-time refinement moves the code
    to $\widehat{\boldsymbol{z}}$ so that
    $G(\widehat{\boldsymbol{z}})\in\mathcal{M}$ better matches the sparse
    observations $\boldsymbol{b}$ through $M_{\Omega}$.}
    \label{fig:conditional_prior_geometry}
\end{figure}
```

**Observação para a revisão:** A Figura 1 é incluída por figures/view_conditional_latent_prior.png. Setas e símbolos internos exigem edição da figura, além do LaTeX.

**Ação sugerida nesta análise:** Separar estimativa inicial e refinada e representar explicitamente a comparação de MΩ G(z) com b.

**Cruzamento com outros revisores:** `R1-m03`. A associação é temática, não equivalência literal dos pedidos.


### R3-A04 — Densidade de probabilidade na Figura 3

**Fonte:** PDF de pareceres, página 3, Additional comments.  
**Natureza:** Figura e terminologia  
**Temas comuns:** T08, T11

**Original do revisor:**

> Use “probability density” in Figure 3 to avoid confusion with the bulk-density log.

**Tradução para português:**

Usar “probability density” na Figura 3 para evitar confusão com o perfil de densidade da rocha.

**Trecho/local do manuscrito relacionado:**

**Página 15 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> Figure 3: Empirical cross-well Vc distributions for the pooled training wells and the held-out F03-4 well. The held-out distribution is shifted toward lower clay-volume values while retaining partial overlap with the training distribution. Table 2: Pearson correlations between the available input channels and the target properties in the cross-well Vc and single-well F03-4 porosity benchmarks. dataset channel target […]

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 56–65.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

Labels neste trecho: `fig:eda_crosswell_distribution`.

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.68\linewidth]
    {eda_crosswell_vc_distribution.png}
    \caption{Empirical cross-well $V_c$ distributions for the pooled training
    wells and the held-out F03-4 well. The held-out distribution is shifted
    toward lower clay-volume values while retaining partial overlap with the
    training distribution.}
    \label{fig:eda_crosswell_distribution}
\end{figure}
```

**Observação para a revisão:** O eixo está dentro de figures/eda_crosswell_vc_distribution.png. Corrigir apenas a legenda LaTeX não altera o rótulo da imagem.

**Ação sugerida nesta análise:** Trocar o rótulo estatístico para Probability density e reservar Bulk density para RHOB.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R1-m03`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-A05 — Explicitar que Ridge é regressão Ridge

**Fonte:** PDF de pareceres, página 3, Additional comments.  
**Natureza:** Texto e método  
**Temas comuns:** T07, T08

**Original do revisor:**

> Introduce Ridge explicitly as ridge regression and state its role in the workflow.

**Tradução para português:**

Introduzir Ridge explicitamente como regressão Ridge e indicar seu papel no fluxo.

**Trecho/local do manuscrito relacionado:**

**Página 9 do PDF.** Passagem relacionada localizada nesta análise; o parecer geral não necessariamente indicou esta página.

> For the Ridge variant, h(eu) = W eu + c, with parameters (c W,bc) ∈arg min W,c ( ntr ntr X i=1 ∥zi −W eui −c∥2 2 + α∥W∥2 F ) , (7) where α ≥0 controls the Ridge penalty [Hoerl and Kennard, 1970]. After fitting, h(eu) = c W eu+bc, and, suppressing tildes, the contextual anchor is z0 = h(u) ∈Rk. The decoder G determines which

**Localização no LaTeX recebido:**

**Passagens relacionadas confirmadas no projeto.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 93–97.**

```latex
evidence. With Ridge as a low-capacity latent-prior reference, CLP-CSGM achieves
lower mean RMSE than the direct autoencoder baseline in seven of nine cross-well
training-size--measurement-ratio cells. The unweighted mean of the nine
cell-level RMSE values decreases from 0.0350 to 0.0286, an overall reduction of
about 18\%; the largest gains occur in the most data-limited cells. In the
```

**`sections/03_method.tex`, linhas 189–226.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `eq:ridge_latent_training_objective`.

```latex
\paragraph{Log-conditioned latent prior.}
Each training target is encoded as
$\boldsymbol{z}_i=E(\widetilde{\boldsymbol{y}}_i)$, and
$h:\mathbb{R}^{p}\rightarrow\mathbb{R}^{k}$ is fitted from
$\{(\widetilde{\boldsymbol{u}}_i,\boldsymbol{z}_i)\}_{i=1}^{n_{\mathrm{tr}}}$.
For the Ridge variant,
$h(\widetilde{\boldsymbol{u}})
=W\widetilde{\boldsymbol{u}}+\boldsymbol{c}$, with parameters

\begin{equation}
(\widehat{W},\widehat{\boldsymbol{c}})
\in
\operatorname*{arg\,min}_{W,\boldsymbol{c}}
\left\{
\frac{1}{n_{\mathrm{tr}}}
\sum_{i=1}^{n_{\mathrm{tr}}}
\left\|
\boldsymbol{z}_i
-
W\widetilde{\boldsymbol{u}}_i
-
\boldsymbol{c}
\right\|_2^2
+
\alpha\|W\|_F^2
\right\},
\label{eq:ridge_latent_training_objective}
\end{equation}

where $\alpha\geq 0$ controls the Ridge penalty
\citep{hoerl1970ridge}. After fitting,
$h(\widetilde{\boldsymbol{u}})
=\widehat{W}\widetilde{\boldsymbol{u}}
+\widehat{\boldsymbol{c}}$, and, suppressing tildes, the contextual anchor is
$\boldsymbol{z}_0=h(\boldsymbol{u})\in\mathbb{R}^{k}$. The decoder $G$
determines which target windows can be represented, and $h$ uses dense-log
context to center the latent search. Ridge is the primary low-capacity
implementation; an MLP variant is evaluated in the sensitivity analysis.
```

**Ação sugerida nesta análise:** Explicar que Ridge estima o código latente a partir das entradas; diferenciá-la do comparador de regressão direta.

**Cruzamento com outros revisores:** `R1-M06`, `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### Anotações do revisor 3 no manuscrito

O ZIP atribui este PDF ao Reviewer 3. O autor interno das anotações aparece como “Author”; a atribuição ao revisor 3 vem da pasta do pacote, não desse campo de metadados.


### R3-P01 — Objetivo pouco claro no resumo

**Fonte:** PDF anotado do manuscrito, página 1.  
**Natureza:** Texto  
**Temas comuns:** T08, T01

**Original do revisor:**

> here i think it is hard to understand the goal. consider rephrase it. you have a contiuous well log and a few sparse core data, the goal is to have a continuous prediction over that property, like permeability

**Tradução para português:**

Aqui está difícil entender o objetivo. Considere reformular: há um perfil contínuo e alguns dados esparsos de testemunho; o objetivo é prever continuamente essa propriedade, por exemplo permeabilidade.

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Localização da anotação do revisor.

> Dense well logs provide continuous indirect observations of subsurface properties, whereas target-property curves may be available only at selected depths. Direct supervised models can combine dense logs with sparse target observations, but they typically use the sparse entries as input features and do not require the reconstruction to honor them as measurements. We introduce Conditional Latent-Prior Compressed […]

**Texto extraído diretamente do destaque no PDF:**

> Dense well logs provide continuous indirect observations of subsurface properties, whereas target-property curves may be available only at selected depths. Direct supervised models can combine dense logs with sparse target observations, but they typically use the sparse entries as input features and do not require the reconstruction to honor them as measurements.

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 79–89.**

```latex
Dense well logs provide continuous indirect observations of subsurface
properties, whereas target-property curves may be available only at selected
depths. Direct supervised models can combine dense logs with sparse target
observations, but they typically use the sparse entries as input features and do
not require the reconstruction to honor them as measurements. We introduce
Conditional Latent-Prior Compressed Sensing with Generative Models (CLP-CSGM), a
latent inverse-problem framework that separates these roles. A decoder learned
from target-property windows defines the admissible reconstruction range. Dense
wireline logs predict a context-dependent latent anchor, and sparse target-log
observations refine the estimate through an explicit data-fidelity term at
inference time.
```

**Ação sugerida nesta análise:** Apresentar primeiro a tarefa. Não adotar a referência a testemunhos como descrição dos experimentos principais, que usam subamostragem de alvos processados.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M11`, `R1-m01`, `R1-m02`, `R2-01`, `R2-02`. A associação é temática, não equivalência literal dos pedidos.


### R3-P02 — Enquadramento como problema inverso

**Fonte:** PDF anotado do manuscrito, página 1.  
**Natureza:** Justificativa conceitual  
**Temas comuns:** T08, T10

**Original do revisor:**

> the probelm doesn’t sound like an inverse problem, or it i s very weakly linked to inverse problem. the position operator M just mask a few points, from the prediciton to get the actual sparse observation. it is better to call it missing data reconstruciton.

**Tradução para português:**

O problema não parece inverso, ou está apenas fracamente ligado a problemas inversos. O operador de posição M apenas mascara pontos da previsão para obter a observação esparsa. Seria melhor chamá-lo de reconstrução de dados ausentes.

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Localização da anotação do revisor.

> latent inverse-problem framework that separates these roles. A decoder learned from target-property windows defines the admissible reconstruction range. Dense wireline logs predict a context-dependent latent anchor, and sparse target-log observations refine the estimate through an explicit data-fidelity term at inference time. We evaluate CLP-CSGM on public Dutch North Sea well-log data reorganized into a […]

**Texto extraído diretamente do destaque no PDF:**

> inverse-problem framework

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 84–85.**

```latex
Conditional Latent-Prior Compressed Sensing with Generative Models (CLP-CSGM), a
latent inverse-problem framework that separates these roles. A decoder learned
```

**`sections/03_method.tex`, linhas 243–276.**

Contexto no fonte: `\section{Method}`; `\subsection{Test-time reconstruction}`.

Labels neste trecho: `eq:clp_csgm_objective`.

```latex
At inference time, $G$ and $h$ remain fixed, and only the latent code of the
current target window is optimized. Given standardized dense logs
$\widetilde{\boldsymbol{u}}\in\mathbb{R}^{p}$ and sparse target observations
$\widetilde{\boldsymbol{b}}\in\mathbb{R}^{m}$, define
$\boldsymbol{z}_0=h(\widetilde{\boldsymbol{u}})$. CLP-CSGM computes

\begin{equation}
\begin{aligned}
\widehat{\boldsymbol{z}}_{\mathrm{num}}
&\in
\operatorname*{arg\,min}_{\boldsymbol{z}\in\mathbb{R}^{k}}
\left\{
\left\|
M_{\Omega}G(\boldsymbol{z})
-
\widetilde{\boldsymbol{b}}
\right\|_2^2
+
\lambda
\left\|
\boldsymbol{z}
-
\boldsymbol{z}_0
\right\|_2^2
\right\},
\\
\widehat{\boldsymbol{y}}
&=
\boldsymbol{\mu}_y
+
D_yG\left(\widehat{\boldsymbol{z}}_{\mathrm{num}}\right).
\end{aligned}
\label{eq:clp_csgm_objective}
\end{equation}
```

**Ação sugerida nesta análise:** Explicar a não unicidade de recuperar N valores de m<N observações. É possível manter a formulação inversa e nomear com clareza a tarefa de reconstrução sob dados ausentes, sem alegar inversão física não modelada.

**Cruzamento com outros revisores:** `R1-M07`, `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P03 — Conjunto admissível de reconstrução

**Fonte:** PDF anotado do manuscrito, página 1.  
**Natureza:** Texto  
**Temas comuns:** T08

**Original do revisor:**

> very difficult to understand

**Tradução para português:**

Muito difícil de entender.

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Localização da anotação do revisor.

> admissible reconstruction range. Dense wireline logs predict a context-dependent latent anchor, and sparse target-log observations refine the estimate through an explicit data-fidelity term at inference time. We evaluate CLP-CSGM on public Dutch North Sea well-log data reorganized into a held-out-well clay-volume benchmark and a single-well porosity stress test with weak dense context. A limited-sample permeability […]

**Texto extraído diretamente do destaque no PDF:**

> admissible reconstruction range

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 85–89.**

```latex
latent inverse-problem framework that separates these roles. A decoder learned
from target-property windows defines the admissible reconstruction range. Dense
wireline logs predict a context-dependent latent anchor, and sparse target-log
observations refine the estimate through an explicit data-fidelity term at
inference time.
```

**`sections/03_method.tex`, linhas 169–187.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `eq:decoder_range`.

```latex
\begin{equation}
\mathcal{M}
=
\operatorname{Im}(G)
=
\left\{
G(\boldsymbol{z}):
\boldsymbol{z}\in\mathbb{R}^{k}
\right\}
\subseteq
\mathbb{R}^{N}.
\label{eq:decoder_range}
\end{equation}

Target-well reconstruction therefore optimizes only
$\boldsymbol{z}\in\mathbb{R}^{k}$ and produces
$\widehat{\boldsymbol{y}}
=G(\widehat{\boldsymbol{z}})\in\mathcal{M}$; it does not update the network
parameters or search over arbitrary vectors in $\mathbb{R}^{N}$.
```

**Ação sugerida nesta análise:** O comentário marca admissible reconstruction range: explicar como conjunto de curvas que o decoder pode produzir; não sugerir garantia de admissibilidade física.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P04 — Significado de rho no resumo

**Fonte:** PDF anotado do manuscrito, página 1.  
**Natureza:** Texto e notação  
**Temas comuns:** T08

**Original do revisor:**

> what is rho here?

**Tradução para português:**

O que é rho aqui?

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Localização da anotação do revisor.

> rho=0.20\). The direct autoencoder becomes competitive as more target entries are observed. Ablations against prior-only and measurement-only CSGM variants indicate that contextual latent anchoring and sparse-measurement consistency make complementary contributions whose relative importance varies across regimes. The preferred latent-prior architecture also depends on the dataset: an MLP prior yields lower mean RMSE […]

**Texto extraído diretamente do destaque no PDF:**

> \(\rho=0.20\).

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 98–100.**

```latex
GR-only porosity benchmark, CLP-CSGM has the lowest mean RMSE at the sparsest
ratios, including 0.016 versus 0.018 at
\(\rho=0.20\). The direct autoencoder becomes competitive as more target entries
```

**`sections/03_method.tex`, linhas 13–18.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

```latex
Let $\boldsymbol{y}\in\mathbb{R}^{N}$ denote a target-property window of $N$
depth samples and let $\boldsymbol{u}\in\mathbb{R}^{p}$ denote the vectorized
dense-log channels over the same interval. Target values are observed at the
ordered coordinate set
$\Omega=\{\omega_1,\ldots,\omega_m\}\subset\{1,\ldots,N\}$, where
$1\leq m<N$ and the measurement ratio is $\rho=m/N$.
```

**Observação para a revisão:** A definição rho=m/N está no método, mas não acompanha sua ocorrência no resumo. A localização posterior não resolve necessariamente a dúvida do leitor no resumo.

**Ação sugerida nesta análise:** Definir a fração de observações ou usar porcentagem em palavras no resumo. Conferir também fração nominal e efetiva devido ao arredondamento e ao mínimo de quatro pontos.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P05 — Contexto fraco de um único canal

**Fonte:** PDF anotado do manuscrito, página 1.  
**Natureza:** Texto  
**Temas comuns:** T08, T02

**Original do revisor:**

> what is weak single-channel context?

**Tradução para português:**

O que significa contexto fraco de um único canal?

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Localização da anotação do revisor.

> weak single-channel context. CLP-CSGM thus provides a complementary sparse-assimilation framework for petrophysical settings with limited training support, cross-well shift, or incomplete dense context. Powered by Editorial Manager® and ProduXion Manager® from Aries Systems Corporation

**Texto extraído diretamente do destaque no PDF:**

> weak single-channel context.

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 104–109.**

```latex
contributions whose relative importance varies across regimes. The preferred
latent-prior architecture also depends on the dataset: an MLP prior yields lower
mean RMSE in the cross-well grid, whereas Ridge yields lower mean RMSE under
weak single-channel context. CLP-CSGM thus provides a complementary
sparse-assimilation framework for petrophysical settings with limited training
support, cross-well shift, or incomplete dense context.
```

**`sections/05_experiments.tex`, linhas 47–60.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Single-well F03-4 GR-only porosity benchmark}`.

```latex
The second benchmark reconstructs length-$N=64$ windows of the processed
density-derived F03-4 porosity curve using gamma ray as the only dense input.
Acoustic travel time is excluded because it is nearly collinear with the target
(Pearson $r\approx0.9996$) and would provide a near-direct proxy. The resulting
GR-only setting is a weak-context test. The target follows the processing of
\citet{zhang2025rockphysicslogs}, in which the density curve underlying F03-4
porosity is itself predicted from sonic and gamma ray.

Sparse observations are sampled at
$\rho\in\{0.20,0.30,0.40,0.50,0.60\}$. Depth is partitioned before window
construction, preventing overlap among training, validation, and test windows.
The benchmark is a controlled weak-context sparse-assimilation test. It is not
presented as a recommendation of GR-only input for operational porosity
prediction.
```

**Ação sugerida nesta análise:** Dizer que o teste usa somente GR como entrada contínua e explicar o objetivo dessa restrição.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M02`, `R1-M05`, `R1-M09`, `R1-m01`, `R1-m02`, `R1-m04`, `R2-01`, `R2-02`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


### R3-P06 — Regressão Ridge e prior de baixa capacidade

**Fonte:** PDF anotado do manuscrito, página 1.  
**Natureza:** Texto e método  
**Temas comuns:** T07, T08

**Original do revisor:**

> are you using ridge regression? please make it easier to understand

**Tradução para português:**

Vocês estão usando regressão Ridge? Por favor, tornem isso mais fácil de entender.

**Trecho/local do manuscrito relacionado:**

**Página 1 do PDF.** Localização da anotação do revisor.

> With Ridge as a low-capacity latent-prior reference, CLP-CSGM achieves lower mean RMSE than the direct autoencoder baseline in seven of nine cross-well training-size--measurement-ratio cells. The unweighted mean of the nine cell-level RMSE values decreases from 0.0350 to 0.0286, an overall reduction of about 18\%; the largest gains occur in the most data-limited cells. In the GR-only porosity benchmark, CLP-CSGM has […]

**Texto extraído diretamente do destaque no PDF:**

> low-capacity latent-prior

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`main.tex`, linhas 93–97.**

```latex
evidence. With Ridge as a low-capacity latent-prior reference, CLP-CSGM achieves
lower mean RMSE than the direct autoencoder baseline in seven of nine cross-well
training-size--measurement-ratio cells. The unweighted mean of the nine
cell-level RMSE values decreases from 0.0350 to 0.0286, an overall reduction of
about 18\%; the largest gains occur in the most data-limited cells. In the
```

**`sections/03_method.tex`, linhas 189–226.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `eq:ridge_latent_training_objective`.

```latex
\paragraph{Log-conditioned latent prior.}
Each training target is encoded as
$\boldsymbol{z}_i=E(\widetilde{\boldsymbol{y}}_i)$, and
$h:\mathbb{R}^{p}\rightarrow\mathbb{R}^{k}$ is fitted from
$\{(\widetilde{\boldsymbol{u}}_i,\boldsymbol{z}_i)\}_{i=1}^{n_{\mathrm{tr}}}$.
For the Ridge variant,
$h(\widetilde{\boldsymbol{u}})
=W\widetilde{\boldsymbol{u}}+\boldsymbol{c}$, with parameters

\begin{equation}
(\widehat{W},\widehat{\boldsymbol{c}})
\in
\operatorname*{arg\,min}_{W,\boldsymbol{c}}
\left\{
\frac{1}{n_{\mathrm{tr}}}
\sum_{i=1}^{n_{\mathrm{tr}}}
\left\|
\boldsymbol{z}_i
-
W\widetilde{\boldsymbol{u}}_i
-
\boldsymbol{c}
\right\|_2^2
+
\alpha\|W\|_F^2
\right\},
\label{eq:ridge_latent_training_objective}
\end{equation}

where $\alpha\geq 0$ controls the Ridge penalty
\citep{hoerl1970ridge}. After fitting,
$h(\widetilde{\boldsymbol{u}})
=\widehat{W}\widetilde{\boldsymbol{u}}
+\widehat{\boldsymbol{c}}$, and, suppressing tildes, the contextual anchor is
$\boldsymbol{z}_0=h(\boldsymbol{u})\in\mathbb{R}^{k}$. The decoder $G$
determines which target windows can be represented, and $h$ uses dense-log
context to center the latent search. Ridge is the primary low-capacity
implementation; an MLP variant is evaluated in the sensitivity analysis.
```

**Ação sugerida nesta análise:** Explicitar algoritmo, entrada e saída do prior, sem acumular qualificadores abstratos.

**Cruzamento com outros revisores:** `R1-M06`, `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P07 — Símbolos nas equações introdutórias

**Fonte:** PDF anotado do manuscrito, página 5.  
**Natureza:** Texto e notação  
**Temas comuns:** T08

**Original do revisor:**

> you need to explain the symbols in your equations

**Tradução para português:**

É necessário explicar os símbolos das equações.

**Trecho/local do manuscrito relacionado:**

**Página 5 do PDF.** Localização da anotação do revisor.

> Such a direct map [u, b] 7→y need not enforce MΩby ≈b at inference time. Here, sparse target values are instead treated as explicit reconstruction measurements. Compressed sensing represents observations through an explicit forward operator [Donoho, 2006, Candes et al., 2006a,b]. Fixed-basis sparsity may be restrictive when cross-well variability and distribution shift alter the signal representation [Bora et al., […]

**Texto extraído diretamente do destaque no PDF:**

> Such a direct map [u, b] 7→y 57 need not enforce MΩ by n by ≈b at inference time.

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/01_introduction.tex`, linhas 12–17.**

Contexto no fonte: `\section{Introduction}`.

```latex
A direct map
$[\boldsymbol{u},\boldsymbol{b}]\mapsto\boldsymbol{y}$ need not satisfy
$M_{\Omega}\widehat{\boldsymbol{y}}\approx\boldsymbol{b}$ at the measured
coordinates at inference time. We examine this computational role of sparse
target-well information; the study does not question the general validity of
supervised petrophysical prediction.
```

**`sections/01_introduction.tex`, linhas 43–62.**

Contexto no fonte: `\section{Introduction}`.

Labels neste trecho: `eq:intro_clp_csgm`.

```latex
\begin{equation}
\widehat{\boldsymbol{z}}
\in
\operatorname*{arg\,min}_{\boldsymbol{z}\in\mathbb{R}^{k}}
\left\{
\left\|
M_{\Omega}G(\boldsymbol{z})-\boldsymbol{b}
\right\|_2^2
+
\lambda
\left\|
\boldsymbol{z}-\boldsymbol{z}_0
\right\|_2^2
\right\},
\qquad
\widehat{\boldsymbol{y}}
=
G(\widehat{\boldsymbol{z}}).
\label{eq:intro_clp_csgm}
\end{equation}
```

**Ação sugerida nesta análise:** Definir u, b, y estimado, Ω e MΩ antes da utilização ou mover a equação para a seção de método.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P08 — Im(G) pode parecer parte imaginária

**Fonte:** PDF anotado do manuscrito, página 6.  
**Natureza:** Texto e notação  
**Temas comuns:** T08, T11

**Original do revisor:**

> The Im is like imaginary part of G where actually it indicates the range. please consider use another symbol

**Tradução para português:**

Im parece a parte imaginária de G, embora indique o conjunto imagem. Considere usar outro símbolo.

**Trecho/local do manuscrito relacionado:**

**Página 6 do PDF.** Localização da anotação do revisor.

> CSGM replaces fixed-basis sparsity with the range Im(G) of a learned generator or decoder [Bora

**Texto extraído diretamente do destaque no PDF:**

> Im(G)

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 107–116.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

```latex
Subsequent equations omit tildes when standardized coordinates are understood.
The formulation assumes that target windows are adequately approximated by the
decoder range
$\mathcal{M}=\operatorname{Im}(G)\subset\mathbb{R}^{N}$, with
$G:\mathbb{R}^{k}\rightarrow\mathbb{R}^{N}$ and $k<N$, and that the dense logs
contain enough information for
$h:\mathbb{R}^{p}\rightarrow\mathbb{R}^{k}$ to provide a useful anchor
$\boldsymbol{z}_0=h(\boldsymbol{u})$. Their adequacy is assessed empirically
through decoder reconstruction, structural ablations, and latent-prior
sensitivity.
```

**`sections/03_method.tex`, linhas 169–181.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `eq:decoder_range`.

```latex
\begin{equation}
\mathcal{M}
=
\operatorname{Im}(G)
=
\left\{
G(\boldsymbol{z}):
\boldsymbol{z}\in\mathbb{R}^{k}
\right\}
\subseteq
\mathbb{R}^{N}.
\label{eq:decoder_range}
\end{equation}
```

**Ação sugerida nesta análise:** A notação é matematicamente válida, mas pode ser substituída por range(G) ou R_G para facilitar a leitura.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R1-m03`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P09 — Ruído de medição e lambda

**Fonte:** PDF anotado do manuscrito, página 8.  
**Natureza:** Esclarecimento técnico  
**Temas comuns:** T09

**Original do revisor:**

> how do you set the measurement noise? and how do you determine the lambda, the regularization coefficient.

**Tradução para português:**

Como vocês definem o ruído de medição e determinam lambda, o coeficiente de regularização?

**Trecho/local do manuscrito relacionado:**

**Página 8 do PDF.** Localização da anotação do revisor.

> where η is additive noise. Under the Gaussian model, η ∼N(0, σ2 ηIm); the noiseless case has η = 0. Dense logs and sparse target observations have distinct roles. The dense-log vector u determines the latent anchor z0 = h(u); b enters the data-fidelity term ∥MΩG(z) −b∥2 2. Reconstruction thus combines contextual information with explicit target-well measurement evidence. Standardization and modeling assumptions. All […]

**Texto extraído diretamente do destaque no PDF:**

> where η is additive noise. Under the Gaussian model, η ∼N(0, σ2 η 2 ηIm); the noiseless case has η = 0.

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 49–65.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

Labels neste trecho: `eq:measurement_model`.

```latex
The sparse target measurements satisfy

\begin{equation}
\boldsymbol{b}
=
M_{\Omega}\boldsymbol{y}
+
\boldsymbol{\eta},
\qquad
\boldsymbol{b},\boldsymbol{\eta}\in\mathbb{R}^{m},
\label{eq:measurement_model}
\end{equation}

where $\boldsymbol{\eta}$ is additive noise. Under the Gaussian model,
$\boldsymbol{\eta}\sim
\mathcal{N}(\boldsymbol{0},\sigma_{\eta}^{2}I_m)$; the noiseless case has
$\boldsymbol{\eta}=\boldsymbol{0}$.
```

**`sections/03_method.tex`, linhas 278–289.**

Contexto no fonte: `\section{Method}`; `\subsection{Test-time reconstruction}`.

```latex
The data-fidelity term enforces sparse-measurement consistency, and the latent
penalty regularizes the search toward the log-conditioned anchor. The parameter
$\lambda\geq 0$ is selected on validation windows for each measurement ratio
and seed.

Because $G$ is nonlinear, the objective is generally nonconvex and is solved
with Adam \citep{kingma2015adam}. The first run starts from
$\boldsymbol{z}_0$, additional restarts use
$\boldsymbol{z}_0+\boldsymbol{\epsilon}_r$ with small Gaussian perturbations,
and the candidate with the lowest objective is retained. The resulting
$\widehat{\boldsymbol{z}}_{\mathrm{num}}$ is the numerical optimizer output,
not a certified global minimizer.
```

**`sections/05_experiments.tex`, linhas 124–127.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Training, validation, and test protocol}`.

```latex
Noise is applied after subsampling, with
$\sigma_{\eta}=0.01$ for cross-well $V_c$ and
$\sigma_{\eta}=0.02$ for F03-4 porosity, independently across coordinates and
windows.
```

**Ação sugerida nesta análise:** Referenciar os níveis de ruído e a grade de validação já informados e esclarecer sua escala, justificativa e valores selecionados.

**Cruzamento:** sem pedido equivalente explícito de outro revisor neste tema; consultar o mapa para comentários relacionados do próprio revisor.


### R3-P10 — Covariância

**Fonte:** PDF anotado do manuscrito, página 8.  
**Natureza:** Esclarecimento técnico  
**Temas comuns:** T09

**Original do revisor:**

> covariance?

**Tradução para português:**

Covariância?

**Trecho/local do manuscrito relacionado:**

**Página 8 do PDF.** Localização da anotação do revisor.

> Standardization and modeling assumptions. All preprocessing statistics are computed from training windows only. Let µy, sy and µu, su be the componentwise training means and standard deviations, with Dy = diag(sy) and Du = diag(su). We use ey = D−1 y (y−µy) and eu = D−1 u (u−µu). Sparse measurements are standardized with DΩ= MΩDyM⊤ Ωand eb = D−1 Ω(b −MΩµy); near-zero standard deviations use the implementation […]

**Texto extraído diretamente do destaque no PDF:**

> standard deviations, Sparse mea

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 62–65.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

```latex
where $\boldsymbol{\eta}$ is additive noise. Under the Gaussian model,
$\boldsymbol{\eta}\sim
\mathcal{N}(\boldsymbol{0},\sigma_{\eta}^{2}I_m)$; the noiseless case has
$\boldsymbol{\eta}=\boldsymbol{0}$.
```

**`sections/03_method.tex`, linhas 75–107.**

Contexto no fonte: `\section{Method}`; `\subsection{Problem setup}`.

Labels neste trecho: `eq:inverse_target_standardization`.

```latex
\paragraph{Standardization and modeling assumptions.}
All preprocessing statistics are computed exclusively from the training windows.
Let $\boldsymbol{\mu}_y,\boldsymbol{s}_y\in\mathbb{R}^{N}$ and
$\boldsymbol{\mu}_u,\boldsymbol{s}_u\in\mathbb{R}^{p}$ denote the
componentwise training means and standard deviations, with
$D_y=\operatorname{diag}(\boldsymbol{s}_y)$ and
$D_u=\operatorname{diag}(\boldsymbol{s}_u)$. The standardized target and
dense-log windows are
$\widetilde{\boldsymbol{y}}
=D_y^{-1}(\boldsymbol{y}-\boldsymbol{\mu}_y)$ and
$\widetilde{\boldsymbol{u}}
=D_u^{-1}(\boldsymbol{u}-\boldsymbol{\mu}_u)$.

The sparse measurements use the corresponding target-coordinate statistics:
$D_{\Omega}=M_{\Omega}D_yM_{\Omega}^{\top}$ and
$\widetilde{\boldsymbol{b}}
=D_{\Omega}^{-1}
(\boldsymbol{b}-M_{\Omega}\boldsymbol{\mu}_y)$.
Zero or nearly zero standard deviations are handled by the numerical safeguard
used in the implementation. All learned components and latent refinement operate
in standardized coordinates, and the final prediction is returned to the
original target scale through

\begin{equation}
\widehat{\boldsymbol{y}}
=
\boldsymbol{\mu}_y
+
D_y\widehat{\widetilde{\boldsymbol{y}}}.
\label{eq:inverse_target_standardization}
\end{equation}

Subsequent equations omit tildes when standardized coordinates are understood.
```

**`sections/03_method.tex`, linhas 340–368.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

Labels neste trecho: `prop:map`.

```latex
\begin{proposition}[MAP and Tikhonov interpretation]
\label{prop:map}
Suppressing tildes, assume
$p(\boldsymbol{b}\mid\boldsymbol{z},\boldsymbol{u})
=p(\boldsymbol{b}\mid\boldsymbol{z})$ and

$\boldsymbol{b}\mid\boldsymbol{z}
\sim
\mathcal{N}
\left(
M_{\Omega}G(\boldsymbol{z}),
\sigma_{\eta}^{2}I_m
\right)$,
while
$\boldsymbol{z}\mid\boldsymbol{u}
\sim
\mathcal{N}
\left(
\boldsymbol{z}_0,
\sigma_z^{2}I_k
\right)$,
where
$\boldsymbol{z}_0=h(\boldsymbol{u})$,
$\sigma_{\eta}>0$, and $\sigma_z>0$.
Then every MAP estimator satisfies

\begin{equation}
\widehat{\boldsymbol{z}}_{\mathrm{MAP}}
\in
```

**Observação para a revisão:** O texto apresenta ruído isotrópico e depois padronização por coordenada. Esses são os trechos a esclarecer quanto à escala e à covariância. A localização não constitui auditoria do código nem confirmação de erro na implementação.

**Ação sugerida nesta análise:** O comentário é ambíguo e marca a padronização. Explicar desvio padrão versus covariância; conferir a transformação da covariância do ruído e sua compatibilidade com a interpretação MAP. Essa auditoria adicional é nossa sugestão, não um pedido detalhado do revisor.

**Cruzamento:** sem pedido equivalente explícito de outro revisor neste tema; consultar o mapa para comentários relacionados do próprio revisor.


### R3-P11 — Identidade do baseline

**Fonte:** PDF anotado do manuscrito, página 9.  
**Natureza:** Texto e comparação  
**Temas comuns:** T06, T07

**Original do revisor:**

> also not clear what method was used as the baseline in this paper.

**Tradução para português:**

Também não está claro qual método foi usado como baseline no artigo.

**Trecho/local do manuscrito relacionado:**

**Página 9 do PDF.** Localização da anotação do revisor.

> For the Ridge variant, h(eu) = W eu + c, with parameters (c W,bc) ∈arg min W,c ( ntr ntr X i=1 ∥zi −W eui −c∥2 2 + α∥W∥2 F ) , (7) where α ≥0 controls the Ridge penalty [Hoerl and Kennard, 1970]. After fitting, h(eu) = c W eu+bc, and, suppressing tildes, the contextual anchor is z0 = h(u) ∈Rk. The decoder G determines which

**Texto extraído diretamente do destaque no PDF:**

> Ridge variant, ( 1 ntr X

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 223–226.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

```latex
$\boldsymbol{z}_0=h(\boldsymbol{u})\in\mathbb{R}^{k}$. The decoder $G$
determines which target windows can be represented, and $h$ uses dense-log
context to center the latent search. Ridge is the primary low-capacity
implementation; an MLP variant is evaluated in the sensitivity analysis.
```

**`sections/05_experiments.tex`, linhas 159–166.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
The direct baselines are ML only, which uses only $\boldsymbol{u}$; MLP
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts the target directly; PCA
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts target principal-component
scores; and AE $[\boldsymbol{u},\boldsymbol{b}]$, which predicts codes of a
separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference
configuration.
Prior-only Ridge and measurement-only CSGM are structural ablations, while the
MLP latent prior is used only for sensitivity analysis.
```

**Ação sugerida nesta análise:** Separar método proposto, prior interno, ablações e comparadores diretos em uma tabela de papéis e entradas.

**Cruzamento com outros revisores:** `R1-M06`, `R1-M08`, `R1-m04`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


### R3-P12 — Por que prever z em vez de y?

**Fonte:** PDF anotado do manuscrito, página 9.  
**Natureza:** Justificativa e evidência experimental  
**Temas comuns:** T06, T07

**Original do revisor:**

> any reason for mapping u to z instead of y directly? what is the point of compressing the data in here?

**Tradução para português:**

Há alguma razão para mapear u em z em vez de diretamente em y? Qual é a finalidade de comprimir os dados aqui?

**Trecho/local do manuscrito relacionado:**

**Página 9 do PDF.** Localização da anotação do revisor.

> Log-conditioned latent prior. Each training target is encoded as zi = E(eyi), and h : Rp →Rk is fitted from {(eui, zi)}ntr i=1. For the Ridge variant, h(eu) = W eu + c, with parameters (c W,bc) ∈arg min W,c ( ntr ntr X i=1 ∥zi −W eui −c∥2 2 + α∥W∥2 F ) , (7) where α ≥0 controls the Ridge penalty [Hoerl and Kennard, 1970]. After fitting, h(eu) = c W eu+bc, and, suppressing tildes, the contextual anchor is z0 = h(u) […]

**Texto extraído diretamente do destaque no PDF:**

> and h : Rp →Rk is fitted from {(eu eui, zi)}ntr i=1 ntr i=1

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 127–133.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

```latex
\paragraph{Generative decoder.}
Let
$E_{\theta_E}:\mathbb{R}^{N}\rightarrow\mathbb{R}^{k}$ and
$G_{\theta_G}:\mathbb{R}^{k}\rightarrow\mathbb{R}^{N}$, with $k<N$, denote
the encoder and decoder. Given standardized training targets
$\{\widetilde{\boldsymbol{y}}_i\}_{i=1}^{n_{\mathrm{tr}}}$, their parameters
are obtained from
```

**`sections/03_method.tex`, linhas 183–193.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

```latex
Target-well reconstruction therefore optimizes only
$\boldsymbol{z}\in\mathbb{R}^{k}$ and produces
$\widehat{\boldsymbol{y}}
=G(\widehat{\boldsymbol{z}})\in\mathcal{M}$; it does not update the network
parameters or search over arbitrary vectors in $\mathbb{R}^{N}$.

\paragraph{Log-conditioned latent prior.}
Each training target is encoded as
$\boldsymbol{z}_i=E(\widetilde{\boldsymbol{y}}_i)$, and
$h:\mathbb{R}^{p}\rightarrow\mathbb{R}^{k}$ is fitted from
$\{(\widetilde{\boldsymbol{u}}_i,\boldsymbol{z}_i)\}_{i=1}^{n_{\mathrm{tr}}}$.
```

**`sections/05_experiments.tex`, linhas 159–166.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Baselines and evaluation metrics}`.

```latex
The direct baselines are ML only, which uses only $\boldsymbol{u}$; MLP
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts the target directly; PCA
$[\boldsymbol{u},\boldsymbol{b}]$, which predicts target principal-component
scores; and AE $[\boldsymbol{u},\boldsymbol{b}]$, which predicts codes of a
separately trained target autoencoder. CLP-CSGM Ridge is the proposed reference
configuration.
Prior-only Ridge and measurement-only CSGM are structural ablations, while the
MLP latent prior is used only for sensitivity analysis.
```

**Ação sugerida nesta análise:** Explicar restrição de representação e refinamento; demonstrar o benefício com comparadores e diagnóstico. Menor dimensão não garante melhor desempenho.

**Cruzamento com outros revisores:** `R1-M06`, `R1-M08`, `R1-m04`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


### R3-P13 — Resultados intermediários

**Fonte:** PDF anotado do manuscrito, página 9.  
**Natureza:** Resultado adicional solicitado  
**Temas comuns:** T07

**Original do revisor:**

> any of these intermediate results were shown in the paper?

**Tradução para português:**

Algum desses resultados intermediários foi mostrado no artigo?

**Trecho/local do manuscrito relacionado:**

**Página 9 do PDF.** Localização da anotação do revisor.

> Offline training fits an autoencoder on standardized target windows and a latent-prior predictor on paired dense-log windows and encoded targets. All learned components remain frozen during target-well reconstruction. Generative decoder. Let EθE : RN →Rk and GθG : Rk →RN, with k < N, denote the encoder and decoder. Given standardized training targets {eyi}ntr i=1, their parameters are obtained from (bθE, bθG) ∈arg […]

**Texto extraído diretamente do destaque no PDF:**

> 3.2 Learned model component 148

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 307–331.**

Contexto no fonte: `\section{Method}`; `\subsection{Test-time reconstruction}`.

Labels neste trecho: `eq:measurement_only_ablation`.

```latex
\paragraph{Ablation variants.}
The \emph{latent-prior-only} variant suppresses refinement and uses
$\widehat{\boldsymbol{z}}_{\mathrm{prior}}=\boldsymbol{z}_0$, followed by
$\widehat{\boldsymbol{y}}_{\mathrm{prior}}
=\boldsymbol{\mu}_y+
D_yG(\widehat{\boldsymbol{z}}_{\mathrm{prior}})$.
The \emph{measurement-only CSGM} variant sets $\lambda=0$ and computes

\begin{equation}
\widehat{\boldsymbol{z}}_{\mathrm{meas}}
\in
\operatorname*{arg\,min}_{\boldsymbol{z}\in\mathbb{R}^{k}}
\left\|
M_{\Omega}G(\boldsymbol{z})
-
\widetilde{\boldsymbol{b}}
\right\|_2^2.
\label{eq:measurement_only_ablation}
\end{equation}

The latent-prior-only variant isolates prediction from dense-log context. The
measurement-only variant recovers measurement-only generative compressed
sensing \citep{bora2017compressed}. Comparison with the full objective in
Eq.~\eqref{eq:clp_csgm_objective} separates the contributions of contextual
anchoring and sparse-measurement assimilation. Results are reported in
```

**`sections/06_results.tex`, linhas 215–235.**

Contexto no fonte: `\section{Results}`; `\subsection{Ablation analysis}`.

Labels neste trecho: `subsec:ablation_analysis`.

```latex
\subsection{Ablation analysis}
\label{subsec:ablation_analysis}

The structural ablations isolate the two terms of the CLP-CSGM objective.
Prior-only returns $G(h(\boldsymbol{u}))$ without refinement. Measurement-only
sets $\lambda=0$ and performs CSGM-style inversion without the log-conditioned
anchor \citep{bora2017compressed}. Comparing these variants with the full model
quantifies the contributions of contextual anchoring and explicit
sparse-measurement assimilation
(Remark~\ref{rem:csgm_relation}).

In the cross-well benchmark, CLP-CSGM Ridge has lower mean RMSE than prior-only
in all nine cells, with reductions from $20.5\%$ to $60.6\%$
(Table~\ref{tab:ablation_crosswell}). It also has lower mean RMSE than
measurement-only CSGM in seven cells. The exceptions are step $8$ with
$\rho\in\{0.05,0.10\}$, where measurement-only CSGM has lower mean RMSE and the
Ridge anchor is not beneficial. At steps $16$ and $32$, the full model has
lower mean RMSE than both ablations at every tested ratio.

\begin{table}[t]
    \centering
```

**Observação para a revisão:** A variante prior-only fornece um resultado intermediário relacionado, mas não equivale necessariamente a mostrar separadamente todos os estágios pedidos, como reconstrução do autoencoder, previsão inicial e refinamento.

**Ação sugerida nesta análise:** Apresentar diagnóstico do autoencoder, previsão sem refinamento e previsão refinada; verificar resultados já presentes no suplemento antes de executar novos testes.

**Cruzamento com outros revisores:** `R1-M06`. A associação é temática, não equivalência literal dos pedidos.


### R3-P14 — Setas e símbolos da Figura 1

**Fonte:** PDF anotado do manuscrito, página 10.  
**Natureza:** Figura e notação  
**Temas comuns:** T08, T11

**Original do revisor:**

> also, the plot is very hard to understand. why z_tilda decode to G(z0)? the blue line below leads to nothing. what is M = Im(G)? M is the index function, lm(G) is the range of decoded measurements (the target).

**Tradução para português:**

O gráfico é muito difícil de entender. Por que z com til decodifica em G(z0)? A linha azul inferior não leva a nada. O que é M=Im(G)? M é a função de índice e Im(G) é o conjunto das medições decodificadas (o alvo).

**Trecho/local do manuscrito relacionado:**

**Página 10 do PDF.** Localização da anotação do revisor.

> Figure 1: Geometry of the log-conditioned latent prior. Dense logs determine z0 = h(u), and test- time refinement moves the code to bz so that G(bz) ∈M better matches the sparse observations b through MΩ. target windows can be represented, and h uses dense-log context to center the latent search. Ridge is the primary low-capacity implementation; an MLP variant is evaluated in the sensitivity analysis. 3.3 Test-time […]

**Texto extraído diretamente do destaque no PDF:**

> ent p G(bz s den t prior. D bz) ∈M

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 228–238.**

Contexto no fonte: `\section{Method}`; `\subsection{Learned model components}`.

Labels neste trecho: `fig:conditional_prior_geometry`.

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.66\linewidth]
    {figures/view_conditional_latent_prior.png}
    \caption{Geometry of the log-conditioned latent prior. Dense logs determine
    $\boldsymbol{z}_0=h(\boldsymbol{u})$, and test-time refinement moves the code
    to $\widehat{\boldsymbol{z}}$ so that
    $G(\widehat{\boldsymbol{z}})\in\mathcal{M}$ better matches the sparse
    observations $\boldsymbol{b}$ through $M_{\Omega}$.}
    \label{fig:conditional_prior_geometry}
\end{figure}
```

**Observação para a revisão:** A anotação refere-se à Figura 1, cujo arquivo é figures/view_conditional_latent_prior.png. A legenda e a chamada foram localizadas; o pedido também envolve elementos internos da imagem.

**Ação sugerida nesta análise:** Redesenhar os mapeamentos e distinguir os símbolos. A confusão do revisor entre conjunto de saídas e operador deve ser esclarecida; não assumir que os objetos matemáticos sejam iguais.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R1-m03`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P15 — Proposições confusas

**Fonte:** PDF anotado do manuscrito, página 12.  
**Natureza:** Texto e teoria  
**Temas comuns:** T10

**Original do revisor:**

> it is very confusing the author listed these propositions.

**Tradução para português:**

É muito confusa a forma como os autores listaram essas proposições.

**Trecho/local do manuscrito relacionado:**

**Página 12 do PDF.** Localização da anotação do revisor.

> Proposition 1 (MAP and Tikhonov interpretation). Suppressing tildes, assume p(b | z, u) = p(b | z) and b | z ∼N MΩG(z), σ2 ηIm  , while z | u ∼N z0, σ2 zIk  , where z0 = h(u), ση > 0, and σz > 0. Then every MAP estimator satisfies bzMAP ∈arg min z∈Rk n ∥MΩG(z) −b∥2 2 + λ ∥z −z0∥2 o , λ = σ2 η σ2z . (10) Thus, CLP-CSGM is a Tikhonov-regularized latent inverse problem. Its center is the log- conditioned anchor, not […]

**Texto extraído diretamente do destaque no PDF:**

> Proposition 1 (MAP and Tikhonov interpretation). Suppressing tildes, assume p(b | z, u) = p(b | 184

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/03_method.tex`, linhas 334–337.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

Labels neste trecho: `subsec:interpretation_scope`.

```latex
\subsection{Interpretation and theoretical context}
\label{subsec:interpretation_scope}

The theoretical results required to interpret CLP-CSGM are stated below.
```

**`sections/03_method.tex`, linhas 407–444.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

Labels neste trecho: `prop:existence`, `eq:existence_argmin`.

```latex
\begin{proposition}[Existence of a minimizer]
\label{prop:existence}
Assume that $G:\mathbb{R}^{k}\rightarrow\mathbb{R}^{N}$ is continuous and
$\lambda>0$. For fixed $M_{\Omega}$,
$\widetilde{\boldsymbol{b}}\in\mathbb{R}^{m}$, and
$\boldsymbol{z}_0\in\mathbb{R}^{k}$, the objective
$\mathcal{J}(\boldsymbol{z})
=
\|M_{\Omega}G(\boldsymbol{z})
-\widetilde{\boldsymbol{b}}\|_2^2
+
\lambda\|\boldsymbol{z}-\boldsymbol{z}_0\|_2^2$
is continuous and coercive. Consequently,

\begin{equation}
\operatorname*{arg\,min}_{\boldsymbol{z}\in\mathbb{R}^{k}}
\mathcal{J}(\boldsymbol{z})
\neq
\varnothing.
\label{eq:existence_argmin}
\end{equation}
\end{proposition}

For any global minimizer $\widehat{\boldsymbol{z}}$,
$\mathcal{J}(\widehat{\boldsymbol{z}})
\leq
\mathcal{J}(\boldsymbol{z}_0)
=
\|M_{\Omega}G(\boldsymbol{z}_0)
-\widetilde{\boldsymbol{b}}\|_2^2$.
Exact minimization therefore cannot worsen the optimized objective relative to
the latent-prior-only candidate, although it does not guarantee lower error with
respect to the unknown target. For
$\widehat{\boldsymbol{z}}_{\mathrm{num}}$, this comparison holds only when
$\mathcal{J}(\widehat{\boldsymbol{z}}_{\mathrm{num}})
\leq
\mathcal{J}(\boldsymbol{z}_0)$.

```

**`sections/03_method.tex`, linhas 595–610.**

Contexto no fonte: `\section{Method}`; `\subsection{Interpretation and theoretical context}`.

```latex
Proposition~\ref{prop:conditional_local_recovery} provides a sufficient
conditional interpretation, not a verified recovery guarantee for the trained
decoder. The latent penalty localizes the relevant S-REC condition around
$\boldsymbol{z}_0$. The bound separates decoder representation error,
measurement residual and noise, latent-prior mismatch, and local measurement
separability. This formulation differs from classical RIP analysis for
fixed-basis sparse vectors and follows the decoder-range perspective of
generative compressed sensing
\citep{candes2006stable,bora2017compressed}.

The proposition does not establish that coordinate restriction satisfies
S-REC on the trained decoder range; this depends on local decoder secants not
being concentrated on a few coordinates. The complete proof, the extension to
an inexact numerical optimizer, and the corresponding empirical separability
diagnostics are provided in
Appendix~\ref{app:conditional_local_recovery} and
```

**Observação para a revisão:** Os três resultados estão na subseção Interpretation and theoretical context: MAP/Tikhonov, existência e recuperação condicional. As provas estão em sections/09_appendix.tex, com labels app:map_proof, app:existence_proof e app:conditional_local_recovery.

**Ação sugerida nesta análise:** Introduzir a finalidade de cada resultado e reduzir o peso das derivações clássicas no corpo; manter as hipóteses explícitas.

**Cruzamento com outros revisores:** `R1-M07`. A associação é temática, não equivalência literal dos pedidos.


### R3-P16 — Significado de 30 dB

**Fonte:** PDF anotado do manuscrito, página 14.  
**Natureza:** Esclarecimento técnico  
**Temas comuns:** T09

**Original do revisor:**

> what do you mean by 30dB?

**Tradução para português:**

O que vocês querem dizer com 30 dB?

**Trecho/local do manuscrito relacionado:**

**Página 14 do PDF.** Localização da anotação do revisor.

> The cross-well benchmark reconstructs Vc from 30 dB noisy logs, using F02-1, F03-2, and F06- 1 for development and F03-4 for testing. The single-well benchmark reconstructs F03-4 porosity from gamma ray alone. In both, b is obtained by coordinate subsampling of the processed target. Appendix D provides a limited-sample Lapa permeability case based on Alvarez et al. [2025]. Table 1: Summary of the two F3 benchmarks. […]

**Texto extraído diretamente do destaque no PDF:**

> 30 dB noisy logs,

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 25–30.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
The cross-well benchmark reconstructs $V_c$ from $30~\mathrm{dB}$ noisy logs,
using F02-1, F03-2, and F06-1 for model development and holding out F03-4 for
testing. The single-well benchmark reconstructs F03-4 porosity using gamma ray
as the only dense input. In both benchmarks, the sparse target vector
$\boldsymbol{b}$ is generated by coordinate subsampling of the corresponding
processed target curve.
```

**`sections/05_experiments.tex`, linhas 124–127.**

Contexto no fonte: `\section{Experimental setup}`; `\subsection{Training, validation, and test protocol}`.

```latex
Noise is applied after subsampling, with
$\sigma_{\eta}=0.01$ for cross-well $V_c$ and
$\sigma_{\eta}=0.02$ for F03-4 porosity, independently across coordinates and
windows.
```

**Observação para a revisão:** A referência a 30 dB diz respeito aos logs; os valores sigma_eta do protocolo descrevem ruído acrescentado às observações esparsas. A redação deve distinguir as duas operações e explicitar a definição de SNR utilizada.

**Ação sugerida nesta análise:** Definir SNR e como o ruído é gerado por canal; separar ruído dos perfis de entrada e ruído das observações esparsas.

**Cruzamento:** sem pedido equivalente explícito de outro revisor neste tema; consultar o mapa para comentários relacionados do próprio revisor.


### R3-P17 — Definição de Vc

**Fonte:** PDF anotado do manuscrito, página 14.  
**Natureza:** Texto e notação  
**Temas comuns:** T08, T01

**Original do revisor:**

> what is Vc?

**Tradução para português:**

O que é Vc?

**Trecho/local do manuscrito relacionado:**

**Página 14 do PDF.** Localização da anotação do revisor.

> Table 1: Summary of the two F3 benchmarks. The cross-well target Vc is derived from gamma ray through the Larionov relation, whereas the single-well target is processed porosity. dataset well target n_rows depth_min depth_max target_mean target_std target_min target_max cross_train F02-1 Vc 1149.7056 1174.6992 0.1215 0.0407 0.0513 0.1934 cross_train F03-2 Vc 951.9000 1073.4000 0.1184 0.0490 0.0393 0.2688 cross_train […]

**Texto extraído diretamente do destaque no PDF:**

> target Vc is processe

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 10–18.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
The experiments use processed well-log curves described by
\citet{zhang2025rockphysicslogs}; no new field data were acquired. We use
the FS2--FS6 intervals of wells F02-1, F03-2, F03-4, and F06-1 from the F3 block
in the Dutch North Sea. Sonic and gamma-ray logs are available in all four
wells; density is measured in F02-1 and F03-2 and predicted from sonic and gamma
ray in F03-4 and F06-1. The processed porosity target is density-derived, while
clay-volume fraction $V_c$ is obtained from gamma ray through the Larionov
relation. Geological, acquisition, and rock-physics details are provided in the
source study.
```

**`sections/04_data_and_eda.tex`, linhas 25–30.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
The cross-well benchmark reconstructs $V_c$ from $30~\mathrm{dB}$ noisy logs,
using F02-1, F03-2, and F06-1 for model development and holding out F03-4 for
testing. The single-well benchmark reconstructs F03-4 porosity using gamma ray
as the only dense input. In both benchmarks, the sparse target vector
$\boldsymbol{b}$ is generated by coordinate subsampling of the corresponding
processed target curve.
```

**Ação sugerida nesta análise:** Definir a propriedade, a escala e a procedência do alvo no primeiro uso e nas legendas pertinentes.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M11`, `R1-m01`, `R1-m02`, `R2-01`, `R2-02`. A associação é temática, não equivalência literal dos pedidos.


### R3-P18 — Density é ambíguo

**Fonte:** PDF anotado do manuscrito, página 15.  
**Natureza:** Figura e terminologia  
**Temas comuns:** T08, T11

**Original do revisor:**

> use probability density, pdf, or other words. the density is confusing as there is the RHOB density log.

**Tradução para português:**

Use densidade de probabilidade, PDF ou outra expressão. Density é confuso porque existe o perfil de densidade RHOB.

**Trecho/local do manuscrito relacionado:**

**Página 15 do PDF.** Localização da anotação do revisor.

> Figure 3: Empirical cross-well Vc distributions for the pooled training wells and the held-out F03-4 well. The held-out distribution is shifted toward lower clay-volume values while retaining partial overlap with the training distribution. Table 2: Pearson correlations between the available input channels and the target properties in the cross-well Vc and single-well F03-4 porosity benchmarks. dataset channel target […]

**Texto extraído diretamente do destaque no PDF:**

> Empirical cross-well Vc distributions held-out distribution is shifted towa

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/04_data_and_eda.tex`, linhas 56–65.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

Labels neste trecho: `fig:eda_crosswell_distribution`.

```latex
\begin{figure}[t]
    \centering
    \includegraphics[width=0.68\linewidth]
    {eda_crosswell_vc_distribution.png}
    \caption{Empirical cross-well $V_c$ distributions for the pooled training
    wells and the held-out F03-4 well. The held-out distribution is shifted
    toward lower clay-volume values while retaining partial overlap with the
    training distribution.}
    \label{fig:eda_crosswell_distribution}
\end{figure}
```

**Observação para a revisão:** A palavra Density está na imagem figures/eda_crosswell_vc_distribution.png. O trecho localizado identifica precisamente a figura que deve ser corrigida.

**Ação sugerida nesta análise:** Corrigir o rótulo do eixo da distribuição para Probability density.

**Cruzamento com outros revisores:** `R1-m01`, `R1-m02`, `R1-m03`, `R2-01`. A associação é temática, não equivalência literal dos pedidos.


### R3-P19 — Figura 7 e Vc derivado de GR

**Fonte:** PDF anotado do manuscrito, página 22.  
**Natureza:** Figura, justificativa e comparação  
**Temas comuns:** T01, T02, T06, T11

**Original do revisor:**

> very difficult to see the results. also, the Vc is not a very good target, it might be calculated from GR. you could  have predicted it almost perfectly using just GR, see the correlation 0.97

**Tradução para português:**

É muito difícil ver os resultados. Além disso, Vc não é um alvo muito bom: pode ter sido calculado de GR. Seria possível prevê-lo quase perfeitamente apenas com GR; veja a correlação 0,97.

**Trecho/local do manuscrito relacionado:**

**Página 22 do PDF.** Localização da anotação do revisor.

> Figure 7: Held-out F03-4 Vc reconstruction at step 16, seed 7, and ρ = 0.20. The target is the processed Larionov-derived curve, and the sparse observations are coordinate subsamples of that curve. absolute error and that the embargo changes training support, but neither modification reverses the ordering of CLP-CSGM Ridge against AE [u, b] and measurement-only CSGM at the tested ratios. 6.2 Single-well F03 GR-only […]

**Texto extraído diretamente do destaque no PDF:**

> Figure 7: Held-out F03-4 Vc reconstruction at step 16, seed 7, and ρ = 0.20. The target is the processed Larionov-derived curve, and the sparse observations are coordinate subsamples of that Figure 7: Held-out F03-4c reconstruction at step, seed, and. The target is the processed Larionov-derived curve, and the sparse observations are coordinate subsamples of that curve. proces curve.

O destaque pode cortar palavras ou fórmulas; o excerto contextual acima ajuda a identificar a passagem.

**Localização no LaTeX recebido:**

**Correspondência com a anotação e contexto explicativo.** Os arquivos e as linhas foram identificados nesta análise.

**`sections/06_results.tex`, linhas 88–95.**

Contexto no fonte: `\section{Results}`; `\subsection{Cross-well low-data results}`.

Labels neste trecho: `fig:crosswell_depth_profile`.

```latex
\begin{figure}[!htbp]
    \centering
    \includegraphics[width=\linewidth]{10_depth_profile_crosswell_vc.png}
    \caption{Held-out F03-4 $V_c$ reconstruction at step $16$, seed $7$, and
    $\rho=0.20$. The target is the processed Larionov-derived curve, and the
    sparse observations are coordinate subsamples of that curve.}
    \label{fig:crosswell_depth_profile}
\end{figure}
```

**`sections/04_data_and_eda.tex`, linhas 67–75.**

Contexto no fonte: `\section{Data and exploratory analysis}`; `\subsection{Field settings and benchmark provenance}`.

```latex
\paragraph{Channel-target correlations and input design.}
Table~\ref{tab:eda_correlations} reports descriptive Pearson correlations used
to justify the input channels; these values are not supplied to the models.
In the cross-well data, gamma ray is strongly correlated with $V_c$ because
the target is Larionov-derived from that log
\citep{zhang2025rockphysicslogs}. To avoid a near-direct shortcut, the main
dense input $\boldsymbol{u}$ excludes gamma ray and uses sonic, density,
acoustic impedance, and $V_p$. Gamma ray is reintroduced only in the
sensitivity analysis of Appendix~\ref{app:crosswell_gr_embargo}.
```

**`sections/09_appendix.tex`, linhas 585–594.**

Contexto no fonte: `\section{Supplementary Diagnostics}`; `\subsection{Cross-well Channel-Ablation and Embargo-Style Boundary Check}`.

```latex
Table~\ref{tab:crosswell_gr_embargo_robustness} shows three stable patterns.
Within each boundary protocol, including GR in \(u\) lowers absolute RMSE, as
expected because \(V_c\) is Larionov-derived from GR. Under the no-GR protocol
matching the main benchmark, CLP-CSGM Ridge has the lowest RMSE relative to AE
\([u,b]\) and measurement-only CSGM throughout the tested grid. The embargo-style
boundary check changes the absolute values without reversing this ordering.
These checks support the main interpretation: the cross-well \(V_c\) benchmark
tests sparse target-log reconstruction under distribution shift without a GR
shortcut in \(u\), and the reported gains are not solely an artifact of
train/validation boundary overlap.
```

**Observação para a revisão:** A Figura 7 usa figures/10_depth_profile_crosswell_vc.png. O apêndice com GR já oferece informação pertinente à crítica sobre o alvo derivado; ainda é necessário responder à legibilidade e à comparação adequada.

**Ação sugerida nesta análise:** Melhorar a figura e responder à origem derivada do alvo. Explicar GR excluído no cenário principal e discutir o cenário com GR; não usar correlação como prova de validade prática.

**Cruzamento com outros revisores:** `R1-M01`, `R1-M02`, `R1-M05`, `R1-M08`, `R1-M09`, `R1-M11`, `R1-m03`, `R1-m04`, `R2-02`, `R2-03`. A associação é temática, não equivalência literal dos pedidos.


## Síntese editorial — AE (não é um quarto revisor)

**AE-01 — Original da síntese do editor associado:**

> The reviewers agree that the manuscript addresses an important problem in petrophysical log reconstruction and presents a technically interesting latent-space framework supported by open code and thorough ablation studies.
> 
> However, an important concern of the reviewers is that the empirical validation remains too limited to support broad claims. The primary targets are derived quantities (e.g., clay volume and porosity), and the sparse observations are obtained through artificial subsampling, meaning that the experiments demonstrate reconstruction under controlled benchmark conditions rather than validation against independent measurements. The reviewers therefore recommend moderating claims regarding practical applicability and transferability. Still regarding applicability, one reviewer underlines that prediction of clay volume and porosity is a simple linear problem and a clear evidence over why the proposed ML workflow is superior to traditional linear methods is therefore needed. Additional concerns include the limited number of wells, the small number of random realizations, and the sensitivity of the results to observation placement. Lastly, maybe because of the use of Ai to edit the text, the manuscript is sometimes hard to understand, with the stacking of terminologies such as, for example, "low-capacity latent-prior reference", "admissible reconstruction range",etc. The paper readability should therefore be checked carefully (and if AI was used to improve the text then it should be acknoweldged according to the Elsevier policy).
> 
> Overall, the reviewers find the work interesting but believe that additional validation, stronger benchmarking, and more cautious interpretation of the results are needed to support the conclusions.

**Tradução:** Os revisores concordam que o manuscrito trata de um problema importante e apresenta uma abordagem latente tecnicamente interessante, apoiada por código aberto e ablações detalhadas. Contudo, a validação empírica é limitada para sustentar afirmações amplas. Os alvos principais são derivados e as observações são obtidas por subamostragem artificial; os experimentos demonstram reconstrução em benchmark controlado, não validação contra medições independentes. Pedem moderar alegações de aplicação e transferência e demonstrar vantagem sobre métodos lineares tradicionais. Também preocupam a quantidade de poços, as poucas realizações e a sensibilidade à posição das observações. O texto é difícil em alguns pontos pelo acúmulo de termos; sua legibilidade deve ser revisada e, se IA foi usada para melhorá-lo, isso deve ser declarado conforme a política da Elsevier. Consideram o trabalho interessante, mas pedem validação adicional, comparações mais fortes e interpretação mais cautelosa.

**Temas comuns:** T01, T02, T03, T04, T05, T06 e T08.

**Localização confirmada no LaTeX:** `sections/08_conclusion.tex`, linhas 58–70, contém a declaração de IA.

**Passagem relevante já presente:** página 30 do manuscrito, “Declaration of generative AI and AI-assisted technologies in the manuscript preparation process”. Há declaração explícita de uso de ChatGPT no texto e nas figuras. Logo, não se deve responder como se ela estivesse ausente; é necessário indicar sua localização e verificar sua adequação à política aplicável. Este levantamento não realizou auditoria dessa política.

## Cobertura e próximos passos

| Bloco | Quantidade de itens identificados |
|---|---:|
| R1 — maiores | 11 |
| R1 — menores | 5 |
| R2 — numerados | 5 |
| R3 — numerados | 6 |
| R3 — adicionais | 5 |
| R3 — anotações textuais do PDF | 19 |
| **Total de IDs de revisores** | **51** |
| AE — síntese editorial, separada | 1 |

O total de 51 IDs não representa 51 problemas independentes: inclui repetições entre o parecer geral e as anotações do mesmo revisor. A anotação sem texto foi conferida, mas não recebeu um pedido inventado.

A localização dos 51 IDs no projeto está concluída. O passo seguinte é revisar cada pedido e registrar seu estado: já atendido; precisa ser mais explícito; exige resultado adicional; ou requer justificativa fundamentada. Esta atualização não modifica o manuscrito nem afirma que novos experimentos foram executados.
