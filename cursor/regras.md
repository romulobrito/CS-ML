# Regras do projeto (Cursor)

## Log de conversas (`agent.log`)

- O arquivo `agent.log` na raiz do repositorio registra, de forma resumida e clara, cada iteracao do chat com o assistente.
- **Obrigatorio:** ao final de cada iteracao do chat (apos a resposta do assistente estar concluida para aquele turno), atualize `agent.log` acrescentando uma nova entrada com a estrutura abaixo.

### Estrutura de cada entrada

1. **Pergunta do usuario** — o que foi pedido (essencial, sem ruido).
2. **Raciocinio empregado** — como o problema foi interpretado e quais passos ou decisoes foram considerados.
3. **Conclusao** — resultado (arquivos criados/alterados, decisao tomada, ou pendencias).

### Formato sugerido (copiar e preencher)

```
--- [ISO-8601 data/hora UTC opcional]
Pergunta do usuario:
...

Raciocinio empregado:
...

Conclusao:
...
```

Use apenas caracteres ASCII nos registros, para evitar problemas de encoding entre ambientes.

---

## Producao de texto LaTeX (`paper/`)

Estas regras complementam o estilo ja usado em `experiments_synthetic.tex` e `main.tex`.

### Idioma e codificacao

- O draft atual do artigo esta em **ingles**; novas secoes no mesmo ficheiro devem manter o mesmo registo.
- Ficheiros `.tex`: **ASCII** em comentarios, labels, nomes de ficheiros referenciados e `\texttt{}` quando possivel; o `inputenc` UTF-8 pode ser usado no PDF final para simbolos matematicos, mas evite caracteres nao ASCII nos comentarios de codigo-fonte para consistencia entre ambientes.

### Ligacao aos resultados do pipeline

- Tabelas com metricas agregadas devem ser **rastreaveis** a um run concreto: indique `run_id` (ex. `20260417_114526`), caminho `outputs/.../summary.csv`, e o comando de perfil (`--profile phase0_baseline`, `--profile solver_comparison`, etc.).
- **Arredondamentos:** alinhar com o Phase~0 (tipicamente **tres casas decimais** para RMSE/MAE medios e desvio entre seeds), reproduzindo os valores do CSV, nao arredondamentos ad hoc.
- Quando uma nova corrida substituir numeros, atualize **tabela, legenda e paragrafo de protocolo** em conjunto.

### Figuras

- **Phase~0:** `paper/figures/phase0_baseline/` (graficos `01_` a `10_`). O `\graphicspath` em `main.tex` aponta para esta pasta; nao use o mesmo nome de ficheiro noutro diretorio sem caminho explicito.
- **Etapa 1 (solver comparison):** graficos por run em `paper/figures/solver_comparison/runs/<run_id>/`. Como os nomes `01_`...`10_` coincidem com o Phase~0, use **caminho relativo completo** no `\includegraphics{...}` (ver `experiments_solver_etapa1.tex`), em vez de depender apenas de `\graphicspath`.
- Opcional: para uma versao "canonica" sem `run_id` no caminho, copiar ou ligar simbolicamente a pasta da run para um nome estavel **e** documentar no comentario do `.tex`.

### Estrutura

- Novos blocos experimentais grandes: preferir `\input{nome.tex}` desde `main.tex` (como `experiments_solver_etapa1.tex`) em vez de inflar indefinidamente um unico ficheiro.
- Manter `\label{...}` unicos por documento; prefixos sugeridos: `sec:exp_`, `tab:`, `fig:`.

### Checklist ao fechar uma etapa experimental no paper

1. CSV e `PROTOCOL.txt` guardados em `outputs/.../runs/<run_id>/`.
2. Figuras copiadas ou referenciadas sob `paper/figures/...`.
3. Tabela e texto no `.tex` atualizados; `run_id` mencionado.
4. `paper/roadmap_proximos_passos.md`: checkboxes da etapa, se aplicavel.
5. Entrada correspondente em `agent.log` (ASCII).
