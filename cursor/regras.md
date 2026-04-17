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
