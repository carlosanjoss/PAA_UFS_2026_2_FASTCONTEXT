# FastContext — Adjudicação cega

Cada item desta etapa apresentou divergência entre os dois avaliadores
independentes originais.

## Regras

- Não tente descobrir a condição experimental.
- Não consulte `adjudication_manifest.csv`.
- Não solicite os escores dos dois avaliadores anteriores.
- Avalie novamente o item de forma independente.
- Preencha somente as métricas listadas em `metrics_to_adjudicate`.
- Use exatamente as mesmas escalas do `evaluation_codebook.md`.
- Não altere `blind_id`, pergunta, resposta ou contexto.
- Use `adjudication_notes` somente quando necessário.

A pontuação do terceiro avaliador será usada como decisão de adjudicação
para a métrica em que houve divergência.
