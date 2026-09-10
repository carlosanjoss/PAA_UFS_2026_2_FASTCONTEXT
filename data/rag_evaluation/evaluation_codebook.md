# FastContext — Protocolo de avaliação humana cega

## Regras gerais

- Avalie cada item de forma independente.
- Não tente descobrir se a resposta veio de No-RAG, Optimized ou Semantic.
- Não consulte `hidden_manifest.csv`.
- Não compare respostas com outros avaliadores antes de concluir sua própria
  planilha.
- Preencha apenas números permitidos e, quando necessário, use `notes`.
- Não altere `blind_id`, `question`, `answer` ou `retrieved_context`.

## Avaliação de qualidade geral

Nesta planilha, as citações de chunks foram ocultadas para reduzir
vazamento da condição experimental.

### correctness

- `0`: resposta incorreta, contraditória ou essencialmente errada.
- `1`: parcialmente correta, mas contém erro, imprecisão ou limitação
  relevante.
- `2`: tecnicamente correta para a pergunta.

### completeness

- `0`: não responde o que foi pedido ou omite quase tudo que seria
  necessário.
- `1`: responde parcialmente, mas falta informação importante.
- `2`: resposta suficientemente completa para a pergunta proposta.

### clarity

- `0`: confusa, ambígua, incoerente ou difícil de usar.
- `1`: compreensível, mas com problemas de organização, precisão ou
  concisão.
- `2`: clara, direta e adequadamente formulada.

### hallucination_present

- `0`: não foi identificada afirmação factual inventada ou claramente não
  sustentada.
- `1`: existe pelo menos uma afirmação factual inventada, incorreta ou
  apresentada sem base plausível.

Se houver dúvida técnica relevante, registre a dúvida em `notes`;
não invente uma justificativa.

## Avaliação de groundedness

Nesta planilha, você recebe exatamente o Top-5 recuperado que foi enviado
ao modelo naquela execução.

### groundedness

- `0`: a resposta não é sustentada pelo contexto recuperado, contradiz o
  contexto ou faz afirmações substanciais ausentes das evidências.
- `1`: parte importante da resposta é sustentada, mas há afirmações
  relevantes não apoiadas, extrapolações ou cobertura incompleta.
- `2`: as afirmações substantivas da resposta são sustentadas pelo contexto
  recuperado ou a abstenção é apropriada diante do contexto.

A avaliação de groundedness não deve premiar uma resposta apenas porque
possui uma citação sintaticamente válida. O critério é sustentação factual
pelo conteúdo recuperado.

## Abstenções

Respostas que afirmam que o contexto é insuficiente devem ser avaliadas
pelo comportamento observado:

- uma abstenção pode receber boa avaliação se realmente não houver
  evidência suficiente;
- uma abstenção pode receber avaliação baixa se o contexto disponível
  responder claramente à pergunta.

## Independência

Cada item é avaliado por duas pessoas. Divergências serão analisadas
somente após o encerramento das avaliações independentes.
