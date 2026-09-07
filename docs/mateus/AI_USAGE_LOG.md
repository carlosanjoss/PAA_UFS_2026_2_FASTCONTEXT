# Registro de Uso de IA — Mateus Almeida

Declaração de uso de IA generativa na Atividade 1 de PAA (UFS 2026.2).

## Ferramenta

- **Ferramenta:** Kiro (assistente de desenvolvimento em IDE).
- **Modelo:** identificado no ambiente como "Auto" (seleção automática de modelo);
  o modelo exato não é exposto pela ferramenta.
- **Datas de uso:** setembro/2026 (Fases 0 a 6 do escopo de Mateus).

## Objetivo do uso

Auxiliar Mateus a concluir suas duas branches de responsabilidade
(`feature/classical-algorithms` e `feature/tests`), preservando o caráter
acadêmico: algoritmos implementados manualmente, analisáveis e testados.

## Tarefas em que a IA auxiliou

1. **Fase 0 — auditoria:** clonar o repositório, inspecionar branches, ler
   contratos de retrieval/RAG, rodar testes baseline e mapear divergências.
2. **Fase 1 — Merge Sort:** implementação manual instrumentada + testes + prova
   de corretude.
3. **Fase 2 — Quick Sort:** implementação manual (mediana de três) + testes.
4. **Fase 3 — Top-k:** min-heap manual + testes.
5. **Fase 4 — corretude/complexidade:** consolidação teórica ligada ao código.
6. **Fase 5 — testes:** consolidação, testes de busca contra as assinaturas do
   Wilson, integração de ranking, cobertura.
7. **Fase 6 — fechamento:** organização de commits, PRs e handoffs.

## Instruções/prompts principais dadas por Mateus

- Executar cada fase somente após aprovação humana explícita.
- Manter os algoritmos 100% desacoplados dos contratos de Wilson/Carlos.
- Não usar `sorted()`/`list.sort()` nos algoritmos; não usar `heapq` escondendo
  a lógica do Top-k.
- Não alterar retrievers, contratos, RAG, corpus ou chunking dos colegas.
- Pivô do Quick Sort por mediana de três, deixando claro que o pior caso segue
  O(N²).
- Registrar Docker como NÃO EXECUTADO enquanto indisponível.
- Não fazer merge automático; parar para revisão ao fim de cada fase.

## Sugestões da IA aproveitadas

- Centralizar o critério de desempate em `src/algorithms/ordering.py`
  (`precedes`, `default_key`), reutilizado pelos três algoritmos.
- Padrão de retorno uniforme `algoritmo(items) -> (resultado, Stats)`.
- Instrumentação: comparações, e métricas específicas por algoritmo
  (merges/moves; partitions/swaps/max_depth; insertions/replacements/max_heap_size).
- Min-heap "pior no topo" para o Top-k em O(N log k).
- Teste de regressão do contrato de ordenação sem acoplar aos retrievers.

## Sugestões rejeitadas / corrigidas por decisão humana

- Não integrar os algoritmos aos retrievers nesta etapa (manter puros).
- Não reconciliar os contratos divergentes de Wilson e Carlos (decisão de equipe).
- Manter `docs/CORRECTNESS.md` na raiz de `docs/` (não espelhar em `docs/mateus/`).

## Erros/limitações encontrados durante o processo

- Ambiente sem Docker: reprodutibilidade em Docker Linux não pôde ser validada.
- Python local 3.13.1 (README pede 3.11): não impediu os testes do escopo.
- Contrato de retrieval divergente entre `feature/retrieval` e `feature/rag`:
  identificado e registrado, não resolvido (fora do escopo de Mateus).

## Como os resultados foram validados

- Execução de `pytest` no `.venv` local a cada fase.
- 76 testes no escopo de Mateus, 100% de cobertura em `src/algorithms`.
- Comparação com oráculo `sorted()` **apenas nos testes**.
- Exemplos canônicos do documento de contexto reproduzidos.

## Decisões que foram humanas (de Mateus)

- Aprovar cada fase antes da seguinte.
- Escolher a estratégia de pivô e a política de desacoplamento.
- Definir a estratégia de commits/branches e a não realização de merge.
- Aceitar os handoffs como base para revisão externa e apresentação.

Nada neste registro afirma resultados não verificados. Estados de teste usam
rótulos explícitos (PASSOU / NÃO EXECUTADO).
