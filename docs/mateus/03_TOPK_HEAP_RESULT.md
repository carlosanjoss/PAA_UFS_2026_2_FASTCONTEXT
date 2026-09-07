# Relatório 3 — Top-k com Min-Heap

Fase 3 do escopo de Mateus Almeida (`feature/classical-algorithms`).
Documento de handoff para revisão humana e auditoria externa.

## Objetivo

Selecionar os `k` melhores candidatos sem ordenar todos os `N`, segundo o
critério canônico do projeto (score DESC; em empate, chunk_id ASC), usando uma
min-heap **implementada manualmente** (sem `heapq` escondendo a lógica),
instrumentada, com saída final em ordem de ranking. Totalmente desacoplado dos
contratos de retrieval e RAG.

## Arquivos criados/alterados

| Arquivo | Situação | Conteúdo |
|---|---|---|
| `src/algorithms/topk_heap.py` | criado (antes vazio) | Top-k com min-heap manual |
| `tests/test_topk_heap.py` | criado | 18 testes |

Reutiliza `src/algorithms/ordering.py` (critério central). Nenhum arquivo de
colegas foi alterado.

## Estrutura da heap

Min-heap de pares `(chave, item)` com tamanho máximo `k`, armazenada em lista
(filhos de `i` em `2i+1` e `2i+2`).

Ponto-chave: a **raiz é o PIOR** item (o menos preferido) entre os `k` melhores
vistos até agora. A relação de ordem da heap é `_worse(a, b) = precedes(b, a)` —
ou seja, o "mínimo" da heap é o candidato de menor score (desempate: maior
chunk_id). Assim, ao processar um novo item:

- se a heap ainda tem menos de `k` elementos: insere e faz `sift_up`;
- se a heap está cheia: compara o novo item com a raiz (pior atual); se o novo
  **precede** a raiz, substitui a raiz e faz `sift_down`.

No fim, a heap contém exatamente os `k` melhores. `sift_up` e `sift_down` são
escritos à mão.

## Uso ou não de biblioteca — justificativa acadêmica

**Não** foi usado `heapq` nem qualquer estrutura pronta. Toda a lógica de heap
(`_sift_up`, `_sift_down`, manutenção da propriedade e extração ordenada) é
manual e instrumentável, atendendo à exigência de que o algoritmo Top-k seja
analisável e que as comparações não fiquem escondidas em biblioteca.

## Tratamento de casos de borda

| Caso | Comportamento |
|---|---|
| `k <= 0` | retorna `[]` (não constrói heap) |
| `items` vazio | retorna `[]` |
| `k = 1` | retorna o melhor item |
| `k >= N` | retorna todos os itens ordenados |
| empates de score | resolvidos por chunk_id ASC via `precedes`, inclusive na fronteira do corte |

Observação sobre `k=0`: o contrato `RetrievalResult` do Carlos exige `top_k > 0`.
Como este algoritmo é puro, ele apenas devolve `[]` para `k<=0`; o tratamento de
`k=0` na camada de retrieval (antes de construir um `RetrievalResult`) é
responsabilidade da integração e será coberto na fase de testes de retrieval.

## Ordenação final

A heap não está em ordem de ranking. `_heap_to_sorted` extrai repetidamente a
raiz (o pior restante) para o fim de uma lista auxiliar (heapsort) e depois
inverte, produzindo a ordem final **score DESC / chunk_id ASC**, coerente com o
que o retrieval espera.

## Instrumentação

- `comparisons`: comparações de chave (na comparação com a raiz e em cada
  comparação de `sift_up`/`sift_down`).
- `insertions`: itens inseridos enquanto a heap tinha espaço (`< k`).
- `replacements`: substituições da raiz (heap cheia, item melhor).
- `max_heap_size`: maior tamanho atingido pela heap (limitado por `k`).

## Complexidade

| Métrica | Custo | Justificativa |
|---|---|---|
| Tempo | O(N log k) | N itens; cada operação de heap custa O(log k) |
| Espaço | O(min(N, k)) | memória efetiva; O(k) é o limite superior |

Observação sobre memória: a heap nunca contém mais que `min(N, k)` elementos.
Portanto a memória efetiva é **O(min(N, k))**, com **O(k)** como limite superior
(atingido quando `N >= k`). Quando `k > N`, a heap guarda apenas os `N` itens.

Por que não é preciso ordenar todos os N: quando `k << N`, manter apenas os `k`
melhores numa heap de tamanho `k` evita o custo O(N log N) da ordenação integral;
processa-se cada item em O(log k) e descartam-se cedo os candidatos piores que o
pior dos k atuais. Esta é a base da configuração "Otimizada" (C) do projeto.

## Testes executados

Ambiente: Windows, Python 3.13.1, pytest 9.1.1, `.venv` local (não versionado).

Comando:

```bash
python -m pytest tests/test_topk_heap.py -v
```

Resultado: **PASSOU — 18 passed in 0.15s.**

Execução conjunta dos três algoritmos clássicos:

```bash
python -m pytest tests/test_merge_sort.py tests/test_quick_sort.py tests/test_topk_heap.py -q
# 49 passed
```

Matriz de casos:

| Caso | Teste |
|---|---|
| `k = 0` | `test_k_zero` |
| `k < 0` | `test_k_negative` |
| `k = 1` | `test_k_one` |
| `k = N` | `test_k_equals_n` |
| `k > N` | `test_k_greater_than_n` |
| lista vazia | `test_empty_list` |
| lista vazia e `k = 0` | `test_empty_list_and_k_zero` |
| saída ordenada DESC | `test_output_ordered_desc` |
| desempate por chunk_id | `test_tie_breaks_by_chunk_id_asc` |
| empate exatamente no corte | `test_ties_at_cutoff` |
| muitos empates | `test_many_ties` |
| scores negativos e zero | `test_negative_and_zero_scores` |
| os k são realmente os melhores (60 casos aleatórios) | `test_selects_actual_best_random` |
| tamanho do resultado limitado a k | `test_result_size_capped_at_k` |
| determinismo (resultado + comparações) | `test_determinism` |
| entrada não mutada | `test_input_not_mutated` |
| heap nunca excede k | `test_max_heap_size_bounded_by_k` |
| contadores de instrumentação | `test_instrumentation_counts` |

O oráculo `sorted()` + fatiamento é usado **apenas no teste** como referência.

## Exemplo de entrada/saída

Entrada (4 candidatos, `k = 2`):

```python
[(0.72, {"chunk_id": "chunk_10"}),
 (0.91, {"chunk_id": "chunk_02"}),
 (0.72, {"chunk_id": "chunk_04"}),
 (0.30, {"chunk_id": "chunk_07"})]
```

Saída:

```text
[(0.91, 'chunk_02'), (0.72, 'chunk_04')]
comparisons = 4   insertions = 2   replacements = 1   max_heap_size = 2
```

Entre os dois candidatos empatados em 0.72, entra `chunk_04` (chunk_id menor),
e a heap nunca ultrapassa tamanho 2 (= k).

## Limitações

- Assume `score` finito e `chunk_id` comparável como string.
- O tratamento de `k=0` na camada de retrieval (contrato exige `top_k > 0`) é da
  integração, não deste algoritmo puro.
- Integração com os retrievers **não foi feita** (decisão registrada).

## Decisões tomadas pelo Kiro

1. Min-heap com "pior no topo" (`_worse = precedes` invertido), permitindo
   descartar candidatos em O(log k) sem ordenar tudo.
2. `_heap_to_sorted` reordena a saída para ranking final sem usar `sorted()`.
3. Métricas escolhidas (`insertions`, `replacements`, `max_heap_size`) para
   evidenciar empiricamente o comportamento O(N log k) e O(k) nos experimentos.
4. Mantido o padrão `algoritmo(items) -> (resultado, Stats)` e o `ordering.py`.

## Itens que Mateus deve revisar manualmente

- Confirmar que o tratamento de `k=0` deve ocorrer na camada de retrieval (por
  causa de `top_k > 0` no contrato do Carlos) — será validado na fase de testes.
- Validar se as métricas do Top-k atendem às necessidades dos experimentos.

## Estado de ambiente

- Docker Linux: **NÃO EXECUTADO** — Docker indisponível nesta máquina.
- Testes locais: **PASSOU** (18/18; 49/49 com Merge e Quick Sort).
