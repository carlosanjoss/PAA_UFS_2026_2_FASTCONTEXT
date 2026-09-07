# Relatório 1 — Merge Sort

Fase 1 do escopo de Mateus Almeida (`feature/classical-algorithms`).
Documento de handoff para revisão humana e auditoria externa.

## Objetivo

Implementar manualmente o Merge Sort que ordena candidatos de recuperação
segundo o critério canônico do projeto (score DESC; em empate, chunk_id ASC),
com contagem de comparações, sem usar `sorted()` / `list.sort()`, e totalmente
desacoplado dos contratos de retrieval (Wilson) e RAG (Carlos).

## Arquivos criados/alterados

| Arquivo | Situação | Conteúdo |
|---|---|---|
| `src/algorithms/ordering.py` | criado | critério canônico compartilhado (`precedes`, `default_key`) |
| `src/algorithms/merge_sort.py` | criado (antes vazio) | Merge Sort manual instrumentado |
| `tests/test_merge_sort.py` | criado (antes vazio) | 14 testes |
| `docs/CORRECTNESS.md` | criado (antes vazio) | prova de corretude do Merge Sort |

Nenhum arquivo de colegas foi alterado. Retrievers, contratos e RAG intactos.

## Assinatura das funções

```python
# src/algorithms/ordering.py
def default_key(item) -> tuple[float, str]         # extrai (score, chunk_id)
def precedes(key_a, key_b) -> bool                 # a vem antes de b?

# src/algorithms/merge_sort.py
@dataclass
class MergeSortStats:
    comparisons: int   # comparações de chave (operação elementar)
    merges: int        # número de operações de merge
    moves: int         # atribuições de elementos nos merges

def merge_sort(items, key=default_key) -> tuple[list, MergeSortStats]
def merge_sort_keys(keys) -> tuple[list, MergeSortStats]   # utilitário/testes
```

`merge_sort` não modifica a lista de entrada.

## Descrição do algoritmo

1. Pré-computa a chave `(score, chunk_id)` de cada item uma única vez.
2. Divide a lista ao meio (`mid = n // 2`) recursivamente até 0/1 elemento.
3. Intercala (`_merge`) as duas metades ordenadas comparando chaves com
   `precedes`, escolhendo `right` apenas quando ele precede estritamente `left`
   (mantém estabilidade em empates).
4. Reconstrói a lista de itens originais na ordem final.

## Regra de desempate

Centralizada em `ordering.precedes`: `score` decrescente; em empate,
`chunk_id` crescente. Reutilizável por Quick Sort e Top-k para evitar
divergência entre os três algoritmos.

## Como as comparações são contadas

Cada avaliação de `precedes(key_right, key_left)` dentro do laço de merge
incrementa `stats.comparisons`. Essa é a operação elementar analisada na
complexidade. `merges` e `moves` são métricas auxiliares.

## Recorrência

```text
T(N) = 2 T(N/2) + Θ(N)
```

O termo `Θ(N)` corresponde ao merge linear das duas metades.

## Complexidade

| Caso | Tempo | Espaço auxiliar |
|---|---|---|
| Melhor | Θ(N log N) | Θ(N) |
| Médio | Θ(N log N) | Θ(N) |
| Pior | Θ(N log N) | Θ(N) |

Espaço adicional: pilha de recursão Θ(log N). Comparações no pior caso
limitadas por `N · ⌈log₂ N⌉`.

## Prova / resumo de corretude

Detalhada em `docs/CORRECTNESS.md`. Resumo:

- **Invariante do merge:** `merged` mantém-se ordenada e contém sempre o menor
  elemento global ainda não consumido; em empate escolhe `left` (estabilidade).
- **Indução na recursão:** caso base 0/1 trivial; passo usa hipótese de indução
  nas metades + corretude do merge; a divisão cobre todos os elementos sem
  sobreposição, logo não há perda nem duplicação.

## Testes executados

Ambiente: Windows, Python 3.13.1, pytest 9.1.1, em `.venv` local (não versionado).

Comando:

```bash
python -m pytest tests/test_merge_sort.py -v
```

Resultado: **PASSOU — 14 passed in 0.20s.**

Matriz de casos:

| Caso | Teste |
|---|---|
| lista vazia | `test_empty_list` |
| um elemento | `test_single_element` |
| dois elementos (troca) | `test_two_elements_need_swap` |
| já ordenada | `test_already_sorted` |
| ordem inversa | `test_reverse_order` |
| desempate por chunk_id (exemplo do doc) | `test_tie_breaks_by_chunk_id_asc` |
| muitos empates | `test_many_ties` |
| scores negativos e zero | `test_negative_and_zero_scores` |
| equivalência com oráculo `sorted()` (50 casos aleatórios) | `test_equivalence_with_reference_random` |
| preservação de todos os elementos | `test_preserves_all_elements` |
| determinismo | `test_determinism` |
| entrada não mutada | `test_input_not_mutated` |
| limite superior de comparações | `test_comparison_count_upper_bound` |
| variante por chaves | `test_keys_variant` |

O oráculo `sorted()` é usado **apenas no teste** como referência; a
implementação avaliada não o utiliza.

## Exemplo de entrada/saída

Entrada (formato `(score, chunk_dict)` dos retrievers do Wilson):

```python
[(0.72, {"chunk_id": "chunk_10"}),
 (0.91, {"chunk_id": "chunk_02"}),
 (0.72, {"chunk_id": "chunk_04"})]
```

Saída ordenada:

```text
[(0.91, 'chunk_02'), (0.72, 'chunk_04'), (0.72, 'chunk_10')]
comparisons = 3   merges = 2   moves = 5
```

Coincide com o resultado esperado da seção 8 do documento de contexto.

## Limitações

- Assume `score` finito e `chunk_id` comparável como string (garantido pelo
  contrato do projeto).
- A integração com os retrievers (substituir o `.sort()` do Wilson pelo Merge
  Sort) **não foi feita** — por decisão registrada, mantém-se apenas o
  algoritmo puro nesta fase.

## Decisões tomadas pelo Kiro

1. Criado módulo `ordering.py` para centralizar o critério de desempate,
   evitando divergência futura entre Merge/Quick/Top-k.
2. `default_key` aceita tanto `(score, payload)` quanto objetos/dicts com
   `score`/`chunk_id`, mantendo o algoritmo agnóstico aos contratos.
3. Merge estável (escolhe `left` em empate), embora `chunk_id` único torne
   empates de chave inexistentes entre itens distintos — mantido por segurança.
4. Contadores expostos via `MergeSortStats` para alimentar
   `RetrievalMetrics.comparisons` / `sorting_time_ns` na integração futura.

## Itens que Mateus deve revisar manualmente

- Confirmar se a assinatura `(lista, stats)` é a preferida para a integração com
  os retrievers, ou se prefere um objeto único.
- Confirmar a decisão de manter os algoritmos puros e fazer a fiação nos
  retrievers em fase separada (alinhando com o Wilson).
- Validar se `docs/CORRECTNESS.md` deve permanecer na raiz de `docs/` (já
  existia vazio na main) ou ser espelhado em `docs/mateus/`.

## Estado de ambiente

- Docker Linux: **NÃO EXECUTADO** — Docker indisponível nesta máquina.
- Testes locais: **PASSOU** (14/14).
