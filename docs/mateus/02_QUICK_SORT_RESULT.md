# Relatório 2 — Quick Sort

Fase 2 do escopo de Mateus Almeida (`feature/classical-algorithms`).
Documento de handoff para revisão humana e auditoria externa.

## Objetivo

Implementar manualmente o Quick Sort como algoritmo clássico de comparação com
o Merge Sort, ordenando candidatos segundo o critério canônico do projeto
(score DESC; em empate, chunk_id ASC), com pivô determinístico documentado,
particionamento manual e instrumentação. Sem `sorted()` / `list.sort()`.
Totalmente desacoplado dos contratos de retrieval e RAG.

## Arquivos criados/alterados

| Arquivo | Situação | Conteúdo |
|---|---|---|
| `src/algorithms/quick_sort.py` | criado (antes vazio) | Quick Sort manual instrumentado |
| `tests/test_quick_sort.py` | criado (antes vazio) | 17 testes |

Reutiliza `src/algorithms/ordering.py` (critério central). Nenhum arquivo de
colegas foi alterado.

## Assinatura das funções

```python
@dataclass
class QuickSortStats:
    comparisons: int   # comparações de chave (operação elementar)
    partitions: int    # número de particionamentos
    swaps: int         # trocas de elementos
    max_depth: int     # profundidade máxima da recursão

def quick_sort(items, key=default_key) -> tuple[list, QuickSortStats]
def quick_sort_keys(keys) -> tuple[list, QuickSortStats]   # utilitário/testes
```

Mantém o padrão `algoritmo(items) -> (resultado, Stats)`. Não modifica a lista
de entrada (opera sobre uma cópia de pares `(chave, item)`).

## Estratégia de pivô

**Mediana de três** (primeiro, meio, último elementos do intervalo).

- **Determinística:** para a mesma entrada, sempre escolhe o mesmo pivô — o que
  garante testes reprodutíveis e instrumentação estável.
- **Motivação:** evita a degeneração para O(N²) nas entradas já ordenadas e em
  ordem inversa, que são exatamente os piores casos do pivô "último elemento".
  Nessas entradas a mediana de três aproxima o pivô do valor central,
  equilibrando as partições.

## Particionamento

Esquema de **Lomuto** adaptado ao comparador canônico `precedes`:

1. escolhe o pivô por mediana de três e o move para o fim do intervalo (`high`);
2. percorre `low..high-1`, movendo para a esquerda (fronteira `i`) todo elemento
   que **precede** o pivô;
3. ao final, posiciona o pivô em `i`, deixando à esquerda os que o precedem e à
   direita os demais;
4. recursão nas duas partições, excluindo o pivô já posicionado.

## Critério de ordenação

Centralizado em `ordering.precedes` (score DESC, chunk_id ASC), o mesmo do Merge
Sort e do futuro Top-k. Não há regra duplicada.

## Instrumentação

- `comparisons`: incrementado a cada avaliação de `precedes` — tanto no
  particionamento quanto na escolha do pivô (mediana de três), pois ambas são
  comparações de chave (operação elementar).
- `partitions`: número de chamadas de particionamento.
- `swaps`: trocas efetivas de elementos (trocas triviais `i == j` não contam).
- `max_depth`: maior profundidade de recursão atingida (útil para observar
  degeneração de pior caso).

## Complexidade

| Caso | Tempo | Observação |
|---|---|---|
| Melhor | Θ(N log N) | partições balanceadas |
| Médio | Θ(N log N) | pivô razoável na maioria das entradas |
| Pior | Θ(N²) | partições sistematicamente desbalanceadas |

Espaço: O(log N) na pilha de recursão no caso equilibrado; O(N) na pilha no
pior caso (recursão degenerada). O particionamento é in-place.

### Caso que leva a O(N²)

Com pivô por mediana de três, entradas ordenadas/inversas **não** são mais o
pior caso. O pior caso quadrático ocorre para entradas adversárias construídas
de modo que a mediana de três selecione repetidamente um extremo, gerando
partições de tamanhos 0 e N-1 a cada nível. Também há degradação quando quase
todas as chaves são iguais (particionamento pouco efetivo), embora o resultado
permaneça correto. O relatório associa explicitamente o pior caso à escolha de
pivô utilizada.

## Testes executados

Ambiente: Windows, Python 3.13.1, pytest 9.1.1, `.venv` local (não versionado).

Comando:

```bash
python -m pytest tests/test_quick_sort.py -v
```

Resultado: **PASSOU — 17 passed in 0.16s.**

Execução conjunta com Merge Sort:

```bash
python -m pytest tests/test_merge_sort.py tests/test_quick_sort.py -q
# 31 passed
```

Matriz de casos:

| Caso | Teste |
|---|---|
| lista vazia | `test_empty_list` |
| um elemento | `test_single_element` |
| dois elementos (troca) | `test_two_elements_need_swap` |
| já ordenada | `test_already_sorted` |
| ordem inversa | `test_reverse_order` |
| desempate por chunk_id | `test_tie_breaks_by_chunk_id_asc` |
| muitos empates | `test_many_ties` |
| scores negativos e zero | `test_negative_and_zero_scores` |
| equivalência com oráculo (50 casos aleatórios) | `test_equivalence_with_reference_random` |
| preservação de elementos | `test_preserves_all_elements` |
| determinismo (resultado + instrumentação) | `test_determinism` |
| entrada não mutada | `test_input_not_mutated` |
| entrada ordenada tratada sem degenerar | `test_sorted_input_is_handled_efficiently` |
| entrada em ordem inversa | `test_reverse_sorted_input` |
| todos scores iguais (grande) | `test_all_equal_scores_large` |
| contagem de comparações positiva | `test_comparison_count_positive` |
| variante por chaves | `test_keys_variant` |

O oráculo `sorted()` é usado **apenas no teste** como referência.

## Exemplo de entrada/saída

Entrada:

```python
[(0.72, {"chunk_id": "chunk_10"}),
 (0.91, {"chunk_id": "chunk_02"}),
 (0.72, {"chunk_id": "chunk_04"})]
```

Saída:

```text
[(0.91, 'chunk_02'), (0.72, 'chunk_04'), (0.72, 'chunk_10')]
comparisons = 5   partitions = 1   swaps = 2   max_depth = 1
```

## Limitações

- Assume `score` finito e `chunk_id` comparável como string.
- Existe pior caso O(N²) para entradas adversárias (documentado acima); não foi
  adotada aleatorização de pivô para preservar o determinismo exigido nos testes.
- Integração com os retrievers **não foi feita** (decisão registrada: algoritmos
  puros nesta fase).

## Decisões tomadas pelo Kiro

1. Escolhido pivô por **mediana de três** (determinístico) em vez de "último
   elemento", para evitar degeneração em entradas ordenadas/inversas mantendo
   reprodutibilidade nos testes.
2. As comparações da escolha de pivô também são contabilizadas, por serem
   comparações de chave legítimas.
3. `max_depth` adicionado à instrumentação para tornar observável a qualidade do
   balanceamento das partições (evidência empírica de pior caso).
4. Mantido o padrão `(resultado, Stats)` e o critério central de `ordering.py`.

## Itens que Mateus deve revisar manualmente

- Confirmar a estratégia de pivô (mediana de três) para a apresentação; se o
  grupo preferir "último elemento" para simplificar a análise de pior caso em
  entradas ordenadas, é uma troca consciente a discutir.
- Validar se as métricas `partitions/swaps/max_depth` são suficientes para os
  experimentos da Gabriela ou se algo mais é necessário.

## Estado de ambiente

- Docker Linux: **NÃO EXECUTADO** — Docker indisponível nesta máquina.
- Testes locais: **PASSOU** (17/17; 31/31 com Merge Sort).
