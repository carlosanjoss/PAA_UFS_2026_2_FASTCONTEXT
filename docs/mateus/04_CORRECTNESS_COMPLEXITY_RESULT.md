# Relatório 4 — Corretude e Complexidade

Síntese teórica dos algoritmos clássicos do FastContext, vinculada ao código real em `src/algorithms/`.

Provas detalhadas:

```text
docs/CORRECTNESS.md
```

---

## 1. Critério de ordenação

Ordem canônica:

1. score decrescente;
2. em empate, `chunk_id` crescente.

---

## 2. Merge Sort

Arquivo: `src/algorithms/merge_sort.py`.

Recorrência:

```text
T(N) = 2T(N/2) + Θ(N)
```

Pelo Teorema Mestre:

```text
T(N) = Θ(N log N)
```

| Caso | Tempo |
|---|---:|
| Melhor | `Θ(N log N)` |
| Médio | `Θ(N log N)` |
| Pior | `Θ(N log N)` |

Espaço auxiliar: `Θ(N)` mais pilha recursiva `Θ(log N)`.

---

## 3. Quick Sort

Arquivo: `src/algorithms/quick_sort.py`.

Estratégia:

- particionamento de Lomuto;
- pivô por mediana de três.

Caso balanceado:

```text
T(N) = 2T(N/2) + Θ(N) = Θ(N log N)
```

Pior caso:

```text
T(N) = T(N-1) + Θ(N) = Θ(N²)
```

| Caso | Tempo |
|---|---:|
| Melhor | `Θ(N log N)` |
| Médio | `Θ(N log N)` |
| Pior | `Θ(N²)` |

Espaço: `O(log N)` balanceado e `O(N)` no pior caso.

---

## 4. Top-k com min-heap

Arquivo: `src/algorithms/topk_heap.py`.

Tempo:

```text
O(N log k)
```

Espaço:

```text
O(min(N, k)) <= O(k)
```

É vantajoso quando `k << N` porque evita ordenar todos os candidatos.

---

## 5. Busca Binária

Arquivo: `src/algorithms/binary_search.py`.

Recorrência:

```text
T(N) = T(N/2) + Θ(1)
```

Logo:

```text
T(N) = Θ(log N)
```

| Caso | Tempo |
|---|---:|
| Melhor | `Θ(1)` |
| Médio | `Θ(log N)` |
| Pior | `Θ(log N)` |

Espaço: `O(1)`.

### Instrumentação atual

A implementação contabiliza as comparações de chave realmente executadas:

```text
elements[mid] == target -> +1
```

Se necessário:

```text
elements[mid] < target -> +1
```

Portanto, uma iteração pode registrar 1 ou 2 comparações. Isso melhora a fidelidade experimental do contador sem alterar a complexidade assintótica.

---

## 6. Busca Linear

Na busca linear clássica:

| Caso | Tempo |
|---|---:|
| Melhor | `Θ(1)` |
| Médio | `Θ(N)` |
| Pior | `Θ(N)` |

No `LinearRetriever` do FastContext, todos os chunks precisam ser pontuados para produzir o ranking. Portanto, o scoring é:

```text
Θ(N)
```

Depois ocorre Merge Sort:

```text
Θ(N log N)
```

O ranking completo domina assintoticamente o pipeline Linear.

---

## 7. Índice Invertido

O índice invertido mapeia:

```text
term -> posting list of chunks
```

Se `T` representa o total de termos processados, a construção pode ser descrita, em termos gerais, como:

```text
O(T)
```

Na consulta, cada termo é localizado no vocabulário ordenado por busca binária. Para vocabulário de tamanho `V`:

```text
O(log V)
```

por termo, além do custo de percorrer/unir as posting lists acessadas.

---

## 8. Comparação resumida

| Componente | Melhor | Médio | Pior | Espaço auxiliar principal |
|---|---:|---:|---:|---:|
| Merge Sort | `Θ(N log N)` | `Θ(N log N)` | `Θ(N log N)` | `Θ(N)` + pilha `Θ(log N)` |
| Quick Sort | `Θ(N log N)` | `Θ(N log N)` | `Θ(N²)` | `O(log N)` a `O(N)` |
| Top-k Heap | `O(N log k)` | `O(N log k)` | `O(N log k)` | `O(k)` |
| Binary Search | `Θ(1)` | `Θ(log N)` | `Θ(log N)` | `O(1)` |
| Linear scoring | `Θ(N)` | `Θ(N)` | `Θ(N)` | depende da representação |
| Inverted Index lookup | depende dos termos/postings | depende dos termos/postings | depende dos termos/postings | proporcional ao índice |

---

## 9. Relação com as estratégias de retrieval

### Linear

```text
TF-IDF scoring over N chunks + Merge Sort
```

Custo dominante:

```text
Θ(N log N)
```

### Indexed

Com `C` candidatos, `C <= N`:

```text
Binary Search + Inverted Index + scoring + Θ(C log C)
```

### Optimized

```text
Binary Search + Inverted Index + scoring + O(C log k)
```

### Semantic

BGE + FAISS funciona como baseline complementar e não substitui a análise dos algoritmos clássicos.

---

## 10. Benchmark Merge Sort × Quick Sort

Benchmark concluído:

```text
Raw rows: 200
Summary rows: 40
Comparable cases: 100
Fingerprint mismatches: 0
```

Matriz:

```text
2 algorithms
× 5 sizes
× 4 scenarios
× 5 repetitions
= 200 measured runs
```

Tamanhos:

```text
100, 250, 500, 1000, 1305
```

Cenários:

```text
random
already_sorted
reverse_sorted
many_ties
```

---

## 11. Estado de verificação

- Corretude detalhada: `docs/CORRECTNESS.md`.
- Benchmark Merge × Quick: concluído.
- Binary Search: contador de comparações atualizado.
- Testes gerais devem ser executados novamente após mudanças de algoritmo.
- Docker ainda precisa de validação final antes da entrega.
