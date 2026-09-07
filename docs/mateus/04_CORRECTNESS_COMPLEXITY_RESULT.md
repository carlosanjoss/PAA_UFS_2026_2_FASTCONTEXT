# Relatório 4 — Corretude e Complexidade

Fase 4 do escopo de Mateus Almeida. Síntese teórica ligada ao código real em
`src/algorithms/`. Provas detalhadas em `docs/CORRECTNESS.md`.

## Critério de ordenação (comum aos três algoritmos)

Definido em `src/algorithms/ordering.py` (`precedes`): score decrescente; em
empate, chunk_id crescente. Operação elementar analisada: **comparação de
chave** via `precedes`.

---

## 1. Merge Sort

Arquivo: `src/algorithms/merge_sort.py`.

- **Modelo de entrada:** lista de candidatos convertidos em chaves
  `(score, chunk_id)`.
- **Operação elementar:** comparação de chave no merge (`stats.comparisons`).
- **Recorrência:**

  ```text
  T(N) = 2 T(N/2) + Θ(N)
  ```

  O termo linear é o custo do merge. Pelo Teorema Mestre (caso 2, a=2, b=2,
  f(N)=Θ(N), N^{log_b a}=N): **T(N) = Θ(N log N)**.

- **Casos:**

  | Caso | Tempo |
  |---|---|
  | Melhor | Θ(N log N) |
  | Médio | Θ(N log N) |
  | Pior | Θ(N log N) |

  A divisão é sempre ao meio, independente da entrada, logo os três casos
  coincidem.

- **Espaço:** Θ(N) auxiliar (listas intermediárias do merge) + Θ(log N) de pilha
  recursiva.
- **Relação teoria ↔ código:** `_merge_sort_recursive` faz a divisão `pairs[:mid]`
  / `pairs[mid:]` (as duas subchamadas de `T(N/2)`) e `_merge` realiza o trabalho
  Θ(N). No exemplo com 3 itens: comparisons=3, merges=2, moves=5.

---

## 2. Quick Sort

Arquivo: `src/algorithms/quick_sort.py`.

- **Estratégia de pivô:** mediana de três (primeiro, meio, último),
  determinística. **Reduz os casos triviais de degeneração** (entradas já
  ordenadas e em ordem inversa deixam de ser o pior caso), **mas o pior caso
  continua sendo O(N²)** para entradas adversárias.
- **Operação elementar:** comparação de chave no particionamento e na escolha do
  pivô (`stats.comparisons`).
- **Recorrências:**
  - Balanceado (melhor/médio): `T(N) = 2 T(N/2) + Θ(N)` → **Θ(N log N)**.
  - Desbalanceado (pior): `T(N) = T(N-1) + Θ(N)` → **Θ(N²)**.

- **Casos:**

  | Caso | Tempo |
  |---|---|
  | Melhor | Θ(N log N) |
  | Médio | Θ(N log N) |
  | Pior | Θ(N²) |

- **Motivo do pior caso quadrático:** quando o pivô selecionado é sistematicamente
  um extremo, cada particionamento separa 1 elemento de N-1, gerando N níveis de
  recursão com custo linear cada. Com mediana de três isso não acontece em
  entradas ordenadas/inversas, mas pode ser induzido por entradas adversárias
  específicas ou quando quase todas as chaves são iguais.
- **Espaço:** O(log N) de pilha no caso balanceado; O(N) no pior caso. O
  particionamento é in-place.
- **Relação teoria ↔ código:** `_partition` (Lomuto) executa o trabalho Θ(N) por
  nível; `_median_of_three` escolhe o pivô; `max_depth` na instrumentação torna
  observável o balanceamento (proxy empírico do caso em que se está).

---

## 3. Top-k com min-heap

Arquivo: `src/algorithms/topk_heap.py`.

- **Objetivo:** obter os `k` melhores sem ordenar todos os `N`.
- **Operação elementar:** comparação de chave na heap (`stats.comparisons`).
- **Complexidade:**

  | Métrica | Custo |
  |---|---|
  | Tempo | O(N log k) |
  | Espaço | O(k) |

- **Por que não é preciso ordenar todos os N:** mantém-se uma heap de tamanho no
  máximo `k`. Cada item é processado em O(log k): ou entra (enquanto há espaço)
  ou é comparado com o pior atual (raiz) e descartado se não for melhor. Ao todo
  são O(N) operações de O(log k), portanto **O(N log k)**. Quando `k << N`, isso
  é assintoticamente melhor que ordenar tudo (O(N log N)).
- **Justificativa da heap de tamanho k:** guardar apenas os `k` melhores limita a
  memória a **O(k)** e o custo por operação a O(log k). A raiz "pior no topo"
  permite decidir em uma comparação se um novo item pode melhorar o conjunto.
- **Relação teoria ↔ código:** `_sift_up`/`_sift_down` custam O(log k);
  `max_heap_size` confirma empiricamente o limite `k`; `replacements`/`insertions`
  evidenciam quantas vezes o conjunto dos melhores foi atualizado.

---

## 4. Busca binária (implementação do Wilson)

Arquivo: `src/algorithms/binary_search.py`. Análise ligada ao código real.

- **Pré-condição:** vetor ordenado não decrescente (vocabulário do índice
  invertido).
- **Operação elementar:** uma comparação de igualdade por iteração
  (`comparisons += 1`), conforme o código.
- **Recorrência:** `T(N) = T(N/2) + Θ(1)` → **Θ(log N)**.

  | Caso | Tempo |
  |---|---|
  | Melhor | Θ(1) (acerta no meio na 1ª comparação) |
  | Médio | Θ(log N) |
  | Pior | Θ(log N) (ausente ou nos extremos) |

- **Espaço:** O(1) (implementação iterativa, sem recursão).
- **Corretude:** ver `docs/CORRECTNESS.md` §4 — invariante "se existe, está em
  `[left, right]`"; intervalo encolhe a cada passo; termina encontrando o alvo ou
  esvaziando o intervalo.

---

## 5. Comparação entre os algoritmos

| Algoritmo | Melhor | Médio | Pior | Espaço | Uso no projeto |
|---|---:|---:|---:|---:|---|
| Merge Sort | Θ(N log N) | Θ(N log N) | Θ(N log N) | Θ(N) + Θ(log N) pilha | ordenação estável e previsível de candidatos |
| Quick Sort | Θ(N log N) | Θ(N log N) | Θ(N²) | O(log N)–O(N) pilha | comparação clássica; in-place |
| Top-k Heap | O(N log k) | O(N log k) | O(N log k) | O(k) | seleção dos k melhores (config. otimizada) |
| Busca binária | Θ(1) | Θ(log N) | Θ(log N) | O(1) | localizar termo no vocabulário ordenado |

### Diferenças teóricas relevantes

- **Merge vs Quick:** o Merge Sort garante Θ(N log N) em todos os casos e é
  estável, ao custo de Θ(N) de memória extra. O Quick Sort é in-place e
  costuma ter constantes menores na prática, mas tem pior caso O(N²) — mitigado,
  não eliminado, pela mediana de três.
- **Ordenar tudo vs Top-k:** ordenar (Merge/Quick) custa Θ(N log N) e entrega a
  lista completa; o Top-k custa O(N log k) e entrega só os `k` melhores. Para
  `k << N`, o Top-k é a escolha assintoticamente superior — base da configuração
  "Otimizada" (C) do projeto.
- **Busca vs ordenação:** a busca binária resolve localização em Θ(log N) sobre
  dados **já ordenados**; ordenar é o pré-requisito que habilita busca binária e
  seleção eficiente.

---

## 6. Estado de verificação

- Provas de corretude: `docs/CORRECTNESS.md` (Merge Sort, Quick Sort, Top-k,
  busca binária do Wilson).
- Testes: 49 passaram (Merge 14, Quick 17, Top-k 18) — ver relatórios 01–03.
- Docker Linux: **NÃO EXECUTADO** (indisponível nesta máquina).
- Nenhum arquivo de colega (retrievers/contratos) foi alterado.
