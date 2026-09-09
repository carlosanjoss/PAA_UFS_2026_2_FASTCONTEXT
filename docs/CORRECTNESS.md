# Corretude dos Algoritmos Clássicos — FastContext

Este documento reúne as justificativas de corretude dos algoritmos clássicos implementados manualmente pela equipe. As provas são vinculadas ao código real em `src/algorithms/`.

## Critério de ordenação canônico

Definido em `src/algorithms/ordering.py`.

Um item `a` precede `b` quando:

```text
a.score > b.score
```

ou, em empate:

```text
a.score == b.score and a.chunk_id < b.chunk_id
```

---

## 1. Merge Sort

Arquivo: `src/algorithms/merge_sort.py`.

### 1.1 Objetivo

Mostrar que `merge_sort(items)` produz uma permutação dos itens de entrada totalmente ordenada segundo a ordem canônica.

### 1.2 Pré-condições

- `items` é uma lista finita;
- cada item possui uma chave `(score, chunk_id)`;
- `score` e `chunk_id` são comparáveis.

### 1.3 Pós-condições

O resultado contém exatamente os mesmos itens da entrada e está ordenado segundo `score DESC` e `chunk_id ASC`.

### 1.4 Invariante de `_merge`

Considere `left` e `right` já ordenadas.

> `merged` contém exatamente os elementos já consumidos de `left` e `right` e permanece ordenada segundo a ordem canônica.

**Inicialização:** antes da primeira iteração, `i = 0`, `j = 0` e `merged = []`; a invariante vale trivialmente.

**Manutenção:** a cada passo, o algoritmo compara o próximo elemento de `left` com o próximo de `right` e escolhe aquele que deve aparecer primeiro segundo a ordem canônica. Anexá-lo a `merged` preserva a ordenação.

**Término:** quando uma metade se esgota, o restante da outra metade já está ordenado e pode ser anexado sem violar a ordem.

### 1.5 Indução

Para `n <= 1`, a lista já está ordenada.

Para `n >= 2`, a entrada é dividida em duas partes menores. Pela hipótese de indução, ambas são corretamente ordenadas pelas chamadas recursivas. Pela corretude de `_merge`, a combinação final também é ordenada e contém todos os elementos exatamente uma vez.

Logo, Merge Sort é correto.

---

## 2. Quick Sort

Arquivo: `src/algorithms/quick_sort.py`.

### 2.1 Objetivo

Mostrar que `quick_sort(items)` retorna uma permutação totalmente ordenada segundo a ordem canônica.

### 2.2 Particionamento

A implementação utiliza particionamento de Lomuto após escolher o pivô por mediana de três.

Invariante:

> Os itens à esquerda da fronteira `i` precedem o pivô; os itens entre `i` e a posição atualmente examinada não precedem o pivô.

**Inicialização:** as duas regiões começam vazias.

**Manutenção:** se o elemento atual precede o pivô, ele é movido para a região esquerda; caso contrário, permanece na região direita.

**Término:** o pivô é colocado entre as duas regiões. Todos os itens à esquerda o precedem e todos os itens à direita não o precedem.

### 2.3 Mediana de três

`_median_of_three` escolhe deterministicamente um pivô entre primeiro, meio e último. A estratégia altera o balanceamento, mas não a corretude do particionamento.

### 2.4 Indução

Intervalos com 0 ou 1 elemento já estão ordenados. Para intervalos maiores, o particionamento posiciona corretamente o pivô e gera duas regiões menores. Pela hipótese de indução, ambas são ordenadas; logo, esquerda + pivô + direita está ordenada.

Portanto, Quick Sort é correto.

---

## 3. Top-k com min-heap

Arquivo: `src/algorithms/topk_heap.py`.

### 3.1 Objetivo

Mostrar que `top_k(items, k)` retorna exatamente os `k` melhores elementos, ou todos quando `k >= N`.

### 3.2 Propriedade da heap

A heap mantém o pior elemento atual na raiz.

### 3.3 Invariante principal

> Após processar os primeiros `t` itens, a heap contém exatamente os `min(t, k)` melhores itens observados até aquele momento.

**Inicialização:** com zero itens, a heap vazia satisfaz a invariante.

**Manutenção:** se há espaço, o item entra. Se a heap está cheia e o novo item é melhor que a raiz, a raiz é substituída. Caso contrário, o item é descartado.

**Término:** após os `N` itens, a heap contém os `min(N, k)` melhores. A etapa final apenas os ordena segundo a ordem canônica.

Logo, Top-k é correto.

---

## 4. Busca Binária

Arquivo: `src/algorithms/binary_search.py`.

### 4.1 Pré-condição

A lista está ordenada em ordem não decrescente.

### 4.2 Invariante

> Se `target` estiver presente, seu índice está no intervalo fechado `[left, right]`.

**Inicialização:** `left = 0` e `right = n - 1` cobrem toda a lista.

**Manutenção:** se `elements[mid] == target`, o alvo é encontrado. Se `elements[mid] < target`, pela ordenação ele só pode estar à direita e `left = mid + 1`. Caso contrário, só pode estar à esquerda e `right = mid - 1`.

**Término:** o intervalo encolhe estritamente. Quando `left > right`, ele está vazio e o alvo não existe.

Logo, a busca binária é correta.

### 4.3 Instrumentação de comparações

A implementação atual conta as comparações de chave realmente executadas:

```text
elements[mid] == target -> +1
```

Se a igualdade falhar:

```text
elements[mid] < target -> +1
```

Assim:

- alvo encontrado no `mid`: 1 comparação na iteração;
- decisão esquerda/direita: 2 comparações na iteração.

A complexidade permanece:

| Caso | Complexidade |
|---|---:|
| Melhor | `Θ(1)` |
| Médio | `Θ(log N)` |
| Pior | `Θ(log N)` |

---

## 5. Limites das provas

As provas assumem entradas válidas segundo os contratos do projeto. A instrumentação é auxiliar e não altera o resultado dos algoritmos.
