# Corretude dos Algoritmos Clássicos — FastContext

Este documento reúne as justificativas de corretude dos algoritmos clássicos
implementados manualmente pela equipe. As provas são ligadas ao código real em
`src/algorithms/`.

Critério de ordenação canônico (ver `src/algorithms/ordering.py`):

> Um item `a` precede `b` se `a.score > b.score`, ou, em empate
> (`a.score == b.score`), se `a.chunk_id < b.chunk_id`.

---

## 1. Merge Sort

Arquivo: `src/algorithms/merge_sort.py`.

### Objetivo da prova

Mostrar que `merge_sort(items)` produz uma permutação de `items` totalmente
ordenada segundo o critério canônico (score DESC, chunk_id ASC).

### Hipóteses e pré-condições

- `items` é uma lista finita (possivelmente vazia).
- Cada item admite extração de uma chave `(score, chunk_id)` via `key`.
- `precedes` define uma ordem total estrita sobre as chaves (`score` real e
  finito; `chunk_id` string). Como `chunk_id` é único no projeto, não há duas
  chaves iguais entre itens distintos.

### Pós-condições

- O resultado contém exatamente os mesmos elementos de `items` (permutação).
- Para índices consecutivos `r[i]` e `r[i+1]` do resultado, não vale
  `precedes(key(r[i+1]), key(r[i]))` — isto é, nenhum par está fora de ordem.

### Invariante do merge (`_merge`)

Sejam `left` e `right` já ordenadas. O laço mantém a invariante:

> `merged` está ordenada e contém todos os elementos já consumidos de `left[:i]`
> e `right[:j]`; além disso, todo elemento em `merged` precede (ou empata,
> mantendo estabilidade) qualquer elemento ainda não consumido em `left[i:]` e
> `right[j:]`.

- **Inicialização:** antes do laço, `i = j = 0` e `merged = []`. A invariante
  vale trivialmente (nada consumido, nada em `merged`).
- **Manutenção:** a cada iteração compara-se `key(left[i])` e `key(right[j])`.
  Escolhe-se `right[j]` somente quando ele precede estritamente `left[i]`; caso
  contrário escolhe-se `left[i]`. O elemento escolhido é o menor (segundo o
  critério) entre os dois candidatos mínimos das duas metades ordenadas, logo é
  o menor global dentre os não consumidos. Anexá-lo a `merged` preserva a ordem.
  Em empate, escolhe-se `left[i]` primeiro, garantindo estabilidade.
- **Término:** o laço termina quando uma das metades se esgota; os laços de
  drenagem anexam o restante da outra metade, que já está ordenado e é maior ou
  igual a tudo que já está em `merged`. Assim `merged` fica totalmente ordenada
  e contém todos os elementos de `left` e `right` exatamente uma vez.

### Indução da recursão (`_merge_sort_recursive`)

Indução forte sobre `n = len(pairs)`.

- **Caso base (`n <= 1`):** uma lista de 0 ou 1 elemento já está ordenada; o
  algoritmo a devolve inalterada. ✔
- **Passo (`n >= 2`):** as duas metades têm tamanho `< n`. Por hipótese de
  indução, `left` e `right` retornam ordenadas e são permutações das metades
  originais. Pela corretude de `_merge`, o resultado é a intercalação ordenada
  contendo todos os elementos das duas metades exatamente uma vez — ou seja, uma
  permutação ordenada de `pairs`. ✔

Como a divisão `pairs[:mid]` / `pairs[mid:]` cobre todos os elementos sem
sobreposição, nenhum elemento é perdido ou duplicado.

### Casos de borda

- Lista vazia e unitária: cobertos pelo caso base.
- Empates de score: resolvidos por `chunk_id` ASC dentro de `precedes`; a
  estabilidade do merge garante determinismo.
- Scores negativos e zero: `precedes` usa comparação numérica direta, válida
  para qualquer real finito.

### Limites da prova

- A prova assume `score` finito (garantido pelo contrato `RetrievedChunk` do
  projeto) e `chunk_id` comparável como string.
- A prova cobre a ordenação; a instrumentação (contadores) é auxiliar e não
  afeta a corretude do resultado ordenado.

---

## 2. Quick Sort

Arquivo: `src/algorithms/quick_sort.py`.

### Objetivo da prova

Mostrar que `quick_sort(items)` produz uma permutação de `items` totalmente
ordenada segundo o critério canônico (score DESC, chunk_id ASC).

### Corretude do particionamento (`_partition`, esquema de Lomuto)

Após escolher o pivô por mediana de três e movê-lo para `high`, o laço mantém a
invariante sobre o índice `i` (fronteira dos elementos que precedem o pivô):

> Ao início de cada iteração `j`, todo elemento em `pairs[low..i-1]` precede o
> pivô, e todo elemento em `pairs[i..j-1]` não precede o pivô.

- **Inicialização:** `i = low`, `j = low`; ambos os intervalos são vazios. ✔
- **Manutenção:** compara-se `pairs[j]` com o pivô. Se `precedes(pairs[j], pivô)`,
  troca-se `pairs[i]` com `pairs[j]` e incrementa-se `i`, estendendo o prefixo
  dos que precedem; caso contrário `pairs[j]` fica na região dos que não
  precedem. A invariante se mantém. ✔
- **Término:** após o laço, `pairs[low..i-1]` precedem o pivô e `pairs[i..high-1]`
  não. A troca final `pairs[i] <-> pairs[high]` coloca o pivô em `i`, com todos
  os que o precedem à esquerda e os demais à direita. O pivô fica na sua posição
  final de ordenação.

### Corretude do pivô (`_median_of_three`)

Retorna sempre um índice válido dentre `{low, mid, high}` correspondente ao
elemento mediano segundo `precedes`. É determinístico e não altera o conteúdo do
vetor (apenas seleciona um índice), logo não afeta a permutação — apenas a
qualidade do balanceamento.

### Indução da recursão (`_quick_sort_recursive`)

Indução forte sobre o tamanho `m = high - low + 1` do subintervalo.

- **Caso base (`low >= high`):** 0 ou 1 elemento já está ordenado. ✔
- **Passo:** o particionamento posiciona o pivô corretamente e separa o
  intervalo em `[low, p-1]` e `[p+1, high]`, ambos de tamanho `< m`. Por hipótese
  de indução, as duas recursões os ordenam. Como todos à esquerda precedem o
  pivô e todos à direita não o precedem, a concatenação `esquerda + pivô +
  direita` está totalmente ordenada. ✔

Não há perda nem duplicação: o particionamento é uma permutação in-place do
subintervalo, e o pivô é contado exatamente uma vez.

### Casos de borda

- Vazio/unitário: caso base.
- Empates de score: `precedes` desempata por chunk_id ASC; o resultado é
  determinístico. Muitas chaves iguais degradam o balanceamento (ver
  complexidade) mas não a corretude.
- Scores negativos/zero: comparação numérica direta em `precedes`.

---

## 3. Top-k com min-heap

Arquivo: `src/algorithms/topk_heap.py`.

### Objetivo da prova

Mostrar que `top_k(items, k)` retorna exatamente os `k` melhores itens (ou
todos, se `k >= N`) na ordem de ranking canônica.

### Propriedade de min-heap "pior no topo"

A relação da heap é `_worse(a, b) = precedes(b, a)`: o "mínimo" é o item de
menor preferência. `_sift_up` e `_sift_down` restauram a propriedade de que todo
pai é pior (ou igual) que seus filhos. Portanto `heap[0]` é sempre o **pior**
elemento presente na heap.

### Invariante do laço principal

> Após processar os primeiros `t` itens, a heap contém exatamente os
> `min(t, k)` melhores dentre eles, e sua raiz é o pior desse conjunto.

- **Inicialização:** heap vazia; vale trivialmente para `t = 0`.
- **Manutenção:** ao processar o item `t+1`:
  - se `|heap| < k`, ele entra (os melhores ainda cabem todos);
  - se `|heap| = k`, compara-se com a raiz (pior dos k atuais). Se o novo item o
    precede (é melhor), substitui a raiz; caso contrário é descartado, pois é
    pior que os `k` já mantidos e não pode figurar entre os `k` melhores.
  Em ambos os casos a heap passa a conter os `min(t+1, k)` melhores. ✔
- **Término:** após `N` itens, a heap contém os `min(N, k)` melhores.

### Ordenação final (`_heap_to_sorted`)

Extrai repetidamente a raiz (pior restante) para o fim de uma lista e inverte,
produzindo a saída do melhor para o pior (score DESC / chunk_id ASC). É um
heapsort correto sobre os elementos da heap.

### Casos de borda

- `k <= 0` ou `items` vazio: retorna `[]` sem construir heap.
- `k >= N`: todos os itens entram; a ordenação final devolve todos ordenados.
- Empates: `precedes` desempata por chunk_id ASC, inclusive na fronteira do
  corte (o item com chunk_id menor é preferido).

### Limites da prova

- Assume `score` finito e `chunk_id` comparável. A prova cobre a seleção e a
  ordenação; a instrumentação é auxiliar.

---

## 4. Busca binária (implementação do Wilson)

Arquivo: `src/algorithms/binary_search.py` (autoria do Wilson,
`feature/retrieval`). A análise abaixo é ligada ao **código real** dessa
implementação.

### Código analisado

```python
def binary_search(elements, target):
    comparisons = 0
    left = 0
    right = len(elements) - 1
    while left <= right:
        mid = (left + right) // 2
        comparisons += 1
        if elements[mid] == target:
            return mid, comparisons
        elif elements[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return None, comparisons
```

### Pré-condições

- `elements` está ordenado em ordem **não decrescente** (no projeto, o
  vocabulário ordenado do índice invertido).
- Os elementos são comparáveis por `==` e `<`.

### Invariante do laço

> Se `target` está em `elements`, então seu índice está no intervalo fechado
> `[left, right]`.

- **Inicialização:** `left = 0`, `right = n-1` cobre todo o vetor. ✔
- **Manutenção:** calcula-se `mid`. Se `elements[mid] == target`, retorna `mid`
  (corretude direta). Se `elements[mid] < target`, como o vetor é ordenado,
  `target` só pode estar à direita, logo `left = mid + 1` preserva a invariante.
  Caso contrário `target` só pode estar à esquerda, logo `right = mid - 1`. ✔
- **Término:** o intervalo encolhe a cada iteração (pois `mid` está sempre em
  `[left, right]` e é removido). Quando `left > right`, o intervalo é vazio e o
  elemento não existe: retorna `None`. Como o intervalo diminui estritamente, o
  laço termina.

### Corretude

Pela invariante, se `target` existe ele nunca é excluído do intervalo, e como o
laço testa `mid` a cada passo, ele será encontrado antes de o intervalo esvaziar.
Se não existe, o intervalo esvazia e retorna `None`. A contagem `comparisons`
soma uma comparação de chave por iteração (a igualdade em `elements[mid]`),
métrica coerente para a análise de PAA.

### Observação

Esta implementação conta **uma** comparação por iteração (a de igualdade); a
comparação `<`/`>` subsequente não é contabilizada separadamente. A análise de
complexidade abaixo respeita essa contagem do código real.
