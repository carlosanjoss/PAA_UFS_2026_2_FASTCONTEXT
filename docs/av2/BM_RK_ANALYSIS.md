# AV2 — Análise de Corretude e Complexidade: Boyer–Moore e Rabin–Karp

**Autor:** Mateus Almeida
**Trilha D — Processamento de Cadeias (string matching exato)**
**Algoritmos:** Boyer–Moore (bad-character) e Rabin–Karp (rolling hash)

Notação: `n = len(text)`, `m = len(pattern)`.

---

## 0. Contrato comum

Ambos os algoritmos implementam a interface `StringMatcher`:

- `search(text, pattern) -> StringMatchResult`;
- `validate_input`: `pattern` vazio → `ValueError`; tipos não-`str` → `TypeError`;
- resultado com `positions` (base-zero, crescente, **com sobreposições**),
  `comparisons` (apenas comparações reais de caractere),
  `preprocessing_time_ns` e `search_time_ns` medidos separadamente;
- texto vazio → `()`; padrão maior que o texto → `()`;
- busca **case-sensitive**; índices sobre caracteres Unicode (não bytes).

---

## 1. Boyer–Moore (bad-character)

### 1.1 Variante
Apenas a heurística **bad-character**. **Não** há *good-suffix*. Portanto
não é o Boyer–Moore completo, e o pior caso de busca é `O(n·m)`.

### 1.2 Pré-processamento — tabela de última ocorrência
Para cada caractere do padrão, guarda o índice da sua ocorrência mais à
direita. Caracteres fora do padrão são tratados como `-1`.

```
build_last_occurrence(pattern):
    last = {}
    for i in 0..m-1:
        last[pattern[i]] = i
    return last
```

Exemplo `pattern = "ABCA"` → `{A:3, B:1, C:2}`.

### 1.3 Pseudocódigo da busca

```
search(text, pattern):
    last = build_last_occurrence(pattern)
    alignment = 0
    while alignment <= n - m:
        j = m - 1
        while j >= 0 and text[alignment + j] == pattern[j]:
            comparisons += 1
            j -= 1
        if j < 0:
            record(alignment)
            alignment += 1            # preserva sobreposição
        else:
            comparisons += 1          # a comparação que falhou
            c = text[alignment + j]
            shift = j - last.get(c, -1)
            alignment += max(shift, 1)
```

> Observação de implementação: no código, `comparisons` é incrementado
> **antes** de cada comparação (inclusive a que falha), então toda
> comparação real de caractere é contabilizada exatamente uma vez.

### 1.4 Corretude
- **Nenhuma ocorrência perdida.** O deslocamento bad-character alinha a
  última ocorrência (no padrão) do caractere `c` que causou o *mismatch*
  com a posição desse caractere no texto. Nenhum alinhamento descartado
  poderia conter um casamento, pois naquele ponto o caractere `c` do texto
  não se alinharia a nenhuma posição compatível do padrão.
- **Progresso.** Quando `shift <= 0` (a última ocorrência está à direita
  de `j`), o avanço é forçado para 1, garantindo que `alignment` sempre
  cresce.
- **Terminação.** `alignment` cresce em ≥ 1 por iteração e o laço termina
  em `alignment > n - m`.
- **Todas as ocorrências / sobreposição.** Após um casamento completo, o
  avanço é de exatamente 1, de modo que ocorrências sobrepostas (ex.:
  `"aaaaa"`/`"aaa"` → `(0,1,2)`) são todas registradas.

### 1.5 Complexidade

| Aspecto | Custo |
|---|---|
| Pré-processamento | `O(m)` tempo; `O(k)` espaço, `k` = nº de caracteres distintos no padrão |
| Busca — caso favorável | sublinear (saltos grandes em textos com alfabeto rico) |
| Busca — pior caso | `O(n·m)` (sem good-suffix; ex.: texto/padrão muito repetitivos) |
| Espaço auxiliar | `O(k)` |

---

## 2. Rabin–Karp (rolling hash)

### 2.1 Função de hash
Hash polinomial em base `b` módulo primo `q`:

```
hash(s) = ( s[0]*b^(m-1) + s[1]*b^(m-2) + ... + s[m-1]*b^0 ) mod q
```

Parâmetros padrão: `b = 256`, `q = 1_000_000_007`. Configuráveis apenas
para testes de colisão.

### 2.2 Pseudocódigo

```
search(text, pattern):
    if m > n: return []
    high = b^(m-1) mod q
    hp = hash(pattern)                 # pré-processamento
    hw = hash(text[0..m-1])            # janela inicial
    for start in 0..n-m:
        if hw == hp:                   # NÃO é prova de match
            ok = True
            for o in 0..m-1:
                comparisons += 1       # verificação real
                if text[start+o] != pattern[o]:
                    ok = False; break
            if ok: record(start)
        if start < n-m:                # roll
            hw = (hw - code(text[start])*high) mod q
            hw = (hw*b + code(text[start+m])) mod q
```

### 2.3 Corretude
- **Sem falso positivo.** Igualdade de hash **nunca** é tratada como
  casamento. Toda coincidência de hash dispara verificação caractere a
  caractere; uma colisão (hashes iguais, conteúdo diferente) é rejeitada.
  Comprovado no teste `test_collision_does_not_produce_false_positive`
  (`base=10, modulus=11`, `hash("19")==hash("08")`, resultado `()`).
- **Sem falso negativo.** Se `text[start..start+m-1] == pattern`, então os
  hashes são iguais; como a verificação confirma caractere a caractere, a
  ocorrência é registrada.
- **Rolling hash correto.** `high = b^(m-1) mod q` remove a contribuição do
  caractere que sai; multiplicar por `b` e somar o novo caractere insere o
  que entra. Em Python, `%` com módulo positivo nunca retorna negativo.
- **Terminação / sobreposição.** O laço percorre todos os `start` de `0` a
  `n-m` e registra qualquer casamento, inclusive sobrepostos.

### 2.4 Complexidade

| Aspecto | Custo |
|---|---|
| Pré-processamento (hash do padrão + janela + `high`) | `O(m)` |
| Roll por janela | `O(1)` |
| Caso esperado/médio | `O(n + m)` (verificações completas raras) |
| Pior caso | `O(n·m)` (muitas colisões ou texto altamente repetitivo) |
| Espaço auxiliar | `O(1)` além da lista de posições |

> O pior caso `O(n·m)` não é escondido: ocorre quando a verificação é
> acionada com frequência.

---

## 3. Definição das métricas (ambos)

- **`comparisons`**: número de comparações **reais** entre caracteres do
  texto e do padrão. **Não** conta igualdade de hashes, acesso a
  dicionário, cálculo de deslocamento nem operações do rolling hash.
- **`preprocessing_time_ns`**: tempo de construção das estruturas
  dependentes do padrão (tabela de última ocorrência no BM; hashes e
  `high` no RK). Cronômetro aberto imediatamente antes e fechado logo após
  essa construção.
- **`search_time_ns`**: tempo do núcleo de busca apenas. Não inclui I/O,
  JSON, HTTP, corpus nem escrita de resultados.

---

## 4. Equivalência com a baseline (Naive, de Wilson)

O `NaiveMatcher` é o oráculo de posições. Testes em
`tests/test_av2_matching_equivalence.py` exigem:

```
boyer_moore.positions == naive.positions
rabin_karp.positions  == naive.positions
```

em casos canônicos e em 800 casos aleatórios com seed fixa (fuzzing
diferencial). Apenas `positions` é comparado; `comparisons`, tempos e
metadados podem divergir legitimamente entre algoritmos.

**Status:** equivalência **EXECUTADA e PASSOU** no ambiente local
(Python 3.13.1, pytest 9.1.1).

---

## 5. Relação com o FastContext

- Boyer–Moore e Rabin–Karp fazem **busca literal exata**: encontram as
  posições em que o padrão aparece textualmente no texto.
- Uma ocorrência literal **não implica relevância temática/semântica**.
- Portanto, BM e RK **não substituem** a recuperação semântica (FAISS/RAG)
  da AV1. São ferramentas complementares de busca exata sobre os chunks.

---

## 6. Limitações

- **Boyer–Moore:** apenas bad-character; pior caso `O(n·m)`.
- **Rabin–Karp:** pior caso `O(n·m)` com colisões/alta repetição; métricas
  internas de hash não expostas (contrato sem `metadata`).
- Benchmark agregado e gráficos pertencem ao Carlos; cenários/corpus ao
  Yann; baseline Naive/KMP ao Wilson.
