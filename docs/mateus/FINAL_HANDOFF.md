# Handoff Final — Mateus Almeida (FastContext / PAA UFS 2026.2)

Documento autocontido para revisão humana e auditoria externa (incl. ChatGPT).
Escopo: `feature/classical-algorithms` e `feature/tests`.

## A. Resumo executivo

Foram implementados manualmente, testados e documentados os três algoritmos
clássicos do escopo de Mateus — **Merge Sort**, **Quick Sort** e **Top-k com
min-heap** — todos respeitando o critério canônico de ordenação (score DESC;
empate por chunk_id ASC), com instrumentação de comparações e demais métricas.
Nenhum `sorted()`/`list.sort()`/`heapq` esconde a lógica avaliada. Os algoritmos
são puros e desacoplados dos contratos de retrieval (Wilson) e RAG (Carlos).
Corretude e complexidade estão documentadas e ligadas ao código. A suíte de
testes do escopo passa (76 testes) com 100% de cobertura em `src/algorithms`.
Docker Linux não foi executado (indisponível na máquina).

## B. Checklist do escopo

| Requisito | Status | Evidência |
|---|---|---|
| Merge Sort manual | ✅ | `src/algorithms/merge_sort.py`, `tests/test_merge_sort.py` |
| Quick Sort manual | ✅ | `src/algorithms/quick_sort.py`, `tests/test_quick_sort.py` |
| Top-k min-heap manual | ✅ | `src/algorithms/topk_heap.py`, `tests/test_topk_heap.py` |
| Comparação instrumentada | ✅ | `*Stats.comparisons` nos três + testes |
| Critério de desempate central | ✅ | `src/algorithms/ordering.py` |
| Sem `sorted()`/`heapq` na lógica | ✅ | revisão de código; oráculo só em teste |
| Casos de borda | ✅ | matrizes nos relatórios 01–03/05 |
| Complexidade documentada | ✅ | `docs/mateus/04_...md` |
| Recorrência do Merge Sort | ✅ | `docs/CORRECTNESS.md`, `04_...md` |
| Corretude do Merge Sort | ✅ | `docs/CORRECTNESS.md` §1 |
| Testes específicos passam | ✅ | 76 passed (escopo) |
| Integração sem duplicar contrato | ✅ | algoritmos puros; regressão de ordenação |
| Docker | ⚠️ | NÃO EXECUTADO (indisponível) |
| Relatórios de fase | ✅ | `docs/mateus/01–05`, `AI_USAGE_LOG`, este arquivo |
| Integração real nos retrievers | ⚠️ | pendente (decisão de equipe) |

Legenda: ✅ pronto · ⚠️ pendente/parcial · ❌ não feito.

## C. Arquivos alterados/criados

### Branch `feature/classical-algorithms` (algoritmos + provas + handoffs 01–04)
- `src/algorithms/ordering.py` (novo)
- `src/algorithms/merge_sort.py`
- `src/algorithms/quick_sort.py`
- `src/algorithms/topk_heap.py`
- `tests/test_merge_sort.py`
- `tests/test_quick_sort.py`
- `tests/test_topk_heap.py`
- `tests/test_ordering.py`
- `docs/CORRECTNESS.md`
- `docs/mateus/01_MERGE_SORT_RESULT.md`
- `docs/mateus/02_QUICK_SORT_RESULT.md`
- `docs/mateus/03_TOPK_HEAP_RESULT.md`
- `docs/mateus/04_CORRECTNESS_COMPLEXITY_RESULT.md`

### Branch `feature/tests` (consolidação de testes + handoffs de fechamento)
- `tests/test_search_consolidated.py` (busca linear/binária — assinaturas do Wilson)
- `tests/test_ranking_integration.py` (regressão do contrato de ordenação)
- `docs/mateus/05_TESTS_RESULT.md`
- `docs/mateus/AI_USAGE_LOG.md`
- `docs/mateus/FINAL_HANDOFF.md`
- (mais os arquivos de algoritmo/teste herdados de `feature/classical-algorithms`
  quando esta for integrada — ver seção I)

`binary_search.py` e `linear_search.py` NÃO são commitados por Mateus: são de
autoria do Wilson e foram usados apenas localmente para testar.

## D. Funções públicas relevantes

```python
# ordering.py
default_key(item) -> (float, str)          # extrai (score, chunk_id)
precedes(key_a, key_b) -> bool             # score DESC, chunk_id ASC

# merge_sort.py
merge_sort(items, key=default_key) -> (list, MergeSortStats)

# quick_sort.py  (pivô: mediana de três)
quick_sort(items, key=default_key) -> (list, QuickSortStats)

# topk_heap.py   (min-heap manual, O(N log k), O(k))
top_k(items, k, key=default_key) -> (list, TopKStats)
```

Padrão uniforme: `algoritmo(items) -> (resultado, Stats)`. Entrada não é mutada.

## E. Integração com retrieval / contrato

- Os retrievers do Wilson produzem candidatos `(score, chunk_dict)` e hoje
  ordenam com `candidates.sort(key=lambda it: (-it[0], it[1]["chunk_id"]))` e
  fatiam `[:k]`. Os algoritmos de Mateus reproduzem exatamente essa ordenação
  (verificado em `tests/test_ranking_integration.py`), de modo que substituir o
  `.sort()` por Merge/Quick e o fatiamento por `top_k` é direto.
- `default_key` aceita tanto `(score, payload)` quanto objetos/dicts, então
  também é compatível com `RetrievedChunk` (que expõe `score` e `chunk_id`).
- **Contrato divergente:** `feature/retrieval` (`base.py`, Wilson) usa
  `RetrievalResult(query, k, retriever_name, chunks: List, ...)`; `feature/rag`
  (`models.py`, Carlos) usa `RetrievalResult(query, algorithm, top_k, chunks:
  tuple, ...)` com validações fortes. A unificação é decisão de equipe, fora do
  escopo puro de Mateus.

## F. Evidência de testes (resumo real dos comandos executados)

```text
# escopo de Mateus + cobertura
python -m pytest tests/test_merge_sort.py tests/test_quick_sort.py \
  tests/test_topk_heap.py tests/test_ordering.py \
  tests/test_search_consolidated.py tests/test_ranking_integration.py \
  --cov=src/algorithms --cov-report=term-missing -q
=> 76 passed ; cobertura src/algorithms = 100%

# suíte completa da branch feature/tests
python -m pytest -q
=> 79 passed   (inclui test_config.py: 3)

# Docker Linux
=> NÃO EXECUTADO (docker indisponível na máquina)
```

Ambiente: Windows, Python 3.13.1, pytest 9.1.1, pytest-cov 7.1.0, `.venv` local
não versionado.

## G. Complexidade (tabela comparativa)

| Algoritmo | Melhor | Médio | Pior | Espaço |
|---|---:|---:|---:|---:|
| Merge Sort | Θ(N log N) | Θ(N log N) | Θ(N log N) | Θ(N) + Θ(log N) pilha |
| Quick Sort | Θ(N log N) | Θ(N log N) | Θ(N²) | O(log N)–O(N) pilha |
| Top-k Heap | O(N log k) | O(N log k) | O(N log k) | O(k) |
| Busca binária (Wilson) | Θ(1) | Θ(log N) | Θ(log N) | O(1) |

## H. Corretude (resumo)

- **Merge Sort:** invariante do merge (mantém o menor global no topo; estável em
  empate) + indução forte na recursão; divisão cobre todos os elementos.
- **Quick Sort:** invariante da fronteira de Lomuto; pivô posicionado
  corretamente; indução nas duas partições.
- **Top-k:** invariante "após t itens, a heap contém os min(t,k) melhores; a raiz
  é o pior"; ordenação final por heapsort invertido.
- **Busca binária:** invariante "se existe, está em [left,right]"; intervalo
  encolhe; termina achando ou esvaziando.

Detalhes em `docs/CORRECTNESS.md`.

## I. Questões pendentes (dependências externas)

- **Wilson (`feature/retrieval`):** `binary_search.py`/`linear_search.py`/
  `inverted_index.py` e retrievers; a integração dos algoritmos de Mateus depende
  do merge dessa branch e da unificação do contrato.
- **Carlos (`feature/rag`):** contrato `models.py`/adapter; tratamento de `k=0`
  no nível de `RetrievalResult` (exige `top_k > 0`).
- **Merge para `main`:** nada foi integrado à `main` ainda; `feature/tests`
  depende de `feature/classical-algorithms` para os testes de algoritmo rodarem.
- **Docker Linux:** executar quando disponível.

## J. Riscos (pontos que o professor pode questionar)

1. Implementação manual real (não delegada a biblioteca) — evidenciável no código.
2. Pior caso O(N²) do Quick Sort apesar da mediana de três — documentado.
3. `k=0` e o contrato `top_k>0` — tratado no algoritmo; camada pendente.
4. Contrato de retrieval divergente entre colegas — fora do escopo, registrado.
5. Docker não executado — declarado explicitamente, sem fabricar evidência.

## K. Material para apresentação (bullets)

- O sistema recupera candidatos, ordena por relevância e seleciona os melhores;
  a parte de Mateus cuida da ordenação e da seleção Top-k.
- Merge Sort garante desempenho previsível Θ(N log N) em qualquer entrada.
- Quick Sort é a alternativa in-place; usamos mediana de três para evitar os
  piores casos triviais, mas o pior caso teórico continua O(N²).
- Top-k com min-heap pega os k melhores em O(N log k) sem ordenar tudo — ideal
  quando k é muito menor que N.
- Todos usam o mesmo critério: maior score primeiro; empate pelo id do chunk.
- Cada algoritmo conta comparações, permitindo comparar custo experimentalmente.
- 76 testes passam com 100% de cobertura nos algoritmos; Docker ainda pendente.
- Os algoritmos são independentes do LLM e do RAG — o núcleo funciona sem IA.

## L. Perguntas prováveis do professor (com respostas curtas)

1. **Por que Merge Sort é Θ(N log N) sempre?** Porque divide sempre ao meio e o
   merge é linear; a recorrência `2T(N/2)+Θ(N)` resolve em Θ(N log N).
2. **Onde está o pior caso do Quick Sort?** Em partições sistematicamente
   desbalanceadas (pivô extremo repetido / muitas chaves iguais): O(N²).
3. **A mediana de três elimina o pior caso?** Não; reduz os casos triviais
   (ordenado/inverso), mas o pior caso teórico permanece O(N²).
4. **Por que Top-k é O(N log k)?** N itens processados, cada operação de heap de
   tamanho k custa O(log k).
5. **Por que a memória do Top-k é O(k)?** A heap nunca guarda mais que k itens.
6. **Como garantem o desempate?** Função única `precedes` (score DESC, chunk_id
   ASC) reutilizada pelos três algoritmos.
7. **Usaram biblioteca pronta?** Não na lógica avaliada; `sorted()` aparece só
   como oráculo nos testes; a heap é manual (sem `heapq`).
8. **Como medem as comparações?** Contador incrementado a cada comparação de
   chave dentro de cada algoritmo.
9. **Como sabem que está correto?** Provas por invariante/indução em
   `CORRECTNESS.md` + testes contra oráculo e casos de borda.
10. **Como isso se conecta ao retrieval?** Os algoritmos reproduzem a ordenação
    que os retrievers já usam; a fiação é direta (teste de regressão comprova).
11. **E o caso k=0?** O algoritmo retorna vazio; o contrato de resultado exige
    top_k>0, então a camada de retrieval trata k=0 antes de montar o resultado.
12. **Rodaram em Docker?** Não; declarado como NÃO EXECUTADO por indisponibilidade.

## M. Declaração de IA

O desenvolvimento foi assistido pelo Kiro, com aprovação humana por fase e
decisões de arquitetura tomadas por Mateus. Detalhes e prompts em
`docs/mateus/AI_USAGE_LOG.md`. Nenhum resultado não verificado foi declarado.
