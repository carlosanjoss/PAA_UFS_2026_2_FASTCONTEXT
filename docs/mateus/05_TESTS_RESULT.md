# Relatório 5 — Testes (feature/tests)

Fase 5 do escopo de Mateus Almeida (`feature/tests`).
Documento de handoff para revisão humana e auditoria externa.

## Objetivo

Consolidar os testes automatizados do escopo de Mateus: Merge Sort, Quick Sort,
Top-k, o critério de ordenação, e as buscas linear/binária do Wilson (contra as
assinaturas reais), além de testes de integração de ranking como regressão do
contrato de ordenação. Avaliar cobertura. Sem alterar contratos ou lógica dos
colegas. Registrar Docker como NÃO EXECUTADO se indisponível.

## Como o ambiente de teste foi montado (transparência)

O trabalho ocorre na branch `feature/tests`. Estado observado:

- `feature/tests` e `main` estão no mesmo commit (`4bf0890`); os arquivos de
  algoritmo e de teste estavam vazios.
- As implementações de Merge/Quick/Top-k + `ordering.py` e seus testes foram
  produzidos por Mateus (Fases 1–3) e estão presentes no working tree (ainda não
  commitados, conforme instrução).
- Para testar as buscas do Wilson, os arquivos `src/algorithms/binary_search.py`
  e `src/algorithms/linear_search.py` foram trazidos **apenas para o working
  tree local**, de forma controlada, com:

  ```bash
  git checkout origin/feature/retrieval -- \
      src/algorithms/binary_search.py \
      src/algorithms/linear_search.py
  ```

  Nenhuma lógica do Wilson foi alterada; os arquivos foram usados exatamente como
  na `feature/retrieval`. Nenhum commit/push/merge foi feito.

- Ambiente Python: `.venv` local (não versionado), Python 3.13.1, pytest 9.1.1,
  pytest-cov 7.1.0. Dependências instaladas via `requirements.txt`.

## Arquivos de teste (autoria de Mateus)

| Arquivo | Testes | Escopo |
|---|---:|---|
| `tests/test_merge_sort.py` | 14 | Merge Sort |
| `tests/test_quick_sort.py` | 17 | Quick Sort |
| `tests/test_topk_heap.py` | 18 | Top-k min-heap |
| `tests/test_ordering.py` | 7 | critério canônico (`precedes`, `default_key`) |
| `tests/test_search_consolidated.py` | 15 | busca linear e binária (assinaturas reais do Wilson) |
| `tests/test_ranking_integration.py` | 5 | regressão do contrato de ordenação (algoritmos ↔ estilo retriever) |

## Matriz de casos testados

### Merge Sort / Quick Sort (comuns)
vazio; 1 elemento; 2 elementos; já ordenado; ordem inversa; empates; muitos
empates; scores negativos/zero; equivalência com oráculo `sorted()`; preservação
de elementos; determinismo; entrada não mutada; contagem de comparações.
Quick Sort adiciona: entrada ordenada tratada sem degenerar (`max_depth < N`),
ordem inversa, todos scores iguais.

### Top-k
`k=0`; `k<0`; `k=1`; `k=N`; `k>N`; lista vazia; vazio+`k=0`; saída DESC;
desempate por chunk_id; empate na fronteira do corte; muitos empates;
negativos/zero; "os k são de fato os melhores" (60 casos aleatórios); tamanho
limitado a k; determinismo; entrada não mutada; heap nunca excede k; contadores.

### Busca binária (assinatura real do Wilson)
coleção vazia; um elemento (presente/ausente); termo no início; no meio; no fim;
ausente; limite logarítmico de comparações.

### Busca linear (assinatura real do Wilson)
coleção vazia; um elemento; início; meio; fim; ausente; primeira ocorrência em
duplicatas.

### Integração de ranking (regressão)
Merge Sort e Quick Sort reproduzem a ordenação `(-score, chunk_id)` usada hoje
pelos retrievers; Top-k equivale a ordenar (estilo retriever) e fatiar os k
primeiros; os três algoritmos concordam no ranking completo; exemplo canônico da
seção 8 do documento de contexto.

## Resultado local

Comando (escopo de Mateus + cobertura de algoritmos):

```bash
python -m pytest tests/test_merge_sort.py tests/test_quick_sort.py \
  tests/test_topk_heap.py tests/test_ordering.py \
  tests/test_search_consolidated.py tests/test_ranking_integration.py \
  --cov=src/algorithms --cov-report=term-missing -q
```

Resultado: **PASSOU — 76 passed in 0.79s.**

Cobertura de `src/algorithms`:

```text
Name                               Stmts   Miss  Cover
src/algorithms/__init__.py             0      0   100%
src/algorithms/binary_search.py       14      0   100%
src/algorithms/inverted_index.py       0      0   100%
src/algorithms/linear_search.py        8      0   100%
src/algorithms/merge_sort.py          48      0   100%
src/algorithms/ordering.py            22      0   100%
src/algorithms/quick_sort.py          61      0   100%
src/algorithms/topk_heap.py           65      0   100%
TOTAL                                218      0   100%
```

Suíte completa da branch (inclui `test_config.py`, 3 testes):

```bash
python -m pytest -q
# 79 passed in 0.50s
```

Distribuição coletada: test_config 3, test_merge_sort 14, test_ordering 7,
test_quick_sort 17, test_ranking_integration 5, test_search_consolidated 15,
test_topk_heap 18.

## Resultado Docker

**Docker Linux: NÃO EXECUTADO.**
Motivo: `docker` não está disponível/instalado nesta máquina (verificado com
`Get-Command docker` → não encontrado). A reprodutibilidade em Docker Linux
permanece pendente e deve ser executada por quem tiver o Docker Desktop
disponível, com:

```bash
docker compose run --rm fastcontext python -m pytest
```

## Regressões e bugs

- Nenhum bug encontrado nas implementações de Mateus durante a consolidação.
- Nenhuma regressão detectada.
- `test_ranking_integration.py` funciona como **teste de regressão do contrato de
  ordenação**: se algum algoritmo divergir do critério `(-score, chunk_id)` que
  os retrievers assumem, o teste falha.

## Testes que NÃO puderam ser realizados (e por quê)

- **Integração real com os retrievers do Wilson** (`LinearRetriever`,
  `IndexedRetriever`, `OptimizedRetriever`) e com `RetrievalResult`: NÃO
  REALIZADA nesta fase. Motivo: por decisão registrada, os algoritmos de Mateus
  permanecem puros e desacoplados; a fiação nos retrievers e a unificação do
  contrato divergente (Wilson `base.py` × Carlos `models.py`) dependem de
  alinhamento da equipe. O Wilson já mantém testes próprios de retriever em
  `feature/retrieval`.
- **`k=0` no nível de `RetrievalResult`**: o contrato do Carlos exige `top_k > 0`,
  então o caso precisa ser tratado na camada de retrieval antes de construir o
  resultado. Testado no nível de algoritmo (`top_k` retorna `[]`), mas o
  comportamento de camada não foi testado aqui por não alterarmos os retrievers.
- **Docker**: NÃO EXECUTADO (ver acima).

## Decisões tomadas pelo Kiro

1. Consolidação sem duplicar os testes de retriever já existentes do Wilson;
   foco no escopo de Mateus + regressão de ordenação.
2. Testes de busca escritos contra as assinaturas REAIS do Wilson
   `func(elements, target) -> (index, comparisons)`.
3. Arquivos do Wilson trazidos apenas ao working tree, documentado acima, sem
   commit e sem alteração de lógica.
4. Cobertura medida em `src/algorithms` (100%).

## Itens que Mateus deve revisar manualmente

- Executar a suíte em Docker Linux quando disponível e atualizar este relatório.
- Decidir, com a equipe, quando integrar os algoritmos aos retrievers e como
  unificar o contrato divergente (fora do escopo puro de Mateus).
- Confirmar se deseja commitar as implementações em `feature/classical-algorithms`
  antes de consolidar os testes em `feature/tests` (hoje ambas compartilham o
  working tree local, sem commit).

## Estado de ambiente

- Testes locais: **PASSOU** (76/76 no escopo; 79/79 na suíte completa).
- Cobertura `src/algorithms`: **100%**.
- Docker Linux: **NÃO EXECUTADO** — Docker indisponível nesta máquina.
