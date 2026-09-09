# Arquitetura do FastContext

## 1. Visão geral

O **FastContext** é um sistema experimental de recuperação de contexto desenvolvido para a disciplina **Projeto e Análise de Algoritmos — UFS 2026.2**. O objetivo é analisar como diferentes algoritmos, estruturas de dados e estratégias de representação afetam o custo, a escalabilidade e a qualidade da recuperação de contexto antes de esse contexto ser entregue a um modelo de linguagem.

O projeto mantém duas camadas separadas:

1. **Recuperação e análise algorítmica**, independente de LLM.
2. **RAG e geração de resposta**, utilizada como camada complementar.

---

## 2. Pipeline geral

```text
FastAPI Documentation
        |
        v
Corpus acquisition and freezing
        |
        v
Normalization
        |
        v
Heading-aware chunking
        |
        v
Context chunks
        |
        +---------------------------+
        |                           |
        v                           v
Lexical representation       Semantic representation
TF-IDF                       BGE embeddings
        |                           |
        v                           v
Classical retrieval          FAISS retrieval
        |                           |
        +-------------+-------------+
                      |
                      v
               RetrievalResult
                      |
             +--------+--------+
             |                 |
             v                 v
      Retrieval only       RAG Pipeline
                                |
                                v
                             Ollama
                                |
                                v
                         Grounded answer
```

---

## 3. Corpus experimental congelado

| Propriedade | Valor |
|---|---|
| Versão do FastAPI | `0.141.0` |
| Idioma | Inglês |
| Documentos preparados | `153` |
| Chunks | `1305` |
| Máximo por chunk | `400` tokens |
| Overlap | `60` tokens |
| Seed experimental | `42` |
| Arquivo excluído do corpus preparado | `release-notes.md` |

Fingerprint SHA-256:

```text
e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d
```

A exclusão de `release-notes.md` foi aplicada somente ao corpus experimental preparado. O arquivo bruto original não foi apagado. Após o início da construção do ground truth, o corpus deve ser tratado como imutável.

---

## 4. Chunking

O chunking é **heading-aware** e preserva a estrutura da documentação. Cada chunk mantém, conforme disponível:

- `chunk_id`;
- `source_path`;
- `section_title`;
- `content`;
- `token_count`;
- metadados adicionais.

O `chunk_id` funciona como identificador estável durante retrieval, pooling, ground truth e avaliação.

---

## 5. Contratos de retrieval

### 5.1 `RetrievedChunk`

Campos principais:

- `chunk_id`;
- `content`;
- `source_path`;
- `section_title`;
- `score`;
- `rank`;
- `token_count`, opcional;
- `metadata`, opcional.

### 5.2 `RetrievalMetrics`

Campo obrigatório:

- `retrieval_time_ns`.

Campos opcionais:

- `sorting_time_ns`;
- `index_build_time_ns`;
- `comparisons`;
- `chunks_scored`;
- `candidates_found`;
- `peak_memory_mb`.

Uma métrica indisponível é representada por **`None`**, e não por zero.

### 5.3 `RetrievalResult`

Contém:

- `query`;
- `algorithm`;
- `top_k`;
- `chunks`;
- `metrics`;
- `metadata`, opcional.

---

## 6. Critério de ranking

Todos os algoritmos clássicos seguem a mesma ordem canônica:

1. score decrescente;
2. em empate, `chunk_id` crescente.

Formalmente, `a` precede `b` quando:

```text
a.score > b.score
```

ou:

```text
a.score == b.score and a.chunk_id < b.chunk_id
```

---

## 7. Configuração A — Linear

```text
Query
  |
  v
TF-IDF
  |
  v
Sequential scan over all chunks
  |
  v
Cosine similarity
  |
  v
Manual Merge Sort
  |
  v
Top-k
```

Custos dominantes esperados:

```text
Scoring:  Θ(N)
Sorting:  Θ(N log N)
```

---

## 8. Configuração B — Indexed

```text
Query
  |
  v
Query terms
  |
  v
Sorted vocabulary
  |
  v
Manual Binary Search
  |
  v
Inverted Index
  |
  v
Candidate set
  |
  v
TF-IDF scoring
  |
  v
Manual Merge Sort
  |
  v
Top-k
```

O índice invertido reduz o conjunto de candidatos antes do ranking. A busca binária é implementada manualmente sobre o vocabulário ordenado.

### Instrumentação da busca binária

A implementação atual contabiliza as comparações de chave realmente executadas:

- `elements[mid] == target`: `+1`;
- se não houver igualdade, `elements[mid] < target`: `+1`.

Assim, uma iteração usa 1 comparação se encontrar o alvo e 2 quando precisa decidir esquerda ou direita. A complexidade continua `Θ(log N)`.

---

## 9. Configuração C — Optimized

```text
Query
  |
  v
Inverted Index + Binary Search
  |
  v
Candidate set
  |
  v
TF-IDF scoring
  |
  v
Bounded min-heap
  |
  v
Top-k
```

Seleção:

```text
O(N log k)
```

Ordenação completa:

```text
O(N log N)
```

A vantagem é mais relevante quando `k << N`.

---

## 10. Configuração D — Semantic

Componentes:

- `BAAI/bge-small-en-v1.5`;
- embeddings normalizados;
- dimensão `384`;
- FAISS `IndexFlatIP`.

```text
Query
  |
  v
BGE embedding
  |
  v
Normalized query vector
  |
  v
FAISS IndexFlatIP
  |
  v
Nearest vectors
  |
  v
chunk_id mapping
  |
  v
Top-k
```

A busca semântica funciona como baseline complementar e não substitui os algoritmos clássicos obrigatórios.

---

## 11. Persistência semântica

Artefatos persistidos:

```text
data/processed/semantic/
```

Antes do reuso, o sistema verifica compatibilidade com:

- fingerprint do corpus;
- quantidade de chunks;
- modelo;
- dimensão;
- normalização;
- tipo de índice FAISS.

Quando compatível, o índice é carregado em vez de reconstruído.

---

## 12. Algoritmos clássicos

Implementados em `src/algorithms/`:

- Linear Search;
- Binary Search;
- Inverted Index;
- Merge Sort;
- Quick Sort;
- Top-k Heap.

Bibliotecas podem ser utilizadas como referência ou validação, mas não para substituir o algoritmo avaliado na medição principal.

---

## 13. Merge Sort

```text
T(N) = 2T(N/2) + Θ(N)
```

Pelo Teorema Mestre:

```text
T(N) = Θ(N log N)
```

A prova de corretude está em `docs/CORRECTNESS.md`.

---

## 14. Quick Sort

A implementação usa particionamento de Lomuto e pivô determinístico por mediana de três.

| Caso | Complexidade |
|---|---:|
| Melhor | `Θ(N log N)` |
| Médio | `Θ(N log N)` |
| Pior | `Θ(N²)` |

A mediana de três reduz casos triviais de degeneração, mas não elimina o pior caso teórico.

---

## 15. Top-k Heap

Complexidade:

```text
O(N log k)
```

Espaço:

```text
O(k)
```

---

## 16. Camada de aplicação

O `FastContextService` centraliza as operações de:

- retrieval;
- retrieval + geração.

A interface Streamlit reutiliza essa camada e não implementa algoritmos de domínio.

---

## 17. RAG e LLM

Fluxo:

```text
Query
  |
  v
Retriever
  |
  v
Retrieved chunks
  |
  v
Context conversion
  |
  v
RAG prompt
  |
  v
LLM provider
  |
  v
Answer + citations
```

Retrieval e geração possuem tempos independentes.

Provider utilizado no desenvolvimento:

```text
Ollama
```

Modelo principal:

```text
qwen2.5:3b
```

Se o provider estiver indisponível, retrieval-only continua funcionando.

---

## 18. Streamlit

Entrada:

```text
app/streamlit_app.py
```

A interface permite selecionar estratégia e `top-k`, executar retrieval ou RAG, visualizar métricas, chunks, resposta, citações e status do sistema.

A interface não altera corpus, ground truth, índice ou resultados experimentais.

---

## 19. Experimentos

A camada experimental é independente da interface.

Diretórios principais:

```text
experiments/
scripts/
src/evaluation/
reports/
```

Metodologia: `docs/experiments.md`.

---

## 20. Testes e invariantes

Comandos principais:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
```

Invariantes:

1. corpus experimental congelado;
2. ranking `score DESC` + `chunk_id ASC`;
3. retrieval separado de geração;
4. `None` diferente de zero;
5. índice semântico validado antes do reuso;
6. Streamlit não escreve resultados experimentais;
7. ground truth humano independente dos rankings;
8. resultados brutos preservados.
