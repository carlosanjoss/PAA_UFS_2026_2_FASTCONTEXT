# Metodologia Experimental — FastContext

## 1. Objetivo

Os experimentos avaliam as estratégias de recuperação em termos de:

- tempo;
- custo algorítmico;
- memória;
- escalabilidade;
- qualidade da recuperação.

A camada experimental consome os retrievers existentes sem alterar sua lógica interna para favorecer resultados.

---

## 2. Corpus experimental

| Propriedade | Valor |
|---|---|
| Fonte | Documentação oficial do FastAPI |
| Versão | `0.141.0` |
| Idioma | Inglês |
| Documentos | `153` |
| Chunks | `1305` |
| Máximo por chunk | `400` tokens |
| Overlap | `60` tokens |
| Seed | `42` |

Fingerprint:

```text
e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d
```

### 2.1 Exclusão de `release-notes.md`

Antes:

```text
2859 chunks
```

Depois:

```text
1305 chunks
```

Contribuição do arquivo excluído:

```text
1554 chunks ≈ 54.36%
```

A exclusão foi aplicada somente ao corpus experimental preparado para reduzir forte concentração em um único documento.

---

## 3. Estratégias avaliadas

| Estratégia | Componentes |
|---|---|
| Linear | TF-IDF + sequential scan + Merge Sort |
| Indexed | Inverted Index + Binary Search + TF-IDF + Merge Sort |
| Optimized | Inverted Index + Binary Search + TF-IDF + bounded Top-k heap |
| Semantic | BGE-small + FAISS |

---

## 4. Tamanhos do corpus

Frações:

```text
25%
50%
75%
100%
```

Os subconjuntos são determinísticos, aninhados e preservam a ordem relativa original dos chunks.

Para `1305` chunks:

| Fração | Chunks |
|---:|---:|
| 25% | 326 |
| 50% | 652 |
| 75% | 978 |
| 100% | 1305 |

---

## 5. Consultas

Total:

```text
30 queries
```

Categorias:

- Tutorial;
- Dependencies;
- Security;
- Middleware/CORS;
- Background Tasks;
- WebSockets;
- Deployment;
- Testing;
- Advanced.

---

## 6. Matriz experimental principal

Cada cenário possui `5` repetições medidas.

```text
4 algorithms
× 4 corpus sizes
× 30 queries
× 5 repetitions
= 2400 measured retrieval executions
```

Cada retrieval executa:

```text
top_k = 10
```

A mesma lista ordenada é reutilizada para calcular:

```text
k ∈ {1, 3, 5, 10}
```

Os valores de `k` não multiplicam o total para 9600 execuções.

---

## 7. Warm-up

Quando configurado, warm-up é executado antes das medições.

Warm-up:

- não entra nas médias;
- não entra nos desvios;
- não entra nas 2400 execuções medidas.

---

## 8. Métricas de desempenho

Quando disponíveis:

- tempo externo total;
- tempo interno de retrieval;
- tempo de ordenação;
- tempo de construção de índice;
- tempo de carregamento de índice;
- pico aproximado de memória;
- comparações;
- chunks pontuados;
- candidatos encontrados;
- quantidade de resultados.

Na estratégia Semantic também podem aparecer:

- query embedding time;
- FAISS search time;
- `index_source`;
- `persistence_status`.

---

## 9. Instrumentação da busca binária

A implementação atual registra as comparações de chave efetivamente executadas:

```text
elements[mid] == target  -> +1
```

Se a igualdade falhar:

```text
elements[mid] < target   -> +1
```

Logo:

- encontro no `mid`: 1 comparação na iteração;
- decisão esquerda/direita: 2 comparações na iteração.

A complexidade continua `Θ(log N)`.

---

## 10. Medição de tempo

Unidade bruta:

```text
nanoseconds
```

Relatórios podem converter para milissegundos.

Setup de retriever é separado da execução de query. Em Semantic, carregamento do modelo e do índice não é confundido com latência de consulta.

---

## 11. Memória

O profiling de memória ocorre em uma execução separada da execução cronometrada.

Essa execução adicional não conta entre as 2400 medições de retrieval.

A métrica baseada em `tracemalloc` representa principalmente memória gerenciada pelo Python. Alocações nativas de FAISS, NumPy e PyTorch podem não estar totalmente refletidas.

---

## 12. Estatísticas

São calculados:

- `count`;
- média;
- mediana;
- desvio padrão amostral;
- mínimo;
- máximo.

Para uma única observação:

```text
std = 0
```

---

## 13. Métricas de qualidade

### Precision@k

```text
Precision@k = relevant retrieved in top-k / k
```

### Recall@k

```text
Recall@k = relevant retrieved in top-k /
           relevant chunks available in the evaluated corpus
```

### Reciprocal Rank

```text
RR = 1 / rank of first relevant result
```

Se não houver relevante recuperado:

```text
RR = 0
```

### Mean Reciprocal Rank

```text
MRR = mean(RR)
```

### Hit Rate@k

```text
1, se houver pelo menos um relevante em top-k
0, caso contrário
```

---

## 14. Ground truth

A relevância é binária.

Um chunk é relevante quando:

1. responde diretamente à pergunta; ou
2. contém informação necessária para construir uma resposta tecnicamente correta.

Sobreposição lexical isolada não é suficiente.

---

## 15. Pooling

Para cada query são obtidos os primeiros `20` resultados de:

- Linear;
- Indexed;
- Optimized;
- Semantic.

A união forma o pool de julgamento.

Duplicatas entre sistemas são julgadas uma única vez por anotador.

---

## 16. Pool observado

```text
30 queries
963 unique query/chunk judgments per annotation pass
```

| Estatística | Candidatos por query |
|---|---:|
| Mínimo | 27 |
| Média | 32.10 |
| Máximo | 38 |

Dois anotadores primários implicam:

```text
1926 primary human judgments
```

---

## 17. Blind annotation

Os arquivos entregues aos avaliadores não mostram:

- algoritmo de origem;
- rank original;
- score original.

---

## 18. Protocolo humano

Cada query possui:

```text
2 independent primary annotators
```

Fluxo:

```text
agreement -> final primary label
```

```text
disagreement -> third-person adjudication
```

O adjudicador deve ser diferente dos dois avaliadores primários.

---

## 19. Concordância

A infraestrutura calcula:

- pares avaliados;
- acordos;
- desacordos;
- taxa de concordância;
- Cohen's kappa por pares;
- estatísticas por query.

Como os pares variam entre queries, um kappa agregado deve ser tratado apenas como estatística descritiva adicional.

---

## 20. Limitação do pooling

O ground truth é pooled. Portanto, não há garantia de que todos os chunks relevantes existentes nos `1305` chunks tenham sido julgados.

O Recall deve ser interpretado em relação ao conjunto relevante identificado no pool julgado, e não como recall absoluto de todo o corpus.

---

## 21. Qualidade em subconjuntos

Nos subconjuntos de 25%, 50% e 75%, chunks relevantes do ground truth completo podem estar ausentes.

Nesse caso:

- o denominador do Recall considera apenas relevantes disponíveis no subconjunto;
- se nenhum relevante estiver disponível, a qualidade fica não avaliável;
- o valor não é convertido artificialmente para zero.

---

## 22. Persistência semântica

O corpus completo reutiliza o índice FAISS persistido quando compatível.

Fingerprint esperado:

```text
e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d
```

Compatibilidade considera fingerprint, chunk count, modelo, dimensão, normalização e tipo FAISS.

---

## 23. Benchmark Merge Sort × Quick Sort

Matriz:

| Parâmetro | Valores |
|---|---|
| Algoritmos | Merge Sort, Quick Sort |
| Tamanhos | 100, 250, 500, 1000, 1305 |
| Cenários | random, already_sorted, reverse_sorted, many_ties |
| Repetições | 5 |

Total:

```text
2 × 5 × 4 × 5 = 200 measured sorting executions
```

Validação concluída:

```text
Raw rows: 200
Summary rows: 40
Comparable cases: 100
Fingerprint mismatches: 0
```

O benchmark de sorting está concluído.

---

## 24. Quick Sort e ordem de entrada

A implementação usa mediana de três. Portanto, `already_sorted` e `reverse_sorted` são cenários de sensibilidade, não automaticamente pior caso.

Pior caso teórico:

```text
O(N²)
```

---

## 25. Artefatos

```text
experiments/raw_results/
experiments/processed_results/
reports/tables/
reports/figures/
```

Resultados brutos não devem ser editados manualmente.

---

## 26. Estado atual

### Concluído

- runner experimental;
- métricas de retrieval;
- estatísticas;
- profiling;
- pooling;
- adjudicação;
- reporting;
- benchmark Merge Sort × Quick Sort.

### Aguardando avaliação humana

- anotações;
- adjudicação dos desacordos;
- ground truth final.

### Depois do ground truth

- benchmark final das 2400 execuções;
- Precision@k;
- Recall@k;
- MRR;
- Hit Rate;
- tabelas e gráficos finais;
- análise quantitativa.

---

## 27. Reprodutibilidade

Antes do benchmark final:

```bash
python -m pytest -q
python -m ruff check .
python -m mypy src
```

A execução final deve usar corpus congelado, ground truth concluído, configuração versionada, código testado e commit Git identificado.
