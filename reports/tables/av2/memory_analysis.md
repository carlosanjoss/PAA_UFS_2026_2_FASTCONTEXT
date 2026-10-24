# AV2 — Análise de memória

## Experimento

- Corpus: FastAPI 0.141.0.
- Cenários: 24.
- Algoritmos: 4.
- Combinações de memória: 96.
- Medição: Python tracemalloc.
- Memória medida separadamente dos tempos.

## Padrão OAuth2 — 100% do corpus

| Algoritmo | Pico (MiB) | Diferença vs Naive |
|---|---:|---:|
| naive | 0.216934 | +0.00% |
| kmp | 0.252003 | +16.17% |
| boyer-moore | 0.221176 | +1.96% |
| rabin-karp | 0.220047 | +1.43% |

## Interpretação

Os valores representam picos de alocações rastreadas pelo Python em execuções específicas.

As medições incluem objetos produzidos pelo executor, como estruturas de resultados e listas de posições. Os textos de entrada já estavam carregados antes do início do rastreamento.

Portanto, não se trata de uma medição isolada da memória auxiliar dos algoritmos.

O tracemalloc também não mede toda a memória residente do processo nem, necessariamente, alocações nativas.

O custo espacial teórico deve ser discutido separadamente, a partir das estruturas de dados e das análises de complexidade.

O executor conserva todos os resultados de cada cenário durante a medição, de modo que cenários com muitas ocorrências podem apresentar picos maiores.

Os casos sintéticos foram apresentados em gráfico separado porque têm distribuições e quantidades de ocorrências distintas do corpus real.

A memória foi medida uma vez por combinação. Não foram calculados desvios-padrão ou intervalos de confiança para essa métrica.

## Arquivos

- memory_real_patterns.png
- memory_synthetic_patterns.png
- memory_analysis.csv
