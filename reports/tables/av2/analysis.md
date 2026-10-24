# AV2 — Análise descritiva do benchmark

Padrão em destaque: `P-OAUTH2`. Escala de referência: 100%.

| Algoritmo | Tempo médio (ms) | Desvio-padrão (ms) | Speedup vs Naive |
|---|---:|---:|---:|
| naive | 148.276 | 5.449 | 1.00x |
| kmp | 64.410 | 4.680 | 2.30x |
| boyer-moore | 27.780 | 2.283 | 5.34x |
| rabin-karp | 322.026 | 25.597 | 0.46x |

## Metodologia e limites

- Os tempos são médias de execuções medidas, sem incluir o aquecimento.
- Cada cenário compara algoritmos sobre os mesmos textos e padrões.
- Os tempos incluem o custo de pré-processamento executado por busca.
- A razão de médias é descritiva; não representa significância estatística.
- Comparações de Rabin-Karp não contabilizam atualizações de hashes.
- Casos sintéticos não representam a distribuição natural de textos FastAPI.
- Os dados da AV1 de recuperação e geração não são comparáveis diretamente.
- Os gráficos são derivados de summary.csv; os resultados originais permanecem intactos.
