# AV2 × RAG — comparação de recuperação

Protocolo: 30 perguntas, Top-5, cinco repetições medidas e um aquecimento por condição.

| Condição | P@5 conhecida | P@5 máxima | Recall@5 | MRR@5 | Hit Rate@5 | Tempo (ms) | Não julgados |
|---|---:|---:|---:|---:|---:|---:|---:|
| Semântica | 0.7600 | 0.7600 | 0.4191 | 0.9361 | 1.0000 | 24.45 | 0 |
| Literal | 0.5067 | 0.7400 | 0.2636 | 0.6944 | 0.7667 | 38.63 | 35 |
| Híbrida | 0.7133 | 0.7333 | 0.3801 | 0.8678 | 0.9333 | 62.69 | 3 |

## Conclusões descritivas

- Precisão conhecida da semântica: 0.7600.
- Melhor cenário possível da literal: 0.7400.
- Melhor cenário possível da híbrida: 0.7333.
- O híbrido foi 2.56 vezes mais lento que a semântica.

## Limitações

- As métricas usam relevância conhecida no pool original da AV1.
- Resultados ainda não julgados não são negativos confirmados.
- O limite superior considera todos os não julgados relevantes.
- As cinco repetições não equivalem a perguntas independentes.
- Não foi realizado teste de significância estatística.
- Não se mediu qualidade de respostas geradas pelo LLM nesta etapa.
- O desempenho não prova que toda fusão híbrida seja inferior.
