# FastContext — Resultados finais da avaliação humana

## Integridade

- Arquivos preenchidos: 10.
- SHA-256 únicos entre os arquivos preenchidos: 10.
- Métricas com concordância exata de 100%: 5/5.

## Qualidade geral

| Métrica | Condição | Média | Mediana | Taxa no escore máximo |
|---|---|---:|---:|---:|
| correctness | no_rag | 1.9333 | 2.00 | 93.33% |
| correctness | optimized | 1.8667 | 2.00 | 93.33% |
| correctness | semantic | 1.9333 | 2.00 | 93.33% |
| completeness | no_rag | 1.8000 | 2.00 | 80.00% |
| completeness | optimized | 1.8667 | 2.00 | 93.33% |
| completeness | semantic | 1.9000 | 2.00 | 90.00% |
| clarity | no_rag | 1.8000 | 2.00 | 80.00% |
| clarity | optimized | 1.9333 | 2.00 | 93.33% |
| clarity | semantic | 1.9000 | 2.00 | 90.00% |
| hallucination_present | no_rag | 0.0000 | 0.00 | 0.00% |
| hallucination_present | optimized | 0.0000 | 0.00 | 0.00% |
| hallucination_present | semantic | 0.0000 | 0.00 | 0.00% |

## Groundedness

| Condição | Média | Mediana | Taxa no escore 2 |
|---|---:|---:|---:|
| optimized | 1.8667 | 2.00 | 93.33% |
| semantic | 2.0000 | 2.00 | 100.00% |

## Testes pareados

| Métrica | Teste | Comparação | p | p Holm | Efeito |
|---|---|---|---:|---:|---:|
| correctness | friedman | all_conditions | 0.939413 |  | 0.0021 |
| correctness | wilcoxon_signed_rank | no_rag_vs_optimized | 0.414216 | 1 | 0.5000 |
| correctness | wilcoxon_signed_rank | no_rag_vs_semantic | 1 | 1 | 0.0000 |
| correctness | wilcoxon_signed_rank | optimized_vs_semantic | 0.457614 | 1 | -0.4000 |
| completeness | friedman | all_conditions | 0.467666 |  | 0.0253 |
| completeness | wilcoxon_signed_rank | no_rag_vs_optimized | 0.414216 | 0.828432 | -0.3333 |
| completeness | wilcoxon_signed_rank | no_rag_vs_semantic | 0.256839 | 0.770518 | -0.4286 |
| completeness | wilcoxon_signed_rank | optimized_vs_semantic | 0.705457 | 0.828432 | -0.2000 |
| clarity | friedman | all_conditions | 0.156118 |  | 0.0619 |
| clarity | wilcoxon_signed_rank | no_rag_vs_optimized | 0.0455003 | 0.136501 | -1.0000 |
| clarity | wilcoxon_signed_rank | no_rag_vs_semantic | 0.256839 | 0.513679 | -0.4286 |
| clarity | wilcoxon_signed_rank | optimized_vs_semantic | 0.563703 | 0.563703 | 0.3333 |
| hallucination_present | cochran_q | all_conditions | 1 |  |  |
| hallucination_present | mcnemar_exact | no_rag_vs_optimized | 1 | 1 | 0.0000 |
| hallucination_present | mcnemar_exact | no_rag_vs_semantic | 1 | 1 | 0.0000 |
| hallucination_present | mcnemar_exact | optimized_vs_semantic | 1 | 1 | 0.0000 |
| groundedness | wilcoxon_signed_rank | optimized_vs_semantic | 0.157299 |  | -1.0000 |

Os testes são pareados por `query_id`. As comparações pós-hoc usam correção de Holm.
