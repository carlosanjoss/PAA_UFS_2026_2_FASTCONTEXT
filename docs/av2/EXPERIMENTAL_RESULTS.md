
# FastContext AV2 — Resultados Experimentais

## 1. Objetivo

Avaliar o desempenho de quatro algoritmos de casamento
exato de cadeias:

- Naive
- Knuth-Morris-Pratt (KMP)
- Boyer-Moore com heurística bad-character
- Rabin-Karp com rolling hash

A análise investiga corretude, tempo de execução,
comparações de caracteres e alocações de memória.

Os resultados não representam uma avaliação de
recuperação semântica nem de geração por LLM.

## 2. Corpus

Corpus congelado da AV1:

- Fonte: documentação oficial do FastAPI
- Versão: 0.141.0
- Idioma: inglês
- Documentos: 153
- Chunks: 1.305
- Caracteres na escala completa: 766.904

Fingerprint SHA-256 de referência:

e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d

O corpus original foi preservado.

## 3. Configuração experimental

Foram utilizados:

- 4 algoritmos
- 8 padrões
- 3 escalas de entrada: 25%, 50% e 100%
- 5 repetições medidas por combinação
- 1 execução de aquecimento não contabilizada

Total:

24 cenários × 4 algoritmos × 5 repetições
= 480 medições temporais.

Os dados foram preparados pelo protocolo
determinístico de Yann Gabriel.

Os padrões reais foram pesquisados em cada chunk
separadamente, sem permitir ocorrências que
atravessem fronteiras de chunks.

Os dois padrões sintéticos foram executados sobre
textos artificiais contínuos, construídos por
repetição de sementes.

## 4. Metodologia de medição

A avaliação temporal utilizou perf_counter_ns.

O carregamento inicial do corpus não foi incluído
no tempo medido.

As medições consideraram:

- Tempo interno de pré-processamento
- Tempo interno de busca
- Tempo total da execução dos matchers
- Comparações diretas de caracteres
- Número de ocorrências

O tempo total de execução é medido externamente
à chamada de busca e inclui a sobrecarga de
invocação e construção dos objetos de resultado.

A validação de corretude foi executada fora da
região de medição temporal.

A ordem de execução dos algoritmos foi alternada
entre repetições.

Para cada cenário, os resultados foram comparados
com as posições produzidas pelo Naive.

Além disso, o número de ocorrências foi conferido
com os valores esperados dos cenários preparados.

## 5. Resultados de tempo — escala de 100%

Tempos médios em milissegundos, com cinco repetições.

| Padrão | Naive | KMP | Boyer-Moore | Rabin-Karp |
|---|---:|---:|---:|---:|
| Ausente | 151.587 | 68.575 | 9.496 | 330.259 |
| dependency | 164.115 | 65.313 | 18.736 | 296.667 |
| background tasks | 170.309 | 64.633 | 18.212 | 341.085 |
| OAuth2 | 148.276 | 64.410 | 27.780 | 322.026 |
| WebSocket | 181.546 | 75.376 | 24.541 | 309.020 |
| a — curto | 189.457 | 77.127 | 151.907 | 244.022 |
| ABA — sintético | 220.668 | 110.318 | 205.168 | 413.689 |
| AAAA — sintético | 363.001 | 120.494 | 363.293 | 684.100 |

Menor tempo médio por padrão:

- Boyer-Moore: 5 padrões
- KMP: 3 padrões
- Naive: 0 padrões
- Rabin-Karp: 0 padrões

Essas contagens são descritivas e se restringem
às implementações, ao ambiente e às entradas avaliadas.

## 6. Escalabilidade — padrão OAuth2

| Corpus | Boyer-Moore (ms) | KMP (ms) | Naive (ms) | Rabin-Karp (ms) |
|---|---:|---:|---:|---:|
| 25% | 8.193 | 19.909 | 60.916 | 99.172 |
| 50% | 14.669 | 35.918 | 90.781 | 173.710 |
| 100% | 27.780 | 64.410 | 148.276 | 322.026 |

Foram encontradas:

- 24 ocorrências na escala de 25%
- 37 ocorrências na escala de 50%
- 103 ocorrências na escala de 100%

Os quatro algoritmos retornaram as mesmas
ocorrências para cada escala.

Os resultados apresentam aumento do tempo médio
à medida que cresce o volume de caracteres.

Três escalas são suficientes para caracterizar
a tendência observada, mas não para demonstrar
empiricamente os limites assintóticos.

## 7. Speedup em relação ao Naive

Define-se:

Speedup(A) = tempo_médio_Naive / tempo_médio_A

Para OAuth2 em 100% do corpus:

| Algoritmo | Tempo médio (ms) | Speedup |
|---|---:|---:|
| Naive | 148.276 | 1.00x |
| KMP | 64.410 | 2.30x |
| Boyer-Moore | 27.780 | 5.34x |
| Rabin-Karp | 322.026 | 0.46x |

O Boyer-Moore apresentou o melhor resultado
temporal nesse cenário.

O valor de 0.46x do Rabin-Karp indica que
sua execução foi mais lenta que a baseline.

As razões de médias não são testes de
significância estatística.

## 8. Análise por características da entrada

### 8.1. Padrões reais e pouco frequentes

O Boyer-Moore apresentou vantagem nos padrões
reais mais longos ou menos frequentes, além
do padrão ausente.

Sua heurística bad-character permite deslocar
o padrão após determinadas divergências,
evitando verificar todos os alinhamentos.

A implementação avaliada utiliza apenas essa
heurística, sem good-suffix.

### 8.2. Padrões repetitivos

O KMP apresentou o menor tempo para o padrão
sintético AAAA.

Tempos médios:

- KMP: 120.494 ms
- Naive: 363.001 ms
- Boyer-Moore: 363.293 ms
- Rabin-Karp: 684.100 ms

Esse comportamento é compatível com o uso da
tabela LPS para reaproveitar informações de
prefixos e sufixos.

A entrada é artificial e adversarial.
Ela não representa a distribuição natural
da documentação FastAPI.

### 8.3. Ocorrências sobrepostas

No cenário sintético ABA, o KMP também
apresentou o menor tempo médio:

- KMP: 110.318 ms
- Boyer-Moore: 205.168 ms
- Naive: 220.668 ms
- Rabin-Karp: 413.689 ms

Os algoritmos preservaram as ocorrências
sobrepostas previstas pelo contrato.

### 8.4. Rabin-Karp e contagem de operações

O Rabin-Karp apresentou baixo número de
comparações diretas de caracteres em vários
cenários.

Entretanto, seu contador não inclui o trabalho
de atualização dos hashes das janelas.

Portanto, essas contagens não representam todas
as operações computacionais realizadas.

O maior tempo temporal da implementação não
contradiz sua complexidade assintótica esperada.

## 9. Análise de memória

A memória foi analisada com tracemalloc,
separadamente do benchmark temporal.

Foram realizadas 96 medições:

24 cenários × 4 algoritmos.

Cada combinação teve uma medição de pico.

### 9.1. OAuth2 em 100% do corpus

| Algoritmo | Pico de alocações (MiB) |
|---|---:|
| Naive | 0.216934 |
| KMP | 0.252003 |
| Boyer-Moore | 0.221176 |
| Rabin-Karp | 0.220047 |

O KMP apresentou o maior pico observado
nesse cenário.

Entretanto, o resultado inclui alocações
realizadas pelo executor e objetos de saída.

### 9.2. Padrões com muitas ocorrências

Nas entradas sintéticas, os picos observados
foram aproximadamente:

- ABA: 15.33 MiB
- AAAA: 35.61 MiB

Os quatro algoritmos apresentaram valores
muito próximos.

Isso é consistente com o efeito do grande
número de posições armazenadas nos resultados.

O padrão curto e frequente "a" também elevou
o pico de alocações no corpus real, atingindo
aproximadamente 1.31–1.35 MiB na escala completa.

### 9.3. Limites da medição

Os resultados de tracemalloc:

- Não representam a RAM residente total
  do processo.
- Não incluem necessariamente alocações
  efetuadas por bibliotecas nativas.
- Não isolam a memória auxiliar dos algoritmos.
- Incluem os resultados mantidos pelo executor.
- Não incluem o custo inicial de carregar
  os textos antes da medição.

A métrica é apresentada como pico de
alocações Python durante a execução.

Os valores estão em MiB (1024 × 1024 bytes).

Como foi realizada uma medição de memória
por combinação, não são apresentados desvios-
padrão ou intervalos de confiança dessa métrica.

## 10. Complexidade teórica

Considere n = comprimento do texto e
m = comprimento do padrão.

As análises consideram espaço auxiliar,
desconsiderando a saída.

| Algoritmo | Tempo de pior caso | Espaço auxiliar |
|---|---|---|
| Naive | O(nm) | O(1) |
| KMP | O(n + m) | O(m) |
| Boyer-Moore bad-character | O(nm) | O(k), k = caracteres distintos do padrão |
| Rabin-Karp | O(nm) | O(1) |

Para Rabin-Karp, o custo esperado de busca,
sob hipóteses usuais de poucas colisões, é
O(n + m).

O custo de armazenar as posições encontradas
deve ser adicionado quando considerado
o espaço total de saída.

As propriedades assintóticas decorrem da
análise dos algoritmos, e não de um ajuste
estatístico realizado sobre as medições.

## 11. Ameaças à validade

As principais limitações são:

1. Os algoritmos foram avaliados em Python,
   cujo custo de interpretação influencia
   a latência observada.

2. O corpus é restrito à documentação
   FastAPI em inglês.

3. As entradas sintéticas apresentam
   distribuições artificiais de caracteres.

4. A máquina, a carga do sistema operacional
   e a frequência do processador podem
   introduzir variações de tempo.

5. Foram utilizadas cinco repetições por
   combinação, com resultados descritivos.

6. Não foram realizados testes de
   significância estatística nesta análise.

7. O Naive foi usado como oráculo diferencial;
   a correção dessa referência também
   depende dos testes e argumentos formais.

8. O campo comparisons não contabiliza
   a mesma classe completa de operações
   em todas as implementações.

9. Os resultados dos chunks do corpus real
   não devem ser interpretados como
   equivalentes aos resultados sintéticos.

10. As medições de memória incluem
    os objetos de saída.

11. Os custos de recuperação semântica
    e geração RAG não fazem parte das
    medições temporais da AV2.

## 12. Reprodutibilidade

Preparação dos dados:

python -m scripts.av2.prepare_inputs

Testes da AV2:

python -m pytest tests -k av2 -q

Execução temporal:

python -m scripts.av2.run_benchmarks --repetitions 5 --warmup 1 --output-dir experiments/av2/final

Gráficos de tempo:

python -m scripts.av2.generate_plots

Medição complementar de memória:

python -m scripts.av2.run_benchmarks --repetitions 1 --warmup 0 --measure-memory --output-dir experiments/av2/memory_probe

Gráficos de memória:

python -m scripts.av2.analyze_memory

## 13. Artefatos

Resultados temporais:

- experiments/av2/final/raw_results.csv
- experiments/av2/final/summary.csv

Resultados de memória:

- experiments/av2/memory_probe/memory_results.csv

Tabelas:

- reports/tables/av2/speedups.csv
- reports/tables/av2/analysis.md
- reports/tables/av2/memory_analysis.csv
- reports/tables/av2/memory_analysis.md

Figuras:

- reports/figures/av2/scaling_time.png
- reports/figures/av2/character_comparisons.png
- reports/figures/av2/full_scale_patterns.png
- reports/figures/av2/memory_real_patterns.png
- reports/figures/av2/memory_synthetic_patterns.png

## 14. Conclusão

Nos cenários avaliados, o desempenho dos
algoritmos variou conforme as características
do padrão e do texto.

O Boyer-Moore com heurística bad-character
apresentou os melhores resultados em cinco
dos oito padrões de escala completa.

O KMP apresentou os melhores resultados
nos três cenários restantes, incluindo
padrões frequentes e entradas sintéticas
repetitivas ou sobrepostas.

O Naive forneceu uma referência simples
de corretude e tempo.

O Rabin-Karp apresentou maior custo temporal
nessas implementações, apesar das poucas
comparações diretas de caracteres registradas
em diversos casos.

Os resultados demonstram que a escolha do
algoritmo deve considerar as características
da entrada, os custos de pré-processamento,
a implementação utilizada e as operações
efetivamente medidas.

As conclusões são descritivas e limitadas
ao protocolo experimental executado.
