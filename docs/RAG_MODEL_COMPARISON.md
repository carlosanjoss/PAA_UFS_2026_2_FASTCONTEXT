# Comparação auditada de modelos RAG

## Objetivo

Este relatório documenta a integração dos resultados existentes da comparação
entre `qwen2.5:3b` e `llama3.2:3b` ao FastContext. A aplicação não recalcula o
experimento: ela lê, valida e apresenta os artefatos congelados de
`experiments/rag_model_comparison_results/`.

Os resultados são descritivos para estes modelos quantizados, este corpus,
estes prompts e este ambiente. Não foi realizado teste de hipótese e nenhuma
diferença deve ser interpretada como superioridade estatisticamente
significativa ou universal.

## Desenho experimental

| Elemento | Configuração |
|---|---|
| Baseline | Qwen2.5 3B, `Q4_K_M` |
| Candidato | Llama 3.2 3B, `Q4_K_M` |
| Condições | `no_rag`, `optimized`, `semantic` |
| Perguntas | 30 por condição, 90 prompts congelados |
| Tentativas | 180, dois modelos por prompt |
| Geração | temperatura 0, seed 42, limite inicial de 256 tokens |
| Contexto RAG | Top-5 chunks |
| Avaliação | protocolo cego com cinco avaliadores |

O conjunto contém 179 respostas completas e um truncamento. O truncamento é a
resposta do Qwen na condição `optimized` para a questão 29; ela não foi
incluída nas médias de geração nem em comparações humanas que exigem o par
completo.

## Indicadores auditados

- 180 tentativas de geração;
- 179 respostas completas;
- 1 truncamento;
- 90 prompts congelados;
- 596 julgamentos humanos;
- 415 comparações válidas por critério, de 420 planejadas;
- 5 comparações ausentes devido ao único par incompleto.

Essas contagens são verificadas pelo carregador da API. Uma inconsistência ou
arquivo ausente torna somente este endpoint indisponível com resposta HTTP 503;
as páginas anteriores continuam operacionais.

## Resultados automáticos

| Modelo | Condição | Completas | Latência média (ms) | Citações aceitas |
|---|---|---:|---:|---:|
| Qwen2.5 3B | Sem RAG | 30/30 | 5.431,058 | não se aplica |
| Llama 3.2 3B | Sem RAG | 30/30 | 9.401,872 | não se aplica |
| Qwen2.5 3B | RAG otimizado | 29/30 | 19.166,623 | 23/29 |
| Llama 3.2 3B | RAG otimizado | 30/30 | 18.868,119 | 28/30 |
| Qwen2.5 3B | RAG semântico | 30/30 | 20.949,349 | 16/30 |
| Llama 3.2 3B | RAG semântico | 30/30 | 20.746,954 | 24/30 |

A conformidade de citações é uma validação estrutural: verifica se os IDs
citados pertencem ao contexto autorizado. Ela não mede, por si só, correção
factual, completude ou groundedness.

## Avaliação humana agregada

As notas de correção, completude, clareza e groundedness usam escala de 1 a 5.
Alucinação é um indicador binário cuja média menor é melhor. Groundedness não
foi avaliado em `no_rag`, pois não há contexto recuperado a fundamentar.

| Modelo | Condição | Correção | Completude | Clareza | Groundedness | Alucinação |
|---|---|---:|---:|---:|---:|---:|
| Qwen2.5 3B | Sem RAG | 3,250 | 1,967 | 3,083 | não se aplica | 18,33% |
| Llama 3.2 3B | Sem RAG | 3,333 | 3,100 | 3,600 | não se aplica | 53,33% |
| Qwen2.5 3B | RAG otimizado | 4,448 | 4,034 | 4,241 | 4,552 | 1,72% |
| Llama 3.2 3B | RAG otimizado | 4,200 | 3,467 | 4,100 | 4,633 | 13,33% |
| Qwen2.5 3B | RAG semântico | 4,533 | 4,200 | 4,383 | 4,600 | 11,67% |
| Llama 3.2 3B | RAG semântico | 4,367 | 3,667 | 4,033 | 4,517 | 11,67% |

As médias não substituem a distribuição das notas e não demonstram distância
intervalar entre categorias da escala ordinal.

## Concordância entre avaliadores

| Critério | Pares | Concordância exata | Diferença absoluta média |
|---|---:|---:|---:|
| Groundedness | 119 | 68,07% | 0,412 |
| Clareza | 179 | 56,42% | 0,458 |
| Completude | 179 | 52,51% | 0,525 |
| Correção | 179 | 54,75% | 0,575 |
| Alucinação | 179 | 92,18% | 0,078 |

Concordância exata é a proporção de pares com a mesma nota. Não foram
calculados kappa ou alpha, e os empates não passaram por adjudicação. A API
publica apenas estes agregados, sem revelar as avaliações individuais.

## Recuperação semântica

O pipeline semântico é:

```text
texto -> BAAI/bge-small-en-v1.5 -> embedding normalizado de 384 dimensões
      -> FAISS IndexFlatIP -> Top-5 chunks
```

Com embeddings normalizados, o produto interno usado por `IndexFlatIP`
corresponde à ordenação por similaridade do cosseno. Os cinco chunks foram
congelados no prompt para que os dois modelos recebessem exatamente o mesmo
contexto em cada execução.

Referências:

- Xiao, Shitao et al. *C-Pack: Packed Resources For General Chinese
  Embeddings*. SIGIR 2024. arXiv:2309.07597.
  <https://arxiv.org/abs/2309.07597>
- FAISS. `IndexFlatIP` API.
  <https://faiss.ai/cpp_api/struct/structfaiss_1_1IndexFlatIP.html>

## Integração web e privacidade

O endpoint `GET /api/rag/model-comparison` lê somente resumos auditados. O
dashboard apresenta filtros por modelo, condição e métrica, além de gráficos de
latência, conformidade de citações, qualidade humana, alucinação e concordância.

Os arquivos administrativos e identificadores cegos permanecem preservados nos
artefatos experimentais para rastreabilidade. A resposta pública não contém
`reviewer_id`, `blind_id`, observações, alocações, perguntas, respostas nem
chaves de execução.

## Fontes de dados

Os principais arquivos consumidos são:

- `generation_status_summary.json` e `generation_status.csv`;
- `frozen_prompts.manifest.json` e `frozen_prompts.jsonl`;
- `model_provenance.json`;
- `analysis/auditoria_180.json`;
- `analysis/resumo_por_modelo_condicao.csv`;
- `avaliacao_humana_cega/distribuicao_v1/analise_modelos_descegada/`;
- `avaliacao_humana_cega/distribuicao_v1/analise_concordancia_cega/`;
- `config/rag_model_comparison.yaml` e `config/retrieval.yaml`.

## Reproduzir a visualização

```powershell
.venv\Scripts\python.exe -m uvicorn src.web_api:app --host 127.0.0.1 --port 8000
cd frontend
npm run dev
```

Abra `http://127.0.0.1:5173/` e selecione **Modelos RAG**. Para servir o build
pelo FastAPI, execute `npm run build` antes de iniciar o servidor e acesse
`http://127.0.0.1:8000/`.

## Limitações

- o conjunto possui 30 perguntas por condição;
- existe um par incompleto por truncamento;
- as notas humanas são ordinais e agregadas descritivamente;
- não houve teste de significância, intervalo de confiança ou adjudicação;
- conformidade de citações não equivale a verdade factual;
- os resultados não devem ser extrapolados para outros modelos, quantizações,
  prompts, corpora ou ambientes.
