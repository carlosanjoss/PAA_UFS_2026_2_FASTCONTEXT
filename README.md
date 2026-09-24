# FastContext

> Protótipo acadêmico de recuperação de contexto para aplicações RAG, desenvolvido na disciplina **Projeto e Análise de Algoritmos — UFS 2026.2**, com foco em corretude, eficiência, escalabilidade, qualidade de recuperação e impacto downstream em geração com LLM.

**Repositório:** https://github.com/carlosanjoss/PAA_UFS_2026_2_FASTCONTEXT

**Instituição:** Universidade Federal de Sergipe — UFS  
**Programa:** Programa de Pós-Graduação em Ciência da Computação — PROCC/UFS  
**Disciplina:** Projeto e Análise de Algoritmos  
**Semestre:** 2026.2  
**Corpus:** documentação oficial do FastAPI 0.141.0  
**Idioma do corpus:** inglês  
**Python:** 3.11

---

## 1. Equipe

| Integrante | Matrícula |
|---|---:|
| Carlos Eduardo de Melo Pereira dos Anjos | 202611010794 |
| Yann Gabriel Freire de Carvalho | 202621002286 |
| Mateus Almeida | 202611006441 |
| Gabriella de Jesus Santos | 202611006998 |
| Wilson Dias Martins Neto | 202611009022 |

---

## 2. Visão geral

O **FastContext** investiga como diferentes estratégias algorítmicas de recuperação de contexto afetam:

- corretude;
- tempo de execução;
- uso de memória;
- número de comparações;
- escalabilidade;
- Precision@k;
- Recall@k;
- MRR;
- Hit Rate;
- qualidade do contexto fornecido a um LLM;
- groundedness e qualidade final de respostas RAG.

O sistema recebe uma consulta textual e retorna os `k` chunks mais relevantes da documentação oficial do FastAPI.

O projeto compara estratégias lexicais clássicas implementadas pela equipe com uma baseline semântica baseada em embeddings e FAISS.

---

## 3. Pergunta de pesquisa

> **Como escolhas de representação, busca, ordenação e seleção Top-k afetam a eficiência, a corretude, a escalabilidade e a qualidade do contexto recuperado para uso em uma aplicação RAG?**

Uma segunda etapa downstream avalia:

> **Como a estratégia de recuperação afeta a latência, a fundamentação e a qualidade da resposta produzida por um mesmo modelo de linguagem em um sistema RAG?**

---

## 4. Corpus

O corpus foi obtido da documentação oficial do FastAPI, congelada na versão:

```text
FastAPI 0.141.0
```

Recorte utilizado:

```text
docs/en/docs
```

Configuração final:

```text
Documentos processados: 153
Chunks: 1305
Chunk máximo: 400 tokens
Overlap: 60 tokens
Idioma: inglês
Formato: Markdown
```

O arquivo `release-notes.md` foi excluído apenas do corpus experimental preparado, porque concentrava 1.554 dos 2.859 chunks da preparação inicial e introduzia forte assimetria na distribuição.

Fingerprint final do corpus:

```text
e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d
```

Esse corpus foi congelado antes dos experimentos finais.

---

## 5. Estratégias de recuperação

### 5.1 Linear

Pipeline:

```text
query
  -> TF-IDF
  -> cosine similarity em todos os chunks
  -> Merge Sort
  -> Top-k
```

Características:

- busca sequencial;
- todos os chunks são pontuados;
- ordenação manual com Merge Sort;
- desempate determinístico.

---

### 5.2 Indexed

Pipeline:

```text
query
  -> índice invertido
  -> busca binária manual no vocabulário
  -> candidatos
  -> TF-IDF
  -> Merge Sort
  -> Top-k
```

Objetivo:

- reduzir o espaço de candidatos;
- preservar o ranking lexical;
- demonstrar uso de indexação e busca binária.

---

### 5.3 Optimized

Pipeline:

```text
query
  -> índice invertido
  -> busca binária manual
  -> candidatos
  -> TF-IDF
  -> bounded min-heap Top-k
```

Diferença principal:

- evita ordenar todos os candidatos;
- mantém apenas os melhores `k`;
- reduz o custo de seleção final.

---

### 5.4 Semantic

Pipeline:

```text
query
  -> BAAI/bge-small-en-v1.5
  -> embedding
  -> FAISS IndexFlatIP
  -> Top-k semântico
```

Modelo:

```text
BAAI/bge-small-en-v1.5
```

Índice:

```text
FAISS IndexFlatIP
```

Quantidade de vetores:

```text
1305
```

---

## 6. Algoritmos implementados

O projeto inclui implementações e análises de:

- busca linear;
- busca binária;
- índice invertido;
- Merge Sort;
- Quick Sort;
- bounded min-heap para Top-k;
- divisão e conquista;
- TF-IDF;
- similaridade do cosseno.

A baseline semântica utiliza bibliotecas externas apenas como comparação complementar e não substitui os algoritmos clássicos exigidos pela atividade.

---

## 7. Corretude

O projeto contém justificativas formais de corretude e critérios determinísticos de ordenação.

Desempate final:

```text
score DESC
chunk_id ASC
```

Com isso, duas execuções sobre os mesmos dados e scores produzem a mesma ordenação.

Documentação relacionada:

```text
docs/CORRECTNESS.md
```

---

## 8. Ambiente

Ambiente principal utilizado nos experimentos finais:

```text
OS: Windows 11 Home 64 bits
Python: 3.11.9
CPU: Intel Core i5-11400H
Cores/threads: 6 / 12
RAM: aproximadamente 15.78 GiB
Execução: ambiente virtual local (.venv)
```

Também há suporte a Docker com Python 3.11.

---

## 9. Instalação local

### 9.1 Clonar o repositório

```bash
git clone https://github.com/carlosanjoss/PAA_UFS_2026_2_FASTCONTEXT.git
cd PAA_UFS_2026_2_FASTCONTEXT
```

### 9.2 Criar ambiente virtual

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 9.3 Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## 10. Configuração

Copie o arquivo de exemplo:

```powershell
Copy-Item .env.example .env
```

Linux/macOS:

```bash
cp .env.example .env
```

Os principais arquivos de configuração estão em:

```text
config/
├── corpus.yaml
├── experiments.yaml
├── retrieval.yaml
├── rag_experiments.yaml
└── human_evaluation.yaml
```

---

## 11. Preparação do corpus

Execute:

```bash
python scripts/download_fastapi_docs.py
```

Depois:

```bash
python scripts/prepare_corpus.py
```

E construa os índices necessários:

```bash
python scripts/build_index.py
```

O corpus experimental final deve resultar em:

```text
153 documentos
1305 chunks
fingerprint:
e08b47b55d7237f38bcaba48c79a468a40b92e4f3120e968ae9ece999f5e8c2d
```

Se o fingerprint divergir, o corpus não é idêntico ao utilizado nos experimentos finais.

---

## 12. CLI

O projeto expõe comandos via CLI.

### Verificar saúde

```bash
fastcontext health
```

### Listar algoritmos

```bash
fastcontext algorithms
```

### Recuperar contexto

Exemplo:

```bash
fastcontext retrieve "How can I create a FastAPI dependency?" --top-k 5
```

### Executar RAG

```bash
fastcontext ask "How can I create a FastAPI dependency?" --top-k 5
```

---

## 13. Interface web TypeScript

A interface é uma aplicação React + TypeScript servida pela API FastAPI. Para
gerar o frontend e iniciar a aplicação completa:

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn src.web_api:app --host 127.0.0.1 --port 8000
```

Abra `http://127.0.0.1:8000`. Durante desenvolvimento, execute `npm run dev`
em `frontend/`; o Vite encaminha chamadas `/api` para a porta `8000`.

A interface permite:

- fazer perguntas e ler a resposta antes da análise técnica;
- escolher algoritmo, `Top-k` e modo retrieval/RAG;
- inspecionar latência, comparações, memória, etapas e chunks recuperados;
- comparar tempo, trabalho computacional, precisão, recall, MRR e hit rate;
- explorar Merge Sort versus Quick Sort por tamanho e cenário de entrada;
- consultar apenas relatórios persistidos, sem reexecutar experimentos.

---

## 14. Ollama

O pipeline RAG foi testado com:

```text
qwen2.5:3b
```

Modelo alternativo da aplicação:

```text
qwen3:1.7b
```

Para uso normal:

```bash
ollama pull qwen2.5:3b
ollama pull qwen3:1.7b
```

Inicie:

```bash
ollama serve
```

No experimento downstream final foi utilizado **somente**:

```text
provider: ollama
model: qwen2.5:3b
temperature: 0
seed: 42
think: false
max_tokens: 256
fallback: desabilitado
```

---

## 15. Ground truth

O ground truth final contém:

```text
30 consultas
963 pares consulta-chunk avaliados
1926 julgamentos humanos primários
343 atribuições de relevância
296 chunks relevantes únicos
```

Concordância antes da adjudicação:

```text
Observed agreement: 77.57%
Cohen's kappa: aproximadamente 0.5320
Discordâncias: 216
```

Arquivo final:

```text
data/ground_truth/ground_truth.final.json
```

### Observação metodológica

O ground truth foi construído a partir de pooling Top-20.

Consequentemente:

- `Precision@k` para os sistemas avaliados em `k <= 10` é adequadamente sustentada pelo pool;
- `Recall@k` deve ser interpretado como recall relativo ao conjunto de relevantes julgados no pool;
- não representa relevância absoluta de todo o corpus.

---

## 16. Experimento principal de retrieval

Configuração:

```text
30 queries
4 algoritmos
4 tamanhos de corpus
5 repetições
```

Total:

```text
30 × 4 × 4 × 5 = 2400 execuções
```

Frações:

```text
25%
50%
75%
100%
```

Top-k avaliados:

```text
1
3
5
10
```

Os resultados finais já estão congelados.

**Não execute novamente com `--force` sobre os artefatos finais existentes.**

Arquivos:

```text
experiments/raw_results/results.csv
experiments/processed_results/summary.csv
```

---

## 17. Resultados de desempenho

Tempo médio de retrieval:

| Corpus | Linear | Indexed | Optimized | Semantic |
|---:|---:|---:|---:|---:|
| 326 chunks | 4.402 ms | 3.849 ms | **2.806 ms** | 20.019 ms |
| 652 chunks | 9.067 ms | 7.453 ms | **5.591 ms** | 20.675 ms |
| 978 chunks | 14.433 ms | 11.836 ms | **9.051 ms** | 22.180 ms |
| 1305 chunks | 18.855 ms | 15.299 ms | **11.264 ms** | 20.442 ms |

No corpus completo:

```text
Optimized vs Linear:
1.674× speedup
-40.26% de tempo

Optimized vs Indexed:
1.358× speedup
-26.37% de tempo
```

---

## 18. Comparações

No corpus completo:

```text
Linear:
10502.97 comparações médias

Indexed:
10807.13 comparações médias

Optimized:
1678.20 comparações médias
```

O Optimized realizou aproximadamente:

```text
84.02% menos comparações que Linear
```

---

## 19. Qualidade de retrieval

### Lexical

Os algoritmos:

```text
linear
indexed
optimized
```

produziram exatamente a mesma qualidade lexical.

No corpus completo:

| k | Precision | Recall | MRR | Hit Rate |
|---:|---:|---:|---:|---:|
| 1 | 0.8667 | 0.0946 | 0.8667 | 0.8667 |
| 3 | 0.7778 | 0.2571 | 0.9167 | 0.9667 |
| 5 | 0.6333 | 0.3197 | 0.9167 | 0.9667 |
| 10 | 0.5400 | 0.5118 | 0.9208 | 1.0000 |

### Semantic

| k | Precision | Recall | MRR | Hit Rate |
|---:|---:|---:|---:|---:|
| 1 | 0.9000 | 0.1168 | 0.9000 | 0.9000 |
| 3 | 0.7778 | 0.2617 | 0.9278 | 0.9667 |
| 5 | **0.7600** | **0.4191** | **0.9361** | **1.0000** |
| 10 | **0.5767** | **0.5759** | **0.9361** | **1.0000** |

---

## 20. Merge Sort versus Quick Sort

Também foi executado benchmark isolado de ordenação.

Configuração:

```text
200 execuções
5 repetições
2 algoritmos
4 cenários
5 tamanhos
```

Cenários:

```text
sorted
ties
random
reverse
```

Tamanhos:

```text
100
250
500
1000
1305
```

Em 100 comparações pareadas entre outputs:

```text
mismatches = 0
```

Ou seja, Merge Sort e Quick Sort produziram rankings equivalentes nos testes realizados.

---

## 21. Experimento downstream RAG

Uma segunda fase experimental avaliou a geração final.

Condições:

```text
No-RAG
Optimized RAG
Semantic RAG
```

Total:

```text
30 queries × 3 condições = 90 gerações
```

### Controle lexical

Antes das gerações finais foi comprovado que:

```text
Linear
Indexed
Optimized
```

produzem, para todas as 30 queries:

```text
Top-5 idêntico
Prompt RAG idêntico
```

Resultado do audit:

```text
queries_checked: 30
top_k: 5
all_top_k_identical: true
all_prompts_identical: true
```

Assim, `optimized` foi utilizado como representante lexical no downstream.

---

## 22. Resultados automáticos do RAG

### Retrieval dentro do downstream

| Condição | Precision@5 | Recall@5 |
|---|---:|---:|
| Optimized | 0.6333 | 0.3197 |
| Semantic | **0.7600** | **0.4191** |

### Latência

| Métrica | No-RAG | Optimized | Semantic |
|---|---:|---:|---:|
| Generation | 0.539 s | 8.257 s | 8.816 s |
| End-to-end | 0.539 s | 8.268 s | 8.839 s |
| Retrieval | N/A | 10.64 ms | 23.03 ms |
| Palavras/resposta | 18.10 | 36.63 | 41.17 |

A geração domina o tempo end-to-end.

### Citações

Das 60 respostas RAG:

```text
57/60
```

terminaram em estado válido pelo protocolo de citação.

Foram observados:

```text
51/60 respostas com pelo menos um retry de citação
```

Casos especiais:

```text
q11 optimized -> invalid_uncited_answer
q13 optimized -> valid_abstention
q16 semantic  -> invalid_uncited_answer
q18 optimized -> invalid_abstention
```

---

## 23. Avaliação humana

A avaliação humana foi realizada de forma cega.

Estrutura:

```text
90 respostas de qualidade geral × 2 avaliadores
60 respostas RAG de groundedness × 2 avaliadores
```

Total:

```text
300 julgamentos humanos
```

Cinco avaliadores participaram.

Carga por avaliador:

```text
36 avaliações de qualidade
24 avaliações de groundedness
60 julgamentos por pessoa
```

### Métricas

Qualidade:

```text
correctness: 0–2
completeness: 0–2
clarity: 0–2
hallucination_present: 0–1
```

Groundedness:

```text
groundedness: 0–2
```

---

## 24. Concordância humana

Resultado:

```text
correctness:
100% de concordância

completeness:
100% de concordância

clarity:
100% de concordância

hallucination_present:
100% de concordância

groundedness:
100% de concordância
```

Kappa:

```text
correctness: 1.0000
completeness: 1.0000
clarity: 1.0000
groundedness: 1.0000
```

Para `hallucination_present`, o kappa é indefinido porque todos os julgamentos ficaram na mesma categoria.

Auditoria de integridade:

```text
completed files: 10
unique completed SHA-256: 10
literal duplicate file hashes: 0
perfect agreement metrics: 5/5
```

---

## 25. Resultados humanos por condição

### Correctness

```text
No-RAG:    1.9333
Optimized: 1.8667
Semantic:  1.9333
```

Friedman:

```text
p = 0.9394
```

Sem diferença estatisticamente significativa.

### Completeness

```text
No-RAG:    1.8000
Optimized: 1.8667
Semantic:  1.9000
```

Friedman:

```text
p = 0.4677
```

Sem diferença estatisticamente significativa.

### Clarity

```text
No-RAG:    1.8000
Optimized: 1.9333
Semantic:  1.9000
```

Friedman:

```text
p = 0.1561
```

Na comparação No-RAG versus Optimized:

```text
p bruto = 0.0455
p Holm = 0.1365
```

Portanto, a diferença não permanece significativa após correção de múltiplas comparações.

### Hallucination

```text
No-RAG:    0/30
Optimized: 0/30
Semantic:  0/30
```

Não houve alucinação marcada pelos avaliadores.

### Groundedness

```text
Optimized: 1.8667
Semantic:  2.0000
```

Distribuição:

```text
Semantic:
30/30 com groundedness = 2

Optimized:
28/30 com groundedness = 2
2/30 com groundedness = 0
```

Wilcoxon:

```text
p = 0.1573
```

O Semantic apresentou vantagem descritiva, mas sem significância estatística na amostra de 30 queries.

---

## 26. Interpretação principal

Os resultados apontam o seguinte padrão:

```text
Semantic
  -> maior Precision@5
  -> maior Recall@5
  -> maior relevância das evidências citadas
  -> maior groundedness descritivo
```

Entretanto, a amostra downstream não forneceu evidência estatística suficiente para afirmar superioridade do Semantic em:

- correctness;
- completeness;
- clarity;
- groundedness.

O resultado mais consistente é o trade-off entre:

```text
Optimized:
menor custo de retrieval

Semantic:
melhor recuperação semântica
```

---

## 27. Testes

Execute todos os testes:

```bash
python -m pytest -q
```

Testes sem integração:

```bash
python -m pytest -m "not integration" -q
```

Lint:

```bash
python -m ruff check .
```

Type checking:

```bash
python -m mypy src
```

---

## 28. Docker

Build:

```bash
docker compose build
```

Subir aplicação:

```bash
docker compose up
```

Executar testes:

```bash
docker compose run --rm tests python -m pytest -q
```

Os nomes dos serviços podem variar de acordo com a versão atual do `docker-compose.yml`.

---

## 29. Estrutura do projeto

```text
.
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.tsx
│   └── package.json
│
├── config/
│   ├── corpus.yaml
│   ├── experiments.yaml
│   ├── retrieval.yaml
│   ├── rag_experiments.yaml
│   └── human_evaluation.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── chunks/
│   ├── ground_truth/
│   └── rag_evaluation/
│
├── docs/
│   ├── AI_USAGE.md
│   ├── architecture.md
│   ├── CORRECTNESS.md
│   ├── experiments.md
│   └── VIDEO.md
│
├── experiments/
│   ├── run_experiments.py
│   ├── run_rag_experiments.py
│   ├── raw_results/
│   ├── processed_results/
│   └── rag_results/
│
├── presentation/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── scripts/
│   ├── build_index.py
│   ├── prepare_human_evaluation.py
│   ├── analyze_human_evaluation.py
│   └── finalize_human_evaluation.py
│
├── src/
│   ├── algorithms/
│   ├── app/
│   ├── rag/
│   ├── retrieval/
│   ├── services/
│   └── utils/
│
├── tests/
│
├── .env.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 30. Principais figuras

Os experimentos geram figuras como:

```text
reports/figures/
├── algorithm_vs_precision_at_10.png
├── algorithm_vs_time.png
├── comparisons_vs_corpus_size.png
├── corpus_size_vs_memory.png
├── corpus_size_vs_time.png
├── k_vs_precision.png
├── k_vs_recall.png
├── lexical_vs_semantic_tradeoff.png
├── merge_vs_quick_comparisons.png
├── merge_vs_quick_time.png
└── quick_sort_input_sensitivity.png
```

---

## 31. Reprodutibilidade e congelamento dos resultados

### Retrieval principal

Hashes finais:

```text
experiments/raw_results/results.csv
677776E1E414C25D1AE623E8C46BBFC1B7908F9C63875254C9918A54ADAEFCAA

experiments/processed_results/summary.csv
686B2E286EB1C10586E4E16A4EDFA911C819080D85E8E390282FB92D1B0CE8A5

reports/tables/performance_by_algorithm.csv
B158322F3EBA53140CF6329E6D91C2B972FFD895E99B8076AC02D48DAAAD1BBD

reports/tables/quality_by_algorithm.csv
94F0C89DA8D95867AFA60F32D4C9C3149D84F1DDA0B6525BD0FB319578E1DEF6

reports/tables/sorting_comparison.csv
AE2234F20AF7E4C13064A0FCE8AED7A305414F33B3BB48BC627D9C0A51A0F28F
```

### RAG final

```text
experiments/rag_results/generations.csv
77160C5A3B2F10EA7E76C01B755B289CED4C73ABC325141791400A6EA9B092B4

experiments/rag_results/generations.jsonl
B9467285E9094203319583845D5D8CE8DA966A38DA6C6B03D4C9D666DA22DE7B

experiments/rag_results/checkpoint.json
A3C4E44099A836E005894B4D4C044D55EDE86833CCCB950882CFD22D026E8DCC

experiments/rag_results/lexical_equivalence_audit.json
4E81D94E28C9C93C1D16E6303165C25B599087F3CC676BC6C95A489AE4F9E15B
```

---

## 32. Importante: não sobrescrever benchmarks finais

Os resultados finais já foram obtidos e auditados.

Não utilize:

```bash
python experiments/run_experiments.py --force
```

sobre o benchmark principal existente.

Também não utilize novamente:

```bash
python experiments/run_rag_experiments.py --force
```

sobre as 90 gerações finais.

Esses comandos só devem ser utilizados para uma reprodução limpa em uma nova cópia/ambiente, nunca para sobrescrever os artefatos congelados desta entrega.

---

## 33. Relatório

O relatório técnico final apresenta:

- definição formal do problema;
- corpus e chunking;
- algoritmos;
- pseudocódigo;
- prova de corretude;
- modelo RAM;
- análise assintótica;
- recorrências;
- metodologia experimental;
- resultados de retrieval;
- benchmark de ordenação;
- avaliação de qualidade;
- experimento downstream RAG;
- avaliação humana;
- testes estatísticos;
- ameaças à validade;
- uso de IA generativa;
- contribuições individuais;
- reprodutibilidade.

---

## 34. Uso de IA generativa

Ferramentas de IA generativa foram utilizadas como apoio em atividades como:

- revisão de código;
- documentação;
- planejamento experimental;
- análise de resultados;
- elaboração de testes;
- discussão metodológica;
- revisão textual.

O uso de IA não substituiu:

- execução dos experimentos;
- validação dos algoritmos;
- análise de corretude;
- inspeção dos resultados;
- julgamento humano do ground truth;
- avaliação humana downstream.

A declaração detalhada está documentada no projeto.

---

## 35. Limitações

Principais limitações:

- corpus restrito à documentação do FastAPI;
- corpus somente em inglês;
- ground truth construído por pooling;
- recall relativo ao conjunto julgado;
- apenas 30 queries no experimento de qualidade;
- baseline semântica baseada em um único encoder;
- geração downstream com um único LLM;
- fixed seed reduz, mas não elimina, variabilidade de geração;
- `tracemalloc` não representa toda a memória nativa utilizada por FAISS/PyTorch;
- resultados de latência dependem do hardware e ambiente;
- concordância humana perfeita deve ser interpretada com cautela;
- ausência de significância estatística não demonstra equivalência entre condições.

---

## 36. Conclusão

O FastContext demonstra que otimizações algorítmicas clássicas podem reduzir de forma expressiva o custo do retrieval sem alterar a qualidade do ranking lexical.

A configuração `optimized`, baseada em índice invertido, busca binária e bounded min-heap, foi a estratégia lexical mais eficiente.

A recuperação semântica baseada em BGE-small + FAISS apresentou melhores métricas de qualidade, particularmente em Precision@5 e Recall@5, ao custo de maior latência de retrieval.

No pipeline completo RAG, entretanto, o custo do retrieval foi pequeno diante do tempo de geração do LLM.

A avaliação downstream mostrou vantagem descritiva do Semantic em groundedness, mas não encontrou diferenças estatisticamente significativas de correctness, completeness, clarity ou groundedness entre as condições avaliadas na amostra disponível.

---

## 37. Vídeo da atividade

A URL do vídeo final deve ser inserida aqui antes da entrega:

```text
[(https://www.youtube.com/watch?v=wviXO29yymo&t=3s)]
```

---

## 38. Versão final da entrega

Após o commit final na `main`, registre:

```text
Branch: main
Data: 20/09/2026
```

Para obter o hash:

```bash
git rev-parse HEAD
```

ou:

```bash
git log -1 --oneline
```

---

## 39. Referências principais

- FastAPI Documentation.
- FastAPI GitHub Repository.
- Cormen, Leiserson, Rivest e Stein. *Introduction to Algorithms*.
- Kleinberg e Tardos. *Algorithm Design*.
- Skiena. *The Algorithm Design Manual*.
- Sedgewick e Wayne. *Algorithms*.
- Lewis et al. *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
- FAISS documentation.
- Sentence Transformers documentation.
- BAAI/bge-small-en-v1.5 model documentation .

---

## 40. Status

```text
Corpus final:              CONCLUÍDO
Ground truth:              CONCLUÍDO
Retrieval benchmark:       CONCLUÍDO
Sorting benchmark:         CONCLUÍDO
RAG downstream:            CONCLUÍDO
Human evaluation:          CONCLUÍDO
Statistical analysis:      CONCLUÍDO
Frontend React/TypeScript: CONCLUÍDO
API FastAPI:               CONCLUÍDO
Tests:                     CONCLUÍDO
Final report:              CONCLUÍDO
Final commit:              CONCLUÍDO
Video URL:                 CONCLUÍDO
```

---

**FastContext — UFS / PROCC — 2026.2**
