# FastContext

Protótipo de recuperação de contexto sobre a documentação oficial do FastAPI desenvolvido para a disciplina Projeto e Análise de Algoritmos da Universidade Federal de Sergipe.

## Objetivo

Investigar como diferentes estratégias de busca, ordenação, indexação e recuperação afetam a eficiência, a escalabilidade e a qualidade do contexto recuperado para aplicações de IA generativa.

## Corpus

Documentação oficial do FastAPI em inglês.

Versão utilizada:

FastAPI 0.141.0

Fonte:

https://github.com/fastapi/fastapi

## Estratégias previstas

- Busca linear
- Busca binária
- Índice invertido
- Merge Sort
- Quick Sort
- Top-k com heap
- TF-IDF
- Busca semântica com Sentence Transformers
- FAISS
- RAG com LLM

## Tecnologias

- Python 3.11
- Docker
- pytest
- scikit-learn
- Streamlit
- Sentence Transformers
- FAISS
- Ollama

## Interface Streamlit

Com o ambiente virtual ativo e o corpus preparado, inicie a interface com:

```bash
streamlit run app/streamlit_app.py
```

A interface usa o corpus real em `data/chunks/chunks.jsonl`, os quatro
retrievers registrados (`linear`, `indexed`, `optimized` e `semantic`) e a
camada `FastContextService`. O modo **Retrieval only** não chama um LLM e
continua disponível se Ollama estiver desligado. O modo **Retrieval + LLM** usa
o pipeline RAG configurado e preserva os resultados de retrieval quando a
geração falhar.

O provider, URL e modelos são lidos de `config/retrieval.yaml` com sobreposição
das variáveis em `.env`. A configuração padrão espera Ollama em
`http://localhost:11434`, o modelo `qwen2.5:3b` e, se configurado, o modelo de
fallback. Para usar RAG, deixe um desses modelos disponível no Ollama. O índice
semântico persistido em `data/processed/semantic/` é reutilizado somente quando
for compatível com o corpus e a configuração de embeddings.

## Ambiente Docker e testes

### Aplicação

```bash
docker compose up -d fastcontext
```

Acesse <http://localhost:8501> ou execute scripts manualmente:

```bash
docker compose exec fastcontext python scripts/prepare_corpus.py
```

O serviço `fastcontext` inicia a mesma interface Streamlit na porta 8501.

### Corpus

O download e a preparação do corpus são etapas independentes:

```bash
python -m scripts.download_corpus
python -m scripts.prepare_corpus
```

O primeiro comando baixa e valida a documentação oficial do FastAPI. O segundo
normaliza os arquivos Markdown e gera os chunks em `data/chunks/chunks.jsonl`.

### Testes

```bash
docker compose run --rm tests
```

O serviço instala as dependências e executa `python -m pytest -v` no container,
aguarda o término e remove o container.

Parar a aplicação:

```bash
docker compose down
```

## Estrutura

```text
config/        Configurações do projeto
data/          Corpus e dados processados
src/           Código-fonte
tests/         Testes automatizados
experiments/   Experimentos e resultados
reports/       Relatório e figuras
presentation/  Apresentação
docs/          Documentação técnica
