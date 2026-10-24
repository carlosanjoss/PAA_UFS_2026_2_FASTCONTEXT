# AV2 — Integração web

## Objetivo

A integração apresenta os quatro algoritmos de casamento exato da AV2 no mesmo
produto da AV1. Ela reutiliza `StringMatchingService`,
`create_default_registry()` e os artefatos finais já persistidos. Nenhuma
requisição HTTP modifica o corpus, os CSVs ou o protocolo experimental.

## Arquitetura

```text
React/TypeScript
  ├─ Busca textual ──────> /api/av2/string-matching/*
  └─ Experimentos AV2 ───> /api/av2/experiments
                                 │
FastAPI                          ├─ experiments/av2/final/*.csv
  ├─ router de string matching   ├─ experiments/av2/memory_probe/*.csv
  ├─ loader experimental         ├─ reports/tables/av2/*
  └─ APIs preservadas da AV1     └─ data/av2/*.json
```

O corpus é carregado uma vez por processo pelo dependency provider com cache.
Os artefatos experimentais também são lidos e validados uma vez por processo.
Ausência ou inconsistência de arquivo produz resposta `503` somente na API da
AV2; as páginas e rotas da AV1 permanecem disponíveis.

## Endpoints

### `GET /api/av2/string-matching/algorithms`

Lista os nomes canônicos e o tamanho do corpus carregado.

### `POST /api/av2/string-matching/search`

Entrada:

```json
{
  "pattern": "OAuth2",
  "algorithm": "boyer-moore",
  "max_results": 10
}
```

`max_results` limita somente os chunks serializados, entre 1 e 50. As contagens
globais são calculadas sobre todo o corpus. A resposta inclui conteúdo apenas
dos chunks exibidos, posições em offsets Python, contagens e tempos em
nanossegundos. A interface converte o conteúdo em pontos de código Unicode antes
de realçar as posições, sem HTML injetado.

### `GET /api/av2/experiments`

Retorna metadados derivados, 8 padrões, 24 cenários, 96 resumos temporais, 96
registros de memória, 96 speedups e o ambiente sanitizado. As 480 linhas brutas
são contadas para validação da matriz, mas não são enviadas ao navegador.

## Frontend

- `StringMatchingPage.tsx`: busca real, comparação exploratória e evidências.
- `AV2ExperimentsPage.tsx`: sete gráficos, filtros, tabela, CSV e ambiente.
- Os tipos AV1 e AV2 permanecem separados em `types.ts`.
- As páginas usam lazy loading e o mesmo design system da AV1.
- Comparações interativas são rotuladas como exploratórias; resultados oficiais
  vêm somente dos artefatos congelados.

## Reprodução

Backend:

```bash
python -m pytest tests -k av2 -q
python -m pytest tests -q
python -m ruff check src scripts/av2 tests
python -m mypy src
uvicorn src.web_api:app --host 127.0.0.1 --port 8000
```

Frontend:

```bash
cd frontend
npm ci
npm run typecheck
npm run build
npm run dev
```

## Limitações

- Cinco repetições temporais por combinação, sem inferência estatística.
- Memória medida separadamente por `tracemalloc`, não como RAM total.
- Objetos de saída fazem parte do pico rastreado.
- Padrões sintéticos são microbenchmarks, não textos naturais.
- Comparações de Rabin-Karp não incluem operações de rolling hash.
- Resultados dependem das entradas e implementações avaliadas.
