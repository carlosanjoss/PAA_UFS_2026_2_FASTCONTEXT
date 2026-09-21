# FastContext Web Implementation

## Architecture

The presentation layer is a React 19 application written in TypeScript and
built with Vite. It communicates with a small FastAPI adapter that reuses the
existing `ApplicationContainer`, retriever registry, `FastContextService`, RAG
pipeline, health checks, and persisted experiment reports.

```text
React/TypeScript -> /api -> FastAPI -> existing FastContext services
                         -> reports/tables/*.csv
```

The browser never runs experiments or rewrites result files. The comparison
dashboard reads the three persisted report families:

- `performance_by_algorithm.csv` for time, comparisons, scored chunks, and
  memory;
- `quality_by_algorithm.csv` for precision, recall, MRR, and hit rate;
- `sorting_comparison.csv` for Merge Sort versus Quick Sort curves.

Missing measurements remain `null`/`N/D`; they are not converted to zero.

## User experience

The application has three focused views:

1. **Visão geral** summarizes the corpus, strategies, fastest measured method,
   and best MRR, with introductory efficiency and quality charts.
2. **Perguntar** puts the grounded answer first, followed by runtime cards, the
   real retrieval trace, and expandable evidence chunks.
3. **Comparações** separates efficiency, retrieval quality, and sorting
   behavior. Corpus fraction, top-k, and input scenario are explicit filters.

The visual system follows a modern analytical dashboard: pale neutral rail,
white cards, coral interaction color, restrained shadows, responsive grids,
and readable annotations explaining whether higher or lower values are better.

## Running locally

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn src.web_api:app --host 127.0.0.1 --port 8000
```

For frontend development, keep the API on port 8000 and run `npm run dev` in
`frontend/`. Vite proxies `/api` requests to the Python process.

## Deployment

The Docker image builds the TypeScript client in a Node stage, copies the
generated static assets into the Python runtime, and serves the SPA and API on
port 8000. This keeps the deployed frontend and backend on the same origin.
