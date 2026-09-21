"""HTTP API and production static host for the FastContext web interface."""

from __future__ import annotations

import csv
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.app.bootstrap import create_application
from src.app.health import check_application_health
from src.app.models import ApplicationContainer
from src.app.presentation import build_retrieval_trace
from src.rag.providers.base import GenerationConfig, LLMProviderError
from src.retrieval.models import RetrievalResult
from src.retrieval.registry import RetrieverRegistryError
from src.services.factory import create_fastcontext_service
from src.services.fastcontext import FastContextResult, RAGNotConfiguredError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIRECTORY = REPOSITORY_ROOT / "reports" / "tables"
FRONTEND_DIST = REPOSITORY_ROOT / "frontend" / "dist"

Algorithm = Literal["linear", "indexed", "optimized", "semantic"]


class QueryRequest(BaseModel):
    """Validated request submitted by the TypeScript client."""

    query: str = Field(min_length=1, max_length=2_000)
    algorithm: Algorithm = "indexed"
    top_k: int = Field(default=5, ge=1, le=10)
    use_rag: bool = True


app = FastAPI(
    title="FastContext API",
    description="Retrieval, RAG, health, and persisted experiment metrics.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@lru_cache(maxsize=1)
def get_application() -> ApplicationContainer:
    """Build shared application resources once per API process."""

    return create_application()


@lru_cache(maxsize=4)
def get_service(algorithm: Algorithm) -> Any:
    """Create and cache one real retriever service per strategy."""

    application = get_application()
    return create_fastcontext_service(
        algorithm=algorithm,
        registry=application.registry,
        rag_pipeline=application.rag_pipeline,
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    """Expose runtime availability and prepared corpus size."""

    try:
        application = get_application()
        report = check_application_health(application)
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="O runtime FastContext não pôde ser inicializado.",
        ) from exc

    return {
        "status": report.status,
        "corpus_size": application.corpus_size,
        "algorithms": list(application.registry.available_names()),
        "components": [
            {
                "name": component.name,
                "status": component.status,
                "message": component.message,
            }
            for component in report.components
        ],
    }


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    """Return persisted experimental reports without rerunning experiments."""

    reports = {
        "performance": REPORTS_DIRECTORY / "performance_by_algorithm.csv",
        "quality": REPORTS_DIRECTORY / "quality_by_algorithm.csv",
        "sorting": REPORTS_DIRECTORY / "sorting_comparison.csv",
    }
    missing = [path.name for path in reports.values() if not path.is_file()]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="Relatórios experimentais ausentes: " + ", ".join(missing),
        )

    return {name: _read_csv_records(path) for name, path in reports.items()} | {
        "generated_from": [path.name for path in reports.values()]
    }


@app.post("/api/query")
def query(request: QueryRequest) -> dict[str, Any]:
    """Run a real retrieval or grounded RAG request."""

    normalized_query = request.query.strip()
    if not normalized_query:
        raise HTTPException(status_code=422, detail="A pergunta não pode estar vazia.")

    try:
        service = get_service(request.algorithm)
        application = get_application()
        if request.use_rag and application.provider.is_available():
            result = service.ask(
                query=normalized_query,
                top_k=request.top_k,
                generation_config=GenerationConfig(
                    temperature=0.0,
                    max_tokens=256,
                    think=False,
                ),
            )
            return _serialize_rag_result(result)

        retrieval = service.retrieve(
            query=normalized_query,
            top_k=request.top_k,
        )
        fallback_reason = None
        if request.use_rag:
            fallback_reason = "O provedor de LLM configurado está indisponível."
        return _serialize_retrieval_result(
            retrieval,
            fallback_reason=fallback_reason,
        )
    except (RetrieverRegistryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (LLMProviderError, RAGNotConfiguredError) as exc:
        try:
            retrieval = service.retrieve(
                query=normalized_query,
                top_k=request.top_k,
            )
        except Exception as retrieval_exc:
            raise HTTPException(
                status_code=503,
                detail="A geração e a recuperação ficaram indisponíveis.",
            ) from retrieval_exc
        return _serialize_retrieval_result(
            retrieval,
            fallback_reason=str(exc) or type(exc).__name__,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="A consulta não pôde ser concluída.",
        ) from exc


def _serialize_rag_result(result: FastContextResult) -> dict[str, Any]:
    retrieval = result.retrieval
    response = _serialize_retrieval_result(retrieval)
    response.update(
        {
            "mode": "rag",
            "answer": result.rag.answer,
            "provider": result.rag.provider,
            "model": result.rag.model,
            "citations": list(result.rag.citations),
            "citation_valid": result.rag.citation_valid,
        }
    )
    response["metrics"]["generation_time_ms"] = _ns_to_ms(result.generation_time_ns)
    response["metrics"]["total_time_ms"] = _ns_to_ms(result.end_to_end_time_ns)
    return response


def _serialize_retrieval_result(
    result: RetrievalResult,
    *,
    fallback_reason: str | None = None,
) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "query": result.query,
        "algorithm": result.algorithm,
        "top_k": result.top_k,
        "mode": "retrieval",
        "answer": None,
        "provider": None,
        "model": None,
        "citations": [],
        "citation_valid": None,
        "fallback_reason": fallback_reason,
        "metrics": {
            "retrieval_time_ms": _ns_to_ms(metrics.retrieval_time_ns),
            "sorting_time_ms": _optional_ns_to_ms(metrics.sorting_time_ns),
            "index_build_time_ms": _optional_ns_to_ms(metrics.index_build_time_ns),
            "generation_time_ms": None,
            "total_time_ms": _ns_to_ms(metrics.retrieval_time_ns),
            "comparisons": metrics.comparisons,
            "chunks_scored": metrics.chunks_scored,
            "candidates_found": metrics.candidates_found,
            "peak_memory_mb": metrics.peak_memory_mb,
        },
        "trace": [
            {"title": step.title, "detail": step.detail}
            for step in build_retrieval_trace(result)
        ],
        "chunks": [
            {
                "rank": chunk.rank,
                "chunk_id": chunk.chunk_id,
                "score": chunk.score,
                "section_title": chunk.section_title,
                "source_path": chunk.source_path,
                "token_count": chunk.token_count,
                "content": chunk.content,
            }
            for chunk in result.chunks
        ],
    }


def _read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            {key: _coerce_csv_value(key, value) for key, value in row.items()}
            for row in reader
        ]


def _coerce_csv_value(key: str, value: str | None) -> str | int | float | None:
    if value is None or not value.strip():
        return None
    if key in {"algorithm", "scenario"}:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if not math.isfinite(number):
        return None
    if key in {"corpus_chunks", "query_count", "k", "item_count", "sample_count"}:
        return int(number)
    return number


def _ns_to_ms(value: int) -> float:
    return value / 1_000_000


def _optional_ns_to_ms(value: int | None) -> float | None:
    return None if value is None else _ns_to_ms(value)


if FRONTEND_DIST.is_dir():
    assets_directory = FRONTEND_DIST / "assets"
    if assets_directory.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=assets_directory),
            name="frontend-assets",
        )

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        """Serve built frontend assets and the SPA entry point."""

        requested = (FRONTEND_DIST / path).resolve()
        if requested.is_relative_to(FRONTEND_DIST) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")


def run() -> None:
    """Run the production web server from the console script."""

    uvicorn.run(
        "src.web_api:app",
        host="127.0.0.1",
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    run()
