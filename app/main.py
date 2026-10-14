"""FastAPI application for ExperimentIQ.

Provides the HTTP API for the RAG pipeline.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import RAGPipeline
from .schemas import AskRequest, AskResponse, HealthResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    app.state.pipeline = RAGPipeline()
    app.state.pipeline.initialize()
    yield
    # Cleanup on shutdown


app = FastAPI(
    title="ExperimentIQ API",
    description="Query-routing RAG assistant for MLflow experiment tracking",
    version="0.1.0",
    lifespan=lifespan,
)

# Health check uses HealthResponse from schemas (imported above)


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Ask endpoint - run a question through the pipeline."""
    pipeline = app.state.pipeline

    try:
        result = pipeline.answer(
            query=request.query,
            route=request.route,
            top_k=request.top_k,
            llm_params=request.llm_params,
        )
        return AskResponse(
            query=request.query,
            answer=result.answer,
            route=result.route,
            sources=result.sources,
            confidence=result.confidence,
            hallucination_score=None,
            raw=result.raw,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    pipeline = app.state.pipeline

    components = {
        "mlflow": "ready",
        "qdrant": "ready" if pipeline.qdrant_client.count() >= 0 else "check",
        "llm": "not_configured",
    }

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        components=components,
    )


@app.get("/status")
async def status():
    """Detailed status endpoint."""
    pipeline = app.state.pipeline

    return {
        "pipeline_initialized": pipeline._initialized,
        "mlflow_client": "ready",
        "qdrant_client": "ready",
        "qdrant_collection_count": pipeline.qdrant_client.count(),
        "llm_provider": pipeline.llm_provider,
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "ExperimentIQ API - POST to /ask to query", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
