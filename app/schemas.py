"""Pydantic request/response schemas for the ExperimentIQ API."""

from pydantic import BaseModel, Field
from typing import Any, Optional


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    query: str = Field(..., min_length=1, max_length=10000, description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of retrieved items to return")
    route: Optional[str] = Field(
        default="auto",
        description="Routing decision: 'auto' (default), 'structured', or 'unstructured'",
    )
    llm_params: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional overrides for LLM call parameters (model, temperature, max_tokens)",
    )

    model_config = {"populate_by_name": True}


class SourceDocument(BaseModel):
    """A single retrieved source document."""

    type: str = Field(..., description="Source type: 'run', 'config', 'log', 'dataset'")
    id: str = Field(..., description="Unique identifier for the source")
    title: str = Field(..., description="Human-readable title")
    snippet: str = Field(..., description="Truncated content snippet for context")
    score: float = Field(..., ge=0.0, le=1.0, description="Retrieval relevance score")
    metadata: Optional[dict[str, Any]] = Field(
        default=None, description="Raw metadata from the source"
    )


class AskResponse(BaseModel):
    """Response body for the /ask endpoint."""

    query: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer")
    route: str = Field(..., description="Which retrieval path was used")
    sources: list[SourceDocument] = Field(default_factory=list, description="Retrieved source documents")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Answer confidence score"
    )
    hallucination_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Hallucination risk score (0=grounded, 1=hallucinated)"
    )
    raw: dict[str, Any] = Field(default_factory=dict, description="Internal pipeline state for debugging")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="ExperimentIQ version")
    components: dict[str, str] = Field(..., description="Status of each backend component")


class MetricValue(BaseModel):
    """A single metric value from an MLflow run."""

    key: str
    value: float
    epoch: int
    unit: Optional[str] = None
    timestamp: Optional[int] = None


class RunSummary(BaseModel):
    """Summary of an MLflow run."""

    run_id: str
    experiment_id: str
    run_name: Optional[str] = None
    status: str
    start_time: Optional[int] = None
    end_time: Optional[int] = None
    params: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, MetricValue] = Field(default_factory=dict)
    tags: dict[str, str] = Field(default_factory=dict)


class StructuredResult(BaseModel):
    """Result from the structured retrieval path."""

    query_type: str = Field(..., description="Type: 'metric_search', 'param_filter', 'run_summary'")
    results: list[RunSummary]
    query_log: str = Field(..., description="Normalized query string used for the API call")


class UnstructuredResult(BaseModel):
    """Result from the unstructured vector retrieval path."""

    query_type: str = Field(..., description="Type: 'log_retrieval', 'config_retrieval', 'general_retrieval'")
    documents: list[SourceDocument]
    query_log: str = Field(..., description="Normalized query string")


class PipelineResult(BaseModel):
    """Full pipeline result with route and both retrievals available."""

    answer: str
    route: str
    structured_results: Optional[StructuredResult] = None
    unstructured_results: Optional[UnstructuredResult] = None
    sources: list[SourceDocument]
    confidence: float
    hallucination_score: Optional[float] = None
    raw: dict[str, Any]
