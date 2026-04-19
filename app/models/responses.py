"""
Pydantic response schemas — serialised by FastAPI on egress.

These match the TypeScript types defined in ``frontend/src/types/index.ts``
so the API contract is honoured.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Auth ──────────────────────────────────────────────────────────────────────


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class UserResponse(BaseModel):
    user_id: str
    email: str
    display_name: str = ""
    avatar_url: str = ""
    role: str = "user"


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentResponse(BaseModel):
    """Maps to frontend ``DocumentItem``."""

    id: str
    title: str
    filename: str
    status: Literal["indexed", "processing", "failed", "queued", "uploaded"]
    pages: int = 0
    chunks: int = 0
    images: int = 0
    size_bytes: int = 0
    indexed_at: str | None = None
    processing_duration_s: float | None = None
    progress: int | None = None
    processing_stage: str | None = None
    error: str | None = None


class IngestResponse(BaseModel):
    """POST /ingest response."""

    document_id: str
    filename: str
    status: Literal["indexed", "processing", "failed", "queued", "uploaded"]
    total_chunks: int = 0
    total_pages: int = 0
    tables_found: int = 0
    processing_time_s: float = 0.0
    message: str = "Document ingested successfully"


class BatchIngestResponse(BaseModel):
    """POST /ingest/zip response — multiple documents from a ZIP."""

    documents: list[IngestResponse]
    total_files: int
    skipped_files: list[str] = []
    message: str = "ZIP uploaded — processing documents in background"


# ── Query ─────────────────────────────────────────────────────────────────────


class SourceResponse(BaseModel):
    """A single source citation in a query response."""

    document_id: str
    document_title: str
    filename: str
    page: int
    section: str
    text_preview: str
    score: float
    modality: Literal["text", "image", "table"] = "text"


class LatencyResponse(BaseModel):
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0


class RagasScoresResponse(BaseModel):
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0


class QueryResponse(BaseModel):
    """POST /query response — maps to frontend ``QueryResponse``."""

    answer: str
    sources: list[SourceResponse] = Field(default_factory=list)
    latency: LatencyResponse = Field(default_factory=LatencyResponse)
    ragas: RagasScoresResponse | None = None
    model: str = ""
    cost: QueryCostResponse | None = None


# ── Evaluation ────────────────────────────────────────────────────────────────


class EvalMetricsResponse(BaseModel):
    hit_at_k: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    hallucination_rate: float = 0.0
    avg_latency_ms: float = 0.0
    total_questions: int = 0


class EvalQuestionResponse(BaseModel):
    id: str
    question: str
    category: str
    hit_at_k: bool
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    latency_ms: float
    answer: str
    ground_truth: str


class EvalReportResponse(BaseModel):
    """GET /evaluate/results — maps to frontend ``EvalReport``."""

    id: str
    run_at: str
    llm_provider: str
    top_k: int
    metrics: EvalMetricsResponse
    questions: list[EvalQuestionResponse] = Field(default_factory=list)


# ── Health ────────────────────────────────────────────────────────────────────


class ServiceHealthResponse(BaseModel):
    name: str
    url: str
    status: Literal["healthy", "unhealthy", "degraded", "unknown"]
    latency_ms: float = 0.0
    uptime_s: float = 0.0
    version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class PipelineLatencyResponse(BaseModel):
    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    llm_ms: float = 0.0


class HealthResponse(BaseModel):
    """GET /health — maps to frontend ``HealthReport``."""

    overall: Literal["healthy", "unhealthy", "degraded", "unknown"]
    checked_at: str
    services: list[ServiceHealthResponse] = Field(default_factory=list)
    pipeline: PipelineLatencyResponse = Field(default_factory=PipelineLatencyResponse)


# ── Generic ───────────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    detail: str
    error_type: str = "error"
    request_id: str | None = None


# ── Chat Sessions ─────────────────────────────────────────────────────────────


class DocumentMetaResponse(BaseModel):
    """Lightweight document metadata attached to a chat message."""
    id: str
    title: str | None = None
    filename: str | None = None
    status: str | None = None


class ChatMessageResponse(BaseModel):
    id: str
    role: Literal["user", "assistant"]
    content: str
    timestamp: str
    sources: list[SourceResponse] = Field(default_factory=list)
    latency: LatencyResponse | None = None
    ragas: RagasScoresResponse | None = None
    model: str | None = None
    cost: QueryCostResponse | None = None
    document_ids: list[str] | None = None
    document_metadata: list[DocumentMetaResponse] | None = None


class ChatSessionResponse(BaseModel):
    id: str
    title: str
    messages: list[ChatMessageResponse] = Field(default_factory=list)
    created_at: str
    updated_at: str


class ChatSessionSummary(BaseModel):
    id: str
    title: str
    message_count: int = 0
    created_at: str
    updated_at: str


# ── Usage / Cost Analytics ────────────────────────────────────────────────────


class UsageEventResponse(BaseModel):
    """A single LLM call within a query."""

    operation: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_total: float = 0.0
    timestamp: str = ""


class OperationBreakdown(BaseModel):
    """Token/cost breakdown for one operation type."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    calls: int = 0


class QueryCostResponse(BaseModel):
    """Per-query cost info included in query responses."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    breakdown_by_operation: dict[str, OperationBreakdown] = Field(default_factory=dict)
    event_count: int = 0


class UsageRecordResponse(BaseModel):
    """A single usage record (one query) in the history list."""

    id: str
    query_id: str
    query_text: str = ""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    events: list[UsageEventResponse] = Field(default_factory=list)
    breakdown_by_operation: dict[str, OperationBreakdown] = Field(default_factory=dict)
    breakdown_by_model: dict[str, OperationBreakdown] = Field(default_factory=dict)
    created_at: str = ""


class UsageSummaryResponse(BaseModel):
    """Aggregated usage summary for a user."""

    user_id: str
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_queries: int = 0
    avg_cost_per_query: float = 0.0
    avg_tokens_per_query: float = 0.0
    breakdown_by_model: dict[str, OperationBreakdown] = Field(default_factory=dict)
    breakdown_by_operation: dict[str, OperationBreakdown] = Field(default_factory=dict)
    daily_costs: dict[str, float] = Field(default_factory=dict)
