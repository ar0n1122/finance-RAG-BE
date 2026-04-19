"""
Pydantic request schemas — validated by FastAPI on ingress.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentMetaItem(BaseModel):
    """Lightweight document metadata sent from the frontend."""
    id: str
    title: str | None = None
    filename: str | None = None
    status: str | None = None


class QueryRequest(BaseModel):
    """POST /query — ask a question against indexed documents."""

    question: str = Field(..., min_length=3, max_length=2000, description="Natural-language question")
    session_id: str | None = Field(
        default=None,
        description="Chat session ID — used to fetch conversation history for context",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results after reranking")
    modalities: list[Literal["text", "image", "table"]] = Field(
        default=["text", "image", "table"],
        description="Content types to include in retrieval",
    )
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict retrieval to these document IDs (null = all)",
    )
    document_metadata: list[DocumentMetaItem] | None = Field(
        default=None,
        description="Frontend-provided document metadata to avoid extra DB lookup",
    )
    include_ragas: bool = Field(
        default=False,
        description="Run RAGAS inline evaluation on this query",
    )
    llm_provider: Literal["ollama", "openai"] = Field(
        default="ollama",
        description="LLM provider to use for generation",
    )
    rag_strategy: Literal["basic", "self_correcting", "agentic", "adaptive"] | None = Field(
        default=None,
        description="Override default RAG strategy (null = use server default)",
    )


class EvaluateRequest(BaseModel):
    """POST /evaluate — run an offline evaluation benchmark."""

    top_k: int = Field(default=5, ge=1, le=20)
    llm_provider: Literal["ollama", "openai"] = "ollama"
    metrics: list[str] = Field(
        default=["hit_at_k", "mrr", "faithfulness", "answer_relevancy", "context_precision"],
        description="Metrics to compute",
    )


class GoogleAuthRequest(BaseModel):
    """POST /auth/google — exchange a Google ID token for a JWT."""

    id_token: str = Field(..., min_length=10, description="Google OAuth ID token from client-side sign-in")
