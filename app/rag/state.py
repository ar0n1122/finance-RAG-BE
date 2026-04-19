"""
Shared RAG pipeline state — TypedDict for LangGraph nodes.
"""

from __future__ import annotations

from typing import Any, TypedDict

from app.models.domain import GenerationOutput, RetrievalResult, Source


class RAGState(TypedDict, total=False):
    """
    Shared state flowing through LangGraph RAG nodes.

    total=False makes all keys optional so nodes only need to set
    the keys they produce.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    query: str
    document_ids: list[str]
    document_context: str  # human-readable document inventory for the LLM

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieved_chunks: list[RetrievalResult]
    sources: list[Source]
    retrieval_timings: dict[str, float]

    # ── Grading / Filtering ───────────────────────────────────────────────────
    relevant_chunks: list[RetrievalResult]
    irrelevant_count: int

    # ── Generation ────────────────────────────────────────────────────────────
    answer: str
    model: str
    generation_latency_ms: float

    # ── Reflection / Self-correction ──────────────────────────────────────────
    reflection: str
    is_hallucination: bool
    needs_correction: bool
    correction_attempts: int
    max_corrections: int

    # ── Query rewriting ───────────────────────────────────────────────────────
    rewritten_query: str
    retrieval_attempts: int
    max_retrieval_attempts: int

    # ── Final ─────────────────────────────────────────────────────────────────
    final_answer: str
    final_sources: list[Source]
    total_latency_ms: float
    strategy: str
    metadata: dict[str, Any]
