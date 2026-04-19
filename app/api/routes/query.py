"""
Query route — RAG pipeline query endpoint.

The RAG pipeline (retrieval, LLM generation, reflection) is entirely
synchronous and can take 30–240+ seconds.  We offload the heavy work to
a thread via ``asyncio.to_thread`` so the asyncio event loop stays free
to serve other requests (chat CRUD, health checks, etc.).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field as dc_field
from typing import Any

from fastapi import APIRouter, HTTPException

from app.api.dependencies import (
    get_firestore_client,
    get_generation_pipeline,
    get_input_guard,
    get_output_guard,
    get_rag_registry,
    get_tracing_provider,
)
from app.rag.document_resolver import resolve_document_ids
from app.auth.dependencies import OptionalUser
from app.core.config import get_settings
from app.core.exceptions import (
    GenerationError,
    GuardrailViolationError,
    RetrievalError,
)
from app.core.logging import get_logger
from app.models.requests import QueryRequest
from app.models.responses import (
    LatencyResponse,
    OperationBreakdown,
    QueryCostResponse,
    QueryResponse,
    SourceResponse,
)
from app.monitoring.cost_tracker import CostTracker
from app.monitoring.prometheus import QUERY_COUNT, QUERY_ERRORS

logger = get_logger(__name__)

router = APIRouter(tags=["query"])


# ── Internal result container ─────────────────────────────────────────────────

@dataclass
class _PipelineResult:
    """Carries everything produced by the blocking pipeline back to the
    async handler so it can build the HTTP response."""

    answer: str
    sources: list[Any]
    model: str
    retrieval_timings: dict[str, float]
    generation_ms: float
    total_ms: float
    chunks: list[Any]
    cost_tracker: CostTracker
    firestore: Any


# ── Blocking pipeline (runs in a worker thread) ──────────────────────────────

def _run_pipeline_sync(
    *,
    sanitised_query: str,
    body: QueryRequest,
    user_id: str | None,
    strategy_name: str,
    t_total: float,
    query_id: str,
) -> _PipelineResult:
    """Execute the full RAG pipeline synchronously.

    Called from ``asyncio.to_thread`` so the event loop is never blocked.
    ``asyncio.to_thread`` copies the current ``contextvars.Context`` into
    the new thread, so ``CostTracker`` (which uses a ``ContextVar``) works
    correctly.
    """
    cost_tracker = CostTracker(user_id=user_id, session_id=body.session_id)

    with cost_tracker:
        # ── Resolve document scope ────────────────────────────────────
        firestore = get_firestore_client()
        gen_pipeline = get_generation_pipeline()
        user_docs_from_frontend = (
            [m.model_dump() for m in body.document_metadata]
            if body.document_metadata
            else None
        )
        document_ids = resolve_document_ids(
            firestore,
            sanitised_query,
            user_id=user_id,
            explicit_ids=body.document_ids,
            gen_pipeline=gen_pipeline,
            user_docs=user_docs_from_frontend,
            session_id=body.session_id,
        )

        # ── Build document context for the generation prompt ──────────
        doc_context = ""
        user_docs_list = user_docs_from_frontend
        if user_docs_list is None and user_id:
            user_docs_list = firestore.list_documents(user_id=user_id)
        if user_docs_list:
            indexed = [
                d for d in user_docs_list if d.get("status") == "indexed"
            ]
            if indexed:
                lines = [
                    f"- {d.get('title') or d.get('filename', 'Untitled')}"
                    for d in indexed
                ]
                doc_context = (
                    "The user has the following documents uploaded:\n"
                    + "\n".join(lines) + "\n\n"
                )

        # ── RAG pipeline ──────────────────────────────────────────────
        registry = get_rag_registry()
        strategy = registry.get(strategy_name)

        QUERY_COUNT.labels(strategy=strategy_name).inc()

        state = strategy.invoke({
            "query": sanitised_query,
            "document_ids": document_ids,
            "document_context": doc_context,
        })

        answer = state.get("final_answer", state.get("answer", ""))
        sources = state.get("final_sources", state.get("sources", []))
        model = state.get("model", "unknown")
        retrieval_timings = state.get("retrieval_timings", {})
        generation_ms = state.get("generation_latency_ms", 0.0)
        total_ms = (time.perf_counter() - t_total) * 1000

        # ── Output guardrails ─────────────────────────────────────────
        output_guard = get_output_guard()
        chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
        out_result = output_guard.check(answer, chunks, sanitised_query)
        if not out_result.passed:
            logger.warning("output_guardrail_failed", violations=out_result.violations)

    return _PipelineResult(
        answer=answer,
        sources=sources,
        model=model,
        retrieval_timings=retrieval_timings,
        generation_ms=generation_ms,
        total_ms=total_ms,
        chunks=chunks,
        cost_tracker=cost_tracker,
        firestore=firestore,
    )


# ── Async route handler ──────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query(body: QueryRequest, user: OptionalUser) -> QueryResponse:
    """Execute a RAG query against ingested documents."""
    settings = get_settings()
    t_total = time.perf_counter()
    query_id = uuid.uuid4().hex[:16]

    strategy_name = body.rag_strategy or settings.rag_strategy.value

    # ── Input guardrails (fast, regex-based — safe on the event loop) ─────
    try:
        input_guard = get_input_guard()
        guard_result = input_guard.check(body.question)
        sanitised_query = guard_result.sanitised_query
    except GuardrailViolationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ── Tracing ───────────────────────────────────────────────────────────
    tracer = get_tracing_provider()
    trace = tracer.trace(
        name="rag_query",
        input={"query": sanitised_query, "strategy": strategy_name},
        user_id=user.user_id if user else None,
    )

    user_id = user.user_id if user else None

    try:
        # ── Offload the heavy sync pipeline to a worker thread ────────
        result = await asyncio.to_thread(
            _run_pipeline_sync,
            sanitised_query=sanitised_query,
            body=body,
            user_id=user_id,
            strategy_name=strategy_name,
            t_total=t_total,
            query_id=query_id,
        )

        # ── Build response (lightweight, stays on event loop) ─────────
        source_responses = [
            SourceResponse(
                document_id=s.document_id,
                document_title=getattr(s, "document_title", ""),
                filename=getattr(s, "filename", ""),
                page=s.page or 0,
                section=getattr(s, "section", "") or "",
                text_preview=getattr(s, "text_preview", getattr(s, "text", ""))[:500],
                score=getattr(s, "score", getattr(s, "relevance_score", 0.0)),
                modality=getattr(s, "modality", "text"),
            )
            for s in result.sources
        ]

        latency = LatencyResponse(
            dense_ms=result.retrieval_timings.get("dense_ms", 0.0),
            sparse_ms=result.retrieval_timings.get("sparse_ms", 0.0),
            fusion_ms=result.retrieval_timings.get("fusion_ms", 0.0),
            rerank_ms=result.retrieval_timings.get("rerank_ms", 0.0),
            llm_ms=result.generation_ms,
            total_ms=result.total_ms,
        )

        cost_resp = QueryCostResponse(
            total_prompt_tokens=result.cost_tracker.total_prompt_tokens,
            total_completion_tokens=result.cost_tracker.total_completion_tokens,
            total_tokens=result.cost_tracker.total_tokens,
            total_cost=result.cost_tracker.total_cost,
            breakdown_by_operation={
                k: OperationBreakdown(**v)
                for k, v in result.cost_tracker.breakdown_by_operation().items()
            },
            event_count=len(result.cost_tracker.events),
        )

        # ── Save usage record (fire-and-forget, in thread) ────────────
        try:
            usage_data = result.cost_tracker.to_firestore_record(
                query_id=query_id,
                query_text=sanitised_query,
            )
            await asyncio.to_thread(
                result.firestore.save_usage_record, query_id, usage_data,
            )
        except Exception:
            logger.warning("usage_record_save_failed", exc_info=True)

        # ── Tracing update ────────────────────────────────────────────
        tracer.generation(
            trace_id=trace.id,
            name="rag_generation",
            model=result.model,
            input=sanitised_query,
            output=result.answer[:500],
        )

        return QueryResponse(
            answer=result.answer,
            sources=source_responses,
            latency=latency,
            model=result.model,
            cost=cost_resp,
        )

    except (RetrievalError, GenerationError) as exc:
        QUERY_ERRORS.labels(strategy=strategy_name, error_type=type(exc).__name__).inc()
        logger.exception("rag_query_error", strategy=strategy_name)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        QUERY_ERRORS.labels(strategy=strategy_name, error_type="unknown").inc()
        logger.exception("rag_query_error", strategy=strategy_name)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc
    finally:
        tracer.flush()
