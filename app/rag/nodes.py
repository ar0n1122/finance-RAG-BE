"""
Shared graph node functions used across multiple RAG strategies.

Each node function takes ``RAGState`` and returns a partial state update dict.
They are intentionally decoupled from a specific graph so they can be composed
freely by any strategy.
"""

from __future__ import annotations

import time
from typing import Any

from app.core.logging import get_logger
from app.generation.pipeline import GenerationPipeline
from app.generation.prompts import PromptManager
from app.models.domain import Source
from app.monitoring.cost_tracker import (
    OPERATION_GRADING,
    OPERATION_REFLECTION,
    OPERATION_REWRITE,
)
from app.rag.state import RAGState
from app.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


# ── Node factories ────────────────────────────────────────────────────────────
# Factory functions return closures so the nodes can capture injected services
# without polluting the graph state.


def make_retrieve_node(retriever: HybridRetriever):
    """Create a *retrieve* node that runs hybrid search."""

    def retrieve(state: RAGState) -> dict[str, Any]:
        query = state.get("rewritten_query") or state["query"]
        doc_ids = state.get("document_ids")

        output, timings = retriever.retrieve(query, document_ids=doc_ids)

        return {
            "retrieved_chunks": output.results,
            "sources": output.sources,
            "retrieval_timings": timings.to_dict(),
        }

    return retrieve


def make_generate_node(gen_pipeline: GenerationPipeline, prompt_name: str = "qa_v1"):
    """Create a *generate* node that calls the LLM pipeline."""

    def generate(state: RAGState) -> dict[str, Any]:
        # Use relevant_chunks if grading was done, else raw retrieved
        chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
        query = state["query"]
        doc_context = state.get("document_context", "")

        t0 = time.perf_counter()
        output = gen_pipeline.generate(
            query, chunks, prompt_name=prompt_name, document_context=doc_context,
        )
        elapsed = (time.perf_counter() - t0) * 1000

        logger.debug("generate_answer", answer_preview=output.answer[:300], ms=round(elapsed, 1))

        return {
            "answer": output.answer,
            "final_answer": output.answer,
            "final_sources": output.sources or [_to_source(r) for r in chunks],
            "model": output.model,
            "generation_latency_ms": elapsed,
        }

    return generate


def make_grade_documents_node(gen_pipeline: GenerationPipeline):
    """Create a *grade_documents* node that filters irrelevant chunks."""

    prompt_manager = gen_pipeline._prompts

    def grade_documents(state: RAGState) -> dict[str, Any]:
        chunks = state.get("retrieved_chunks", [])
        query = state["query"]
        template = prompt_manager.get("grading_v1")

        relevant = []
        irrelevant = 0

        for chunk in chunks:
            _, user_prompt = template.format(question=query, document=chunk.text)
            response = gen_pipeline.generate_raw(user_prompt, temperature=0.0, max_tokens=64, operation=OPERATION_GRADING)
            if "yes" in response.text.lower():
                relevant.append(chunk)
            else:
                irrelevant += 1

        logger.info("grade_documents", total=len(chunks), relevant=len(relevant), irrelevant=irrelevant)
        return {
            "relevant_chunks": relevant,
            "irrelevant_count": irrelevant,
        }

    return grade_documents


def make_reflection_node(gen_pipeline: GenerationPipeline):
    """Create a *reflect* node that checks for hallucinations."""

    prompt_manager = gen_pipeline._prompts

    def reflect(state: RAGState) -> dict[str, Any]:
        chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
        answer = state.get("answer", "")
        query = state["query"]

        # Hallucination check
        hal_template = prompt_manager.get("hallucination_check_v1")
        context = "\n\n".join(c.text for c in chunks)
        _, hal_prompt = hal_template.format(context=context, answer=answer)
        hal_resp = gen_pipeline.generate_raw(hal_prompt, temperature=0.0, max_tokens=64, operation=OPERATION_REFLECTION)
        hal_text = hal_resp.text.strip().lower()
        logger.debug("hallucination_raw", raw_repr=repr(hal_text), length=len(hal_text))
        is_hallucination = bool(hal_text) and "yes" in hal_text

        # Reflection for correction
        ref_template = prompt_manager.get("reflection_v1")
        _, ref_prompt = ref_template.format(question=query, context=context, answer=answer)
        ref_resp = gen_pipeline.generate_raw(ref_prompt, temperature=0.0, max_tokens=128, operation=OPERATION_REFLECTION)
        resp_text = ref_resp.text.strip()
        logger.debug("reflection_raw", raw_repr=repr(resp_text), length=len(resp_text))
        if not resp_text:
            # Empty LLM response — treat as approved (benefit of the doubt)
            needs_correction = False
        else:
            resp_upper = resp_text.upper()
            needs_correction = "APPROVED" not in resp_upper

        attempts = state.get("correction_attempts", 0) + 1

        update: dict[str, Any] = {
            "reflection": ref_resp.text,
            "is_hallucination": is_hallucination,
            "needs_correction": needs_correction,
            "correction_attempts": attempts,
        }

        # Do NOT overwrite the answer with reflection feedback.
        # If correction is needed, the loop routes back to the generate node
        # which will produce a fresh answer from the original chunks.

        logger.info("reflection", hallucination=is_hallucination, needs_correction=needs_correction, attempt=attempts)
        return update

    return reflect


def make_rewrite_query_node(gen_pipeline: GenerationPipeline):
    """Create a *rewrite_query* node for adaptive retrieval."""

    prompt_manager = gen_pipeline._prompts

    def rewrite_query(state: RAGState) -> dict[str, Any]:
        query = state["query"]
        template = prompt_manager.get("query_rewrite_v1")
        _, user_prompt = template.format(question=query)
        response = gen_pipeline.generate_raw(user_prompt, temperature=0.3, max_tokens=256, operation=OPERATION_REWRITE)

        rewritten = response.text.strip()
        attempts = state.get("retrieval_attempts", 0) + 1
        logger.info("query_rewrite", original=query[:80], rewritten=rewritten[:80], attempt=attempts)

        return {
            "rewritten_query": rewritten,
            "retrieval_attempts": attempts,
        }

    return rewrite_query


# ── Condition functions ───────────────────────────────────────────────────────


def should_correct(state: RAGState) -> str:
    """Routing condition: 'correct' if needs correction and under limit, else 'done'."""
    max_corrections = state.get("max_corrections", 2)
    if state.get("needs_correction") and state.get("correction_attempts", 0) < max_corrections:
        return "correct"
    return "done"


def should_retry_retrieval(state: RAGState) -> str:
    """Routing condition: 'retry' if too many irrelevant docs, else 'generate'."""
    max_attempts = state.get("max_retrieval_attempts", 2)
    irrelevant = state.get("irrelevant_count", 0)
    relevant = len(state.get("relevant_chunks", []))
    attempts = state.get("retrieval_attempts", 0)

    # Retry if majority of docs are irrelevant and under limit
    if relevant == 0 and irrelevant > 0 and attempts < max_attempts:
        return "retry"
    if relevant < 2 and irrelevant > relevant and attempts < max_attempts:
        return "retry"
    return "generate"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _to_source(r: Any) -> Source:
    meta = getattr(r, "metadata", {}) or {}
    return Source(
        document_id=r.document_id,
        document_title=meta.get("title", ""),
        filename=meta.get("filename", ""),
        page=r.page,
        section=getattr(r, "section", "") or "",
        text_preview=r.text[:500],
        score=r.score,
        modality=getattr(r, "content_type", "text"),
    )
