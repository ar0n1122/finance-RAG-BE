"""
Adaptive RAG — the *optimal combined* strategy.

Dynamically selects and combines the best elements from basic,
self-correcting, and agentic strategies based on query complexity
and initial retrieval confidence.

Pipeline:
  1. Retrieve
  2. Assess retrieval confidence
  3. Low confidence → grade + rewrite + re-retrieve (agentic path)
     High confidence → proceed directly
  4. Generate (qa_v2 prompt)
  5. Reflect for hallucinations
  6. Correct if needed (bounded loop)
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, StateGraph

from app.core.logging import get_logger
from app.generation.pipeline import GenerationPipeline
from app.rag.nodes import (
    make_generate_node,
    make_grade_documents_node,
    make_reflection_node,
    make_retrieve_node,
    make_rewrite_query_node,
    should_correct,
)
from app.rag.state import RAGState
from app.rag.strategies import RAGStrategy
from app.retrieval.hybrid import HybridRetriever

logger = get_logger(__name__)


def _make_confidence_router(min_score: float = 0.4, min_results: int = 2):
    """
    Create a routing function that decides whether retrieval results
    are confident enough to generate directly, or need grading/rewriting.
    """

    def assess_confidence(state: RAGState) -> str:
        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return "low_confidence"

        top_scores = [c.score for c in chunks[:5]]
        avg_score = sum(top_scores) / len(top_scores) if top_scores else 0.0

        if len(chunks) >= min_results and avg_score >= min_score:
            logger.debug("adaptive_confidence", decision="high", avg_score=round(avg_score, 3))
            return "high_confidence"
        else:
            logger.debug("adaptive_confidence", decision="low", avg_score=round(avg_score, 3), count=len(chunks))
            return "low_confidence"

    return assess_confidence


def _make_grade_then_route():
    """After grading, decide whether to rewrite or generate."""

    def route_after_grade(state: RAGState) -> str:
        relevant = state.get("relevant_chunks", [])
        attempts = state.get("retrieval_attempts", 0)
        max_attempts = state.get("max_retrieval_attempts", 2)

        if len(relevant) < 2 and attempts < max_attempts:
            return "rewrite"
        return "generate"

    return route_after_grade


class AdaptiveRAG(RAGStrategy):
    """
    Adaptive RAG — combines the best of all strategies with dynamic routing.

    - High-confidence results → fast path (like basic)
    - Low-confidence results → grade + rewrite loop (like agentic)
    - Always reflects for hallucinations (like self-correcting)
    - Bounded correction loops for safety
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        gen_pipeline: GenerationPipeline,
        *,
        max_corrections: int = 2,
        max_retrieval_attempts: int = 2,
        confidence_threshold: float = 0.4,
    ) -> None:
        self._retriever = retriever
        self._gen_pipeline = gen_pipeline
        self._max_corrections = max_corrections
        self._max_retrieval_attempts = max_retrieval_attempts
        self._confidence_threshold = confidence_threshold
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "adaptive"

    def invoke(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        state["max_corrections"] = self._max_corrections
        state["max_retrieval_attempts"] = self._max_retrieval_attempts
        state["correction_attempts"] = 0
        state["retrieval_attempts"] = 0
        result = self._graph.invoke(state)
        result["total_latency_ms"] = (time.perf_counter() - t0) * 1000
        result["strategy"] = self.name
        return result  # type: ignore[return-value]

    def _build_graph(self) -> StateGraph:
        g = StateGraph(RAGState)

        # Nodes
        g.add_node("retrieve", make_retrieve_node(self._retriever))
        g.add_node("grade", make_grade_documents_node(self._gen_pipeline))
        g.add_node("rewrite", make_rewrite_query_node(self._gen_pipeline))
        g.add_node("generate", make_generate_node(self._gen_pipeline, prompt_name="qa_v2"))
        g.add_node("reflect", make_reflection_node(self._gen_pipeline))

        # Entry
        g.set_entry_point("retrieve")

        # After retrieval: assess confidence
        g.add_conditional_edges(
            "retrieve",
            _make_confidence_router(min_score=self._confidence_threshold),
            {
                "high_confidence": "generate",  # Fast path
                "low_confidence": "grade",       # Quality check path
            },
        )

        # After grading: rewrite or generate
        g.add_conditional_edges(
            "grade",
            _make_grade_then_route(),
            {
                "rewrite": "rewrite",
                "generate": "generate",
            },
        )

        g.add_edge("rewrite", "retrieve")  # Re-retrieve with better query
        g.add_edge("generate", "reflect")

        # After reflection: correct or finish
        g.add_conditional_edges(
            "reflect",
            should_correct,
            {
                "correct": "generate",
                "done": END,
            },
        )

        return g.compile()
