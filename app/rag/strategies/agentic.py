"""
Agentic RAG — retrieve → grade → (re-retrieve or generate) → reflect.

Adds document relevance grading and query rewriting to improve retrieval
quality before generation.
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.generation.pipeline import GenerationPipeline
from app.rag.nodes import (
    make_generate_node,
    make_grade_documents_node,
    make_reflection_node,
    make_retrieve_node,
    make_rewrite_query_node,
    should_correct,
    should_retry_retrieval,
)
from app.rag.state import RAGState
from app.rag.strategies import RAGStrategy
from app.retrieval.hybrid import HybridRetriever


class AgenticRAG(RAGStrategy):
    """
    Agentic RAG with document grading and query rewriting.

    Pipeline: retrieve → grade → (rewrite + re-retrieve | generate) → reflect → correct?

    Best for: Complex questions where initial retrieval quality may be poor
    and query reformulation would help.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        gen_pipeline: GenerationPipeline,
        *,
        max_corrections: int = 1,
        max_retrieval_attempts: int = 2,
    ) -> None:
        self._retriever = retriever
        self._gen_pipeline = gen_pipeline
        self._max_corrections = max_corrections
        self._max_retrieval_attempts = max_retrieval_attempts
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "agentic"

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

        g.add_node("retrieve", make_retrieve_node(self._retriever))
        g.add_node("grade", make_grade_documents_node(self._gen_pipeline))
        g.add_node("rewrite", make_rewrite_query_node(self._gen_pipeline))
        g.add_node("generate", make_generate_node(self._gen_pipeline, prompt_name="qa_v2"))
        g.add_node("reflect", make_reflection_node(self._gen_pipeline))

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "grade")

        # After grading: retry retrieval or proceed to generate
        g.add_conditional_edges(
            "grade",
            should_retry_retrieval,
            {
                "retry": "rewrite",
                "generate": "generate",
            },
        )

        g.add_edge("rewrite", "retrieve")  # Re-retrieve with rewritten query
        g.add_edge("generate", "reflect")

        g.add_conditional_edges(
            "reflect",
            should_correct,
            {
                "correct": "generate",
                "done": END,
            },
        )

        return g.compile()
