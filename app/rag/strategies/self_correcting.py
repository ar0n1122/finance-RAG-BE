"""
Self-correcting RAG — retrieve → generate → reflect → correct loop.
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.generation.pipeline import GenerationPipeline
from app.rag.nodes import (
    make_generate_node,
    make_reflection_node,
    make_retrieve_node,
    should_correct,
)
from app.rag.state import RAGState
from app.rag.strategies import RAGStrategy
from app.retrieval.hybrid import HybridRetriever


class SelfCorrectingRAG(RAGStrategy):
    """
    Retrieve → Generate → Reflect → optionally correct loop.

    The reflection node checks for hallucinations and factual accuracy.
    If issues are found and correction attempts remain, it loops back
    to generation with the corrected answer.

    Best for: High-accuracy queries where hallucination must be minimised.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        gen_pipeline: GenerationPipeline,
        *,
        max_corrections: int = 2,
    ) -> None:
        self._retriever = retriever
        self._gen_pipeline = gen_pipeline
        self._max_corrections = max_corrections
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "self_correcting"

    def invoke(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        state["max_corrections"] = self._max_corrections
        state["correction_attempts"] = 0
        result = self._graph.invoke(state)
        result["total_latency_ms"] = (time.perf_counter() - t0) * 1000
        result["strategy"] = self.name
        return result  # type: ignore[return-value]

    def _build_graph(self) -> StateGraph:
        g = StateGraph(RAGState)

        g.add_node("retrieve", make_retrieve_node(self._retriever))
        g.add_node("generate", make_generate_node(self._gen_pipeline))
        g.add_node("reflect", make_reflection_node(self._gen_pipeline))

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", "reflect")

        g.add_conditional_edges(
            "reflect",
            should_correct,
            {
                "correct": "generate",  # Loop back for correction
                "done": END,
            },
        )

        return g.compile()
