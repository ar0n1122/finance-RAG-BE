"""
Basic RAG — simple retrieve → generate pipeline (no self-correction).
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.generation.pipeline import GenerationPipeline
from app.rag.nodes import make_generate_node, make_retrieve_node
from app.rag.state import RAGState
from app.rag.strategies import RAGStrategy
from app.retrieval.hybrid import HybridRetriever


class BasicRAG(RAGStrategy):
    """
    Simplest RAG: retrieve → generate.

    Best for: Low-latency queries where context is usually sufficient.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        gen_pipeline: GenerationPipeline,
    ) -> None:
        self._retriever = retriever
        self._gen_pipeline = gen_pipeline
        self._graph = self._build_graph()

    @property
    def name(self) -> str:
        return "basic"

    def invoke(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        result = self._graph.invoke(state)
        result["total_latency_ms"] = (time.perf_counter() - t0) * 1000
        result["strategy"] = self.name
        return result  # type: ignore[return-value]

    def _build_graph(self) -> StateGraph:
        g = StateGraph(RAGState)

        g.add_node("retrieve", make_retrieve_node(self._retriever))
        g.add_node("generate", make_generate_node(self._gen_pipeline))

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "generate")
        g.add_edge("generate", END)

        return g.compile()
