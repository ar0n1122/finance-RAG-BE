"""
Hybrid retriever — orchestrates dense + sparse → RRF → rerank pipeline.

This is the primary retrieval entry-point used by RAG strategies.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.core.logging import get_logger
from app.ingestion.embeddings.base import EmbeddingProvider
from app.models.domain import RetrievalOutput, RetrievalResult, Source
from app.retrieval.dense import DenseRetriever
from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.reranker import Reranker
from app.retrieval.sparse import SparseRetriever

logger = get_logger(__name__)


@dataclass
class RetrievalTimings:
    """Breakdown of retrieval stage latencies (ms)."""

    dense_ms: float = 0.0
    sparse_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "dense_ms": round(self.dense_ms, 1),
            "sparse_ms": round(self.sparse_ms, 1),
            "fusion_ms": round(self.fusion_ms, 1),
            "rerank_ms": round(self.rerank_ms, 1),
            "total_ms": round(self.total_ms, 1),
        }


class HybridRetriever:
    """
    Full retrieval pipeline: dense + sparse → RRF fusion → cross-encoder rerank.

    Usage
    -----
    ```python
    retriever = HybridRetriever(qdrant_client, embedder)
    output, timings = retriever.retrieve("What is the revenue for Q3?")
    ```
    """

    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingProvider,
        *,
        collection: str = "documents",
        reranker: Reranker | None = None,
    ) -> None:
        settings = get_settings()

        self._dense = DenseRetriever(client, embedder, collection)
        self._sparse = SparseRetriever(client, collection)
        self._rrf = ReciprocalRankFusion(k=settings.rrf_k)
        self._reranker = reranker or Reranker(
            model_name=settings.reranker_model or settings.embedding_api_model,
            base_url=settings.reranker_api_base_url or settings.embedding_api_base_url or settings.ollama_base_url,
            api_key=settings.reranker_api_key or settings.embedding_api_key,
            api_format=settings.reranker_api_format,
        )

        # Configurable limits
        self._dense_top_k = settings.dense_top_k
        self._sparse_top_k = settings.sparse_top_k
        self._rerank_top_k = settings.rerank_top_k
        self._enable_rerank = settings.reranker_enabled

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
        skip_rerank: bool = False,
    ) -> tuple[RetrievalOutput, RetrievalTimings]:
        """
        Execute the full hybrid retrieval pipeline.

        Parameters
        ----------
        query : str
            The user query.
        top_k : int | None
            Final number of results. Defaults to ``rerank_top_k`` from settings.
        document_ids : list[str] | None
            Filter to these document IDs only.
        skip_rerank : bool
            If True, skip the cross-encoder reranking stage.

        Returns
        -------
        tuple[RetrievalOutput, RetrievalTimings]
            The retrieval output (results + sources) and per-stage timings.
        """
        final_k = top_k or self._rerank_top_k
        timings = RetrievalTimings()
        t_total = time.perf_counter()

        # ── Stage 1: Parallel retrieval ──────────────────────────────────────
        dense_results, timings.dense_ms = self._dense.search(
            query, top_k=self._dense_top_k, document_ids=document_ids,
        )
        sparse_results, timings.sparse_ms = self._sparse.search(
            query, top_k=self._sparse_top_k, document_ids=document_ids,
        )

        # ── Stage 2: Fusion ──────────────────────────────────────────────────
        t_fuse = time.perf_counter()
        # Merge with generous top_k to feed into reranker
        fused = self._rrf.fuse(
            dense_results,
            sparse_results,
            top_k=self._dense_top_k + self._sparse_top_k,
        )
        timings.fusion_ms = (time.perf_counter() - t_fuse) * 1000

        # ── Stage 3: Rerank ──────────────────────────────────────────────────
        if not skip_rerank and self._enable_rerank and fused:
            try:
                final_results, timings.rerank_ms = self._reranker.rerank(
                    query, fused, top_k=final_k,
                )
            except Exception as exc:
                logger.warning("reranker_failed_fallback_to_fusion", error=str(exc))
                final_results = fused[:final_k]
                timings.rerank_ms = 0.0
        else:
            final_results = fused[:final_k]
            timings.rerank_ms = 0.0

        timings.total_ms = (time.perf_counter() - t_total) * 1000

        # ── Build output ─────────────────────────────────────────────────────
        output = RetrievalOutput(
            results=final_results,
            sources=[self._to_source(r) for r in final_results],
        )

        logger.info(
            "hybrid_retrieval",
            dense=len(dense_results),
            sparse=len(sparse_results),
            fused=len(fused),
            final=len(final_results),
            **timings.to_dict(),
        )
        return output, timings

    def retrieve_text_only(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: list[str] | None = None,
    ) -> tuple[RetrievalOutput, RetrievalTimings]:
        """Convenience wrapper that filters to text content only."""
        # Dense supports content_type filter
        final_k = top_k or self._rerank_top_k
        timings = RetrievalTimings()
        t_total = time.perf_counter()

        dense_results, timings.dense_ms = self._dense.search(
            query,
            top_k=self._dense_top_k,
            content_type="text",
            document_ids=document_ids,
        )
        sparse_results, timings.sparse_ms = self._sparse.search(
            query, top_k=self._sparse_top_k, document_ids=document_ids,
        )

        t_fuse = time.perf_counter()
        fused = self._rrf.fuse(dense_results, sparse_results, top_k=self._dense_top_k + self._sparse_top_k)
        timings.fusion_ms = (time.perf_counter() - t_fuse) * 1000

        if self._enable_rerank and fused:
            final_results, timings.rerank_ms = self._reranker.rerank(query, fused, top_k=final_k)
        else:
            final_results = fused[:final_k]

        timings.total_ms = (time.perf_counter() - t_total) * 1000
        output = RetrievalOutput(
            results=final_results,
            sources=[self._to_source(r) for r in final_results],
        )
        return output, timings

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _to_source(r: RetrievalResult) -> Source:
        return Source(
            document_id=r.document_id,
            document_title=r.metadata.get("title", ""),
            filename=r.metadata.get("filename", ""),
            page=r.page,
            section=r.section or "",
            text_preview=r.text[:500],
            score=r.score,
            modality=r.content_type.value if hasattr(r.content_type, "value") else str(r.content_type),
        )
