"""
Dense vector retriever — Qdrant HNSW cosine similarity on named vectors.
"""

from __future__ import annotations

import time

from qdrant_client import QdrantClient, models

from app.core.logging import get_logger
from app.ingestion.embeddings.base import EmbeddingProvider
from app.models.domain import ContentType, RetrievalResult

logger = get_logger(__name__)


class DenseRetriever:
    """Semantic search via Qdrant named dense vectors."""

    def __init__(
        self,
        client: QdrantClient,
        embedder: EmbeddingProvider,
        collection: str = "documents",
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._collection = collection

    def search(
        self,
        query: str,
        top_k: int = 20,
        content_type: str | None = None,
        document_ids: list[str] | None = None,
    ) -> tuple[list[RetrievalResult], float]:
        """
        Run a dense vector search.

        Returns (results, elapsed_ms).
        """
        t0 = time.perf_counter()
        query_vector = self._embedder.embed_query(query)

        qdrant_filter = self._build_filter(content_type, document_ids)
        points = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="text",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        results = [self._to_result(p) for p in points.points]
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("dense_search", query_preview=query[:80], results=len(results), ms=round(elapsed, 1))
        return results, elapsed

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(
        content_type: str | None,
        document_ids: list[str] | None,
    ) -> models.Filter | None:
        conditions: list[models.FieldCondition] = []
        if content_type:
            conditions.append(models.FieldCondition(
                key="content_type", match=models.MatchValue(value=content_type),
            ))
        if document_ids:
            conditions.append(models.FieldCondition(
                key="document_id", match=models.MatchAny(any=document_ids),
            ))
        return models.Filter(must=conditions) if conditions else None

    @staticmethod
    def _to_result(point: models.ScoredPoint) -> RetrievalResult:
        p = point.payload or {}
        return RetrievalResult(
            chunk_id=str(point.id),
            document_id=p.get("document_id", ""),
            text=p.get("text", ""),
            content_type=ContentType(p.get("content_type", "text")),
            score=point.score or 0.0,
            page=p.get("page", 0),
            section=p.get("section"),
            metadata=p.get("metadata", {}),
        )
