"""
Sparse vector retriever — BM25-weighted sparse vectors via Qdrant.

Falls back to full-text search when no sparse vectors are indexed.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from qdrant_client import QdrantClient, models

from app.core.logging import get_logger
from app.models.domain import ContentType, RetrievalResult

logger = get_logger(__name__)

# Simple stop-words for BM25 tokenisation (extend as needed)
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between",
    "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "in", "out", "on", "off", "over", "under",
    "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "what", "which", "who", "whom", "this",
    "that", "these", "those", "i", "me", "my", "myself", "we", "our",
    "ours", "ourselves", "you", "your", "yours", "he", "him", "his",
    "she", "her", "hers", "it", "its", "they", "them", "their",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class SparseRetriever:
    """BM25-style sparse retrieval via Qdrant sparse named vectors."""

    def __init__(
        self,
        client: QdrantClient,
        collection: str = "documents",
        *,
        k1: float = 1.5,
        b: float = 0.75,
        avg_dl: float = 256.0,
    ) -> None:
        self._client = client
        self._collection = collection
        self._k1 = k1
        self._b = b
        self._avg_dl = avg_dl

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        document_ids: list[str] | None = None,
    ) -> tuple[list[RetrievalResult], float]:
        """
        Run a sparse vector search using locally-computed BM25 sparse query.

        Returns (results, elapsed_ms).
        """
        t0 = time.perf_counter()
        tokens = self._tokenise(query)
        if not tokens:
            return [], 0.0

        sparse_vector = self._to_sparse_vector(tokens)
        qdrant_filter = self._build_filter(document_ids)

        try:
            points = self._client.query_points(
                collection_name=self._collection,
                query=models.SparseVector(
                    indices=sparse_vector["indices"],
                    values=sparse_vector["values"],
                ),
                using="text-sparse",
                query_filter=qdrant_filter,
                limit=top_k,
                with_payload=True,
            )
            results = [self._to_result(p) for p in points.points]
        except Exception:
            logger.warning("sparse_search_fallback", reason="sparse vector search failed, trying full-text")
            results = self._fulltext_fallback(query, top_k, qdrant_filter)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug("sparse_search", query_preview=query[:80], results=len(results), ms=round(elapsed, 1))
        return results, elapsed

    # ── Full-text fallback ────────────────────────────────────────────────────

    def _fulltext_fallback(
        self,
        query: str,
        top_k: int,
        qdrant_filter: models.Filter | None,
    ) -> list[RetrievalResult]:
        """Fall back to Qdrant full-text match when sparse vectors aren't available."""
        scroll_filter_conditions: list[models.Condition] = [
            models.FieldCondition(key="text", match=models.MatchText(text=query)),
        ]
        if qdrant_filter and qdrant_filter.must:
            scroll_filter_conditions.extend(qdrant_filter.must)

        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=models.Filter(must=scroll_filter_conditions),
            limit=top_k,
            with_payload=True,
        )
        return [self._to_result_from_record(p, score=1.0) for p in points]

    # ── BM25 sparse vector encoding ──────────────────────────────────────────

    def _tokenise(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]

    def _to_sparse_vector(self, tokens: list[str]) -> dict:
        """
        Compute TF-weighted sparse query vector. Each unique token gets a
        deterministic integer index via hash and a BM25-style TF weight.
        """
        tf = Counter(tokens)
        dl = len(tokens)
        indices: list[int] = []
        values: list[float] = []
        for token, freq in tf.items():
            idx = abs(hash(token)) % (2**31)  # Deterministic integer index
            tf_score = (freq * (self._k1 + 1)) / (
                freq + self._k1 * (1 - self._b + self._b * dl / self._avg_dl)
            )
            # IDF approximation — log(N/df) ≈ constant boost for single query
            weight = tf_score * math.log(10.0)  # Mild IDF proxy
            indices.append(idx)
            values.append(round(weight, 6))
        return {"indices": indices, "values": values}

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_filter(document_ids: list[str] | None) -> models.Filter | None:
        if not document_ids:
            return None
        return models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchAny(any=document_ids)),
        ])

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

    @staticmethod
    def _to_result_from_record(record: models.Record, score: float) -> RetrievalResult:
        p = record.payload or {}
        return RetrievalResult(
            chunk_id=str(record.id),
            document_id=p.get("document_id", ""),
            text=p.get("text", ""),
            content_type=ContentType(p.get("content_type", "text")),
            score=score,
            page=p.get("page", 0),
            section=p.get("section"),
            metadata=p.get("metadata", {}),
        )
