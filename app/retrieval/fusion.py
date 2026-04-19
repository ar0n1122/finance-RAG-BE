"""
Reciprocal Rank Fusion (RRF) — merges ranked lists from dense and sparse retrievers.
"""

from __future__ import annotations

from collections import defaultdict

from app.core.logging import get_logger
from app.models.domain import RetrievalResult

logger = get_logger(__name__)


class ReciprocalRankFusion:
    """
    Merge multiple ranked result lists using RRF.

    RRF_score(d) = Σ  1 / (k + rank_i(d))

    where k is a damping constant (default 60) and rank_i(d) is the 1-based
    rank of document d in the i-th result list.
    """

    def __init__(self, k: int = 60) -> None:
        if k < 1:
            raise ValueError("RRF k must be >= 1")
        self._k = k

    def fuse(
        self,
        *result_lists: list[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Fuse multiple ranked lists and return re-ranked results.

        Each result is identified by its chunk_id. When duplicates appear
        across lists, the payload from the highest-scored occurrence is kept.

        Parameters
        ----------
        *result_lists : list[RetrievalResult]
            One or more ranked result lists, each sorted by descending relevance.
        top_k : int | None
            Limit the number of returned results. None → all.

        Returns
        -------
        list[RetrievalResult]
            Fused results sorted by descending RRF score.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        best_result: dict[str, RetrievalResult] = {}

        for ranked_list in result_lists:
            for rank_0, result in enumerate(ranked_list):
                rrf_scores[result.chunk_id] += 1.0 / (self._k + rank_0 + 1)
                # Keep the occurrence with the highest original score
                existing = best_result.get(result.chunk_id)
                if existing is None or result.score > existing.score:
                    best_result[result.chunk_id] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        if top_k is not None:
            sorted_ids = sorted_ids[:top_k]

        fused: list[RetrievalResult] = []
        for cid in sorted_ids:
            r = best_result[cid]
            # Replace the original score with the RRF score for downstream ranking
            fused.append(RetrievalResult(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                content_type=r.content_type,
                score=rrf_scores[cid],
                page=r.page,
                section=r.section,
                metadata=r.metadata,
            ))

        logger.debug(
            "rrf_fused",
            input_lists=len(result_lists),
            unique_chunks=len(rrf_scores),
            returned=len(fused),
        )
        return fused
