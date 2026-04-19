"""
Custom retrieval quality metrics — Hit@K, MRR, Recall@K, Precision@K.

These metrics evaluate retrieval quality independently of the LLM,
using document-level or chunk-level ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.models.domain import RetrievalResult


@dataclass
class RetrievalMetrics:
    """Computed retrieval quality metrics for a single query."""

    hit_at_k: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    k: int = 5

    def to_dict(self) -> dict[str, float]:
        return {
            f"hit@{self.k}": round(self.hit_at_k, 4),
            f"mrr": round(self.mrr, 4),
            f"recall@{self.k}": round(self.recall_at_k, 4),
            f"precision@{self.k}": round(self.precision_at_k, 4),
            f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
        }


def compute_retrieval_metrics(
    retrieved: list[RetrievalResult],
    relevant_ids: set[str],
    k: int = 5,
) -> RetrievalMetrics:
    """
    Compute retrieval quality metrics.

    Parameters
    ----------
    retrieved : list[RetrievalResult]
        Retrieved chunks, ordered by rank (best first).
    relevant_ids : set[str]
        Set of ground-truth relevant chunk_ids or document_ids.
    k : int
        Cut-off for @K metrics.

    Returns
    -------
    RetrievalMetrics
    """
    top_k = retrieved[:k]
    top_k_ids = [r.chunk_id for r in top_k]

    # Hit@K: 1 if any relevant doc in top-K
    hit = 1.0 if any(cid in relevant_ids for cid in top_k_ids) else 0.0

    # MRR: Reciprocal rank of first relevant result
    mrr = 0.0
    for i, cid in enumerate(top_k_ids):
        if cid in relevant_ids:
            mrr = 1.0 / (i + 1)
            break

    # Recall@K
    retrieved_relevant = sum(1 for cid in top_k_ids if cid in relevant_ids)
    recall = retrieved_relevant / len(relevant_ids) if relevant_ids else 0.0

    # Precision@K
    precision = retrieved_relevant / k if k > 0 else 0.0

    # NDCG@K
    import math

    dcg = 0.0
    for i, cid in enumerate(top_k_ids):
        if cid in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)  # +2 because rank is 1-indexed

    # Ideal DCG: all relevant first
    ideal_relevant = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return RetrievalMetrics(
        hit_at_k=hit,
        mrr=mrr,
        recall_at_k=recall,
        precision_at_k=precision,
        ndcg_at_k=ndcg,
        k=k,
    )


def compute_average_metrics(
    all_metrics: list[RetrievalMetrics],
) -> dict[str, float]:
    """Average metrics across multiple queries."""
    if not all_metrics:
        return {}

    n = len(all_metrics)
    k = all_metrics[0].k
    return {
        f"avg_hit@{k}": round(sum(m.hit_at_k for m in all_metrics) / n, 4),
        "avg_mrr": round(sum(m.mrr for m in all_metrics) / n, 4),
        f"avg_recall@{k}": round(sum(m.recall_at_k for m in all_metrics) / n, 4),
        f"avg_precision@{k}": round(sum(m.precision_at_k for m in all_metrics) / n, 4),
        f"avg_ndcg@{k}": round(sum(m.ndcg_at_k for m in all_metrics) / n, 4),
    }
