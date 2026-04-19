"""Tests for retrieval metrics."""

from __future__ import annotations

from app.evaluation.metrics import (
    compute_average_metrics,
    compute_retrieval_metrics,
)
from app.models.domain import ContentType, RetrievalResult


def _make_results(ids: list[str]) -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id=cid,
            document_id="doc-1",
            text=f"Text {cid}",
            content_type=ContentType.TEXT,
            score=1.0 - i * 0.1,
            page=1,
        )
        for i, cid in enumerate(ids)
    ]


class TestRetrievalMetrics:
    def test_perfect_retrieval(self):
        retrieved = _make_results(["a", "b", "c"])
        m = compute_retrieval_metrics(retrieved, {"a", "b"}, k=3)
        assert m.hit_at_k == 1.0
        assert m.mrr == 1.0
        assert m.recall_at_k == 1.0
        assert m.precision_at_k > 0.0

    def test_no_relevant(self):
        retrieved = _make_results(["x", "y", "z"])
        m = compute_retrieval_metrics(retrieved, {"a", "b"}, k=3)
        assert m.hit_at_k == 0.0
        assert m.mrr == 0.0
        assert m.recall_at_k == 0.0

    def test_partial_hit(self):
        retrieved = _make_results(["x", "a", "y"])
        m = compute_retrieval_metrics(retrieved, {"a"}, k=3)
        assert m.hit_at_k == 1.0
        assert m.mrr == 0.5  # Found at rank 2

    def test_average(self):
        m1 = compute_retrieval_metrics(_make_results(["a"]), {"a"}, k=3)
        m2 = compute_retrieval_metrics(_make_results(["x"]), {"a"}, k=3)
        avg = compute_average_metrics([m1, m2])
        assert avg["avg_mrr"] == 0.5

    def test_ndcg(self):
        retrieved = _make_results(["a", "b", "c"])
        m = compute_retrieval_metrics(retrieved, {"a", "b"}, k=3)
        assert m.ndcg_at_k > 0.0
        assert m.ndcg_at_k <= 1.0
