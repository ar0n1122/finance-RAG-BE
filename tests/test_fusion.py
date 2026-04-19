"""Tests for RRF fusion."""

from __future__ import annotations

from app.models.domain import ContentType, RetrievalResult
from app.retrieval.fusion import ReciprocalRankFusion


def _make_result(chunk_id: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"Text for {chunk_id}",
        content_type=ContentType.TEXT,
        score=score,
        page=1,
    )


class TestRRF:
    def test_single_list(self):
        rrf = ReciprocalRankFusion(k=60)
        results = [_make_result("a", 0.9), _make_result("b", 0.8)]
        fused = rrf.fuse(results)
        assert len(fused) == 2
        assert fused[0].chunk_id == "a"

    def test_merge_two_lists(self):
        rrf = ReciprocalRankFusion(k=60)
        list1 = [_make_result("a", 0.9), _make_result("b", 0.8)]
        list2 = [_make_result("b", 0.95), _make_result("c", 0.7)]
        fused = rrf.fuse(list1, list2)

        # "b" appears in both lists, should rank higher
        ids = [r.chunk_id for r in fused]
        assert "b" in ids
        assert len(fused) == 3  # a, b, c

    def test_top_k(self):
        rrf = ReciprocalRankFusion(k=60)
        results = [_make_result(f"chunk-{i}", 1.0 - i * 0.1) for i in range(10)]
        fused = rrf.fuse(results, top_k=3)
        assert len(fused) == 3

    def test_dedup(self):
        rrf = ReciprocalRankFusion(k=60)
        list1 = [_make_result("a", 0.9)]
        list2 = [_make_result("a", 0.8)]
        fused = rrf.fuse(list1, list2)
        assert len(fused) == 1
        # Should keep the higher original score's payload
        assert fused[0].chunk_id == "a"

    def test_invalid_k(self):
        import pytest

        with pytest.raises(ValueError):
            ReciprocalRankFusion(k=0)
