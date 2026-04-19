"""Tests for output guardrails."""

from __future__ import annotations

from app.guardrails.output_guard import OutputGuard
from app.models.domain import ContentType, RetrievalResult


def _make_chunks() -> list[RetrievalResult]:
    return [
        RetrievalResult(
            chunk_id="c1",
            document_id="doc-1",
            text="Revenue for Q3 2024 was $5.2 billion, representing a 12% increase year-over-year.",
            content_type=ContentType.TEXT,
            score=0.9,
            page=1,
        ),
        RetrievalResult(
            chunk_id="c2",
            document_id="doc-1",
            text="Operating expenses decreased by 3% compared to the previous quarter.",
            content_type=ContentType.TEXT,
            score=0.8,
            page=2,
        ),
    ]


class TestOutputGuard:
    def setup_method(self):
        self.guard = OutputGuard()
        self.chunks = _make_chunks()

    def test_good_answer(self):
        answer = "Revenue for Q3 2024 was $5.2 billion [Source 1], with operating expenses down 3% [Source 2]."
        result = self.guard.check(answer, self.chunks, "What was the revenue?")
        assert result.passed is True

    def test_empty_answer(self):
        result = self.guard.check("", self.chunks, "What was the revenue?")
        assert result.passed is False
        assert any("Empty" in v for v in result.violations)

    def test_no_citations_warning(self):
        answer = "Revenue was really good and things are looking up."
        result = self.guard.check(answer, self.chunks, "What was the revenue?")
        assert any("citation" in w.lower() for w in result.warnings)

    def test_invalid_citation_warning(self):
        answer = "Revenue was $5.2B [Source 1] and profit was high [Source 99]."
        result = self.guard.check(answer, self.chunks, "What was revenue?")
        assert any("Invalid citation" in w for w in result.warnings)

    def test_grounding_score(self):
        answer = "Revenue for Q3 2024 was $5.2 billion. Operating expenses decreased by 3%."
        result = self.guard.check(answer, self.chunks, "financial summary")
        assert result.grounding_score > 0.0
