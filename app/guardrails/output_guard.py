"""
Output guardrails — post-processing validation for LLM responses.

Checks:
1. Hallucination detection (context grounding)
2. Relevance verification
3. Toxicity / inappropriate content
4. Citation verification
5. Response quality
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.domain import RetrievalResult

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[Source\s*(\d+)\]", re.IGNORECASE)

# Toxicity / refusal indicators (basic keyword-based)
_TOXIC_PATTERNS = [
    re.compile(r"\b(kill|murder|harm|attack|destroy)\s+(people|humans|someone)", re.IGNORECASE),
    re.compile(r"\b(hate\s+speech|racial\s+slur)", re.IGNORECASE),
]

_REFUSAL_PHRASES = [
    "i cannot",
    "i can't",
    "i'm not able to",
    "as an ai",
    "i don't have access",
    "i'm unable to",
]


@dataclass
class OutputGuardResult:
    """Result of output guardrail checks."""

    passed: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    citation_coverage: float = 0.0
    grounding_score: float = 1.0
    sanitised_answer: str = ""


class OutputGuard:
    """
    Validates LLM output before returning to the user.

    Operates without calling the LLM again — uses heuristic and
    text-overlap methods for speed.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = settings.guardrails_enabled
        self._min_grounding = 0.3  # Minimum fraction of answer grounded in context

    def check(
        self,
        answer: str,
        context_chunks: list[RetrievalResult],
        query: str,
    ) -> OutputGuardResult:
        """
        Run all output guardrail checks.

        Returns OutputGuardResult with pass/fail and diagnostics.
        Does NOT raise — returns result so the caller can decide policy.
        """
        result = OutputGuardResult(sanitised_answer=answer)

        if not self._enabled:
            return result

        # 1. Empty answer
        self._check_empty(answer, result)

        # 2. Toxicity
        self._check_toxicity(answer, result)

        # 3. Refusal detection
        self._check_refusal(answer, result)

        # 4. Grounding / hallucination (heuristic)
        self._check_grounding(answer, context_chunks, result)

        # 5. Citation coverage
        self._check_citations(answer, context_chunks, result)

        # 6. Relevance to query (keyword overlap)
        self._check_relevance(answer, query, result)

        if result.violations:
            logger.warning("output_guardrail_violation", violations=result.violations)
            result.passed = False

        if result.warnings:
            logger.info("output_guardrail_warning", warnings=result.warnings)

        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_empty(self, answer: str, result: OutputGuardResult) -> None:
        if not answer.strip():
            result.violations.append("Empty answer generated")

    def _check_toxicity(self, answer: str, result: OutputGuardResult) -> None:
        for pattern in _TOXIC_PATTERNS:
            if pattern.search(answer):
                result.violations.append("Potentially toxic content detected")
                return

    def _check_refusal(self, answer: str, result: OutputGuardResult) -> None:
        lower = answer.lower()
        for phrase in _REFUSAL_PHRASES:
            if phrase in lower:
                result.warnings.append("Answer may contain a refusal / hedge")
                return

    def _check_grounding(
        self,
        answer: str,
        context_chunks: list[RetrievalResult],
        result: OutputGuardResult,
    ) -> None:
        """
        Heuristic grounding check: measures what fraction of answer sentences
        have significant overlap with the context.
        """
        if not context_chunks:
            result.warnings.append("No context provided — cannot verify grounding")
            result.grounding_score = 0.0
            return

        context_text = " ".join(c.text.lower() for c in context_chunks)
        context_words = set(context_text.split())

        # Split answer into sentences
        sentences = re.split(r"[.!?]\s+", answer)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]

        if not sentences:
            result.grounding_score = 1.0
            return

        grounded = 0
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            if not sentence_words:
                continue
            overlap = len(sentence_words & context_words) / len(sentence_words)
            if overlap > 0.3:  # At least 30% word overlap
                grounded += 1

        score = grounded / len(sentences) if sentences else 1.0
        result.grounding_score = round(score, 3)

        if score < self._min_grounding:
            result.warnings.append(
                f"Low grounding score ({score:.1%}) — answer may contain hallucinations"
            )

    def _check_citations(
        self,
        answer: str,
        context_chunks: list[RetrievalResult],
        result: OutputGuardResult,
    ) -> None:
        """Check that citations reference valid source indices."""
        cited = set()
        for match in _CITATION_RE.finditer(answer):
            idx = int(match.group(1))
            cited.add(idx)

        if not cited:
            result.warnings.append("No citations found in answer")
            result.citation_coverage = 0.0
            return

        valid = sum(1 for idx in cited if 1 <= idx <= len(context_chunks))
        invalid = cited - set(range(1, len(context_chunks) + 1))

        if invalid:
            result.warnings.append(f"Invalid citation indices: {sorted(invalid)}")

        result.citation_coverage = valid / len(context_chunks) if context_chunks else 0.0

    def _check_relevance(
        self,
        answer: str,
        query: str,
        result: OutputGuardResult,
    ) -> None:
        """Basic keyword overlap between query and answer."""
        query_words = set(query.lower().split()) - {"the", "a", "an", "is", "are", "what", "how", "why", "when", "where", "who"}
        answer_words = set(answer.lower().split())

        if not query_words:
            return

        overlap = len(query_words & answer_words) / len(query_words)
        if overlap < 0.1:
            result.warnings.append("Answer may not be relevant to the query")
