"""
Input guardrails — pre-processing validation for user queries.

Checks:
1. Prompt injection detection
2. PII detection in queries
3. Topic / scope relevance
4. Query length and quality
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.exceptions import GuardrailViolationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Prompt injection patterns ─────────────────────────────────────────────────

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"system\s*prompt\s*:", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.IGNORECASE),
    re.compile(r"act\s+as\s+(if\s+you|a|an)", re.IGNORECASE),
    re.compile(r"override\s+(your|the)\s+(instructions|rules|guidelines)", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all|your)", re.IGNORECASE),
    re.compile(r"new\s+instruction[s]?\s*:", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
    re.compile(r"\[\s*INST\s*\]", re.IGNORECASE),
]

# ── PII patterns ──────────────────────────────────────────────────────────────

_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.]?\d{2}[-.]?\d{4}\b"),  # SSN
    re.compile(r"\b\d{16}\b"),  # Credit card (basic)
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),  # Email
    re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"),  # Phone
]


@dataclass
class InputGuardResult:
    """Result of input guardrail checks."""

    passed: bool = True
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sanitised_query: str = ""


class InputGuard:
    """
    Validates user queries before they enter the RAG pipeline.

    Configurable via settings — each check can be enabled/disabled.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._enabled = settings.guardrails_enabled
        self._max_query_length = settings.max_input_length

    def check(self, query: str) -> InputGuardResult:
        """
        Run all input guardrail checks.

        Raises GuardrailViolationError if a blocking violation is detected.
        Returns InputGuardResult with any warnings.
        """
        result = InputGuardResult(sanitised_query=query.strip())

        if not self._enabled:
            return result

        # 1. Length check
        self._check_length(query, result)

        # 2. Empty / low quality
        self._check_quality(query, result)

        # 3. Prompt injection
        self._check_injection(query, result)

        # 4. PII in query
        self._check_pii(query, result)

        if result.violations:
            logger.warning("input_guardrail_violation", violations=result.violations)
            raise GuardrailViolationError(
                f"Query blocked: {'; '.join(result.violations)}"
            )

        if result.warnings:
            logger.info("input_guardrail_warning", warnings=result.warnings)

        return result

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_length(self, query: str, result: InputGuardResult) -> None:
        if len(query) > self._max_query_length:
            result.passed = False
            result.violations.append(
                f"Query too long ({len(query)} chars, max {self._max_query_length})"
            )

    def _check_quality(self, query: str, result: InputGuardResult) -> None:
        stripped = query.strip()
        if not stripped:
            result.passed = False
            result.violations.append("Empty query")
            return
        if len(stripped.split()) < 2:
            result.warnings.append("Very short query — results may be imprecise")

    def _check_injection(self, query: str, result: InputGuardResult) -> None:
        for pattern in _INJECTION_PATTERNS:
            if pattern.search(query):
                result.passed = False
                result.violations.append("Potential prompt injection detected")
                return

    def _check_pii(self, query: str, result: InputGuardResult) -> None:
        for pattern in _PII_PATTERNS:
            if pattern.search(query):
                result.warnings.append("Query may contain PII — consider rephrasing")
                return
