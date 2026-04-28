"""
Custom exception hierarchy.

Every exception carries a human-readable ``detail`` message suitable for API
responses and an optional ``status_code`` so the global exception handler can
map it directly to an HTTP status.
"""

from __future__ import annotations


class RAGException(Exception):
    """Base exception for all RAG platform errors."""

    status_code: int = 500

    def __init__(self, detail: str = "An unexpected error occurred", *, status_code: int | None = None):
        self.detail = detail
        if status_code is not None:
            self.status_code = status_code
        super().__init__(self.detail)


# ── Auth ──────────────────────────────────────────────────────────────────────


class AuthenticationError(RAGException):
    """Credentials are missing or invalid."""

    status_code = 401

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(detail, status_code=self.status_code)


class AuthorizationError(RAGException):
    """Authenticated but not permitted."""

    status_code = 403

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(detail, status_code=self.status_code)


# ── Pipeline ──────────────────────────────────────────────────────────────────


class IngestionError(RAGException):
    """Document upload, parsing, or indexing failed."""

    status_code = 422

    def __init__(self, detail: str = "Document ingestion failed"):
        super().__init__(detail, status_code=self.status_code)


class RetrievalError(RAGException):
    """Vector search or fusion stage failed."""

    status_code = 502

    def __init__(self, detail: str = "Retrieval failed"):
        super().__init__(detail, status_code=self.status_code)


class GenerationError(RAGException):
    """LLM answer generation failed."""

    status_code = 502

    def __init__(self, detail: str = "Generation failed"):
        super().__init__(detail, status_code=self.status_code)


class EvaluationError(RAGException):
    """Benchmark or RAGAS evaluation failed."""

    status_code = 500

    def __init__(self, detail: str = "Evaluation failed"):
        super().__init__(detail, status_code=self.status_code)


# ── Infrastructure ────────────────────────────────────────────────────────────


class StorageError(RAGException):
    """GCS, Firestore, or Qdrant storage operation failed."""

    status_code = 502

    def __init__(self, detail: str = "Storage operation failed"):
        super().__init__(detail, status_code=self.status_code)


class ServiceUnavailableError(RAGException):
    """An external service (Qdrant, Ollama, OpenAI) is unreachable."""

    status_code = 503

    def __init__(self, detail: str = "Service unavailable"):
        super().__init__(detail, status_code=self.status_code)


# ── Guardrails ────────────────────────────────────────────────────────────────


class GuardrailViolationError(RAGException):
    """Input or output guardrail check failed."""

    status_code = 400

    def __init__(self, detail: str = "Guardrail violation", *, violation_type: str = "unknown"):
        self.violation_type = violation_type
        super().__init__(detail, status_code=self.status_code)


# ── Not Found ─────────────────────────────────────────────────────────────────


class NotFoundError(RAGException):
    """Requested resource does not exist."""

    status_code = 404

    def __init__(self, detail: str = "Resource not found"):
        super().__init__(detail, status_code=self.status_code)


# ── Rate limiting ─────────────────────────────────────────────────────────────


class RateLimitError(RAGException):
    """Free-tier usage limit exhausted."""

    status_code = 429

    def __init__(self, detail: str = "You have exhausted your free use limit. Contact Admin"):
        super().__init__(detail, status_code=self.status_code)
