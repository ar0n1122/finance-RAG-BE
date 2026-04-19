"""Tests for exception hierarchy."""

from __future__ import annotations

from app.core.exceptions import (
    AuthenticationError,
    GenerationError,
    GuardrailViolationError,
    IngestionError,
    NotFoundError,
    RAGException,
    RetrievalError,
    ServiceUnavailableError,
    StorageError,
)


class TestExceptions:
    def test_base_exception(self):
        exc = RAGException("test error")
        assert exc.message == "test error"
        assert exc.status_code == 500

    def test_auth_error(self):
        exc = AuthenticationError("bad token")
        assert exc.status_code == 401
        assert isinstance(exc, RAGException)

    def test_generation_error(self):
        exc = GenerationError("LLM down")
        assert exc.status_code == 502

    def test_guardrail_error(self):
        exc = GuardrailViolationError("injection detected")
        assert exc.status_code == 400

    def test_not_found(self):
        exc = NotFoundError("doc-123")
        assert exc.status_code == 404

    def test_ingestion_error(self):
        exc = IngestionError("parse failed")
        assert exc.status_code == 422

    def test_service_unavailable(self):
        exc = ServiceUnavailableError("qdrant down")
        assert exc.status_code == 503

    def test_storage_error(self):
        exc = StorageError("GCS write failed")
        assert exc.status_code == 502

    def test_retrieval_error(self):
        exc = RetrievalError("search failed")
        assert exc.status_code == 502
