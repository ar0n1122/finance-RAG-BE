"""Tests for domain models and request/response schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.domain import ContentType, DocumentStatus, RetrievalResult
from app.models.requests import EvaluateRequest, GoogleAuthRequest, QueryRequest


class TestDomainModels:
    def test_content_type_enum(self):
        assert ContentType.TEXT.value == "text"
        assert ContentType.TABLE.value == "table"
        assert ContentType.IMAGE.value == "image"

    def test_document_status(self):
        assert DocumentStatus.UPLOADED.value == "uploaded"
        assert DocumentStatus.PROCESSING.value == "processing"
        assert DocumentStatus.PROCESSED.value == "processed"
        assert DocumentStatus.ERROR.value == "error"

    def test_retrieval_result_defaults(self):
        r = RetrievalResult(
            chunk_id="c1",
            document_id="doc-1",
            text="hello",
            content_type=ContentType.TEXT,
            score=0.5,
            page=1,
        )
        assert r.section is None
        assert r.metadata == {}


class TestRequestValidation:
    def test_query_request_valid(self):
        req = QueryRequest(query="What is revenue?")
        assert req.query == "What is revenue?"

    def test_query_request_empty(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_request_with_settings(self):
        req = QueryRequest(
            query="Revenue?",
            rag_strategy="basic",
            document_ids=["doc-1"],
        )
        assert req.rag_strategy == "basic"
        assert req.document_ids == ["doc-1"]

    def test_google_auth_request(self):
        req = GoogleAuthRequest(token="abc123")
        assert req.token == "abc123"

    def test_evaluate_request(self):
        req = EvaluateRequest(
            questions=[{"question": "Q1", "ground_truth": "A1"}],
        )
        assert len(req.questions) == 1
