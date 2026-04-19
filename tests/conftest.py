"""
Shared test fixtures.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

# Set test environment variables before importing app modules
os.environ.update({
    "RAG_DEBUG": "true",
    "RAG_JWT_SECRET_KEY": "test-secret-key-for-unit-tests",
    "RAG_GOOGLE_CLIENT_ID": "test-client-id",
    "RAG_GCP_PROJECT_ID": "test-project",
    "RAG_GCS_BUCKET": "test-bucket",
    "RAG_QDRANT_HOST": "localhost",
    "RAG_QDRANT_PORT": "6333",
    "RAG_OLLAMA_MODEL": "test-model",
    "RAG_OLLAMA_BASE_URL": "http://localhost:11434",
})


@pytest.fixture
def mock_qdrant_client():
    """Mocked QdrantClient."""
    client = MagicMock()
    client.get_collections.return_value = MagicMock(collections=[])
    return client


@pytest.fixture
def mock_firestore():
    """Mocked FirestoreClient."""
    client = MagicMock()
    client.list_documents.return_value = []
    client.get_document.return_value = None
    return client


@pytest.fixture
def mock_gcs():
    """Mocked CloudStorageClient."""
    client = MagicMock()
    client.upload_file.return_value = "gs://test/file.pdf"
    return client


@pytest.fixture
def sample_chunks():
    """Sample RetrievalResult objects for testing."""
    from app.models.domain import ContentType, RetrievalResult

    return [
        RetrievalResult(
            chunk_id="chunk-1",
            document_id="doc-1",
            text="Revenue for Q3 2024 was $5.2 billion, up 12% year-over-year.",
            content_type=ContentType.TEXT,
            score=0.95,
            page=1,
            section="Financial Summary",
            metadata={"company": "ACME Corp", "year": "2024"},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            document_id="doc-1",
            text="Operating expenses decreased by 3% compared to the previous quarter.",
            content_type=ContentType.TEXT,
            score=0.87,
            page=2,
            section="Expenses",
            metadata={"company": "ACME Corp", "year": "2024"},
        ),
        RetrievalResult(
            chunk_id="chunk-3",
            document_id="doc-2",
            text="The company expanded into three new markets in Southeast Asia.",
            content_type=ContentType.TEXT,
            score=0.72,
            page=5,
            section="Strategy",
            metadata={"company": "ACME Corp", "year": "2024"},
        ),
    ]
