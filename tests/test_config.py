"""Tests for core configuration."""

from __future__ import annotations

from app.core.config import EmbeddingModel, Settings, get_settings


class TestSettings:
    def test_defaults_from_env(self):
        """Values come from .env file, not hardcoded in config.py."""
        s = get_settings()
        # These should be set by .env.local (default profile)
        assert s.app_name == "rag-pipeline"
        assert s.qdrant_host == "localhost"
        assert s.qdrant_collection == "documents"
        assert s.chunk_size == 1024

    def test_embedding_dim_bge(self):
        s = Settings(
            embedding_model=EmbeddingModel.BGE_LARGE,
            jwt_secret="test",
            google_client_id="test",
            gcp_project_id="test",
        )
        assert s.embedding_dim == 1024

    def test_embedding_dim_nomic(self):
        s = Settings(
            embedding_model=EmbeddingModel.NOMIC,
            jwt_secret="test",
            google_client_id="test",
            gcp_project_id="test",
        )
        assert s.embedding_dim == 768
