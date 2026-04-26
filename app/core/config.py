"""
Application settings via pydantic-settings.

All configuration is loaded from environment variables with the ``RAG_`` prefix,
or from env files at the project root.  The ``RAG_ENV`` variable controls which
profile is active:

    RAG_ENV=local  →  .env.local   (Ollama, local inference)
    RAG_ENV=dev    →  .env.dev     (OpenRouter, GCP deployment)

The env file is the single source of truth for all operational values.
This module only defines the schema; string settings default to empty
so that a missing env file fails loudly rather than silently using
hardcoded values.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolved once at import time — avoids reading os.environ on every access.
_RAG_ENV: str = os.environ.get("RAG_ENV", "local").strip().strip('"').lower()
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent  # backend/
_ENV_FILE = _BACKEND_DIR / f".env.{_RAG_ENV}"


class EmbeddingModel(str, Enum):
    """Supported text-embedding models."""

    BGE_LARGE = "BAAI/bge-large-en-v1.5"
    NOMIC = "nomic-ai/nomic-embed-text-v1.5"
    OPENAI_SMALL = "openai/text-embedding-3-small"
    OPENAI_LARGE = "openai/text-embedding-3-large"
    NVIDIA_NEMOTRON = "nvidia/llama-nemotron-embed-vl-1b-v2"


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    RECURSIVE = "recursive"
    DOCLING_HYBRID = "docling_hybrid"


class RAGStrategy(str, Enum):
    """Available RAG pipeline strategies."""

    BASIC = "basic"
    SELF_CORRECTING = "self_correcting"
    AGENTIC = "agentic"
    ADAPTIVE = "adaptive"


class Settings(BaseSettings):
    """Central configuration — single source of truth for the entire backend."""

    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE),
        env_prefix="RAG_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Environment Profile ──────────────────────────────────────────────────
    env: str = _RAG_ENV  # "local" or "dev" — informational, set by RAG_ENV

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = ""
    app_version: str = ""
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    allowed_origins: list[str] = Field(default_factory=list)

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_allowed_origins(cls, v: object) -> object:
        """Accept pipe-separated origins (avoids gcloud --update-env-vars comma parsing).
        Also accepts JSON arrays and plain comma-separated strings.
        """
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json
                return json.loads(v)
            # Pipe-separated: preferred format for Cloud Run env vars
            if "|" in v:
                return [o.strip() for o in v.split("|") if o.strip()]
            # Comma-separated fallback
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_host: str = ""
    qdrant_port: int = 6333
    qdrant_api_key: str | None = None
    qdrant_collection: str = ""
    qdrant_use_https: bool = False
    qdrant_timeout: int = 60

    # ── GCP ───────────────────────────────────────────────────────────────────
    gcp_project_id: str = ""
    gcs_bucket: str = ""
    firestore_database: str = ""
    google_application_credentials: str | None = None  # path to SA key JSON

    # ── Google OAuth ─────────────────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""

    # ── JWT ───────────────────────────────────────────────────────────────────
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24

    # ── LLM ───────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "openrouter"] = "ollama"
    llm_fallback_provider: Literal["ollama", "openai", "openrouter", "none"] = "none"
    llm_max_retries: int = 3
    llm_retry_backoff: float = 1.0  # seconds, multiplied by attempt number
    llm_timeout: float = 60.0  # seconds

    # Ollama
    ollama_model: str = ""
    ollama_base_url: str = ""

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = ""

    # OpenRouter (OpenAI-compatible, https://openrouter.ai)
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    openrouter_base_url: str = ""
    openrouter_embedding_model: str = ""

    # LLM Fallback (OpenAI-compatible endpoint)
    llm_fallback_model: str = ""
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: str = ""

    # ── Embeddings ────────────────────────────────────────────────────────────
    # Provider base URL (Ollama, OpenAI, Azure, Groq, Vertex, …)
    embedding_provider: Literal["ollama", "openai", "openrouter"] = "ollama"
    embedding_model: EmbeddingModel = EmbeddingModel.NOMIC
    embedding_api_model: str = ""
    embedding_api_base_url: str = ""  # blank → falls back to ollama_base_url
    embedding_api_key: str = ""
    embedding_api_format: Literal["openai", "ollama"] = "openai"
    embedding_batch_size: int = 32

    # ── Retrieval ─────────────────────────────────────────────────────────────
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rerank_top_k: int = 5
    rrf_k: int = 60
    reranker_enabled: bool = True
    reranker_model: str = ""  # blank → falls back to embedding_api_model
    reranker_api_base_url: str = ""  # blank → falls back to embedding_api_base_url / ollama_base_url
    reranker_api_key: str = ""  # blank → falls back to embedding_api_key
    reranker_api_format: Literal["openai", "ollama"] = "openai"

    # ── PDF Conversion (Docling) ───────────────────────────────────────────────
    # "full" = all models loaded, batch_size=4, best accuracy (~1.5 GB RAM)
    # "slim" = same Docling pipeline but with aggressive memory-saving knobs:
    #          batch_size=1, images_scale=0.25, table_mode=fast,
    #          force_backend_text=True (skip layout-model text extraction)
    #          Peak ~600–800 MB in subprocess, API server stays at ~200 MB.
    docling_mode: Literal["full", "slim"] = "full"

    docling_do_ocr: bool = False
    docling_do_table_structure: bool = True
    docling_table_mode: Literal["fast", "accurate"] = "accurate"
    docling_do_cell_matching: bool = True
    docling_document_timeout: float | None = 600.0  # seconds per document; None = no limit
    docling_num_threads: int = 1   # keep low to avoid OOM on CPU
    docling_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    docling_images_scale: float = 0.5  # page render scale; lower = less RAM

    # Path for local 24-h upload cache files (blank = system temp dir)
    upload_cache_dir: str = ""

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.DOCLING_HYBRID
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_chars: int = 60

    # ── RAG Pipeline ──────────────────────────────────────────────────────────
    rag_strategy: RAGStrategy = RAGStrategy.ADAPTIVE

    # ── Prompts ───────────────────────────────────────────────────────────────
    prompt_version: str = ""

    # ── Guardrails ────────────────────────────────────────────────────────────
    guardrails_enabled: bool = True
    max_input_length: int = 2000

    # ── Monitoring ────────────────────────────────────────────────────────────
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    # ── Evaluation ────────────────────────────────────────────────────────────
    benchmark_dataset_path: str = ""

    # ── Derived ───────────────────────────────────────────────────────────────
    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension based on selected mo del."""
        dims = {
            EmbeddingModel.BGE_LARGE: 1024,
            EmbeddingModel.NOMIC: 768,
            EmbeddingModel.OPENAI_SMALL: 1536,
            EmbeddingModel.OPENAI_LARGE: 3072,
            EmbeddingModel.NVIDIA_NEMOTRON: 2048,
        }
        return dims.get(self.embedding_model, 768)

    @field_validator("embedding_model", mode="before")
    @classmethod
    def _normalize_embedding_model(cls, v: str) -> str:
        """Accept short aliases like 'bge-large' or 'nomic'."""
        aliases = {
            "bge-large": EmbeddingModel.BGE_LARGE.value,
            "bge": EmbeddingModel.BGE_LARGE.value,
            "nomic": EmbeddingModel.NOMIC.value,
            "openai-small": EmbeddingModel.OPENAI_SMALL.value,
            "openai-large": EmbeddingModel.OPENAI_LARGE.value,
            "nvidia-nemotron": EmbeddingModel.NVIDIA_NEMOTRON.value,
            "nemotron": EmbeddingModel.NVIDIA_NEMOTRON.value,
        }
        return aliases.get(v.lower(), v) if isinstance(v, str) else v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    The active profile is determined by ``RAG_ENV`` (default: ``local``).
    """
    return Settings()
