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
from pydantic_settings import BaseSettings, EnvSettingsSource, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.sources.providers.dotenv import DotEnvSettingsSource


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


class _PipeAwareEnvSource(EnvSettingsSource):
    """EnvSettingsSource that falls back to the raw string when json.loads fails.

    pydantic-settings normally calls json.loads() on any list/dict field before
    field validators run.  For fields like ``allowed_origins`` that accept
    pipe-separated input (``a|b|c``), this would crash with a JSONDecodeError.
    By catching the ValueError and returning the raw string, we let the
    field_validator handle all format variants (JSON, pipe, comma).
    """

    def decode_complex_value(self, field_name: str, field_info: object, value: str) -> object:
        try:
            return super().decode_complex_value(field_name, field_info, value)
        except ValueError:
            return value  # pass raw string through to field_validator


class _PipeAwareDotEnvSource(DotEnvSettingsSource):
    """DotEnvSettingsSource that falls back to the raw string when json.loads fails.

    Same rationale as _PipeAwareEnvSource but for values read from .env files.
    """

    def decode_complex_value(self, field_name: str, field_info: object, value: str) -> object:
        try:
            return super().decode_complex_value(field_name, field_info, value)
        except ValueError:
            return value  # pass raw string through to field_validator


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
        Trailing slashes are stripped — browsers send Origin without trailing slash.
        """
        def _clean(origins: list[str]) -> list[str]:
            return [o.strip().rstrip("/") for o in origins if o.strip()]

        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json
                return _clean(json.loads(v))
            if "|" in v:
                return _clean(v.split("|"))
            return _clean(v.split(","))
        if isinstance(v, list):
            return _clean(v)
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
    llm_provider: Literal["ollama", "openai", "openrouter", "fireworks"] = "ollama"
    llm_fallback_provider: Literal["ollama", "openai", "openrouter", "fireworks", "none"] = "none"
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

    # Fireworks AI (OpenAI-compatible, https://fireworks.ai)
    fireworks_api_key: str = ""
    fireworks_model: str = ""
    fireworks_base_url: str = "https://api.fireworks.ai/inference/v1"

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
    # "full"   = all models loaded, batch_size=4, best accuracy (~1.5 GB RAM)
    # "medium" = batch_size=2, force_backend_text=True, table_mode follows config
    #            (~800-900 MB RAM). Sweet spot: 2× faster than slim, more
    #            accurate tables than slim (ACCURATE vs forced FAST). Safe at 2Gi.
    # "slim"   = same Docling pipeline but with aggressive memory-saving knobs:
    #          batch_size=1, images_scale=0.25, table_mode=fast,
    #          force_backend_text=True (skip layout-model text extraction)
    #          Peak ~600–800 MB in subprocess, API server stays at ~200 MB.
    docling_mode: Literal["full", "medium", "slim"] = "full"

    # Layout model: "heron" (default, 42.9M params RT-DETR v2) or
    # "egret_medium" (19.5M params D-Fine, ~54 MB lighter peak RAM)
    docling_layout_model: str = "heron"
    # Pages per subprocess batch. Each batch runs in a fresh OS process so the
    # ONNX C++ heap (~1 GB) is fully freed between batches. 0 = no splitting.
    docling_pages_per_batch: int = 25

    docling_do_table_structure: bool = True
    docling_table_mode: Literal["fast", "accurate"] = "accurate"
    docling_do_cell_matching: bool = True
    docling_document_timeout: float | None = 1800.0  # seconds per document; None = no limit
    docling_num_threads: int = 1   # keep low to avoid OOM on CPU
    docling_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    # Path for local 24-h upload cache files (blank = system temp dir)
    upload_cache_dir: str = ""

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.DOCLING_HYBRID
    chunk_size: int = 1024
    chunk_overlap: int = 128
    min_chunk_chars: int = 60
    # Hard cap on chunks per document. Docling triplet-format can produce
    # 20k+ chunks from table-heavy financial PDFs; this prevents runaway
    # embedding API costs and processing time. 0 = no cap.
    max_chunks_per_document: int = 10000

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
    # ── Redis / Rate Limiting ─────────────────────────────────────────────────
    # Set RAG_REDIS_URL to a blank string to disable Redis (rate limiting becomes
    # a no-op — useful when Redis is not available in a given environment).
    redis_url: str = "redis://localhost:6379/0"
    rate_limit_docs: int = 3      # max documents a user may upload (lifetime)
    rate_limit_queries: int = 15  # max queries a user may submit (lifetime)
    # Emails listed here are seeded into the Redis SET "rl:exempt" at startup.
    # Users whose email is in that set bypass both doc-upload and query limits.
    # You can also manage the set directly: redis-cli SADD rl:exempt user@example.com
    rate_limit_exempt_emails: list[str] = []

    @field_validator("rate_limit_exempt_emails", mode="before")
    @classmethod
    def parse_exempt_emails(cls, v: object) -> object:
        """Accept comma-separated, pipe-separated, JSON array, or empty string.

        Pipe separator is required when setting via gcloud --update-env-vars
        because gcloud treats commas as argument delimiters (same as allowed_origins).
        """
        if isinstance(v, str):
            v = v.strip()
            if not v:
                return []
            if v.startswith("["):
                import json
                return [e.strip() for e in json.loads(v) if str(e).strip()]
            if "|" in v:
                return [e.strip() for e in v.split("|") if e.strip()]
            return [e.strip() for e in v.split(",") if e.strip()]
        return v
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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Replace the default EnvSettingsSource with one that tolerates
        non-JSON values for list fields (e.g. pipe-separated RAG_ALLOWED_ORIGINS).
        Without this, pydantic-settings calls json.loads() on the raw string
        *before* any field_validator runs, crashing on pipe-separated input.
        """
        return (
            init_settings,
            _PipeAwareEnvSource(settings_cls),
            _PipeAwareDotEnvSource(settings_cls),
            file_secret_settings,
        )

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
