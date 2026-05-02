"""
FastAPI dependency injection — service factories and shared singletons.

All heavy services (Qdrant client, embedders, retrievers, RAG strategies)
are created once and cached for the application lifetime.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

from qdrant_client import QdrantClient

from app.core.config import get_settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.evaluation.ragas_eval import RAGASEvaluator
    from app.generation.pipeline import GenerationPipeline
    from app.guardrails.input_guard import InputGuard
    from app.guardrails.output_guard import OutputGuard
    from app.ingestion.embeddings.base import EmbeddingProvider
    from app.ingestion.pipeline import IngestionPipeline
    from app.monitoring.tracing import TracingProvider
    from app.rag.strategies import RAGStrategy, RAGStrategyRegistry
    from app.retrieval.hybrid import HybridRetriever
    from app.storage.cloud_storage import CloudStorageClient
    from app.storage.firestore import FirestoreClient
    from app.storage.redis_client import RedisClient

logger = get_logger(__name__)


# ── Singletons ────────────────────────────────────────────────────────────────


@functools.lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()
    host = settings.qdrant_host
    # If the host is a full URL (cloud-hosted), use the url= parameter.
    # If it's a bare hostname, use host/port/https separately.
    if host.startswith("http://") or host.startswith("https://"):
        client = QdrantClient(
            url=host,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
        )
    else:
        client = QdrantClient(
            host=host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key or None,
            https=settings.qdrant_use_https,
            timeout=settings.qdrant_timeout,
        )
    logger.info("qdrant_connected", host=host, port=settings.qdrant_port)
    return client


@functools.lru_cache
def get_firestore_client() -> "FirestoreClient":
    # from app.storage.firestore import FirestoreClient
    # return FirestoreClient()
    
    from google.cloud import firestore as gcp_firestore
    from app.storage.firestore import FirestoreClient

    settings = get_settings()

    client = gcp_firestore.Client(
        project=settings.gcp_project_id,
        database=settings.firestore_database,
    )
    return FirestoreClient(client=client)


@functools.lru_cache
def get_cloud_storage_client() -> "CloudStorageClient":
    from google.cloud import storage as gcp_storage
    from app.storage.cloud_storage import CloudStorageClient

    settings = get_settings()
    client = gcp_storage.Client(project=settings.gcp_project_id or None)
    return CloudStorageClient(client=client, bucket_name=settings.gcs_bucket)


@functools.lru_cache
def get_text_embedder() -> "EmbeddingProvider":
    settings = get_settings()
    from app.ingestion.embeddings.ollama import OllamaEmbedder

    if settings.embedding_provider == "openrouter":
        # OpenRouter exposes /v1/embeddings — strip /v1 from base URL since
        # OllamaEmbedder already appends /v1/embeddings in openai format.
        base = settings.openrouter_base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        return OllamaEmbedder(
            model=settings.openrouter_embedding_model,
            base_url=base,
            api_key=settings.openrouter_api_key,
            dimension=settings.embedding_dim,
            batch_size=settings.embedding_batch_size,
            api_format="openai",
        )

    return OllamaEmbedder(
        model=settings.embedding_api_model,
        base_url=settings.embedding_api_base_url or settings.ollama_base_url,
        api_key=settings.embedding_api_key,
        dimension=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
        api_format=settings.embedding_api_format,
    )


def _build_openrouter_embedding_base_url() -> str:
    """Return the OpenRouter base URL suitable for embedding/reranker clients."""
    settings = get_settings()
    base = settings.openrouter_base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base


@functools.lru_cache
def get_hybrid_retriever() -> "HybridRetriever":
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.reranker import Reranker

    settings = get_settings()

    reranker = None
    if settings.reranker_enabled:
        if settings.embedding_provider == "openrouter":
            base = _build_openrouter_embedding_base_url()
            reranker = Reranker(
                model_name=settings.reranker_model or settings.openrouter_embedding_model,
                base_url=settings.reranker_api_base_url or base,
                api_key=settings.reranker_api_key or settings.openrouter_api_key,
                api_format="openai",
            )

    return HybridRetriever(
        client=get_qdrant_client(),
        embedder=get_text_embedder(),
        reranker=reranker,
    )


@functools.lru_cache
def get_generation_pipeline() -> "GenerationPipeline":
    from app.generation.fireworks_provider import FireworksProvider
    from app.generation.ollama_provider import OllamaProvider
    from app.generation.openai_provider import OpenAIProvider
    from app.generation.openrouter_provider import OpenRouterProvider
    from app.generation.pipeline import GenerationPipeline

    settings = get_settings()

    # ── Primary provider ──────────────────────────────────────────────────
    if settings.llm_provider == "fireworks":
        primary = FireworksProvider()
    elif settings.llm_provider == "openrouter":
        primary = OpenRouterProvider()
    elif settings.llm_provider == "openai":
        primary = OpenAIProvider(
            model=settings.openai_model,
            base_url="https://api.openai.com/v1",
            api_key=settings.openai_api_key,
        )
    else:
        primary = OllamaProvider()

    # ── Fallback provider ─────────────────────────────────────────────────
    fallback = None
    if settings.llm_fallback_provider == "openrouter":
        fallback = OpenRouterProvider()
    elif settings.llm_fallback_provider == "fireworks":
        fallback = FireworksProvider()
    elif settings.llm_fallback_provider == "openai":
        fallback = OpenAIProvider()
    elif settings.llm_fallback_provider == "ollama":
        fallback = OllamaProvider()
    elif settings.llm_fallback_model:
        # Legacy: fallback model set without explicit provider
        fallback = OpenAIProvider()

    return GenerationPipeline(primary=primary, fallback=fallback)


@functools.lru_cache
def get_ingestion_pipeline() -> "IngestionPipeline":
    from app.ingestion.document_converter import DocumentConverter
    from app.ingestion.pipeline import IngestionPipeline

    settings = get_settings()

    # Select chunker based on config
    if settings.chunking_strategy.value == "docling_hybrid":
        from app.ingestion.chunking.docling_hybrid import DoclingHybridChunker

        chunker = DoclingHybridChunker(
            max_tokens=settings.chunk_size,
            min_chunk_chars=settings.min_chunk_chars,
        )
    else:
        from app.ingestion.chunking.recursive import RecursiveChunker

        chunker = RecursiveChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_chunk_chars=settings.min_chunk_chars,
        )

    return IngestionPipeline(
        settings=settings,
        converter=DocumentConverter(),
        chunker=chunker,
        text_embedder=get_text_embedder(),
        qdrant=get_qdrant_client(),
        firestore=get_firestore_client(),
        gcs=get_cloud_storage_client(),
    )


@functools.lru_cache
def get_rag_registry() -> "RAGStrategyRegistry":
    from app.rag.strategies import RAGStrategyRegistry
    from app.rag.strategies.adaptive import AdaptiveRAG
    from app.rag.strategies.agentic import AgenticRAG
    from app.rag.strategies.basic import BasicRAG
    from app.rag.strategies.self_correcting import SelfCorrectingRAG

    retriever = get_hybrid_retriever()
    gen_pipeline = get_generation_pipeline()

    registry = RAGStrategyRegistry()
    registry.register(BasicRAG(retriever, gen_pipeline))
    registry.register(SelfCorrectingRAG(retriever, gen_pipeline))
    registry.register(AgenticRAG(retriever, gen_pipeline))
    registry.register(AdaptiveRAG(retriever, gen_pipeline))
    return registry


@functools.lru_cache
def get_input_guard() -> "InputGuard":
    from app.guardrails.input_guard import InputGuard

    return InputGuard()


@functools.lru_cache
def get_output_guard() -> "OutputGuard":
    from app.guardrails.output_guard import OutputGuard

    return OutputGuard()


@functools.lru_cache
def get_tracing_provider() -> "TracingProvider":
    settings = get_settings()
    if settings.langfuse_public_key and settings.langfuse_secret_key:
        from app.monitoring.langfuse_provider import LangfuseProvider

        return LangfuseProvider()
    else:
        from app.monitoring.langfuse_provider import NoopProvider

        return NoopProvider()


@functools.lru_cache
def get_ragas_evaluator() -> "RAGASEvaluator":
    from app.evaluation.ragas_eval import RAGASEvaluator

    return RAGASEvaluator()


@functools.lru_cache
def get_redis_client() -> "RedisClient":
    """Return the shared async Redis client (fail-safe: no-op if URL is blank)."""
    from app.storage.redis_client import RedisClient

    settings = get_settings()
    # A blank redis_url disables Redis — callers must handle RedisUnavailableError.
    url = settings.redis_url or "redis://localhost:6379/0"
    client = RedisClient(url=url)
    logger.info("redis_client_created", url=url)
    return client


# ── Health verification ───────────────────────────────────────────────────────


def verify_services() -> dict[str, bool]:
    """Check connectivity of all external services."""
    status: dict[str, bool] = {}

    # Qdrant
    try:
        client = get_qdrant_client()
        client.get_collections()
        status["qdrant"] = True
    except Exception:
        status["qdrant"] = False

    # Firestore
    try:
        fs = get_firestore_client()
        # Light-weight check
        status["firestore"] = True
    except Exception:
        status["firestore"] = False

    # GCS
    try:
        gcs = get_cloud_storage_client()
        status["cloud_storage"] = True
    except Exception:
        status["cloud_storage"] = False

    # LLM provider
    try:
        pipeline = get_generation_pipeline()
        status["llm"] = pipeline._primary.health_check()
    except Exception:
        status["llm"] = False

    return status
