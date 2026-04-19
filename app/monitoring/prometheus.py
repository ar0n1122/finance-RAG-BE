"""
Prometheus metrics — application-level counters, histograms, and gauges.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Lazy Prometheus import ────────────────────────────────────────────────────
# prometheus_client is optional — if not installed, metrics are silently no-op.

try:
    from prometheus_client import Counter, Gauge, Histogram, Info, generate_latest

    _ENABLED = True
except ImportError:
    _ENABLED = False

    class _NoopMetric:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, *args: Any, **kwargs: Any) -> "_NoopMetric":
            return self

        def inc(self, *args: Any, **kwargs: Any) -> None:
            pass

        def dec(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set(self, *args: Any, **kwargs: Any) -> None:
            pass

        def observe(self, *args: Any, **kwargs: Any) -> None:
            pass

        def info(self, *args: Any, **kwargs: Any) -> None:
            pass

    Counter = Histogram = Gauge = Info = _NoopMetric  # type: ignore[misc, assignment]

    def generate_latest() -> bytes:  # type: ignore[misc]
        return b""


# ── Request metrics ───────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "rag_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0],
)

# ── RAG pipeline metrics ─────────────────────────────────────────────────────

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval pipeline latency",
    ["strategy"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_duration_seconds",
    "LLM generation latency",
    ["model"],
    buckets=[0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
)

RERANK_LATENCY = Histogram(
    "rag_rerank_duration_seconds",
    "Reranking latency",
    [],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

QUERY_COUNT = Counter(
    "rag_queries_total",
    "Total RAG queries",
    ["strategy"],
)

QUERY_ERRORS = Counter(
    "rag_query_errors_total",
    "Total RAG query errors",
    ["strategy", "error_type"],
)

# ── Ingestion metrics ────────────────────────────────────────────────────────

INGESTION_COUNT = Counter(
    "rag_ingestions_total",
    "Total document ingestions",
    ["status"],
)

INGESTION_LATENCY = Histogram(
    "rag_ingestion_duration_seconds",
    "Document ingestion latency",
    [],
    buckets=[1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
)

CHUNKS_CREATED = Counter(
    "rag_chunks_created_total",
    "Total chunks created during ingestion",
    [],
)

# ── Evaluation metrics ───────────────────────────────────────────────────────

EVAL_SCORE = Histogram(
    "rag_eval_score",
    "Evaluation scores",
    ["metric"],
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── System gauges ─────────────────────────────────────────────────────────────

ACTIVE_CONNECTIONS = Gauge(
    "rag_active_connections",
    "Active WebSocket/HTTP connections",
    [],
)

DOCUMENTS_INDEXED = Gauge(
    "rag_documents_indexed",
    "Number of indexed documents",
    [],
)

# ── Build info ────────────────────────────────────────────────────────────────

BUILD_INFO = Info(
    "rag_build",
    "RAG application build info",
)


def get_metrics() -> bytes:
    """Return Prometheus metrics in exposition format."""
    return generate_latest()
