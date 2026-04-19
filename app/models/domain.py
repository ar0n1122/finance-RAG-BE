"""
Internal domain models — not exposed in the API, used across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────────────────


class ContentType(str, Enum):
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


# ── Ingestion ─────────────────────────────────────────────────────────────────


@dataclass
class ParsedDocument:
    """Result from document conversion (Docling or fallback).

    Carries the DoclingDocument AST (when available) plus exported
    markdown and file-level metadata.  Passed to chunking strategies.
    """

    document_id: str
    filename: str
    markdown: str
    total_pages: int
    docling_document: Any = None  # DoclingDocument when available
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedContent:
    """A single content block extracted from a PDF page (legacy)."""

    content: str | bytes
    content_type: ContentType
    page: int
    section: str | None = None
    document_id: str = ""
    source_file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk ready for embedding and indexing."""

    chunk_id: str
    text: str
    content_type: ContentType
    document_id: str
    page: int
    section: str | None
    chunk_index: int
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional enriched text (e.g. with heading breadcrumbs prepended)
    enriched_text: str | None = None


@dataclass
class IngestionResult:
    """Summary returned after ingesting a single document."""

    document_id: str
    status: DocumentStatus
    total_chunks: int = 0
    total_pages: int = 0
    tables_found: int = 0
    vectors_indexed: int = 0
    processing_time_s: float = 0.0
    error: str | None = None


# ── Retrieval ─────────────────────────────────────────────────────────────────


@dataclass
class RetrievalResult:
    """A single retrieval hit with score."""

    chunk_id: str
    document_id: str
    text: str
    content_type: ContentType
    score: float
    page: int
    section: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalOutput:
    """Aggregate retrieval output with timing info."""

    results: list[RetrievalResult] = field(default_factory=list)
    sources: list["Source"] = field(default_factory=list)
    image_results: list[RetrievalResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    total_dense_candidates: int = 0
    total_sparse_candidates: int = 0


# ── Generation ────────────────────────────────────────────────────────────────


@dataclass
class Source:
    """Citation source attached to a generated answer."""

    document_id: str
    document_title: str
    filename: str
    page: int
    section: str
    text_preview: str
    score: float
    modality: str = "text"  # text | image | table


@dataclass
class GenerationOutput:
    """LLM generation result."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_latency_ms: float = 0.0
    citations: list[str] = field(default_factory=list)


# ── RAG ───────────────────────────────────────────────────────────────────────


@dataclass
class RAGResult:
    """Final output from a RAG strategy execution."""

    answer: str
    sources: list[Source] = field(default_factory=list)
    model: str = ""
    strategy: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    evaluation: dict[str, float] | None = None
    guardrail_flags: list[str] = field(default_factory=list)
    retries: int = 0
    citations: list[str] = field(default_factory=list)


# ── User ──────────────────────────────────────────────────────────────────────


@dataclass
class User:
    """Authenticated user profile (from Firestore)."""

    user_id: str
    email: str
    display_name: str = ""
    avatar_url: str = ""
    role: str = "user"

    @property
    def id(self) -> str:
        return self.user_id
    created_at: datetime | None = None
    last_login: datetime | None = None
    settings: dict[str, Any] = field(default_factory=dict)
