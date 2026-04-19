"""
Docling HybridChunker — structure-aware chunking for financial reports.

**Primary chunking strategy for this project.**

Uses docling's built-in ``HybridChunker`` which:
  1. Understands the document structure via the DoclingDocument AST
  2. Respects element boundaries (tables are atomic, paragraphs are not
     split mid-sentence where possible)
  3. Merges undersized peer chunks sharing the same heading hierarchy
  4. Provides heading-chain context for each chunk automatically
  5. Serialises tables into a triplet format optimised for embedding

The chunker takes a ``DoclingDocument`` (not plain text) and produces
``Chunk`` objects with rich metadata including headings, page numbers,
chunk type (prose / table), and financial metadata extracted from the
filename.

Falls back to ``RecursiveChunker`` when docling is not installed.

Design decisions:
  - ``max_tokens=400``: raw chunks ~400 tokens.  After ``contextualize()``
    prepends heading breadcrumbs (~50-80 tokens), total stays ≤512 within
    the built-in tokenizer's limit and well within the embedding model's
    context window (8 192 for nomic, 512 for BGE).
  - ``merge_peers=True``: merges undersized sibling chunks sharing the
    same heading hierarchy to reduce chunk count without losing context.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.core.logging import get_logger
from app.models.domain import Chunk, ContentType, ParsedDocument

logger = get_logger(__name__)

# ── Financial metadata extraction ─────────────────────────────────────────────
# Examples: "Apple-2023-10K.pdf", "TCS_2024_Q1_earnings.pdf"
_FIN_PATTERN = re.compile(
    r"(?P<company>[A-Za-z]+)[_-](?P<year>\d{4})[_-](?P<type>\w+)",
    re.IGNORECASE,
)


def _parse_financial_metadata(filename: str) -> dict[str, str]:
    """Best-effort extraction of company / year / report-type from filename."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    m = _FIN_PATTERN.search(stem)
    if m:
        return {
            "company": m.group("company").title(),
            "type_of_report": m.group("type").upper(),
            "fiscal_year": m.group("year"),
        }
    return {"company": "", "type_of_report": "", "fiscal_year": ""}


def _build_context_header(doc_name: str, headings: list[str]) -> str:
    """
    Build a compact breadcrumb prefix from the heading chain.

    Examples::

        [Apple-2023-10K | PART II > Item 7. MD&A]
        [Apple-2023-10K]   (no headings)
    """
    if headings:
        return f"[{doc_name} | {' > '.join(headings)}]"
    return f"[{doc_name}]"


class DoclingHybridChunker:
    """
    Structure-aware chunker using docling's ``HybridChunker``.

    Produces chunks that:
      - Respect document boundaries (tables are atomic, paragraphs intact)
      - Include heading breadcrumbs as enriched context (for embedding)
      - Carry rich metadata: chunk_type, headings, pages, financial data
      - Are optimised for embedding and retrieval

    Falls back to ``RecursiveChunker`` when docling is not installed or
    when the ``ParsedDocument`` has no ``docling_document``.
    """

    name = "docling_hybrid"

    def __init__(
        self,
        max_tokens: int = 350,
        merge_peers: bool = True,
        min_chunk_chars: int = 60,
    ) -> None:
        self._max_tokens = max_tokens
        self._merge_peers = merge_peers
        self._min_chunk_chars = min_chunk_chars

        # Lazy-import — docling may not be installed
        try:
            from docling.chunking import HybridChunker

            self._chunker = HybridChunker(
                max_tokens=max_tokens,
                merge_peers=merge_peers,
            )
            self._available = True
        except ImportError:
            self._chunker = None
            self._available = False
            logger.warning(
                "docling_chunker_not_available",
                hint="Install docling: pip install docling docling-core",
            )

    # ── ChunkingStrategy interface ────────────────────────────────────────────

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        """
        Chunk the parsed document using docling's ``HybridChunker``.

        Falls back to ``RecursiveChunker`` when docling is not installed
        or when no ``DoclingDocument`` is present in *parsed*.
        """
        if not self._available or parsed.docling_document is None:
            logger.warning(
                "docling_chunker_fallback",
                document_id=parsed.document_id,
                reason="docling unavailable or no DoclingDocument — using recursive splitter",
            )
            from app.ingestion.chunking.recursive import RecursiveChunker

            return RecursiveChunker().chunk(parsed)

        return self._docling_chunk(parsed)

    # ── Core chunking logic ───────────────────────────────────────────────────

    def _docling_chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        """
        Run docling's HybridChunker on the DoclingDocument AST.

        Closely mirrors the proven pattern from ``docling_hybrid/chunker.py``.
        """
        from docling_core.types.doc.labels import DocItemLabel

        dl_doc = parsed.docling_document
        fin_meta = _parse_financial_metadata(parsed.filename)
        doc_name = parsed.filename.rsplit(".", 1)[0] if "." in parsed.filename else parsed.filename

        _TABLE_LABELS = {DocItemLabel.TABLE}

        chunks: list[Chunk] = []
        global_idx = 0

        for doc_chunk in self._chunker.chunk(dl_doc=dl_doc):
            raw_text = doc_chunk.text
            if len(raw_text.strip()) < self._min_chunk_chars:
                continue

            # Contextualized text (headings prepended) — what gets embedded
            enriched_text = self._chunker.contextualize(chunk=doc_chunk)

            # Extract heading chain
            headings: list[str] = doc_chunk.meta.headings or []

            # Extract page numbers from provenance
            pages: set[int] = set()
            for doc_item in doc_chunk.meta.doc_items:
                for prov in doc_item.prov:
                    pages.add(prov.page_no)

            # Determine chunk type: table if ANY doc_item is a table
            is_table = any(
                item.label in _TABLE_LABELS for item in doc_chunk.meta.doc_items
            )
            chunk_type = "table" if is_table else "prose"

            # Derive table name from deepest heading near the table
            table_name = headings[-1] if is_table and headings else ""

            # Context header breadcrumb
            context_header = _build_context_header(doc_name, headings)

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=raw_text,
                enriched_text=enriched_text,
                content_type=ContentType.TABLE if is_table else ContentType.TEXT,
                document_id=parsed.document_id,
                page=min(pages) if pages else 0,
                section=" > ".join(headings) if headings else None,
                chunk_index=global_idx,
                token_count=len(enriched_text.split()),
                metadata={
                    "source_file": parsed.filename,
                    "chunk_type": chunk_type,
                    "table_name": table_name,
                    "headings": " > ".join(headings) if headings else "",
                    "page_min": min(pages) if pages else 0,
                    "page_max": max(pages) if pages else 0,
                    "context_header": context_header,
                    **fin_meta,
                },
            ))
            global_idx += 1

        table_count = sum(1 for c in chunks if c.metadata.get("chunk_type") == "table")
        logger.info(
            "docling_chunking_complete",
            document_id=parsed.document_id,
            filename=parsed.filename,
            total_chunks=len(chunks),
            tables=table_count,
            prose=len(chunks) - table_count,
        )
        return chunks
