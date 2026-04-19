"""
Lightweight PDF → Markdown converter using PyMuPDF (``pymupdf4llm``).

Designed for **memory-constrained servers** (≤ 1–2 GB RAM) where Docling's
ML-based pipeline (~1.5 GB model memory) cannot run.

``pymupdf4llm`` extracts:
  - Text with reading-order awareness
  - Tables as Markdown (pipe-delimited)
  - Headings inferred from font size / bold
  - Page numbers

It does **not** use ML models — purely algorithmic extraction,
so RAM usage stays at ~50–100 MB even for 200-page PDFs.

Trade-offs vs Docling:
  ✅ 10–20× less RAM
  ✅ 3–5× faster
  ✅ No model download / cold start
  ❌ Table detection is heuristic (font/spacing), not ML-based
  ❌ No heading-hierarchy AST — headings inferred from font size
  ❌ Complex multi-column layouts may not preserve reading order

For financial reports (10-K, 10-Q, annual reports), the accuracy is
still very good because these documents are well-structured with
consistent formatting.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.models.domain import ParsedDocument

logger = get_logger(__name__)


class LiteConverter:
    """
    Lightweight PDF → Markdown converter using ``pymupdf4llm``.

    Drop-in replacement for ``DocumentConverter`` on memory-constrained
    servers.  Produces a ``ParsedDocument`` with markdown text and page
    count — compatible with ``RecursiveChunker``.

    Set ``RAG_DOCLING_MODE=lite`` to enable this instead of Docling.
    """

    def __init__(self) -> None:
        try:
            import pymupdf4llm  # noqa: F401
            import pymupdf  # noqa: F401

            self._available = True
            logger.info("lite_converter_ready", backend="pymupdf4llm")
        except ImportError:
            self._available = False
            logger.warning(
                "pymupdf4llm_not_installed",
                hint="Install: pip install pymupdf4llm",
            )

    @property
    def is_available(self) -> bool:
        return self._available

    def convert(
        self,
        source: Path | bytes,
        *,
        document_id: str = "",
        filename: str = "",
    ) -> ParsedDocument:
        """
        Convert a PDF to Markdown using pymupdf4llm.

        Parameters
        ----------
        source :
            File path or raw PDF bytes.
        document_id :
            Unique document identifier.
        filename :
            Original upload filename.

        Returns
        -------
        ParsedDocument
            Contains markdown, page count, and metadata.
            ``docling_document`` is always ``None`` (no AST available).
        """
        if not self._available:
            raise IngestionError(
                "pymupdf4llm is not installed. "
                "Install it: pip install pymupdf4llm"
            )

        import pymupdf
        import pymupdf4llm

        try:
            # Open the PDF
            if isinstance(source, bytes):
                doc = pymupdf.open(stream=source, filetype="pdf")
            else:
                p = Path(source)
                if not p.exists():
                    raise IngestionError(f"File not found: {p}")
                doc = pymupdf.open(str(p))

            total_pages = len(doc)
            doc.close()

            # Extract markdown — pymupdf4llm handles tables, headings, text
            if isinstance(source, bytes):
                # pymupdf4llm needs a file path or pymupdf.Document
                import tempfile

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(source)
                    tmp_path = tmp.name

                try:
                    markdown = pymupdf4llm.to_markdown(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            else:
                markdown = pymupdf4llm.to_markdown(str(source))

            file_hash = hashlib.sha256(
                source if isinstance(source, bytes) else Path(source).read_bytes()
            ).hexdigest()

            logger.info(
                "document_converted",
                document_id=document_id,
                filename=filename,
                pages=total_pages,
                markdown_chars=len(markdown),
                converter="pymupdf4llm",
            )

            return ParsedDocument(
                document_id=document_id,
                filename=filename or (source.name if isinstance(source, Path) else "upload.pdf"),
                markdown=markdown,
                total_pages=total_pages,
                docling_document=None,  # No AST — chunking falls back to RecursiveChunker
                metadata={
                    "converter": "pymupdf4llm",
                    "conversion_status": "success",
                    "file_hash": file_hash,
                },
            )

        except IngestionError:
            raise
        except Exception as exc:
            logger.error(
                "lite_conversion_failed",
                document_id=document_id,
                filename=filename,
                error=str(exc),
            )
            raise IngestionError(
                f"PDF conversion failed for '{filename}': {exc}"
            ) from exc
