"""
DEPRECATED — replaced by ``document_converter.py`` which uses Docling.

This module previously used PyMuPDF + PDFPlumber for PDF extraction.
The Docling ``DocumentConverter`` now handles all PDF-to-structured-document
conversion, producing richer output (heading hierarchy, page provenance,
table detection, reading-order preservation).

If you need the legacy parser for some reason, install the ``legacy-pdf``
optional extras::

    pip install -e ".[legacy-pdf]"

Then import and use ``LegacyPDFParser`` below.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Re-export notice
__all__ = ["LegacyPDFParser"]


class LegacyPDFParser:
    """
    *Deprecated* — use ``app.ingestion.document_converter.DocumentConverter``.

    Thin wrapper kept for backward compatibility.  Requires PyMuPDF and
    pdfplumber to be installed (optional extras ``legacy-pdf``).
    """

    def __init__(self) -> None:
        warnings.warn(
            "LegacyPDFParser is deprecated. Use DocumentConverter instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def parse(self, pdf_path: Path, document_id: str = "") -> list[Any]:
        raise NotImplementedError(
            "LegacyPDFParser is deprecated. "
            "Use app.ingestion.document_converter.DocumentConverter instead."
        )
