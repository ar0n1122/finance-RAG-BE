"""
Recursive character text-splitter — **fallback** chunking strategy.

Used when docling is not available.  Operates on the exported markdown
text from document conversion (or raw text for non-PDF files).

For the primary strategy, see ``docling_hybrid.py`` which uses docling's
structure-aware ``HybridChunker``.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from app.core.logging import get_logger
from app.models.domain import Chunk, ContentType, ParsedDocument

logger = get_logger(__name__)

# Financial-report filename pattern
# Examples: "Apple-2023-10K.pdf", "TCS_2024_annual_report.pdf"
_FIN_PATTERN = re.compile(
    r"(?P<company>[A-Za-z]+)[_-](?P<year>\d{4})[_-](?P<type>\w+)",
    re.IGNORECASE,
)


def _parse_financial_metadata(filename: str) -> dict[str, str]:
    """Best-effort extraction of company/year/type from filename stem."""
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    m = _FIN_PATTERN.search(stem)
    if m:
        return {
            "company": m.group("company").title(),
            "type_of_report": m.group("type").upper(),
            "fiscal_year": m.group("year"),
        }
    return {"company": "", "type_of_report": "", "fiscal_year": ""}


# Heuristic: a chunk that has 2+ lines starting with "|" is likely a markdown table
_TABLE_LINE_RE = re.compile(r"^\s*\|", re.MULTILINE)


class RecursiveChunker:
    """
    Recursive character text-splitter with configurable separators.

    Fallback chunker that works on plain markdown / text when docling's
    ``HybridChunker`` is not available.  Tables in markdown format are
    detected and treated as atomic units (never split).
    """

    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_chars: int = 60,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars
        self.separators = separators or ["\n\n\n", "\n\n", "\n", ". ", " ", ""]

    # ── ChunkingStrategy interface ────────────────────────────────────────────

    def chunk(self, parsed: ParsedDocument) -> list[Chunk]:
        """
        Split the parsed document's markdown text into chunks.

        Parameters
        ----------
        parsed :
            The ``ParsedDocument`` from document conversion.  Uses
            ``parsed.markdown`` as the input text.
        """
        text = parsed.markdown
        if not text or not text.strip():
            return []

        fin_meta = _parse_financial_metadata(parsed.filename)
        splits = self._recursive_split(text)
        chunks: list[Chunk] = []

        for idx, split in enumerate(splits):
            stripped = split.strip()
            if len(stripped) < self.min_chunk_chars:
                continue

            is_table = self._looks_like_table(stripped)

            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=stripped,
                content_type=ContentType.TABLE if is_table else ContentType.TEXT,
                document_id=parsed.document_id,
                page=0,  # page info not available from plain text
                section=self._detect_section(stripped),
                chunk_index=idx,
                token_count=len(stripped.split()),
                metadata={
                    "source_file": parsed.filename,
                    "chunk_type": "table" if is_table else "prose",
                    **fin_meta,
                },
            ))

        logger.info(
            "recursive_chunking_complete",
            document_id=parsed.document_id,
            total_chunks=len(chunks),
            tables=sum(1 for c in chunks if c.metadata.get("chunk_type") == "table"),
        )
        return chunks

    # ── Splitting logic ───────────────────────────────────────────────────────

    def _recursive_split(self, text: str) -> list[str]:
        """Split *text* recursively by separator hierarchy with overlap."""
        return self._split(text, list(self.separators))

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            return self._hard_split(text)

        sep = separators[0]
        remaining_seps = separators[1:]

        if not sep:
            return self._hard_split(text)

        parts = text.split(sep)
        merged: list[str] = []
        current = ""

        for part in parts:
            candidate = f"{current}{sep}{part}" if current else part
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(part) > self.chunk_size:
                    merged.extend(self._split(part, remaining_seps))
                    current = ""
                else:
                    current = part

        if current:
            merged.append(current)

        # Apply overlap
        if self.chunk_overlap > 0 and len(merged) > 1:
            merged = self._apply_overlap(merged)

        return merged

    def _hard_split(self, text: str) -> list[str]:
        """Character-level split as a last resort."""
        result: list[str] = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i : i + self.chunk_size]
            if chunk.strip():
                result.append(chunk)
        return result

    def _apply_overlap(self, parts: list[str]) -> list[str]:
        """Prepend overlap from the previous chunk to each subsequent chunk."""
        result = [parts[0]]
        for i in range(1, len(parts)):
            prev = parts[i - 1]
            overlap_text = prev[-self.chunk_overlap :] if len(prev) > self.chunk_overlap else prev
            result.append(overlap_text + parts[i])
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _detect_section(text: str) -> str | None:
        """Heuristic: first line if it's short and looks like a heading."""
        first_line = text.strip().split("\n")[0].strip()
        if 5 < len(first_line) < 120 and not first_line.endswith("."):
            return first_line
        return None

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        """Heuristic: 2+ lines starting with '|' indicates a markdown table."""
        return len(_TABLE_LINE_RE.findall(text)) >= 2
