"""
ChunkingStrategy protocol — the contract every chunker must satisfy.

New chunking approaches (e.g. semantic, agentic) should implement this
protocol; the rest of the pipeline works unchanged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.models.domain import Chunk, ParsedDocument


@runtime_checkable
class ChunkingStrategy(Protocol):
    """Interface for document chunking strategies."""

    @property
    def name(self) -> str:
        """Human-readable strategy name (used in config / logs)."""
        ...

    def chunk(
        self,
        parsed: ParsedDocument,
    ) -> list[Chunk]:
        """
        Split a parsed document into indexable chunks.

        Parameters
        ----------
        parsed:
            Converted document carrying a DoclingDocument AST and/or
            exported markdown text.

        Returns
        -------
        list[Chunk]
            Flat list of chunks with metadata, ready for embedding.
        """
        ...
