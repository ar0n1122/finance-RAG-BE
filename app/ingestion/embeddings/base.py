"""
EmbeddingProvider protocol — the contract every text/image embedder satisfies.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for text embedding models."""

    @property
    def dimension(self) -> int:
        """Output embedding dimension."""
        ...

    @property
    def model_name(self) -> str:
        """Model identifier."""
        ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.  Returns list of float vectors."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string.  May use a different prefix/instruction."""
        ...
