"""
BGE-compatible embedding provider via Ollama.

Delegates to an Ollama instance.  By default uses ``nomic-embed-text``
(768-dim) since BGE-large is not natively available in Ollama.
Swap in ``mxbai-embed-large`` (1024-dim) for a closer equivalent.
"""

from __future__ import annotations

from app.ingestion.embeddings.ollama import OllamaEmbedder


class BGEEmbedder(OllamaEmbedder):
    """BGE-equivalent embeddings via Ollama."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        dimension: int = 768,
        batch_size: int = 32,
        **kwargs: object,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            dimension=dimension,
            batch_size=batch_size,
        )
