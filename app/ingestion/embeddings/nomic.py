"""
nomic-embed-text embedding provider via Ollama (768-dim).

Delegates to an Ollama instance running ``nomic-embed-text``.
No in-process model loading — all inference happens in Ollama.
"""

from __future__ import annotations

from app.ingestion.embeddings.ollama import OllamaEmbedder


class NomicEmbedder(OllamaEmbedder):
    """nomic-embed-text via Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        **kwargs: object,
    ) -> None:
        super().__init__(
            model="nomic-embed-text:latest",
            base_url=base_url,
            dimension=768,
            batch_size=batch_size,
        )
