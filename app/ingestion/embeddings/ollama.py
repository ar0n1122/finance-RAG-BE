"""
Provider-agnostic embedding client.

Supports two wire formats via the ``api_format`` parameter:

- ``"openai"`` (default) — Standard ``POST /v1/embeddings`` format.
  Works with OpenAI, Azure OpenAI, Groq, Google Vertex AI (OpenAI-compat),
  and Ollama (which exposes ``/v1/embeddings`` alongside its native API).

- ``"ollama"`` — Ollama-native ``POST /api/embed`` format.
  Useful when hitting Ollama directly at ``http://localhost:11434``.

By defaulting to the OpenAI format the same class works with *any*
provider — just swap ``base_url`` and ``api_key``.
"""

from __future__ import annotations

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class OllamaEmbedder:
    """Provider-agnostic embedding client (OpenAI-compatible by default)."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        dimension: int = 768,
        batch_size: int = 32,
        api_format: str = "openai",
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._dimension = dimension
        self._batch_size = batch_size
        self._api_format = api_format
        self._client = httpx.Client(timeout=_TIMEOUT)

    # ── EmbeddingProvider protocol ────────────────────────────────────────────

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def model_name(self) -> str:
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            embeddings = self._call_embed(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._call_embed([text])[0]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_embed(self, texts: list[str]) -> list[list[float]]:
        if self._api_format == "ollama":
            return self._call_ollama_native(texts)
        return self._call_openai_compat(texts)

    def _call_openai_compat(self, texts: list[str]) -> list[list[float]]:
        """``POST {base_url}/v1/embeddings`` — OpenAI-compatible format."""
        url = f"{self._base_url}/v1/embeddings"
        headers = self._auth_headers()
        payload = {"model": self._model, "input": texts}

        resp = self._client.post(url, json=payload, headers=headers)
        resp.raise_for_status()

        data = resp.json()["data"]
        # Sort by index to guarantee order matches input
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def _call_ollama_native(self, texts: list[str]) -> list[list[float]]:
        """``POST {base_url}/api/embed`` — Ollama-native format."""
        url = f"{self._base_url}/api/embed"
        payload = {"model": self._model, "input": texts}

        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["embeddings"]

    def _auth_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
