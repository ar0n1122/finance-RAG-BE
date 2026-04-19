"""
Embedding-based reranker — provider-agnostic.

Computes embeddings for the query and each candidate via an external
API, then ranks by cosine similarity.  No model weights are loaded
in-process.

Supports two wire formats (matching ``OllamaEmbedder``):

- ``"openai"`` — ``POST {base_url}/v1/embeddings``
  (OpenAI, Azure, Groq, Vertex AI, Ollama's compat layer)
- ``"ollama"`` — ``POST {base_url}/api/embed``
  (Ollama-native)
"""

from __future__ import annotations

import math
import time

import httpx

from app.core.logging import get_logger
from app.models.domain import RetrievalResult

logger = get_logger(__name__)

_DEFAULT_MODEL = "qllama/bge-reranker-v2-m3:latest"
_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)


class Reranker:
    """Embedding-based reranker using cosine similarity (provider-agnostic)."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        base_url: str = "http://localhost:11434",
        api_key: str = "",
        api_format: str = "openai",
        **kwargs: object,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._api_format = api_format
        self._client = httpx.Client(timeout=_TIMEOUT)

    # ── Public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> tuple[list[RetrievalResult], float]:
        """
        Rerank *results* against *query* and return the top_k.

        Returns (reranked_results, elapsed_ms).
        """
        if not results:
            return [], 0.0

        t0 = time.perf_counter()

        # Embed query + all candidate texts in one batch
        texts = [query] + [r.text for r in results]
        embeddings = self._embed(texts)

        query_emb = embeddings[0]
        doc_embs = embeddings[1:]

        # Cosine similarity scoring
        raw_scores = [self._cosine(query_emb, d) for d in doc_embs]

        # Normalise scores to [0, 1]
        min_s = min(raw_scores)
        max_s = max(raw_scores)
        span = max_s - min_s if max_s != min_s else 1.0

        scored = []
        for result, raw in zip(results, raw_scores):
            normalised = (raw - min_s) / span
            scored.append(RetrievalResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                text=result.text,
                content_type=result.content_type,
                score=normalised,
                page=result.page,
                section=result.section,
                metadata=result.metadata,
            ))

        scored.sort(key=lambda r: r.score, reverse=True)
        top = scored[:top_k]

        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(
            "reranked",
            model=self._model_name,
            input=len(results),
            returned=len(top),
            ms=round(elapsed, 1),
        )
        return top, elapsed

    def warm_up(self) -> None:
        """Send a trivial embed request so the provider loads the model."""
        try:
            self._embed(["warmup"])
        except Exception:
            logger.warning("reranker_warmup_failed", model=self._model_name)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._api_format == "ollama":
            return self._embed_ollama(texts)
        return self._embed_openai(texts)

    def _embed_openai(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/v1/embeddings"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        resp = self._client.post(url, json={"model": self._model_name, "input": texts}, headers=headers)
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        url = f"{self._base_url}/api/embed"
        resp = self._client.post(url, json={"model": self._model_name, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
