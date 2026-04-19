"""
LLM generation pipeline — retry, fallback, and citation extraction.
"""

from __future__ import annotations

import re
import time
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.config import get_settings
from app.core.exceptions import GenerationError, ServiceUnavailableError
from app.core.logging import get_logger
from app.generation.base import LLMProvider, LLMResponse
from app.generation.prompts import PromptManager
from app.models.domain import GenerationOutput, RetrievalResult, Source
from app.monitoring.cost_tracker import (
    OPERATION_GENERATION,
    get_current_tracker,
)

logger = get_logger(__name__)

_CITATION_RE = re.compile(r"\[Source\s*(\d+)\]", re.IGNORECASE)


class GenerationPipeline:
    """
    Orchestrates prompt formatting → LLM generation → citation extraction.

    Features:
    - Automatic retry with exponential backoff on transient failures
    - Fallback to secondary provider when primary is unavailable
    - Citation extraction and source linking
    """

    def __init__(
        self,
        primary: LLMProvider,
        *,
        fallback: LLMProvider | None = None,
        prompt_manager: PromptManager | None = None,
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        self._prompts = prompt_manager or PromptManager()

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        context_chunks: list[RetrievalResult],
        *,
        prompt_name: str = "qa_v1",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        document_context: str = "",
    ) -> GenerationOutput:
        """
        Generate an answer from context chunks with retry + fallback.

        Returns GenerationOutput with the answer, cited sources, model info,
        and latency.
        """
        t0 = time.perf_counter()

        # Build context string with source numbering
        context_str = self._format_context(context_chunks)

        # Get prompt template
        template = self._prompts.get(prompt_name)
        fmt_kwargs: dict[str, str] = {
            "context": context_str,
            "question": query,
        }
        if "document_context" in template.required_vars:
            fmt_kwargs["document_context"] = document_context
        system_prompt, user_prompt = template.format(**fmt_kwargs)

        # Generate with retry + fallback
        response = self._generate_with_fallback(
            user_prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            operation=OPERATION_GENERATION,
        )

        # Extract citations and map to sources
        cited_sources = self._extract_citations(response.text, context_chunks)

        elapsed = (time.perf_counter() - t0) * 1000

        return GenerationOutput(
            answer=response.text,
            sources=cited_sources,
            model=f"{response.provider}/{response.model}",
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            generation_latency_ms=elapsed,
        )

    def generate_raw(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        operation: str = "raw",
    ) -> LLMResponse:
        """
        Raw generation without context formatting — used by RAG nodes
        for grading, query rewriting, reflection, etc.
        """
        return self._generate_with_fallback(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            operation=operation,
        )

    # ── Retry + Fallback ──────────────────────────────────────────────────────

    def _generate_with_fallback(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        operation: str = "raw",
    ) -> LLMResponse:
        """Try primary with retries, then fall back to secondary provider."""
        try:
            return self._call_with_retry(
                self._primary, prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                operation=operation,
            )
        except (GenerationError, ServiceUnavailableError) as primary_err:
            if self._fallback is None:
                raise
            logger.warning(
                "llm_fallback",
                primary=self._primary.provider_name,
                fallback=self._fallback.provider_name,
                error=str(primary_err),
            )
            try:
                return self._call_with_retry(
                    self._fallback, prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    operation=operation,
                )
            except Exception as fallback_err:
                raise GenerationError(
                    f"Both providers failed. Primary: {primary_err}. Fallback: {fallback_err}"
                ) from fallback_err

    @retry(
        retry=retry_if_exception_type((GenerationError, ServiceUnavailableError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    def _call_with_retry(
        self,
        provider: LLMProvider,
        prompt: str,
        *,
        system_prompt: str | None,
        temperature: float,
        max_tokens: int,
        operation: str = "raw",
    ) -> LLMResponse:
        response = provider.generate(
            prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Record usage in the per-request cost tracker (if active)
        tracker = get_current_tracker()
        if tracker is not None:
            tracker.record_llm(
                operation=operation,
                provider=response.provider,
                model=response.model,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        return response

    # ── Context + Citation helpers ────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: list[RetrievalResult]) -> str:
        """Format chunks into numbered source blocks for the prompt."""
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            header = f"[Source {i}]"
            if chunk.section:
                header += f" (Section: {chunk.section})"
            if chunk.page:
                header += f" (Page {chunk.page})"
            parts.append(f"{header}\n{chunk.text}")
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_citations(
        answer: str,
        context_chunks: list[RetrievalResult],
    ) -> list[Source]:
        """Extract [Source N] citations from the answer and resolve to Source objects."""
        cited_indices = set()
        for match in _CITATION_RE.finditer(answer):
            idx = int(match.group(1)) - 1  # Convert to 0-based
            if 0 <= idx < len(context_chunks):
                cited_indices.add(idx)

        sources: list[Source] = []
        for idx in sorted(cited_indices):
            chunk = context_chunks[idx]
            meta = chunk.metadata or {}
            sources.append(Source(
                document_id=chunk.document_id,
                document_title=meta.get("title", ""),
                filename=meta.get("filename", ""),
                page=chunk.page,
                section=chunk.section or "",
                text_preview=chunk.text[:500],
                score=chunk.score,
                modality=chunk.content_type.value if hasattr(chunk.content_type, "value") else str(chunk.content_type),
            ))
        return sources
