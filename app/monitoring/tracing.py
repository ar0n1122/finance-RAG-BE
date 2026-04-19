"""
Tracing provider abstraction — agnostic interface for LLM observability.

Supports Langfuse (default) with easy swap to LangSmith or custom backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator


class TracingProvider(ABC):
    """Protocol for LLM call tracing / observability."""

    @abstractmethod
    def trace(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        """Create a new trace."""
        ...

    @abstractmethod
    def span(
        self,
        trace_id: str,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Create a span within a trace."""
        ...

    @abstractmethod
    def generation(
        self,
        trace_id: str,
        name: str,
        *,
        model: str = "",
        input: str = "",
        output: str = "",
        usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Record an LLM generation event."""
        ...

    @abstractmethod
    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        *,
        comment: str = "",
    ) -> None:
        """Attach a score to a trace."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending traces."""
        ...
