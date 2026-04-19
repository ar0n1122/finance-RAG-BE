"""
Langfuse tracing provider implementation (Langfuse SDK v4).
"""

from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.monitoring.tracing import TracingProvider

logger = get_logger(__name__)


class LangfuseProvider(TracingProvider):
    """
    Langfuse-based tracing for LLM observability (v4 SDK).

    Falls back to NoopProvider if Langfuse is not configured or unavailable.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client: Any = None
        self._enabled = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

        if self._enabled:
            try:
                from langfuse import Langfuse

                self._client = Langfuse(
                    public_key=settings.langfuse_public_key,
                    secret_key=settings.langfuse_secret_key,
                    host=settings.langfuse_host,
                )
                logger.info("langfuse_connected", host=settings.langfuse_host)
            except Exception as exc:
                logger.warning("langfuse_init_failed", error=str(exc))
                self._enabled = False

    def trace(
        self,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> Any:
        if not self._enabled or not self._client:
            return _NoopTrace()
        try:
            span = self._client.start_observation(
                name=name,
                as_type="span",
                input=input,
                metadata={
                    **(metadata or {}),
                    **({"user_id": user_id} if user_id else {}),
                    **({"session_id": session_id} if session_id else {}),
                } or None,
            )
            return _LangfuseTrace(span)
        except Exception as exc:
            logger.warning("langfuse_trace_failed", error=str(exc))
            return _NoopTrace()

    def span(
        self,
        trace_id: str,
        name: str,
        *,
        input: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        if not self._enabled or not self._client:
            return _NoopSpan()
        try:
            span = self._client.start_observation(
                name=name,
                as_type="span",
                input=input,
                metadata=metadata,
            )
            return span
        except Exception as exc:
            logger.warning("langfuse_span_failed", error=str(exc))
            return _NoopSpan()

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
        if not self._enabled or not self._client:
            return None
        try:
            gen = self._client.start_observation(
                name=name,
                as_type="generation",
                model=model,
                input=input,
                metadata=metadata,
            )
            usage_details = {}
            if usage:
                usage_details = {
                    "input": usage.get("prompt_tokens", 0),
                    "output": usage.get("completion_tokens", 0),
                }
            gen.update(output=output, usage_details=usage_details or None)
            gen.end()
            return gen
        except Exception as exc:
            logger.warning("langfuse_generation_failed", error=str(exc))
            return None

    def score(
        self,
        trace_id: str,
        name: str,
        value: float,
        *,
        comment: str = "",
    ) -> None:
        if not self._enabled or not self._client:
            return
        try:
            self._client.create_score(
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
        except Exception as exc:
            logger.warning("langfuse_score_failed", error=str(exc))

    def flush(self) -> None:
        if self._client:
            try:
                self._client.flush()
            except Exception:
                pass


class _LangfuseTrace:
    """Wrapper around a Langfuse v4 span that exposes a trace-like interface."""

    def __init__(self, span: Any) -> None:
        self._span = span

    @property
    def id(self) -> str:
        return self._span.trace_id

    def update(self, **kwargs: Any) -> None:
        try:
            self._span.update(**kwargs)
        except Exception:
            pass

    def end(self) -> None:
        try:
            self._span.end()
        except Exception:
            pass


class NoopProvider(TracingProvider):
    """No-op tracing provider — used when tracing is disabled."""

    def trace(self, name: str, **kwargs: Any) -> Any:
        return _NoopTrace()

    def span(self, trace_id: str, name: str, **kwargs: Any) -> Any:
        return _NoopSpan()

    def generation(self, trace_id: str, name: str, **kwargs: Any) -> Any:
        return None

    def score(self, trace_id: str, name: str, value: float, **kwargs: Any) -> None:
        pass

    def flush(self) -> None:
        pass


class _NoopTrace:
    """Placeholder trace object."""

    id = "noop"

    def span(self, **kwargs: Any) -> "_NoopSpan":
        return _NoopSpan()

    def generation(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass

    def end(self) -> None:
        pass


class _NoopSpan:
    """Placeholder span object."""

    def end(self, **kwargs: Any) -> None:
        pass

    def update(self, **kwargs: Any) -> None:
        pass
