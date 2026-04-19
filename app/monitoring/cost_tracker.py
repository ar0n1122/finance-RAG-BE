"""
Token usage & cost tracking — per-request accumulation with accurate pricing.

Uses contextvars so all LLM calls within a single request (generation,
grading, reflection, rewriting, document resolution) are automatically
captured and attributed to the requesting user.
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


# ── Pricing table (USD per 1 M tokens) ───────────────────────────────────────
# Format: {provider: {model_substring: {"input": $, "output": $}}}
# "default" key is used when no model-specific price is found.

PRICING: dict[str, dict[str, dict[str, float]]] = {
    "ollama": {
        "default": {"input": 0.0, "output": 0.0},
    },
    "openrouter": {
        "default": {"input": 0.0, "output": 0.0},
        # Free-tier models
        "qwen/qwen3.6-plus:free": {"input": 0.0, "output": 0.0},
        "google/gemma-3-27b-it:free": {"input": 0.0, "output": 0.0},
        "deepseek/deepseek-r1:free": {"input": 0.0, "output": 0.0},
        "meta-llama/llama-4-maverick:free": {"input": 0.0, "output": 0.0},
        # Paid models
        "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "openai/gpt-4o": {"input": 2.50, "output": 10.00},
        "openai/gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "openai/gpt-4.1": {"input": 2.00, "output": 8.00},
        "anthropic/claude-sonnet-4": {"input": 3.00, "output": 15.00},
        "anthropic/claude-3.5-sonnet": {"input": 3.00, "output": 15.00},
        "anthropic/claude-3-haiku": {"input": 0.25, "output": 1.25},
        "google/gemini-2.0-flash-001": {"input": 0.10, "output": 0.40},
        "google/gemini-2.5-pro-preview": {"input": 1.25, "output": 10.00},
        "deepseek/deepseek-r1": {"input": 0.55, "output": 2.19},
        "deepseek/deepseek-chat-v3-0324": {"input": 0.27, "output": 1.10},
        "meta-llama/llama-4-maverick": {"input": 0.20, "output": 0.60},
        "qwen/qwen3-235b-a22b": {"input": 0.20, "output": 0.60},
        # Embedding models
        "nvidia/llama-nemotron-embed-vl-1b-v2:free": {"input": 0.0, "output": 0.0},
    },
    "openai": {
        "default": {"input": 0.15, "output": 0.60},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
        "gpt-4.1": {"input": 2.00, "output": 8.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "text-embedding-3-small": {"input": 0.02, "output": 0.0},
        "text-embedding-3-large": {"input": 0.13, "output": 0.0},
    },
}


def _lookup_price(provider: str, model: str) -> dict[str, float]:
    """Find the best matching price for a provider/model pair."""
    provider_prices = PRICING.get(provider.lower(), {})
    # Exact match
    if model in provider_prices:
        return provider_prices[model]
    # Substring match (e.g. "openai/gpt-4o-mini" matches "gpt-4o-mini")
    model_lower = model.lower()
    for key, price in provider_prices.items():
        if key != "default" and key.lower() in model_lower:
            return price
    # Reverse: check if model contains any pricing key
    for key, price in provider_prices.items():
        if key != "default" and model_lower in key.lower():
            return price
    return provider_prices.get("default", {"input": 0.0, "output": 0.0})


def compute_cost(
    provider: str, model: str, prompt_tokens: int, completion_tokens: int,
) -> tuple[float, float, float]:
    """Return (input_cost, output_cost, total_cost) in USD."""
    price = _lookup_price(provider, model)
    input_cost = (prompt_tokens / 1_000_000) * price["input"]
    output_cost = (completion_tokens / 1_000_000) * price["output"]
    return input_cost, output_cost, input_cost + output_cost


# ── Usage event ───────────────────────────────────────────────────────────────

# Operation types for fine-grained breakdown
OPERATION_GENERATION = "generation"
OPERATION_GRADING = "grading"
OPERATION_REFLECTION = "reflection"
OPERATION_REWRITE = "query_rewrite"
OPERATION_DOC_RESOLVE = "document_resolve"
OPERATION_EMBEDDING = "embedding"
OPERATION_RERANK = "rerank"


@dataclass
class UsageEvent:
    """A single LLM / embedding call with token counts and cost."""

    operation: str  # e.g. "generation", "grading", "reflection"
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_total: float = 0.0
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cost_input": self.cost_input,
            "cost_output": self.cost_output,
            "cost_total": self.cost_total,
            "timestamp": self.timestamp,
        }


# ── Request-scoped cost tracker ──────────────────────────────────────────────

_current_tracker: ContextVar["CostTracker | None"] = ContextVar(
    "cost_tracker", default=None,
)


def get_current_tracker() -> "CostTracker | None":
    """Return the active per-request tracker (if any)."""
    return _current_tracker.get()


class CostTracker:
    """
    Accumulates token usage and costs across all LLM/embedding calls
    within a single request scope.
    """

    def __init__(self, user_id: str | None = None, session_id: str | None = None):
        self.tracker_id: str = uuid.uuid4().hex[:16]
        self.user_id: str | None = user_id
        self.session_id: str | None = session_id
        self.events: list[UsageEvent] = []
        self._token: Any = None

    def __enter__(self) -> "CostTracker":
        self._token = _current_tracker.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _current_tracker.reset(self._token)

    # ── Recording ─────────────────────────────────────────────────────────

    def record_llm(
        self,
        operation: str,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> UsageEvent:
        """Record an LLM call and compute its cost."""
        total = prompt_tokens + completion_tokens
        cost_in, cost_out, cost_total = compute_cost(
            provider, model, prompt_tokens, completion_tokens,
        )
        event = UsageEvent(
            operation=operation,
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            cost_input=cost_in,
            cost_output=cost_out,
            cost_total=cost_total,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.events.append(event)
        logger.debug(
            "cost_recorded",
            operation=operation,
            provider=provider,
            model=model,
            tokens=total,
            cost=f"${cost_total:.6f}",
        )
        return event

    def record_embedding(
        self,
        provider: str,
        model: str,
        token_count: int,
    ) -> UsageEvent:
        """Record an embedding call (input tokens only)."""
        cost_in, _, cost_total = compute_cost(provider, model, token_count, 0)
        event = UsageEvent(
            operation=OPERATION_EMBEDDING,
            provider=provider,
            model=model,
            prompt_tokens=token_count,
            completion_tokens=0,
            total_tokens=token_count,
            cost_input=cost_in,
            cost_output=0.0,
            cost_total=cost_total,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self.events.append(event)
        return event

    # ── Aggregation ───────────────────────────────────────────────────────

    @property
    def total_prompt_tokens(self) -> int:
        return sum(e.prompt_tokens for e in self.events)

    @property
    def total_completion_tokens(self) -> int:
        return sum(e.completion_tokens for e in self.events)

    @property
    def total_tokens(self) -> int:
        return sum(e.total_tokens for e in self.events)

    @property
    def total_cost(self) -> float:
        return sum(e.cost_total for e in self.events)

    def breakdown_by_operation(self) -> dict[str, dict[str, Any]]:
        """Group events by operation type and sum tokens/costs."""
        breakdown: dict[str, dict[str, Any]] = {}
        for e in self.events:
            if e.operation not in breakdown:
                breakdown[e.operation] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            b = breakdown[e.operation]
            b["prompt_tokens"] += e.prompt_tokens
            b["completion_tokens"] += e.completion_tokens
            b["total_tokens"] += e.total_tokens
            b["cost"] += e.cost_total
            b["calls"] += 1
        return breakdown

    def breakdown_by_model(self) -> dict[str, dict[str, Any]]:
        """Group events by model and sum tokens/costs."""
        breakdown: dict[str, dict[str, Any]] = {}
        for e in self.events:
            key = f"{e.provider}/{e.model}"
            if key not in breakdown:
                breakdown[key] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost": 0.0,
                    "calls": 0,
                }
            b = breakdown[key]
            b["prompt_tokens"] += e.prompt_tokens
            b["completion_tokens"] += e.completion_tokens
            b["total_tokens"] += e.total_tokens
            b["cost"] += e.cost_total
            b["calls"] += 1
        return breakdown

    def to_firestore_record(self, query_id: str, query_text: str = "") -> dict[str, Any]:
        """Serialize the full tracker state for Firestore storage."""
        return {
            "tracker_id": self.tracker_id,
            "user_id": self.user_id or "anonymous",
            "session_id": self.session_id,
            "query_id": query_id,
            "query_text": query_text[:200],
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "events": [e.to_dict() for e in self.events],
            "breakdown_by_operation": self.breakdown_by_operation(),
            "breakdown_by_model": self.breakdown_by_model(),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    def to_response_dict(self) -> dict[str, Any]:
        """Lightweight summary for inclusion in query responses."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "breakdown_by_operation": self.breakdown_by_operation(),
            "event_count": len(self.events),
        }
