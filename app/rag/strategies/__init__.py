"""
RAG strategy protocol and registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.rag.state import RAGState


class RAGStrategy(ABC):
    """Base class for RAG pipeline strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique strategy identifier."""
        ...

    @abstractmethod
    def invoke(self, state: RAGState) -> RAGState:
        """Execute the RAG pipeline and return final state."""
        ...

    def __repr__(self) -> str:
        return f"<RAGStrategy: {self.name}>"


class RAGStrategyRegistry:
    """Registry mapping strategy names to implementations."""

    def __init__(self) -> None:
        self._strategies: dict[str, RAGStrategy] = {}

    def register(self, strategy: RAGStrategy) -> None:
        self._strategies[strategy.name] = strategy

    def get(self, name: str) -> RAGStrategy:
        if name not in self._strategies:
            available = list(self._strategies.keys())
            raise KeyError(f"RAG strategy '{name}' not found. Available: {available}")
        return self._strategies[name]

    def list_strategies(self) -> list[str]:
        return sorted(self._strategies.keys())
