"""
RAGAS evaluation wrapper — end-to-end RAG quality scoring.

Wraps the RAGAS library to compute:
- Faithfulness (answer grounded in context)
- Answer relevancy (answer addresses the question)
- Context precision (retrieved context is relevant)
- Context recall (relevant context was retrieved)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RAGASResult:
    """Scores from a RAGAS evaluation run."""

    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    overall_score: float = 0.0
    latency_ms: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, float | str | None]:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "context_recall": round(self.context_recall, 4),
            "overall_score": round(self.overall_score, 4),
            "latency_ms": round(self.latency_ms, 1),
            "error": self.error,
        }


@dataclass
class EvalSample:
    """A single evaluation sample."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str = ""


class RAGASEvaluator:
    """
    Wraps RAGAS library for end-to-end RAG evaluation.

    Falls back to heuristic scoring when RAGAS dependencies are unavailable
    or when evaluation fails.
    """

    def __init__(self, llm_model: str = "gpt-4o-mini") -> None:
        self._llm_model = llm_model
        self._ragas_available = self._check_ragas()

    def _check_ragas(self) -> bool:
        try:
            import ragas  # noqa: F401
            return True
        except ImportError:
            logger.warning("ragas_not_available", msg="RAGAS not installed, using heuristic fallback")
            return False

    def evaluate_single(self, sample: EvalSample) -> RAGASResult:
        """Evaluate a single question-answer pair."""
        t0 = time.perf_counter()

        if self._ragas_available:
            try:
                result = self._evaluate_with_ragas([sample])
                result.latency_ms = (time.perf_counter() - t0) * 1000
                return result
            except Exception as exc:
                logger.warning("ragas_eval_error", error=str(exc))

        # Heuristic fallback
        result = self._heuristic_eval(sample)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def evaluate_batch(self, samples: list[EvalSample]) -> list[RAGASResult]:
        """Evaluate a batch of samples."""
        if not samples:
            return []

        if self._ragas_available:
            try:
                return [self.evaluate_single(s) for s in samples]
            except Exception as exc:
                logger.warning("ragas_batch_error", error=str(exc))

        return [self._heuristic_eval(s) for s in samples]

    def evaluate_batch_aggregate(self, samples: list[EvalSample]) -> RAGASResult:
        """Evaluate batch and return averaged scores."""
        results = self.evaluate_batch(samples)
        if not results:
            return RAGASResult()

        n = len(results)
        avg = RAGASResult(
            faithfulness=sum(r.faithfulness for r in results) / n,
            answer_relevancy=sum(r.answer_relevancy for r in results) / n,
            context_precision=sum(r.context_precision for r in results) / n,
            context_recall=sum(r.context_recall for r in results) / n,
            latency_ms=sum(r.latency_ms for r in results),
        )
        avg.overall_score = (
            avg.faithfulness + avg.answer_relevancy
            + avg.context_precision + avg.context_recall
        ) / 4
        return avg

    # ── RAGAS integration ─────────────────────────────────────────────────────

    def _evaluate_with_ragas(self, samples: list[EvalSample]) -> RAGASResult:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [s.question for s in samples],
            "answer": [s.answer for s in samples],
            "contexts": [s.contexts for s in samples],
            "ground_truth": [s.ground_truth or "" for s in samples],
        }
        dataset = Dataset.from_dict(data)

        metrics = [faithfulness, answer_relevancy, context_precision]
        # Only include context_recall if ground truth is available
        if any(s.ground_truth for s in samples):
            metrics.append(context_recall)

        result = evaluate(dataset, metrics=metrics)

        return RAGASResult(
            faithfulness=result.get("faithfulness", 0.0),
            answer_relevancy=result.get("answer_relevancy", 0.0),
            context_precision=result.get("context_precision", 0.0),
            context_recall=result.get("context_recall", 0.0),
            overall_score=(
                result.get("faithfulness", 0.0)
                + result.get("answer_relevancy", 0.0)
                + result.get("context_precision", 0.0)
                + result.get("context_recall", 0.0)
            ) / 4,
        )

    # ── Heuristic fallback ────────────────────────────────────────────────────

    def _heuristic_eval(self, sample: EvalSample) -> RAGASResult:
        """
        Fast heuristic evaluation using text overlap.
        Not as accurate as RAGAS with an LLM judge, but works offline.
        """
        import re

        answer_words = set(re.findall(r"\w+", sample.answer.lower()))
        context_text = " ".join(sample.contexts).lower()
        context_words = set(re.findall(r"\w+", context_text))
        query_words = set(re.findall(r"\w+", sample.question.lower()))

        # Faithfulness: fraction of answer words found in context
        faith = len(answer_words & context_words) / max(len(answer_words), 1)

        # Answer relevancy: fraction of query words found in answer
        relevancy = len(query_words & answer_words) / max(len(query_words), 1)

        # Context precision: fraction of context words found in answer
        ctx_precision = len(context_words & answer_words) / max(len(context_words), 1)

        # Context recall: if ground truth available
        ctx_recall = 0.0
        if sample.ground_truth:
            gt_words = set(re.findall(r"\w+", sample.ground_truth.lower()))
            ctx_recall = len(gt_words & context_words) / max(len(gt_words), 1)

        overall = (faith + relevancy + ctx_precision + ctx_recall) / 4

        return RAGASResult(
            faithfulness=min(faith, 1.0),
            answer_relevancy=min(relevancy, 1.0),
            context_precision=min(ctx_precision, 1.0),
            context_recall=min(ctx_recall, 1.0),
            overall_score=min(overall, 1.0),
        )
