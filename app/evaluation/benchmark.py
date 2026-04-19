"""
Benchmark runner — runs evaluation across a dataset and aggregates results.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.core.logging import get_logger
from app.evaluation.metrics import (
    RetrievalMetrics,
    compute_average_metrics,
    compute_retrieval_metrics,
)
from app.evaluation.ragas_eval import EvalSample, RAGASEvaluator, RAGASResult
from app.models.domain import RetrievalResult

logger = get_logger(__name__)


@dataclass
class BenchmarkQuestion:
    """A benchmark question with ground truth."""

    question: str
    ground_truth_answer: str = ""
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevant_document_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results for a single benchmark question."""

    question: str
    answer: str
    contexts: list[str]
    retrieval_metrics: RetrievalMetrics | None = None
    ragas_result: RAGASResult | None = None
    latency_ms: float = 0.0
    strategy: str = ""


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""

    total_questions: int = 0
    avg_retrieval_metrics: dict[str, float] = field(default_factory=dict)
    avg_ragas_scores: dict[str, Any] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    results: list[BenchmarkResult] = field(default_factory=list)
    run_timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_questions": self.total_questions,
            "avg_retrieval_metrics": self.avg_retrieval_metrics,
            "avg_ragas_scores": self.avg_ragas_scores,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "run_timestamp": self.run_timestamp,
            "results": [
                {
                    "question": r.question,
                    "answer": r.answer[:200],
                    "strategy": r.strategy,
                    "latency_ms": round(r.latency_ms, 1),
                    "retrieval": r.retrieval_metrics.to_dict() if r.retrieval_metrics else {},
                    "ragas": r.ragas_result.to_dict() if r.ragas_result else {},
                }
                for r in self.results
            ],
        }


class BenchmarkRunner:
    """
    Runs end-to-end evaluation benchmarks.

    Requires a RAG pipeline function (query → answer, contexts, retrieved)
    and a list of benchmark questions.
    """

    def __init__(self, ragas_evaluator: RAGASEvaluator | None = None) -> None:
        self._evaluator = ragas_evaluator or RAGASEvaluator()

    def run(
        self,
        questions: list[BenchmarkQuestion],
        rag_fn: Any,  # Callable[[str], tuple[str, list[str], list[RetrievalResult]]]
        *,
        strategy_name: str = "adaptive",
    ) -> BenchmarkReport:
        """
        Run benchmark evaluation.

        Parameters
        ----------
        questions : list[BenchmarkQuestion]
            Benchmark dataset.
        rag_fn : callable
            Function(query) -> (answer, contexts, retrieved_chunks).
        strategy_name : str
            Name of the RAG strategy being evaluated.
        """
        from datetime import datetime, timezone

        results: list[BenchmarkResult] = []
        retrieval_metrics_list: list[RetrievalMetrics] = []

        for i, q in enumerate(questions):
            logger.info("benchmark_question", index=i + 1, total=len(questions))
            t0 = time.perf_counter()

            try:
                answer, contexts, retrieved = rag_fn(q.question)
            except Exception as exc:
                logger.warning("benchmark_error", question=q.question[:80], error=str(exc))
                results.append(BenchmarkResult(
                    question=q.question,
                    answer=f"ERROR: {exc}",
                    contexts=[],
                    strategy=strategy_name,
                ))
                continue

            elapsed = (time.perf_counter() - t0) * 1000

            # Retrieval metrics (if ground truth available)
            ret_metrics = None
            if q.relevant_chunk_ids or q.relevant_document_ids:
                relevant_ids = set(q.relevant_chunk_ids) or set(q.relevant_document_ids)
                ret_metrics = compute_retrieval_metrics(retrieved, relevant_ids)
                retrieval_metrics_list.append(ret_metrics)

            # RAGAS evaluation
            sample = EvalSample(
                question=q.question,
                answer=answer,
                contexts=contexts,
                ground_truth=q.ground_truth_answer,
            )
            ragas_result = self._evaluator.evaluate_single(sample)

            results.append(BenchmarkResult(
                question=q.question,
                answer=answer,
                contexts=contexts,
                retrieval_metrics=ret_metrics,
                ragas_result=ragas_result,
                latency_ms=elapsed,
                strategy=strategy_name,
            ))

        # Aggregate
        report = BenchmarkReport(
            total_questions=len(questions),
            avg_retrieval_metrics=compute_average_metrics(retrieval_metrics_list),
            avg_ragas_scores=self._avg_ragas(results),
            avg_latency_ms=sum(r.latency_ms for r in results) / max(len(results), 1),
            results=results,
            run_timestamp=datetime.now(timezone.utc).isoformat(),
        )

        logger.info(
            "benchmark_complete",
            questions=report.total_questions,
            avg_latency=round(report.avg_latency_ms, 1),
        )
        return report

    @staticmethod
    def _avg_ragas(results: list[BenchmarkResult]) -> dict[str, float]:
        scored = [r for r in results if r.ragas_result is not None]
        if not scored:
            return {}
        n = len(scored)
        return {
            "faithfulness": round(sum(r.ragas_result.faithfulness for r in scored) / n, 4),  # type: ignore
            "answer_relevancy": round(sum(r.ragas_result.answer_relevancy for r in scored) / n, 4),  # type: ignore
            "context_precision": round(sum(r.ragas_result.context_precision for r in scored) / n, 4),  # type: ignore
            "context_recall": round(sum(r.ragas_result.context_recall for r in scored) / n, 4),  # type: ignore
            "overall_score": round(sum(r.ragas_result.overall_score for r in scored) / n, 4),  # type: ignore
        }

    @staticmethod
    def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
        """Load benchmark questions from a JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [
            BenchmarkQuestion(
                question=q["question"],
                ground_truth_answer=q.get("ground_truth", ""),
                relevant_chunk_ids=q.get("relevant_chunk_ids", []),
                relevant_document_ids=q.get("relevant_document_ids", []),
                metadata=q.get("metadata", {}),
            )
            for q in data
        ]
