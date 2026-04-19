"""
Evaluation route — run RAGAS evaluations.

The evaluation loop invokes the RAG pipeline multiple times (once per
benchmark question), which is entirely synchronous and can run for
minutes.  All heavy work is offloaded to a thread via
``asyncio.to_thread`` so the event loop stays responsive.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.dependencies import (
    get_firestore_client,
    get_ragas_evaluator,
    get_rag_registry,
)
from app.auth.dependencies import RequiredUser
from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.ragas_eval import EvalSample
from app.models.requests import EvaluateRequest
from app.models.responses import (
    EvalMetricsResponse,
    EvalQuestionResponse,
    EvalReportResponse,
)

logger = get_logger(__name__)

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


def _load_benchmark_questions(path: str) -> list[dict]:
    """Load benchmark questions from JSON file."""
    p = Path(path)
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("questions", [])


def _run_evaluation_sync(
    questions: list[dict],
    body: EvaluateRequest,
    user_id: str,
) -> EvalReportResponse:
    """Execute the full evaluation loop synchronously (runs in a worker thread)."""
    settings = get_settings()
    t0 = time.perf_counter()
    evaluator = get_ragas_evaluator()
    registry = get_rag_registry()
    strategy = registry.get(settings.rag_strategy.value)

    question_results: list[EvalQuestionResponse] = []

    for i, q in enumerate(questions):
        q_text = q.get("question", "")
        ground_truth = q.get("ground_truth", "")
        category = q.get("category", "general")
        qt0 = time.perf_counter()

        try:
            state = strategy.invoke({"query": q_text, "document_ids": []})
            answer = state.get("final_answer", state.get("answer", ""))
            chunks = state.get("relevant_chunks") or state.get("retrieved_chunks", [])
            contexts = [c.text for c in chunks]
        except Exception as exc:
            logger.warning("eval_query_error", question=q_text[:80], error=str(exc))
            answer = f"Error: {exc}"
            contexts = []

        sample = EvalSample(
            question=q_text, answer=answer,
            contexts=contexts, ground_truth=ground_truth,
        )
        ragas_result = evaluator.evaluate_single(sample)
        latency_ms = (time.perf_counter() - qt0) * 1000

        question_results.append(EvalQuestionResponse(
            id=str(i + 1),
            question=q_text,
            category=category,
            hit_at_k=len(contexts) > 0,
            faithfulness=ragas_result.faithfulness,
            answer_relevancy=ragas_result.answer_relevancy,
            context_precision=ragas_result.context_precision,
            latency_ms=round(latency_ms, 1),
            answer=answer,
            ground_truth=ground_truth,
        ))

    elapsed = time.perf_counter() - t0
    n = len(question_results) or 1

    report_id = str(uuid.uuid4())[:8]
    report = EvalReportResponse(
        id=report_id,
        run_at=datetime.now(timezone.utc).isoformat(),
        llm_provider=body.llm_provider,
        top_k=body.top_k,
        metrics=EvalMetricsResponse(
            hit_at_k=sum(1 for q in question_results if q.hit_at_k) / n,
            mrr=0.0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            faithfulness=sum(q.faithfulness for q in question_results) / n,
            answer_relevancy=sum(q.answer_relevancy for q in question_results) / n,
            context_precision=sum(q.context_precision for q in question_results) / n,
            hallucination_rate=0.0,
            avg_latency_ms=sum(q.latency_ms for q in question_results) / n,
            total_questions=len(question_results),
        ),
        questions=question_results,
    )

    # Save to Firestore
    try:
        fs = get_firestore_client()
        fs.save_evaluation(report_id, {
            "user_id": user_id,
            "llm_provider": body.llm_provider,
            "top_k": body.top_k,
            "total_questions": len(question_results),
            "metrics": report.metrics.model_dump(),
            "processing_time": round(elapsed, 2),
        })
    except Exception as exc:
        logger.warning("eval_save_error", error=str(exc))

    return report


@router.post("", response_model=EvalReportResponse)
async def run_evaluation(body: EvaluateRequest, user: RequiredUser) -> EvalReportResponse:
    """Run evaluation using benchmark dataset."""
    settings = get_settings()

    questions = _load_benchmark_questions(settings.benchmark_dataset_path)
    if not questions:
        raise HTTPException(
            status_code=404,
            detail=f"No benchmark questions found at {settings.benchmark_dataset_path}. "
                   "Create a JSON file with a list of {question, ground_truth, category} objects.",
        )

    return await asyncio.to_thread(
        _run_evaluation_sync, questions, body, user.user_id,
    )


@router.get("/results", response_model=EvalReportResponse | None)
async def get_latest_evaluation() -> EvalReportResponse | None:
    """Get the most recent evaluation report."""
    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    data = await loop.run_in_executor(None, fs.get_latest_evaluation)
    if not data:
        return None
    return EvalReportResponse(
        id=data.get("evaluation_id", ""),
        run_at=data.get("started_at", ""),
        llm_provider=data.get("llm_provider", "ollama"),
        top_k=data.get("top_k", 5),
        metrics=EvalMetricsResponse(**data.get("metrics", {})),
        questions=[],
    )
