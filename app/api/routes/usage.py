"""
Usage & cost analytics routes — per-user token consumption and cost breakdown.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query

from app.api.dependencies import get_firestore_client, get_redis_client
from app.auth.dependencies import OptionalUser
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.responses import (
    OperationBreakdown,
    UsageRecordResponse,
    UsageSummaryResponse,
    UsageEventResponse,
    UserLimitsResponse,
)
from app.storage.redis_client import RedisUnavailableError

logger = get_logger(__name__)

router = APIRouter(tags=["usage"])


@router.get("/usage/summary", response_model=UsageSummaryResponse)
async def usage_summary(user: OptionalUser) -> UsageSummaryResponse:
    """Get aggregated usage summary for the authenticated user."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    firestore = get_firestore_client()
    loop = asyncio.get_running_loop()
    summary = await loop.run_in_executor(
        None, lambda: firestore.get_usage_summary(user.user_id),
    )

    return UsageSummaryResponse(
        user_id=summary["user_id"],
        total_prompt_tokens=summary["total_prompt_tokens"],
        total_completion_tokens=summary["total_completion_tokens"],
        total_tokens=summary["total_tokens"],
        total_cost=summary["total_cost"],
        total_queries=summary["total_queries"],
        avg_cost_per_query=summary["avg_cost_per_query"],
        avg_tokens_per_query=summary["avg_tokens_per_query"],
        breakdown_by_model={
            k: OperationBreakdown(**v) for k, v in summary["breakdown_by_model"].items()
        },
        breakdown_by_operation={
            k: OperationBreakdown(**v) for k, v in summary["breakdown_by_operation"].items()
        },
        daily_costs=summary["daily_costs"],
    )


@router.get("/usage/history", response_model=list[UsageRecordResponse])
async def usage_history(
    user: OptionalUser,
    limit: int = Query(default=50, ge=1, le=500),
    start_date: str | None = Query(default=None, description="ISO date filter start (inclusive)"),
    end_date: str | None = Query(default=None, description="ISO date filter end (inclusive)"),
) -> list[UsageRecordResponse]:
    """Get detailed usage records for the authenticated user."""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    firestore = get_firestore_client()
    loop = asyncio.get_running_loop()
    records = await loop.run_in_executor(
        None,
        lambda: firestore.get_usage_records(
            user.user_id, limit=limit, start_date=start_date, end_date=end_date,
        ),
    )

    return [
        UsageRecordResponse(
            id=r["id"],
            query_id=r.get("query_id", ""),
            query_text=r.get("query_text", ""),
            total_prompt_tokens=r.get("total_prompt_tokens", 0),
            total_completion_tokens=r.get("total_completion_tokens", 0),
            total_tokens=r.get("total_tokens", 0),
            total_cost=r.get("total_cost", 0.0),
            events=[UsageEventResponse(**e) for e in r.get("events", [])],
            breakdown_by_operation={
                k: OperationBreakdown(**v) for k, v in r.get("breakdown_by_operation", {}).items()
            },
            breakdown_by_model={
                k: OperationBreakdown(**v) for k, v in r.get("breakdown_by_model", {}).items()
            },
            created_at=r.get("created_at", ""),
        )
        for r in records
    ]


@router.get("/usage/limits", response_model=UserLimitsResponse)
async def usage_limits(user: OptionalUser) -> UserLimitsResponse:
    """Return the rate-limit config and the user's current counters from Redis.

    Always returns the configured limits so an exempt user can still see what
    limits apply to regular accounts. ``is_exempt=True`` signals the UI to
    present the matrix in informational-only mode.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    settings = get_settings()
    redis = get_redis_client()

    # ── Exemption check ───────────────────────────────────────────────────────
    try:
        is_exempt = await redis.sismember("rl:exempt", user.email)
    except RedisUnavailableError:
        is_exempt = False

    # ── Current counters ──────────────────────────────────────────────────────
    try:
        doc_count = await redis.get_int(f"rl:docs:{user.user_id}")
    except RedisUnavailableError:
        doc_count = 0

    try:
        query_count = await redis.get_int(f"rl:queries:{user.user_id}")
    except RedisUnavailableError:
        query_count = 0

    return UserLimitsResponse(
        doc_limit=settings.rate_limit_docs,
        query_limit=settings.rate_limit_queries,
        doc_count=doc_count,
        query_count=query_count,
        is_exempt=is_exempt,
    )
