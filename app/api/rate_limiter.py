"""
Rate limiting FastAPI dependencies.

Two async callables are provided for injection via ``Depends()``:

  - ``check_doc_upload_limit``  — for ``POST /ingest*`` routes
  - ``check_query_limit``       — for ``POST /query`` (optional-user variant
                                    is inlined in the route itself)

Both are **fail-open**: if Redis is unreachable the request is allowed
through and a warning is logged. Redis downtime must not take down the API.

Counter increments (on success) happen in the route handlers, NOT here,
so failed uploads/queries never consume quota.
"""

from __future__ import annotations

from fastapi import Depends

from app.api.dependencies import get_redis_client
from app.auth.dependencies import RequiredUser
from app.core.config import get_settings
from app.core.exceptions import RateLimitError
from app.core.logging import get_logger
from app.storage.redis_client import RedisClient, RedisUnavailableError

logger = get_logger(__name__)

_LIMIT_MSG = "You have exhausted your free use limit. Contact Admin"


async def check_doc_upload_limit(
    user: RequiredUser,
    redis: RedisClient = Depends(get_redis_client),
) -> None:
    """Raise ``RateLimitError`` (429) if the user has hit the document upload limit.

    Injected as a FastAPI dependency on ingest routes — runs before the route
    body executes. Counter increment happens in the route after a successful upload.
    """
    settings = get_settings()
    # ── Exemption check ───────────────────────────────────────────────────────
    try:
        if await redis.sismember("rl:exempt", user.email):
            logger.debug("doc_upload_limit_exempt", user_id=user.user_id, email=user.email)
            return  # exempt — skip all rate limiting
    except RedisUnavailableError:
        pass  # fail-open: proceed to counter check

    try:
        count = await redis.get_int(f"rl:docs:{user.user_id}")
    except RedisUnavailableError:
        logger.warning("redis_unavailable_doc_limit_skipped", user_id=user.user_id)
        return  # fail-open

    if count >= settings.rate_limit_docs:
        logger.info("doc_upload_limit_exceeded", user_id=user.user_id, count=count, limit=settings.rate_limit_docs)
        raise RateLimitError(_LIMIT_MSG)
