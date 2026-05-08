"""
Application lifecycle events (startup / shutdown).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan — runs on startup and shutdown."""
    settings = get_settings()
    # Cloud Run always sets K_SERVICE. Force JSON logs there so every log line
    # is a queryable jsonPayload in Log Explorer — regardless of RAG_DEBUG.
    # Locally (no K_SERVICE) respect the debug flag for readable console output.
    in_cloud_run = bool(os.environ.get("K_SERVICE"))
    setup_logging(log_level=settings.log_level, json_logs=in_cloud_run or not settings.debug)

    logger.info(
        "application_starting",
        env=settings.env,
        app_name=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        qdrant=f"{settings.qdrant_host}:{settings.qdrant_port}",
        embedding_model=settings.embedding_model.value,
        llm_provider=settings.llm_provider,
        rag_strategy=settings.rag_strategy.value,
    )

    # Startup: verify critical connections
    try:
        from app.api.dependencies import verify_services

        verify_services()
        logger.info("all_services_verified")
    except Exception as exc:
        logger.warning("service_verification_partial_failure", error=str(exc))

    # Redis connectivity check (non-fatal — rate limiting fails-open when Redis is down)
    try:
        from app.api.dependencies import get_redis_client

        redis = get_redis_client()
        reachable = await redis.ping()
        if reachable:
            logger.info("redis_connected", url=settings.redis_url)
            # Seed the exemption SET from config (idempotent — SADD ignores duplicates)
            if settings.rate_limit_exempt_emails:
                await redis.sadd("rl:exempt", *settings.rate_limit_exempt_emails)
                logger.info(
                    "rate_limit_exempt_seeded",
                    count=len(settings.rate_limit_exempt_emails),
                    emails=settings.rate_limit_exempt_emails,
                )
        else:
            logger.warning("redis_unreachable_rate_limiting_disabled", url=settings.redis_url)
    except Exception as exc:
        logger.warning("redis_startup_check_failed", error=str(exc))

    # Mark documents stuck in "processing" from a previous crash as "failed"
    try:
        from app.api.dependencies import get_firestore_client

        fs = get_firestore_client()
        docs = fs.list_documents()
        stuck = [d for d in docs if d.get("status") == "processing"]
        for doc in stuck:
            doc_id = doc.get("id", "")
            fs.update_document(doc_id, {
                "status": "failed",
                "error": "Application restarted while document was processing. Use re-index to retry.",
                "processing_stage": None,
            })
            logger.warning("stuck_document_marked_failed", document_id=doc_id, filename=doc.get("filename", ""))
        if stuck:
            logger.info("stuck_documents_recovered", count=len(stuck))
    except Exception as exc:
        logger.warning("stuck_document_recovery_failed", error=str(exc))

    yield

    # Shutdown
    logger.info("application_shutting_down")

    # Close Redis connection pool cleanly
    try:
        from app.api.dependencies import get_redis_client
        await get_redis_client().close()
    except Exception:
        pass
