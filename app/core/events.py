"""
Application lifecycle events (startup / shutdown).
"""

from __future__ import annotations

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
    setup_logging(log_level=settings.log_level, json_logs=not settings.debug)

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
