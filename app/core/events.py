"""
Application lifecycle events (startup / shutdown).
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# One-liner that initialises Docling's DocumentConverter — which triggers the
# huggingface_hub model download for layout + table ONNX weights.
# Must be kept in sync with document_converter.py's pipeline options.
# ---------------------------------------------------------------------------
_DOCLING_PRELOAD_SCRIPT = (
    # 1. Pre-download Docling layout + table ONNX models (used by the subprocess worker)
    "from docling.datamodel.base_models import InputFormat;"
    "from docling.datamodel.pipeline_options import PdfPipelineOptions;"
    "from docling.document_converter import DocumentConverter as DC, PdfFormatOption;"
    "opts = PdfPipelineOptions();"
    "opts.do_ocr = False;"
    "opts.do_table_structure = False;"
    "opts.generate_page_images = False;"
    "opts.generate_picture_images = False;"
    "DC(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)});"
    # 2. Pre-download sentence-transformers/all-MiniLM-L6-v2 tokenizer
    # (used by docling_core HybridChunker for token counting — loaded in the main API process)
    "from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer;"
    "get_default_tokenizer();"
    "print('docling_models_ok', flush=True)"
)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/


async def _preload_docling_models() -> None:
    """
    Ensure Docling ONNX/ML models are present in the local cache.

    Runs a short-lived subprocess (keeping the API server's heap clean) that
    initialises Docling's DocumentConverter.  On first run this triggers a
    ~300 MB download from HuggingFace; on subsequent runs it exits in < 5 s
    because the weights are already cached.

    After the subprocess exits (success or failure), HF_HUB_OFFLINE=1 and
    TRANSFORMERS_OFFLINE=1 are written into the parent process environment so
    that all future ingestion subprocesses inherit offline mode — preventing
    any further attempts to reach huggingface.co at runtime.

    If HF_HUB_OFFLINE is already set to "1" in the environment (e.g. a Docker
    image that pre-downloaded models at build time), the subprocess is skipped
    and only the env-var propagation step runs.
    """
    already_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"

    if not already_offline:
        # Allow HF downloads inside the preload subprocess (override any
        # inherited offline flag that might have been set externally).
        preload_env = {**os.environ, "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0"}

        logger.info("docling_model_preload_start", hint="downloads on first run, fast on subsequent starts")
        try:
            loop = asyncio.get_event_loop()
            result: subprocess.CompletedProcess = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [sys.executable, "-c", _DOCLING_PRELOAD_SCRIPT],
                    capture_output=True,
                    text=True,
                    timeout=600,  # 10-minute ceiling for slow connections
                    cwd=str(_BACKEND_ROOT),
                    env=preload_env,
                ),
            )
            if result.returncode == 0:
                logger.info("docling_model_preload_complete")
            else:
                logger.warning(
                    "docling_model_preload_failed",
                    returncode=result.returncode,
                    stderr=(result.stderr or "")[-800:],
                    hint=(
                        "HuggingFace may be unreachable. "
                        "Set HF_ENDPOINT env var to use a mirror, "
                        "or pre-download models manually with: "
                        "python -c \"" + _DOCLING_PRELOAD_SCRIPT + "\""
                    ),
                )
        except subprocess.TimeoutExpired:
            logger.warning("docling_model_preload_timeout", timeout_s=600)
        except Exception as exc:
            logger.warning("docling_model_preload_error", error=str(exc))
    else:
        logger.info("docling_model_preload_skipped", reason="HF_HUB_OFFLINE already set — models assumed cached")

    # Lock HF to offline mode for all subsequent ingestion subprocesses.
    # os.environ changes are inherited by child processes spawned after this point.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("hf_offline_mode_enabled")


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

    # Pre-download Docling ML models (layout + table detection) and lock HF offline mode.
    # Non-fatal: a warning is logged if HuggingFace is unreachable, but the server
    # still starts.  The first PDF upload will fail if models are missing; subsequent
    # uploads work because the cache is populated on the first successful download.
    await _preload_docling_models()

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
