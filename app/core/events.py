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
# Pre-download Docling's layout/table ONNX models + sentence-transformers
# tokenizer.  Runs in a subprocess to keep the API server heap clean.
#
# In the Docker image, models are already cached at build time (Dockerfile
# RUN step), so this subprocess exits in <5 s on Cloud Run.
# On local dev / first run, it downloads ~300 MB of ONNX weights.
#
# Note: torchvision must be the CPU wheel (not PyPI's CUDA build) for the
# import chain to work: granite_vision → AutoProcessor → torchvision.
# The Dockerfile installs it from https://download.pytorch.org/whl/cpu.
# ---------------------------------------------------------------------------
_DOCLING_PRELOAD_SCRIPT = (
    # 1. Pre-download layout (Heron) + table (TableFormer) ONNX models
    "from docling.datamodel.base_models import InputFormat;"
    "from docling.datamodel.pipeline_options import PdfPipelineOptions;"
    "from docling.document_converter import DocumentConverter as DC, PdfFormatOption;"
    "opts = PdfPipelineOptions();"
    "opts.do_ocr = False;"
    "opts.do_chart_extraction = False;"
    "opts.do_code_enrichment = False;"
    "opts.do_formula_enrichment = False;"
    "opts.generate_page_images = False;"
    "opts.generate_picture_images = False;"
    "opts.generate_parsed_pages = False;"
    "DC(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)});"
    # 2. Pre-download sentence-transformers/all-MiniLM-L6-v2 tokenizer
    # (used by HybridChunker for token counting in the main API process)
    "from docling_core.transforms.chunker.tokenizer.huggingface import get_default_tokenizer;"
    "get_default_tokenizer();"
    "print('docling_models_ok', flush=True)"
)

_BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent  # backend/


async def _preload_docling_models() -> None:
    """
    Ensure Docling ONNX models and the HuggingFace tokenizer are cached locally.

    Runs a short-lived subprocess that initialises docling's DocumentConverter
    (downloading layout + table ONNX weights on first run) and the
    sentence-transformers tokenizer used by HybridChunker.  On Cloud Run
    both are already in the image cache, so this exits in < 5 s.

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
