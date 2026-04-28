"""
Ingest route — upload and process documents.

Upload is synchronous (fast: GCS upload + Firestore record).
Chunking, embedding, and vectorisation run in a dedicated background
thread pool so they don't block other API endpoints.
"""

from __future__ import annotations

import asyncio
import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, UploadFile

from app.api.dependencies import get_cloud_storage_client, get_firestore_client, get_ingestion_pipeline, get_redis_client
from app.api.rate_limiter import check_doc_upload_limit
from app.auth.dependencies import RequiredUser
from app.core.config import get_settings
from app.core.exceptions import IngestionError, RateLimitError
from app.storage.redis_client import RedisClient, RedisUnavailableError
from app.core.logging import get_logger
from app.models.domain import DocumentStatus
from app.models.responses import BatchIngestResponse, IngestResponse
from app.monitoring.prometheus import INGESTION_COUNT, INGESTION_LATENCY

logger = get_logger(__name__)

router = APIRouter(tags=["ingest"])

# Dedicated executor so heavy ingestion work doesn't starve the default
# executor used by other async endpoints (document listing, queries, etc.).
# PDF conversion runs in a subprocess (memory-isolated), so threads here
# only wait on subprocess I/O + do embedding network calls.
# max_workers=1: sequential processing prevents concurrent subprocess
# OOM when system RAM is limited (e.g. 8-16 GB).  Bump to 2 if you have
# ≥32 GB RAM and want parallel document processing.
_INGEST_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ingest")


def _run_processing(document_id: str, filename: str, content_type: str, user_id: str) -> None:
    """Synchronous heavy processing — called from a background thread.

    Content bytes are NOT passed; ``pipeline.process()`` loads from
    the local cache written during the upload phase (or falls back to GCS).
    """
    t0 = time.perf_counter()
    try:
        pipeline = get_ingestion_pipeline()
        pipeline.process(
            document_id=document_id,
            filename=filename,
            content=None,
            content_type=content_type,
            user_id=user_id,
        )
        elapsed = time.perf_counter() - t0
        INGESTION_COUNT.labels(status="success").inc()
        INGESTION_LATENCY.observe(elapsed)
        logger.info("background_processing_complete", document_id=document_id, elapsed_s=round(elapsed, 2))
    except Exception as exc:
        INGESTION_COUNT.labels(status="error").inc()
        logger.exception("background_processing_failed", document_id=document_id, error=str(exc))
        # Mark document as failed in Firestore so the UI doesn't show it stuck at "processing"
        try:
            from app.api.dependencies import get_firestore_client
            fs = get_firestore_client()
            fs.update_document(document_id, {
                "status": "failed",
                "processing_stage": "error",
                "error_message": str(exc)[:500],
            })
        except Exception:
            logger.debug("failed_status_update_error", document_id=document_id)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile,
    user: RequiredUser,
    _limit: None = Depends(check_doc_upload_limit),
    redis: RedisClient = Depends(get_redis_client),
) -> IngestResponse:
    """Upload a document and kick off async processing."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content_type = file.content_type or "application/octet-stream"
    allowed = {"application/pdf", "text/plain", "text/markdown", "text/csv"}
    if content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Allowed: {', '.join(allowed)}",
        )

    file_bytes = await file.read()

    # Phase 1 — fast synchronous: upload to GCS + create Firestore record
    try:
        pipeline = get_ingestion_pipeline()
        document_id = pipeline.upload(
            filename=file.filename,
            content=file_bytes,
            content_type=content_type,
            user_id=user.user_id,
        )
    except Exception as exc:
        logger.exception("upload_failed", filename=file.filename)
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    # Increment the upload counter now that the file is safely stored.
    # Failures before this point (bad file, GCS error) do not consume quota.
    try:
        await redis.incr(f"rl:docs:{user.user_id}")
    except RedisUnavailableError:
        logger.warning("redis_unavailable_doc_counter_skip", user_id=user.user_id)

    # Phase 2 — kick off heavy processing in background (dedicated executor)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(_INGEST_EXECUTOR, _run_processing, document_id, file.filename, content_type, user.user_id)

    return IngestResponse(
        document_id=document_id,
        filename=file.filename,
        status="processing",
        message="Document uploaded — processing in background",
    )


# ── Batch / ZIP constraints ───────────────────────────────────────────────────
_BATCH_MAX_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB
_BATCH_MAX_FILES = 10


@router.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(
    files: list[UploadFile],
    user: RequiredUser,
    redis: RedisClient = Depends(get_redis_client),
) -> BatchIngestResponse:
    """Upload multiple PDF files at once and process each in the background."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > _BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files ({len(files)}). Maximum allowed is {_BATCH_MAX_FILES}.",
        )

    # ── Rate limit check (batch) ──────────────────────────────────────────────
    settings = get_settings()
    # Exemption check — members of 'rl:exempt' bypass all upload limits
    _batch_exempt = False
    try:
        _batch_exempt = await redis.sismember("rl:exempt", user.email)
    except RedisUnavailableError:
        pass  # fail-open
    if not _batch_exempt:
        try:
            current_doc_count = await redis.get_int(f"rl:docs:{user.user_id}")
            if current_doc_count >= settings.rate_limit_docs:
                raise RateLimitError("You have exhausted your free use limit. Contact Admin")
            # Reject the whole batch if it would push the user over the limit.
            if current_doc_count + len(files) > settings.rate_limit_docs:
                remaining = settings.rate_limit_docs - current_doc_count
                raise RateLimitError(
                    f"Batch would exceed your upload limit. You may upload at most {remaining} more document(s)."
                )
        except RateLimitError:
            raise
        except RedisUnavailableError:
            logger.warning("redis_unavailable_batch_limit_skipped", user_id=user.user_id)

    # Read all files and validate total size + types
    file_data: list[tuple[str, bytes]] = []
    total_size = 0
    for f in files:
        if not f.filename:
            raise HTTPException(status_code=400, detail="One of the files has no filename.")
        ct = f.content_type or "application/octet-stream"
        if ct not in {"application/pdf"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type for '{f.filename}': {ct}. Only PDF files are allowed.",
            )
        data = await f.read()
        total_size += len(data)
        if total_size > _BATCH_MAX_SIZE_BYTES:
            raise HTTPException(
                status_code=400,
                detail="Total upload size exceeds the 50 MB limit. Please upload fewer or smaller files.",
            )
        file_data.append((f.filename, data))

    pipeline = get_ingestion_pipeline()
    loop = asyncio.get_running_loop()
    documents: list[IngestResponse] = []
    skipped: list[str] = []

    for filename, content in file_data:
        try:
            document_id = pipeline.upload(
                filename=filename,
                content=content,
                content_type="application/pdf",
                user_id=user.user_id,
            )
        except Exception as exc:
            logger.exception("batch_upload_failed", filename=filename)
            skipped.append(filename)
            continue

        loop.run_in_executor(
            _INGEST_EXECUTOR,
            _run_processing,
            document_id,
            filename,
            "application/pdf",
            user.user_id,
        )

        # Increment counter per successfully accepted upload.
        try:
            await redis.incr(f"rl:docs:{user.user_id}")
        except RedisUnavailableError:
            logger.warning("redis_unavailable_batch_counter_skip", user_id=user.user_id)

        documents.append(
            IngestResponse(
                document_id=document_id,
                filename=filename,
                status="processing",
                message="Processing in background",
            )
        )

    return BatchIngestResponse(
        documents=documents,
        total_files=len(documents),
        skipped_files=skipped,
        message=f"{len(documents)} PDF(s) uploaded — processing in background",
    )


@router.post("/ingest/zip", response_model=BatchIngestResponse)
async def ingest_zip(file: UploadFile, user: RequiredUser) -> BatchIngestResponse:
    """Upload a ZIP archive containing PDF files and process each one."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    content_type = file.content_type or "application/octet-stream"
    if content_type not in {"application/zip", "application/x-zip-compressed", "application/octet-stream"}:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a ZIP file.",
        )

    file_bytes = await file.read()

    # Validate total ZIP size
    if len(file_bytes) > _BATCH_MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=400,
            detail="ZIP file exceeds the 50 MB size limit. Please upload a smaller archive.",
        )

    # Validate it's a real ZIP
    if not zipfile.is_zipfile(io.BytesIO(file_bytes)):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid ZIP archive.",
        )

    with zipfile.ZipFile(io.BytesIO(file_bytes), "r") as zf:
        # Security: reject ZIPs with path-traversal entries
        for entry in zf.namelist():
            if entry.startswith("/") or ".." in entry:
                raise HTTPException(status_code=400, detail="ZIP contains unsafe file paths.")

        # Collect PDF entries (ignore directories and non-PDF files)
        pdf_entries: list[str] = []
        skipped: list[str] = []
        for entry in zf.namelist():
            if entry.endswith("/"):
                continue  # skip directories
            if PurePosixPath(entry).suffix.lower() == ".pdf":
                pdf_entries.append(entry)
            else:
                skipped.append(PurePosixPath(entry).name)

        if not pdf_entries:
            raise HTTPException(
                status_code=400,
                detail="ZIP contains no PDF files. Only PDF files are processed.",
            )

        if len(pdf_entries) > _BATCH_MAX_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"ZIP contains {len(pdf_entries)} PDF files. Maximum allowed is {_BATCH_MAX_FILES}.",
            )

        # Upload + background-process each PDF
        pipeline = get_ingestion_pipeline()
        loop = asyncio.get_running_loop()
        documents: list[IngestResponse] = []

        for entry in pdf_entries:
            pdf_bytes = zf.read(entry)
            pdf_filename = PurePosixPath(entry).name

            try:
                document_id = pipeline.upload(
                    filename=pdf_filename,
                    content=pdf_bytes,
                    content_type="application/pdf",
                    user_id=user.user_id,
                )
            except Exception as exc:
                logger.exception("zip_upload_failed", filename=pdf_filename)
                skipped.append(pdf_filename)
                continue

            # Kick off background processing
            loop.run_in_executor(
                _INGEST_EXECUTOR,
                _run_processing,
                document_id,
                pdf_filename,
                "application/pdf",
                user.user_id,
            )

            documents.append(
                IngestResponse(
                    document_id=document_id,
                    filename=pdf_filename,
                    status="processing",
                    message="Processing in background",
                )
            )

    return BatchIngestResponse(
        documents=documents,
        total_files=len(documents),
        skipped_files=skipped,
        message=f"{len(documents)} PDF(s) uploaded from ZIP — processing in background",
    )
