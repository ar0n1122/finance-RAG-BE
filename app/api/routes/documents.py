"""
Document management routes — list, get, delete documents.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.dependencies import get_cloud_storage_client, get_firestore_client, get_qdrant_client
from app.auth.dependencies import OptionalUser, RequiredUser
from app.core.exceptions import NotFoundError
from app.models.responses import DocumentResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentResponse])
async def list_documents(user: OptionalUser) -> list[DocumentResponse]:
    """List all documents. Optionally filtered to user's documents."""
    fs = get_firestore_client()
    user_id = user.id if user else None
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(None, lambda: fs.list_documents(user_id=user_id))
    result = []
    for doc in docs:
        try:
            result.append(_to_response(doc))
        except Exception:
            pass  # skip malformed docs rather than failing the whole list
    return result


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str) -> DocumentResponse:
    """Get a specific document by ID."""
    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    doc = await loop.run_in_executor(None, lambda: fs.get_document(document_id))
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    return _to_response(doc)


@router.delete("/{document_id}")
async def delete_document(document_id: str, user: RequiredUser) -> dict[str, str]:
    """Delete a document — removes from Qdrant, GCS, and Firestore."""
    fs = get_firestore_client()
    gcs = get_cloud_storage_client()
    qdrant = get_qdrant_client()

    loop = asyncio.get_running_loop()
    doc = await loop.run_in_executor(None, lambda: fs.get_document(document_id))
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    def _delete_sync() -> None:
        # Delete from Qdrant (filter by document_id payload)
        try:
            from qdrant_client import models
            qdrant.delete(
                collection_name="documents",
                points_selector=models.FilterSelector(
                    filter=models.Filter(must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id),
                        ),
                    ]),
                ),
            )
        except Exception:
            pass  # Collection might not exist yet

        # Delete from GCS
        gcs.delete_document_files(document_id)

        # Delete from Firestore
        fs.delete_document(document_id)

    await loop.run_in_executor(None, _delete_sync)

    return {"status": "deleted", "document_id": document_id}


@router.post("/{document_id}/reindex")
async def reindex_document(document_id: str, user: RequiredUser) -> dict[str, str]:
    """Re-process a failed or uploaded document. Uses local cache if available, otherwise fetches from GCS."""
    from app.api.dependencies import get_ingestion_pipeline
    from app.models.domain import DocumentStatus

    fs = get_firestore_client()
    loop = asyncio.get_running_loop()
    doc = await loop.run_in_executor(None, lambda: fs.get_document(document_id))
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    status = doc.get("status", "")
    if status not in ("failed", "uploaded", "queued", "processing"):
        raise HTTPException(status_code=400, detail=f"Cannot reindex document with status '{status}'")

    filename = doc.get("filename", doc.get("source_file", ""))
    content_type = doc.get("content_type", "application/pdf")

    # Mark as processing
    await loop.run_in_executor(
        None,
        lambda: fs.update_document(document_id, {"status": DocumentStatus.PROCESSING.value, "error": None}),
    )

    # Kick off background processing — pipeline loads from local cache or GCS automatically.
    from app.api.routes.ingest import _INGEST_EXECUTOR

    def _run():
        try:
            pipeline = get_ingestion_pipeline()
            pipeline.process(
                document_id=document_id,
                filename=filename,
                content=None,
                content_type=content_type,
                user_id=user.user_id,
            )
        except Exception:
            pass  # pipeline.process already updates Firestore on failure

    loop.run_in_executor(_INGEST_EXECUTOR, _run)

    return {"status": "processing", "document_id": document_id, "message": "Re-processing started"}


_VALID_STATUSES = {"indexed", "processing", "failed", "queued", "uploaded"}


def _to_response(doc: dict) -> DocumentResponse:
    raw_status = doc.get("status") or "queued"
    status = raw_status if raw_status in _VALID_STATUSES else "failed"
    filename = doc.get("filename") or doc.get("source_file") or doc.get("name") or ""
    return DocumentResponse(
        id=doc.get("id") or "",
        title=doc.get("title") or filename,
        filename=filename,
        status=status,
        pages=int(doc.get("pages") or 0),
        chunks=int(doc.get("chunk_count") or 0),
        images=int(doc.get("images") or 0),
        size_bytes=int(doc.get("size_bytes") or doc.get("file_size_bytes") or doc.get("size") or 0),
        indexed_at=doc.get("indexed_at") or doc.get("ingested_at"),
        processing_duration_s=doc.get("processing_duration_s") or doc.get("processing_time_seconds"),
        progress=doc.get("progress"),
        processing_stage=doc.get("processing_stage"),
        error=doc.get("error"),
    )
