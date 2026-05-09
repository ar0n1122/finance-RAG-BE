"""
Ingestion pipeline — orchestrates: convert → chunk → embed → index → store.

Uses Docling ``DocumentConverter`` for PDF→DoclingDocument conversion,
then Docling ``HybridChunker`` (or recursive fallback) for structure-aware
chunking, dense embedding, and Qdrant vector indexing.

PDF conversion + chunking runs in an **isolated subprocess** to prevent
Docling's heavy ML models from crashing the API server with OOM errors.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

_QDRANT_RETRYABLE = (OSError, ConnectionError, ResponseHandlingException)

from app.core.config import Settings, get_settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.ingestion.chunking.base import ChunkingStrategy
from app.ingestion.document_converter import DocumentConverter
from app.ingestion.embeddings.base import EmbeddingProvider
from app.models.domain import Chunk, ContentType, DocumentStatus, IngestionResult
from app.storage.cloud_storage import CloudStorageClient
from app.storage.firestore import FirestoreClient

logger = get_logger(__name__)

# Maximum chars sent to the text embedding model in one go.
_MAX_EMBED_CHARS = 8_000


class IngestionPipeline:
    """
    End-to-end document ingestion.

    Flow:
      1. Upload raw file to GCS
      2. Convert PDF → ``ParsedDocument`` (Docling ``DocumentConverter``)
      3. Chunk → ``list[Chunk]`` (Docling ``HybridChunker`` or recursive fallback)
      4. Embed text chunks → dense vectors
      5. Upsert to Qdrant (dense + full-text index)
      6. Upload processing manifest to GCS
      7. Update Firestore document metadata
    """

    def __init__(
        self,
        settings: Settings,
        converter: DocumentConverter,
        chunker: ChunkingStrategy,
        text_embedder: EmbeddingProvider,
        qdrant: QdrantClient,
        firestore: FirestoreClient,
        gcs: CloudStorageClient,
    ) -> None:
        self._settings = settings
        self._converter = converter
        self._chunker = chunker
        self._text_embedder = text_embedder
        self._qdrant = qdrant
        self._firestore = firestore
        self._gcs = gcs
        self._collection = settings.qdrant_collection

    # ── Progress helpers ─────────────────────────────────────────────────────

    def _update_progress(self, document_id: str, progress: int, stage: str) -> None:
        """Write processing progress (0-100) and stage label to Firestore."""
        try:
            self._firestore.update_document(document_id, {
                "progress": progress,
                "processing_stage": stage,
            })
        except Exception:
            logger.debug("progress_update_failed", document_id=document_id, progress=progress)

    # ── Public API ────────────────────────────────────────────────────────────

    def upload(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        user_id: str = "",
    ) -> str:
        """
        Phase 1 — fast: upload raw file to GCS and create Firestore record.

        Returns the ``document_id``.
        """
        document_id = str(uuid.uuid4())
        file_hash = hashlib.sha256(content).hexdigest()

        logger.info(
            "upload_started",
            document_id=document_id,
            filename=filename,
            size_bytes=len(content),
            user_id=user_id,
        )

        # Create Firestore record (status=processing)
        self._firestore.create_document(document_id, {
            "title": filename,
            "filename": filename,
            "source_file": filename,
            "content_type": content_type,
            "status": DocumentStatus.PROCESSING.value,
            "file_size_bytes": len(content),
            "size_bytes": len(content),
            "file_hash_sha256": file_hash,
            "uploaded_by": user_id,
            "user_id": user_id,
        })

        # Upload raw file to GCS
        gcs_raw_path = f"raw/{document_id}/{filename}"
        self._gcs.upload_bytes(content, gcs_raw_path, content_type=content_type)

        # Cache locally so re-processing avoids a GCS re-download for up to 24 h.
        self._write_cache(document_id, filename, content)

        logger.info("upload_complete", document_id=document_id)
        return document_id

    def process(
        self,
        document_id: str,
        filename: str,
        content: bytes | None = None,
        content_type: str = "application/pdf",
        user_id: str = "",
    ) -> IngestionResult:
        """
        Phase 2 — heavy: convert, chunk, embed, index.

        Expects the Firestore record and GCS file to already exist (from ``upload``).
        If *content* is ``None``, the file is loaded from the local 24-h cache
        (written by ``upload()``) or downloaded from GCS as a fallback.

        PDF conversion + chunking runs in an isolated subprocess so Docling's
        ML models don't consume memory in the API server process.
        """
        start = time.perf_counter()

        if content is None:
            content = self._read_cache(document_id, filename)
            if content is not None:
                logger.debug("process_using_cache", document_id=document_id)
            else:
                logger.info("process_downloading_from_gcs", document_id=document_id)
                content = self._gcs.download_bytes(f"raw/{document_id}/{filename}")
                self._write_cache(document_id, filename, content)

        logger.info("processing_started", document_id=document_id, filename=filename)

        # ── Detect first-run model download ───────────────────────────────────
        # On Cloud Run cold starts (no model cache), docling downloads ~500 MB of
        # AI models from HuggingFace before processing starts.  Warn the user so
        # the long wait doesn't look like a hang.
        _hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        _docling_cache = Path.home() / ".cache" / "docling"
        if not _hf_cache.exists() and not _docling_cache.exists():
            self._update_progress(
                document_id, 2,
                "AI models are still loading in the background — "
                "this upload may take longer than usual."
            )

        self._update_progress(document_id, 5, "converting")

        try:
            # 1+2. Convert PDF → chunks (subprocess for PDFs, in-process for others)
            if content_type == "application/pdf":
                chunks, total_pages = self._convert_and_chunk_subprocess(
                    document_id=document_id,
                    filename=filename,
                    content=content,
                )
            else:
                parsed = self._converter.convert(
                    source=content,
                    document_id=document_id,
                    filename=filename,
                )
                self._update_progress(document_id, 25, "chunking")
                chunks = self._chunker.chunk(parsed)
                total_pages = parsed.total_pages

            self._update_progress(document_id, 40, "embedding")

            # 3. Ensure Qdrant collection exists
            self._ensure_collection()

            # 4. Embed + index text/table chunks
            vectors_indexed = self._embed_and_index(chunks)
            self._update_progress(document_id, 90, "finalizing")

            elapsed = time.perf_counter() - start
            tables_found = sum(
                1 for c in chunks if c.content_type == ContentType.TABLE
            )

            # 6. Update Firestore (status=indexed)
            self._firestore.update_document(document_id, {
                "status": DocumentStatus.INDEXED.value,
                "progress": 100,
                "processing_stage": "complete",
                "total_pages": total_pages,
                "total_chunks": len(chunks),
                "chunk_count": len(chunks),
                "pages": total_pages,
                "chunks": len(chunks),
                "tables_found": tables_found,
                "processing_time_seconds": round(elapsed, 2),
                "processing_duration_s": round(elapsed, 2),
            })

            result = IngestionResult(
                document_id=document_id,
                status=DocumentStatus.INDEXED,
                total_chunks=len(chunks),
                total_pages=total_pages,
                tables_found=tables_found,
                vectors_indexed=vectors_indexed,
                processing_time_s=round(elapsed, 2),
            )
            logger.info(
                "processing_complete",
                document_id=document_id,
                total_chunks=result.total_chunks,
                total_pages=result.total_pages,
                tables=result.tables_found,
                elapsed_s=result.processing_time_s,
            )
            # Remove the local cache file now that the document is indexed.
            self._delete_cache(document_id, filename)
            return result

        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._firestore.update_document(document_id, {
                "status": DocumentStatus.FAILED.value,
                "error": str(exc),
                "processing_time_seconds": round(elapsed, 2),
            })
            logger.error("processing_failed", document_id=document_id, error=str(exc))
            raise IngestionError(f"Processing failed for '{filename}': {exc}") from exc

    # ── Subprocess-based PDF conversion ───────────────────────────────────────

    def _convert_and_chunk_subprocess(
        self,
        document_id: str,
        filename: str,
        content: bytes,
    ) -> tuple[list[Chunk], int]:
        """
        Run Docling conversion + chunking in an isolated subprocess.

        Returns ``(chunks, total_pages)``.

        The subprocess loads Docling models, processes the PDF, writes JSON
        output, and exits — releasing all model memory.  If it OOMs or
        crashes, only the worker process dies; the API server is unaffected.
        """
        # Ensure the PDF is on disk for the subprocess to read
        pdf_path = self._cache_path(document_id, filename)
        if not pdf_path.exists():
            pdf_path.write_bytes(content)

        output_path = pdf_path.with_suffix(".chunks.json")

        # The worker module lives at app/ingestion/docling_worker.py
        # Run from the backend root so `python -m app.ingestion.docling_worker` works
        backend_root = Path(__file__).resolve().parent.parent.parent

        timeout = self._settings.docling_document_timeout or 900

        logger.info(
            "subprocess_convert_start",
            document_id=document_id,
            filename=filename,
            pdf_path=str(pdf_path),
            timeout=timeout,
        )

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m", "app.ingestion.docling_worker",
                    str(pdf_path),
                    str(output_path),
                    document_id,
                    filename,
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(backend_root),
            )

            # ── Read output ───────────────────────────────────────────────
            if not output_path.exists():
                stderr_tail = (result.stderr or "")[-1000:]
                raise IngestionError(
                    f"Worker produced no output (exit={result.returncode}). "
                    f"stderr: {stderr_tail}"
                )

            raw = json.loads(output_path.read_text(encoding="utf-8"))

            if raw.get("status") != "success":
                error_msg = raw.get("error", "unknown worker error")
                error_type = raw.get("error_type", "")
                raise IngestionError(
                    f"Subprocess conversion failed ({error_type}): {error_msg}"
                )

            # ── Reconstruct Chunk objects ─────────────────────────────────
            chunks = [
                Chunk(
                    chunk_id=c["chunk_id"],
                    text=c["text"],
                    enriched_text=c.get("enriched_text"),
                    content_type=ContentType(c["content_type"]),
                    document_id=c["document_id"],
                    page=c["page"],
                    section=c.get("section"),
                    chunk_index=c["chunk_index"],
                    token_count=c["token_count"],
                    metadata=c.get("metadata", {}),
                )
                for c in raw["chunks"]
            ]

            total_pages = raw.get("total_pages", 0)

            logger.info(
                "subprocess_convert_complete",
                document_id=document_id,
                total_chunks=len(chunks),
                total_pages=total_pages,
                exit_code=result.returncode,
            )

            return chunks, total_pages

        except subprocess.TimeoutExpired:
            logger.error(
                "subprocess_convert_timeout",
                document_id=document_id,
                timeout=timeout,
            )
            raise IngestionError(
                f"PDF conversion timed out after {timeout}s for '{filename}'"
            )
        except json.JSONDecodeError as exc:
            raise IngestionError(
                f"Worker output is not valid JSON for '{filename}': {exc}"
            ) from exc
        finally:
            # Always clean up the output JSON (success or failure)
            try:
                output_path.unlink(missing_ok=True)
            except OSError:
                pass

    def ingest(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        user_id: str = "",
    ) -> IngestionResult:
        """
        Run the full ingestion pipeline (upload + process) synchronously.
        Kept for backward compatibility.
        """
        document_id = self.upload(filename, content, content_type, user_id)
        return self.process(document_id, filename, content, content_type, user_id)
    # ── Local upload cache ────────────────────────────────────────────────────

    _CACHE_TTL = 86_400  # 24 hours in seconds

    def _cache_dir(self) -> Path:
        base = Path(self._settings.upload_cache_dir) if self._settings.upload_cache_dir else Path(tempfile.gettempdir())
        d = base / "rag_uploads"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _cache_path(self, document_id: str, filename: str) -> Path:
        # Use document_id prefix so filenames with special chars are safe.
        safe_name = Path(filename).name
        return self._cache_dir() / f"{document_id}_{safe_name}"

    def _write_cache(self, document_id: str, filename: str, content: bytes) -> None:
        try:
            path = self._cache_path(document_id, filename)
            path.write_bytes(content)
            logger.debug("upload_cache_written", document_id=document_id, path=str(path), size=len(content))
        except OSError as exc:
            logger.warning("upload_cache_write_failed", document_id=document_id, error=str(exc))

    def _read_cache(self, document_id: str, filename: str) -> bytes | None:
        try:
            path = self._cache_path(document_id, filename)
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > self._CACHE_TTL:
                path.unlink(missing_ok=True)
                logger.debug("upload_cache_expired", document_id=document_id)
                return None
            return path.read_bytes()
        except OSError as exc:
            logger.warning("upload_cache_read_failed", document_id=document_id, error=str(exc))
            return None

    def _delete_cache(self, document_id: str, filename: str) -> None:
        try:
            path = self._cache_path(document_id, filename)
            path.unlink(missing_ok=True)
            logger.debug("upload_cache_deleted", document_id=document_id)
        except OSError:
            pass
    # ── Qdrant collection setup ───────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection with text dense vector + sparse + payload indexes."""
        # Retry on transient DNS/network failures (common after long CPU-bound processing).
        last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                existing = [c.name for c in self._qdrant.get_collections().collections]
                break
            except _QDRANT_RETRYABLE as exc:
                last_exc = exc
                logger.warning("qdrant_connect_retry", attempt=_attempt + 1, error=f"{type(exc).__name__}: {exc}")
                if _attempt < 2:
                    time.sleep(2 ** _attempt)  # 1s, 2s
        else:
            raise last_exc  # type: ignore[misc]
        if self._collection in existing:
            return

        text_dim = self._text_embedder.dimension

        self._qdrant.create_collection(
            collection_name=self._collection,
            vectors_config={
                "text": models.VectorParams(
                    size=text_dim,
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=200),
                ),
            },
            sparse_vectors_config={
                "text-sparse": models.SparseVectorParams(),
            },
        )

        # Keyword payload indexes — mirrors user's proven Qdrant schema
        for field_name, schema in [
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("content_type", models.PayloadSchemaType.KEYWORD),
            ("section", models.PayloadSchemaType.KEYWORD),
            ("metadata.company", models.PayloadSchemaType.KEYWORD),
            ("metadata.type_of_report", models.PayloadSchemaType.KEYWORD),
            ("metadata.chunk_type", models.PayloadSchemaType.KEYWORD),
            ("metadata.fiscal_year", models.PayloadSchemaType.KEYWORD),
        ]:
            self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=schema,
            )

        # Integer payload indexes
        for field_name in ["page", "metadata.page_min", "metadata.page_max"]:
            self._qdrant.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.INTEGER,
            )

        # Full-text index for keyword search leg
        self._qdrant.create_payload_index(
            collection_name=self._collection,
            field_name="text",
            field_schema=models.TextIndexParams(
                type=models.TextIndexType.TEXT,
                tokenizer=models.TokenizerType.WORD,
                min_token_len=2,
                max_token_len=20,
                lowercase=True,
            ),
        )

        logger.info(
            "qdrant_collection_created",
            collection=self._collection,
            text_dim=text_dim,
        )

    # ── Embedding + indexing ──────────────────────────────────────────────────

    def _embed_and_index(self, chunks: list[Chunk], batch_size: int = 32) -> int:
        """Embed and upsert text/table chunks to Qdrant.  Returns count."""
        if not chunks:
            return 0

        total = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts: list[str] = []
            for c in batch:
                t = c.enriched_text or c.text
                if len(t) > _MAX_EMBED_CHARS:
                    t = t[:_MAX_EMBED_CHARS]
                texts.append(t)

            embeddings = self._text_embedder.embed_documents(texts)

            points: list[models.PointStruct] = []
            for chunk, vector in zip(batch, embeddings):
                points.append(models.PointStruct(
                    id=chunk.chunk_id,
                    vector={"text": vector},
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "content_type": chunk.content_type.value,
                        "text": chunk.text,
                        "page": chunk.page,
                        "section": chunk.section or "",
                        "chunk_index": chunk.chunk_index,
                        "token_count": chunk.token_count,
                        "metadata": chunk.metadata,
                    },
                ))

            for _attempt in range(3):
                try:
                    self._qdrant.upsert(collection_name=self._collection, points=points)
                    break
                except _QDRANT_RETRYABLE as exc:
                    if _attempt == 2:
                        raise
                    logger.warning("qdrant_upsert_retry", attempt=_attempt + 1, error=f"{type(exc).__name__}: {exc}")
                    time.sleep(2 ** _attempt)
            total += len(points)

        logger.debug("text_vectors_indexed", count=total)
        return total
